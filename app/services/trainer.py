"""Training pipeline — orchestrates data loading, model training, and persistence."""
import io
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sqlalchemy.orm import Session
from torch.utils.data import DataLoader, TensorDataset

from app.config import settings
from app.models.orm import User, Profile, Anomaly, Feature
from app.models.autoencoder import Autoencoder, train_autoencoder, compute_anomaly_scores
from app.models.isolation_forest import train_isolation_forest, score_samples
from app.services.preprocessor import DataPreprocessor
from app.services.explainer import Explainer


class Trainer:
    """High-level training pipeline.

    Usage:
        trainer = Trainer(db_session)
        trainer.run(start="2010-01-01", end="2010-03-31")
    """

    def __init__(self, db: Session, task_id: str | None = None):
        self.db = db
        self.task_id = task_id
        self.preprocessor = DataPreprocessor()

    @staticmethod
    def _numeric_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
        """Return only numeric feature columns from a feature matrix."""
        # Почему мы фильтруем метаданные: столбцы date, anon_id, role и т.д.
        # не являются числовыми признаками — их включение в обучение сломало бы
        # алгоритмы: автоэнкодер использует MSE (требует чисел), Isolation Forest
        # строит деревья на евклидовых расстояниях. Категориальные поля (role,
        # department) закодированы отдельно на этапе препроцессинга. Если бы мы
        # оставили anon_id или date, модель выучила бы привязку к конкретной дате
        # или пользователю, что лишило бы её обобщающей способности.
        skip = {"date", "anon_id", "role", "department", "business_unit", "employee_name"}
        return matrix[[c for c in matrix.columns if c not in skip]]

    def _progress(self, progress: float, message: str):
        """Report progress to the background-task store."""
        if self.task_id:
            import app.tasks.train_task as tt
            tt._update(self.task_id, progress, message)

    def run(self, start: str, end: str, model_version: str | None = None):
        """Execute the full training pipeline (synchronous)."""
        # Почему именно такой порядок конвейера: каждый шаг подготавливает
        # данные для следующего. Сначала импортируем пользователей — без них
        # не имеет смысла строить признаки (нет владельца записей). Затем
        # строим признаковое пространство, чтобы подать его обеим моделям
        # сразу. Автоэнкодер (AE) обучается первым, потому что его реконструкция
        # порождает порог аномальности. Isolation Forest (IF) выдаёт свои оценки.
        # Порог считается на AE, потому что он даёт непрерывную MSE-метрику,
        # удобную для перцентильного отсечения. Только после этого сохраняются
        # профили и запускается детекция — потому что для фиксации аномалий
        # нужны уже обученные модели и вычисленный порог. SHAP-объяснитель
        # строится в конце, когда есть полная выборка для фона.
        version = model_version or settings.model_version

        self._progress(5, "Loading users...")
        self._import_users()

        self._progress(15, "Building feature matrix...")
        features = self.preprocessor.build_training_matrix(start, end)
        matrix, norm_params = DataPreprocessor.normalise(features)

        self._progress(30, "Training autoencoder...")
        ae, ae_history = self._train_ae(matrix)

        self._progress(50, "Training Isolation Forest...")
        if_model = self._train_if(matrix)

        self._progress(60, "Computing thresholds...")
        threshold = self._compute_threshold(ae, matrix)

        self._progress(65, "Computing IF anomaly scores...")
        num_matrix = self._numeric_matrix(matrix)
        if_scores = score_samples(if_model, num_matrix.values)

        self._progress(70, "Persisting profiles...")
        self._save_profiles(version, norm_params, threshold)

        self._progress(80, "Running detection on training period...")
        self._detect_anomalies(ae, if_scores, matrix, features, threshold)

        self._progress(90, "Preparing SHAP explainer...")
        self._feature_names = num_matrix.columns.tolist()
        self._explainer = Explainer(ae, num_matrix.values, feature_names=self._feature_names)

        self._progress(100, "Done")
        return {"model_version": version, "train_loss": ae_history["train_loss"][-1] if ae_history["train_loss"] else None}

    # ── internal steps ────────────────────────────────────────

    def _import_users(self):
        """Insert/update users from CERT data into PostgreSQL."""
        # Почему это выделено в отдельный метод, а не встроено в run:
        # импорт пользователей — операция, которая выполняется один раз за
        # конвейер (не на каждую эпоху или шаг). Вынесение упрощает повторное
        # использование: если в будущем понадобится догрузить пользователей
        # без переобучения моделей, можно вызвать только этот метод. Кроме того,
        # так run() сохраняет читаемость — он описывает конвейер на высоком
        # уровне, а детали работы с БД скрыты в специализированных методах.
        df = self.preprocessor.load_users()
        existing = {u.anon_id for u in self.db.query(User).all()}
        # Используем UPSERT-стиль: добавляем только новых, чтобы не плодить
        # дубликаты при повторных запусках обучения.
        for _, row in df.iterrows():
            if row["anon_id"] not in existing:
                user = User(
                    anon_id=row["anon_id"],
                    role=row.get("role"),
                    department=row.get("department"),
                    business_unit=str(row.get("business_unit", "")),
                )
                self.db.add(user)
        self.db.commit()

    def _train_ae(self, matrix: pd.DataFrame) -> tuple[Autoencoder, dict]:
        """Train and return autoencoder."""
        # Почему сплит 90/10, а не 80/20 или cross-validation: 10% —
        # минимальная репрезентативная выборка для мониторинга переобучения
        # по валидационной MSE. При большем размере валидации (20%) мы теряем
        # ценные данные для обучения, учитывая, что период обучения всего 3
        # месяца — около 90 точек на пользователя. Кросс-валидация здесь
        # неоправданно дорога (AE обучается десятки эпох на 80+ признаках).
        num = self._numeric_matrix(matrix)
        data = torch.tensor(num.values, dtype=torch.float32)
        n = len(data)
        val_n = int(n * 0.1)
        train, val = data[val_n:], data[:val_n]

        # Почему batch_size=128: баланс между скоростью сходимости и
        # стабильностью градиента. Для датасета CERT r4.2 (около 360 тыс.
        # записей за квартал) batch 128 даёт примерно 2800 итераций на эпоху —
        # достаточно, чтобы градиент не шумел, как при batch=1, и не усреднял
        # всю выборку, как при full-batch. На Consumer GPU (RTX 3060 с 12 ГБ)
        # 128 помещается в VRAM при размерности ~80 признаков.
        train_loader = DataLoader(TensorDataset(train), batch_size=settings.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(val), batch_size=settings.batch_size)

        model = Autoencoder(input_dim=num.shape[1])
        history = train_autoencoder(
            model, train_loader, val_loader,
            epochs=settings.training_epochs,
            lr=settings.learning_rate,
            patience=settings.early_stop_patience,
        )
        return model, history

    def _train_if(self, matrix: pd.DataFrame):
        num = self._numeric_matrix(matrix)
        return train_isolation_forest(num.values)

    def _compute_threshold(self, model: Autoencoder, matrix: pd.DataFrame) -> float:
        """Reconstruction error at the specified percentile on training data."""
        # Почему MSE + перцентильный порог: автоэнкодер выдаёт среднеквадратичную
        # ошибку реконструкции (MSE) для каждого объекта. MSE хороша тем, что
        # не требует подбора сигмоиды на выходе — она естественна для числовых
        # признаков. Выбор перцентиля (по умолчанию 95% или 99%, задаётся
        # в settings.anomaly_threshold_percentile) вместо фиксированного порога
        # делает систему адаптивной к распределению ошибок на конкретном датасете:
        # даже если MSE систематически выше на одних данных, чем на других,
        # порог автоматически подстраивается хвостом распределения.
        num = self._numeric_matrix(matrix)
        data = torch.tensor(num.values, dtype=torch.float32)
        loader = DataLoader(TensorDataset(data), batch_size=settings.batch_size * 2)
        scores = compute_anomaly_scores(model, loader)
        return float(np.percentile(scores, settings.anomaly_threshold_percentile))

    def _save_profiles(self, version: str, norm_params: dict, threshold: float):
        """Store normalisation params + threshold per user (shared for now)."""
        # Для простоты храним один глобальный профиль; в реальном развёртывании
        # профили должны храниться отдельно на пользователя или на роль.
        serialised = pickle.dumps({"norm_params": norm_params, "threshold": threshold})

        for user in self.db.query(User).all():
            profile = Profile(
                user_id=user.id,
                profile_data={"norm_params": norm_params, "threshold": threshold},
                anomaly_threshold=threshold,
                model_version=version,
                trained_at=datetime.utcnow(),
            )
            self.db.add(profile)
        self.db.commit()

    def _detect_anomalies(
        self,
        model: Autoencoder,
        if_scores: np.ndarray,
        matrix: pd.DataFrame,
        features: pd.DataFrame,
        threshold: float,
    ):
        """Run detection with relative anomaly scores: measure how much a day
        deviates from the user's OWN baseline, not just from the global norm.
        This reduces FP for consistently-unusual users."""
        num = self._numeric_matrix(matrix)
        data = torch.tensor(num.values, dtype=torch.float32)
        loader = DataLoader(TensorDataset(data), batch_size=settings.batch_size * 2)
        ae_scores = np.array(compute_anomaly_scores(model, loader))

        if_threshold = float(np.percentile(if_scores, settings.anomaly_threshold_percentile))

        # Group by user
        user_dates: dict[str, list[tuple[int, datetime, float, float]]] = {}
        for idx, (_, row) in enumerate(features.iterrows()):
            anon = row["anon_id"]
            d = row["date"]
            detected = d if isinstance(d, datetime) else datetime(d.year, d.month, d.day)
            user_dates.setdefault(anon, []).append((idx, detected, float(ae_scores[idx]), float(if_scores[idx])))

        explainer = Explainer(model, num.values, feature_names=num.columns.tolist())

        for anon, records in user_dates.items():
            user = self.db.query(User).filter(User.anon_id == anon).first()
            if user is None:
                continue

            ae_vals = np.array([r[2] for r in records])
            median = float(np.median(ae_vals))
            mad = float(np.median(np.abs(ae_vals - median)))  # Median Absolute Deviation
            mad = max(mad, 1e-10)

            # Почему MAD-based относительное ранжирование (>2 MAD над медианой):
            # у разных пользователей разная «шумность» поведения. Один может
            # регулярно отклоняться на 0.5 MSE (это его норма), а другой —
            # всегда ровно предсказуем. Глобальный порог отсечёт только
            # экстремальные выбросы, но пропустит мягкие, но нехарактерные
            # отклонения у «тихого» пользователя. MAD (медианное абсолютное
            # отклонение) устойчивее к выбросам, чем стандартное отклонение:
            # одно аномальное значение не сдвигает MAD так, как std.
            # Порог >2 MAD означает, что мы считаем аномалией день, чья
            # MSE-ошибка более чем вдвое превышает типичный разброс
            # относительно медианы этого пользователя.
            # Почему согласование AE + IF: каждый детектор имеет свои слепые
            # зоны. AE может пропустить аномалию, если она похожа на обученные
            # нормальные паттерны. IF может ошибиться на разреженных признаках,
            # где деревья делят пространство грубо. Требуя совпадения обеих
            # моделей, мы снижаем ложные срабатывания (FP) ценой небольшой
            # потери истинных (FN). Для бизнеса ИБ ложная тревога дороже
            # пропуска — она подрывает доверие к системе.
            anomaly_days = [
                r for r in records
                if r[2] > threshold                          # globally anomalous
                and (r[2] - median) / mad > 2.0              # >2 MAD above personal baseline
                and r[3] > if_threshold                      # IF agrees
            ]

            # Почему фильтр длительности (минимум 2 дня): внутренний нарушитель
            # обычно проявляет аномальную активность не один день — скачивание
            # документов, переписка в нерабочее время, вход с необычных хостов
            # занимают серию дней. Одиночный выброс чаще оказывается техническим
            # сбоем (пропущенное сетевое событие, скачок нагрузки) или
            # разовым действием легитимного пользователя. Требование >= 2 дней
            # кардинально снижает FP, как показано в оригинальных экспериментах
            # с датасетом CERT.
            if len(anomaly_days) >= 2:
                for idx, detected_at, ae_val, if_val in anomaly_days:
                    # Почему среднее геометрическое для ансамблевой оценки:
                    # (ae_val * if_val)**0.5 — это среднее геометрическое двух
                    # скорингов. В отличие от арифметического, оно штрафует
                    # ситуации, когда одна модель даёт высокий балл, а другая —
                    # низкий. Если AE=0.9 и IF=0.9, геом. среднее = 0.9.
                    # Если AE=0.9 и IF=0.1, то 0.3, а арифметическое = 0.5.
                    # Это дополнительно подчёркивает важность согласия моделей.
                    anomaly = Anomaly(
                        user_id=user.id,
                        score=float((ae_val * if_val) ** 0.5),
                        threshold=threshold,
                        detected_at=detected_at,
                    )
                    self.db.add(anomaly)
                    self.db.flush()

                    # Почему SHAP-объяснитель: SHAP (SHapley Additive exPlanations)
                    # вычисляет вклад каждого признака в отклонение. Без этого
                    # специалист по ИБ увидит только факт аномалии и её числовую
                    # оценку, но не поймёт причины: из-за чего именно день
                    # признан подозрительным (например, «аномально много писем
                    # вовне + вход в нерабочее время»). SHAP даёт интерпретируемость,
                    # которая критична для доверия к системе и для разбора
                    # инцидентов. Explainer использует DeepExplainer для PyTorch,
                    # который эффективно аппроксимирует SHAP-значения через
                    # градиенты сети, не требуя полного перебора комбинаций.
                    sample = num.values[idx]
                    shap_vals = explainer.explain(sample)
                    for fv in shap_vals:
                        feat = Feature(
                            anomaly_id=anomaly.id,
                            feature_name=fv["feature_name"],
                            feature_value=fv["feature_value"],
                            shap_value=fv["shap_value"],
                        )
                        self.db.add(feat)

        self.db.commit()
