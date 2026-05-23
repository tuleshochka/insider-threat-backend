"""
Загрузка, очистка, нормализация и агрегация данных датасета CERT r4.2.

Препроцессор отвечает за превращение «сырых» CSV-файлов датасета (логины,
файловые операции, почта, устройства, HTTP) в матрицу признаков, пригодную
для обучения моделей машинного обучения.

Все признаки агрегируются на уровне «пользователь × день», поскольку:
  - инсайдерские угрозы проявляются в дневных паттернах поведения;
  - более мелкая гранулярность (почасовой) даёт слишком разрежённые векторы;
  - более крупная (неделя) сглаживает аномальные всплески.

Источник: датасет CERT Insider Threat r4.2 [1].
"""
import os
from datetime import datetime, date
from typing import Literal

import pandas as pd
import numpy as np

from app.config import settings


class DataPreprocessor:
    """
    Загружает «сырые» CSV-файлы CERT, чистит, нормализует и собирает
    признаковые векторы для каждого пользователя за каждый день.

    Поток обработки:
      1) Загрузка справочника пользователей (users.csv) — кешируется.
      2) Чтение событийных таблиц чанками (по 500k строк) с фильтрацией по
         дате и ранней остановкой (~30% от оценённого объёма).
      3) Агрегация каждой таблицы в признаки {date, user}.
      4) Объединение всех агрегатов внешним (outer) merge — для дней,
         когда по одной системе данные есть, а по другой нет.
      5) Присоединение метаданных пользователей (роль, отдел) и
         психометрики (Big Five).
      6) Исключение дней с известными аномалиями (опционально).

    Зачем такой подход:
      - датасет CERT может занимать до 90+ ГБ на диске; чанкование
        предотвращает переполнение памяти;
      - агрегация per-day per-user снижает размерность и делает
        признаки интерпретируемыми для аналитика ИБ;
      - outer merge + fillna(0) гарантирует, что для дней без активности
        по какому-то каналу будет 0, а не пропуск (NaN ломает большинство
        моделей машинного обучения).
    """

    def __init__(self, data_dir: str | None = None):
        """
        Args:
            data_dir: Путь к каталогу с CSV-файлами датасета.
                      Если None — используется значение из настроек приложения.
        """
        self.data_dir = data_dir or settings.dataset_path
        self._users_df: pd.DataFrame | None = None
        self._features_df: pd.DataFrame | None = None

    # ── public API ────────────────────────────────────────────

    def load_users(self) -> pd.DataFrame:
        """
        Загружает и кеширует справочник пользователей из users.csv.

        Зачем нужен отдельный метод:
          - users.csv читается один раз и переиспользуется всеми методами;
          - содержит «статичные» атрибуты (роль, отдел), которые
            присоединяются к признакам на финальном шаге.

        Преобразования:
          - user_id → anon_id (единообразие имён с другими таблицами);
          - start_date / end_date приводятся к datetime (для возможной
            фильтрации по дате найма/увольнения — расширяемость);
          - бизнес-единица (business_unit) и функциональная единица
            (functional_unit) явно приводятся к строке — в исходных данных
            они могут быть числами с плавающей точкой из-за особенностей
            парсинга CSV в CERT.
        """
        if self._users_df is not None:
            return self._users_df
        path = os.path.join(self.data_dir, "users.csv")
        cols = {"user_id": str, "role": str, "department": str}
        df = pd.read_csv(path, dtype=cols)
        if "business_unit" in df.columns:
            df["business_unit"] = df["business_unit"].astype(str)
        if "functional_unit" in df.columns:
            df["functional_unit"] = df["functional_unit"].astype(str)
        if "start_date" in df.columns:
            df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
        if "end_date" in df.columns:
            df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
        df.rename(columns={"user_id": "anon_id"}, inplace=True)
        self._users_df = df
        return df

    def build_training_matrix(
        self,
        start: str | None,
        end: str | None,
        exclude_anomaly_dates: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Основной метод: загружает события за период [start, end],
        агрегирует их в признаки {date, user} и возвращает матрицу
        для обучения модели.

        Шаги:
          1) Загрузка всех событийных таблиц (logon, file, email, device, http).
          2) Агрегация каждой таблицы в свой набор признаков.
          3) Последовательный внешний merge — так как не каждый пользователь
             проявляет активность во всех системах каждый день.
          4) Заполнение NaN → 0 (отсутствие активности = нулевая интенсивность).
          5) Присоединение метаданных (роль, отдел, бизнес-единица).
          6) Присоединение психометрики (Big Five) — если файл существует.
          7) Исключение дней с известными аномалиями (чтобы модель не училась
             на «шумных» днях, где аномалии — это не инсайдер, а сбой сбора).

        Почему порядок merge важен:
          - начинаем с logon (самая плотная таблица — логины есть почти
            каждый день у каждого активного пользователя);
          - затем присоединяем остальные; outer merge не теряет строки,
            даже если в logon нет записи для данного (user, date).

        Args:
            start: Начало периода (включительно), e.g. "2010-01-01".
            end:   Конец периода (включительно), e.g. "2010-03-31".
            exclude_anomaly_dates: Список дат для исключения (известные
                                   инциденты, не связанные с инсайдерами).

        Returns:
            pd.DataFrame с колонками: date, anon_id, feature_*, role,
            department, business_unit, O, C, E, A, N (Big Five).
        """
        users = self.load_users()
        logon = self._load_table("logon", start, end)
        files = self._load_table("file", start, end)
        email = self._load_table("email", start, end)
        dev   = self._load_table("device", start, end)
        http  = self._load_table("http", start, end)

        # ── aggregate per user + day ──
        aggs = []
        aggs.append(self._agg_logon(logon))
        aggs.append(self._agg_file(files))
        aggs.append(self._agg_email(email))
        aggs.append(self._agg_device(dev))
        if not http.empty:
            aggs.append(self._agg_http(http))

        # Merge all aggregations on (date, anon_id)
        merged = aggs[0]
        for a in aggs[1:]:
            merged = merged.merge(a, on=["date", "anon_id"], how="outer")

        merged.fillna(0, inplace=True)

        # Attach user metadata
        meta = users[["anon_id", "role", "department", "business_unit"]].copy()
        result = merged.merge(meta, on="anon_id", how="left")

        # Attach psychometric data (Big Five)
        psych_path = os.path.join(self.data_dir, "psychometric.csv")
        if os.path.exists(psych_path):
            psych = pd.read_csv(psych_path, usecols=["user_id", "O", "C", "E", "A", "N"])
            psych.rename(columns={"user_id": "anon_id"}, inplace=True)
            result = result.merge(psych, on="anon_id", how="left")

        # Exclude known anomaly days
        if exclude_anomaly_dates:
            exclude = pd.to_datetime(exclude_anomaly_dates).date
            mask = result["date"].isin(exclude)
            result = result[~mask]

        self._features_df = result
        return result

    # ── internal helpers ──────────────────────────────────────

    def _load_table(self, name: str, start: str | None, end: str | None) -> pd.DataFrame:
        """
        Читает CSV-файл событийной таблицы чанками, фильтрует по дате.

        Зачем chunked CSV loading с chunksize=500_000:
          - исходные CSV-файлы CERT занимают десятки гигабайт; загрузка
            целиком в память невозможна на рядовом ноутбуке;
          - 500 000 строк на чанк — компромисс между частотой I/O
            (больше чанков — медленнее диск) и потреблением памяти
            (меньше чанков — больше RAM на чанк);
          - low_memory=False предотвращает фрагментацию dtypes при
            чанковом чтении (pandas иначе может угадать разные типы
            для одной колонки в разных чанках).

        Зачем early stop на ~30% от оценённого числа строк:
          - датасет содержит ~18 месяцев данных; для обучения мы часто
            берём первые 3–6 месяцев;
          - оценка LARGE_EST основана на реальных размерах CERT r4.2:
            email — самый «толстый» (переписка очень активная), logon и
            file — средние, device и http — сравнительно небольшие;
          - умножая на 0.3, мы покрываем ~3 месяца из 18 (~17%), но с
            запасом, так как не все строки проходят фильтр по дате.
          - Ранняя остановка экономит время на переборе миллионов строк,
            которые заведомо не войдут в период training_window.

        Зачем явный format="%m/%d/%Y %H:%M:%S":
          - даты в CERT записаны в американском формате (месяц/день/год),
            что отличается от ISO-8601; без явного формата pandas может
            неверно интерпретировать (01/02 → 2 января или 1 февраля).
          - errors="coerce" превращает некорректные записи в NaT
            (Not a Time), что безопаснее ошибки парсинга.

        Почему end_dt = end + 1 день с условием < (а не <=):
          - убирает неоднозначность полуночного времени: если end =
            "2010-03-31", то строки с 2010-03-31 23:59 войдут, а
            2010-04-01 00:00 — нет (строгое меньше).
          - Это эквивалентно `date < end + 1 day`.

        Args:
            name: Имя таблицы (logon, file, email, device, http).
            start, end: Границы периода фильтрации.

        Returns:
            pd.DataFrame с отфильтрованными строками и колонкой date
            типа datetime.
        """
        path = os.path.join(self.data_dir, f"{name}.csv")
        usecols, dtypes = self._schema(name)

        # Determine early-stop threshold (process ~30% of estimated rows)
        LARGE_EST = {"email": 5_000_000, "file": 1_500_000, "logon": 2_000_000, "device": 500_000, "http": 1_000_000}
        stop_at = LARGE_EST.get(name, 0) * 0.3

        chunks: list[pd.DataFrame] = []
        reader = pd.read_csv(path, usecols=usecols, dtype=dtypes, chunksize=500_000, low_memory=False)
        start_dt = pd.to_datetime(start) if start else None
        end_dt = pd.to_datetime(end) + pd.Timedelta(days=1) if end else None  # include full end day
        total_read = 0
        for chunk in reader:
            total_read += len(chunk)
            chunk["date"] = pd.to_datetime(chunk["date"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
            if start_dt is not None:
                chunk = chunk[chunk["date"] >= start_dt]
            if end_dt is not None:
                chunk = chunk[chunk["date"] < end_dt]  # strict < to include end day
            if not chunk.empty:
                chunks.append(chunk)
            # Stop after reading ~30% of estimated rows (before filtering)
            if total_read >= stop_at:
                break

        if not chunks:
            return pd.DataFrame()
        return pd.concat(chunks, ignore_index=True)

    @staticmethod
    def _schema(name: str) -> tuple[list[str], dict]:
        """
        Возвращает (список колонок для чтения, словарь dtypes) для таблицы.

        Зачем явно задавать dtypes:
          - CERT-датасет содержит поля, которые pandas может неверно
            интерпретировать: user_id как число, activity как булево и т.д.
          - Явное указание str для идентификаторов гарантирует, что
            "001" не превратится в 1;
          - size и attachments указаны как int64 — это числовые метрики,
            над которыми будут выполняться sum/mean;
          - id (идентификатор сессии) — str, так как это UUID-подобные
            коды, арифметика с ними не нужна.

        Колонка 'date' намеренно не типизируется:
          - pandas прочитает её как str, а преобразование в datetime
            выполняется позже (в _load_table) с явным форматом;
          - это даёт контроль над обработкой ошибок формата.

        Почему в email нужна колонка 'from' (зарезервированное слово):
          - это название поля в датасете CERT; для различения отправленных
            и полученных писем сравниваем user с from.
        """
        if name == "logon":
            return (["date", "user", "pc", "activity", "id"],
                    {"user": str, "pc": str, "activity": str, "id": str})
        if name == "file":
            return (["date", "user", "pc", "filename", "id"],
                    {"user": str, "pc": str, "filename": str, "id": str})
        if name == "email":
            return (["date", "user", "to", "from", "size", "attachments", "id"],
                    {"user": str, "to": str, "from": str, "size": "int64", "attachments": "int64", "id": str})
        if name == "device":
            return (["date", "user", "pc", "activity", "id"],
                    {"user": str, "pc": str, "activity": str, "id": str})
        if name == "http":
            return (["date", "user", "url", "id"],
                    {"user": str, "url": str, "id": str})
        return (["date", "user"], {"user": str})

    # ── aggregation functions ─────────────────────────────────

    def _agg_logon(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Агрегирует логины в признаки per-user per-day.

        Зачем именно эти признаки:
          - logon_count: общее количество сессий — базовая мера активности;
          - logon_unique_pc: число уникальных ПК — резкое расширение
            круга используемых машин может указывать на разведку
            (insider подбирается к чужим данным);
          - after_hours_logons: логины в нерабочее время (до 8:00,
            после 18:00) — типичный паттерн инсайдера, работающего
            скрытно в часы, когда коллег нет;
          - weekend_logons: то же для выходных (суббота, воскресенье).

        Зачем выделяются час (hour) и день недели (dow):
          - эти временные признаки извлекаются один раз из datetime
            и переиспользуются во всех агрегациях одного вызова;
          - dayofweek: 0 = понедельник, 6 = воскресенье (стандарт
            Python/pandas); условие (>= 5) означает сб/вс.

        Зачем dt.date (date_only), а не группировка по datetime:
          - группировка по полной дате-времени дала бы срез по
            секундам — бессмысленно для анализа дневных паттернов;
          - мы собираем дневные признаки, поэтому группируем по дате
            без времени.
        """
        df["date_only"] = df["date"].dt.date
        df["hour"] = df["date"].dt.hour
        df["dow"] = df["date"].dt.dayofweek  # 0=Mon..6=Sun
        grp = df.groupby(["date_only", "user"], as_index=False)
        result = grp.agg(
            logon_count=("id", "count"),
            logon_unique_pc=("pc", "nunique"),
            after_hours_logons=("hour", lambda x: ((x < 8) | (x >= 18)).sum()),
            weekend_logons=("dow", lambda x: (x >= 5).sum()),
        )
        result.rename(columns={"user": "anon_id", "date_only": "date"}, inplace=True)
        return result

    def _agg_file(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Агрегирует файловые операции.

        Признаки:
          - file_operations: общее число операций с файлами;
          - file_unique_pc: число разных ПК, с которых выполнялись
            операции — аномалия, если пользователь вдруг начал работать
            с файлами с машин, где обычно этого не делает;
          - file_unique_names: число уникальных имён файлов — может
            указывать на массовое копирование (много разных файлов
            за короткое время, что нехарактерно для обычной работы);
          - after_hours_files: файловые операции вне рабочего времени.

        В датасете CERT нет разделения на чтение/запись в колонке
        'activity' для файлов (в отличие от logon), поэтому общее
        количество — единственная доступная метрика.
        """
        df["date_only"] = df["date"].dt.date
        df["hour"] = df["date"].dt.hour
        grp = df.groupby(["date_only", "user"], as_index=False)
        result = grp.agg(
            file_operations=("id", "count"),
            file_unique_pc=("pc", "nunique"),
            file_unique_names=("filename", "nunique"),
            after_hours_files=("hour", lambda x: ((x < 8) | (x >= 18)).sum()),
        )
        result.rename(columns={"user": "anon_id", "date_only": "date"}, inplace=True)
        return result

    def _agg_email(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Агрегирует почтовую активность с разделением на отправленные
        и полученные письма.

        Зачем sent/received split:
          - резкий рост исходящей переписки (особенно с вложениями)
            может указывать на кражу данных;
          - рост входящей — менее подозрителен (спам, массовые рассылки);
          - размер вложений (attachments) — специфический индикатор:
            массовая рассылка с большими вложениями — классический вектор
            эксфильтрации.

        Логика разделения:
          - sent: строки, где user == from (пользователь — отправитель);
          - received: строки, где user != from (пользователь — получатель,
            либо в копии, либо указан в to).

        Почему outer merge между sent и received:
          - у пользователя может быть день, когда он только писал
            письма, но не получал (или наоборот); outer + fillna(0)
            корректно обрабатывает этот случай, не теряя строки.
        """
        df["date_only"] = df["date"].dt.date
        df["hour"] = df["date"].dt.hour
        # sent: user is the sender
        sent = df[df["user"] == df["from"]].groupby(["date_only", "user"], as_index=False).agg(
            email_sent=("id", "count"),
            email_size_total=("size", "sum"),
            email_attachments=("attachments", "sum"),
            email_unique_recipients=("to", "nunique"),
            after_hours_email=("hour", lambda x: ((x < 8) | (x >= 18)).sum()),
        )
        received = df[df["user"] != df["from"]].groupby(["date_only", "user"], as_index=False).agg(
            email_received=("id", "count"),
        )
        result = sent.merge(received, on=["date_only", "user"], how="outer").fillna(0)
        result.rename(columns={"user": "anon_id", "date_only": "date"}, inplace=True)
        return result

    def _agg_device(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Агрегирует события подключения устройств (USB, внешние диски).

        Признаки:
          - device_operations: общее число подключений/отключений;
          - after_hours_device: подключения вне рабочего времени;
          - weekend_device: подключения в выходные.

        Зачем отдельный признак для устройств:
          - в исследованиях CERT [2] показано, что инсайдеры часто
            используют USB-накопители для кражи данных;
          - активность с устройствами в нерабочее время — сильный
            маркер аномалии.
        """
        df["date_only"] = df["date"].dt.date
        df["hour"] = df["date"].dt.hour
        df["dow"] = df["date"].dt.dayofweek
        all_dev = df.groupby(["date_only", "user"], as_index=False).agg(
            device_operations=("id", "count"),
            after_hours_device=("hour", lambda x: ((x < 8) | (x >= 18)).sum()),
            weekend_device=("dow", lambda x: (x >= 5).sum()),
        )
        result = all_dev
        result.rename(columns={"user": "anon_id", "date_only": "date"}, inplace=True)
        return result

    def _agg_http(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Агрегирует HTTP-запросы (веб-сёрфинг).

        Признаки:
          - http_requests: общее число запросов;
          - http_unique_urls: число уникальных URL — резкий рост
            может указывать на разведку или посещение нехарактерных
            для пользователя ресурсов;
          - after_hours_http: веб-активность вне рабочего времени.

        Примечание: URL-адреса не анализируются по содержанию
        из соображений приватности — датасет CERT содержит URL,
        но мы используем только метрику уникальности.
        """
        df["date_only"] = df["date"].dt.date
        df["hour"] = df["date"].dt.hour
        grp = df.groupby(["date_only", "user"], as_index=False)
        result = grp.agg(
            http_requests=("id", "count"),
            http_unique_urls=("url", "nunique"),
            after_hours_http=("hour", lambda x: ((x < 8) | (x >= 18)).sum()),
        )
        result.rename(columns={"user": "anon_id", "date_only": "date"}, inplace=True)
        return result

    # ── normalisation helpers (to be called by trainer) ───────

    @staticmethod
    def normalise(features: pd.DataFrame, fit_params: dict | None = None) -> tuple[pd.DataFrame, dict]:
        """
        Стандартизация (Z-score) числовых признаков.

        Формула: X' = (X - mean) / std

        Зачем стандартизация (mean/std), а не min-max:
          - большинство моделей детекции аномалий (Isolation Forest,
            One-Class SVM, автоэнкодеры) чувствительны к масштабу
            признаков;
          - min-max нормализация чувствительна к выбросам (а у нас
            как раз аномалии — это выбросы), поэтому Z-score более
            устойчив;
          - Z-score приводит признаки к распределению N(0,1), что
            позволяет модели корректно сравнивать признаки разной
            размерности (логины: 0–50, размер почты: 0–10^7).

        Зачем пропускаем {date, anon_id, role, department, business_unit,
        employee_name}:
          - date — идентификатор строки, не признак;
          - anon_id — идентификатор пользователя (категория);
          - role, department, business_unit — категориальные метаданные;
          - employee_name — имя (бесполезно для ML, может быть полезно
            для отладки/логов безопасности).

        Почему std == 0 заменяется на 1.0:
          - у некоторых признаков может быть нулевая вариация (все
            значения одинаковы), тогда деление на 0 даёт NaN;
          - std = 1.0 означает, что такой признак «замораживается» в
            значении (X - mean) / 1.0 = X - mean, то есть просто
            центрируется без нормирования масштаба — это корректно,
            так как модель не получит от него полезной информации.

        Два режима работы (fit vs transform):
          - fit_params=None: вычисляем mean и std по переданным данным
            (режим обучения) и возвращаем их вместе с нормализованными
            данными;
          - fit_params задан: применяем уже известные mean и std
            (режим трансформации — для тестовых данных и инференса).
          - Это позволяет нормализовать обучающую выборку и затем
            применять те же параметры к валидационной/тестовой,
            избегая data leakage.

        Args:
            features: Исходная матрица признаков.
            fit_params: Словарь с ключами 'mean', 'std', 'columns'
                        для повторного применения (из тренировки).

        Returns:
            (нормализованный DataFrame, fit_params).
        """
        skip = {"date", "anon_id", "role", "department", "business_unit", "employee_name"}
        feat_cols = [c for c in features.columns if c not in skip]
        values = features[feat_cols].values.astype(np.float64)

        if fit_params is None:
            mean = values.mean(axis=0)
            std = values.std(axis=0)
            std[std == 0] = 1.0
            fit_params = {"mean": mean.tolist(), "std": std.tolist(), "columns": feat_cols}
        else:
            mean = np.array(fit_params["mean"])
            std = np.array(fit_params["std"])

        normalised = (values - mean) / std
        features = features.copy()
        features[feat_cols] = normalised
        return features, fit_params
