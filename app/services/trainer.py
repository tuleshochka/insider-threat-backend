"""Training pipeline — orchestrates data loading, model training, and persistence."""
import os
import io
import pickle
import collections
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sqlalchemy.orm import Session
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from app.config import settings
from app.models.orm import User, Profile, Anomaly, Feature
from app.models.autoencoder import Autoencoder, train_autoencoder, compute_anomaly_scores
from app.models.isolation_forest import train_isolation_forest, score_samples
from app.models.bilstm import BiLSTMClassifier, FocalLoss, TOKEN_MAP, INV_TOKEN_MAP, MAX_SEQ_LEN
from app.services.preprocessor import DataPreprocessor


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
        skip = {"date", "anon_id", "role", "department", "business_unit", "employee_name"}
        return matrix[[c for c in matrix.columns if c not in skip]]

    def _progress(self, progress: float, message: str):
        """Report progress to the background-task store."""
        if self.task_id:
            import app.tasks.train_task as tt
            tt._update(self.task_id, progress, message)

    def _load_malicious_user_dates(self) -> set[tuple[str, str]]:
        """Загружает даты инсайдерской активности из файла ответов (answers)."""
        malicious = set()
        try:
            gt = pd.read_csv(settings.ground_truth_path)
            for _, row in gt.iterrows():
                try:
                    start_dt = pd.to_datetime(row["start"])
                    end_dt = pd.to_datetime(row["end"])
                    cur = start_dt
                    while cur <= end_dt:
                        malicious.add((str(row["user"]), cur.date().strftime("%Y-%m-%d")))
                        cur += pd.Timedelta(days=1)
                except Exception:
                    pass
        except Exception as e:
            print(f"Error loading ground truth: {e}")
        return malicious

    def _build_sequences(self, user_day_list: list[tuple[str, str]], start: str, end: str) -> np.ndarray:
        """Формирует последовательности токенов действий из сырых логов."""
        # 1. Загрузка сырых таблиц за период
        logon = self.preprocessor._load_table("logon", start, end)
        files = self.preprocessor._load_table("file", start, end)
        email = self.preprocessor._load_table("email", start, end)
        dev = self.preprocessor._load_table("device", start, end)
        http = self.preprocessor._load_table("http", start, end)
        
        user_day_events = collections.defaultdict(list)
        
        # Агрегация всех логов
        if not logon.empty:
            for _, row in logon.iterrows():
                u, d = str(row["user"]), row["date"]
                day_str = d.strftime("%Y-%m-%d")
                token = "LOGON" if str(row.get("activity", "logon")).strip().lower() == "logon" else "LOGOFF"
                user_day_events[(u, day_str)].append((d.timestamp(), TOKEN_MAP.get(token, 1)))
                
        if not dev.empty:
            for _, row in dev.iterrows():
                u, d = str(row["user"]), row["date"]
                day_str = d.strftime("%Y-%m-%d")
                token = "USB_CONNECT" if str(row.get("activity", "connect")).strip().lower() == "connect" else "USB_DISCONNECT"
                user_day_events[(u, day_str)].append((d.timestamp(), TOKEN_MAP.get(token, 1)))
                
        if not files.empty:
            for _, row in files.iterrows():
                u, d = str(row["user"]), row["date"]
                day_str = d.strftime("%Y-%m-%d")
                is_usb = (
                    str(row.get("to_removable_media", "")).lower() == "true" or
                    str(row.get("from_removable_media", "")).lower() == "true"
                )
                act = str(row.get("activity", "open")).strip().lower()
                if "open" in act: token = "FILE_OPEN_USB" if is_usb else "FILE_OPEN"
                elif "write" in act: token = "FILE_WRITE_USB" if is_usb else "FILE_WRITE"
                elif "copy" in act: token = "FILE_COPY_USB" if is_usb else "FILE_COPY"
                elif "delete" in act: token = "FILE_DELETE_USB" if is_usb else "FILE_DELETE"
                else: token = "FILE_OPEN_USB" if is_usb else "FILE_OPEN"
                user_day_events[(u, day_str)].append((d.timestamp(), TOKEN_MAP.get(token, 1)))
                
        if not email.empty:
            for _, row in email.iterrows():
                u, d = str(row["user"]), row["date"]
                day_str = d.strftime("%Y-%m-%d")
                to_field = str(row.get("to", ""))
                has_external = any(r.strip() and "@dtaa.com" not in r for r in to_field.split(";"))
                token = "EMAIL_SEND_EXT" if has_external else "EMAIL_SEND_INT"
                user_day_events[(u, day_str)].append((d.timestamp(), TOKEN_MAP.get(token, 1)))
                
        if not http.empty:
            for _, row in http.iterrows():
                u, d = str(row["user"]), row["date"]
                day_str = d.strftime("%Y-%m-%d")
                user_day_events[(u, day_str)].append((d.timestamp(), TOKEN_MAP.get("HTTP_BROWSE", 16)))
                
        # Сборка векторов
        sequences = []
        for u, day_str in user_day_list:
            events = user_day_events.get((u, day_str), [])
            events.sort(key=lambda x: x[0])
            seq = [e[1] for e in events]
            
            if len(seq) > MAX_SEQ_LEN:
                seq = seq[-MAX_SEQ_LEN:]
            else:
                seq = seq + [TOKEN_MAP["<PAD>"]] * (MAX_SEQ_LEN - len(seq))
            sequences.append(seq)
            
        return np.array(sequences, dtype=np.int32)

    def _train_bilstm(self, sequences: np.ndarray, labels: np.ndarray) -> BiLSTMClassifier:
        """Обучает Bi-LSTM классификатор на последовательностях токенов."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        class BiLSTMTrainingDataset(torch.utils.data.Dataset):
            def __init__(self, seqs, lbls):
                self.seqs = torch.LongTensor(seqs)
                self.lbls = torch.FloatTensor(lbls)
            def __len__(self):
                return len(self.lbls)
            def __getitem__(self, idx):
                return self.seqs[idx], self.lbls[idx]
                
        dataset = BiLSTMTrainingDataset(sequences, labels)
        
        class_counts = np.bincount(labels.astype(np.int32))
        class_counts = np.maximum(class_counts, 1)
        
        # Если в подвыборку попал только один класс (например, нет атак),
        # обучаемся на обычном DataLoader без сэмплера.
        if len(class_counts) < 2 or class_counts[1] == 0:
            loader = DataLoader(dataset, batch_size=64, shuffle=True)
        else:
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[labels.astype(np.int32)]
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights, num_samples=len(sample_weights), replacement=True
            )
            loader = DataLoader(dataset, batch_size=64, sampler=sampler)
            
        model = BiLSTMClassifier().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        model.train()
        for epoch in range(10):
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
        model.to("cpu")
        return model

    def run(self, start: str, end: str, model_version: str | None = None):
        """Execute the full training pipeline (synchronous)."""
        version = model_version or settings.model_version

        self._progress(5, "Loading users...")
        self._import_users()

        self._progress(15, "Building feature matrix...")
        features = self.preprocessor.build_training_matrix(start, end)
        matrix, norm_params = DataPreprocessor.normalise(features)

        self._progress(30, "Training autoencoder (Level 1)...")
        ae, ae_history = self._train_ae(matrix)

        self._progress(50, "Training Isolation Forest (Level 1)...")
        if_model = self._train_if(matrix)

        self._progress(60, "Computing thresholds...")
        threshold = self._compute_threshold(ae, matrix)

        self._progress(65, "Filtering Level 1 for Level 2 training...")
        num_matrix = self._numeric_matrix(matrix)
        if_scores = score_samples(if_model, num_matrix.values)
        
        # Получаем комбинированный скор первого уровня
        data_t = torch.tensor(num_matrix.values, dtype=torch.float32)
        loader_all = DataLoader(TensorDataset(data_t), batch_size=256)
        ae_scores = np.array(compute_anomaly_scores(ae, loader_all))
        
        ae_norm = (ae_scores - ae_scores.min()) / (ae_scores.max() - ae_scores.min() + 1e-10)
        ensemble = np.sqrt(ae_norm * if_scores)
        
        # Отбор топ-20% наиболее аномальных дней
        top_k = int(len(ensemble) * 0.20)
        top_k = max(top_k, min(100, len(ensemble)))
        keep_idx = np.argsort(-ensemble)[:top_k]
        
        selected_user_days = []
        selected_labels = []
        malicious_dates = self._load_malicious_user_dates()
        
        for idx in keep_idx:
            row = features.iloc[idx]
            u = str(row["anon_id"])
            d_str = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])
            selected_user_days.append((u, d_str))
            selected_labels.append(1 if (u, d_str) in malicious_dates else 0)

        self._progress(70, "Persisting profiles...")
        self._save_profiles(version, norm_params, threshold)

        self._progress(75, "Training Bi-LSTM (Level 2)...")
        sequences = self._build_sequences(selected_user_days, start, end)
        bilstm_model = self._train_bilstm(sequences, np.array(selected_labels))

        self._progress(85, "Saving trained models to disk...")
        model_dir = settings.model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Сохранение StandardScaler
        scaler = StandardScaler()
        scaler.fit(num_matrix.values)
        with open(os.path.join(model_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
            
        with open(os.path.join(model_dir, "feature_cols.pkl"), "wb") as f:
            pickle.dump(num_matrix.columns.tolist(), f)
            
        with open(os.path.join(model_dir, "isolation_forest.pkl"), "wb") as f:
            pickle.dump(if_model, f)
            
        torch.save(ae.state_dict(), os.path.join(model_dir, "autoencoder.pt"))
        torch.save(bilstm_model.state_dict(), os.path.join(model_dir, "bilstm.pt"))

        self._progress(90, "Running detection on training period...")
        self._detect_anomalies(ae, bilstm_model, matrix, features, threshold, start, end)

        self._progress(100, "Done")
        return {"model_version": version, "train_loss": ae_history["train_loss"][-1] if ae_history["train_loss"] else None}

    # ── internal steps ────────────────────────────────────────

    def _import_users(self):
        df = self.preprocessor.load_users()
        existing = {u.anon_id for u in self.db.query(User).all()}
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
        num = self._numeric_matrix(matrix)
        data = torch.tensor(num.values, dtype=torch.float32)
        n = len(data)
        val_n = int(n * 0.1)
        if val_n == 0 and n >= 2:
            val_n = 1

        if val_n > 0:
            train, val = data[val_n:], data[:val_n]
            train_loader = DataLoader(TensorDataset(train), batch_size=settings.batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(val), batch_size=settings.batch_size)
        else:
            train_loader = DataLoader(TensorDataset(data), batch_size=settings.batch_size, shuffle=True)
            val_loader = None

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
        num = self._numeric_matrix(matrix)
        data = torch.tensor(num.values, dtype=torch.float32)
        loader = DataLoader(TensorDataset(data), batch_size=settings.batch_size * 2)
        scores = compute_anomaly_scores(model, loader)
        return float(np.percentile(scores, settings.anomaly_threshold_percentile))

    def _save_profiles(self, version: str, norm_params: dict, threshold: float):
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
        ae: Autoencoder,
        bilstm: BiLSTMClassifier,
        matrix: pd.DataFrame,
        features: pd.DataFrame,
        threshold: float,
        start: str,
        end: str
    ):
        """Детектирует аномалии на обучающей выборке по гибридному пайплайну."""
        num = self._numeric_matrix(matrix)
        data = torch.tensor(num.values, dtype=torch.float32)
        loader = DataLoader(TensorDataset(data), batch_size=128)
        ae_scores = np.array(compute_anomaly_scores(ae, loader))
        
        # Фильтрация 1-го уровня
        ae_norm = (ae_scores - ae_scores.min()) / (ae_scores.max() - ae_scores.min() + 1e-10)
        top_k = int(len(ae_norm) * 0.20)
        top_k = max(top_k, min(100, len(ae_norm)))
        l1_pass_idx = np.argsort(-ae_norm)[:top_k]
        
        passed_user_days = []
        passed_indices = []
        for idx in l1_pass_idx:
            row = features.iloc[idx]
            u = str(row["anon_id"])
            d_str = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])
            passed_user_days.append((u, d_str))
            passed_indices.append(idx)
            
        if not passed_user_days:
            return
            
        # Формирование последовательностей для Level 2
        sequences = self._build_sequences(passed_user_days, start, end)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bilstm.to(device)
        bilstm.eval()
        
        with torch.no_grad():
            seq_tensor = torch.LongTensor(sequences).to(device)
            logits_list = []
            attn_list = []
            for i in range(0, len(seq_tensor), 64):
                batch_x = seq_tensor[i:i+64]
                logits, attn = bilstm.get_attention(batch_x)
                logits_list.append(logits.cpu())
                attn_list.append(attn.cpu())
            
            probs = torch.sigmoid(torch.cat(logits_list)).numpy()
            attns = torch.cat(attn_list).numpy()
            
        bilstm.to("cpu")
        
        l2_threshold = 0.5
        
        for i, idx in enumerate(passed_indices):
            prob = float(probs[i])
            if prob > l2_threshold:
                row = features.iloc[idx]
                anon = str(row["anon_id"])
                d = row["date"]
                detected_at = d if isinstance(d, datetime) else datetime(d.year, d.month, d.day)
                
                user = self.db.query(User).filter(User.anon_id == anon).first()
                if not user:
                    continue
                    
                anomaly = Anomaly(
                    user_id=user.id,
                    score=prob,
                    threshold=l2_threshold,
                    detected_at=detected_at,
                    status="pending"
                )
                self.db.add(anomaly)
                self.db.flush()
                
                # Запись весов внимания Attention в качестве объяснений (XAI)
                seq_tokens = sequences[i]
                attn_weights = attns[i]
                
                for step_idx, token_id in enumerate(seq_tokens):
                    if token_id == 0:  # Пропускаем <PAD>
                        continue
                    token_name = INV_TOKEN_MAP.get(token_id, f"UNK_{token_id}")
                    weight = float(attn_weights[step_idx])
                    
                    feat = Feature(
                        anomaly_id=anomaly.id,
                        feature_name=f"Шаг {step_idx + 1}: {token_name}",
                        feature_value=weight,
                        shap_value=weight
                    )
                    self.db.add(feat)
                    
        self.db.commit()
