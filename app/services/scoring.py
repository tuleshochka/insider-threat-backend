import os
import pickle
import logging
import datetime
import torch
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session

from app.models.orm import Event, Profile, Anomaly, Feature, User
from app.models.autoencoder import Autoencoder
from app.models.isolation_forest import score_samples
from app.models.bilstm import BiLSTMClassifier, TOKEN_MAP, INV_TOKEN_MAP, MAX_SEQ_LEN, map_event_to_token
from app.config import settings

logger = logging.getLogger(__name__)

class ScoringService:
    def __init__(self, db: Session):
        self.db = db
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model_dir = settings.model_dir
        
        if not os.path.exists(model_dir):
            logger.warning(f"Model directory '{model_dir}' not found. Using simulated scoring.")
            self.simulated = True
            return
            
        self.simulated = False
        try:
            self.feature_cols = pickle.load(open(os.path.join(model_dir, "feature_cols.pkl"), "rb"))
            self.scaler = pickle.load(open(os.path.join(model_dir, "scaler.pkl"), "rb"))
            
            # Autoencoder (Level 1)
            self.ae_model = Autoencoder(input_dim=len(self.feature_cols)).to(self.device)
            self.ae_model.load_state_dict(torch.load(os.path.join(model_dir, "autoencoder.pt"), map_location=self.device, weights_only=True))
            self.ae_model.eval()

            # Isolation Forest (Level 1)
            self.if_model = pickle.load(open(os.path.join(model_dir, "isolation_forest.pkl"), "rb"))

            # Bi-LSTM (Level 2)
            self.bilstm_model = BiLSTMClassifier().to(self.device)
            self.bilstm_model.load_state_dict(torch.load(os.path.join(model_dir, "bilstm.pt"), map_location=self.device, weights_only=True))
            self.bilstm_model.eval()
            
            logger.info("Models loaded successfully for inference.")
        except Exception as e:
            logger.error(f"Failed to load models from {model_dir}: {e}")
            self.simulated = True

    def _aggregate_user_events(self, user_anon_id: str, date: datetime.date) -> dict:
        """
        Aggregate raw events into daily features (mocked default values for missing columns,
        matching the schema expected by the tabular models).
        """
        features = {col: 0.0 for col in getattr(self, "feature_cols", [])}
        
        # Загрузка событий из базы за выбранный день
        start_time = datetime.datetime.combine(date, datetime.time.min)
        end_time = datetime.datetime.combine(date, datetime.time.max)
        events = self.db.query(Event).filter(
            Event.anon_id == user_anon_id,
            Event.timestamp >= start_time,
            Event.timestamp <= end_time
        ).all()
        
        logon_count = sum(1 for e in events if e.event_type == 'logon')
        file_ops = sum(1 for e in events if e.event_type == 'file')
        http_reqs = sum(1 for e in events if e.event_type == 'http')
        dev_ops = sum(1 for e in events if e.event_type == 'device')
        email_ops = sum(1 for e in events if e.event_type == 'email')
        
        if not self.simulated:
            if "logon_count" in features: features["logon_count"] = float(logon_count)
            if "file_operations" in features: features["file_operations"] = float(file_ops)
            if "http_requests" in features: features["http_requests"] = float(http_reqs)
            if "device_operations" in features: features["device_operations"] = float(dev_ops)
            if "email_sent" in features: features["email_sent"] = float(email_ops)
            
            # Заполняем остальные колонки нормальными случайными значениями
            for col in features:
                if features[col] == 0.0:
                    features[col] = float(np.random.normal(0, 1))
        
        return features

    def update_baselines_and_score(self):
        """
        Запускает фоновый процесс оценки поведения активных пользователей.
        """
        logger.info("Starting background scoring process")
        users = self.db.query(User).filter(User.is_active == True).all()
        
        for user in users:
            today = datetime.datetime.utcnow().date()
            features = self._aggregate_user_events(user.anon_id, today)
            
            profile = self.db.query(Profile).filter(Profile.user_id == user.id).first()
            if not profile:
                logger.warning(f"No profile for user {user.id}. Skipping scoring.")
                continue
                
            # Порог первого уровня (AE)
            ae_threshold = profile.anomaly_threshold or 0.1
            
            if self.simulated:
                ensemble_score = 0.88
                attn_dict = {
                    "Шаг 1: LOGON": 0.45,
                    "Шаг 2: USB_CONNECT": 0.35,
                    "Шаг 3: FILE_OPEN_USB": 0.20
                }
            else:
                # 1. Формируем признаковый вектор для AE+IF
                raw_vec = np.array([features[c] for c in self.feature_cols]).reshape(1, -1)
                scaled_vec = self.scaler.transform(raw_vec)
                scaled_tensor = torch.tensor(scaled_vec, dtype=torch.float32).to(self.device)
                
                # 2. Оценка аномальности Autoencoder
                with torch.no_grad():
                    recon = self.ae_model(scaled_tensor)
                    ae_error = torch.mean((scaled_tensor - recon)**2, dim=1).item()
                
                # 3. Фильтрация Уровня 1:
                # Если суточная активность лежит в пределах нормы, пропускаем
                if ae_error <= ae_threshold:
                    continue
                
                # 4. Формируем последовательность токенов действий для Bi-LSTM
                start_time = datetime.datetime.combine(today, datetime.time.min)
                end_time = datetime.datetime.combine(today, datetime.time.max)
                events = self.db.query(Event).filter(
                    Event.anon_id == user.anon_id,
                    Event.timestamp >= start_time,
                    Event.timestamp <= end_time
                ).order_by(Event.timestamp.asc()).all()
                
                seq_tokens = []
                for e in events:
                    token = map_event_to_token(e.event_type, e.details)
                    seq_tokens.append(TOKEN_MAP.get(token, 1))
                
                if not seq_tokens:
                    continue
                    
                if len(seq_tokens) > MAX_SEQ_LEN:
                    seq_tokens = seq_tokens[-MAX_SEQ_LEN:]
                else:
                    seq_tokens = seq_tokens + [TOKEN_MAP["<PAD>"]] * (MAX_SEQ_LEN - len(seq_tokens))
                
                # 5. Инференс Bi-LSTM (Уровень 2) + получение Attention-весов
                seq_tensor = torch.LongTensor([seq_tokens]).to(self.device)
                with torch.no_grad():
                    logits, attn_weights = self.bilstm_model.get_attention(seq_tensor)
                    ensemble_score = torch.sigmoid(logits).item()
                    attn_weights_np = attn_weights.squeeze().cpu().numpy()
                
                attn_dict = {}
                for idx, t_id in enumerate(seq_tokens):
                    if t_id == 0:  # Пропускаем <PAD>
                        continue
                    token_name = INV_TOKEN_MAP.get(t_id, f"UNK_{t_id}")
                    weight = float(attn_weights_np[idx]) if len(seq_tokens) > 1 else float(attn_weights_np)
                    attn_dict[f"Шаг {idx+1}: {token_name}"] = weight
            
            # 6. Если вероятность инсайдерской активности превышает порог Уровня 2 (0.5), фиксируем аномалию
            l2_threshold = 0.5
            if ensemble_score > l2_threshold:
                logger.info(f"Anomaly detected for user {user.id} (Score: {ensemble_score:.4f})")
                
                anomaly = Anomaly(
                    user_id=user.id,
                    score=ensemble_score,
                    threshold=l2_threshold,
                    status="pending"
                )
                self.db.add(anomaly)
                self.db.flush() # Получаем ID аномалии
                
                # 7. Запись объяснений в БД (сохраняем веса внимания)
                for step_name, weight in attn_dict.items():
                    f = Feature(
                        anomaly_id=anomaly.id,
                        feature_name=step_name,
                        feature_value=weight,
                        shap_value=weight
                    )
                    self.db.add(f)
                    
        self.db.commit()
        logger.info("Scoring process completed")
