import logging
import datetime
import pandas as pd
import numpy as np
import shap
from sqlalchemy.orm import Session

from app.models.orm import Event, Profile, Anomaly, Feature, User
from app.config import settings

logger = logging.getLogger(__name__)

class ScoringService:
    def __init__(self, db: Session):
        self.db = db
        # In a real app, models would be loaded from disk or MLflow here
        # self.rf_model = load_model("rf_smote")
        # self.xgb_model = load_model("xgb_smote")
        # self.lgb_model = load_model("lgb_smote")
        # self.explainer = shap.TreeExplainer(self.xgb_model)
        pass

    def _aggregate_user_events(self, user_id: int, date: datetime.date) -> dict:
        """
        Aggregate raw events into daily features.
        """
        events = self.db.query(Event).filter(
            Event.anon_id == str(user_id), # Assuming anon_id matches for simplicity
            # Normally we'd filter by date
        ).all()
        
        # Simple aggregation placeholder
        logon_count = sum(1 for e in events if e.event_type == 'logon')
        file_ops = sum(1 for e in events if e.event_type == 'file')
        http_reqs = sum(1 for e in events if e.event_type == 'http')
        
        return {
            "logon_count": logon_count,
            "file_operations": file_ops,
            "http_requests": http_reqs,
            # we'd add after_hours logic etc.
        }

    def update_baselines_and_score(self):
        """
        Run nightly or hourly to update baselines, extract features,
        run the ensemble model and flag anomalies.
        """
        logger.info("Starting background scoring process")
        # 1. Fetch all active users
        users = self.db.query(User).filter(User.is_active == True).all()
        
        for user in users:
            # 2. Aggregate features for the current period
            today = datetime.datetime.utcnow().date()
            features = self._aggregate_user_events(user.id, today)
            
            # 3. Fetch user profile (baseline)
            profile = self.db.query(Profile).filter(Profile.user_id == user.id).first()
            if not profile:
                logger.warning(f"No profile for user {user.id}. Skipping scoring.")
                continue
            
            # 4. Compute z-scores and lags
            # This requires historical data which we'd normally get from a Time Series DB or ClickHouse
            z_scores = {
                "logon_count_zscore": 0.5, # Placeholder
                "file_operations_zscore": 1.2,
                "http_requests_zscore": 3.5 # High deviation placeholder
            }
            
            # Combine all features into a vector
            # feature_vector = np.array([...])
            
            # 5. Run Ensemble Inference
            # p_rf = self.rf_model.predict_proba(feature_vector)[:, 1]
            # p_xgb = self.xgb_model.predict_proba(feature_vector)[:, 1]
            # p_lgb = self.lgb_model.predict_proba(feature_vector)[:, 1]
            # ensemble_score = float((p_rf + p_xgb + p_lgb) / 3.0)
            ensemble_score = 0.88 # Simulated high risk score
            
            threshold = profile.anomaly_threshold or 0.85
            
            # 6. Flag Anomaly if threshold exceeded
            if ensemble_score > threshold:
                logger.info(f"Anomaly detected for user {user.id} (Score: {ensemble_score})")
                
                # Create Anomaly record
                anomaly = Anomaly(
                    user_id=user.id,
                    score=ensemble_score,
                    threshold=threshold,
                    status="pending"
                )
                self.db.add(anomaly)
                self.db.flush() # Get anomaly.id
                
                # 7. Compute SHAP values for Explainability
                # shap_values = self.explainer.shap_values(feature_vector)
                
                # Simulated SHAP values
                shap_data = [
                    {"name": "http_requests_zscore", "value": 3.5, "shap": 0.45},
                    {"name": "after_hours_ratio", "value": 0.8, "shap": 0.21},
                    {"name": "device_operations_lag3_avg", "value": 15, "shap": 0.15}
                ]
                
                for sd in shap_data:
                    f = Feature(
                        anomaly_id=anomaly.id,
                        feature_name=sd["name"],
                        feature_value=sd["value"],
                        shap_value=sd["shap"]
                    )
                    self.db.add(f)
                    
        self.db.commit()
        logger.info("Scoring process completed")
