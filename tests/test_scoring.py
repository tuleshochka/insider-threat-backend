import pytest
from unittest.mock import MagicMock
from app.services.scoring import ScoringService
from app.models.orm import User, Profile, Anomaly, Feature, Event

def test_scoring_service_initialization():
    db = MagicMock()
    service = ScoringService(db)
    
    # Assert models were loaded successfully and simulated=False
    assert service.simulated is False
    assert hasattr(service, 'ae_model')
    assert hasattr(service, 'bilstm_model')
    assert hasattr(service, 'scaler')
    assert hasattr(service, 'feature_cols')

def test_scoring_service_inference():
    db = MagicMock()
    
    # Mocking User
    mock_user = User(id=1, anon_id="mock_user_1", is_active=True)
    db.query.return_value.filter.return_value.all.return_value = [mock_user]
    
    # Mocking Profile
    mock_profile = Profile(id=1, user_id=1, anomaly_threshold=0.01) # Very low to force anomaly
    db.query.return_value.filter.return_value.first.return_value = mock_profile
    
    # Mocking Events for sequence generation
    mock_events = [
        Event(anon_id="mock_user_1", event_type="logon", details={"activity": "logon"}),
        Event(anon_id="mock_user_1", event_type="device", details={"activity": "connect"}),
        Event(anon_id="mock_user_1", event_type="file", details={"activity": "open", "is_usb": True}),
    ]
    
    # Override query behavior
    def mock_query(model):
        m = MagicMock()
        if model == User:
            m.filter.return_value.all.return_value = [mock_user]
        elif model == Profile:
            m.filter.return_value.first.return_value = mock_profile
        else: # Event
            m.filter.return_value.all.return_value = mock_events
            m.filter.return_value.order_by.return_value.all.return_value = mock_events
        return m
        
    db.query.side_effect = mock_query
    
    # Create Anomaly id simulation
    def add_side_effect(obj):
        if isinstance(obj, Anomaly):
            obj.id = 100
    db.add.side_effect = add_side_effect
    
    service = ScoringService(db)
    service.update_baselines_and_score()
    
    # Check if an Anomaly and Features were added
    add_calls = db.add.call_args_list
    assert len(add_calls) >= 1
    
    added_objects = [call[0][0] for call in add_calls]
    anomalies = [obj for obj in added_objects if isinstance(obj, Anomaly)]
    features = [obj for obj in added_objects if isinstance(obj, Feature)]
    
    assert len(anomalies) == 1
    assert anomalies[0].user_id == 1
    assert anomalies[0].score > 0.0
    
    assert len(features) > 0
    feature_names = [f.feature_name for f in features]
    # Check that at least the first step is recorded
    assert any("Шаг 1: LOGON" in name for name in feature_names)
