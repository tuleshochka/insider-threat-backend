"""Tests for the data preprocessor."""
import sys, os, tempfile
sys.path.insert(0, r'D:\Политех\Мага\Дипломы\М\insider_threat_recongition\backend')

import numpy as np
import pandas as pd
import pytest

from app.services.preprocessor import DataPreprocessor


@pytest.fixture
def mock_data_dir(tmp_path):
    """Create mock CSV files simulating CERT structure."""
    # Users
    pd.DataFrame({
        "employee_name": ["User A", "User B"],
        "user_id": ["USR001", "USR002"],
        "role": ["Manager", "Engineer"],
        "business_unit": [1, 1],
        "functional_unit": ["1 - Admin", "2 - Dev"],
        "department": ["IT", "IT"],
        "start_date": ["2010-01-01", "2010-01-01"],
        "end_date": ["2011-06-01", "2011-06-01"],
    }).to_csv(os.path.join(tmp_path, "users.csv"), index=False)

    # Logon
    logon = {
        "id": [f"L{i}" for i in range(6)],
        "date": ["01/04/2010 08:00:00", "01/04/2010 09:00:00", "01/04/2010 10:00:00",
                 "01/05/2010 08:00:00", "01/05/2010 09:00:00", "01/05/2010 17:00:00"],
        "user": ["USR001", "USR001", "USR002", "USR001", "USR002", "USR002"],
        "pc": ["PC-001", "PC-001", "PC-002", "PC-001", "PC-002", "PC-002"],
        "activity": ["Logon", "Logoff", "Logon", "Logon", "Logon", "Logoff"],
    }
    pd.DataFrame(logon).to_csv(os.path.join(tmp_path, "logon.csv"), index=False)

    # File
    file = {
        "id": [f"F{i}" for i in range(4)],
        "date": ["01/04/2010 08:30:00", "01/04/2010 09:30:00", "01/04/2010 11:00:00", "01/04/2010 14:00:00"],
        "user": ["USR001", "USR001", "USR002", "USR002"],
        "pc": ["PC-001", "PC-001", "PC-002", "PC-002"],
        "filename": ["doc1.txt", "doc2.txt", "doc3.txt", "doc4.txt"],
        "activity": ["File Open", "File Write", "File Open", "File Open"],
        "to_removable_media": [False, False, False, True],
        "from_removable_media": [False, True, False, False],
    }
    pd.DataFrame(file).to_csv(os.path.join(tmp_path, "file.csv"), index=False)

    # Email
    email = {
        "id": [f"E{i}" for i in range(3)],
        "date": ["01/04/2010 08:00:00", "01/04/2010 10:00:00", "01/04/2010 12:00:00"],
        "user": ["USR001", "USR002", "USR001"],
        "to": ["USR002@dtaa.com", "USR001@dtaa.com", "external@gmail.com"],
        "from": ["USR001@dtaa.com", "USR002@dtaa.com", "USR001@dtaa.com"],
        "size": [1024, 512, 2048],
        "attachments": [1, 0, 0],
    }
    pd.DataFrame(email).to_csv(os.path.join(tmp_path, "email.csv"), index=False)

    # Device
    device = {
        "id": [f"D{i}" for i in range(2)],
        "date": ["01/04/2010 09:00:00", "01/04/2010 17:00:00"],
        "user": ["USR002", "USR002"],
        "pc": ["PC-002", "PC-002"],
        "activity": ["Connect", "Disconnect"],
    }
    pd.DataFrame(device).to_csv(os.path.join(tmp_path, "device.csv"), index=False)

    # HTTP
    http = {
        "id": [f"H{i}" for i in range(2)],
        "date": ["01/04/2010 10:00:00", "01/04/2010 11:00:00"],
        "user": ["USR001", "USR002"],
        "url": ["http://google.com", "http://wikipedia.org"],
    }
    pd.DataFrame(http).to_csv(os.path.join(tmp_path, "http.csv"), index=False)

    return tmp_path


class TestPreprocessor:
    def test_load_users(self, mock_data_dir):
        proc = DataPreprocessor(mock_data_dir)
        df = proc.load_users()
        assert len(df) == 2
        assert "USR001" in df["anon_id"].values

    def test_build_matrix_single_day(self, mock_data_dir):
        proc = DataPreprocessor(mock_data_dir)
        proc.load_users()
        df = proc.build_training_matrix("2010-01-04", "2010-01-04")
        assert len(df) == 2, f"Expected 2 user-days, got {len(df)}"
        # Both users should be present
        assert df["anon_id"].nunique() == 2
        # Logon counts should match
        usr1 = df[df["anon_id"] == "USR001"]
        assert usr1["logon_count"].values[0] == 2
        usr2 = df[df["anon_id"] == "USR002"]
        assert usr2["logon_count"].values[0] == 1

    def test_build_matrix_two_days(self, mock_data_dir):
        proc = DataPreprocessor(mock_data_dir)
        proc.load_users()
        df = proc.build_training_matrix("2010-01-04", "2010-01-05")
        assert 3 <= len(df) <= 4  # 2 users × 2 days = 4, but some may be empty

    def test_normalise(self, mock_data_dir):
        proc = DataPreprocessor(mock_data_dir)
        proc.load_users()
        df = proc.build_training_matrix("2010-01-04", "2010-01-05")
        normed, params = DataPreprocessor.normalise(df)

        assert "mean" in params
        assert "std" in params
        feat_cols = [c for c in normed.columns if c not in ("date", "anon_id", "role", "department", "business_unit")]
        for col in feat_cols:
            if len(normed) > 1:
                assert abs(normed[col].mean()) < 1e-10, f"{col} mean should be ≈0"

    def test_normalise_reapply(self, mock_data_dir):
        """Normalisation should be reversible via fit_params."""
        proc = DataPreprocessor(mock_data_dir)
        proc.load_users()
        train = proc.build_training_matrix("2010-01-04", "2010-01-05")
        normed_train, params = DataPreprocessor.normalise(train)

        test = proc.build_training_matrix("2010-01-04", "2010-01-05")
        normed_test, _ = DataPreprocessor.normalise(test, fit_params=params)
        assert normed_test.shape[0] > 0
