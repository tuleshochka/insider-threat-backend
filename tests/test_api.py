"""Tests for FastAPI endpoints (using SQLite for local testing)."""
import sys, os
sys.path.insert(0, r'D:\Политех\Мага\Дипломы\М\insider_threat_recongition\backend')

os.environ["DB_TYPE"] = "sqlite"

from fastapi.testclient import TestClient
from main import app
from app.models.db import engine, Base

Base.metadata.create_all(bind=engine)

client = TestClient(app)


class TestHealth:
    def test_health_returns_ok(self):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_has_version(self):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        assert "version" in resp.json()


class TestTrain:
    def test_trigger_training_returns_202(self):
        resp = client.post("/api/v1/train")
        assert resp.status_code == 202
        data = resp.json()
        assert "task_id" in data

    def test_training_status_returns_running(self):
        resp = client.get("/api/v1/train/some_task_123")
        assert resp.status_code == 200
        assert "status" in resp.json()


class TestUsers:
    def test_list_users_returns_paginated(self):
        resp = client.get("/api/v1/users")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "total" in data

    def test_user_detail_404(self):
        resp = client.get("/api/v1/users/999999")
        assert resp.status_code == 404


class TestAnomalies:
    def test_list_anomalies_returns_list(self):
        resp = client.get("/api/v1/anomalies")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_anomaly_detail_404(self):
        resp = client.get("/api/v1/anomalies/99999")
        assert resp.status_code == 404

    def test_invalid_review_status(self):
        resp = client.patch("/api/v1/anomalies/99999", json={"status": "invalid"})
        assert resp.status_code in (404, 422)


class TestDashboard:
    def test_dashboard_returns_stats(self):
        resp = client.get("/api/v1/dashboard")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_anomalies" in data
        assert "pending_anomalies" in data
        assert "max_score" in data
