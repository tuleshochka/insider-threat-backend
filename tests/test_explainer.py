"""Tests for the SHAP explainer."""
import sys
sys.path.insert(0, r'D:\Политех\Мага\Дипломы\М\insider_threat_recongition\backend')

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from app.models.autoencoder import Autoencoder, train_autoencoder
from app.services.explainer import Explainer


class TestExplainer:
    def test_explain_returns_features(self):
        """explain() should return a list of feature dicts."""
        np.random.seed(42)
        torch.manual_seed(42)

        # Train a small model
        normal = torch.rand(300, 8)
        loader = DataLoader(TensorDataset(normal), batch_size=32)
        model = Autoencoder(input_dim=8)
        train_autoencoder(model, loader, None, epochs=5, lr=0.01, patience=2)

        bg = np.random.randn(50, 8)
        anomaly = np.random.randn(8) * 3 + 5

        explainer = Explainer(model, bg)
        result = explainer.explain(anomaly)

        assert len(result) == 8, f"Expected 8 features, got {len(result)}"
        for item in result:
            assert "feature_name" in item
            assert "feature_value" in item
            assert "shap_value" in item

    def test_result_sorted_by_importance(self):
        """Results should be sorted by absolute shap value descending."""
        torch.manual_seed(42)
        normal = torch.rand(100, 6)
        loader = DataLoader(TensorDataset(normal), batch_size=32)
        model = Autoencoder(input_dim=6)
        train_autoencoder(model, loader, None, epochs=5, lr=0.01, patience=2)

        explainer = Explainer(model, np.random.randn(30, 6))
        result = explainer.explain(np.random.randn(6))

        abs_vals = [abs(r["shap_value"]) for r in result]
        for i in range(len(abs_vals) - 1):
            assert abs_vals[i] >= abs_vals[i + 1], f"Not sorted at index {i}"
