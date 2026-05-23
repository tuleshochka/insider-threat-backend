"""Tests for the PyTorch autoencoder."""
import sys
sys.path.insert(0, r'D:\Политех\Мага\Дипломы\М\insider_threat_recongition\backend')

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from app.models.autoencoder import Autoencoder, train_autoencoder, compute_anomaly_scores


def _random_data(n=200, dim=16) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.rand(n, dim)


class TestAutoencoder:
    def test_architecture(self):
        """Model should have encoder and decoder with correct dimensions."""
        model = Autoencoder(input_dim=16, hidden_dim=8)
        assert model.encoder is not None
        assert model.decoder is not None

        x = torch.rand(4, 16)
        out = model(x)
        assert out.shape == (4, 16), f"Expected (4,16), got {out.shape}"

    def test_reconstruction_small_loss(self):
        """After training, reconstruction loss should be low."""
        data = _random_data(1000, 24)
        train = data[:800]
        val = data[800:]
        train_loader = DataLoader(TensorDataset(train), batch_size=64)
        val_loader = DataLoader(TensorDataset(val), batch_size=64)

        model = Autoencoder(input_dim=24, hidden_dim=8)
        history = train_autoencoder(model, train_loader, val_loader, epochs=20, lr=0.01, patience=3)

        assert len(history["train_loss"]) > 0
        assert history["train_loss"][-1] < 0.1, f"Final train loss too high: {history['train_loss'][-1]:.4f}"
        assert history["val_loss"][-1] < 0.15, f"Final val loss too high: {history['val_loss'][-1]:.4f}"

    def test_anomaly_scores_higher_for_outliers(self):
        """Outliers should have higher anomaly scores than normal data."""
        torch.manual_seed(42)
        normal = torch.rand(500, 16)
        outliers = torch.rand(20, 16) * 5 + 10  # far from normal range

        train_loader = DataLoader(TensorDataset(normal), batch_size=64)
        model = Autoencoder(input_dim=16)
        train_autoencoder(model, train_loader, None, epochs=10, lr=0.01, patience=3)

        normal_loader = DataLoader(TensorDataset(normal), batch_size=64)
        outlier_loader = DataLoader(TensorDataset(outliers), batch_size=64)

        normal_scores = np.array(compute_anomaly_scores(model, normal_loader))
        outlier_scores = np.array(compute_anomaly_scores(model, outlier_loader))

        assert outlier_scores.mean() > normal_scores.mean(), \
            f"Expected outliers ({outlier_scores.mean():.4f}) > normal ({normal_scores.mean():.4f})"

    def test_different_hidden_dim(self):
        """Should work with various hidden dimensions."""
        for hidden in [4, 8, 16, 32]:
            model = Autoencoder(input_dim=32, hidden_dim=hidden)
            x = torch.rand(2, 32)
            out = model(x)
            assert out.shape == (2, 32), f"Failed for hidden_dim={hidden}"
