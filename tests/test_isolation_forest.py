"""Tests for Isolation Forest wrapper."""
import sys
sys.path.insert(0, r'D:\Политех\Мага\Дипломы\М\insider_threat_recongition\backend')

import numpy as np
from app.models.isolation_forest import train_isolation_forest, score_samples


class TestIsolationForest:
    def test_train_and_score(self):
        """Should train and return scores for new data."""
        np.random.seed(42)
        X_train = np.random.randn(200, 10)
        X_test = np.random.randn(50, 10)

        model = train_isolation_forest(X_train, contamination=0.05)
        scores = score_samples(model, X_test)

        assert scores.shape == (50,), f"Expected (50,), got {scores.shape}"
        assert 0 <= scores.min() <= 1, f"Scores should be in [0,1], got {scores.min():.4f}"
        assert 0 <= scores.max() <= 1, f"Scores should be in [0,1], got {scores.max():.4f}"

    def test_outliers_get_higher_scores(self):
        """Outliers should score higher than normal points."""
        np.random.seed(42)
        normal = np.random.randn(300, 8)
        outliers = np.random.randn(10, 8) * 10

        model = train_isolation_forest(normal, contamination=0.05)
        normal_scores = score_samples(model, normal)
        outlier_scores = score_samples(model, outliers)

        assert outlier_scores.mean() > normal_scores.mean(), \
            f"Expected outliers ({outlier_scores.mean():.4f}) > normal ({normal_scores.mean():.4f})"

    def test_deterministic_with_seed(self):
        """Should give same results with random_state=42."""
        np.random.seed(42)
        X = np.random.randn(100, 5)

        model1 = train_isolation_forest(X, contamination=0.1)
        model2 = train_isolation_forest(X, contamination=0.1)

        s1 = score_samples(model1, X)
        s2 = score_samples(model2, X)
        np.testing.assert_array_almost_equal(s1, s2)
