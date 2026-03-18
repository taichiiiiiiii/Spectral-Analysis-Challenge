"""KMM (Kernel Mean Matching) のテスト"""
import numpy as np
import pytest


@pytest.fixture
def domain_data():
    np.random.seed(42)
    X_source = np.random.randn(80, 50)
    X_target = np.random.randn(30, 50) + 2.0
    y_source = X_source[:, 0] * 2 + np.random.randn(80) * 0.5
    return X_source, X_target, y_source


class TestKMM:

    def test_weights_shape(self, domain_data):
        """重みの形状が正しいこと"""
        from src.preprocessing.issue50_kmm import compute_kmm_weights
        X_s, X_t, _ = domain_data
        weights = compute_kmm_weights(X_s, X_t)
        assert weights.shape == (80,)

    def test_weights_positive(self, domain_data):
        """重みが非負であること"""
        from src.preprocessing.issue50_kmm import compute_kmm_weights
        X_s, X_t, _ = domain_data
        weights = compute_kmm_weights(X_s, X_t)
        assert (weights >= -1e-6).all()

    def test_weights_no_nan(self, domain_data):
        """重みにNaNがないこと"""
        from src.preprocessing.issue50_kmm import compute_kmm_weights
        X_s, X_t, _ = domain_data
        weights = compute_kmm_weights(X_s, X_t)
        assert not np.any(np.isnan(weights))

    def test_weighted_pls_predict(self, domain_data):
        """重み付きPLSで予測できること"""
        from src.preprocessing.issue50_kmm import compute_kmm_weights, weighted_pls_predict
        X_s, X_t, y_s = domain_data
        weights = compute_kmm_weights(X_s, X_t)
        preds = weighted_pls_predict(X_s, y_s, X_t, weights, n_components=2)
        assert preds.shape == (30,)
        assert not np.any(np.isnan(preds))

    def test_different_B(self, domain_data):
        """異なるB値で動作すること"""
        from src.preprocessing.issue50_kmm import compute_kmm_weights
        X_s, X_t, _ = domain_data
        for B in [5.0, 10.0, 50.0]:
            weights = compute_kmm_weights(X_s, X_t, B=B)
            assert (weights <= B + 1e-6).all()
