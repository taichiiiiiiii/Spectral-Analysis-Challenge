"""Weighted TCA のテスト"""
import numpy as np
import pytest


@pytest.fixture
def domain_data():
    """ソース・ターゲットドメインのダミーデータ"""
    np.random.seed(42)
    X_source = np.random.randn(80, 50)
    X_target = np.random.randn(30, 50) + 2.0
    y_source = X_source[:, 0] * 2 + np.random.randn(80) * 0.5
    return X_source, X_target, y_source


class TestWeightedTCA:

    def test_output_shape(self, domain_data):
        """変換後の形状が正しいこと"""
        from src.preprocessing.weighted_tca import weighted_tca_transform
        X_s, X_t, y_s = domain_data
        Z_s, Z_t = weighted_tca_transform(X_s, X_t, y_s, n_components=5)
        assert Z_s.shape == (80, 5)
        assert Z_t.shape == (30, 5)

    def test_no_nan(self, domain_data):
        """変換後にNaNがないこと"""
        from src.preprocessing.weighted_tca import weighted_tca_transform
        X_s, X_t, y_s = domain_data
        Z_s, Z_t = weighted_tca_transform(X_s, X_t, y_s, n_components=5)
        assert not np.any(np.isnan(Z_s))
        assert not np.any(np.isnan(Z_t))

    def test_reduces_domain_distance(self, domain_data):
        """変換後にドメイン間の平均距離が縮まること"""
        from src.preprocessing.weighted_tca import weighted_tca_transform
        X_s, X_t, y_s = domain_data
        dist_before = np.linalg.norm(X_s.mean(axis=0) - X_t.mean(axis=0))
        Z_s, Z_t = weighted_tca_transform(X_s, X_t, y_s, n_components=10)
        dist_after = np.linalg.norm(Z_s.mean(axis=0) - Z_t.mean(axis=0))
        assert dist_after < dist_before

    def test_weights_affect_result(self, domain_data):
        """重みが結果に影響すること（uniform vs custom）"""
        from src.preprocessing.weighted_tca import weighted_tca_transform
        X_s, X_t, y_s = domain_data
        # デフォルト（KMM的な重み）
        Z_s1, _ = weighted_tca_transform(X_s, X_t, y_s, n_components=5)
        # 均一重み
        w_uniform = np.ones(len(X_s))
        Z_s2, _ = weighted_tca_transform(
            X_s, X_t, y_s, n_components=5, source_weights=w_uniform
        )
        assert not np.allclose(Z_s1, Z_s2)

    def test_custom_weights(self, domain_data):
        """カスタム重みを渡せること"""
        from src.preprocessing.weighted_tca import weighted_tca_transform
        X_s, X_t, y_s = domain_data
        weights = np.random.rand(80) + 0.1
        Z_s, Z_t = weighted_tca_transform(
            X_s, X_t, y_s, n_components=5, source_weights=weights
        )
        assert Z_s.shape == (80, 5)
