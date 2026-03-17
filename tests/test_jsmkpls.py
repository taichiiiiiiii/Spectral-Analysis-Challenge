"""JSMKPLS (Joint Statistical and Manifold alignment in Kernel PLS subspace) のテスト"""
import numpy as np
import pytest


@pytest.fixture
def domain_data():
    np.random.seed(42)
    X_source = np.random.randn(60, 50)
    X_target = np.random.randn(20, 50) + 2.0
    return X_source, X_target


class TestJSMKPLS:

    def test_output_shape(self, domain_data):
        """変換後の形状が正しいこと"""
        from src.preprocessing.jsmkpls import jsmkpls_transform
        X_s, X_t = domain_data
        Z_s, Z_t = jsmkpls_transform(X_s, X_t, n_components=5)
        assert Z_s.shape == (60, 5)
        assert Z_t.shape == (20, 5)

    def test_no_nan(self, domain_data):
        """NaNがないこと"""
        from src.preprocessing.jsmkpls import jsmkpls_transform
        X_s, X_t = domain_data
        Z_s, Z_t = jsmkpls_transform(X_s, X_t, n_components=5)
        assert not np.any(np.isnan(Z_s))
        assert not np.any(np.isnan(Z_t))

    def test_reduces_domain_distance(self, domain_data):
        """ドメイン間距離が縮まること"""
        from src.preprocessing.jsmkpls import jsmkpls_transform
        X_s, X_t = domain_data
        dist_before = np.linalg.norm(X_s.mean(axis=0) - X_t.mean(axis=0))
        Z_s, Z_t = jsmkpls_transform(X_s, X_t, n_components=10)
        dist_after = np.linalg.norm(Z_s.mean(axis=0) - Z_t.mean(axis=0))
        assert dist_after < dist_before

    def test_different_n_components(self, domain_data):
        """異なるn_componentsで動作すること"""
        from src.preprocessing.jsmkpls import jsmkpls_transform
        X_s, X_t = domain_data
        for n in [3, 5, 10]:
            Z_s, Z_t = jsmkpls_transform(X_s, X_t, n_components=n)
            assert Z_s.shape[1] == n

    def test_hyperparameters(self, domain_data):
        """ハイパーパラメータを変えても動作すること"""
        from src.preprocessing.jsmkpls import jsmkpls_transform
        X_s, X_t = domain_data
        Z_s1, _ = jsmkpls_transform(X_s, X_t, n_components=5, mu=0.1, lam=0.1)
        Z_s2, _ = jsmkpls_transform(X_s, X_t, n_components=5, mu=10.0, lam=10.0)
        assert not np.allclose(Z_s1, Z_s2)
