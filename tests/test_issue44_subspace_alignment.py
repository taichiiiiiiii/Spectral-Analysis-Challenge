"""Subspace Alignment のテスト"""
import numpy as np
import pytest


@pytest.fixture
def domain_data():
    np.random.seed(42)
    X_source = np.random.randn(80, 50)
    X_target = np.random.randn(30, 50) + 2.0
    return X_source, X_target


class TestSubspaceAlignment:

    def test_output_shape(self, domain_data):
        """変換後の形状が正しいこと"""
        from src.preprocessing.issue44_subspace_alignment import subspace_align
        X_s, X_t = domain_data
        Z_s, Z_t = subspace_align(X_s, X_t, n_components=10)
        assert Z_s.shape == (80, 10)
        assert Z_t.shape == (30, 10)

    def test_no_nan(self, domain_data):
        """NaNがないこと"""
        from src.preprocessing.issue44_subspace_alignment import subspace_align
        X_s, X_t = domain_data
        Z_s, Z_t = subspace_align(X_s, X_t, n_components=10)
        assert not np.any(np.isnan(Z_s))
        assert not np.any(np.isnan(Z_t))

    def test_reduces_domain_distance(self, domain_data):
        """ドメイン間距離が縮まること"""
        from src.preprocessing.issue44_subspace_alignment import subspace_align
        X_s, X_t = domain_data
        dist_before = np.linalg.norm(X_s.mean(axis=0) - X_t.mean(axis=0))
        Z_s, Z_t = subspace_align(X_s, X_t, n_components=10)
        dist_after = np.linalg.norm(Z_s.mean(axis=0) - Z_t.mean(axis=0))
        assert dist_after < dist_before

    def test_different_n_components(self, domain_data):
        """異なるn_componentsで動作すること"""
        from src.preprocessing.issue44_subspace_alignment import subspace_align
        X_s, X_t = domain_data
        for n in [5, 10, 20]:
            Z_s, Z_t = subspace_align(X_s, X_t, n_components=n)
            assert Z_s.shape[1] == n
            assert Z_t.shape[1] == n
