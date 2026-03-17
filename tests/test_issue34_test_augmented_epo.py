"""Issue #34: Test-Augmented EPOのテスト"""
import numpy as np
import pytest


class TestTestAugmentedEPO:

    def test_augmented_epo_shape(self):
        """変換後のデータ形状が正しいこと"""
        from src.modeling.issue34_test_augmented_epo import compute_augmented_epo_projection, apply_epo
        np.random.seed(42)
        X_train = np.random.randn(100, 50)
        X_test = np.random.randn(30, 50)
        groups = np.random.choice(["A", "B", "C"], 100)

        P = compute_augmented_epo_projection(X_train, groups, X_test, n_components=1)
        X_tr = X_train @ P
        X_te = X_test @ P
        assert X_tr.shape == X_train.shape
        assert X_te.shape == X_test.shape

    def test_augmented_epo_no_nan(self):
        """変換後にNaNがないこと"""
        from src.modeling.issue34_test_augmented_epo import compute_augmented_epo_projection
        np.random.seed(42)
        X_train = np.random.randn(100, 50)
        X_test = np.random.randn(30, 50)
        groups = np.random.choice(["A", "B", "C"], 100)

        P = compute_augmented_epo_projection(X_train, groups, X_test, n_components=1)
        assert not np.any(np.isnan(P))

    def test_augmented_epo_projection_symmetric(self):
        """投影行列が対称であること"""
        from src.modeling.issue34_test_augmented_epo import compute_augmented_epo_projection
        np.random.seed(42)
        X_train = np.random.randn(100, 50)
        X_test = np.random.randn(30, 50)
        groups = np.random.choice(["A", "B", "C"], 100)

        P = compute_augmented_epo_projection(X_train, groups, X_test, n_components=1)
        assert np.allclose(P, P.T, atol=1e-10)
