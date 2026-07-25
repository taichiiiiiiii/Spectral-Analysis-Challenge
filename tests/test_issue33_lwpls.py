"""Issue #33: LWPLSのテスト"""
import numpy as np
import pytest


class TestLWPLS:

    def test_lwpls_predict_shape(self):
        """LWPLS予測の形状が正しいこと"""
        from src.modeling.issue33_lwpls import lwpls_predict
        np.random.seed(42)
        X_train = np.random.randn(100, 50)
        y_train = np.random.randn(100)
        X_test = np.random.randn(10, 50)
        preds = lwpls_predict(X_train, y_train, X_test, n_components=2, k=30)
        assert preds.shape == (10,)

    def test_lwpls_no_nan(self):
        """LWPLSの予測にNaNがないこと"""
        from src.modeling.issue33_lwpls import lwpls_predict
        np.random.seed(42)
        X_train = np.random.randn(100, 50)
        y_train = np.random.randn(100)
        X_test = np.random.randn(10, 50)
        preds = lwpls_predict(X_train, y_train, X_test, n_components=2, k=30)
        assert not np.any(np.isnan(preds))

    def test_lwpls_different_k(self):
        """k値を変えても動作すること"""
        from src.modeling.issue33_lwpls import lwpls_predict
        np.random.seed(42)
        X_train = np.random.randn(100, 50)
        y_train = np.random.randn(100)
        X_test = np.random.randn(5, 50)
        for k in [20, 50, 80]:
            preds = lwpls_predict(X_train, y_train, X_test, n_components=2, k=k)
            assert preds.shape == (5,)
