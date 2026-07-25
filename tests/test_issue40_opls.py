"""OPLS (Orthogonal Partial Least Squares) のテスト"""
import numpy as np
import pytest


@pytest.fixture
def regression_data():
    np.random.seed(42)
    n, p = 100, 50
    X = np.random.randn(n, p)
    # yに関連する成分 + 直交ノイズ
    y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(n) * 0.5
    return X, y


class TestOPLS:

    def test_filter_shape(self, regression_data):
        """フィルタ後の形状が同じこと"""
        from src.preprocessing.issue40_opls import OPLSFilter
        X, y = regression_data
        opls = OPLSFilter(n_components=2)
        X_filtered = opls.fit_transform(X, y)
        assert X_filtered.shape == X.shape

    def test_transform_shape(self, regression_data):
        """transform後の形状が正しいこと"""
        from src.preprocessing.issue40_opls import OPLSFilter
        X, y = regression_data
        opls = OPLSFilter(n_components=2)
        opls.fit(X, y)
        X_new = np.random.randn(20, 50)
        X_filtered = opls.transform(X_new)
        assert X_filtered.shape == (20, 50)

    def test_no_nan(self, regression_data):
        """NaNがないこと"""
        from src.preprocessing.issue40_opls import OPLSFilter
        X, y = regression_data
        opls = OPLSFilter(n_components=2)
        X_filtered = opls.fit_transform(X, y)
        assert not np.any(np.isnan(X_filtered))

    def test_removes_orthogonal_variation(self, regression_data):
        """yに直交する変動が減少すること"""
        from src.preprocessing.issue40_opls import OPLSFilter
        X, y = regression_data
        opls = OPLSFilter(n_components=3)
        X_filtered = opls.fit_transform(X, y)
        # フィルタ後のXはyとの相関が維持されつつ、分散が減少
        var_before = np.sum(np.var(X, axis=0))
        var_after = np.sum(np.var(X_filtered, axis=0))
        assert var_after < var_before

    def test_different_n_components(self, regression_data):
        """n_componentsを変えても動作すること"""
        from src.preprocessing.issue40_opls import OPLSFilter
        X, y = regression_data
        for n in [1, 3, 5]:
            opls = OPLSFilter(n_components=n)
            X_filtered = opls.fit_transform(X, y)
            assert X_filtered.shape == X.shape

    def test_orthogonal_weights(self, regression_data):
        """直交重みがyに直交していること"""
        from src.preprocessing.issue40_opls import OPLSFilter
        X, y = regression_data
        opls = OPLSFilter(n_components=2)
        opls.fit(X, y)
        # 直交スコアとyの相関がほぼ0
        T_orth = X @ opls.W_orth_
        for j in range(T_orth.shape[1]):
            corr = np.abs(np.corrcoef(T_orth[:, j], y)[0, 1])
            assert corr < 0.15  # ほぼ無相関
