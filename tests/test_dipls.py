"""di-PLS (Domain-Invariant PLS) のテスト"""
import numpy as np
import pytest


@pytest.fixture
def domain_data():
    np.random.seed(42)
    X_source = np.random.randn(80, 50)
    X_target = np.random.randn(30, 50) + 1.5
    y_source = X_source[:, 0] * 2 + np.random.randn(80) * 0.5
    return X_source, X_target, y_source


class TestDiPLS:

    def test_predict_shape(self, domain_data):
        """予測の形状が正しいこと"""
        from src.preprocessing.dipls_wrapper import fit_predict_dipls
        X_s, X_t, y_s = domain_data
        preds = fit_predict_dipls(X_s, y_s, X_t, n_components=2, dipls_lambda=0.5)
        assert preds.shape == (30,)

    def test_no_nan(self, domain_data):
        """予測にNaNがないこと"""
        from src.preprocessing.dipls_wrapper import fit_predict_dipls
        X_s, X_t, y_s = domain_data
        preds = fit_predict_dipls(X_s, y_s, X_t, n_components=2, dipls_lambda=0.5)
        assert not np.any(np.isnan(preds))

    def test_different_lambda(self, domain_data):
        """異なるlambdaで動作すること"""
        from src.preprocessing.dipls_wrapper import fit_predict_dipls
        X_s, X_t, y_s = domain_data
        for lam in [0.0, 0.5, 1.0, 10.0]:
            preds = fit_predict_dipls(X_s, y_s, X_t, n_components=2, dipls_lambda=lam)
            assert preds.shape == (30,)

    def test_lambda0_correlated_with_pls(self, domain_data):
        """lambda=0のとき通常PLSと高い相関を持つこと"""
        from src.preprocessing.dipls_wrapper import fit_predict_dipls
        from sklearn.cross_decomposition import PLSRegression
        X_s, X_t, y_s = domain_data
        preds_dipls = fit_predict_dipls(X_s, y_s, X_t, n_components=2, dipls_lambda=0.0)
        pls = PLSRegression(n_components=2)
        pls.fit(X_s, y_s)
        preds_pls = pls.predict(X_t).ravel()
        # デフレーション方式の差で完全一致しないが高い相関を持つ
        corr = np.corrcoef(preds_dipls, preds_pls)[0, 1]
        assert corr > 0.9
