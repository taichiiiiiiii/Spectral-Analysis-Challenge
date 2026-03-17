"""Constituent EMSC のテスト"""
import numpy as np
import pytest


@pytest.fixture
def dummy_spectra():
    np.random.seed(42)
    n_samples = 30
    n_features = 100
    wn = np.linspace(4300, 10000, n_features)
    # ベースライン + 成分信号 + ノイズ
    baseline = 0.5 + 0.001 * (wn - 7000)
    X = np.tile(baseline, (n_samples, 1)) + np.random.randn(n_samples, n_features) * 0.05
    X += np.random.randn(n_samples, 1) * 0.3  # 乗算的変動
    return X, wn


class TestConstituentEMSC:

    def test_output_shape(self, dummy_spectra):
        """補正後の形状が同じこと"""
        from src.preprocessing.constituent_emsc import constituent_emsc
        X, wn = dummy_spectra
        X_corr = constituent_emsc(X, wavenumbers=wn)
        assert X_corr.shape == X.shape

    def test_no_nan(self, dummy_spectra):
        """NaNがないこと"""
        from src.preprocessing.constituent_emsc import constituent_emsc
        X, wn = dummy_spectra
        X_corr = constituent_emsc(X, wavenumbers=wn)
        assert not np.any(np.isnan(X_corr))

    def test_reduces_scatter_variation(self, dummy_spectra):
        """散乱変動が減少すること"""
        from src.preprocessing.constituent_emsc import constituent_emsc
        X, wn = dummy_spectra
        X_corr = constituent_emsc(X, wavenumbers=wn)
        # 補正後はサンプル間分散が減少するはず
        var_before = np.var(X, axis=0).mean()
        var_after = np.var(X_corr, axis=0).mean()
        assert var_after < var_before

    def test_with_constituent_spectra(self, dummy_spectra):
        """成分スペクトル指定で動作すること"""
        from src.preprocessing.constituent_emsc import constituent_emsc
        X, wn = dummy_spectra
        # ダミー成分スペクトル
        constituents = np.random.randn(2, X.shape[1])
        X_corr = constituent_emsc(X, wavenumbers=wn, constituent_spectra=constituents)
        assert X_corr.shape == X.shape

    def test_different_poly_order(self, dummy_spectra):
        """異なるpoly_orderで動作すること"""
        from src.preprocessing.constituent_emsc import constituent_emsc
        X, wn = dummy_spectra
        for order in [0, 1, 2, 3]:
            X_corr = constituent_emsc(X, wavenumbers=wn, poly_order=order)
            assert X_corr.shape == X.shape
