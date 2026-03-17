"""スペクトル自己相関特徴量のテスト"""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def dummy_spectra():
    """ダミースペクトルデータ"""
    np.random.seed(42)
    n_samples = 30
    n_features = 100
    X = np.random.randn(n_samples, n_features) * 0.1 + 1.0
    return X


class TestSpectralAutocorrelation:

    def test_output_shape(self, dummy_spectra):
        """出力形状が正しいこと"""
        from src.feature_engineering.spectral_autocorrelation import compute_spectral_autocorrelation
        X = dummy_spectra
        features = compute_spectral_autocorrelation(X, lags=[1, 5, 10, 20])
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(X)
        assert features.shape[1] == 4  # 4 lags

    def test_no_nan(self, dummy_spectra):
        """NaNがないこと"""
        from src.feature_engineering.spectral_autocorrelation import compute_spectral_autocorrelation
        X = dummy_spectra
        features = compute_spectral_autocorrelation(X, lags=[1, 5, 10])
        assert not features.isnull().any().any()

    def test_acf_range(self, dummy_spectra):
        """ACF値が[-1, 1]の範囲内であること"""
        from src.feature_engineering.spectral_autocorrelation import compute_spectral_autocorrelation
        X = dummy_spectra
        features = compute_spectral_autocorrelation(X, lags=[1, 5, 10])
        assert (features.values >= -1.0 - 1e-6).all()
        assert (features.values <= 1.0 + 1e-6).all()

    def test_lag1_high_for_smooth_signal(self):
        """滑らかな信号のlag-1 ACFが高いこと"""
        from src.feature_engineering.spectral_autocorrelation import compute_spectral_autocorrelation
        # 滑らかなサイン波
        x = np.sin(np.linspace(0, 4 * np.pi, 200))
        X = x.reshape(1, -1)
        features = compute_spectral_autocorrelation(X, lags=[1])
        assert features.values[0, 0] > 0.9

    def test_different_lags(self, dummy_spectra):
        """異なるlag設定で動作すること"""
        from src.feature_engineering.spectral_autocorrelation import compute_spectral_autocorrelation
        X = dummy_spectra
        f1 = compute_spectral_autocorrelation(X, lags=[1, 2, 3])
        f2 = compute_spectral_autocorrelation(X, lags=[5, 10, 15, 20, 25])
        assert f1.shape[1] == 3
        assert f2.shape[1] == 5

    def test_column_names(self, dummy_spectra):
        """カラム名にlag値が含まれること"""
        from src.feature_engineering.spectral_autocorrelation import compute_spectral_autocorrelation
        X = dummy_spectra
        features = compute_spectral_autocorrelation(X, lags=[1, 5, 10])
        for col in features.columns:
            assert "acf" in col.lower() or "lag" in col.lower()
