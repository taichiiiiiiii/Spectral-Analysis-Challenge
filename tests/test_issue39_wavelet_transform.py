"""ウェーブレット変換前処理のテスト"""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def dummy_spectra():
    np.random.seed(42)
    n_samples = 20
    n_features = 128  # 2のべき乗でDWT扱いやすい
    X = np.random.randn(n_samples, n_features) * 0.1 + 1.0
    return X


class TestWaveletDenoise:

    def test_output_shape(self, dummy_spectra):
        """デノイズ後の形状が同じこと"""
        from src.preprocessing.issue39_wavelet_transform import wavelet_denoise
        X = dummy_spectra
        X_den = wavelet_denoise(X, wavelet="db4", level=3)
        assert X_den.shape == X.shape

    def test_no_nan(self, dummy_spectra):
        """NaNがないこと"""
        from src.preprocessing.issue39_wavelet_transform import wavelet_denoise
        X = dummy_spectra
        X_den = wavelet_denoise(X, wavelet="db4", level=3)
        assert not np.any(np.isnan(X_den))

    def test_reduces_noise(self):
        """ノイズ付き信号がデノイズされること"""
        from src.preprocessing.issue39_wavelet_transform import wavelet_denoise
        np.random.seed(42)
        t = np.linspace(0, 1, 256)
        clean = np.sin(2 * np.pi * 5 * t)
        noisy = clean + np.random.randn(256) * 0.5
        X = noisy.reshape(1, -1)
        X_den = wavelet_denoise(X, wavelet="db4", level=4)
        # デノイズ後は元信号に近くなる
        mse_before = np.mean((noisy - clean) ** 2)
        mse_after = np.mean((X_den[0] - clean) ** 2)
        assert mse_after < mse_before

    def test_different_wavelets(self, dummy_spectra):
        """異なるウェーブレットで動作すること"""
        from src.preprocessing.issue39_wavelet_transform import wavelet_denoise
        X = dummy_spectra
        for wv in ["db4", "sym6", "coif3"]:
            X_den = wavelet_denoise(X, wavelet=wv, level=2)
            assert X_den.shape == X.shape


class TestWaveletFeatures:

    def test_output_shape(self, dummy_spectra):
        """特徴量の形状が正しいこと"""
        from src.preprocessing.issue39_wavelet_transform import extract_wavelet_features
        X = dummy_spectra
        features = extract_wavelet_features(X, wavelet="db4", level=3)
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(X)
        assert features.shape[1] >= 4  # level+1のエネルギー特徴量

    def test_no_nan(self, dummy_spectra):
        """NaNがないこと"""
        from src.preprocessing.issue39_wavelet_transform import extract_wavelet_features
        X = dummy_spectra
        features = extract_wavelet_features(X, wavelet="db4", level=3)
        assert not features.isnull().any().any()

    def test_energy_positive(self, dummy_spectra):
        """エネルギー特徴量が非負であること"""
        from src.preprocessing.issue39_wavelet_transform import extract_wavelet_features
        X = dummy_spectra
        features = extract_wavelet_features(X, wavelet="db4", level=3)
        energy_cols = [c for c in features.columns if "energy" in c]
        assert len(energy_cols) > 0
        assert (features[energy_cols].values >= 0).all()
