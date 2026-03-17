"""Issue #37: EDA知見に基づく樹種不変特徴量のテスト"""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def dummy_spectra():
    """ダミースペクトルデータ（波数軸付き）"""
    np.random.seed(42)
    n_samples = 50
    # 波数: 4300〜10000 cm⁻¹ に相当する1555点
    wavenumbers = np.linspace(4300, 10000, 1555)
    X = np.random.randn(n_samples, 1555) * 0.1 + 1.0  # 吸光度1.0前後
    return X, wavenumbers


class TestBandRatioFeatures:
    """バンド比特徴量のテスト"""

    def test_band_ratios_shape(self, dummy_spectra):
        """バンド比特徴量の形状が正しいこと"""
        from src.feature_engineering.issue37_species_invariant import compute_band_ratios
        X, wn = dummy_spectra
        ratios = compute_band_ratios(X, wn)
        assert isinstance(ratios, pd.DataFrame)
        assert len(ratios) == len(X)
        assert ratios.shape[1] >= 5  # 少なくとも5種のバンド比

    def test_band_ratios_no_nan(self, dummy_spectra):
        """バンド比にNaNがないこと"""
        from src.feature_engineering.issue37_species_invariant import compute_band_ratios
        X, wn = dummy_spectra
        ratios = compute_band_ratios(X, wn)
        assert not ratios.isnull().any().any()

    def test_band_ratios_scale_invariant(self, dummy_spectra):
        """バンド比がスケーリングに対してロバストであること"""
        from src.feature_engineering.issue37_species_invariant import compute_band_ratios
        X, wn = dummy_spectra
        ratios_orig = compute_band_ratios(X, wn)
        ratios_scaled = compute_band_ratios(X * 2.0, wn)
        # スケーリングしても比はほぼ同じ（分子・分母が同じスケール）
        np.testing.assert_allclose(ratios_orig.values, ratios_scaled.values, rtol=0.01)


class TestNDMIFeatures:
    """正規化差分水分指標のテスト"""

    def test_ndmi_range(self, dummy_spectra):
        """NDMIが[-1, 1]の範囲内であること"""
        from src.feature_engineering.issue37_species_invariant import compute_ndmi
        X, wn = dummy_spectra
        ndmi = compute_ndmi(X, wn)
        assert (ndmi.values >= -1.0 - 1e-6).all()
        assert (ndmi.values <= 1.0 + 1e-6).all()

    def test_ndmi_shape(self, dummy_spectra):
        """NDMIの形状が正しいこと"""
        from src.feature_engineering.issue37_species_invariant import compute_ndmi
        X, wn = dummy_spectra
        ndmi = compute_ndmi(X, wn)
        assert len(ndmi) == len(X)
        assert ndmi.shape[1] >= 3


class TestAreaRatioFeatures:
    """領域面積比のテスト"""

    def test_area_ratios_positive(self, dummy_spectra):
        """面積比が正であること（吸光度が正の場合）"""
        from src.feature_engineering.issue37_species_invariant import compute_area_ratios
        X, wn = dummy_spectra
        X_pos = np.abs(X)  # 正にする
        areas = compute_area_ratios(X_pos, wn)
        assert (areas.values > 0).all()

    def test_area_ratios_shape(self, dummy_spectra):
        """面積比の形状が正しいこと"""
        from src.feature_engineering.issue37_species_invariant import compute_area_ratios
        X, wn = dummy_spectra
        areas = compute_area_ratios(X, wn)
        assert len(areas) == len(X)


class TestContinuumRemoval:
    """コンティニュアム除去のテスト"""

    def test_continuum_removal_range(self, dummy_spectra):
        """CR値が0〜1の範囲であること"""
        from src.feature_engineering.issue37_species_invariant import continuum_removal
        X, wn = dummy_spectra
        X_pos = np.abs(X) + 0.1  # 正にする
        cr = continuum_removal(X_pos, wn)
        assert (cr >= -0.01).all()  # ほぼ0以上
        assert (cr <= 1.01).all()   # ほぼ1以下

    def test_continuum_removal_shape(self, dummy_spectra):
        """CR後の形状が同じであること"""
        from src.feature_engineering.issue37_species_invariant import continuum_removal
        X, wn = dummy_spectra
        X_pos = np.abs(X) + 0.1
        cr = continuum_removal(X_pos, wn)
        assert cr.shape == X.shape


class TestAllFeatures:
    """全特徴量統合テスト"""

    def test_create_all_features_shape(self, dummy_spectra):
        """全特徴量の形状が正しいこと"""
        from src.feature_engineering.issue37_species_invariant import create_species_invariant_features
        X, wn = dummy_spectra
        features = create_species_invariant_features(X, wn)
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(X)
        assert features.shape[1] >= 10  # バンド比+NDMI+面積比+バンド深さ

    def test_create_all_features_no_nan(self, dummy_spectra):
        """全特徴量にNaNがないこと"""
        from src.feature_engineering.issue37_species_invariant import create_species_invariant_features
        X, wn = dummy_spectra
        features = create_species_invariant_features(X, wn)
        assert not features.isnull().any().any()
