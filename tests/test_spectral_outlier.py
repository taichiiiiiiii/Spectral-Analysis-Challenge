"""スペクトル異常値検出のテスト

対応Issue: #8 スペクトル異常値の検出（Hotelling's T²・Q残差）
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/8
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.eda.data_loader import load_train, get_spectral_columns
from src.eda.spectral_outlier import (
    compute_pca_for_outlier,
    compute_hotelling_t2,
    compute_q_residuals,
    detect_spectral_outliers,
)

DATA_DIR = Path("Input_data")


@pytest.fixture(scope="module")
def train_df():
    return load_train(DATA_DIR)


@pytest.fixture(scope="module")
def spectral_cols(train_df):
    return get_spectral_columns(train_df)


@pytest.fixture(scope="module")
def pca_result(train_df, spectral_cols):
    return compute_pca_for_outlier(train_df, spectral_cols, variance_threshold=0.95)


class TestPcaForOutlier:
    def test_returns_dict(self, train_df, spectral_cols):
        result = compute_pca_for_outlier(train_df, spectral_cols)
        assert isinstance(result, dict)

    def test_has_required_keys(self, train_df, spectral_cols):
        result = compute_pca_for_outlier(train_df, spectral_cols)
        for key in ["scores", "loadings", "explained_variance_ratio", "n_components", "X_reconstructed"]:
            assert key in result

    def test_variance_threshold(self, train_df, spectral_cols):
        result = compute_pca_for_outlier(train_df, spectral_cols, variance_threshold=0.95)
        assert result["explained_variance_ratio"].sum() >= 0.95

    def test_scores_shape(self, train_df, spectral_cols):
        result = compute_pca_for_outlier(train_df, spectral_cols)
        assert result["scores"].shape[0] == len(train_df)


class TestHotellingT2:
    def test_returns_series(self, train_df, pca_result):
        result = compute_hotelling_t2(pca_result["scores"])
        assert isinstance(result, np.ndarray)

    def test_length(self, train_df, pca_result):
        result = compute_hotelling_t2(pca_result["scores"])
        assert len(result) == len(train_df)

    def test_non_negative(self, train_df, pca_result):
        result = compute_hotelling_t2(pca_result["scores"])
        assert np.all(result >= 0)


class TestQResiduals:
    def test_returns_array(self, train_df, spectral_cols, pca_result):
        X = train_df[spectral_cols].values
        result = compute_q_residuals(X, pca_result["X_reconstructed"])
        assert isinstance(result, np.ndarray)

    def test_length(self, train_df, spectral_cols, pca_result):
        X = train_df[spectral_cols].values
        result = compute_q_residuals(X, pca_result["X_reconstructed"])
        assert len(result) == len(train_df)

    def test_non_negative(self, train_df, spectral_cols, pca_result):
        X = train_df[spectral_cols].values
        result = compute_q_residuals(X, pca_result["X_reconstructed"])
        assert np.all(result >= 0)


class TestDetectSpectralOutliers:
    def test_returns_dict(self, train_df, spectral_cols):
        result = detect_spectral_outliers(train_df, spectral_cols)
        assert isinstance(result, dict)

    def test_has_required_keys(self, train_df, spectral_cols):
        result = detect_spectral_outliers(train_df, spectral_cols)
        for key in ["t2_outliers", "q_outliers", "combined_outliers", "outlier_indices", "outlier_count"]:
            assert key in result

    def test_outlier_count_reasonable(self, train_df, spectral_cols):
        result = detect_spectral_outliers(train_df, spectral_cols)
        assert 0 <= result["outlier_count"] <= len(train_df) * 0.1

    def test_outlier_indices_are_valid(self, train_df, spectral_cols):
        result = detect_spectral_outliers(train_df, spectral_cols)
        assert all(0 <= i < len(train_df) for i in result["outlier_indices"])
