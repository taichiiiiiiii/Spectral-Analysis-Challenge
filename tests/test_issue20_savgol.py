"""Savitzky-Golay微分のテスト

対応Issue: #20 Savitzky-Golay微分（1次・2次）の実装・効果検証
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/20
"""
import pytest
import numpy as np
from pathlib import Path

from src.eda.data_loader import load_train, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue20_savgol import apply_savgol, evaluate_savgol_combinations

DATA_DIR = Path("Input_data")


@pytest.fixture(scope="module")
def train_df():
    return load_train(DATA_DIR)


@pytest.fixture(scope="module")
def spectral_cols(train_df):
    return get_spectral_columns(train_df)


@pytest.fixture(scope="module")
def X_snv(train_df, spectral_cols):
    return apply_snv(train_df[spectral_cols].values)


class TestApplySavgol:
    def test_returns_ndarray(self, X_snv):
        result = apply_savgol(X_snv, deriv=1)
        assert isinstance(result, np.ndarray)

    def test_shape_preserved(self, X_snv):
        result = apply_savgol(X_snv, deriv=1)
        assert result.shape == X_snv.shape

    def test_first_deriv_different_from_input(self, X_snv):
        result = apply_savgol(X_snv, deriv=1)
        assert not np.allclose(result, X_snv)

    def test_second_deriv_different_from_first(self, X_snv):
        d1 = apply_savgol(X_snv, deriv=1)
        d2 = apply_savgol(X_snv, deriv=2)
        assert not np.allclose(d1, d2)

    def test_custom_window(self, X_snv):
        result = apply_savgol(X_snv, deriv=1, window_length=15, polyorder=3)
        assert result.shape == X_snv.shape

    def test_no_nan(self, X_snv):
        result = apply_savgol(X_snv, deriv=1)
        assert not np.isnan(result).any()

    def test_no_inf(self, X_snv):
        result = apply_savgol(X_snv, deriv=1)
        assert not np.isinf(result).any()


class TestEvaluateSavgolCombinations:
    def test_returns_dataframe(self, train_df, spectral_cols):
        result = evaluate_savgol_combinations(train_df, spectral_cols)
        import pandas as pd
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, train_df, spectral_cols):
        result = evaluate_savgol_combinations(train_df, spectral_cols)
        for col in ["preprocessing", "rmse", "n_components"]:
            assert col in result.columns

    def test_all_combinations_present(self, train_df, spectral_cols):
        result = evaluate_savgol_combinations(train_df, spectral_cols)
        expected = {"Raw", "SNV", "MSC", "SNV+1d", "SNV+2d", "MSC+1d", "MSC+2d"}
        assert set(result["preprocessing"]) == expected

    def test_rmse_positive(self, train_df, spectral_cols):
        result = evaluate_savgol_combinations(train_df, spectral_cols)
        assert (result["rmse"] > 0).all()

    def test_sorted_by_rmse(self, train_df, spectral_cols):
        result = evaluate_savgol_combinations(train_df, spectral_cols)
        assert result["rmse"].is_monotonic_increasing
