"""PLSコンポーネント数事前評価のテスト

対応Issue: #15 PLSコンポーネント数の事前評価
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/15
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.eda.data_loader import load_train, get_spectral_columns
from src.eda.issue15_pls_components import (
    run_loso_cv_pls,
    find_optimal_components,
    compare_log_vs_raw_target,
)

DATA_DIR = Path("Input_data")


@pytest.fixture(scope="module")
def train_df():
    return load_train(DATA_DIR)


@pytest.fixture(scope="module")
def spectral_cols(train_df):
    return get_spectral_columns(train_df)


class TestLosoSvPls:
    def test_returns_dataframe(self, train_df, spectral_cols):
        result = run_loso_cv_pls(train_df, spectral_cols, max_components=5)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, train_df, spectral_cols):
        result = run_loso_cv_pls(train_df, spectral_cols, max_components=5)
        for col in ["n_components", "rmse_mean", "rmse_std"]:
            assert col in result.columns

    def test_row_count(self, train_df, spectral_cols):
        max_c = 5
        result = run_loso_cv_pls(train_df, spectral_cols, max_components=max_c)
        assert len(result) == max_c

    def test_rmse_positive(self, train_df, spectral_cols):
        result = run_loso_cv_pls(train_df, spectral_cols, max_components=5)
        assert (result["rmse_mean"] > 0).all()


class TestOptimalComponents:
    def test_returns_dict(self, train_df, spectral_cols):
        cv_result = run_loso_cv_pls(train_df, spectral_cols, max_components=5)
        result = find_optimal_components(cv_result)
        assert isinstance(result, dict)

    def test_has_required_keys(self, train_df, spectral_cols):
        cv_result = run_loso_cv_pls(train_df, spectral_cols, max_components=5)
        result = find_optimal_components(cv_result)
        for key in ["optimal_n", "optimal_rmse", "overfitting_starts_at"]:
            assert key in result

    def test_optimal_n_in_range(self, train_df, spectral_cols):
        max_c = 5
        cv_result = run_loso_cv_pls(train_df, spectral_cols, max_components=max_c)
        result = find_optimal_components(cv_result)
        assert 1 <= result["optimal_n"] <= max_c


class TestLogVsRawTarget:
    def test_returns_dataframe(self, train_df, spectral_cols):
        result = compare_log_vs_raw_target(train_df, spectral_cols, max_components=5)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, train_df, spectral_cols):
        result = compare_log_vs_raw_target(train_df, spectral_cols, max_components=5)
        for col in ["n_components", "rmse_raw", "rmse_log"]:
            assert col in result.columns

    def test_both_rmse_positive(self, train_df, spectral_cols):
        result = compare_log_vs_raw_target(train_df, spectral_cols, max_components=5)
        assert (result["rmse_raw"] > 0).all()
        assert (result["rmse_log"] > 0).all()
