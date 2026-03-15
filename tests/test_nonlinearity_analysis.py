"""非線形性分析のテスト（Issue #10）"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.eda.data_loader import load_train, get_spectral_columns, get_wavenumbers
from src.eda.nonlinearity_analysis import (
    compute_correlation_by_moisture_range,
    compare_linear_vs_log_correlation,
    find_most_nonlinear_wavenumbers,
)

DATA_DIR = Path("Input_data")


@pytest.fixture(scope="module")
def train_df():
    return load_train(DATA_DIR)


@pytest.fixture(scope="module")
def spectral_cols(train_df):
    return get_spectral_columns(train_df)


class TestCorrelationByMoistureRange:
    def test_returns_dataframe(self, train_df, spectral_cols):
        result = compute_correlation_by_moisture_range(train_df, spectral_cols)
        assert isinstance(result, pd.DataFrame)

    def test_has_three_ranges(self, train_df, spectral_cols):
        result = compute_correlation_by_moisture_range(train_df, spectral_cols)
        assert len(result) == 3

    def test_index_labels(self, train_df, spectral_cols):
        result = compute_correlation_by_moisture_range(train_df, spectral_cols)
        assert "low" in result.index
        assert "mid" in result.index
        assert "high" in result.index

    def test_values_in_range(self, train_df, spectral_cols):
        result = compute_correlation_by_moisture_range(train_df, spectral_cols)
        assert result.values.min() >= -1.0
        assert result.values.max() <= 1.0


class TestLinearVsLogCorrelation:
    def test_returns_dataframe(self, train_df, spectral_cols):
        result = compare_linear_vs_log_correlation(train_df, spectral_cols)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, train_df, spectral_cols):
        result = compare_linear_vs_log_correlation(train_df, spectral_cols)
        for col in ["linear_corr", "log_corr", "log_improves"]:
            assert col in result.columns

    def test_length(self, train_df, spectral_cols):
        result = compare_linear_vs_log_correlation(train_df, spectral_cols)
        assert len(result) == 1555

    def test_log_improves_is_bool(self, train_df, spectral_cols):
        result = compare_linear_vs_log_correlation(train_df, spectral_cols)
        assert result["log_improves"].dtype == bool

    def test_log_improves_is_proportion(self, train_df, spectral_cols):
        """log_improvesの割合は0〜1の間"""
        result = compare_linear_vs_log_correlation(train_df, spectral_cols)
        ratio = result["log_improves"].mean()
        assert 0.0 <= ratio <= 1.0


class TestFindMostNonlinearWavenumbers:
    def test_returns_dataframe(self, train_df, spectral_cols):
        result = find_most_nonlinear_wavenumbers(train_df, spectral_cols)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, train_df, spectral_cols):
        result = find_most_nonlinear_wavenumbers(train_df, spectral_cols)
        for col in ["wavenumber", "corr_variance_across_ranges", "nonlinearity_score"]:
            assert col in result.columns

    def test_sorted_by_score(self, train_df, spectral_cols):
        result = find_most_nonlinear_wavenumbers(train_df, spectral_cols, top_n=10)
        assert result["nonlinearity_score"].is_monotonic_decreasing
