"""変動係数マップのテスト

対応Issue: #16 各波数の変動係数（CV）マップ
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/16
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.eda.data_loader import load_train, get_spectral_columns, get_wavenumbers
from src.eda.issue16_cv_map import (
    compute_cv_map,
    find_high_cv_bands,
    compare_cv_with_correlation,
)

DATA_DIR = Path("Input_data")


@pytest.fixture(scope="module")
def train_df():
    return load_train(DATA_DIR)


@pytest.fixture(scope="module")
def spectral_cols(train_df):
    return get_spectral_columns(train_df)


class TestCvMap:
    def test_returns_series(self, train_df, spectral_cols):
        result = compute_cv_map(train_df, spectral_cols)
        assert isinstance(result, pd.Series)

    def test_length(self, train_df, spectral_cols):
        result = compute_cv_map(train_df, spectral_cols)
        assert len(result) == 1555

    def test_non_negative(self, train_df, spectral_cols):
        result = compute_cv_map(train_df, spectral_cols)
        assert (result >= 0).all()

    def test_index_is_float(self, train_df, spectral_cols):
        result = compute_cv_map(train_df, spectral_cols)
        assert result.index.dtype == float


class TestHighCvBands:
    def test_returns_dataframe(self, train_df, spectral_cols):
        result = find_high_cv_bands(train_df, spectral_cols)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, train_df, spectral_cols):
        result = find_high_cv_bands(train_df, spectral_cols)
        for col in ["wavenumber", "cv", "rank"]:
            assert col in result.columns

    def test_sorted_descending(self, train_df, spectral_cols):
        result = find_high_cv_bands(train_df, spectral_cols, top_n=10)
        assert result["cv"].is_monotonic_decreasing


class TestCvWithCorrelation:
    def test_returns_dataframe(self, train_df, spectral_cols):
        result = compare_cv_with_correlation(train_df, spectral_cols)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, train_df, spectral_cols):
        result = compare_cv_with_correlation(train_df, spectral_cols)
        for col in ["wavenumber", "cv", "abs_correlation", "cv_rank", "corr_rank"]:
            assert col in result.columns

    def test_length(self, train_df, spectral_cols):
        result = compare_cv_with_correlation(train_df, spectral_cols)
        assert len(result) == 1555
