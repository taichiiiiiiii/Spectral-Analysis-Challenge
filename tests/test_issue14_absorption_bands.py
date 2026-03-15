"""近赤外吸収帯照合のテスト

対応Issue: #14 近赤外吸収帯の分光学的照合
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/14
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.eda.data_loader import load_train, get_spectral_columns, get_wavenumbers
from src.eda.issue14_absorption_bands import (
    compute_correlation_map,
    identify_theoretical_bands,
    compare_theory_vs_data,
)

DATA_DIR = Path("Input_data")


@pytest.fixture(scope="module")
def train_df():
    return load_train(DATA_DIR)


@pytest.fixture(scope="module")
def spectral_cols(train_df):
    return get_spectral_columns(train_df)


@pytest.fixture(scope="module")
def wavenumbers(spectral_cols):
    return get_wavenumbers(spectral_cols)


class TestCorrelationMap:
    def test_returns_series(self, train_df, spectral_cols):
        result = compute_correlation_map(train_df, spectral_cols)
        assert isinstance(result, pd.Series)

    def test_length(self, train_df, spectral_cols):
        result = compute_correlation_map(train_df, spectral_cols)
        assert len(result) == 1555

    def test_index_is_float(self, train_df, spectral_cols):
        result = compute_correlation_map(train_df, spectral_cols)
        assert result.index.dtype == float


class TestTheoreticalBands:
    def test_returns_dict(self):
        result = identify_theoretical_bands()
        assert isinstance(result, dict)

    def test_has_water_bands(self):
        result = identify_theoretical_bands()
        assert "water_combination" in result
        assert "water_first_overtone" in result

    def test_wavenumber_values_in_range(self):
        result = identify_theoretical_bands()
        for name, wn in result.items():
            assert 4000 <= wn <= 10000


class TestTheoryVsData:
    def test_returns_dataframe(self, train_df, spectral_cols, wavenumbers):
        corr = compute_correlation_map(train_df, spectral_cols)
        result = compare_theory_vs_data(corr, wavenumbers)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, train_df, spectral_cols, wavenumbers):
        corr = compute_correlation_map(train_df, spectral_cols)
        result = compare_theory_vs_data(corr, wavenumbers)
        for col in ["band_name", "theoretical_wn", "nearest_actual_wn",
                    "correlation_at_theory", "peak_correlation", "peak_wavenumber"]:
            assert col in result.columns

    def test_correlations_in_range(self, train_df, spectral_cols, wavenumbers):
        corr = compute_correlation_map(train_df, spectral_cols)
        result = compare_theory_vs_data(corr, wavenumbers)
        assert result["correlation_at_theory"].abs().max() <= 1.0
        assert result["peak_correlation"].abs().max() <= 1.0
