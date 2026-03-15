"""スペクトル分析のテスト

対応Issue: #2 スペクトルの可視化（樹種別・含水率別）
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/2
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.eda.data_loader import load_train, get_spectral_columns, get_wavenumbers
from src.eda.spectrum_analysis import (
    compute_mean_spectrum_by_species,
    compute_mean_spectrum_by_moisture_range,
    compute_spectral_correlation_with_moisture,
    find_water_absorption_bands,
)


DATA_DIR = Path("Input_data")


@pytest.fixture(scope="module")
def train_df():
    return load_train(DATA_DIR)


@pytest.fixture(scope="module")
def spectral_cols(train_df):
    return get_spectral_columns(train_df)


class TestMeanSpectrumBySpecies:
    def test_returns_dataframe(self, train_df, spectral_cols):
        result = compute_mean_spectrum_by_species(train_df, spectral_cols)
        assert isinstance(result, pd.DataFrame)

    def test_index_is_species(self, train_df, spectral_cols):
        result = compute_mean_spectrum_by_species(train_df, spectral_cols)
        assert result.index.name == "樹種"

    def test_species_count(self, train_df, spectral_cols):
        result = compute_mean_spectrum_by_species(train_df, spectral_cols)
        assert len(result) == 13

    def test_column_count(self, train_df, spectral_cols):
        result = compute_mean_spectrum_by_species(train_df, spectral_cols)
        assert len(result.columns) == 1555


class TestMeanSpectrumByMoistureRange:
    def test_returns_dataframe(self, train_df, spectral_cols):
        result = compute_mean_spectrum_by_moisture_range(train_df, spectral_cols)
        assert isinstance(result, pd.DataFrame)

    def test_has_three_ranges(self, train_df, spectral_cols):
        result = compute_mean_spectrum_by_moisture_range(train_df, spectral_cols)
        assert len(result) == 3

    def test_index_labels(self, train_df, spectral_cols):
        result = compute_mean_spectrum_by_moisture_range(train_df, spectral_cols)
        assert "low" in result.index
        assert "mid" in result.index
        assert "high" in result.index


class TestSpectralCorrelationWithMoisture:
    def test_returns_series(self, train_df, spectral_cols):
        result = compute_spectral_correlation_with_moisture(train_df, spectral_cols)
        assert isinstance(result, pd.Series)

    def test_length(self, train_df, spectral_cols):
        result = compute_spectral_correlation_with_moisture(train_df, spectral_cols)
        assert len(result) == 1555

    def test_values_in_range(self, train_df, spectral_cols):
        result = compute_spectral_correlation_with_moisture(train_df, spectral_cols)
        assert result.min() >= -1.0
        assert result.max() <= 1.0


class TestFindWaterAbsorptionBands:
    def test_returns_dict(self, train_df, spectral_cols):
        wn = get_wavenumbers(spectral_cols)
        corr = compute_spectral_correlation_with_moisture(train_df, spectral_cols)
        result = find_water_absorption_bands(wn, corr)
        assert isinstance(result, dict)

    def test_has_expected_keys(self, train_df, spectral_cols):
        wn = get_wavenumbers(spectral_cols)
        corr = compute_spectral_correlation_with_moisture(train_df, spectral_cols)
        result = find_water_absorption_bands(wn, corr)
        assert "top_positive" in result
        assert "top_negative" in result

    def test_top_bands_are_wavenumbers(self, train_df, spectral_cols):
        wn = get_wavenumbers(spectral_cols)
        corr = compute_spectral_correlation_with_moisture(train_df, spectral_cols)
        result = find_water_absorption_bands(wn, corr)
        assert len(result["top_positive"]) > 0
        assert len(result["top_negative"]) > 0
