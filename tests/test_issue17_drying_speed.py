"""乾燥速度・サンプル数偏り分析のテスト

対応Issue: #17 樹種ごとの乾燥速度とサンプル数の偏り確認
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/17
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.eda.data_loader import load_train, get_spectral_columns
from src.eda.issue17_drying_speed import (
    compute_drying_stats,
    compute_moisture_interval,
    assess_sample_imbalance,
)

DATA_DIR = Path("Input_data")


@pytest.fixture(scope="module")
def train_df():
    return load_train(DATA_DIR)


@pytest.fixture(scope="module")
def spectral_cols(train_df):
    return get_spectral_columns(train_df)


class TestDryingStats:
    def test_returns_dataframe(self, train_df):
        result = compute_drying_stats(train_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_13_species(self, train_df):
        result = compute_drying_stats(train_df)
        assert len(result) == 13

    def test_has_required_columns(self, train_df):
        result = compute_drying_stats(train_df)
        for col in ["species", "sample_count", "moisture_range", "avg_interval"]:
            assert col in result.columns

    def test_positive_values(self, train_df):
        result = compute_drying_stats(train_df)
        assert (result["sample_count"] > 0).all()
        assert (result["moisture_range"] > 0).all()


class TestMoistureInterval:
    def test_returns_dataframe(self, train_df):
        result = compute_moisture_interval(train_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, train_df):
        result = compute_moisture_interval(train_df)
        for col in ["species", "mean_interval", "std_interval", "n_intervals"]:
            assert col in result.columns

    def test_positive_intervals(self, train_df):
        result = compute_moisture_interval(train_df)
        assert (result["mean_interval"] >= 0).all()


class TestSampleImbalance:
    def test_returns_dict(self, train_df, spectral_cols):
        result = assess_sample_imbalance(train_df, spectral_cols)
        assert isinstance(result, dict)

    def test_has_required_keys(self, train_df, spectral_cols):
        result = assess_sample_imbalance(train_df, spectral_cols)
        for key in ["imbalance_ratio", "min_species", "max_species",
                    "needs_reweighting", "recommendation"]:
            assert key in result

    def test_imbalance_ratio_positive(self, train_df, spectral_cols):
        result = assess_sample_imbalance(train_df, spectral_cols)
        assert result["imbalance_ratio"] >= 1.0

    def test_recommendation_is_string(self, train_df, spectral_cols):
        result = assess_sample_imbalance(train_df, spectral_cols)
        assert isinstance(result["recommendation"], str)
