"""ベースラインドリフト・多重共線性のテスト（Issue #11）"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.eda.data_loader import load_train, get_spectral_columns
from src.eda.baseline_analysis import (
    compute_baseline_drift,
    compute_effective_rank,
    compute_adjacent_correlation,
)

DATA_DIR = Path("Input_data")


@pytest.fixture(scope="module")
def train_df():
    return load_train(DATA_DIR)


@pytest.fixture(scope="module")
def spectral_cols(train_df):
    return get_spectral_columns(train_df)


class TestBaselineDrift:
    def test_returns_dict(self, train_df, spectral_cols):
        result = compute_baseline_drift(train_df, spectral_cols)
        assert isinstance(result, dict)

    def test_has_required_keys(self, train_df, spectral_cols):
        result = compute_baseline_drift(train_df, spectral_cols)
        for key in ["mean_drift", "std_drift", "max_drift", "drift_by_species"]:
            assert key in result

    def test_drift_non_negative(self, train_df, spectral_cols):
        result = compute_baseline_drift(train_df, spectral_cols)
        assert result["mean_drift"] >= 0
        assert result["max_drift"] >= 0

    def test_drift_by_species_is_series(self, train_df, spectral_cols):
        result = compute_baseline_drift(train_df, spectral_cols)
        assert isinstance(result["drift_by_species"], pd.Series)

    def test_drift_by_species_count(self, train_df, spectral_cols):
        result = compute_baseline_drift(train_df, spectral_cols)
        assert len(result["drift_by_species"]) == 13


class TestEffectiveRank:
    def test_returns_dict(self, train_df, spectral_cols):
        result = compute_effective_rank(train_df, spectral_cols)
        assert isinstance(result, dict)

    def test_has_required_keys(self, train_df, spectral_cols):
        result = compute_effective_rank(train_df, spectral_cols)
        for key in ["rank_95", "rank_99", "total_features", "compression_ratio_95"]:
            assert key in result

    def test_rank_much_less_than_features(self, train_df, spectral_cols):
        result = compute_effective_rank(train_df, spectral_cols)
        assert result["rank_95"] < result["total_features"] * 0.1

    def test_rank_99_ge_rank_95(self, train_df, spectral_cols):
        result = compute_effective_rank(train_df, spectral_cols)
        assert result["rank_99"] >= result["rank_95"]


class TestAdjacentCorrelation:
    def test_returns_dict(self, train_df, spectral_cols):
        result = compute_adjacent_correlation(train_df, spectral_cols)
        assert isinstance(result, dict)

    def test_has_required_keys(self, train_df, spectral_cols):
        result = compute_adjacent_correlation(train_df, spectral_cols)
        for key in ["mean_adjacent_corr", "min_adjacent_corr", "high_corr_ratio"]:
            assert key in result

    def test_high_correlation(self, train_df, spectral_cols):
        """NIRスペクトルは隣接波数の相関が非常に高いはず"""
        result = compute_adjacent_correlation(train_df, spectral_cols)
        assert result["mean_adjacent_corr"] > 0.9

    def test_high_corr_ratio_is_proportion(self, train_df, spectral_cols):
        result = compute_adjacent_correlation(train_df, spectral_cols)
        assert 0 <= result["high_corr_ratio"] <= 1.0
