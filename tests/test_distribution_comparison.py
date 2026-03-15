"""train/test分布比較のテスト

対応Issue: #4 train/testのスペクトル分布比較（ドメインシフト確認）
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/4
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.eda.data_loader import load_train, load_test, get_spectral_columns
from src.eda.distribution_comparison import (
    compute_pca_projection,
    compute_spectral_statistics,
    compute_domain_shift_score,
    check_species_overlap,
)


DATA_DIR = Path("Input_data")


@pytest.fixture(scope="module")
def train_df():
    return load_train(DATA_DIR)


@pytest.fixture(scope="module")
def test_df():
    return load_test(DATA_DIR)


@pytest.fixture(scope="module")
def spectral_cols(train_df):
    return get_spectral_columns(train_df)


class TestSpeciesOverlap:
    def test_returns_dict(self, train_df, test_df):
        result = check_species_overlap(train_df, test_df)
        assert isinstance(result, dict)

    def test_no_overlap(self, train_df, test_df):
        result = check_species_overlap(train_df, test_df)
        assert result["overlap_count"] == 0
        assert len(result["overlap_species"]) == 0

    def test_species_counts(self, train_df, test_df):
        result = check_species_overlap(train_df, test_df)
        assert result["train_species_count"] == 13
        assert result["test_species_count"] == 6


class TestSpectralStatistics:
    def test_returns_dataframe(self, train_df, test_df, spectral_cols):
        result = compute_spectral_statistics(train_df, test_df, spectral_cols)
        assert isinstance(result, pd.DataFrame)

    def test_has_train_test_rows(self, train_df, test_df, spectral_cols):
        result = compute_spectral_statistics(train_df, test_df, spectral_cols)
        assert "train" in result.index
        assert "test" in result.index

    def test_has_required_columns(self, train_df, test_df, spectral_cols):
        result = compute_spectral_statistics(train_df, test_df, spectral_cols)
        for col in ["mean_intensity", "std_intensity", "min_intensity", "max_intensity"]:
            assert col in result.columns


class TestPCAProjection:
    def test_returns_dict(self, train_df, test_df, spectral_cols):
        result = compute_pca_projection(train_df, test_df, spectral_cols, n_components=10)
        assert isinstance(result, dict)

    def test_has_required_keys(self, train_df, test_df, spectral_cols):
        result = compute_pca_projection(train_df, test_df, spectral_cols, n_components=10)
        assert "train_scores" in result
        assert "test_scores" in result
        assert "explained_variance_ratio" in result

    def test_shape(self, train_df, test_df, spectral_cols):
        n = 10
        result = compute_pca_projection(train_df, test_df, spectral_cols, n_components=n)
        assert result["train_scores"].shape == (1322, n)
        assert result["test_scores"].shape == (550, n)

    def test_explained_variance_sum(self, train_df, test_df, spectral_cols):
        result = compute_pca_projection(train_df, test_df, spectral_cols, n_components=10)
        assert 0 < result["explained_variance_ratio"].sum() <= 1.0


class TestDomainShiftScore:
    def test_returns_dict(self, train_df, test_df, spectral_cols):
        result = compute_domain_shift_score(train_df, test_df, spectral_cols)
        assert isinstance(result, dict)

    def test_has_required_keys(self, train_df, test_df, spectral_cols):
        result = compute_domain_shift_score(train_df, test_df, spectral_cols)
        assert "mean_diff" in result
        assert "std_diff" in result
        assert "shift_level" in result

    def test_shift_level_is_valid(self, train_df, test_df, spectral_cols):
        result = compute_domain_shift_score(train_df, test_df, spectral_cols)
        assert result["shift_level"] in ["low", "medium", "high"]
