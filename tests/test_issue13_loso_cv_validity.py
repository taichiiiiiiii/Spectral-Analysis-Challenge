"""LOSO-CV妥当性確認のテスト

対応Issue: #13 LOSO-CVの妥当性確認（fold別含水率カバレッジ）
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/13
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.eda.data_loader import load_train
from src.eda.issue13_loso_cv_validity import (
    compute_fold_coverage,
    identify_problematic_folds,
    compute_fold_sample_stats,
)

DATA_DIR = Path("Input_data")


@pytest.fixture(scope="module")
def train_df():
    return load_train(DATA_DIR)


class TestFoldCoverage:
    def test_returns_dataframe(self, train_df):
        result = compute_fold_coverage(train_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_13_folds(self, train_df):
        result = compute_fold_coverage(train_df)
        assert len(result) == 13

    def test_has_required_columns(self, train_df):
        result = compute_fold_coverage(train_df)
        for col in ["holdout_species", "holdout_min", "holdout_max",
                    "train_min", "train_max", "is_covered"]:
            assert col in result.columns

    def test_is_covered_is_bool(self, train_df):
        result = compute_fold_coverage(train_df)
        assert result["is_covered"].dtype == bool

    def test_coverage_ratio_valid(self, train_df):
        result = compute_fold_coverage(train_df)
        ratio = result["is_covered"].mean()
        assert 0.0 <= ratio <= 1.0


class TestProblematicFolds:
    def test_returns_dataframe(self, train_df):
        result = identify_problematic_folds(train_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, train_df):
        result = identify_problematic_folds(train_df)
        for col in ["holdout_species", "issue", "recommendation"]:
            assert col in result.columns

    def test_recommendation_is_string(self, train_df):
        result = identify_problematic_folds(train_df)
        if len(result) > 0:
            assert all(isinstance(r, str) for r in result["recommendation"])


class TestFoldSampleStats:
    def test_returns_dataframe(self, train_df):
        result = compute_fold_sample_stats(train_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_13_rows(self, train_df):
        result = compute_fold_sample_stats(train_df)
        assert len(result) == 13

    def test_has_required_columns(self, train_df):
        result = compute_fold_sample_stats(train_df)
        for col in ["holdout_species", "holdout_count", "train_count", "holdout_ratio"]:
            assert col in result.columns

    def test_total_counts_correct(self, train_df):
        result = compute_fold_sample_stats(train_df)
        assert (result["holdout_count"] + result["train_count"] == len(train_df)).all()
