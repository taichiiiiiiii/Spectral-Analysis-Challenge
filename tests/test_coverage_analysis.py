"""含水率カバレッジ分析のテスト

対応Issue: #12 含水率カバレッジの確認（train vs test の外挿リスク）
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/12
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.eda.data_loader import load_train, load_test, get_spectral_columns
from src.eda.coverage_analysis import (
    compute_pc_coverage,
    assess_extrapolation_risk,
    compute_moisture_range_coverage,
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


class TestPcCoverage:
    def test_returns_dict(self, train_df, test_df, spectral_cols):
        result = compute_pc_coverage(train_df, test_df, spectral_cols)
        assert isinstance(result, dict)

    def test_has_required_keys(self, train_df, test_df, spectral_cols):
        result = compute_pc_coverage(train_df, test_df, spectral_cols)
        for key in ["train_scores", "test_scores", "test_outside_ratio", "n_components"]:
            assert key in result

    def test_scores_shape(self, train_df, test_df, spectral_cols):
        result = compute_pc_coverage(train_df, test_df, spectral_cols, n_components=5)
        assert result["train_scores"].shape == (1322, 5)
        assert result["test_scores"].shape == (550, 5)

    def test_outside_ratio_is_proportion(self, train_df, test_df, spectral_cols):
        result = compute_pc_coverage(train_df, test_df, spectral_cols)
        assert 0 <= result["test_outside_ratio"] <= 1.0


class TestExtrapolationRisk:
    def test_returns_dict(self, train_df, test_df, spectral_cols):
        result = assess_extrapolation_risk(train_df, test_df, spectral_cols)
        assert isinstance(result, dict)

    def test_has_required_keys(self, train_df, test_df, spectral_cols):
        result = assess_extrapolation_risk(train_df, test_df, spectral_cols)
        for key in ["risk_level", "outside_ratio", "recommendation"]:
            assert key in result

    def test_risk_level_is_valid(self, train_df, test_df, spectral_cols):
        result = assess_extrapolation_risk(train_df, test_df, spectral_cols)
        assert result["risk_level"] in ["low", "medium", "high"]

    def test_recommendation_is_string(self, train_df, test_df, spectral_cols):
        result = assess_extrapolation_risk(train_df, test_df, spectral_cols)
        assert isinstance(result["recommendation"], str)
        assert len(result["recommendation"]) > 0


class TestMoistureRangeCoverage:
    def test_returns_dict(self, train_df):
        result = compute_moisture_range_coverage(train_df)
        assert isinstance(result, dict)

    def test_has_required_keys(self, train_df):
        result = compute_moisture_range_coverage(train_df)
        for key in ["low_moisture_count", "high_moisture_count",
                    "low_moisture_ratio", "high_moisture_ratio"]:
            assert key in result

    def test_counts_non_negative(self, train_df):
        result = compute_moisture_range_coverage(train_df)
        assert result["low_moisture_count"] >= 0
        assert result["high_moisture_count"] >= 0

    def test_ratios_are_proportions(self, train_df):
        result = compute_moisture_range_coverage(train_df)
        assert 0 <= result["low_moisture_ratio"] <= 1.0
        assert 0 <= result["high_moisture_ratio"] <= 1.0
