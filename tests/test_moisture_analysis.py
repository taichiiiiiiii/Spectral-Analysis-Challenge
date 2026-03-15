"""含水率分析のテスト

対応Issue: #3 含水率の分布確認
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/3
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.eda.data_loader import load_train
from src.eda.moisture_analysis import (
    compute_moisture_stats,
    compute_moisture_stats_by_species,
    detect_outliers,
    check_log_transform_benefit,
    check_sample_number_duplicates,
)


DATA_DIR = Path("Input_data")


@pytest.fixture(scope="module")
def train_df():
    return load_train(DATA_DIR)


class TestMoistureStats:
    def test_returns_series(self, train_df):
        result = compute_moisture_stats(train_df)
        assert isinstance(result, pd.Series)

    def test_has_required_stats(self, train_df):
        result = compute_moisture_stats(train_df)
        for key in ["mean", "std", "min", "max", "median", "skewness"]:
            assert key in result.index

    def test_mean_around_50(self, train_df):
        result = compute_moisture_stats(train_df)
        assert 40 < result["mean"] < 60

    def test_positive_skewness(self, train_df):
        """含水率は右に裾を引いた分布のはず"""
        result = compute_moisture_stats(train_df)
        assert result["skewness"] > 0


class TestMoistureStatsBySpecies:
    def test_returns_dataframe(self, train_df):
        result = compute_moisture_stats_by_species(train_df)
        assert isinstance(result, pd.DataFrame)

    def test_species_count(self, train_df):
        result = compute_moisture_stats_by_species(train_df)
        assert len(result) == 13

    def test_has_required_columns(self, train_df):
        result = compute_moisture_stats_by_species(train_df)
        for col in ["mean", "std", "min", "max", "count"]:
            assert col in result.columns


class TestDetectOutliers:
    def test_returns_series(self, train_df):
        result = detect_outliers(train_df)
        assert isinstance(result, pd.Series)
        assert result.dtype == bool

    def test_length(self, train_df):
        result = detect_outliers(train_df)
        assert len(result) == len(train_df)

    def test_outlier_ratio_is_small(self, train_df):
        result = detect_outliers(train_df)
        assert result.mean() < 0.1


class TestLogTransformBenefit:
    def test_returns_dict(self, train_df):
        result = check_log_transform_benefit(train_df)
        assert isinstance(result, dict)

    def test_has_required_keys(self, train_df):
        result = check_log_transform_benefit(train_df)
        assert "original_skewness" in result
        assert "log_skewness" in result
        assert "recommend_log" in result

    def test_recommend_is_bool(self, train_df):
        result = check_log_transform_benefit(train_df)
        assert isinstance(result["recommend_log"], bool)


class TestSampleNumberDuplicates:
    def test_returns_dict(self, train_df):
        result = check_sample_number_duplicates(train_df)
        assert isinstance(result, dict)

    def test_has_required_keys(self, train_df):
        result = check_sample_number_duplicates(train_df)
        assert "has_duplicates" in result
        assert "unique_samples" in result
        assert "total_rows" in result
        assert "recommended_cv" in result

    def test_recommended_cv_is_valid(self, train_df):
        result = check_sample_number_duplicates(train_df)
        assert result["recommended_cv"] in ["LOSO-CV", "GroupKFold + LOSO-CV"]
