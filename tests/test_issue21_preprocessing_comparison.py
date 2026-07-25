"""前処理組み合わせ比較のテスト

対応Issue: #21 前処理組み合わせ比較（PLS LOSO-CV RMSE）
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/21
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.eda.data_loader import load_train, get_spectral_columns
from src.preprocessing.issue21_preprocessing_comparison import (
    build_preprocessing_pipeline,
    evaluate_preprocessing_combination,
    compare_all_combinations,
)

DATA_DIR = Path("Input_data")


@pytest.fixture(scope="module")
def train_df():
    return load_train(DATA_DIR)


@pytest.fixture(scope="module")
def spectral_cols(train_df):
    return get_spectral_columns(train_df)


class TestBuildPreprocessingPipeline:
    def test_raw_returns_unchanged(self, train_df, spectral_cols):
        X = train_df[spectral_cols].values
        result = build_preprocessing_pipeline(X, method="Raw")
        assert result.shape == X.shape

    def test_snv_pipeline(self, train_df, spectral_cols):
        X = train_df[spectral_cols].values
        result = build_preprocessing_pipeline(X, method="SNV")
        assert result.shape == X.shape

    def test_msc_pipeline(self, train_df, spectral_cols):
        X = train_df[spectral_cols].values
        result = build_preprocessing_pipeline(X, method="MSC")
        assert result.shape == X.shape

    def test_snv_1d_pipeline(self, train_df, spectral_cols):
        X = train_df[spectral_cols].values
        result = build_preprocessing_pipeline(X, method="SNV+1d")
        assert result.shape == X.shape

    def test_snv_2d_pipeline(self, train_df, spectral_cols):
        X = train_df[spectral_cols].values
        result = build_preprocessing_pipeline(X, method="SNV+2d")
        assert result.shape == X.shape

    def test_msc_1d_pipeline(self, train_df, spectral_cols):
        X = train_df[spectral_cols].values
        result = build_preprocessing_pipeline(X, method="MSC+1d")
        assert result.shape == X.shape

    def test_msc_2d_pipeline(self, train_df, spectral_cols):
        X = train_df[spectral_cols].values
        result = build_preprocessing_pipeline(X, method="MSC+2d")
        assert result.shape == X.shape

    def test_invalid_method_raises(self, train_df, spectral_cols):
        X = train_df[spectral_cols].values
        with pytest.raises(ValueError):
            build_preprocessing_pipeline(X, method="INVALID")


class TestEvaluatePreprocessingCombination:
    def test_returns_dict(self, train_df, spectral_cols):
        result = evaluate_preprocessing_combination(train_df, spectral_cols, method="Raw")
        assert isinstance(result, dict)

    def test_has_required_keys(self, train_df, spectral_cols):
        result = evaluate_preprocessing_combination(train_df, spectral_cols, method="Raw")
        for key in ["method", "rmse", "rmse_std", "n_components"]:
            assert key in result

    def test_rmse_near_baseline(self, train_df, spectral_cols):
        """Rawのベースラインは15〜35程度"""
        result = evaluate_preprocessing_combination(train_df, spectral_cols, method="Raw")
        assert 10 < result["rmse"] < 40

    def test_method_name_preserved(self, train_df, spectral_cols):
        result = evaluate_preprocessing_combination(train_df, spectral_cols, method="SNV")
        assert result["method"] == "SNV"


class TestCompareAllCombinations:
    def test_returns_dataframe(self, train_df, spectral_cols):
        result = compare_all_combinations(train_df, spectral_cols)
        assert isinstance(result, pd.DataFrame)

    def test_has_7_rows(self, train_df, spectral_cols):
        result = compare_all_combinations(train_df, spectral_cols)
        assert len(result) == 7

    def test_has_required_columns(self, train_df, spectral_cols):
        result = compare_all_combinations(train_df, spectral_cols)
        for col in ["method", "rmse", "rmse_std", "n_components", "rank"]:
            assert col in result.columns

    def test_sorted_by_rmse(self, train_df, spectral_cols):
        result = compare_all_combinations(train_df, spectral_cols)
        assert result["rmse"].is_monotonic_increasing

    def test_all_methods_present(self, train_df, spectral_cols):
        result = compare_all_combinations(train_df, spectral_cols)
        expected = {"Raw", "SNV", "MSC", "SNV+1d", "SNV+2d", "MSC+1d", "MSC+2d"}
        assert set(result["method"]) == expected
