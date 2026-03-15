"""SNV（Standard Normal Variate）のテスト

対応Issue: #18 SNV（Standard Normal Variate）の実装・効果検証
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/18
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.eda.data_loader import load_train, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv, verify_snv_properties, evaluate_snv_effect

DATA_DIR = Path("Input_data")


@pytest.fixture(scope="module")
def train_df():
    return load_train(DATA_DIR)


@pytest.fixture(scope="module")
def spectral_cols(train_df):
    return get_spectral_columns(train_df)


@pytest.fixture(scope="module")
def X_raw(train_df, spectral_cols):
    return train_df[spectral_cols].values


@pytest.fixture(scope="module")
def X_snv(X_raw):
    return apply_snv(X_raw)


class TestApplySnv:
    def test_returns_ndarray(self, X_raw):
        result = apply_snv(X_raw)
        assert isinstance(result, np.ndarray)

    def test_shape_preserved(self, X_raw, X_snv):
        assert X_snv.shape == X_raw.shape

    def test_mean_near_zero(self, X_snv):
        """各サンプルの平均がほぼ0"""
        row_means = X_snv.mean(axis=1)
        assert np.allclose(row_means, 0, atol=1e-10)

    def test_std_near_one(self, X_snv):
        """各サンプルの標準偏差がほぼ1"""
        row_stds = X_snv.std(axis=1)
        assert np.allclose(row_stds, 1, atol=1e-10)

    def test_dataframe_input(self, train_df, spectral_cols):
        """DataFrameを入力しても動作する"""
        X_df = train_df[spectral_cols]
        result = apply_snv(X_df)
        assert isinstance(result, np.ndarray)
        assert result.shape == (len(train_df), len(spectral_cols))


class TestVerifySnvProperties:
    def test_returns_dict(self, X_snv):
        result = verify_snv_properties(X_snv)
        assert isinstance(result, dict)

    def test_has_required_keys(self, X_snv):
        result = verify_snv_properties(X_snv)
        for key in ["mean_of_row_means", "mean_of_row_stds", "max_abs_row_mean", "passes"]:
            assert key in result

    def test_passes_for_snv(self, X_snv):
        result = verify_snv_properties(X_snv)
        assert result["passes"] is True

    def test_fails_for_raw(self, X_raw):
        result = verify_snv_properties(X_raw)
        assert result["passes"] is False


class TestEvaluateSnvEffect:
    def test_returns_dict(self, train_df, spectral_cols):
        result = evaluate_snv_effect(train_df, spectral_cols)
        assert isinstance(result, dict)

    def test_has_required_keys(self, train_df, spectral_cols):
        result = evaluate_snv_effect(train_df, spectral_cols)
        for key in ["rmse_raw", "rmse_snv", "improvement_pct", "drift_raw", "drift_snv"]:
            assert key in result

    def test_rmse_positive(self, train_df, spectral_cols):
        result = evaluate_snv_effect(train_df, spectral_cols)
        assert result["rmse_raw"] > 0
        assert result["rmse_snv"] > 0

    def test_drift_reduced(self, train_df, spectral_cols):
        """SNV後のドリフトがrawより小さくなる"""
        result = evaluate_snv_effect(train_df, spectral_cols)
        assert result["drift_snv"] < result["drift_raw"]
