"""MSC（Multiple Scatter Correction）のテスト

対応Issue: #19 MSC（Multiple Scatter Correction）の実装・効果検証
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/19
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.eda.data_loader import load_train, get_spectral_columns
from src.preprocessing.issue19_msc import compute_msc_reference, apply_msc, evaluate_msc_effect

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
def reference(X_raw):
    return compute_msc_reference(X_raw)


@pytest.fixture(scope="module")
def X_msc(X_raw, reference):
    return apply_msc(X_raw, reference)


class TestComputeMscReference:
    def test_returns_ndarray(self, X_raw):
        ref = compute_msc_reference(X_raw)
        assert isinstance(ref, np.ndarray)

    def test_shape(self, X_raw):
        ref = compute_msc_reference(X_raw)
        assert ref.shape == (X_raw.shape[1],)

    def test_equals_mean(self, X_raw):
        ref = compute_msc_reference(X_raw)
        assert np.allclose(ref, X_raw.mean(axis=0))


class TestApplyMsc:
    def test_returns_ndarray(self, X_raw, reference):
        result = apply_msc(X_raw, reference)
        assert isinstance(result, np.ndarray)

    def test_shape_preserved(self, X_raw, X_msc):
        assert X_msc.shape == X_raw.shape

    def test_reference_corrected_to_itself(self, reference):
        """リファレンス自身にMSCを適用すると変化しない"""
        result = apply_msc(reference.reshape(1, -1), reference)
        assert np.allclose(result, reference.reshape(1, -1), atol=1e-6)

    def test_dataframe_input(self, train_df, spectral_cols, reference):
        X_df = train_df[spectral_cols]
        result = apply_msc(X_df, reference)
        assert isinstance(result, np.ndarray)
        assert result.shape == (len(train_df), len(spectral_cols))

    def test_no_data_leakage(self, train_df, spectral_cols):
        """testデータにtrainのreferenceを適用できる（leakage防止の構造確認）"""
        X = train_df[spectral_cols].values
        n = len(X)
        train_part = X[: n // 2]
        test_part = X[n // 2 :]
        ref = compute_msc_reference(train_part)
        result = apply_msc(test_part, ref)
        assert result.shape == test_part.shape


class TestEvaluateMscEffect:
    def test_returns_dict(self, train_df, spectral_cols):
        result = evaluate_msc_effect(train_df, spectral_cols)
        assert isinstance(result, dict)

    def test_has_required_keys(self, train_df, spectral_cols):
        result = evaluate_msc_effect(train_df, spectral_cols)
        for key in ["rmse_raw", "rmse_msc", "improvement_pct", "drift_raw", "drift_msc"]:
            assert key in result

    def test_rmse_positive(self, train_df, spectral_cols):
        result = evaluate_msc_effect(train_df, spectral_cols)
        assert result["rmse_raw"] > 0
        assert result["rmse_msc"] > 0

    def test_drift_reduced(self, train_df, spectral_cols):
        result = evaluate_msc_effect(train_df, spectral_cols)
        assert result["drift_msc"] < result["drift_raw"]
