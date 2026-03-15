"""データ読み込みのテスト（TDD: テストを先に書く）"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.eda.data_loader import load_train, load_test, get_spectral_columns, get_wavenumbers


DATA_DIR = Path("Input_data")


class TestLoadTrain:
    def test_returns_dataframe(self):
        df = load_train(DATA_DIR)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        df = load_train(DATA_DIR)
        assert "sample number" in df.columns
        assert "species number" in df.columns
        assert "樹種" in df.columns
        assert "含水率" in df.columns

    def test_shape(self):
        df = load_train(DATA_DIR)
        assert df.shape[0] == 1322
        assert df.shape[1] == 1559

    def test_no_missing_values_in_target(self):
        df = load_train(DATA_DIR)
        assert df["含水率"].isna().sum() == 0

    def test_moisture_range(self):
        df = load_train(DATA_DIR)
        assert df["含水率"].min() >= 0
        assert df["含水率"].max() <= 400


class TestLoadTest:
    def test_returns_dataframe(self):
        df = load_test(DATA_DIR)
        assert isinstance(df, pd.DataFrame)

    def test_has_no_moisture_column(self):
        df = load_test(DATA_DIR)
        assert "含水率" not in df.columns

    def test_shape(self):
        df = load_test(DATA_DIR)
        assert df.shape[0] == 550
        assert df.shape[1] == 1558


class TestGetSpectralColumns:
    def test_returns_list(self):
        df = load_train(DATA_DIR)
        cols = get_spectral_columns(df)
        assert isinstance(cols, list)

    def test_count(self):
        df = load_train(DATA_DIR)
        cols = get_spectral_columns(df)
        assert len(cols) == 1555

    def test_all_numeric(self):
        df = load_train(DATA_DIR)
        cols = get_spectral_columns(df)
        for col in cols:
            assert df[col].dtype in [np.float64, np.float32]


class TestGetWavenumbers:
    def test_returns_array(self):
        df = load_train(DATA_DIR)
        cols = get_spectral_columns(df)
        wn = get_wavenumbers(cols)
        assert isinstance(wn, np.ndarray)

    def test_range(self):
        df = load_train(DATA_DIR)
        cols = get_spectral_columns(df)
        wn = get_wavenumbers(cols)
        assert wn.min() >= 3999
        assert wn.max() <= 9994

    def test_sorted_ascending(self):
        df = load_train(DATA_DIR)
        cols = get_spectral_columns(df)
        wn = get_wavenumbers(cols)
        assert np.all(wn[:-1] <= wn[1:])
