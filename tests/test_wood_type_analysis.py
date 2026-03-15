"""針葉樹 vs 広葉樹分析のテスト（Issue #9）"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.eda.data_loader import load_train, load_test, get_spectral_columns
from src.eda.wood_type_analysis import (
    assign_wood_type,
    compute_mean_spectrum_by_wood_type,
    compute_moisture_stats_by_wood_type,
    compute_wood_type_ratio,
)

DATA_DIR = Path("Input_data")

SOFTWOOD = {"ヒノキ", "ベイスギ", "スプルース", "ベイマツ", "米ヒバ", "スギ"}
HARDWOOD = {"イチョウ", "ウエンジ", "ウォールナット", "クリ", "チェリー", "トチ",
            "ナラ", "ホワイトオーク", "ケヤキ", "クスノキ", "タモ", "チーク", "ヤマザクラ"}


@pytest.fixture(scope="module")
def train_df():
    return load_train(DATA_DIR)


@pytest.fixture(scope="module")
def test_df():
    return load_test(DATA_DIR)


@pytest.fixture(scope="module")
def spectral_cols(train_df):
    return get_spectral_columns(train_df)


@pytest.fixture(scope="module")
def train_with_type(train_df):
    return assign_wood_type(train_df)


class TestAssignWoodType:
    def test_returns_dataframe(self, train_df):
        result = assign_wood_type(train_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_wood_type_column(self, train_df):
        result = assign_wood_type(train_df)
        assert "wood_type" in result.columns

    def test_values_are_valid(self, train_df):
        result = assign_wood_type(train_df)
        assert set(result["wood_type"].unique()).issubset({"softwood", "hardwood"})

    def test_train_softwood_count(self, train_with_type):
        softwood_species = train_with_type[train_with_type["wood_type"] == "softwood"]["樹種"].unique()
        assert len(softwood_species) == 5

    def test_train_hardwood_count(self, train_with_type):
        hardwood_species = train_with_type[train_with_type["wood_type"] == "hardwood"]["樹種"].unique()
        assert len(hardwood_species) == 8


class TestMeanSpectrumByWoodType:
    def test_returns_dataframe(self, train_with_type, spectral_cols):
        result = compute_mean_spectrum_by_wood_type(train_with_type, spectral_cols)
        assert isinstance(result, pd.DataFrame)

    def test_has_two_rows(self, train_with_type, spectral_cols):
        result = compute_mean_spectrum_by_wood_type(train_with_type, spectral_cols)
        assert len(result) == 2

    def test_index_values(self, train_with_type, spectral_cols):
        result = compute_mean_spectrum_by_wood_type(train_with_type, spectral_cols)
        assert "softwood" in result.index
        assert "hardwood" in result.index


class TestMoistureStatsByWoodType:
    def test_returns_dataframe(self, train_with_type):
        result = compute_moisture_stats_by_wood_type(train_with_type)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, train_with_type):
        result = compute_moisture_stats_by_wood_type(train_with_type)
        for col in ["mean", "std", "min", "max", "count"]:
            assert col in result.columns


class TestWoodTypeRatio:
    def test_returns_dict(self, train_df, test_df):
        result = compute_wood_type_ratio(train_df, test_df)
        assert isinstance(result, dict)

    def test_has_required_keys(self, train_df, test_df):
        result = compute_wood_type_ratio(train_df, test_df)
        for key in ["train_softwood_ratio", "train_hardwood_ratio",
                    "test_softwood_ratio", "test_hardwood_ratio"]:
            assert key in result

    def test_ratios_sum_to_one(self, train_df, test_df):
        result = compute_wood_type_ratio(train_df, test_df)
        assert abs(result["train_softwood_ratio"] + result["train_hardwood_ratio"] - 1.0) < 1e-6
        assert abs(result["test_softwood_ratio"] + result["test_hardwood_ratio"] - 1.0) < 1e-6
