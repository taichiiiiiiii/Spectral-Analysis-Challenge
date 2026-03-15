"""針葉樹 vs 広葉樹分析モジュール

対応Issue: #9 針葉樹 vs 広葉樹のスペクトル特性分析
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/9
"""
import pandas as pd
import numpy as np

SOFTWOOD_SPECIES = {"ヒノキ", "ベイスギ", "スプルース", "ベイマツ", "米ヒバ", "スギ"}


def assign_wood_type(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["wood_type"] = df["樹種"].apply(
        lambda s: "softwood" if s in SOFTWOOD_SPECIES else "hardwood"
    )
    return df


def compute_mean_spectrum_by_wood_type(
    df: pd.DataFrame, spectral_cols: list[str]
) -> pd.DataFrame:
    return df.groupby("wood_type")[spectral_cols].mean()


def compute_moisture_stats_by_wood_type(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("wood_type")["含水率"].agg(
        mean="mean", std="std", min="min", max="max", count="count"
    )


def compute_wood_type_ratio(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> dict:
    train = assign_wood_type(train_df)
    test = assign_wood_type(test_df)

    train_counts = train["wood_type"].value_counts(normalize=True)
    test_counts = test["wood_type"].value_counts(normalize=True)

    return {
        "train_softwood_ratio": float(train_counts.get("softwood", 0)),
        "train_hardwood_ratio": float(train_counts.get("hardwood", 0)),
        "test_softwood_ratio": float(test_counts.get("softwood", 0)),
        "test_hardwood_ratio": float(test_counts.get("hardwood", 0)),
    }
