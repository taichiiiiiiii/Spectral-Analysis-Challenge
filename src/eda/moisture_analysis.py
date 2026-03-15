"""含水率分析モジュール（Issue #3）"""
import numpy as np
import pandas as pd
from scipy import stats


def compute_moisture_stats(df: pd.DataFrame) -> pd.Series:
    moisture = df["含水率"]
    return pd.Series({
        "mean": moisture.mean(),
        "std": moisture.std(),
        "min": moisture.min(),
        "max": moisture.max(),
        "median": moisture.median(),
        "skewness": float(stats.skew(moisture)),
    })


def compute_moisture_stats_by_species(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("樹種")["含水率"].agg(
        mean="mean", std="std", min="min", max="max", count="count"
    )


def detect_outliers(df: pd.DataFrame, threshold: float = 3.0) -> pd.Series:
    moisture = df["含水率"]
    z_scores = np.abs(stats.zscore(moisture))
    return pd.Series(z_scores > threshold, index=df.index)


def check_log_transform_benefit(df: pd.DataFrame) -> dict:
    moisture = df["含水率"]
    original_skewness = float(stats.skew(moisture))
    log_skewness = float(stats.skew(np.log1p(moisture)))
    return {
        "original_skewness": original_skewness,
        "log_skewness": log_skewness,
        "recommend_log": abs(log_skewness) < abs(original_skewness),
    }


def check_sample_number_duplicates(df: pd.DataFrame) -> dict:
    total_rows = len(df)
    unique_samples = df["sample number"].nunique()
    has_duplicates = unique_samples < total_rows
    recommended_cv = "GroupKFold + LOSO-CV" if has_duplicates else "LOSO-CV"
    return {
        "has_duplicates": has_duplicates,
        "unique_samples": unique_samples,
        "total_rows": total_rows,
        "recommended_cv": recommended_cv,
    }
