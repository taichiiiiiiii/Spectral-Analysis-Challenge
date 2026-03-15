"""スペクトル分析モジュール

対応Issue: #2 スペクトルの可視化（樹種別・含水率別）
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/2
"""
import numpy as np
import pandas as pd


def compute_mean_spectrum_by_species(
    df: pd.DataFrame, spectral_cols: list[str]
) -> pd.DataFrame:
    return df.groupby("樹種")[spectral_cols].mean()


def compute_mean_spectrum_by_moisture_range(
    df: pd.DataFrame, spectral_cols: list[str]
) -> pd.DataFrame:
    q33, q66 = df["含水率"].quantile([0.33, 0.66])
    labels = pd.cut(
        df["含水率"],
        bins=[-np.inf, q33, q66, np.inf],
        labels=["low", "mid", "high"],
    )
    return df.groupby(labels, observed=True)[spectral_cols].mean()


def compute_spectral_correlation_with_moisture(
    df: pd.DataFrame, spectral_cols: list[str]
) -> pd.Series:
    return df[spectral_cols].corrwith(df["含水率"])


def find_water_absorption_bands(
    wavenumbers: np.ndarray,
    correlation: pd.Series,
    top_n: int = 10,
) -> dict:
    corr_values = correlation.values
    top_positive_idx = np.argsort(corr_values)[-top_n:][::-1]
    top_negative_idx = np.argsort(corr_values)[:top_n]

    return {
        "top_positive": wavenumbers[top_positive_idx],
        "top_negative": wavenumbers[top_negative_idx],
    }
