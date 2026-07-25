"""変動係数マップモジュール

対応Issue: #16 各波数の変動係数（CV）マップ
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/16
"""
import numpy as np
import pandas as pd


def compute_cv_map(df: pd.DataFrame, spectral_cols: list[str]) -> pd.Series:
    X = df[spectral_cols]
    cv = X.std() / (X.mean().abs() + 1e-8)
    cv.index = pd.Index([float(c) for c in spectral_cols], dtype=float)
    return cv


def find_high_cv_bands(
    df: pd.DataFrame, spectral_cols: list[str], top_n: int = 20
) -> pd.DataFrame:
    cv = compute_cv_map(df, spectral_cols)
    top = cv.nlargest(top_n)
    return pd.DataFrame({
        "wavenumber": top.index,
        "cv": top.values,
        "rank": range(1, top_n + 1),
    }).reset_index(drop=True)


def compare_cv_with_correlation(
    df: pd.DataFrame, spectral_cols: list[str]
) -> pd.DataFrame:
    cv = compute_cv_map(df, spectral_cols)
    corr = df[spectral_cols].corrwith(df["含水率"]).abs()
    corr.index = pd.Index([float(c) for c in spectral_cols], dtype=float)

    result = pd.DataFrame({
        "wavenumber": cv.index,
        "cv": cv.values,
        "abs_correlation": corr.values,
    })
    result["cv_rank"] = result["cv"].rank(ascending=False).astype(int)
    result["corr_rank"] = result["abs_correlation"].rank(ascending=False).astype(int)
    return result
