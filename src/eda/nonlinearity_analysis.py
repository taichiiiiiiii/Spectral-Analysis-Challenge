"""非線形性分析モジュール（Issue #10）"""
import numpy as np
import pandas as pd


def compute_correlation_by_moisture_range(
    df: pd.DataFrame, spectral_cols: list[str]
) -> pd.DataFrame:
    q33, q66 = df["含水率"].quantile([0.33, 0.66])
    bins = pd.cut(df["含水率"], bins=[-np.inf, q33, q66, np.inf], labels=["low", "mid", "high"])
    results = {}
    for label, group in df.groupby(bins, observed=True):
        results[label] = group[spectral_cols].corrwith(group["含水率"])
    return pd.DataFrame(results).T


def compare_linear_vs_log_correlation(
    df: pd.DataFrame, spectral_cols: list[str]
) -> pd.DataFrame:
    linear_corr = df[spectral_cols].corrwith(df["含水率"]).abs()
    log_corr = df[spectral_cols].corrwith(np.log1p(df["含水率"])).abs()
    return pd.DataFrame({
        "linear_corr": linear_corr,
        "log_corr": log_corr,
        "log_improves": log_corr > linear_corr,
    })


def find_most_nonlinear_wavenumbers(
    df: pd.DataFrame, spectral_cols: list[str], top_n: int = 20
) -> pd.DataFrame:
    corr_by_range = compute_correlation_by_moisture_range(df, spectral_cols)
    variance_across_ranges = corr_by_range.var(axis=0)
    top_cols = variance_across_ranges.nlargest(top_n).index

    result = pd.DataFrame({
        "wavenumber": [float(c) for c in top_cols],
        "corr_variance_across_ranges": variance_across_ranges[top_cols].values,
        "nonlinearity_score": variance_across_ranges[top_cols].values,
    }).sort_values("nonlinearity_score", ascending=False).reset_index(drop=True)

    return result
