"""乾燥速度・サンプル数偏り分析モジュール

対応Issue: #17 樹種ごとの乾燥速度とサンプル数の偏り確認
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/17
"""
import numpy as np
import pandas as pd


def compute_drying_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sp, group in df.groupby("樹種"):
        moisture = group["含水率"].sort_values(ascending=False)
        moisture_range = moisture.max() - moisture.min()
        intervals = moisture.diff().abs().dropna()
        avg_interval = float(intervals.mean()) if len(intervals) > 0 else 0.0
        rows.append({
            "species": sp,
            "sample_count": len(group),
            "moisture_range": float(moisture_range),
            "avg_interval": avg_interval,
        })
    return pd.DataFrame(rows).sort_values("sample_count").reset_index(drop=True)


def compute_moisture_interval(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sp, group in df.groupby("樹種"):
        moisture = group["含水率"].sort_values(ascending=False).values
        intervals = np.abs(np.diff(moisture))
        rows.append({
            "species": sp,
            "mean_interval": float(intervals.mean()) if len(intervals) > 0 else 0.0,
            "std_interval": float(intervals.std()) if len(intervals) > 1 else 0.0,
            "n_intervals": len(intervals),
        })
    return pd.DataFrame(rows)


def assess_sample_imbalance(
    df: pd.DataFrame, spectral_cols: list[str]
) -> dict:
    counts = df.groupby("樹種").size()
    imbalance_ratio = float(counts.max() / counts.min())
    min_species = str(counts.idxmin())
    max_species = str(counts.idxmax())
    needs_reweighting = imbalance_ratio > 3.0

    if needs_reweighting:
        recommendation = (
            f"不均衡比率{imbalance_ratio:.1f}倍（{min_species}:{counts.min()}件 vs "
            f"{max_species}:{counts.max()}件）。サンプル重み付けを検討。"
        )
    else:
        recommendation = (
            f"不均衡比率{imbalance_ratio:.1f}倍。許容範囲内のため重み付けは不要。"
        )
    return {
        "imbalance_ratio": imbalance_ratio,
        "min_species": min_species,
        "max_species": max_species,
        "needs_reweighting": needs_reweighting,
        "recommendation": recommendation,
    }
