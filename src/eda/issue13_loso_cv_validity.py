"""LOSO-CV妥当性確認モジュール

対応Issue: #13 LOSO-CVの妥当性確認（fold別含水率カバレッジ）
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/13
"""
import pandas as pd
import numpy as np


def compute_fold_coverage(df: pd.DataFrame) -> pd.DataFrame:
    species_list = df["樹種"].unique()
    rows = []
    for sp in species_list:
        holdout = df[df["樹種"] == sp]["含水率"]
        train = df[df["樹種"] != sp]["含水率"]
        is_covered = (holdout.min() >= train.min()) and (holdout.max() <= train.max())
        rows.append({
            "holdout_species": sp,
            "holdout_min": holdout.min(),
            "holdout_max": holdout.max(),
            "train_min": train.min(),
            "train_max": train.max(),
            "is_covered": is_covered,
        })
    return pd.DataFrame(rows)


def identify_problematic_folds(df: pd.DataFrame) -> pd.DataFrame:
    coverage = compute_fold_coverage(df)
    sample_stats = compute_fold_sample_stats(df)
    problems = []

    for _, row in coverage.iterrows():
        issues = []
        if not row["is_covered"]:
            issues.append("含水率レンジ未カバー")
        n_holdout = sample_stats[sample_stats["holdout_species"] == row["holdout_species"]]["holdout_count"].values[0]
        if n_holdout < 30:
            issues.append(f"サンプル数少({n_holdout}件)")
        if issues:
            problems.append({
                "holdout_species": row["holdout_species"],
                "issue": " / ".join(issues),
                "recommendation": "このfoldのRMSEは参考値として扱い、全体平均から除外することを検討",
            })

    return pd.DataFrame(problems) if problems else pd.DataFrame(
        columns=["holdout_species", "issue", "recommendation"]
    )


def compute_fold_sample_stats(df: pd.DataFrame) -> pd.DataFrame:
    species_list = df["樹種"].unique()
    rows = []
    for sp in species_list:
        holdout_count = int((df["樹種"] == sp).sum())
        train_count = len(df) - holdout_count
        rows.append({
            "holdout_species": sp,
            "holdout_count": holdout_count,
            "train_count": train_count,
            "holdout_ratio": holdout_count / len(df),
        })
    return pd.DataFrame(rows)
