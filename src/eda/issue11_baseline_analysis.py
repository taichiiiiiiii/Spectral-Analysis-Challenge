"""ベースラインドリフト・多重共線性分析モジュール

対応Issue: #11 ベースラインドリフトと多重共線性の確認
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/11
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def compute_baseline_drift(
    df: pd.DataFrame, spectral_cols: list[str]
) -> dict:
    # ベースラインドリフト = サンプル間のスペクトル平均値のばらつき
    sample_means = df[spectral_cols].mean(axis=1)
    drift_by_species = df.groupby("樹種").apply(
        lambda g: g[spectral_cols].mean(axis=1).std(), include_groups=False
    )

    return {
        "mean_drift": float(sample_means.std()),
        "std_drift": float(sample_means.std()),
        "max_drift": float(sample_means.max() - sample_means.min()),
        "drift_by_species": drift_by_species,
    }


def compute_effective_rank(
    df: pd.DataFrame, spectral_cols: list[str]
) -> dict:
    X = df[spectral_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    _, s, _ = np.linalg.svd(X_scaled, full_matrices=False)
    total_variance = (s ** 2).sum()
    cumvar = np.cumsum(s ** 2) / total_variance

    rank_95 = int(np.searchsorted(cumvar, 0.95) + 1)
    rank_99 = int(np.searchsorted(cumvar, 0.99) + 1)
    total_features = len(spectral_cols)

    return {
        "rank_95": rank_95,
        "rank_99": rank_99,
        "total_features": total_features,
        "compression_ratio_95": rank_95 / total_features,
        "cumulative_variance": cumvar,
    }


def compute_adjacent_correlation(
    df: pd.DataFrame, spectral_cols: list[str]
) -> dict:
    X = df[spectral_cols].values
    n_cols = X.shape[1]

    adj_corrs = []
    for i in range(n_cols - 1):
        c = np.corrcoef(X[:, i], X[:, i + 1])[0, 1]
        adj_corrs.append(abs(c))

    adj_corrs = np.array(adj_corrs)

    return {
        "mean_adjacent_corr": float(adj_corrs.mean()),
        "min_adjacent_corr": float(adj_corrs.min()),
        "high_corr_ratio": float((adj_corrs > 0.99).mean()),
    }
