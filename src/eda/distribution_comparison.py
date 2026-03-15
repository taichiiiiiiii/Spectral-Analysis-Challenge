"""train/test分布比較モジュール（Issue #4）"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def check_species_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    train_species = set(train_df["樹種"].unique())
    test_species = set(test_df["樹種"].unique())
    overlap = train_species & test_species
    return {
        "overlap_count": len(overlap),
        "overlap_species": list(overlap),
        "train_species_count": len(train_species),
        "test_species_count": len(test_species),
    }


def compute_spectral_statistics(
    train_df: pd.DataFrame, test_df: pd.DataFrame, spectral_cols: list[str]
) -> pd.DataFrame:
    def stats(df):
        vals = df[spectral_cols].values.flatten()
        return {
            "mean_intensity": vals.mean(),
            "std_intensity": vals.std(),
            "min_intensity": vals.min(),
            "max_intensity": vals.max(),
        }

    return pd.DataFrame(
        {"train": stats(train_df), "test": stats(test_df)}
    ).T


def compute_pca_projection(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    spectral_cols: list[str],
    n_components: int = 10,
) -> dict:
    X_train = train_df[spectral_cols].values
    X_test = test_df[spectral_cols].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=n_components)
    train_scores = pca.fit_transform(X_train_scaled)
    test_scores = pca.transform(X_test_scaled)

    return {
        "train_scores": train_scores,
        "test_scores": test_scores,
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }


def compute_domain_shift_score(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    spectral_cols: list[str],
) -> dict:
    train_mean = train_df[spectral_cols].mean()
    test_mean = test_df[spectral_cols].mean()
    train_std = train_df[spectral_cols].std()

    mean_diff = ((test_mean - train_mean) / (train_std + 1e-8)).abs().mean()
    std_diff = (test_df[spectral_cols].std() / (train_std + 1e-8) - 1).abs().mean()

    if mean_diff < 0.1:
        shift_level = "low"
    elif mean_diff < 0.5:
        shift_level = "medium"
    else:
        shift_level = "high"

    return {
        "mean_diff": float(mean_diff),
        "std_diff": float(std_diff),
        "shift_level": shift_level,
    }
