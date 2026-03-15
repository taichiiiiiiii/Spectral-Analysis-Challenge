"""スペクトル異常値検出モジュール（Issue #8）"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats


def compute_pca_for_outlier(
    df: pd.DataFrame,
    spectral_cols: list[str],
    variance_threshold: float = 0.95,
) -> dict:
    X = df[spectral_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=variance_threshold)
    scores = pca.fit_transform(X_scaled)
    X_reconstructed = pca.inverse_transform(scores)

    return {
        "scores": scores,
        "loadings": pca.components_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "n_components": pca.n_components_,
        "X_reconstructed": X_reconstructed,
        "X_scaled": X_scaled,
    }


def compute_hotelling_t2(scores: np.ndarray) -> np.ndarray:
    n, p = scores.shape
    cov = np.cov(scores.T)
    if p == 1:
        cov_inv = np.array([[1.0 / cov]])
    else:
        cov_inv = np.linalg.pinv(cov)
    mean = scores.mean(axis=0)
    diff = scores - mean
    t2 = np.array([d @ cov_inv @ d for d in diff])
    return t2


def compute_q_residuals(X_original: np.ndarray, X_reconstructed: np.ndarray) -> np.ndarray:
    residuals = X_original - X_reconstructed
    return np.sum(residuals ** 2, axis=1)


def detect_spectral_outliers(
    df: pd.DataFrame,
    spectral_cols: list[str],
    alpha: float = 0.05,
) -> dict:
    pca_result = compute_pca_for_outlier(df, spectral_cols)
    scores = pca_result["scores"]
    X_scaled = pca_result["X_scaled"]
    X_reconstructed = pca_result["X_reconstructed"]

    t2 = compute_hotelling_t2(scores)
    q = compute_q_residuals(X_scaled, X_reconstructed)

    t2_threshold = np.percentile(t2, (1 - alpha) * 100)
    q_threshold = np.percentile(q, (1 - alpha) * 100)

    t2_outliers = t2 > t2_threshold
    q_outliers = q > q_threshold
    combined_outliers = t2_outliers | q_outliers
    outlier_indices = np.where(combined_outliers)[0].tolist()

    return {
        "t2_outliers": t2_outliers,
        "q_outliers": q_outliers,
        "combined_outliers": combined_outliers,
        "outlier_indices": outlier_indices,
        "outlier_count": len(outlier_indices),
        "t2_values": t2,
        "q_values": q,
        "t2_threshold": t2_threshold,
        "q_threshold": q_threshold,
    }
