"""SNV（Standard Normal Variate）モジュール

対応Issue: #18 SNV（Standard Normal Variate）の実装・効果検証
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/18
"""
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut


def apply_snv(X) -> np.ndarray:
    """各サンプルのスペクトルにSNV変換を適用する。

    各サンプルについて: x_snv = (x - mean(x)) / std(x)

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    X = np.array(X, dtype=float)
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    return (X - means) / stds


def verify_snv_properties(X_snv: np.ndarray, tol: float = 1e-6) -> dict:
    """SNV変換後の統計的性質を検証する。

    Returns
    -------
    dict with keys:
        mean_of_row_means, mean_of_row_stds, max_abs_row_mean, passes
    """
    row_means = X_snv.mean(axis=1)
    row_stds = X_snv.std(axis=1)
    max_abs_mean = float(np.abs(row_means).max())
    mean_of_stds = float(row_stds.mean())
    passes = max_abs_mean < tol and abs(mean_of_stds - 1.0) < tol
    return {
        "mean_of_row_means": float(row_means.mean()),
        "mean_of_row_stds": mean_of_stds,
        "max_abs_row_mean": max_abs_mean,
        "passes": passes,
    }


def evaluate_snv_effect(df: pd.DataFrame, spectral_cols: list[str], n_components: int = 4) -> dict:
    """SNV前後のPLS LOSO-CV RMSEとベースラインドリフトを比較する。

    Returns
    -------
    dict with keys:
        rmse_raw, rmse_snv, improvement_pct, drift_raw, drift_snv
    """
    X_raw = df[spectral_cols].values
    X_snv = apply_snv(X_raw)
    y = df["含水率"].values
    groups = df["樹種"].values

    logo = LeaveOneGroupOut()

    def loso_rmse(X):
        errors = []
        for train_idx, test_idx in logo.split(X, y, groups):
            pls = PLSRegression(n_components=n_components)
            pls.fit(X[train_idx], y[train_idx])
            pred = pls.predict(X[test_idx]).ravel()
            errors.extend((pred - y[test_idx]) ** 2)
        return float(np.sqrt(np.mean(errors)))

    rmse_raw = loso_rmse(X_raw)
    rmse_snv = loso_rmse(X_snv)
    improvement_pct = float((rmse_raw - rmse_snv) / rmse_raw * 100)

    # ベースラインドリフト: サンプル間のスペクトル平均の標準偏差
    drift_raw = float(X_raw.mean(axis=1).std())
    drift_snv = float(X_snv.mean(axis=1).std())

    return {
        "rmse_raw": rmse_raw,
        "rmse_snv": rmse_snv,
        "improvement_pct": improvement_pct,
        "drift_raw": drift_raw,
        "drift_snv": drift_snv,
    }
