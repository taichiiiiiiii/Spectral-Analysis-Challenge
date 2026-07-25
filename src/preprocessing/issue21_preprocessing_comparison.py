"""前処理組み合わせ比較モジュール

対応Issue: #21 前処理組み合わせ比較（PLS LOSO-CV RMSE）
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/21
"""
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut

from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue19_msc import compute_msc_reference, apply_msc
from src.preprocessing.issue20_savgol import apply_savgol

VALID_METHODS = {"Raw", "SNV", "MSC", "SNV+1d", "SNV+2d", "MSC+1d", "MSC+2d"}


def build_preprocessing_pipeline(X: np.ndarray, method: str, X_ref: np.ndarray = None) -> np.ndarray:
    """指定した前処理メソッドを適用する。

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    method : str, 前処理名（"Raw", "SNV", "MSC", "SNV+1d", "SNV+2d", "MSC+1d", "MSC+2d"）
    X_ref : np.ndarray, MSC用のリファレンス計算元（Noneの場合はXを使用）

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)
    """
    if method not in VALID_METHODS:
        raise ValueError(f"Invalid method '{method}'. Choose from {VALID_METHODS}")

    if X_ref is None:
        X_ref = X

    if method == "Raw":
        return X
    elif method == "SNV":
        return apply_snv(X)
    elif method == "MSC":
        ref = compute_msc_reference(X_ref)
        return apply_msc(X, ref)
    elif method == "SNV+1d":
        return apply_savgol(apply_snv(X), deriv=1)
    elif method == "SNV+2d":
        return apply_savgol(apply_snv(X), deriv=2)
    elif method == "MSC+1d":
        ref = compute_msc_reference(X_ref)
        return apply_savgol(apply_msc(X, ref), deriv=1)
    elif method == "MSC+2d":
        ref = compute_msc_reference(X_ref)
        return apply_savgol(apply_msc(X, ref), deriv=2)


def evaluate_preprocessing_combination(
    df: pd.DataFrame,
    spectral_cols: list[str],
    method: str,
    n_components: int = 4,
) -> dict:
    """指定した前処理のPLS LOSO-CV RMSEを計算する。

    data leakage防止のため、各foldのtrain側でreferenceを計算する。

    Returns
    -------
    dict with keys: method, rmse, rmse_std, n_components
    """
    X_raw = df[spectral_cols].values
    y = df["含水率"].values
    groups = df["樹種"].values
    logo = LeaveOneGroupOut()

    fold_rmses = []
    for train_idx, test_idx in logo.split(X_raw, y, groups):
        X_train = build_preprocessing_pipeline(X_raw[train_idx], method, X_ref=X_raw[train_idx])
        X_test = build_preprocessing_pipeline(X_raw[test_idx], method, X_ref=X_raw[train_idx])
        pls = PLSRegression(n_components=n_components)
        pls.fit(X_train, y[train_idx])
        pred = pls.predict(X_test).ravel()
        fold_rmse = float(np.sqrt(np.mean((pred - y[test_idx]) ** 2)))
        fold_rmses.append(fold_rmse)

    return {
        "method": method,
        "rmse": float(np.mean(fold_rmses)),
        "rmse_std": float(np.std(fold_rmses)),
        "n_components": n_components,
    }


def compare_all_combinations(
    df: pd.DataFrame,
    spectral_cols: list[str],
    n_components: int = 4,
) -> pd.DataFrame:
    """全前処理パターンのPLS LOSO-CV RMSEを一括比較する。

    Returns
    -------
    pd.DataFrame sorted by rmse with columns:
        method, rmse, rmse_std, n_components, rank
    """
    rows = []
    for method in VALID_METHODS:
        result = evaluate_preprocessing_combination(df, spectral_cols, method, n_components)
        rows.append(result)

    result_df = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    result_df["rank"] = range(1, len(result_df) + 1)
    return result_df
