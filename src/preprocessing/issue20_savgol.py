"""Savitzky-Golay微分モジュール

対応Issue: #20 Savitzky-Golay微分（1次・2次）の実装・効果検証
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/20
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut

from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue19_msc import compute_msc_reference, apply_msc


def apply_savgol(
    X,
    deriv: int = 1,
    window_length: int = 11,
    polyorder: int = 2,
) -> np.ndarray:
    """Savitzky-Golay微分を適用する。

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    deriv : int, 微分次数 (1 or 2)
    window_length : int, ウィンドウ幅（奇数）
    polyorder : int, 多項式次数

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    X = np.array(X, dtype=float)
    return savgol_filter(X, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=1)


def _loso_rmse_with_preprocessing(X_raw, y, groups, preprocess_fn, n_components=4):
    """前処理関数を受け取りLOSO-CV RMSEを計算する（data leakage防止）。"""
    logo = LeaveOneGroupOut()
    errors = []
    for train_idx, test_idx in logo.split(X_raw, y, groups):
        X_train = preprocess_fn(X_raw[train_idx], X_raw[train_idx])
        X_test = preprocess_fn(X_raw[test_idx], X_raw[train_idx])
        pls = PLSRegression(n_components=n_components)
        pls.fit(X_train, y[train_idx])
        pred = pls.predict(X_test).ravel()
        errors.extend((pred - y[test_idx]) ** 2)
    return float(np.sqrt(np.mean(errors)))


def evaluate_savgol_combinations(
    df: pd.DataFrame, spectral_cols: list[str], n_components: int = 4
) -> pd.DataFrame:
    """全前処理パターンのPLS LOSO-CV RMSEを比較する。

    Returns
    -------
    pd.DataFrame sorted by rmse with columns:
        preprocessing, rmse, n_components
    """
    X_raw = df[spectral_cols].values
    y = df["含水率"].values
    groups = df["樹種"].values
    logo = LeaveOneGroupOut()

    def loso_rmse(preprocess_fn):
        errors = []
        for train_idx, test_idx in logo.split(X_raw, y, groups):
            X_train = preprocess_fn(X_raw[train_idx], X_raw[train_idx])
            X_test = preprocess_fn(X_raw[test_idx], X_raw[train_idx])
            pls = PLSRegression(n_components=n_components)
            pls.fit(X_train, y[train_idx])
            pred = pls.predict(X_test).ravel()
            errors.extend((pred - y[test_idx]) ** 2)
        return float(np.sqrt(np.mean(errors)))

    def raw_fn(X, _ref):
        return X

    def snv_fn(X, _ref):
        return apply_snv(X)

    def msc_fn(X, X_train):
        ref = compute_msc_reference(X_train)
        return apply_msc(X, ref)

    def snv_1d_fn(X, _ref):
        return apply_savgol(apply_snv(X), deriv=1)

    def snv_2d_fn(X, _ref):
        return apply_savgol(apply_snv(X), deriv=2)

    def msc_1d_fn(X, X_train):
        ref = compute_msc_reference(X_train)
        return apply_savgol(apply_msc(X, ref), deriv=1)

    def msc_2d_fn(X, X_train):
        ref = compute_msc_reference(X_train)
        return apply_savgol(apply_msc(X, ref), deriv=2)

    methods = {
        "Raw": raw_fn,
        "SNV": snv_fn,
        "MSC": msc_fn,
        "SNV+1d": snv_1d_fn,
        "SNV+2d": snv_2d_fn,
        "MSC+1d": msc_1d_fn,
        "MSC+2d": msc_2d_fn,
    }

    rows = []
    for name, fn in methods.items():
        rmse = loso_rmse(fn)
        rows.append({"preprocessing": name, "rmse": rmse, "n_components": n_components})

    result = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    return result
