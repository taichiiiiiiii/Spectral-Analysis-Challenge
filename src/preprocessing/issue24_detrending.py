"""SNV + De-trending モジュール

対応Issue: #24 SNV+De-trending の実装・効果検証
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/24

SNV後に多項式ベースライン除去を適用する。
"""
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut

from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue20_savgol import apply_savgol


def apply_detrending(X: np.ndarray, poly_order: int = 2, wavenumbers: np.ndarray = None) -> np.ndarray:
    """De-trending: 各サンプルから多項式ベースラインを除去する。

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    poly_order : int, 多項式次数（1=線形, 2=2次）
    wavenumbers : np.ndarray of shape (n_features,), 波数軸（Noneの場合はインデックスを使用）

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    X = np.array(X, dtype=float)
    n_samples, n_features = X.shape

    if wavenumbers is None:
        x_axis = np.arange(n_features, dtype=float)
    else:
        x_axis = np.array(wavenumbers, dtype=float)

    # 正規化（数値安定性）
    x_norm = (x_axis - x_axis.mean()) / (x_axis.std() + 1e-10)

    X_dt = np.zeros_like(X)
    for i in range(n_samples):
        coeffs = np.polyfit(x_norm, X[i], poly_order)
        baseline = np.polyval(coeffs, x_norm)
        X_dt[i] = X[i] - baseline

    return X_dt


def apply_snv_detrending(X: np.ndarray, poly_order: int = 2, wavenumbers: np.ndarray = None) -> np.ndarray:
    """SNV + De-trending を適用する。

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    poly_order : int, De-trendingの多項式次数
    wavenumbers : np.ndarray, 波数軸

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)
    """
    X_snv = apply_snv(X)
    return apply_detrending(X_snv, poly_order=poly_order, wavenumbers=wavenumbers)


def evaluate_detrending_effect(
    df: pd.DataFrame,
    spectral_cols: list[str],
    n_components: int = 4,
) -> pd.DataFrame:
    """SNV+DT系の前処理パターンのPLS LOSO-CV RMSEを評価する。

    Returns
    -------
    pd.DataFrame with columns: method, rmse, rmse_std
    """
    X_raw = df[spectral_cols].values
    y = df["含水率"].values
    groups = df["樹種"].values
    logo = LeaveOneGroupOut()

    methods = {}

    # Raw
    def raw_fn(X, _ref):
        return X
    methods["Raw"] = raw_fn

    # SNV
    def snv_fn(X, _ref):
        return apply_snv(X)
    methods["SNV"] = snv_fn

    # DT only (poly=2)
    def dt_fn(X, _ref):
        return apply_detrending(X, poly_order=2)
    methods["DT(poly=2)"] = dt_fn

    # SNV+DT (poly=1)
    def snv_dt1_fn(X, _ref):
        return apply_snv_detrending(X, poly_order=1)
    methods["SNV+DT(poly=1)"] = snv_dt1_fn

    # SNV+DT (poly=2)
    def snv_dt2_fn(X, _ref):
        return apply_snv_detrending(X, poly_order=2)
    methods["SNV+DT(poly=2)"] = snv_dt2_fn

    # SNV+DT+1d
    def snv_dt2_1d_fn(X, _ref):
        return apply_savgol(apply_snv_detrending(X, poly_order=2), deriv=1)
    methods["SNV+DT+1d"] = snv_dt2_1d_fn

    # SNV+DT+2d
    def snv_dt2_2d_fn(X, _ref):
        return apply_savgol(apply_snv_detrending(X, poly_order=2), deriv=2)
    methods["SNV+DT+2d"] = snv_dt2_2d_fn

    rows = []
    for name, fn in methods.items():
        fold_rmses = []
        for train_idx, test_idx in logo.split(X_raw, y, groups):
            X_train = fn(X_raw[train_idx], X_raw[train_idx])
            X_test = fn(X_raw[test_idx], X_raw[train_idx])
            pls = PLSRegression(n_components=n_components)
            pls.fit(X_train, y[train_idx])
            pred = pls.predict(X_test).ravel()
            fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
        rows.append({
            "method": name,
            "rmse": float(np.mean(fold_rmses)),
            "rmse_std": float(np.std(fold_rmses)),
        })

    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
