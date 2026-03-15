"""MSC（Multiple Scatter Correction）モジュール

対応Issue: #19 MSC（Multiple Scatter Correction）の実装・効果検証
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/19
"""
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut


def compute_msc_reference(X) -> np.ndarray:
    """MSCのリファレンス（train平均スペクトル）を計算する。

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)

    Returns
    -------
    np.ndarray of shape (n_features,)
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    return np.array(X, dtype=float).mean(axis=0)


def apply_msc(X, reference: np.ndarray) -> np.ndarray:
    """MSC変換を適用する。

    各サンプル x_i について:
        x_i ≈ a_i * reference + b_i  (線形回帰)
        x_msc_i = (x_i - b_i) / a_i

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    reference : np.ndarray of shape (n_features,)

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    X = np.array(X, dtype=float)
    ref = reference.astype(float)
    X_msc = np.zeros_like(X)
    # 1次線形回帰: x_i = a * ref + b
    ref_with_bias = np.column_stack([ref, np.ones(len(ref))])
    for i in range(len(X)):
        coeffs, _, _, _ = np.linalg.lstsq(ref_with_bias, X[i], rcond=None)
        a, b = coeffs[0], coeffs[1]
        X_msc[i] = (X[i] - b) / a
    return X_msc


def evaluate_msc_effect(df: pd.DataFrame, spectral_cols: list[str], n_components: int = 4) -> dict:
    """MSC前後のPLS LOSO-CV RMSEとベースラインドリフトを比較する。

    ⚠️ LOSO-CVでは各foldのtrain部分でreferenceを計算（data leakage防止）

    Returns
    -------
    dict with keys:
        rmse_raw, rmse_msc, improvement_pct, drift_raw, drift_msc
    """
    X_raw = df[spectral_cols].values
    y = df["含水率"].values
    groups = df["樹種"].values

    logo = LeaveOneGroupOut()

    def loso_rmse_msc():
        errors = []
        for train_idx, test_idx in logo.split(X_raw, y, groups):
            ref = compute_msc_reference(X_raw[train_idx])
            X_train_msc = apply_msc(X_raw[train_idx], ref)
            X_test_msc = apply_msc(X_raw[test_idx], ref)
            pls = PLSRegression(n_components=n_components)
            pls.fit(X_train_msc, y[train_idx])
            pred = pls.predict(X_test_msc).ravel()
            errors.extend((pred - y[test_idx]) ** 2)
        return float(np.sqrt(np.mean(errors)))

    def loso_rmse_raw():
        errors = []
        for train_idx, test_idx in logo.split(X_raw, y, groups):
            pls = PLSRegression(n_components=n_components)
            pls.fit(X_raw[train_idx], y[train_idx])
            pred = pls.predict(X_raw[test_idx]).ravel()
            errors.extend((pred - y[test_idx]) ** 2)
        return float(np.sqrt(np.mean(errors)))

    rmse_raw = loso_rmse_raw()
    rmse_msc = loso_rmse_msc()
    improvement_pct = float((rmse_raw - rmse_msc) / rmse_raw * 100)

    # ドリフト: train全体のMSCで比較
    ref_all = compute_msc_reference(X_raw)
    X_msc_all = apply_msc(X_raw, ref_all)
    drift_raw = float(X_raw.mean(axis=1).std())
    drift_msc = float(X_msc_all.mean(axis=1).std())

    return {
        "rmse_raw": rmse_raw,
        "rmse_msc": rmse_msc,
        "improvement_pct": improvement_pct,
        "drift_raw": drift_raw,
        "drift_msc": drift_msc,
    }
