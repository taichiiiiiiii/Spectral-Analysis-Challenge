"""EMSC（Extended Multiplicative Scatter Correction）モジュール

対応Issue: #25 EMSCの実装・効果検証
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/25

通常のMSCを拡張し、多項式ベースラインと化学的干渉を同時にモデル化する。
"""
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut

from src.preprocessing.issue19_msc import compute_msc_reference, apply_msc


def apply_emsc(
    X: np.ndarray,
    reference: np.ndarray,
    poly_order: int = 2,
) -> np.ndarray:
    """EMSC変換を適用する。

    各サンプル x_i について:
        x_i ≈ a * reference + b + c1*wn + c2*wn^2 + ... (多項式ベースライン)
        x_emsc_i = (x_i - b - c1*wn - c2*wn^2 - ...) / a

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    reference : np.ndarray of shape (n_features,), リファレンススペクトル
    poly_order : int, 多項式ベースラインの次数

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    X = np.array(X, dtype=float)
    ref = reference.astype(float)
    n_samples, n_features = X.shape

    # 波数軸（正規化）
    wn = np.linspace(-1, 1, n_features)

    # 基底行列: [reference, 1, wn, wn^2, ...]
    basis = [ref, np.ones(n_features)]
    for p in range(1, poly_order + 1):
        basis.append(wn ** p)
    basis = np.column_stack(basis)  # (n_features, n_basis)

    X_emsc = np.zeros_like(X)
    for i in range(n_samples):
        coeffs, _, _, _ = np.linalg.lstsq(basis, X[i], rcond=None)
        a = coeffs[0]  # リファレンスの係数
        # ベースライン = b + c1*wn + c2*wn^2 + ...
        baseline = basis[:, 1:] @ coeffs[1:]
        X_emsc[i] = (X[i] - baseline) / a

    return X_emsc


def evaluate_emsc_effect(
    df: pd.DataFrame,
    spectral_cols: list[str],
    n_components: int = 4,
) -> pd.DataFrame:
    """EMSCのPLS LOSO-CV RMSEを評価する。

    Data leakage防止: 各foldのtrain側でリファレンスを計算。

    Returns
    -------
    pd.DataFrame with columns: method, rmse, rmse_std
    """
    X_raw = df[spectral_cols].values
    y = df["含水率"].values
    groups = df["樹種"].values
    logo = LeaveOneGroupOut()

    rows = []

    # Raw baseline
    fold_rmses_raw = []
    for train_idx, test_idx in logo.split(X_raw, y, groups):
        pls = PLSRegression(n_components=n_components)
        pls.fit(X_raw[train_idx], y[train_idx])
        pred = pls.predict(X_raw[test_idx]).ravel()
        fold_rmses_raw.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
    rows.append({
        "method": "Raw",
        "rmse": float(np.mean(fold_rmses_raw)),
        "rmse_std": float(np.std(fold_rmses_raw)),
    })

    # MSC (for comparison)
    fold_rmses_msc = []
    for train_idx, test_idx in logo.split(X_raw, y, groups):
        ref = compute_msc_reference(X_raw[train_idx])
        X_train_msc = apply_msc(X_raw[train_idx], ref)
        X_test_msc = apply_msc(X_raw[test_idx], ref)
        pls = PLSRegression(n_components=n_components)
        pls.fit(X_train_msc, y[train_idx])
        pred = pls.predict(X_test_msc).ravel()
        fold_rmses_msc.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
    rows.append({
        "method": "MSC",
        "rmse": float(np.mean(fold_rmses_msc)),
        "rmse_std": float(np.std(fold_rmses_msc)),
    })

    # EMSC with different polynomial orders
    for poly_order in [1, 2, 3]:
        fold_rmses = []
        for train_idx, test_idx in logo.split(X_raw, y, groups):
            ref = compute_msc_reference(X_raw[train_idx])
            X_train_emsc = apply_emsc(X_raw[train_idx], ref, poly_order=poly_order)
            X_test_emsc = apply_emsc(X_raw[test_idx], ref, poly_order=poly_order)
            pls = PLSRegression(n_components=n_components)
            pls.fit(X_train_emsc, y[train_idx])
            pred = pls.predict(X_test_emsc).ravel()
            fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
        rows.append({
            "method": f"EMSC(poly={poly_order})",
            "rmse": float(np.mean(fold_rmses)),
            "rmse_std": float(np.std(fold_rmses)),
        })

    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
