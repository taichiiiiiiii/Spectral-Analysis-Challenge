"""OSC（Orthogonal Signal Correction）モジュール

対応Issue: #23 OSCの実装・効果検証
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/23

参考: Westerhuis et al. (1998) "Direct Orthogonal Signal Correction"
目的変数と直交する変動成分をスペクトルから除去する。
"""
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut


def compute_osc(X: np.ndarray, y: np.ndarray, n_components: int = 2) -> tuple:
    """OSC（Direct Orthogonal Signal Correction）を計算する。

    Westerhuis法: yと直交するXの変動成分を反復的に除去。

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,)
    n_components : int, 除去する直交成分数

    Returns
    -------
    X_osc : np.ndarray, OSC適用後のスペクトル
    W : np.ndarray, 重み行列（テストデータへの適用用）
    P : np.ndarray, ローディング行列（テストデータへの適用用）
    """
    X_osc = X.copy()
    n_samples, n_features = X.shape
    y_vec = y.reshape(-1, 1) if y.ndim == 1 else y

    W_list = []
    P_list = []

    for _ in range(n_components):
        # yと直交する成分を求める
        # Step 1: Xの最大分散方向（SVD）
        U, S, Vt = np.linalg.svd(X_osc, full_matrices=False)
        t = U[:, 0:1] * S[0]  # 第1スコア

        # Step 2: tをyに直交化
        # t_orth = t - y * (y^T @ t) / (y^T @ y)
        t_orth = t - y_vec @ (y_vec.T @ t) / (y_vec.T @ y_vec)

        # Step 3: 反復精製（NIPALS-like）
        for _iter in range(50):
            # w = X^T @ t_orth / (t_orth^T @ t_orth)
            w = X_osc.T @ t_orth / (t_orth.T @ t_orth)
            w = w / np.linalg.norm(w)

            # t_new = X @ w
            t_new = X_osc @ w

            # yに直交化
            t_new = t_new - y_vec @ (y_vec.T @ t_new) / (y_vec.T @ y_vec)

            # 収束判定
            if np.linalg.norm(t_new - t_orth) / (np.linalg.norm(t_orth) + 1e-10) < 1e-6:
                t_orth = t_new
                break
            t_orth = t_new

        # Step 4: ローディング p = X^T @ t_orth / (t_orth^T @ t_orth)
        p = X_osc.T @ t_orth / (t_orth.T @ t_orth)

        # Step 5: Xから直交成分を除去
        X_osc = X_osc - t_orth @ p.T

        W_list.append(w)
        P_list.append(p)

    W = np.hstack(W_list)
    P = np.hstack(P_list)

    return X_osc, W, P


def apply_osc(X: np.ndarray, W: np.ndarray, P: np.ndarray) -> np.ndarray:
    """学習済みOSCパラメータを新しいデータに適用する。

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    W : np.ndarray of shape (n_features, n_components)
    P : np.ndarray of shape (n_features, n_components)

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)
    """
    X_osc = X.copy()
    n_comp = W.shape[1]
    for i in range(n_comp):
        w = W[:, i:i+1]
        p = P[:, i:i+1]
        t = X_osc @ w
        X_osc = X_osc - t @ p.T
    return X_osc


def evaluate_osc_effect(
    df: pd.DataFrame,
    spectral_cols: list[str],
    n_osc_components_list: list[int] = None,
    n_pls_components: int = 4,
) -> pd.DataFrame:
    """OSCのPLS LOSO-CV RMSEを評価する。

    Data leakage防止: 各foldのtrain側でOSCパラメータを計算。

    Returns
    -------
    pd.DataFrame with columns: method, n_osc_components, rmse, rmse_std
    """
    if n_osc_components_list is None:
        n_osc_components_list = [1, 2, 3, 4, 5]

    X_raw = df[spectral_cols].values
    y = df["含水率"].values
    groups = df["樹種"].values
    logo = LeaveOneGroupOut()

    rows = []

    # Raw baseline
    fold_rmses_raw = []
    for train_idx, test_idx in logo.split(X_raw, y, groups):
        pls = PLSRegression(n_components=n_pls_components)
        pls.fit(X_raw[train_idx], y[train_idx])
        pred = pls.predict(X_raw[test_idx]).ravel()
        fold_rmses_raw.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
    rows.append({
        "method": "Raw",
        "n_osc_components": 0,
        "rmse": float(np.mean(fold_rmses_raw)),
        "rmse_std": float(np.std(fold_rmses_raw)),
    })

    # OSC with different component counts
    for n_osc in n_osc_components_list:
        fold_rmses = []
        for train_idx, test_idx in logo.split(X_raw, y, groups):
            X_osc_train, W, P = compute_osc(
                X_raw[train_idx], y[train_idx], n_components=n_osc
            )
            X_osc_test = apply_osc(X_raw[test_idx], W, P)

            pls = PLSRegression(n_components=n_pls_components)
            pls.fit(X_osc_train, y[train_idx])
            pred = pls.predict(X_osc_test).ravel()
            fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))

        rows.append({
            "method": f"OSC(n={n_osc})",
            "n_osc_components": n_osc,
            "rmse": float(np.mean(fold_rmses)),
            "rmse_std": float(np.std(fold_rmses)),
        })

    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
