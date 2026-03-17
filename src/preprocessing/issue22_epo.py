"""EPO（External Parameter Orthogonalization）モジュール

対応Issue: #22 EPOの実装・効果検証
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/22

参考: Roger, J.M. et al. (2003)
樹種の違いによるスペクトル変動を直交投影で除去する。
"""
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut


def compute_epo_projection(X: np.ndarray, groups: np.ndarray, n_components: int = 5) -> np.ndarray:
    """EPO投影行列を計算する。

    グループ（樹種）間の変動をPCAで抽出し、その空間を除去する投影行列を返す。

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    groups : np.ndarray of shape (n_samples,), グループラベル（樹種）
    n_components : int, 除去する主成分数

    Returns
    -------
    P : np.ndarray of shape (n_features, n_features), 投影行列
    """
    # グループ平均行列を計算
    unique_groups = np.unique(groups)
    group_means = np.array([X[groups == g].mean(axis=0) for g in unique_groups])

    # グループ平均のPCAで樹種間変動の主成分を抽出
    group_means_centered = group_means - group_means.mean(axis=0)
    U, S, Vt = np.linalg.svd(group_means_centered, full_matrices=False)

    # 上位n_components個の主成分（樹種間変動の方向）
    n_comp = min(n_components, len(unique_groups) - 1, Vt.shape[0])
    D = Vt[:n_comp].T  # (n_features, n_comp)

    # 投影行列: P = I - D @ D^T （Dの空間を除去）
    P = np.eye(X.shape[1]) - D @ D.T

    return P


def apply_epo(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    """EPO投影を適用する。

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    P : np.ndarray of shape (n_features, n_features)

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)
    """
    return X @ P


def evaluate_epo_effect(
    df: pd.DataFrame,
    spectral_cols: list[str],
    n_epo_components_list: list[int] = None,
    n_pls_components: int = 4,
) -> pd.DataFrame:
    """EPOのPLS LOSO-CV RMSEを評価する。

    Data leakage防止: 各foldのtrain側でEPO投影行列を計算。

    Parameters
    ----------
    df : pd.DataFrame
    spectral_cols : list[str]
    n_epo_components_list : list[int], 試すEPO成分数のリスト
    n_pls_components : int, PLS成分数

    Returns
    -------
    pd.DataFrame with columns: method, n_epo_components, rmse, rmse_std
    """
    if n_epo_components_list is None:
        n_epo_components_list = [1, 2, 3, 4, 5, 6, 7, 8]

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
        "n_epo_components": 0,
        "rmse": float(np.mean(fold_rmses_raw)),
        "rmse_std": float(np.std(fold_rmses_raw)),
    })

    # EPO with different component counts
    for n_epo in n_epo_components_list:
        fold_rmses = []
        for train_idx, test_idx in logo.split(X_raw, y, groups):
            # train側でEPO投影行列を計算
            P = compute_epo_projection(X_raw[train_idx], groups[train_idx], n_components=n_epo)
            X_train_epo = apply_epo(X_raw[train_idx], P)
            X_test_epo = apply_epo(X_raw[test_idx], P)

            pls = PLSRegression(n_components=n_pls_components)
            pls.fit(X_train_epo, y[train_idx])
            pred = pls.predict(X_test_epo).ravel()
            fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))

        rows.append({
            "method": f"EPO(n={n_epo})",
            "n_epo_components": n_epo,
            "rmse": float(np.mean(fold_rmses)),
            "rmse_std": float(np.std(fold_rmses)),
        })

    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
