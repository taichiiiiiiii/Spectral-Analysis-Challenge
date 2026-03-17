"""ドメイン適応（CORAL / 擬似ラベル）

対応Issue: #28
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/28

CORAL: Sun & Saenko (2016) "Return of Frustratingly Easy Domain Adaptation"
"""
import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut

from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo


def coral_transform(X_source: np.ndarray, X_target: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """CORAL（Correlation Alignment）変換。

    ソースデータの共分散をターゲットの共分散に合わせる。

    Parameters
    ----------
    X_source : np.ndarray of shape (n_source, n_features)
    X_target : np.ndarray of shape (n_target, n_features)
    reg : float, 正則化パラメータ

    Returns
    -------
    np.ndarray of shape (n_source, n_features)
    """
    # 中心化
    mu_s = X_source.mean(axis=0)
    mu_t = X_target.mean(axis=0)
    Xs = X_source - mu_s
    Xt = X_target - mu_t

    n_features = X_source.shape[1]

    # 共分散行列
    Cs = Xs.T @ Xs / (len(Xs) - 1) + reg * np.eye(n_features)
    Ct = Xt.T @ Xt / (len(Xt) - 1) + reg * np.eye(n_features)

    # Cs^{-1/2} と Ct^{1/2}
    Cs_half_inv = np.real(sqrtm(np.linalg.inv(Cs)))
    Ct_half = np.real(sqrtm(Ct))

    # 変換: X_aligned = (X_source - mu_s) @ Cs^{-1/2} @ Ct^{1/2} + mu_t
    X_aligned = Xs @ Cs_half_inv @ Ct_half + mu_t

    return X_aligned


def generate_pseudo_labels(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 4,
) -> np.ndarray:
    """PLSモデルで擬似ラベルを生成する。

    Parameters
    ----------
    X_train : np.ndarray of shape (n_train, n_features)
    y_train : np.ndarray of shape (n_train,)
    X_test : np.ndarray of shape (n_test, n_features)
    n_components : int

    Returns
    -------
    np.ndarray of shape (n_test,)
    """
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_train, y_train)
    return pls.predict(X_test).ravel()


def evaluate_domain_adaptation(
    df: pd.DataFrame,
    spectral_cols: list[str],
    n_pls_components: int = 4,
    n_epo: int = 1,
) -> pd.DataFrame:
    """ドメイン適応手法のLOSO-CV RMSE評価。

    LOSO-CVの各foldで、test樹種をtargetドメインとして適応。

    Returns
    -------
    pd.DataFrame with columns: method, rmse, rmse_std
    """
    X_raw = df[spectral_cols].values
    y = df["含水率"].values
    groups = df["樹種"].values
    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_raw, y, groups))

    rows = []

    # Baseline: EPO + PLS
    fold_rmses = []
    for train_idx, test_idx in folds:
        P = compute_epo_projection(X_raw[train_idx], groups[train_idx], n_components=n_epo)
        X_train = apply_epo(X_raw[train_idx], P)
        X_test = apply_epo(X_raw[test_idx], P)
        pls = PLSRegression(n_components=n_pls_components)
        pls.fit(X_train, y[train_idx])
        pred = pls.predict(X_test).ravel()
        fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
    rows.append({"method": "EPO+PLS", "rmse": np.mean(fold_rmses), "rmse_std": np.std(fold_rmses)})

    # CORAL (on raw spectra)
    fold_rmses = []
    for train_idx, test_idx in folds:
        X_train_coral = coral_transform(X_raw[train_idx], X_raw[test_idx])
        pls = PLSRegression(n_components=n_pls_components)
        pls.fit(X_train_coral, y[train_idx])
        pred = pls.predict(X_raw[test_idx]).ravel()
        fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
    rows.append({"method": "CORAL+PLS", "rmse": np.mean(fold_rmses), "rmse_std": np.std(fold_rmses)})

    # EPO + CORAL
    fold_rmses = []
    for train_idx, test_idx in folds:
        P = compute_epo_projection(X_raw[train_idx], groups[train_idx], n_components=n_epo)
        X_train_epo = apply_epo(X_raw[train_idx], P)
        X_test_epo = apply_epo(X_raw[test_idx], P)
        X_train_coral = coral_transform(X_train_epo, X_test_epo)
        pls = PLSRegression(n_components=n_pls_components)
        pls.fit(X_train_coral, y[train_idx])
        pred = pls.predict(X_test_epo).ravel()
        fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
    rows.append({"method": "EPO+CORAL+PLS", "rmse": np.mean(fold_rmses), "rmse_std": np.std(fold_rmses)})

    # Pseudo-labeling: EPO + PLS with pseudo-labeled test data
    fold_rmses = []
    for train_idx, test_idx in folds:
        P = compute_epo_projection(X_raw[train_idx], groups[train_idx], n_components=n_epo)
        X_train_epo = apply_epo(X_raw[train_idx], P)
        X_test_epo = apply_epo(X_raw[test_idx], P)

        # Generate pseudo labels
        pseudo_y = generate_pseudo_labels(X_train_epo, y[train_idx], X_test_epo, n_components=n_pls_components)

        # Retrain with pseudo-labeled data
        X_combined = np.vstack([X_train_epo, X_test_epo])
        y_combined = np.concatenate([y[train_idx], pseudo_y])
        pls = PLSRegression(n_components=n_pls_components)
        pls.fit(X_combined, y_combined)
        pred = pls.predict(X_test_epo).ravel()
        fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
    rows.append({"method": "EPO+PseudoLabel+PLS", "rmse": np.mean(fold_rmses), "rmse_std": np.std(fold_rmses)})

    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
