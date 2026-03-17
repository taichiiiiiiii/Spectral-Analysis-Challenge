"""スタッキングアンサンブル

対応Issue: #30
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/30

複数ベースモデルのOOF予測をRidge回帰でブレンドする。
"""
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo


def generate_oof_predictions(
    df: pd.DataFrame,
    spectral_cols: list[str],
    n_pls_components: int = 4,
    n_epo: int = 1,
) -> pd.DataFrame:
    """各ベースモデルのLOSO-CV OOF予測を生成する。

    Returns
    -------
    pd.DataFrame of shape (n_samples, n_base_models)
    """
    X_raw = df[spectral_cols].values
    y = df["含水率"].values
    groups = df["樹種"].values
    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_raw, y, groups))
    n_samples = len(df)

    oof = {}

    # Model 1: EPO + PLS
    preds = np.zeros(n_samples)
    for train_idx, test_idx in folds:
        P = compute_epo_projection(X_raw[train_idx], groups[train_idx], n_components=n_epo)
        X_tr = apply_epo(X_raw[train_idx], P)
        X_te = apply_epo(X_raw[test_idx], P)
        pls = PLSRegression(n_components=n_pls_components)
        pls.fit(X_tr, y[train_idx])
        preds[test_idx] = pls.predict(X_te).ravel()
    oof["EPO+PLS"] = preds

    # Model 2: Raw + PLS
    preds = np.zeros(n_samples)
    for train_idx, test_idx in folds:
        pls = PLSRegression(n_components=n_pls_components)
        pls.fit(X_raw[train_idx], y[train_idx])
        preds[test_idx] = pls.predict(X_raw[test_idx]).ravel()
    oof["Raw+PLS"] = preds

    # Model 3: EPO + PCA + SVR
    preds = np.zeros(n_samples)
    for train_idx, test_idx in folds:
        P = compute_epo_projection(X_raw[train_idx], groups[train_idx], n_components=n_epo)
        X_tr = apply_epo(X_raw[train_idx], P)
        X_te = apply_epo(X_raw[test_idx], P)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        pca = PCA(n_components=10)
        X_tr_pca = pca.fit_transform(X_tr_s)
        X_te_pca = pca.transform(X_te_s)
        svr = SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1)
        svr.fit(X_tr_pca, y[train_idx])
        preds[test_idx] = svr.predict(X_te_pca)
    oof["EPO+SVR"] = preds

    # Model 4: SNV + PLS
    preds = np.zeros(n_samples)
    for train_idx, test_idx in folds:
        X_tr = apply_snv(X_raw[train_idx])
        X_te = apply_snv(X_raw[test_idx])
        pls = PLSRegression(n_components=n_pls_components)
        pls.fit(X_tr, y[train_idx])
        preds[test_idx] = pls.predict(X_te).ravel()
    oof["SNV+PLS"] = preds

    return pd.DataFrame(oof)


def evaluate_stacking(
    df: pd.DataFrame,
    spectral_cols: list[str],
    n_pls_components: int = 4,
    n_epo: int = 1,
) -> pd.DataFrame:
    """スタッキングアンサンブルのLOSO-CV RMSE評価。

    Returns
    -------
    pd.DataFrame with columns: method, rmse, rmse_std
    """
    y = df["含水率"].values
    groups = df["樹種"].values
    logo = LeaveOneGroupOut()

    # Generate OOF predictions
    oof_df = generate_oof_predictions(df, spectral_cols, n_pls_components, n_epo)
    oof_matrix = oof_df.values

    folds = list(logo.split(oof_matrix, y, groups))
    rows = []

    # Individual model RMSEs
    for col in oof_df.columns:
        fold_rmses = []
        for train_idx, test_idx in folds:
            pred = oof_df[col].values[test_idx]
            fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
        rows.append({
            "method": col,
            "rmse": float(np.mean(fold_rmses)),
            "rmse_std": float(np.std(fold_rmses)),
        })

    # Simple average ensemble
    fold_rmses = []
    for train_idx, test_idx in folds:
        pred = oof_matrix[test_idx].mean(axis=1)
        fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
    rows.append({
        "method": "SimpleAvg",
        "rmse": float(np.mean(fold_rmses)),
        "rmse_std": float(np.std(fold_rmses)),
    })

    # Stacking with Ridge (nested CV to avoid leakage)
    fold_rmses = []
    for train_idx, test_idx in folds:
        meta_X_train = oof_matrix[train_idx]
        meta_y_train = y[train_idx]
        meta_X_test = oof_matrix[test_idx]

        ridge = Ridge(alpha=1.0)
        ridge.fit(meta_X_train, meta_y_train)
        pred = ridge.predict(meta_X_test)
        fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
    rows.append({
        "method": "Stacking(Ridge)",
        "rmse": float(np.mean(fold_rmses)),
        "rmse_std": float(np.std(fold_rmses)),
    })

    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
