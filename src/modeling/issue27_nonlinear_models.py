"""EPO + 非線形モデル（SVR / XGBoost / RF）

対応Issue: #27
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/27
"""
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo


def evaluate_nonlinear_models(
    df: pd.DataFrame,
    spectral_cols: list[str],
    n_epo: int = 1,
    pca_components_list: list[int] = None,
) -> pd.DataFrame:
    """EPO + 非線形モデルのLOSO-CV RMSE評価。

    Returns
    -------
    pd.DataFrame with columns: model, n_pca, rmse, rmse_std
    """
    if pca_components_list is None:
        pca_components_list = [3, 5, 10, 20]

    X_raw = df[spectral_cols].values
    y = df["含水率"].values
    groups = df["樹種"].values
    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_raw, y, groups))

    rows = []

    # Baseline: EPO + PLS(4)
    fold_rmses = []
    for train_idx, test_idx in folds:
        P = compute_epo_projection(X_raw[train_idx], groups[train_idx], n_components=n_epo)
        X_train = apply_epo(X_raw[train_idx], P)
        X_test = apply_epo(X_raw[test_idx], P)
        pls = PLSRegression(n_components=4)
        pls.fit(X_train, y[train_idx])
        pred = pls.predict(X_test).ravel()
        fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
    rows.append({"model": "EPO+PLS(4)", "n_pca": "N/A", "rmse": np.mean(fold_rmses), "rmse_std": np.std(fold_rmses)})
    print(f"EPO+PLS(4): RMSE={np.mean(fold_rmses):.2f}")

    # SVR with PCA
    for n_pca in pca_components_list:
        fold_rmses = []
        for train_idx, test_idx in folds:
            P = compute_epo_projection(X_raw[train_idx], groups[train_idx], n_components=n_epo)
            X_train = apply_epo(X_raw[train_idx], P)
            X_test = apply_epo(X_raw[test_idx], P)

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            pca = PCA(n_components=n_pca)
            X_train_pca = pca.fit_transform(X_train_s)
            X_test_pca = pca.transform(X_test_s)

            svr = SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1)
            svr.fit(X_train_pca, y[train_idx])
            pred = svr.predict(X_test_pca)
            fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
        rows.append({"model": f"EPO+PCA+SVR", "n_pca": n_pca, "rmse": np.mean(fold_rmses), "rmse_std": np.std(fold_rmses)})
        print(f"EPO+PCA({n_pca})+SVR: RMSE={np.mean(fold_rmses):.2f}")

    # Random Forest with PCA
    for n_pca in pca_components_list:
        fold_rmses = []
        for train_idx, test_idx in folds:
            P = compute_epo_projection(X_raw[train_idx], groups[train_idx], n_components=n_epo)
            X_train = apply_epo(X_raw[train_idx], P)
            X_test = apply_epo(X_raw[test_idx], P)

            pca = PCA(n_components=n_pca)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)

            rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
            rf.fit(X_train_pca, y[train_idx])
            pred = rf.predict(X_test_pca)
            fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
        rows.append({"model": f"EPO+PCA+RF", "n_pca": n_pca, "rmse": np.mean(fold_rmses), "rmse_std": np.std(fold_rmses)})
        print(f"EPO+PCA({n_pca})+RF: RMSE={np.mean(fold_rmses):.2f}")

    # XGBoost with PCA (if available)
    try:
        from xgboost import XGBRegressor
        for n_pca in pca_components_list:
            fold_rmses = []
            for train_idx, test_idx in folds:
                P = compute_epo_projection(X_raw[train_idx], groups[train_idx], n_components=n_epo)
                X_train = apply_epo(X_raw[train_idx], P)
                X_test = apply_epo(X_raw[test_idx], P)

                pca = PCA(n_components=n_pca)
                X_train_pca = pca.fit_transform(X_train)
                X_test_pca = pca.transform(X_test)

                xgb = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
                xgb.fit(X_train_pca, y[train_idx])
                pred = xgb.predict(X_test_pca)
                fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
            rows.append({"model": f"EPO+PCA+XGB", "n_pca": n_pca, "rmse": np.mean(fold_rmses), "rmse_std": np.std(fold_rmses)})
            print(f"EPO+PCA({n_pca})+XGB: RMSE={np.mean(fold_rmses):.2f}")
    except ImportError:
        print("XGBoost not installed, skipping")

    # LightGBM with PCA (if available)
    try:
        import lightgbm as lgb
        for n_pca in pca_components_list:
            fold_rmses = []
            for train_idx, test_idx in folds:
                P = compute_epo_projection(X_raw[train_idx], groups[train_idx], n_components=n_epo)
                X_train = apply_epo(X_raw[train_idx], P)
                X_test = apply_epo(X_raw[test_idx], P)

                pca = PCA(n_components=n_pca)
                X_train_pca = pca.fit_transform(X_train)
                X_test_pca = pca.transform(X_test)

                lgbm = lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42, verbose=-1)
                lgbm.fit(X_train_pca, y[train_idx])
                pred = lgbm.predict(X_test_pca)
                fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
            rows.append({"model": f"EPO+PCA+LGBM", "n_pca": n_pca, "rmse": np.mean(fold_rmses), "rmse_std": np.std(fold_rmses)})
            print(f"EPO+PCA({n_pca})+LGBM: RMSE={np.mean(fold_rmses):.2f}")
    except ImportError:
        print("LightGBM not installed, skipping")

    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
