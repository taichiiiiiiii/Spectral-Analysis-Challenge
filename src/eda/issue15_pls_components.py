"""PLSコンポーネント数事前評価モジュール

対応Issue: #15 PLSコンポーネント数の事前評価
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/15
"""
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def run_loso_cv_pls(
    df: pd.DataFrame,
    spectral_cols: list[str],
    max_components: int = 20,
    use_log: bool = False,
) -> pd.DataFrame:
    species_list = df["樹種"].unique()
    results = []

    for n_comp in range(1, max_components + 1):
        fold_rmses = []
        for sp in species_list:
            train = df[df["樹種"] != sp]
            val = df[df["樹種"] == sp]

            X_train = train[spectral_cols].values
            X_val = val[spectral_cols].values
            y_train = train["含水率"].values
            y_val = val["含水率"].values

            if use_log:
                y_train = np.log1p(y_train)

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)

            pls = PLSRegression(n_components=n_comp, max_iter=500)
            pls.fit(X_train_s, y_train)
            y_pred = pls.predict(X_val_s).ravel()

            if use_log:
                y_pred = np.expm1(y_pred)

            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            fold_rmses.append(rmse)

        results.append({
            "n_components": n_comp,
            "rmse_mean": float(np.mean(fold_rmses)),
            "rmse_std": float(np.std(fold_rmses)),
        })

    return pd.DataFrame(results)


def find_optimal_components(cv_result: pd.DataFrame) -> dict:
    optimal_idx = cv_result["rmse_mean"].idxmin()
    optimal_n = int(cv_result.loc[optimal_idx, "n_components"])
    optimal_rmse = float(cv_result.loc[optimal_idx, "rmse_mean"])

    # 過学習開始点: RMSEが最小値から1%以上増加した後に反転する点
    rmses = cv_result["rmse_mean"].values
    overfitting_starts_at = optimal_n
    for i in range(optimal_idx + 1, len(rmses)):
        if rmses[i] > optimal_rmse * 1.01:
            overfitting_starts_at = int(cv_result.loc[i, "n_components"])
            break

    return {
        "optimal_n": optimal_n,
        "optimal_rmse": optimal_rmse,
        "overfitting_starts_at": overfitting_starts_at,
    }


def compare_log_vs_raw_target(
    df: pd.DataFrame,
    spectral_cols: list[str],
    max_components: int = 20,
) -> pd.DataFrame:
    raw_result = run_loso_cv_pls(df, spectral_cols, max_components, use_log=False)
    log_result = run_loso_cv_pls(df, spectral_cols, max_components, use_log=True)

    return pd.DataFrame({
        "n_components": raw_result["n_components"],
        "rmse_raw": raw_result["rmse_mean"],
        "rmse_log": log_result["rmse_mean"],
    })
