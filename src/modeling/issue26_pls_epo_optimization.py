"""PLS成分数とEPOパラメータの同時最適化

対応Issue: #26
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/26
"""
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut

from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo


def grid_search_pls_epo(
    df: pd.DataFrame,
    spectral_cols: list[str],
    pls_range: list[int] = None,
    epo_range: list[int] = None,
) -> pd.DataFrame:
    """PLS成分数 × EPO成分数のグリッドサーチ。

    Returns
    -------
    pd.DataFrame with columns: n_pls, n_epo, rmse, rmse_std
    """
    if pls_range is None:
        pls_range = list(range(1, 16))
    if epo_range is None:
        epo_range = [0, 1, 2, 3, 4, 5]

    X_raw = df[spectral_cols].values
    y = df["含水率"].values
    groups = df["樹種"].values
    logo = LeaveOneGroupOut()

    # Pre-compute fold indices
    folds = list(logo.split(X_raw, y, groups))

    rows = []
    for n_epo in epo_range:
        for n_pls in pls_range:
            fold_rmses = []
            for train_idx, test_idx in folds:
                if n_epo > 0:
                    P = compute_epo_projection(X_raw[train_idx], groups[train_idx], n_components=n_epo)
                    X_train = apply_epo(X_raw[train_idx], P)
                    X_test = apply_epo(X_raw[test_idx], P)
                else:
                    X_train = X_raw[train_idx]
                    X_test = X_raw[test_idx]

                pls = PLSRegression(n_components=n_pls)
                pls.fit(X_train, y[train_idx])
                pred = pls.predict(X_test).ravel()
                fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))

            rows.append({
                "n_pls": n_pls,
                "n_epo": n_epo,
                "rmse": float(np.mean(fold_rmses)),
                "rmse_std": float(np.std(fold_rmses)),
            })
            print(f"  n_epo={n_epo}, n_pls={n_pls}: RMSE={rows[-1]['rmse']:.2f} ± {rows[-1]['rmse_std']:.2f}")

    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
