"""前処理 + EPO の組み合わせ最適化

対応Issue: #29
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/29
"""
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut

from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue19_msc import compute_msc_reference, apply_msc
from src.preprocessing.issue20_savgol import apply_savgol
from src.preprocessing.issue24_detrending import apply_snv_detrending
from src.preprocessing.issue25_emsc import apply_emsc
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo


def evaluate_epo_preprocessing_combos(
    df: pd.DataFrame,
    spectral_cols: list[str],
    n_pls: int = 4,
    n_epo: int = 1,
) -> pd.DataFrame:
    """EPO + 各種前処理の組み合わせを評価する。

    Returns
    -------
    pd.DataFrame with columns: method, rmse, rmse_std
    """
    X_raw = df[spectral_cols].values
    y = df["含水率"].values
    groups = df["樹種"].values
    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_raw, y, groups))

    combos = {}

    # EPO only (baseline)
    def epo_only(X_train_raw, X_test_raw, train_groups):
        P = compute_epo_projection(X_train_raw, train_groups, n_components=n_epo)
        return apply_epo(X_train_raw, P), apply_epo(X_test_raw, P)
    combos["EPO"] = epo_only

    # EPO → SNV
    def epo_snv(X_train_raw, X_test_raw, train_groups):
        P = compute_epo_projection(X_train_raw, train_groups, n_components=n_epo)
        return apply_snv(apply_epo(X_train_raw, P)), apply_snv(apply_epo(X_test_raw, P))
    combos["EPO→SNV"] = epo_snv

    # SNV → EPO
    def snv_epo(X_train_raw, X_test_raw, train_groups):
        X_train_snv = apply_snv(X_train_raw)
        X_test_snv = apply_snv(X_test_raw)
        P = compute_epo_projection(X_train_snv, train_groups, n_components=n_epo)
        return apply_epo(X_train_snv, P), apply_epo(X_test_snv, P)
    combos["SNV→EPO"] = snv_epo

    # EPO → SG 2d
    def epo_sg2d(X_train_raw, X_test_raw, train_groups):
        P = compute_epo_projection(X_train_raw, train_groups, n_components=n_epo)
        return apply_savgol(apply_epo(X_train_raw, P), deriv=2), apply_savgol(apply_epo(X_test_raw, P), deriv=2)
    combos["EPO→SG2d"] = epo_sg2d

    # SG 2d → EPO
    def sg2d_epo(X_train_raw, X_test_raw, train_groups):
        X_train_sg = apply_savgol(X_train_raw, deriv=2)
        X_test_sg = apply_savgol(X_test_raw, deriv=2)
        P = compute_epo_projection(X_train_sg, train_groups, n_components=n_epo)
        return apply_epo(X_train_sg, P), apply_epo(X_test_sg, P)
    combos["SG2d→EPO"] = sg2d_epo

    # EPO → SNV → SG 1d
    def epo_snv_sg1d(X_train_raw, X_test_raw, train_groups):
        P = compute_epo_projection(X_train_raw, train_groups, n_components=n_epo)
        X_tr = apply_savgol(apply_snv(apply_epo(X_train_raw, P)), deriv=1)
        X_te = apply_savgol(apply_snv(apply_epo(X_test_raw, P)), deriv=1)
        return X_tr, X_te
    combos["EPO→SNV→SG1d"] = epo_snv_sg1d

    # SNV+DT → EPO
    def snvdt_epo(X_train_raw, X_test_raw, train_groups):
        X_train_snvdt = apply_snv_detrending(X_train_raw, poly_order=2)
        X_test_snvdt = apply_snv_detrending(X_test_raw, poly_order=2)
        P = compute_epo_projection(X_train_snvdt, train_groups, n_components=n_epo)
        return apply_epo(X_train_snvdt, P), apply_epo(X_test_snvdt, P)
    combos["SNV+DT→EPO"] = snvdt_epo

    # EPO → EMSC(poly=1)
    def epo_emsc(X_train_raw, X_test_raw, train_groups):
        P = compute_epo_projection(X_train_raw, train_groups, n_components=n_epo)
        X_tr_epo = apply_epo(X_train_raw, P)
        X_te_epo = apply_epo(X_test_raw, P)
        ref = X_tr_epo.mean(axis=0)
        return apply_emsc(X_tr_epo, ref, poly_order=1), apply_emsc(X_te_epo, ref, poly_order=1)
    combos["EPO→EMSC(1)"] = epo_emsc

    rows = []
    for name, fn in combos.items():
        fold_rmses = []
        for train_idx, test_idx in folds:
            X_train, X_test = fn(X_raw[train_idx], X_raw[test_idx], groups[train_idx])
            pls = PLSRegression(n_components=n_pls)
            pls.fit(X_train, y[train_idx])
            pred = pls.predict(X_test).ravel()
            fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
        rows.append({
            "method": name,
            "rmse": float(np.mean(fold_rmses)),
            "rmse_std": float(np.std(fold_rmses)),
        })
        print(f"{name}: RMSE={rows[-1]['rmse']:.2f} ± {rows[-1]['rmse_std']:.2f}")

    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
