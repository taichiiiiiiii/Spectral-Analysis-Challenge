"""前処理深掘り Part3: Box-Cox + SNV/Raw目的変数変換（Part2の残り）"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut
from scipy import stats
from scipy.special import inv_boxcox

from src.eda.data_loader import load_train, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"


def main():
    df = load_train(DATA_DIR)
    spectral_cols = get_spectral_columns(df)
    X_raw = df[spectral_cols].values
    y = df["含水率"].values
    groups = df["樹種"].values
    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_raw, y, groups))
    results = []

    # Box-Cox
    print("--- Box-Cox変換 ---")
    y_bc, lam = stats.boxcox(y + 1)
    print(f"  lambda = {lam:.4f}")
    for n_comp in [3, 4, 5, 6]:
        fold_rmses = []
        for train_idx, test_idx in folds:
            P = compute_epo_projection(X_raw[train_idx], groups[train_idx], n_components=1)
            X_tr = apply_epo(X_raw[train_idx], P)
            X_te = apply_epo(X_raw[test_idx], P)
            pls = PLSRegression(n_components=n_comp)
            pls.fit(X_tr, y_bc[train_idx])
            pred_bc = pls.predict(X_te).ravel()
            pred = inv_boxcox(pred_bc, lam) - 1
            pred = np.clip(pred, 0, None)
            fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
        rmse, std = np.mean(fold_rmses), np.std(fold_rmses)
        results.append({"method": f"EPO+PLS({n_comp})+BoxCox", "rmse": rmse, "rmse_std": std})
        print(f"  EPO+PLS({n_comp})+BoxCox: RMSE={rmse:.2f} ± {std:.2f}")

    # SNV + 目的変数変換
    print("\n--- SNV + 目的変数変換 ---")
    for tname, y_t, inv_fn in [
        ("log", np.log1p(y), np.expm1),
        ("sqrt", np.sqrt(y), lambda x: np.clip(x, 0, None) ** 2),
    ]:
        for n_comp in [2, 3, 4]:
            fold_rmses = []
            for train_idx, test_idx in folds:
                X_tr = apply_snv(X_raw[train_idx])
                X_te = apply_snv(X_raw[test_idx])
                pls = PLSRegression(n_components=n_comp)
                pls.fit(X_tr, y_t[train_idx])
                pred_t = pls.predict(X_te).ravel()
                pred = inv_fn(pred_t)
                fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
            rmse, std = np.mean(fold_rmses), np.std(fold_rmses)
            results.append({"method": f"SNV+PLS({n_comp})+{tname}(y)", "rmse": rmse, "rmse_std": std})
            print(f"  SNV+PLS({n_comp})+{tname}(y): RMSE={rmse:.2f} ± {std:.2f}")

    # Raw + 目的変数変換
    print("\n--- Raw + 目的変数変換 ---")
    for tname, y_t, inv_fn in [
        ("log", np.log1p(y), np.expm1),
        ("sqrt", np.sqrt(y), lambda x: np.clip(x, 0, None) ** 2),
    ]:
        for n_comp in [3, 4, 5]:
            fold_rmses = []
            for train_idx, test_idx in folds:
                pls = PLSRegression(n_components=n_comp)
                pls.fit(X_raw[train_idx], y_t[train_idx])
                pred_t = pls.predict(X_raw[test_idx]).ravel()
                pred = inv_fn(pred_t)
                fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
            rmse, std = np.mean(fold_rmses), np.std(fold_rmses)
            results.append({"method": f"Raw+PLS({n_comp})+{tname}(y)", "rmse": rmse, "rmse_std": std})
            print(f"  Raw+PLS({n_comp})+{tname}(y): RMSE={rmse:.2f} ± {std:.2f}")

    result_df = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    print("\n=== Part3 全結果 ===")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
