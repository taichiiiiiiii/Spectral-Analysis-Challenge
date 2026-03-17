"""TCA深掘り + OPLS+sqrt(y) の追加評価"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut

from src.eda.data_loader import load_train, load_test, get_spectral_columns
from src.preprocessing.tca import tca_transform
from src.preprocessing.opls import OPLSFilter
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue18_snv import apply_snv


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main():
    data_dir = Path("Input_data")
    train_df = load_train(data_dir)
    spectral_cols = get_spectral_columns(train_df)
    X_train = train_df[spectral_cols].values
    y = train_df["含水率"].values
    groups = train_df["樹種"].values
    logo = LeaveOneGroupOut()

    results = []

    # TCA deep dive: different gamma values
    print("=== TCA gamma sweep ===")
    for gamma in [0.001, 0.01, 0.1, None]:  # None = 1/d default
        for n_tca in [10, 15, 20]:
            fold_rmses = []
            for train_idx, test_idx in logo.split(X_train, y, groups):
                X_tr, X_te = X_train[train_idx], X_train[test_idx]
                Z_tr, Z_te = tca_transform(X_tr, X_te, n_components=n_tca, gamma=gamma)
                nc_pls = min(n_tca, 4)
                pls = PLSRegression(n_components=nc_pls)
                pls.fit(Z_tr, y[train_idx])
                pred = pls.predict(Z_te).ravel()
                fold_rmses.append(rmse(y[test_idx], pred))
            mean_r = np.mean(fold_rmses)
            std_r = np.std(fold_rmses)
            g_str = f"{gamma}" if gamma else "auto"
            print(f"  TCA(n={n_tca},g={g_str})+PLS({nc_pls}): RMSE={mean_r:.2f} ± {std_r:.2f}")
            results.append({"method": f"TCA(n={n_tca},g={g_str})+PLS({nc_pls})", "rmse": mean_r, "rmse_std": std_r})

    # TCA + sqrt(y)
    print("\n=== TCA + sqrt(y) ===")
    for n_tca in [10, 15]:
        fold_rmses = []
        for train_idx, test_idx in logo.split(X_train, y, groups):
            X_tr, X_te = X_train[train_idx], X_train[test_idx]
            Z_tr, Z_te = tca_transform(X_tr, X_te, n_components=n_tca)
            pls = PLSRegression(n_components=4)
            pls.fit(Z_tr, np.sqrt(y[train_idx]))
            pred_sqrt = pls.predict(Z_te).ravel()
            pred = pred_sqrt ** 2
            fold_rmses.append(rmse(y[test_idx], pred))
        mean_r = np.mean(fold_rmses)
        std_r = np.std(fold_rmses)
        print(f"  TCA({n_tca})+PLS(4)+sqrt(y): RMSE={mean_r:.2f} ± {std_r:.2f}")
        results.append({"method": f"TCA({n_tca})+PLS(4)+sqrt(y)", "rmse": mean_r, "rmse_std": std_r})

    # TCA linear kernel
    print("\n=== TCA linear kernel ===")
    for n_tca in [10, 20]:
        fold_rmses = []
        for train_idx, test_idx in logo.split(X_train, y, groups):
            X_tr, X_te = X_train[train_idx], X_train[test_idx]
            Z_tr, Z_te = tca_transform(X_tr, X_te, n_components=n_tca, kernel="linear")
            pls = PLSRegression(n_components=4)
            pls.fit(Z_tr, y[train_idx])
            pred = pls.predict(Z_te).ravel()
            fold_rmses.append(rmse(y[test_idx], pred))
        mean_r = np.mean(fold_rmses)
        std_r = np.std(fold_rmses)
        print(f"  TCA_linear({n_tca})+PLS(4): RMSE={mean_r:.2f} ± {std_r:.2f}")
        results.append({"method": f"TCA_linear({n_tca})+PLS(4)", "rmse": mean_r, "rmse_std": std_r})

    # OPLS + sqrt(y)
    print("\n=== OPLS + sqrt(y) ===")
    for n_opls in [1, 2, 3]:
        fold_rmses = []
        for train_idx, test_idx in logo.split(X_train, y, groups):
            opls = OPLSFilter(n_components=n_opls)
            X_tr_f = opls.fit_transform(X_train[train_idx], y[train_idx])
            X_te_f = opls.transform(X_train[test_idx])
            pls = PLSRegression(n_components=4)
            pls.fit(X_tr_f, np.sqrt(y[train_idx]))
            pred_sqrt = pls.predict(X_te_f).ravel()
            pred = pred_sqrt ** 2
            fold_rmses.append(rmse(y[test_idx], pred))
        mean_r = np.mean(fold_rmses)
        std_r = np.std(fold_rmses)
        print(f"  OPLS({n_opls})+PLS(4)+sqrt(y): RMSE={mean_r:.2f} ± {std_r:.2f}")
        results.append({"method": f"OPLS({n_opls})+PLS(4)+sqrt(y)", "rmse": mean_r, "rmse_std": std_r})

    # SNV + OPLS + PLS
    print("\n=== SNV + OPLS + PLS ===")
    X_snv = apply_snv(X_train)
    for n_opls in [1, 2]:
        fold_rmses = []
        for train_idx, test_idx in logo.split(X_snv, y, groups):
            opls = OPLSFilter(n_components=n_opls)
            X_tr_f = opls.fit_transform(X_snv[train_idx], y[train_idx])
            X_te_f = opls.transform(X_snv[test_idx])
            pls = PLSRegression(n_components=2)
            pls.fit(X_tr_f, y[train_idx])
            pred = pls.predict(X_te_f).ravel()
            fold_rmses.append(rmse(y[test_idx], pred))
        mean_r = np.mean(fold_rmses)
        std_r = np.std(fold_rmses)
        print(f"  SNV+OPLS({n_opls})+PLS(2): RMSE={mean_r:.2f} ± {std_r:.2f}")
        results.append({"method": f"SNV+OPLS({n_opls})+PLS(2)", "rmse": mean_r, "rmse_std": std_r})

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (sorted by RMSE)")
    print("=" * 60)
    results_df = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    print(results_df.to_string(index=False))
    results_df.to_csv("outputs/new_preprocessing_eval2.csv", index=False)


if __name__ == "__main__":
    main()
