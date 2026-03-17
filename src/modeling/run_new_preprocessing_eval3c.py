"""新規前処理5手法のLOSO-CV評価 (KMM除外)

di-PLS(結果済), SA, Constituent EMSC, MMD Selection, JSMKPLS
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneGroupOut

from src.eda.data_loader import load_train, load_test, get_spectral_columns, get_wavenumbers
from src.preprocessing.subspace_alignment import subspace_align
from src.preprocessing.constituent_emsc import constituent_emsc
from src.preprocessing.mmd_selection import select_best_preprocessing_detailed
from src.preprocessing.jsmkpls import jsmkpls_transform
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue20_savgol import apply_savgol


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def p(msg):
    print(msg, flush=True)


def main():
    data_dir = Path("Input_data")
    train_df = load_train(data_dir)
    test_df = load_test(data_dir)
    spectral_cols = get_spectral_columns(train_df)

    X_train = train_df[spectral_cols].values
    X_test = test_df[spectral_cols].values
    y = train_df["含水率"].values
    groups = train_df["樹種"].values
    wavenumbers = get_wavenumbers(spectral_cols)
    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_train, y, groups))

    results = []

    # ===== 1. Subspace Alignment =====
    p("=== Subspace Alignment + PLS ===")
    for d in [5, 10, 20]:
        fold_rmses = []
        for train_idx, test_idx in folds:
            Z_tr, Z_te = subspace_align(X_train[train_idx], X_train[test_idx], n_components=d)
            nc_pls = min(d, 4)
            pls = PLSRegression(n_components=nc_pls)
            pls.fit(Z_tr, y[train_idx])
            preds = pls.predict(Z_te).ravel()
            fold_rmses.append(rmse(y[test_idx], preds))
        mean_r = np.mean(fold_rmses)
        std_r = np.std(fold_rmses)
        p(f"  SA(d={d})+PLS({nc_pls}): RMSE={mean_r:.2f} ± {std_r:.2f}")
        results.append({"method": f"SA(d={d})+PLS({nc_pls})", "rmse": mean_r, "rmse_std": std_r})

    # SA + sqrt(y)
    for d in [10, 20]:
        fold_rmses = []
        for train_idx, test_idx in folds:
            Z_tr, Z_te = subspace_align(X_train[train_idx], X_train[test_idx], n_components=d)
            pls = PLSRegression(n_components=4)
            pls.fit(Z_tr, np.sqrt(y[train_idx]))
            preds = pls.predict(Z_te).ravel() ** 2
            fold_rmses.append(rmse(y[test_idx], preds))
        mean_r = np.mean(fold_rmses)
        std_r = np.std(fold_rmses)
        p(f"  SA(d={d})+PLS(4)+sqrt(y): RMSE={mean_r:.2f} ± {std_r:.2f}")
        results.append({"method": f"SA(d={d})+PLS(4)+sqrt(y)", "rmse": mean_r, "rmse_std": std_r})

    # ===== 2. Constituent EMSC =====
    p("\n=== Constituent EMSC + PLS ===")
    X_emsc = constituent_emsc(X_train, wavenumbers=wavenumbers)
    for nc in [2, 4]:
        fold_rmses = []
        for train_idx, test_idx in folds:
            pls = PLSRegression(n_components=nc)
            pls.fit(X_emsc[train_idx], y[train_idx])
            preds = pls.predict(X_emsc[test_idx]).ravel()
            fold_rmses.append(rmse(y[test_idx], preds))
        mean_r = np.mean(fold_rmses)
        std_r = np.std(fold_rmses)
        p(f"  EMSC+PLS({nc}): RMSE={mean_r:.2f} ± {std_r:.2f}")
        results.append({"method": f"EMSC+PLS({nc})", "rmse": mean_r, "rmse_std": std_r})

    # EMSC + EPO
    P_epo = compute_epo_projection(X_emsc, groups, n_components=1)
    X_emsc_epo = apply_epo(X_emsc, P_epo)
    for nc in [2, 4]:
        fold_rmses = []
        for train_idx, test_idx in folds:
            pls = PLSRegression(n_components=nc)
            pls.fit(X_emsc_epo[train_idx], y[train_idx])
            preds = pls.predict(X_emsc_epo[test_idx]).ravel()
            fold_rmses.append(rmse(y[test_idx], preds))
        mean_r = np.mean(fold_rmses)
        std_r = np.std(fold_rmses)
        p(f"  EMSC+EPO(1)+PLS({nc}): RMSE={mean_r:.2f} ± {std_r:.2f}")
        results.append({"method": f"EMSC+EPO(1)+PLS({nc})", "rmse": mean_r, "rmse_std": std_r})

    # EMSC + sqrt(y)
    for nc in [2, 4]:
        fold_rmses = []
        for train_idx, test_idx in folds:
            pls = PLSRegression(n_components=nc)
            pls.fit(X_emsc[train_idx], np.sqrt(y[train_idx]))
            preds = pls.predict(X_emsc[test_idx]).ravel() ** 2
            fold_rmses.append(rmse(y[test_idx], preds))
        mean_r = np.mean(fold_rmses)
        std_r = np.std(fold_rmses)
        p(f"  EMSC+PLS({nc})+sqrt(y): RMSE={mean_r:.2f} ± {std_r:.2f}")
        results.append({"method": f"EMSC+PLS({nc})+sqrt(y)", "rmse": mean_r, "rmse_std": std_r})

    # ===== 3. MMD Preprocessing Selection =====
    p("\n=== MMD Preprocessing Selection ===")
    preprocessings = {
        "Raw": lambda X: X,
        "SNV": apply_snv,
        "SG1d": lambda X: apply_savgol(X, window_length=11, polyorder=2, deriv=1),
        "SG2d": lambda X: apply_savgol(X, window_length=11, polyorder=2, deriv=2),
        "EMSC": lambda X: constituent_emsc(X, wavenumbers=wavenumbers),
    }
    mmd_results = select_best_preprocessing_detailed(
        X_train, y, X_test, preprocessings, n_pls_components=4
    )
    for r in sorted(mmd_results, key=lambda x: x["mmd"]):
        p(f"  {r['name']}: MMD={r['mmd']:.6f}")
    best_pp = min(mmd_results, key=lambda x: x["mmd"])
    p(f"  → Best: {best_pp['name']}")

    X_best = preprocessings[best_pp["name"]](X_train)
    for nc in [2, 4]:
        fold_rmses = []
        for train_idx, test_idx in folds:
            pls = PLSRegression(n_components=nc)
            pls.fit(X_best[train_idx], y[train_idx])
            preds = pls.predict(X_best[test_idx]).ravel()
            fold_rmses.append(rmse(y[test_idx], preds))
        mean_r = np.mean(fold_rmses)
        std_r = np.std(fold_rmses)
        p(f"  MMD-best({best_pp['name']})+PLS({nc}): RMSE={mean_r:.2f} ± {std_r:.2f}")
        results.append({"method": f"MMD-best({best_pp['name']})+PLS({nc})", "rmse": mean_r, "rmse_std": std_r})

    # ===== 4. JSMKPLS (PCA20) =====
    p("\n=== JSMKPLS (PCA20) ===")
    pca20 = PCA(n_components=20).fit(X_train)
    X_pca20 = pca20.transform(X_train)

    for n_comp in [5, 10]:
        fold_rmses = []
        for train_idx, test_idx in folds:
            try:
                Z_tr, Z_te = jsmkpls_transform(X_pca20[train_idx], X_pca20[test_idx],
                                                n_components=n_comp, mu=1.0, lam=1.0, eta=1.0)
                nc_pls = min(n_comp, 4)
                pls = PLSRegression(n_components=nc_pls)
                pls.fit(Z_tr, y[train_idx])
                preds = pls.predict(Z_te).ravel()
                fold_rmses.append(rmse(y[test_idx], preds))
            except Exception as e:
                p(f"    Error: {e}")
                fold_rmses.append(999.0)
        mean_r = np.mean(fold_rmses)
        std_r = np.std(fold_rmses)
        p(f"  JSMKPLS(n={n_comp})+PLS({nc_pls}): RMSE={mean_r:.2f} ± {std_r:.2f}")
        results.append({"method": f"JSMKPLS(n={n_comp})+PLS({nc_pls})", "rmse": mean_r, "rmse_std": std_r})

    # JSMKPLS + sqrt(y)
    fold_rmses = []
    for train_idx, test_idx in folds:
        try:
            Z_tr, Z_te = jsmkpls_transform(X_pca20[train_idx], X_pca20[test_idx], n_components=10)
            pls = PLSRegression(n_components=4)
            pls.fit(Z_tr, np.sqrt(y[train_idx]))
            preds = pls.predict(Z_te).ravel() ** 2
            fold_rmses.append(rmse(y[test_idx], preds))
        except Exception:
            fold_rmses.append(999.0)
    mean_r = np.mean(fold_rmses)
    std_r = np.std(fold_rmses)
    p(f"  JSMKPLS(10)+PLS(4)+sqrt(y): RMSE={mean_r:.2f} ± {std_r:.2f}")
    results.append({"method": "JSMKPLS(10)+PLS(4)+sqrt(y)", "rmse": mean_r, "rmse_std": std_r})

    # Summary
    p("\n" + "=" * 60)
    p("SUMMARY (sorted by RMSE)")
    p("=" * 60)
    results_df = pd.DataFrame(results)
    results_df = results_df[results_df["rmse"] < 900].sort_values("rmse").reset_index(drop=True)
    p(results_df[["method", "rmse", "rmse_std"]].to_string(index=False))
    results_df.to_csv("outputs/new_preprocessing_eval3.csv", index=False)
    p("\nSaved to outputs/new_preprocessing_eval3.csv")


if __name__ == "__main__":
    main()
