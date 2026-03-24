"""di-PLSドメイン適応のアンサンブル統合（PCA次元削減+高速版）

Issue #95 Cycle 3: di-PLSでLOSO-CVおよびテスト予測を行う。

パラメータグリッド:
- n_components: [2, 3, 4, 5, 6]
- dipls_lambda: [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
- 前処理: [raw, SNV, EPO(1), SG2d]
- 目的変数変換: [raw, sqrt]

高速化: PCA(n=50)で1555次元→50次元に削減してからdi-PLSを適用
"""
import sys
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneGroupOut

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eda.data_loader import load_train, load_test, get_spectral_columns
from src.preprocessing.issue43_dipls import fit_predict_dipls
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue20_savgol import apply_savgol


DATA_DIR = PROJECT_ROOT / "Input_data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

PCA_N_COMPONENTS = 50  # di-PLS入力の次元数


def preprocess_pair(X_train_raw, X_test_raw, groups_train, method):
    """前処理を適用する。"""
    if method == "raw":
        return X_train_raw.copy(), X_test_raw.copy()
    elif method == "SNV":
        return apply_snv(X_train_raw), apply_snv(X_test_raw)
    elif method == "EPO1":
        P = compute_epo_projection(X_train_raw, groups_train, n_components=1)
        return apply_epo(X_train_raw, P), apply_epo(X_test_raw, P)
    elif method == "SG2d":
        return (apply_savgol(X_train_raw, deriv=2, window_length=11, polyorder=2),
                apply_savgol(X_test_raw, deriv=2, window_length=11, polyorder=2))
    else:
        raise ValueError(f"Unknown method: {method}")


def run_loso_cv_batch(df_train, spectral_cols, preproc_method, y_transform,
                      n_components_list, lambda_list):
    """1つの(前処理, y変換)に対し、全(n_comp, lambda)のLOSO-CVを実行。"""
    X_raw = df_train[spectral_cols].values
    y = df_train["含水率"].values
    groups = df_train["樹種"].values
    logo = LeaveOneGroupOut()

    # 結果格納
    results = {}
    for nc, lam in product(n_components_list, lambda_list):
        results[(nc, lam)] = {"preds": np.full(len(y), np.nan), "fold_rmses": {}}

    # fold別に処理
    for train_idx, test_idx in logo.split(X_raw, y, groups):
        species = groups[test_idx][0]
        X_train_raw = X_raw[train_idx]
        X_test_raw = X_raw[test_idx]
        y_train = y[train_idx].copy()
        y_test = y[test_idx]
        groups_train = groups[train_idx]

        # 前処理（foldごとに1回だけ）
        X_tr_pp, X_te_pp = preprocess_pair(X_train_raw, X_test_raw,
                                            groups_train, preproc_method)

        # PCA次元削減（trainでfit, testはtransform）
        pca = PCA(n_components=PCA_N_COMPONENTS)
        X_tr_pca = pca.fit_transform(X_tr_pp)
        X_te_pca = pca.transform(X_te_pp)

        # 目的変数変換
        if y_transform == "sqrt":
            y_fit = np.sqrt(y_train)
        else:
            y_fit = y_train

        # 全(n_comp, lambda)の組合せを実行
        for nc in n_components_list:
            for lam in lambda_list:
                try:
                    pred = fit_predict_dipls(
                        X_tr_pca, y_fit, X_te_pca,
                        n_components=nc, dipls_lambda=lam
                    )
                except Exception:
                    pred = np.full(len(test_idx), np.nan)

                # 逆変換
                if y_transform == "sqrt":
                    pred = np.clip(pred, 0, None) ** 2

                results[(nc, lam)]["preds"][test_idx] = pred
                fold_rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
                results[(nc, lam)]["fold_rmses"][species] = fold_rmse

    # 全体RMSE計算
    output = []
    for (nc, lam), data in results.items():
        preds = data["preds"]
        if np.any(np.isnan(preds)):
            continue
        overall_rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
        row = {
            "preproc": preproc_method,
            "y_transform": y_transform,
            "n_components": nc,
            "dipls_lambda": lam,
            "rmse": overall_rmse,
        }
        for species, fold_rmse in data["fold_rmses"].items():
            row[f"rmse_{species}"] = fold_rmse
        output.append(row)

    return output


def run_loso_cv_batch_nopca(df_train, spectral_cols, preproc_method, y_transform,
                            n_components_list, lambda_list):
    """PCAなしバージョン（参考用、1設定のみ）。"""
    X_raw = df_train[spectral_cols].values
    y = df_train["含水率"].values
    groups = df_train["樹種"].values
    logo = LeaveOneGroupOut()

    results = {}
    for nc, lam in product(n_components_list, lambda_list):
        results[(nc, lam)] = {"preds": np.full(len(y), np.nan), "fold_rmses": {}}

    for train_idx, test_idx in logo.split(X_raw, y, groups):
        species = groups[test_idx][0]
        X_train_raw = X_raw[train_idx]
        X_test_raw = X_raw[test_idx]
        y_train = y[train_idx].copy()
        y_test = y[test_idx]
        groups_train = groups[train_idx]

        X_tr_pp, X_te_pp = preprocess_pair(X_train_raw, X_test_raw,
                                            groups_train, preproc_method)
        if y_transform == "sqrt":
            y_fit = np.sqrt(y_train)
        else:
            y_fit = y_train

        for nc in n_components_list:
            for lam in lambda_list:
                try:
                    pred = fit_predict_dipls(
                        X_tr_pp, y_fit, X_te_pp,
                        n_components=nc, dipls_lambda=lam
                    )
                except Exception:
                    pred = np.full(len(test_idx), np.nan)
                if y_transform == "sqrt":
                    pred = np.clip(pred, 0, None) ** 2
                results[(nc, lam)]["preds"][test_idx] = pred
                fold_rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
                results[(nc, lam)]["fold_rmses"][species] = fold_rmse

    output = []
    for (nc, lam), data in results.items():
        preds = data["preds"]
        if np.any(np.isnan(preds)):
            continue
        overall_rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
        row = {
            "preproc": preproc_method,
            "y_transform": y_transform,
            "n_components": nc,
            "dipls_lambda": lam,
            "rmse": overall_rmse,
        }
        for species, fold_rmse in data["fold_rmses"].items():
            row[f"rmse_{species}"] = fold_rmse
        output.append(row)
    return output


def main():
    print("=" * 70)
    print("Issue #95 Cycle 3: di-PLS Domain Adaptation")
    print("=" * 70)

    df_train = load_train(DATA_DIR)
    df_test = load_test(DATA_DIR)
    spectral_cols = get_spectral_columns(df_train)
    print(f"Train: {len(df_train)} samples, Test: {len(df_test)} samples")
    print(f"Spectral features: {len(spectral_cols)}")
    print(f"PCA reduction: {len(spectral_cols)} -> {PCA_N_COMPONENTS}")
    print(f"Species (train): {sorted(df_train['樹種'].unique())}")
    print()

    n_components_list = [2, 3, 4, 5, 6]
    lambda_list = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    preproc_list = ["raw", "SNV", "EPO1", "SG2d"]
    y_transform_list = ["raw", "sqrt"]

    total_combos = len(n_components_list) * len(lambda_list) * len(preproc_list) * len(y_transform_list)
    print(f"Total grid combinations: {total_combos}")
    print()

    all_results = []
    start_time = time.time()
    batch_count = 0
    total_batches = len(preproc_list) * len(y_transform_list)

    for preproc, y_tf in product(preproc_list, y_transform_list):
        batch_count += 1
        batch_start = time.time()
        print(f"[Batch {batch_count}/{total_batches}] preproc={preproc}, y_tf={y_tf} ...")

        batch_results = run_loso_cv_batch(
            df_train, spectral_cols, preproc, y_tf,
            n_components_list, lambda_list
        )
        all_results.extend(batch_results)

        batch_elapsed = time.time() - batch_start
        total_elapsed = time.time() - start_time

        if batch_results:
            best_in_batch = min(batch_results, key=lambda x: x["rmse"])
            print(f"  -> {len(batch_results)} results, best RMSE={best_in_batch['rmse']:.4f} "
                  f"(n_comp={best_in_batch['n_components']}, lambda={best_in_batch['dipls_lambda']}) "
                  f"[{batch_elapsed:.0f}s / total {total_elapsed:.0f}s]")
        else:
            print(f"  -> No valid results [{batch_elapsed:.0f}s]")

    df_results = pd.DataFrame(all_results).sort_values("rmse").reset_index(drop=True)

    # 結果表示
    print()
    print("=" * 70)
    print("LOSO-CV Results (Top 30)")
    print("=" * 70)
    top30 = df_results.head(30)
    for i, row in top30.iterrows():
        print(f"  #{i+1}: RMSE={row['rmse']:.4f} | preproc={row['preproc']}, "
              f"y_tf={row['y_transform']}, n_comp={int(row['n_components'])}, "
              f"lambda={row['dipls_lambda']}")

    # 前処理別ベスト
    print()
    print("=" * 70)
    print("Best RMSE by preprocessing")
    print("=" * 70)
    for preproc in preproc_list:
        subset = df_results[df_results["preproc"] == preproc]
        if len(subset) > 0:
            best_pp = subset.iloc[0]
            print(f"  {preproc}: RMSE={best_pp['rmse']:.4f} (n_comp={int(best_pp['n_components'])}, "
                  f"lambda={best_pp['dipls_lambda']}, y_tf={best_pp['y_transform']})")

    # ベスト設定のfold別RMSE
    best = df_results.iloc[0]
    print()
    print("=" * 70)
    print(f"Best Setting: preproc={best['preproc']}, y_tf={best['y_transform']}, "
          f"n_comp={int(best['n_components'])}, lambda={best['dipls_lambda']}")
    print(f"Overall RMSE: {best['rmse']:.4f}")
    print("=" * 70)
    fold_cols = [c for c in df_results.columns if c.startswith("rmse_")]
    for col in sorted(fold_cols):
        species = col.replace("rmse_", "")
        print(f"  {species}: RMSE={best[col]:.4f}")

    # PCAなしで最良設定を再評価（参考）
    print()
    print("=" * 70)
    print("Verification: Best setting WITHOUT PCA reduction (raw 1555-dim)")
    print("=" * 70)
    nopca_results = run_loso_cv_batch_nopca(
        df_train, spectral_cols, best["preproc"], best["y_transform"],
        [int(best["n_components"])], [best["dipls_lambda"]]
    )
    if nopca_results:
        nopca_best = nopca_results[0]
        print(f"  RMSE (no PCA): {nopca_best['rmse']:.4f}")
        for k, v in nopca_best.items():
            if k.startswith("rmse_"):
                print(f"    {k.replace('rmse_', '')}: {v:.4f}")

    results_path = OUTPUT_DIR / "issue95_dipls_grid_results.csv"
    df_results.to_csv(results_path, index=False)
    print(f"\nGrid results saved to: {results_path}")

    # テスト予測（ベスト設定 - PCA付き）
    print()
    print("=" * 70)
    print("Test Prediction (Best Setting with PCA)")
    print("=" * 70)

    X_train_all = df_train[spectral_cols].values
    X_test_all = df_test[spectral_cols].values
    y_train_all = df_train["含水率"].values
    groups_train_all = df_train["樹種"].values

    best_preproc = best["preproc"]
    best_y_tf = best["y_transform"]
    best_n_comp = int(best["n_components"])
    best_lambda = best["dipls_lambda"]

    X_tr_pp, X_te_pp = preprocess_pair(X_train_all, X_test_all,
                                        groups_train_all, best_preproc)
    pca = PCA(n_components=PCA_N_COMPONENTS)
    X_tr_pca = pca.fit_transform(X_tr_pp)
    X_te_pca = pca.transform(X_te_pp)

    if best_y_tf == "sqrt":
        y_fit = np.sqrt(y_train_all)
    else:
        y_fit = y_train_all

    test_pred = fit_predict_dipls(
        X_tr_pca, y_fit, X_te_pca,
        n_components=best_n_comp, dipls_lambda=best_lambda
    )
    if best_y_tf == "sqrt":
        test_pred = np.clip(test_pred, 0, None) ** 2

    print(f"Test predictions: mean={test_pred.mean():.2f}, "
          f"std={test_pred.std():.2f}, "
          f"min={test_pred.min():.2f}, max={test_pred.max():.2f}")

    sample_submit = pd.read_csv(DATA_DIR / "sample_submit.csv", header=None)
    submit = sample_submit.copy()
    submit.iloc[:, 1] = test_pred
    submit_path = OUTPUT_DIR / "submission_v6_dipls.csv"
    submit.to_csv(submit_path, index=False, header=False)
    print(f"Submission saved to: {submit_path}")

    # テスト予測（PCAなし）
    print()
    print("Test Prediction (Best Setting WITHOUT PCA)")
    test_pred_nopca = fit_predict_dipls(
        X_tr_pp, y_fit, X_te_pp,
        n_components=best_n_comp, dipls_lambda=best_lambda
    )
    if best_y_tf == "sqrt":
        test_pred_nopca = np.clip(test_pred_nopca, 0, None) ** 2

    print(f"Test predictions (no PCA): mean={test_pred_nopca.mean():.2f}, "
          f"std={test_pred_nopca.std():.2f}, "
          f"min={test_pred_nopca.min():.2f}, max={test_pred_nopca.max():.2f}")

    # アンサンブル
    print()
    print("=" * 70)
    print("Ensemble: di-PLS Top-3 + Existing PLS")
    print("=" * 70)

    top3_preds = []
    for i in range(min(3, len(df_results))):
        row = df_results.iloc[i]
        pp = row["preproc"]
        ytf = row["y_transform"]
        nc = int(row["n_components"])
        lam = row["dipls_lambda"]

        xtr, xte = preprocess_pair(X_train_all, X_test_all,
                                    groups_train_all, pp)
        pca_i = PCA(n_components=PCA_N_COMPONENTS)
        xtr_pca = pca_i.fit_transform(xtr)
        xte_pca = pca_i.transform(xte)

        yf = np.sqrt(y_train_all) if ytf == "sqrt" else y_train_all
        p = fit_predict_dipls(xtr_pca, yf, xte_pca, n_components=nc, dipls_lambda=lam)
        if ytf == "sqrt":
            p = np.clip(p, 0, None) ** 2
        top3_preds.append(p)
        print(f"  Top-{i+1}: preproc={pp}, y_tf={ytf}, n_comp={nc}, "
              f"lambda={lam}, RMSE={row['rmse']:.4f}")

    existing_pls_path = OUTPUT_DIR / "submission_v5.csv"
    ensemble_preds = top3_preds.copy()

    if existing_pls_path.exists():
        df_v5 = pd.read_csv(existing_pls_path, header=None)
        pls_pred = df_v5.iloc[:, 1].values
        ensemble_preds.append(pls_pred)
        print(f"  Existing PLS (v5): loaded {len(pls_pred)} predictions")

    ensemble_pred = np.mean(ensemble_preds, axis=0)
    print(f"\nEnsemble ({len(ensemble_preds)} models): "
          f"mean={ensemble_pred.mean():.2f}, "
          f"std={ensemble_pred.std():.2f}")

    submit_ens = sample_submit.copy()
    submit_ens.iloc[:, 1] = ensemble_pred
    ens_path = OUTPUT_DIR / "submission_v6_dipls_ensemble.csv"
    submit_ens.to_csv(ens_path, index=False, header=False)
    print(f"Ensemble submission saved to: {ens_path}")

    # LOSO-CVでのアンサンブル評価
    print()
    print("=" * 70)
    print("LOSO-CV Ensemble Evaluation")
    print("=" * 70)

    X_raw = df_train[spectral_cols].values
    y = df_train["含水率"].values
    groups = df_train["樹種"].values
    logo = LeaveOneGroupOut()

    n_top = min(3, len(df_results))
    top_cv_preds = [np.full(len(y), np.nan) for _ in range(n_top)]

    for train_idx, test_idx in logo.split(X_raw, y, groups):
        X_train_raw = X_raw[train_idx]
        X_test_raw = X_raw[test_idx]
        y_train = y[train_idx]
        groups_train = groups[train_idx]

        for model_idx in range(n_top):
            row = df_results.iloc[model_idx]
            pp = row["preproc"]
            ytf = row["y_transform"]
            nc = int(row["n_components"])
            lam = row["dipls_lambda"]

            xtr, xte = preprocess_pair(X_train_raw, X_test_raw,
                                        groups_train, pp)
            pca_m = PCA(n_components=PCA_N_COMPONENTS)
            xtr_pca = pca_m.fit_transform(xtr)
            xte_pca = pca_m.transform(xte)

            yf = np.sqrt(y_train) if ytf == "sqrt" else y_train
            p = fit_predict_dipls(xtr_pca, yf, xte_pca, n_components=nc, dipls_lambda=lam)
            if ytf == "sqrt":
                p = np.clip(p, 0, None) ** 2
            top_cv_preds[model_idx][test_idx] = p

    ens_cv_pred = np.nanmean(top_cv_preds, axis=0)
    ens_cv_rmse = float(np.sqrt(np.nanmean((ens_cv_pred - y) ** 2)))
    print(f"di-PLS Top-3 Ensemble LOSO-CV RMSE: {ens_cv_rmse:.4f}")

    for train_idx, test_idx in logo.split(X_raw, y, groups):
        species = groups[test_idx][0]
        fold_rmse = float(np.sqrt(np.mean((ens_cv_pred[test_idx] - y[test_idx]) ** 2)))
        print(f"  {species}: RMSE={fold_rmse:.4f}")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
