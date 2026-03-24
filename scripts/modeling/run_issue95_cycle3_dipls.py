"""di-PLSドメイン適応のアンサンブル統合（高速版）

Issue #95 Cycle 3: di-PLSでLOSO-CVおよびテスト予測を行う。
XtXの漸化式更新で高速化。

パラメータグリッド:
- n_components: [3, 4, 5]
- dipls_lambda: [0.1, 1.0, 10.0, 100.0]
- 前処理: [raw, SNV, SG2d]
- 目的変数変換: [raw, sqrt]
"""
import sys
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
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


def nipals_dipls_batch_fast(X_s, y_s, X_t, max_components, lambda_list):
    """複数のlambdaに対してdi-PLSをバッチ実行（XtX漸化式更新で高速化）。

    Returns
    -------
    dict of (n_comp, lam) -> predictions
    """
    n_s, p = X_s.shape

    x_mean = X_s.mean(axis=0)
    y_mean = y_s.mean()

    # 初期XtXを1回だけ計算
    X_sc_init = X_s - x_mean
    X_tc_init = X_t - x_mean
    y_c_init = y_s - y_mean

    XtX_init = X_sc_init.T @ X_sc_init  # 最も重い計算、1回だけ
    Xty_init = X_sc_init.T @ y_c_init

    results = {}

    for lam in lambda_list:
        X_sc = X_sc_init.copy()
        X_tc = X_tc_init.copy()
        y_c = y_c_init.copy()
        XtX = XtX_init.copy()
        Xty = Xty_init.copy()

        W = np.zeros((p, max_components))
        P_load = np.zeros((p, max_components))
        Q = np.zeros(max_components)

        mean_diff = X_sc.mean(axis=0) - X_tc.mean(axis=0)
        D = np.outer(mean_diff, mean_diff)

        for a in range(max_components):
            A_mat = XtX + lam * D + 1e-8 * np.eye(p)
            w = np.linalg.solve(A_mat, Xty)
            w = w / (np.linalg.norm(w) + 1e-12)

            t = X_sc @ w
            tt = t @ t + 1e-12
            # p_load = X_sc^T @ t / (t^T t)  using XtX is wrong since Xty is not t
            # need X_sc^T @ t directly
            Xt_t = XtX @ w * (t @ t) / tt  # NOT correct
            # Actually: X_sc^T @ t = X_sc^T @ (X_sc @ w) = XtX @ w
            Xt_t_vec = XtX @ w
            p_l = Xt_t_vec / tt
            q = y_c @ t / tt

            W[:, a] = w
            P_load[:, a] = p_l
            Q[a] = q

            # Deflate X_sc: X_new = X_sc - outer(t, p_l)
            # Update XtX using recurrence:
            # X_new^T X_new = XtX - XtX @ outer(w, p_l) - outer(p_l, w) @ XtX
            #                 + (t^T t) * outer(p_l, p_l)
            # Since t = X_sc @ w, X_sc^T t = XtX @ w
            XtX_w = XtX @ w  # = X_sc^T @ t
            # X_new^T X_new = XtX - outer(XtX_w, p_l) - outer(p_l, XtX_w) + tt * outer(p_l, p_l)
            XtX = XtX - np.outer(XtX_w, p_l) - np.outer(p_l, XtX_w) + tt * np.outer(p_l, p_l)

            # Update Xty: X_new^T y_new where y_new = y_c - t*q
            # X_new^T y_new = (X_sc - outer(t,p_l))^T (y_c - t*q)
            # = X_sc^T y_c - X_sc^T t * q - p_l * (t^T y_c) + p_l * (t^T t) * q
            # = Xty - XtX_w * q - p_l * (t @ y_c) + p_l * tt * q
            # Note: t @ y_c = q * tt (from definition of q)
            Xty = Xty - XtX_w * q - p_l * (t @ y_c) + p_l * tt * q

            # Deflate X_sc and X_tc for mean_diff update
            X_sc = X_sc - np.outer(t, p_l)
            X_tc = X_tc - X_tc @ np.outer(w, p_l)
            y_c = y_c - t * q

            mean_diff = X_sc.mean(axis=0) - X_tc.mean(axis=0)
            D = np.outer(mean_diff, mean_diff)

            # Store prediction for this n_comp
            nc = a + 1
            R = W[:, :nc] @ np.linalg.inv(P_load[:, :nc].T @ W[:, :nc] + 1e-10 * np.eye(nc))
            B = R @ Q[:nc]
            X_target_c = X_t - x_mean
            pred = X_target_c @ B + y_mean
            results[(nc, lam)] = pred

    return results


def run_loso_cv_batch(df_train, spectral_cols, preproc_method, y_transform,
                      n_components_list, lambda_list):
    """1つの(前処理, y変換)に対し、全(n_comp, lambda)のLOSO-CVを実行。"""
    X_raw = df_train[spectral_cols].values
    y = df_train["含水率"].values
    groups = df_train["樹種"].values
    logo = LeaveOneGroupOut()

    max_comp = max(n_components_list)

    all_preds = {}
    fold_rmses = {}
    for nc, lam in product(n_components_list, lambda_list):
        all_preds[(nc, lam)] = np.full(len(y), np.nan)
        fold_rmses[(nc, lam)] = {}

    fold_count = 0
    for train_idx, test_idx in logo.split(X_raw, y, groups):
        fold_count += 1
        species = groups[test_idx][0]
        X_train_raw = X_raw[train_idx]
        X_test_raw = X_raw[test_idx]
        y_train = y[train_idx].copy()
        y_test = y[test_idx]
        groups_train = groups[train_idx]

        X_tr_pp, X_te_pp = preprocess_pair(X_train_raw, X_test_raw,
                                            groups_train, preproc_method)
        y_fit = np.sqrt(y_train) if y_transform == "sqrt" else y_train

        # バッチdi-PLS（高速版）
        batch_preds = nipals_dipls_batch_fast(X_tr_pp, y_fit, X_te_pp,
                                              max_comp, lambda_list)

        for nc in n_components_list:
            for lam in lambda_list:
                pred = batch_preds.get((nc, lam))
                if pred is None:
                    continue
                if y_transform == "sqrt":
                    pred = np.clip(pred, 0, None) ** 2
                all_preds[(nc, lam)][test_idx] = pred
                fold_rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
                fold_rmses[(nc, lam)][species] = fold_rmse

        sys.stdout.write(f"    fold {fold_count}/13 ({species}) done\n")
        sys.stdout.flush()

    output = []
    for (nc, lam) in product(n_components_list, lambda_list):
        preds = all_preds[(nc, lam)]
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
        for species, fold_rmse in fold_rmses[(nc, lam)].items():
            row[f"rmse_{species}"] = fold_rmse
        output.append(row)

    return output


def main():
    sys.stdout.write("=" * 70 + "\n")
    sys.stdout.write("Issue #95 Cycle 3: di-PLS Domain Adaptation (Fast)\n")
    sys.stdout.write("=" * 70 + "\n")
    sys.stdout.flush()

    df_train = load_train(DATA_DIR)
    df_test = load_test(DATA_DIR)
    spectral_cols = get_spectral_columns(df_train)

    sys.stdout.write(f"Train: {len(df_train)} samples, Test: {len(df_test)} samples\n")
    sys.stdout.write(f"Spectral features: {len(spectral_cols)}\n")
    sys.stdout.write(f"Species (train): {sorted(df_train['樹種'].unique())}\n\n")
    sys.stdout.flush()

    n_components_list = [3, 4, 5]
    lambda_list = [0.1, 1.0, 10.0, 100.0]
    preproc_list = ["raw", "SNV", "SG2d"]
    y_transform_list = ["raw", "sqrt"]

    total_combos = len(n_components_list) * len(lambda_list) * len(preproc_list) * len(y_transform_list)
    total_batches = len(preproc_list) * len(y_transform_list)

    sys.stdout.write(f"Total grid combinations: {total_combos}\n")
    sys.stdout.write(f"Batch groups: {total_batches}\n")
    sys.stdout.write(f"Optimization: XtX computed once per fold, "
                     f"recurrence update for deflation\n\n")
    sys.stdout.flush()

    all_results = []
    start_time = time.time()
    batch_count = 0

    for preproc, y_tf in product(preproc_list, y_transform_list):
        batch_count += 1
        batch_start = time.time()
        sys.stdout.write(f"[Batch {batch_count}/{total_batches}] "
                        f"preproc={preproc}, y_tf={y_tf}\n")
        sys.stdout.flush()

        batch_results = run_loso_cv_batch(
            df_train, spectral_cols, preproc, y_tf,
            n_components_list, lambda_list
        )
        all_results.extend(batch_results)

        batch_elapsed = time.time() - batch_start
        total_elapsed = time.time() - start_time

        if batch_results:
            best_in_batch = min(batch_results, key=lambda x: x["rmse"])
            sys.stdout.write(f"  -> best RMSE={best_in_batch['rmse']:.4f} "
                           f"(n_comp={best_in_batch['n_components']}, "
                           f"lambda={best_in_batch['dipls_lambda']}) "
                           f"[{batch_elapsed:.0f}s / total {total_elapsed:.0f}s]\n\n")
        sys.stdout.flush()

    df_results = pd.DataFrame(all_results).sort_values("rmse").reset_index(drop=True)

    # 結果表示
    sys.stdout.write("\n" + "=" * 70 + "\n")
    sys.stdout.write("LOSO-CV Results (Top 20)\n")
    sys.stdout.write("=" * 70 + "\n")
    for i, row in df_results.head(20).iterrows():
        sys.stdout.write(f"  #{i+1}: RMSE={row['rmse']:.4f} | preproc={row['preproc']}, "
                        f"y_tf={row['y_transform']}, n_comp={int(row['n_components'])}, "
                        f"lambda={row['dipls_lambda']}\n")

    sys.stdout.write("\n" + "=" * 70 + "\n")
    sys.stdout.write("Best RMSE by preprocessing\n")
    sys.stdout.write("=" * 70 + "\n")
    for preproc in preproc_list:
        subset = df_results[df_results["preproc"] == preproc]
        if len(subset) > 0:
            best_pp = subset.iloc[0]
            sys.stdout.write(f"  {preproc}: RMSE={best_pp['rmse']:.4f} "
                           f"(n_comp={int(best_pp['n_components'])}, "
                           f"lambda={best_pp['dipls_lambda']}, "
                           f"y_tf={best_pp['y_transform']})\n")

    best = df_results.iloc[0]
    sys.stdout.write("\n" + "=" * 70 + "\n")
    sys.stdout.write(f"Best Setting: preproc={best['preproc']}, y_tf={best['y_transform']}, "
                    f"n_comp={int(best['n_components'])}, lambda={best['dipls_lambda']}\n")
    sys.stdout.write(f"Overall RMSE: {best['rmse']:.4f}\n")
    sys.stdout.write("=" * 70 + "\n")
    fold_cols = sorted([c for c in df_results.columns if c.startswith("rmse_")])
    for col in fold_cols:
        species = col.replace("rmse_", "")
        sys.stdout.write(f"  {species}: RMSE={best[col]:.4f}\n")

    results_path = OUTPUT_DIR / "issue95_dipls_grid_results.csv"
    df_results.to_csv(results_path, index=False)
    sys.stdout.write(f"\nGrid results saved to: {results_path}\n")
    sys.stdout.flush()

    # テスト予測
    sys.stdout.write("\n" + "=" * 70 + "\n")
    sys.stdout.write("Test Prediction (Best Setting)\n")
    sys.stdout.write("=" * 70 + "\n")
    sys.stdout.flush()

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
    y_fit = np.sqrt(y_train_all) if best_y_tf == "sqrt" else y_train_all

    test_pred = fit_predict_dipls(X_tr_pp, y_fit, X_te_pp,
                                  n_components=best_n_comp, dipls_lambda=best_lambda)
    if best_y_tf == "sqrt":
        test_pred = np.clip(test_pred, 0, None) ** 2

    sys.stdout.write(f"Test predictions: mean={test_pred.mean():.2f}, "
                    f"std={test_pred.std():.2f}, "
                    f"min={test_pred.min():.2f}, max={test_pred.max():.2f}\n")

    sample_submit = pd.read_csv(DATA_DIR / "sample_submit.csv", header=None)
    submit = sample_submit.copy()
    submit.iloc[:, 1] = test_pred
    submit_path = OUTPUT_DIR / "submission_v6_dipls.csv"
    submit.to_csv(submit_path, index=False, header=False)
    sys.stdout.write(f"Submission saved to: {submit_path}\n")
    sys.stdout.flush()

    # アンサンブル
    sys.stdout.write("\n" + "=" * 70 + "\n")
    sys.stdout.write("Ensemble: di-PLS Top-3 + Existing PLS\n")
    sys.stdout.write("=" * 70 + "\n")
    sys.stdout.flush()

    top3_preds = []
    for i in range(min(3, len(df_results))):
        row = df_results.iloc[i]
        pp = row["preproc"]
        ytf = row["y_transform"]
        nc = int(row["n_components"])
        lam = row["dipls_lambda"]

        xtr, xte = preprocess_pair(X_train_all, X_test_all,
                                    groups_train_all, pp)
        yf = np.sqrt(y_train_all) if ytf == "sqrt" else y_train_all
        p = fit_predict_dipls(xtr, yf, xte, n_components=nc, dipls_lambda=lam)
        if ytf == "sqrt":
            p = np.clip(p, 0, None) ** 2
        top3_preds.append(p)
        sys.stdout.write(f"  Top-{i+1}: preproc={pp}, y_tf={ytf}, n_comp={nc}, "
                        f"lambda={lam}, RMSE={row['rmse']:.4f}\n")

    existing_pls_path = OUTPUT_DIR / "submission_v5.csv"
    ensemble_preds = top3_preds.copy()
    if existing_pls_path.exists():
        df_v5 = pd.read_csv(existing_pls_path, header=None)
        pls_pred = df_v5.iloc[:, 1].values
        ensemble_preds.append(pls_pred)
        sys.stdout.write(f"  Existing PLS (v5): loaded {len(pls_pred)} predictions\n")

    ensemble_pred = np.mean(ensemble_preds, axis=0)
    sys.stdout.write(f"\nEnsemble ({len(ensemble_preds)} models): "
                    f"mean={ensemble_pred.mean():.2f}, "
                    f"std={ensemble_pred.std():.2f}\n")

    submit_ens = sample_submit.copy()
    submit_ens.iloc[:, 1] = ensemble_pred
    ens_path = OUTPUT_DIR / "submission_v6_dipls_ensemble.csv"
    submit_ens.to_csv(ens_path, index=False, header=False)
    sys.stdout.write(f"Ensemble submission saved to: {ens_path}\n")
    sys.stdout.flush()

    # LOSO-CVアンサンブル評価
    sys.stdout.write("\n" + "=" * 70 + "\n")
    sys.stdout.write("LOSO-CV Ensemble Evaluation\n")
    sys.stdout.write("=" * 70 + "\n")
    sys.stdout.flush()

    X_raw = df_train[spectral_cols].values
    y = df_train["含水率"].values
    groups = df_train["樹種"].values
    logo = LeaveOneGroupOut()

    n_top = min(3, len(df_results))
    top_cv_preds = [np.full(len(y), np.nan) for _ in range(n_top)]

    # アンサンブル用にもバッチ実行
    for train_idx, test_idx in logo.split(X_raw, y, groups):
        species = groups[test_idx][0]
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
            yf = np.sqrt(y_train) if ytf == "sqrt" else y_train
            p = fit_predict_dipls(xtr, yf, xte, n_components=nc, dipls_lambda=lam)
            if ytf == "sqrt":
                p = np.clip(p, 0, None) ** 2
            top_cv_preds[model_idx][test_idx] = p

        sys.stdout.write(f"  Ensemble fold ({species}) done\n")
        sys.stdout.flush()

    ens_cv_pred = np.nanmean(top_cv_preds, axis=0)
    ens_cv_rmse = float(np.sqrt(np.nanmean((ens_cv_pred - y) ** 2)))
    sys.stdout.write(f"\ndi-PLS Top-3 Ensemble LOSO-CV RMSE: {ens_cv_rmse:.4f}\n")

    for train_idx, test_idx in logo.split(X_raw, y, groups):
        species = groups[test_idx][0]
        fold_rmse = float(np.sqrt(np.mean((ens_cv_pred[test_idx] - y[test_idx]) ** 2)))
        sys.stdout.write(f"  {species}: RMSE={fold_rmse:.4f}\n")

    sys.stdout.write("\nDone!\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
