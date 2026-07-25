"""Issue #71: v4アンサンブル — サンプル重み付けモデル追加 + 拡張探索

v3(15.99)のモデルに加え、サンプル重み付きモデルを候補に追加。
fold-level予測を収集し、最適サブセット・重みを探索。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import time
from itertools import combinations
from scipy.optimize import minimize

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.model_selection import LeaveOneGroupOut, KFold

from src.eda.data_loader import load_train, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue20_savgol import apply_savgol
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue19_msc import compute_msc_reference, apply_msc
from src.preprocessing.issue62_additional_preprocessing import (
    apply_asls, apply_piecewise_msc,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_sample_weights(groups_train, strategy="log_inv"):
    unique, counts = np.unique(groups_train, return_counts=True)
    count_map = dict(zip(unique, counts))
    max_count = counts.max()
    weights = np.zeros(len(groups_train))
    for i, sp in enumerate(groups_train):
        c = count_map[sp]
        if strategy == "log_inv":
            weights[i] = np.log(max_count / c + 1)
        elif strategy == "inv_freq":
            weights[i] = len(groups_train) / (len(unique) * c)
        elif strategy == "sqrt_inv":
            weights[i] = np.sqrt(max_count / c)
        else:
            weights[i] = 1.0
    return weights


def vip_select(X_train, y_train, X_test, n_components=2, threshold=1.5):
    nc = min(n_components, X_train.shape[1] - 1)
    pls = PLSRegression(n_components=nc)
    pls.fit(X_train, y_train)
    T, W, Q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
    p = X_train.shape[1]
    s = np.diag(T.T @ T @ Q.T @ Q).ravel()
    total_s = np.sum(s)
    vip = np.sqrt(p * np.sum(s[None, :] * (W / np.linalg.norm(W, axis=0)) ** 2, axis=1) / total_s)
    mask = vip > threshold
    if mask.sum() < 2:
        mask = np.zeros(p, dtype=bool)
        mask[np.argsort(vip)[-max(2, int(p * 0.1)):]] = True
    return X_train[:, mask], X_test[:, mask]


def predict_fold(X_tr_raw, X_te_raw, y_train, groups_train, config):
    pp = config["pp"]
    # Preprocessing
    if pp == "SNV":
        X_tr, X_te = apply_snv(X_tr_raw), apply_snv(X_te_raw)
    elif pp == "EPO(1)":
        P = compute_epo_projection(X_tr_raw, groups_train, n_components=1)
        X_tr, X_te = apply_epo(X_tr_raw, P), apply_epo(X_te_raw, P)
    elif pp == "SNV+AsLS(1e6)":
        X_tr = apply_asls(apply_snv(X_tr_raw), lam=1e6)
        X_te = apply_asls(apply_snv(X_te_raw), lam=1e6)
    elif pp == "PiecewiseMSC(seg=3)":
        ref = compute_msc_reference(X_tr_raw)
        X_tr = apply_piecewise_msc(X_tr_raw, ref, 3)
        X_te = apply_piecewise_msc(X_te_raw, ref, 3)
    elif pp == "MSC":
        ref = compute_msc_reference(X_tr_raw)
        X_tr, X_te = apply_msc(X_tr_raw, ref), apply_msc(X_te_raw, ref)
    elif pp == "SG2d+EPO(1)":
        X_tr_sg = apply_savgol(X_tr_raw, deriv=2, window_length=7)
        X_te_sg = apply_savgol(X_te_raw, deriv=2, window_length=7)
        P = compute_epo_projection(X_tr_sg, groups_train, n_components=1)
        X_tr, X_te = apply_epo(X_tr_sg, P), apply_epo(X_te_sg, P)
    elif pp == "SNV+EPO(1)":
        X_tr_snv, X_te_snv = apply_snv(X_tr_raw), apply_snv(X_te_raw)
        P = compute_epo_projection(X_tr_snv, groups_train, n_components=1)
        X_tr, X_te = apply_epo(X_tr_snv, P), apply_epo(X_te_snv, P)
    else:
        X_tr, X_te = X_tr_raw.copy(), X_te_raw.copy()

    # Feature selection
    fs = config.get("fs")
    if fs == "VIP(1.5)":
        X_tr, X_te = vip_select(X_tr, y_train, X_te, threshold=1.5)

    # Target transform
    tf = config["tf"]
    y_fit = np.sqrt(y_train) if tf == "sqrt" else y_train.copy()

    # Sample weights
    sw = config.get("sw")
    weights = compute_sample_weights(groups_train, sw) if sw else None

    # Model
    mdl = config["model"]
    if mdl.startswith("PLS("):
        nc = int(mdl.split("(")[1].rstrip(")"))
        nc = min(nc, X_tr.shape[1] - 1)
        m = PLSRegression(n_components=max(1, nc))
        if weights is not None:
            sqrt_w = np.sqrt(weights)
            X_tr_w = X_tr * sqrt_w[:, None]
            y_fit_w = y_fit * sqrt_w
            m.fit(X_tr_w, y_fit_w)
        else:
            m.fit(X_tr, y_fit)
        pred = m.predict(X_te).ravel()
    elif mdl == "Lasso":
        m = Lasso(alpha=0.1, max_iter=10000)
        if weights is not None:
            m.fit(X_tr, y_fit, sample_weight=weights)
        else:
            m.fit(X_tr, y_fit)
        pred = m.predict(X_te)
    elif mdl == "ElasticNet":
        m = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
        if weights is not None:
            m.fit(X_tr, y_fit, sample_weight=weights)
        else:
            m.fit(X_tr, y_fit)
        pred = m.predict(X_te)
    elif mdl == "Ridge":
        m = Ridge(alpha=100)
        if weights is not None:
            m.fit(X_tr, y_fit, sample_weight=weights)
        else:
            m.fit(X_tr, y_fit)
        pred = m.predict(X_te)
    else:
        raise ValueError(mdl)

    if tf == "sqrt":
        pred = np.clip(pred, 0, None) ** 2
    return pred


def optimize_weights_fold(all_preds, y, folds, model_indices, n_restarts=30):
    n = len(model_indices)
    def obj(w):
        w_n = np.abs(w) / np.sum(np.abs(w))
        fold_rmses = []
        for f_idx, (_, test_idx) in enumerate(folds):
            preds = [all_preds[m_idx][f_idx] for m_idx in model_indices]
            ens = sum(w_i * p for w_i, p in zip(w_n, preds))
            ens = np.clip(ens, 0, 200)
            fold_rmses.append(rmse(y[test_idx], ens))
        return np.mean(fold_rmses)
    best_r, best_w = np.inf, np.ones(n) / n
    for s in range(n_restarts):
        w0 = np.random.RandomState(s).dirichlet(np.ones(n))
        res = minimize(obj, w0, method="Nelder-Mead", options={"maxiter": 3000})
        if res.fun < best_r:
            best_r = res.fun
            best_w = np.abs(res.x) / np.sum(np.abs(res.x))
    return best_w, best_r


def eval_ensemble(all_preds, y, folds, model_indices, weights=None):
    fold_rmses = []
    for f_idx, (_, test_idx) in enumerate(folds):
        preds = [all_preds[m_idx][f_idx] for m_idx in model_indices]
        if weights is not None:
            ens = sum(w * p for w, p in zip(weights, preds))
        else:
            ens = np.mean(preds, axis=0)
        ens = np.clip(ens, 0, 200)
        fold_rmses.append(rmse(y[test_idx], ens))
    return np.mean(fold_rmses), np.std(fold_rmses), fold_rmses


def main():
    t0 = time.time()
    df = load_train(DATA_DIR)
    sc = get_spectral_columns(df)
    X_raw = df[sc].values
    y = df["含水率"].values
    groups = df["樹種"].values

    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_raw, y, groups))
    species_names = [np.unique(groups[te])[0] for _, te in folds]

    print("=" * 70)
    print("Issue #71: v4アンサンブル（サンプル重み付け追加）")
    print("  ベースライン v3: 15.99")
    print("=" * 70)

    # v3の5モデル + 重み付きモデル + 追加候補
    configs = [
        # v3 models
        {"name": "EPO(1)+PLS(4)+sqrt", "pp": "EPO(1)", "model": "PLS(4)", "tf": "sqrt"},
        {"name": "SG2d+EPO(1)+PLS(3)+raw", "pp": "SG2d+EPO(1)", "model": "PLS(3)", "tf": "raw"},
        {"name": "PiecewiseMSC+Lasso+raw", "pp": "PiecewiseMSC(seg=3)", "model": "Lasso", "tf": "raw"},
        {"name": "SNV+VIP(1.5)+PLS(4)+sqrt", "pp": "SNV", "model": "PLS(4)", "tf": "sqrt", "fs": "VIP(1.5)"},
        {"name": "SNV+AsLS(1e6)+PLS(2)+sqrt", "pp": "SNV+AsLS(1e6)", "model": "PLS(2)", "tf": "sqrt"},
        # Sample-weighted models (NEW)
        {"name": "SNV+EPO(1)+PLS(4)+raw+logW", "pp": "SNV+EPO(1)", "model": "PLS(4)", "tf": "raw", "sw": "log_inv"},
        {"name": "EPO(1)+PLS(4)+raw+logW", "pp": "EPO(1)", "model": "PLS(4)", "tf": "raw", "sw": "log_inv"},
        {"name": "EPO(1)+PLS(4)+sqrt+logW", "pp": "EPO(1)", "model": "PLS(4)", "tf": "sqrt", "sw": "log_inv"},
        {"name": "SNV+PLS(4)+raw+invW", "pp": "SNV", "model": "PLS(4)", "tf": "raw", "sw": "inv_freq"},
        {"name": "PiecewiseMSC+PLS(4)+raw+logW", "pp": "PiecewiseMSC(seg=3)", "model": "PLS(4)", "tf": "raw", "sw": "log_inv"},
        # Additional diverse models
        {"name": "SNV+PLS(2)+raw", "pp": "SNV", "model": "PLS(2)", "tf": "raw"},
        {"name": "SNV+PLS(2)+sqrt", "pp": "SNV", "model": "PLS(2)", "tf": "sqrt"},
        {"name": "SNV+ElasticNet+sqrt", "pp": "SNV", "model": "ElasticNet", "tf": "sqrt"},
        {"name": "EPO(1)+Lasso+raw", "pp": "EPO(1)", "model": "Lasso", "tf": "raw"},
        {"name": "MSC+PLS(4)+sqrt", "pp": "MSC", "model": "PLS(4)", "tf": "sqrt"},
        {"name": "EPO(1)+PLS(2)+raw", "pp": "EPO(1)", "model": "PLS(2)", "tf": "raw"},
        # More weighted variants
        {"name": "MSC+PLS(4)+raw+logW", "pp": "MSC", "model": "PLS(4)", "tf": "raw", "sw": "log_inv"},
        {"name": "SNV+Lasso+raw+logW", "pp": "SNV", "model": "Lasso", "tf": "raw", "sw": "log_inv"},
    ]

    n_models = len(configs)
    all_preds = []  # model_idx -> list of fold predictions

    print(f"\nモデル数: {n_models}\n")

    for m_idx, cfg in enumerate(configs):
        t1 = time.time()
        fold_preds = []
        fold_rmses = []
        for f_idx, (train_idx, test_idx) in enumerate(folds):
            try:
                pred = predict_fold(
                    X_raw[train_idx], X_raw[test_idx],
                    y[train_idx], groups[train_idx], cfg
                )
                fold_preds.append(pred)
                fold_rmses.append(rmse(y[test_idx], pred))
            except Exception as e:
                fold_preds.append(np.full(len(test_idx), y[train_idx].mean()))
                fold_rmses.append(999.0)
        all_preds.append(fold_preds)
        mean_r = np.mean(fold_rmses)
        print(f"  [{m_idx+1}/{n_models}] {cfg['name']}: {mean_r:.2f} ± {np.std(fold_rmses):.2f} ({time.time()-t1:.1f}s)", flush=True)

    names = [c["name"] for c in configs]

    # Individual rankings
    indiv = [(np.mean([rmse(y[folds[f][1]], all_preds[i][f]) for f in range(len(folds))]), i) for i in range(n_models)]
    indiv.sort()

    print("\n" + "=" * 70)
    print("個別モデルランキング:")
    print("=" * 70)
    for r, i in indiv:
        print(f"  {r:.2f} | {names[i]}")

    # Ensemble search
    print("\n" + "=" * 70)
    print("アンサンブル探索:")
    print("=" * 70)

    results = []

    # Top-K SimpleAvg
    for k in range(2, min(n_models + 1, 12)):
        idx = [i for _, i in indiv[:k]]
        r, s, _ = eval_ensemble(all_preds, y, folds, idx)
        results.append((f"SimpleAvg-Top{k}", r, s, idx, None))

    # Top-K WeightedAvg
    for k in [3, 4, 5, 6, 7, 8, 9, 10]:
        if k > n_models:
            break
        idx = [i for _, i in indiv[:k]]
        w, rw = optimize_weights_fold(all_preds, y, folds, idx)
        _, s, _ = eval_ensemble(all_preds, y, folds, idx, w)
        results.append((f"WeightedAvg-Top{k}", rw, s, idx, w))

    # Exhaustive best subset (k=3,4,5,6)
    for k in [3, 4, 5, 6]:
        best_r, best_combo = np.inf, None
        for combo in combinations(range(n_models), k):
            r, _, _ = eval_ensemble(all_preds, y, folds, list(combo))
            if r < best_r:
                best_r = r
                best_combo = combo
        if best_combo:
            idx = list(best_combo)
            r, s, _ = eval_ensemble(all_preds, y, folds, idx)
            results.append((f"BestCombo{k}-Avg", r, s, idx, None))
            w, rw = optimize_weights_fold(all_preds, y, folds, idx)
            _, s2, _ = eval_ensemble(all_preds, y, folds, idx, w)
            results.append((f"BestCombo{k}-Weighted", rw, s2, idx, w))

    results.sort(key=lambda x: x[1])

    print(f"\nTop 20 アンサンブル:")
    print("-" * 70)
    for name, r, s, idx, w in results[:20]:
        print(f"  RMSE={r:.4f} ± {s:.2f} | {name}")
        if w is not None:
            for i, wi in zip(idx, w):
                if wi > 0.01:
                    print(f"    {wi:.3f}: {names[i]}")
        else:
            for i in idx:
                print(f"    - {names[i]}")
        print()

    best = results[0]
    print("=" * 70)
    print(f"BEST: RMSE = {best[1]:.4f} ± {best[2]:.2f} ({best[0]})")
    print(f"v3ベースライン: 15.99")
    print(f"改善幅: {15.99 - best[1]:.4f}")
    print("=" * 70)

    # Fold details for best
    if best[4] is not None:
        _, _, fr = eval_ensemble(all_preds, y, folds, best[3], best[4])
    else:
        _, _, fr = eval_ensemble(all_preds, y, folds, best[3])
    print("\nFold詳細:")
    for sp, r in zip(species_names, fr):
        print(f"  {sp}: {r:.2f}")

    # Save
    rows = [{"method": n, "rmse": r, "std": s, "models": str([names[i] for i in idx]),
             "weights": str(w) if w is not None else "equal"} for n, r, s, idx, w in results]
    pd.DataFrame(rows).to_csv(OUT_DIR / "issue71_v4_ensemble_results.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'issue71_v4_ensemble_results.csv'}")
    print(f"総実行時間: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
