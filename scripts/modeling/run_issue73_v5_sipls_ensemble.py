"""Issue #73: v5アンサンブル — siPLS/iPLSモデル追加

siPLS(int=30,c=3)+SNV+AsLS+PLS(4)+sqrt = 16.88 を含む多様なモデルで
v3(15.99)を超えるアンサンブルを構築する。
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
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.model_selection import LeaveOneGroupOut

from src.eda.data_loader import load_train, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue20_savgol import apply_savgol
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue19_msc import compute_msc_reference
from src.preprocessing.issue62_additional_preprocessing import apply_asls, apply_piecewise_msc
from src.modeling.issue65_feature_selection import (
    sipls_select, ipls_select, vip_select,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def predict_fold(X_tr_raw, X_te_raw, y_train, groups_train, config):
    pp = config["pp"]
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
    elif pp == "SG2d+EPO(1)":
        X_tr_sg = apply_savgol(X_tr_raw, deriv=2, window_length=7)
        X_te_sg = apply_savgol(X_te_raw, deriv=2, window_length=7)
        P = compute_epo_projection(X_tr_sg, groups_train, n_components=1)
        X_tr, X_te = apply_epo(X_tr_sg, P), apply_epo(X_te_sg, P)
    else:
        X_tr, X_te = X_tr_raw.copy(), X_te_raw.copy()

    # Feature selection
    fs = config.get("fs")
    if fs == "VIP(1.5)":
        X_tr, X_te = vip_select(X_tr, y_train, X_te, threshold=1.5)
    elif fs == "siPLS(30,3)":
        X_tr, X_te, _ = sipls_select(X_tr, y_train, X_te, n_intervals=30, n_components=3, n_combine=3)
    elif fs == "siPLS(20,3)":
        X_tr, X_te, _ = sipls_select(X_tr, y_train, X_te, n_intervals=20, n_components=3, n_combine=3)
    elif fs == "iPLS(50)":
        X_tr, X_te, _ = ipls_select(X_tr, y_train, X_te, n_intervals=50, n_components=3, n_best=1)
    elif fs == "iPLS(20)":
        X_tr, X_te, _ = ipls_select(X_tr, y_train, X_te, n_intervals=20, n_components=3, n_best=1)

    tf = config["tf"]
    y_fit = np.sqrt(y_train) if tf == "sqrt" else y_train.copy()

    mdl = config["model"]
    if mdl.startswith("PLS("):
        nc = int(mdl.split("(")[1].rstrip(")"))
        nc = min(nc, X_tr.shape[1] - 1)
        m = PLSRegression(n_components=max(1, nc))
        m.fit(X_tr, y_fit)
        pred = m.predict(X_te).ravel()
    elif mdl == "Lasso":
        m = Lasso(alpha=0.1, max_iter=10000)
        m.fit(X_tr, y_fit)
        pred = m.predict(X_te)
    elif mdl == "ElasticNet":
        m = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
        m.fit(X_tr, y_fit)
        pred = m.predict(X_te)
    else:
        raise ValueError(mdl)

    if tf == "sqrt":
        pred = np.clip(pred, 0, None) ** 2
    return pred


def eval_ensemble(all_preds, y, folds, idx, weights=None):
    fold_rmses = []
    for f, (_, te) in enumerate(folds):
        preds = [all_preds[i][f] for i in idx]
        ens = sum(w * p for w, p in zip(weights, preds)) if weights else np.mean(preds, axis=0)
        fold_rmses.append(rmse(y[te], np.clip(ens, 0, 200)))
    return np.mean(fold_rmses), np.std(fold_rmses), fold_rmses


def opt_weights(all_preds, y, folds, idx, n_restarts=30):
    n = len(idx)
    def obj(w):
        wn = np.abs(w) / np.sum(np.abs(w))
        r, _, _ = eval_ensemble(all_preds, y, folds, idx, wn)
        return r
    best_r, best_w = np.inf, np.ones(n) / n
    for s in range(n_restarts):
        w0 = np.random.RandomState(s).dirichlet(np.ones(n))
        res = minimize(obj, w0, method="Nelder-Mead", options={"maxiter": 3000})
        if res.fun < best_r:
            best_r = res.fun
            best_w = np.abs(res.x) / np.sum(np.abs(res.x))
    return best_w, best_r


def main():
    t0 = time.time()
    df = load_train(DATA_DIR)
    sc = get_spectral_columns(df)
    X_raw = df[sc].values
    y = df["含水率"].values
    groups = df["樹種"].values

    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_raw, y, groups))
    species = [np.unique(groups[te])[0] for _, te in folds]

    print("=" * 70)
    print("Issue #73: v5アンサンブル（siPLS/iPLS追加）")
    print("  v3ベースライン: 15.99")
    print("=" * 70)

    configs = [
        # v3 original 5 models
        {"name": "EPO(1)+PLS(4)+sqrt", "pp": "EPO(1)", "model": "PLS(4)", "tf": "sqrt"},
        {"name": "SG2d+EPO(1)+PLS(3)+raw", "pp": "SG2d+EPO(1)", "model": "PLS(3)", "tf": "raw"},
        {"name": "PiecewiseMSC+Lasso+raw", "pp": "PiecewiseMSC(seg=3)", "model": "Lasso", "tf": "raw"},
        {"name": "SNV+VIP(1.5)+PLS(4)+sqrt", "pp": "SNV", "model": "PLS(4)", "tf": "sqrt", "fs": "VIP(1.5)"},
        {"name": "SNV+AsLS+PLS(2)+sqrt", "pp": "SNV+AsLS(1e6)", "model": "PLS(2)", "tf": "sqrt"},
        # NEW: siPLS/iPLS models
        {"name": "SNV+AsLS+siPLS(30,3)+PLS(4)+sqrt", "pp": "SNV+AsLS(1e6)", "model": "PLS(4)", "tf": "sqrt", "fs": "siPLS(30,3)"},
        {"name": "SNV+AsLS+siPLS(20,3)+PLS(4)+sqrt", "pp": "SNV+AsLS(1e6)", "model": "PLS(4)", "tf": "sqrt", "fs": "siPLS(20,3)"},
        {"name": "SNV+iPLS(50)+PLS(4)+sqrt", "pp": "SNV", "model": "PLS(4)", "tf": "sqrt", "fs": "iPLS(50)"},
        {"name": "SNV+iPLS(20)+PLS(2)+sqrt", "pp": "SNV", "model": "PLS(2)", "tf": "sqrt", "fs": "iPLS(20)"},
        {"name": "SNV+siPLS(30,3)+PLS(4)+sqrt", "pp": "SNV", "model": "PLS(4)", "tf": "sqrt", "fs": "siPLS(30,3)"},
        # Additional diverse models
        {"name": "SNV+ElasticNet+sqrt", "pp": "SNV", "model": "ElasticNet", "tf": "sqrt"},
        {"name": "SNV+PLS(2)+sqrt", "pp": "SNV", "model": "PLS(2)", "tf": "sqrt"},
        {"name": "EPO(1)+Lasso+raw", "pp": "EPO(1)", "model": "Lasso", "tf": "raw"},
    ]

    n_models = len(configs)
    all_preds = []
    names = [c["name"] for c in configs]

    print(f"\nモデル数: {n_models}\n")
    for m_idx, cfg in enumerate(configs):
        t1 = time.time()
        fold_preds = []
        fold_rmses = []
        for f_idx, (tr, te) in enumerate(folds):
            try:
                pred = predict_fold(X_raw[tr], X_raw[te], y[tr], groups[tr], cfg)
                fold_preds.append(pred)
                fold_rmses.append(rmse(y[te], pred))
            except Exception as e:
                fold_preds.append(np.full(len(te), y[tr].mean()))
                fold_rmses.append(999.0)
        all_preds.append(fold_preds)
        print(f"  [{m_idx+1}/{n_models}] {cfg['name']}: {np.mean(fold_rmses):.2f} ± {np.std(fold_rmses):.2f} ({time.time()-t1:.1f}s)", flush=True)

    # Rankings
    indiv = sorted([(np.mean([rmse(y[folds[f][1]], all_preds[i][f]) for f in range(len(folds))]), i) for i in range(n_models)])
    print("\n個別モデル:")
    for r, i in indiv:
        print(f"  {r:.2f} | {names[i]}")

    # Ensemble search
    print("\n" + "=" * 70)
    results = []

    for k in range(2, min(n_models + 1, 10)):
        idx = [i for _, i in indiv[:k]]
        r, s, _ = eval_ensemble(all_preds, y, folds, idx)
        results.append((f"SimpleAvg-Top{k}", r, s, idx, None))

    for k in [3, 4, 5, 6, 7, 8]:
        if k > n_models:
            break
        idx = [i for _, i in indiv[:k]]
        w, rw = opt_weights(all_preds, y, folds, idx)
        _, s, _ = eval_ensemble(all_preds, y, folds, idx, w)
        results.append((f"WeightedAvg-Top{k}", rw, s, idx, w))

    for k in [3, 4, 5, 6, 7]:
        best_r, best_combo = np.inf, None
        for combo in combinations(range(n_models), k):
            r, _, _ = eval_ensemble(all_preds, y, folds, list(combo))
            if r < best_r:
                best_r = r
                best_combo = combo
        if best_combo:
            idx = list(best_combo)
            results.append((f"BestCombo{k}-Avg", best_r, 0, idx, None))
            w, rw = opt_weights(all_preds, y, folds, idx)
            _, s, _ = eval_ensemble(all_preds, y, folds, idx, w)
            results.append((f"BestCombo{k}-Weighted", rw, s, idx, w))

    results.sort(key=lambda x: x[1])

    print(f"\nTop 15:")
    print("-" * 70)
    for name, r, s, idx, w in results[:15]:
        print(f"  RMSE={r:.4f} | {name}")
        if w is not None:
            for i, wi in zip(idx, w):
                if wi > 0.01:
                    print(f"    {wi:.3f}: {names[i]}")
        else:
            for i in idx:
                print(f"    - {names[i]}")
        print()

    best = results[0]
    # Fold details
    if best[4] is not None:
        _, _, fr = eval_ensemble(all_preds, y, folds, best[3], best[4])
    else:
        _, _, fr = eval_ensemble(all_preds, y, folds, best[3])

    print("=" * 70)
    print(f"BEST: RMSE = {best[1]:.4f} ({best[0]})")
    print(f"v3: 15.99 | 改善: {15.99 - best[1]:.4f}")
    print("Fold詳細:")
    for sp, r in zip(species, fr):
        print(f"  {sp}: {r:.2f}")
    print("=" * 70)

    rows = [{"method": n, "rmse": r, "models": str([names[i] for i in idx])} for n, r, _, idx, _ in results]
    pd.DataFrame(rows).to_csv(OUT_DIR / "issue73_v5_ensemble_results.csv", index=False)
    print(f"\nSaved. 総実行時間: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
