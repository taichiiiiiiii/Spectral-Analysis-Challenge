"""Issue #75: 拡張モデル探索 + スタッキング + 重み付きsiPLS

1. siPLS+サンプル重み付けの検証
2. siPLSパラメータ拡張探索
3. スタッキングメタラーナー
4. 最終アンサンブル最適化
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
from sklearn.model_selection import LeaveOneGroupOut

from src.eda.data_loader import load_train, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue20_savgol import apply_savgol
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue19_msc import compute_msc_reference
from src.preprocessing.issue62_additional_preprocessing import apply_asls, apply_piecewise_msc
from src.modeling.issue65_feature_selection import sipls_select, ipls_select

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_weights(groups_train, strategy="log_inv"):
    unique, counts = np.unique(groups_train, return_counts=True)
    cm = dict(zip(unique, counts))
    mx = counts.max()
    w = np.array([np.log(mx / cm[sp] + 1) if strategy == "log_inv"
                  else len(groups_train) / (len(unique) * cm[sp]) for sp in groups_train])
    return w


def predict_fold(X_tr_raw, X_te_raw, y_train, groups_train, cfg):
    pp = cfg["pp"]
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
        X_sg = apply_savgol(X_tr_raw, deriv=2, window_length=7)
        X_sg_te = apply_savgol(X_te_raw, deriv=2, window_length=7)
        P = compute_epo_projection(X_sg, groups_train, n_components=1)
        X_tr, X_te = apply_epo(X_sg, P), apply_epo(X_sg_te, P)
    else:
        X_tr, X_te = X_tr_raw.copy(), X_te_raw.copy()

    fs = cfg.get("fs")
    if fs and fs.startswith("siPLS"):
        parts = fs.replace("siPLS(", "").rstrip(")").split(",")
        ni, nc = int(parts[0]), int(parts[1])
        X_tr, X_te, _ = sipls_select(X_tr, y_train, X_te, n_intervals=ni, n_components=3, n_combine=nc)
    elif fs and fs.startswith("iPLS"):
        ni = int(fs.replace("iPLS(", "").rstrip(")"))
        X_tr, X_te, _ = ipls_select(X_tr, y_train, X_te, n_intervals=ni, n_components=3, n_best=1)

    tf = cfg["tf"]
    y_fit = np.sqrt(y_train) if tf == "sqrt" else y_train.copy()

    sw = cfg.get("sw")
    weights = compute_weights(groups_train, sw) if sw else None

    mdl = cfg["model"]
    if mdl.startswith("PLS("):
        nc = int(mdl.split("(")[1].rstrip(")"))
        nc = min(nc, X_tr.shape[1] - 1)
        m = PLSRegression(n_components=max(1, nc))
        if weights is not None:
            sqrt_w = np.sqrt(weights)
            m.fit(X_tr * sqrt_w[:, None], y_fit * sqrt_w)
        else:
            m.fit(X_tr, y_fit)
        pred = m.predict(X_te).ravel()
    elif mdl == "Lasso":
        m = Lasso(alpha=0.1, max_iter=10000)
        m.fit(X_tr, y_fit, sample_weight=weights) if weights is not None else m.fit(X_tr, y_fit)
        pred = m.predict(X_te)
    elif mdl == "ElasticNet":
        m = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
        m.fit(X_tr, y_fit, sample_weight=weights) if weights is not None else m.fit(X_tr, y_fit)
        pred = m.predict(X_te)
    elif mdl == "Ridge":
        m = Ridge(alpha=100)
        m.fit(X_tr, y_fit, sample_weight=weights) if weights is not None else m.fit(X_tr, y_fit)
        pred = m.predict(X_te)
    else:
        raise ValueError(mdl)

    if tf == "sqrt":
        pred = np.clip(pred, 0, None) ** 2
    return pred


def eval_ens(preds, y, folds, idx, w=None):
    fr = []
    for f, (_, te) in enumerate(folds):
        ps = [preds[i][f] for i in idx]
        e = sum(wi * p for wi, p in zip(w, ps)) if w is not None else np.mean(ps, axis=0)
        fr.append(rmse(y[te], np.clip(e, 0, 300)))
    return np.mean(fr), np.std(fr), fr


def opt_w(preds, y, folds, idx, nr=30):
    n = len(idx)
    def obj(w):
        wn = np.abs(w) / np.sum(np.abs(w))
        r, _, _ = eval_ens(preds, y, folds, idx, wn)
        return r
    best_r, best_w = np.inf, np.ones(n) / n
    for s in range(nr):
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
    X = df[sc].values
    y = df["含水率"].values
    g = df["樹種"].values
    logo = LeaveOneGroupOut()
    folds = list(logo.split(X, y, g))
    sp_names = [np.unique(g[te])[0] for _, te in folds]

    print("=" * 70)
    print("Issue #75: 拡張モデル探索 + スタッキング")
    print("  現行ベスト v5: 15.28, v6候補: 15.18")
    print("=" * 70)

    # 幅広いモデル候補
    configs = [
        # v5/v6 proven models
        {"name": "EPO+PLS4+sqrt", "pp": "EPO(1)", "model": "PLS(4)", "tf": "sqrt"},
        {"name": "SG2d+EPO+PLS3+raw", "pp": "SG2d+EPO(1)", "model": "PLS(3)", "tf": "raw"},
        {"name": "SNV+AsLS+siPLS(30,3)+PLS4+sqrt", "pp": "SNV+AsLS(1e6)", "model": "PLS(4)", "tf": "sqrt", "fs": "siPLS(30,3)"},
        {"name": "SNV+iPLS(50)+PLS4+sqrt", "pp": "SNV", "model": "PLS(4)", "tf": "sqrt", "fs": "iPLS(50)"},
        {"name": "PMSC+siPLS(30,3)+PLS4+sqrt", "pp": "PiecewiseMSC(seg=3)", "model": "PLS(4)", "tf": "sqrt", "fs": "siPLS(30,3)"},
        # NEW: siPLS parameter variants
        {"name": "SNV+siPLS(40,3)+PLS4+sqrt", "pp": "SNV", "model": "PLS(4)", "tf": "sqrt", "fs": "siPLS(40,3)"},
        {"name": "SNV+siPLS(50,3)+PLS4+sqrt", "pp": "SNV", "model": "PLS(4)", "tf": "sqrt", "fs": "siPLS(50,3)"},
        {"name": "SNV+siPLS(30,2)+PLS4+sqrt", "pp": "SNV", "model": "PLS(4)", "tf": "sqrt", "fs": "siPLS(30,2)"},
        {"name": "SNV+siPLS(30,4)+PLS4+sqrt", "pp": "SNV", "model": "PLS(4)", "tf": "sqrt", "fs": "siPLS(30,4)"},
        # NEW: siPLS + different PLS components
        {"name": "SNV+AsLS+siPLS(30,3)+PLS3+sqrt", "pp": "SNV+AsLS(1e6)", "model": "PLS(3)", "tf": "sqrt", "fs": "siPLS(30,3)"},
        {"name": "SNV+AsLS+siPLS(30,3)+PLS5+sqrt", "pp": "SNV+AsLS(1e6)", "model": "PLS(5)", "tf": "sqrt", "fs": "siPLS(30,3)"},
        {"name": "SNV+iPLS(50)+PLS3+sqrt", "pp": "SNV", "model": "PLS(3)", "tf": "sqrt", "fs": "iPLS(50)"},
        # NEW: siPLS + sample weighting
        {"name": "SNV+siPLS(30,3)+PLS4+sqrt+logW", "pp": "SNV", "model": "PLS(4)", "tf": "sqrt", "fs": "siPLS(30,3)", "sw": "log_inv"},
        {"name": "SNV+iPLS(50)+PLS4+sqrt+logW", "pp": "SNV", "model": "PLS(4)", "tf": "sqrt", "fs": "iPLS(50)", "sw": "log_inv"},
        {"name": "PMSC+siPLS(30,3)+PLS4+sqrt+logW", "pp": "PiecewiseMSC(seg=3)", "model": "PLS(4)", "tf": "sqrt", "fs": "siPLS(30,3)", "sw": "log_inv"},
        # Additional diversity
        {"name": "SNV+ElasticNet+sqrt", "pp": "SNV", "model": "ElasticNet", "tf": "sqrt"},
        {"name": "PMSC+Lasso+raw", "pp": "PiecewiseMSC(seg=3)", "model": "Lasso", "tf": "raw"},
        {"name": "EPO+Lasso+raw", "pp": "EPO(1)", "model": "Lasso", "tf": "raw"},
        {"name": "SNV+PLS2+sqrt", "pp": "SNV", "model": "PLS(2)", "tf": "sqrt"},
        {"name": "SNV+AsLS+PLS2+sqrt", "pp": "SNV+AsLS(1e6)", "model": "PLS(2)", "tf": "sqrt"},
    ]

    n = len(configs)
    all_preds = []
    names = [c["name"] for c in configs]

    print(f"\nモデル数: {n}\n")
    for i, cfg in enumerate(configs):
        t1 = time.time()
        fp = []
        fr = []
        for f, (tr, te) in enumerate(folds):
            try:
                p = predict_fold(X[tr], X[te], y[tr], g[tr], cfg)
                fp.append(p)
                fr.append(rmse(y[te], p))
            except Exception as e:
                fp.append(np.full(len(te), y[tr].mean()))
                fr.append(999.0)
        all_preds.append(fp)
        print(f"  [{i+1}/{n}] {cfg['name']}: {np.mean(fr):.2f} ± {np.std(fr):.2f} ({time.time()-t1:.1f}s)", flush=True)

    # Individual rankings
    indiv = sorted([(np.mean([rmse(y[folds[f][1]], all_preds[i][f]) for f in range(len(folds))]), i) for i in range(n)])
    print("\n個別モデルランキング:")
    for r, i in indiv:
        print(f"  {r:.2f} | {names[i]}")

    # ===== Stacking =====
    print("\n" + "=" * 70)
    print("スタッキング（メタラーナー）")
    print("=" * 70)

    # Collect OOF predictions as features
    n_samples = len(y)
    oof_features = np.zeros((n_samples, n))
    for m_idx in range(n):
        for f_idx, (_, te) in enumerate(folds):
            oof_features[te, m_idx] = all_preds[m_idx][f_idx]

    # Nested stacking: use inner LOSO to train meta-learner
    for meta_name, meta_model in [("Ridge(1)", Ridge(alpha=1)), ("Ridge(10)", Ridge(alpha=10)),
                                   ("Ridge(100)", Ridge(alpha=100)), ("Lasso(0.01)", Lasso(alpha=0.01))]:
        stack_fold_rmses = []
        for f_idx, (tr, te) in enumerate(folds):
            X_meta_tr = oof_features[tr]
            X_meta_te = oof_features[te]
            meta_model_copy = type(meta_model)(**meta_model.get_params())
            meta_model_copy.fit(X_meta_tr, y[tr])
            pred = meta_model_copy.predict(X_meta_te)
            pred = np.clip(pred, 0, 300)
            stack_fold_rmses.append(rmse(y[te], pred))
        print(f"  Stacking({meta_name}): {np.mean(stack_fold_rmses):.4f} ± {np.std(stack_fold_rmses):.2f}")

    # ===== Ensemble search =====
    print("\n" + "=" * 70)
    print("アンサンブル探索")
    print("=" * 70)

    results = []

    # Top-K SimpleAvg
    for k in range(2, min(n + 1, 10)):
        idx = [i for _, i in indiv[:k]]
        r, s, _ = eval_ens(all_preds, y, folds, idx)
        results.append((f"SimpleAvg-Top{k}", r, s, idx, None))

    # Top-K WeightedAvg
    for k in [3, 4, 5, 6, 7, 8]:
        if k > n:
            break
        idx = [i for _, i in indiv[:k]]
        w, rw = opt_w(all_preds, y, folds, idx)
        _, s, _ = eval_ens(all_preds, y, folds, idx, w)
        results.append((f"WeightedAvg-Top{k}", rw, s, idx, w))

    # Exhaustive best combo
    for k in [3, 4, 5, 6, 7]:
        best_r, best_c = np.inf, None
        for combo in combinations(range(n), k):
            r, _, _ = eval_ens(all_preds, y, folds, list(combo))
            if r < best_r:
                best_r = r
                best_c = combo
        if best_c:
            idx = list(best_c)
            results.append((f"BestCombo{k}-Avg", best_r, 0, idx, None))
            w, rw = opt_w(all_preds, y, folds, idx)
            _, s, _ = eval_ens(all_preds, y, folds, idx, w)
            results.append((f"BestCombo{k}-Weighted", rw, s, idx, w))

    results.sort(key=lambda x: x[1])

    print(f"\nTop 20:")
    for name, r, s, idx, w in results[:20]:
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
    if best[4] is not None:
        _, _, fr = eval_ens(all_preds, y, folds, best[3], best[4])
    else:
        _, _, fr = eval_ens(all_preds, y, folds, best[3])
    print("=" * 70)
    print(f"BEST: {best[1]:.4f} ({best[0]})")
    print(f"v5: 15.28 | 改善: {15.28 - best[1]:.4f}")
    print("Fold詳細:")
    for sp, r in zip(sp_names, fr):
        print(f"  {sp}: {r:.2f}")
    print("=" * 70)

    rows = [{"method": nm, "rmse": r, "models": str([names[i] for i in idx])} for nm, r, _, idx, _ in results]
    pd.DataFrame(rows).to_csv(OUT_DIR / "issue75_extended_results.csv", index=False)
    print(f"\n総実行時間: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
