"""Issue #86: サイクル10 - Huber回帰深堀り

H2(Huber, 重み0.29)の発見をさらに深堀り:
1. Huberのepsilonパラメータ変更（1.1, 1.2, 1.35, 1.5, 2.0）
2. 複数の前処理×Huber組み合わせ
3. Huber + GBRのハイブリッドモデル（Huber残差をGBRで補正）
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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
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

def preprocess(X_tr_raw, X_te_raw, groups_train, pp):
    if pp == "SNV":
        return apply_snv(X_tr_raw), apply_snv(X_te_raw)
    elif pp == "EPO(1)":
        P = compute_epo_projection(X_tr_raw, groups_train, n_components=1)
        return apply_epo(X_tr_raw, P), apply_epo(X_te_raw, P)
    elif pp == "SNV+AsLS(1e6)":
        return apply_asls(apply_snv(X_tr_raw), lam=1e6), apply_asls(apply_snv(X_te_raw), lam=1e6)
    elif pp == "PiecewiseMSC(seg=3)":
        ref = compute_msc_reference(X_tr_raw)
        return apply_piecewise_msc(X_tr_raw, ref, 3), apply_piecewise_msc(X_te_raw, ref, 3)
    elif pp == "SG2d+EPO(1)":
        X_sg = apply_savgol(X_tr_raw, deriv=2, window_length=7)
        X_sg_te = apply_savgol(X_te_raw, deriv=2, window_length=7)
        P = compute_epo_projection(X_sg, groups_train, n_components=1)
        return apply_epo(X_sg, P), apply_epo(X_sg_te, P)
    elif pp == "SNV+SG2d":
        return apply_savgol(apply_snv(X_tr_raw), deriv=2, window_length=7), \
               apply_savgol(apply_snv(X_te_raw), deriv=2, window_length=7)
    return X_tr_raw.copy(), X_te_raw.copy()

def feature_select(X_tr, X_te, y_train, fs):
    if not fs:
        return X_tr, X_te
    if fs.startswith("siPLS"):
        parts = fs.replace("siPLS(", "").rstrip(")").split(",")
        ni, nc = int(parts[0]), int(parts[1])
        X_tr, X_te, _ = sipls_select(X_tr, y_train, X_te, n_intervals=ni, n_components=3, n_combine=nc)
    elif fs.startswith("iPLS"):
        ni = int(fs.replace("iPLS(", "").rstrip(")"))
        X_tr, X_te, _ = ipls_select(X_tr, y_train, X_te, n_intervals=ni, n_components=3, n_best=1)
    return X_tr, X_te

def predict_std(X_tr_raw, X_te_raw, y_train, groups_train, cfg):
    pp, fs, tf, nc = cfg["pp"], cfg.get("fs"), cfg["tf"], cfg.get("nc", 4)
    X_tr, X_te = preprocess(X_tr_raw, X_te_raw, groups_train, pp)
    X_tr, X_te = feature_select(X_tr, X_te, y_train, fs)
    nc = max(1, min(nc, X_tr.shape[1]-1))
    y_fit = np.sqrt(y_train) if tf == "sqrt" else y_train.copy()
    pls = PLSRegression(n_components=nc)
    pls.fit(X_tr, y_fit)
    pred = pls.predict(X_te).ravel()
    if tf == "sqrt": pred = np.clip(pred, 0, None) ** 2
    return pred

def predict_gbr(X_tr_raw, X_te_raw, y_train, groups_train, cfg):
    pp, fs, tf, nc = cfg["pp"], cfg.get("fs"), cfg["tf"], cfg.get("nc", 4)
    X_tr, X_te = preprocess(X_tr_raw, X_te_raw, groups_train, pp)
    X_tr, X_te = feature_select(X_tr, X_te, y_train, fs)
    nc = max(1, min(nc, X_tr.shape[1]-1))
    y_fit = np.sqrt(y_train) if tf == "sqrt" else y_train.copy()
    pls = PLSRegression(n_components=nc); pls.fit(X_tr, y_fit)
    T_tr, T_te = pls.transform(X_tr), pls.transform(X_te)
    gbr = GradientBoostingRegressor(
        n_estimators=cfg.get("n_est", 200), max_depth=cfg.get("md", 3),
        learning_rate=cfg.get("lr", 0.05), subsample=0.8, min_samples_leaf=5,
        random_state=42, validation_fraction=0.15, n_iter_no_change=50, tol=0.01)
    gbr.fit(T_tr, y_fit); pred = gbr.predict(T_te)
    if tf == "sqrt": pred = np.clip(pred, 0, None) ** 2
    return pred

def predict_huber(X_tr_raw, X_te_raw, y_train, groups_train, cfg):
    pp, fs, tf, nc = cfg["pp"], cfg.get("fs"), cfg["tf"], cfg.get("nc", 4)
    eps = cfg.get("eps", 1.35)
    alpha = cfg.get("alpha", 0.01)
    X_tr, X_te = preprocess(X_tr_raw, X_te_raw, groups_train, pp)
    X_tr, X_te = feature_select(X_tr, X_te, y_train, fs)
    nc = max(1, min(nc, X_tr.shape[1]-1))
    y_fit = np.sqrt(y_train) if tf == "sqrt" else y_train.copy()
    pls = PLSRegression(n_components=nc); pls.fit(X_tr, y_fit)
    T_tr, T_te = pls.transform(X_tr), pls.transform(X_te)
    sc = StandardScaler()
    T_tr_s, T_te_s = sc.fit_transform(T_tr), sc.transform(T_te)
    huber = HuberRegressor(epsilon=eps, max_iter=200, alpha=alpha)
    huber.fit(T_tr_s, y_fit); pred = huber.predict(T_te_s)
    if tf == "sqrt": pred = np.clip(pred, 0, None) ** 2
    return pred

def main():
    t0 = time.time()
    df = load_train(DATA_DIR)
    sc = get_spectral_columns(df)
    X_raw = df[sc].values
    y = df["含水率"].values
    groups = df["樹種"].values
    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_raw, y, groups))
    sp_names = [np.unique(groups[te])[0] for _, te in folds]

    print("=" * 70)
    print("Issue #86: サイクル10 - Huber回帰深堀り")
    print(f"  前ベスト: 14.1594")
    print("=" * 70)

    models = [
        # Cycle9ベスト9 (H2含む)
        {"name": "M1:EPO+PLS4+sqrt", "pp": "EPO(1)", "nc": 4, "tf": "sqrt", "f": predict_std},
        {"name": "M5:SNV+AsLS+siPLS+PLS5+sqrt", "pp": "SNV+AsLS(1e6)", "nc": 5, "tf": "sqrt", "fs": "siPLS(30,3)", "f": predict_std},
        {"name": "N10:PMSC+siPLS+GBR+sqrt", "pp": "PiecewiseMSC(seg=3)", "nc": 4, "tf": "sqrt", "fs": "siPLS(30,3)", "f": predict_gbr},
        {"name": "G9:SG2d+EPO+GBR+raw", "pp": "SG2d+EPO(1)", "nc": 4, "tf": "raw", "f": predict_gbr},
        {"name": "N2:SNV+SG2d+GBR+raw", "pp": "SNV+SG2d", "nc": 4, "tf": "raw", "f": predict_gbr},
        {"name": "Gnew1:EPO+GBR150+raw", "pp": "EPO(1)", "nc": 4, "tf": "raw", "n_est": 150, "lr": 0.08, "f": predict_gbr},
        {"name": "H2:SNV+iPLS50+Huber(1.35)+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt", "fs": "iPLS(50)", "eps": 1.35, "f": predict_huber},

        # Huber epsilonバリエーション
        {"name": "H2a:SNV+iPLS50+Huber(1.1)+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt", "fs": "iPLS(50)", "eps": 1.1, "f": predict_huber},
        {"name": "H2b:SNV+iPLS50+Huber(1.5)+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt", "fs": "iPLS(50)", "eps": 1.5, "f": predict_huber},
        {"name": "H2c:SNV+iPLS50+Huber(2.0)+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt", "fs": "iPLS(50)", "eps": 2.0, "f": predict_huber},

        # Huber + 異なる前処理
        {"name": "H3:EPO+Huber(1.35)+sqrt", "pp": "EPO(1)", "nc": 4, "tf": "sqrt", "eps": 1.35, "f": predict_huber},
        {"name": "H4:SNV+Huber(1.35)+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt", "eps": 1.35, "f": predict_huber},
        {"name": "H5:PMSC+siPLS+Huber(1.35)+sqrt", "pp": "PiecewiseMSC(seg=3)", "nc": 4, "tf": "sqrt", "fs": "siPLS(30,3)", "eps": 1.35, "f": predict_huber},
        {"name": "H6:SNV+AsLS+siPLS+Huber(1.35)+sqrt", "pp": "SNV+AsLS(1e6)", "nc": 4, "tf": "sqrt", "fs": "siPLS(30,3)", "eps": 1.35, "f": predict_huber},
        {"name": "H7:SG2d+EPO+Huber(1.35)+raw", "pp": "SG2d+EPO(1)", "nc": 4, "tf": "raw", "eps": 1.35, "f": predict_huber},

        # Huber + alpha変更
        {"name": "H8:SNV+iPLS50+Huber(1.35,a001)+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt", "fs": "iPLS(50)", "eps": 1.35, "alpha": 0.001, "f": predict_huber},
        {"name": "H9:SNV+iPLS50+Huber(1.35,a1)+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt", "fs": "iPLS(50)", "eps": 1.35, "alpha": 1.0, "f": predict_huber},

        # 追加PLS
        {"name": "M3:SNV+iPLS50+PLS4+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt", "fs": "iPLS(50)", "f": predict_std},
        {"name": "G11:EPO+GBR+raw", "pp": "EPO(1)", "nc": 4, "tf": "raw", "f": predict_gbr},
    ]

    n_models = len(models)
    all_preds = [[] for _ in range(n_models)]

    print(f"\n--- 個別モデル ({n_models}) ---\n", flush=True)
    for m_idx, cfg in enumerate(models):
        t1 = time.time()
        fold_rmses = []
        for f_idx, (tr, te) in enumerate(folds):
            try:
                pred = cfg["f"](X_raw[tr], X_raw[te], y[tr], groups[tr], cfg)
                all_preds[m_idx].append(pred)
                fold_rmses.append(rmse(y[te], pred))
            except Exception as e:
                all_preds[m_idx].append(np.full(len(te), y[tr].mean()))
                fold_rmses.append(999.0)
                print(f"  ERR {cfg['name']} f{f_idx}: {e}")
        print(f"  [{m_idx+1:>2}/{n_models}] {cfg['name']}: {np.mean(fold_rmses):.2f} ± {np.std(fold_rmses):.2f} ({time.time()-t1:.1f}s)", flush=True)

    def eval_ens(indices, weights=None):
        fold_rmses = []
        for f_idx, (_, te) in enumerate(folds):
            preds = [all_preds[m][f_idx] for m in indices]
            if weights is not None:
                w = np.array(weights); w = w / w.sum()
                ens = sum(wi * p for wi, p in zip(w, preds))
            else:
                ens = np.mean(preds, axis=0)
            fold_rmses.append(rmse(y[te], np.clip(ens, 0, 300)))
        return np.mean(fold_rmses), fold_rmses

    def opt_w(indices, n_restarts=40):
        n = len(indices)
        def obj(w):
            w_n = np.abs(w) / np.sum(np.abs(w))
            r, _ = eval_ens(indices, w_n)
            return r
        best_r, best_w = np.inf, np.ones(n) / n
        for s in range(n_restarts):
            w0 = np.random.RandomState(s).dirichlet(np.ones(n))
            res = minimize(obj, w0, method="Nelder-Mead", options={"maxiter": 5000})
            if res.fun < best_r:
                best_r = res.fun
                best_w = np.abs(res.x) / np.sum(np.abs(res.x))
        return best_w, best_r

    indiv = sorted([(eval_ens([i])[0], i) for i in range(n_models)])
    print("\n個別ランキング:")
    for r, i in indiv:
        print(f"  {r:.2f} | {models[i]['name']}")

    top14 = [i for _, i in indiv[:14]]
    results = []
    for k in [7, 8, 9, 10, 11]:
        best_r, best_combo = np.inf, None
        for combo in combinations(top14, k):
            r, _ = eval_ens(list(combo))
            if r < best_r:
                best_r = r
                best_combo = list(combo)
        if best_combo:
            w, r_w = opt_w(best_combo)
            results.append((f"BestCombo{k}-W", r_w, best_combo, w))
            print(f"  BestCombo{k}-W: {r_w:.4f}", flush=True)
            for i, wi in zip(best_combo, w):
                if wi > 0.02:
                    print(f"    {wi:.3f}: {models[i]['name']}")

    results.sort(key=lambda x: x[1])
    best = results[0]
    print(f"\n{'='*70}")
    print(f"BEST: {best[1]:.4f} ({best[0]})")
    print(f"前ベスト: 14.1594, 改善: {14.1594 - best[1]:+.4f}")
    print(f"{'='*70}")

    _, fr = eval_ens(best[2], best[3])
    print("\nfold詳細:")
    for sp, r in zip(sp_names, fr):
        tag = " ※" if sp == "ベイスギ" else ""
        print(f"  {sp}: {r:.2f}{tag}")

    rows = [{"method": n, "rmse": r, "models": str([models[i]["name"] for i in c])} for n, r, c, w in results]
    pd.DataFrame(rows).to_csv(OUT_DIR / "issue86_cycle10_results.csv", index=False)
    print(f"\n保存: {OUT_DIR / 'issue86_cycle10_results.csv'}")
    print(f"時間: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
