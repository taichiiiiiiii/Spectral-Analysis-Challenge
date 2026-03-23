"""Issue #84: サイクル8 - PLSスコア拡張特徴量 + Huber回帰

1. PLSスコアの交差項・二乗項を追加してGBRに入力
2. Huber回帰（外れ値頑健）をPLSスコアに適用
3. 複数前処理スペクトルの結合（multi-block PLS的アプローチ）
4. ElasticNet on PLS scores
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
from sklearn.linear_model import HuberRegressor, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
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
    elif pp == "multi_SNV_EPO":
        # SNVとEPOの結合
        X_snv_tr, X_snv_te = apply_snv(X_tr_raw), apply_snv(X_te_raw)
        P = compute_epo_projection(X_tr_raw, groups_train, n_components=1)
        X_epo_tr, X_epo_te = apply_epo(X_tr_raw, P), apply_epo(X_te_raw, P)
        return np.hstack([X_snv_tr, X_epo_tr]), np.hstack([X_snv_te, X_epo_te])
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


def get_pls_scores(X_tr, X_te, y_train, nc, tf):
    nc = min(nc, X_tr.shape[1] - 1, max(1, nc))
    y_fit = np.sqrt(y_train) if tf == "sqrt" else y_train.copy()
    pls = PLSRegression(n_components=nc)
    pls.fit(X_tr, y_fit)
    return pls.transform(X_tr), pls.transform(X_te), pls, y_fit


def predict_pls(X_tr, X_te, y_train, nc, tf):
    nc = min(nc, X_tr.shape[1] - 1)
    nc = max(1, nc)
    y_fit = np.sqrt(y_train) if tf == "sqrt" else y_train.copy()
    pls = PLSRegression(n_components=nc)
    pls.fit(X_tr, y_fit)
    pred = pls.predict(X_te).ravel()
    if tf == "sqrt":
        pred = np.clip(pred, 0, None) ** 2
    return pred


def predict_fold_std(X_tr_raw, X_te_raw, y_train, groups_train, cfg):
    pp, fs, tf, nc = cfg["pp"], cfg.get("fs"), cfg["tf"], cfg.get("nc", 4)
    X_tr, X_te = preprocess(X_tr_raw, X_te_raw, groups_train, pp)
    X_tr, X_te = feature_select(X_tr, X_te, y_train, fs)
    return predict_pls(X_tr, X_te, y_train, nc, tf)


def predict_fold_gbr(X_tr_raw, X_te_raw, y_train, groups_train, cfg):
    pp, fs, tf, nc = cfg["pp"], cfg.get("fs"), cfg["tf"], cfg.get("nc", 4)
    X_tr, X_te = preprocess(X_tr_raw, X_te_raw, groups_train, pp)
    X_tr, X_te = feature_select(X_tr, X_te, y_train, fs)
    T_tr, T_te, _, y_fit = get_pls_scores(X_tr, X_te, y_train, nc, tf)
    gbr = GradientBoostingRegressor(
        n_estimators=cfg.get("n_est", 200), max_depth=cfg.get("max_depth", 3),
        learning_rate=cfg.get("lr", 0.05), subsample=0.8,
        min_samples_leaf=5, random_state=42,
        validation_fraction=0.15, n_iter_no_change=50, tol=0.01
    )
    gbr.fit(T_tr, y_fit)
    pred = gbr.predict(T_te)
    if tf == "sqrt":
        pred = np.clip(pred, 0, None) ** 2
    return pred


def predict_fold_gbr_poly(X_tr_raw, X_te_raw, y_train, groups_train, cfg):
    """PLSスコアの多項式特徴量 + GBR"""
    pp, fs, tf, nc = cfg["pp"], cfg.get("fs"), cfg["tf"], cfg.get("nc", 4)
    X_tr, X_te = preprocess(X_tr_raw, X_te_raw, groups_train, pp)
    X_tr, X_te = feature_select(X_tr, X_te, y_train, fs)
    T_tr, T_te, _, y_fit = get_pls_scores(X_tr, X_te, y_train, nc, tf)

    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    T_tr_poly = poly.fit_transform(T_tr)
    T_te_poly = poly.transform(T_te)

    gbr = GradientBoostingRegressor(
        n_estimators=cfg.get("n_est", 200), max_depth=cfg.get("max_depth", 3),
        learning_rate=cfg.get("lr", 0.05), subsample=0.8,
        min_samples_leaf=5, random_state=42,
        validation_fraction=0.15, n_iter_no_change=50, tol=0.01
    )
    gbr.fit(T_tr_poly, y_fit)
    pred = gbr.predict(T_te_poly)
    if tf == "sqrt":
        pred = np.clip(pred, 0, None) ** 2
    return pred


def predict_fold_huber(X_tr_raw, X_te_raw, y_train, groups_train, cfg):
    """PLSスコア + Huber回帰"""
    pp, fs, tf, nc = cfg["pp"], cfg.get("fs"), cfg["tf"], cfg.get("nc", 4)
    X_tr, X_te = preprocess(X_tr_raw, X_te_raw, groups_train, pp)
    X_tr, X_te = feature_select(X_tr, X_te, y_train, fs)
    T_tr, T_te, _, y_fit = get_pls_scores(X_tr, X_te, y_train, nc, tf)

    scaler = StandardScaler()
    T_tr_s = scaler.fit_transform(T_tr)
    T_te_s = scaler.transform(T_te)

    huber = HuberRegressor(epsilon=1.35, max_iter=200, alpha=0.01)
    huber.fit(T_tr_s, y_fit)
    pred = huber.predict(T_te_s)
    if tf == "sqrt":
        pred = np.clip(pred, 0, None) ** 2
    return pred


def predict_fold_enet(X_tr_raw, X_te_raw, y_train, groups_train, cfg):
    """PLSスコアの多項式 + ElasticNet"""
    pp, fs, tf, nc = cfg["pp"], cfg.get("fs"), cfg["tf"], cfg.get("nc", 4)
    X_tr, X_te = preprocess(X_tr_raw, X_te_raw, groups_train, pp)
    X_tr, X_te = feature_select(X_tr, X_te, y_train, fs)
    T_tr, T_te, _, y_fit = get_pls_scores(X_tr, X_te, y_train, nc, tf)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    T_tr_poly = poly.fit_transform(T_tr)
    T_te_poly = poly.transform(T_te)

    scaler = StandardScaler()
    T_tr_s = scaler.fit_transform(T_tr_poly)
    T_te_s = scaler.transform(T_te_poly)

    enet = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=42)
    enet.fit(T_tr_s, y_fit)
    pred = enet.predict(T_te_s)
    if tf == "sqrt":
        pred = np.clip(pred, 0, None) ** 2
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
    print("Issue #84: サイクル8 - PLSスコア拡張特徴量 + Huber回帰")
    print(f"  前ベスト: 14.2173 (Cycle7)")
    print("=" * 70)

    models = [
        # サイクル7ベスト8モデル
        {"name": "M1:EPO+PLS4+sqrt", "pp": "EPO(1)", "nc": 4, "tf": "sqrt", "func": predict_fold_std},
        {"name": "M3:SNV+iPLS50+PLS4+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt", "fs": "iPLS(50)", "func": predict_fold_std},
        {"name": "M5:SNV+AsLS+siPLS+PLS5+sqrt", "pp": "SNV+AsLS(1e6)", "nc": 5, "tf": "sqrt", "fs": "siPLS(30,3)", "func": predict_fold_std},
        {"name": "C1:SNV+iPLS50+PLS5+sqrt", "pp": "SNV", "nc": 5, "tf": "sqrt", "fs": "iPLS(50)", "func": predict_fold_std},
        {"name": "N10:PMSC+siPLS+GBR+sqrt", "pp": "PiecewiseMSC(seg=3)", "nc": 4, "tf": "sqrt", "fs": "siPLS(30,3)", "func": predict_fold_gbr},
        {"name": "G9:SG2d+EPO+GBR+raw", "pp": "SG2d+EPO(1)", "nc": 4, "tf": "raw", "func": predict_fold_gbr},
        {"name": "N2:SNV+SG2d+GBR+raw", "pp": "SNV+SG2d", "nc": 4, "tf": "raw", "func": predict_fold_gbr},
        {"name": "Gnew1:EPO+GBR150_008+raw", "pp": "EPO(1)", "nc": 4, "tf": "raw", "n_est": 150, "lr": 0.08, "func": predict_fold_gbr},

        # 新: PLSスコア多項式 + GBR
        {"name": "P1:EPO+PLS4poly+GBR+raw", "pp": "EPO(1)", "nc": 4, "tf": "raw", "func": predict_fold_gbr_poly},
        {"name": "P2:SNV+PLS4poly+GBR+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt", "func": predict_fold_gbr_poly},
        {"name": "P3:SG2d+EPO+PLS4poly+GBR+raw", "pp": "SG2d+EPO(1)", "nc": 4, "tf": "raw", "func": predict_fold_gbr_poly},

        # 新: Huber回帰
        {"name": "H1:EPO+PLS4+Huber+sqrt", "pp": "EPO(1)", "nc": 4, "tf": "sqrt", "func": predict_fold_huber},
        {"name": "H2:SNV+iPLS50+PLS4+Huber+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt", "fs": "iPLS(50)", "func": predict_fold_huber},
        {"name": "H3:SNV+PLS4+Huber+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt", "func": predict_fold_huber},

        # 新: ElasticNet on poly scores
        {"name": "E1:EPO+PLS4poly+ENet+sqrt", "pp": "EPO(1)", "nc": 4, "tf": "sqrt", "func": predict_fold_enet},
        {"name": "E2:SNV+iPLS50+PLS4poly+ENet+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt", "fs": "iPLS(50)", "func": predict_fold_enet},

        # multi-block PLS
        {"name": "MB1:multi_SNV_EPO+PLS4+sqrt", "pp": "multi_SNV_EPO", "nc": 4, "tf": "sqrt", "func": predict_fold_std},
        {"name": "MB2:multi_SNV_EPO+PLS6+sqrt", "pp": "multi_SNV_EPO", "nc": 6, "tf": "sqrt", "func": predict_fold_std},
    ]

    n_models = len(models)
    all_preds = [[] for _ in range(n_models)]

    print(f"\n--- 個別モデル ({n_models}) ---\n", flush=True)
    for m_idx, cfg in enumerate(models):
        t1 = time.time()
        fold_rmses = []
        for f_idx, (tr, te) in enumerate(folds):
            try:
                pred = cfg["func"](X_raw[tr], X_raw[te], y[tr], groups[tr], cfg)
                all_preds[m_idx].append(pred)
                fold_rmses.append(rmse(y[te], pred))
            except Exception as e:
                all_preds[m_idx].append(np.full(len(te), y[tr].mean()))
                fold_rmses.append(999.0)
                print(f"  ERR {cfg['name']} f{f_idx}: {e}")
        print(f"  [{m_idx+1:>2}/{n_models}] {cfg['name']}: {np.mean(fold_rmses):.2f} ± {np.std(fold_rmses):.2f} ({time.time()-t1:.1f}s)", flush=True)

    # アンサンブル
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

    def opt_w(indices, n_restarts=30):
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

    c7_base = list(range(8))
    w_c7, r_c7 = opt_w(c7_base)
    print(f"\nCycle7-Base: {r_c7:.4f}")

    top14 = [i for _, i in indiv[:14]]
    results = [("Cycle7-Base", r_c7, c7_base, w_c7)]
    for k in [6, 7, 8, 9, 10]:
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
    print(f"前ベスト: 14.2173, 改善: {14.2173 - best[1]:+.4f}")
    print(f"{'='*70}")

    _, fr = eval_ens(best[2], best[3])
    print("\nfold詳細:")
    for sp, r in zip(sp_names, fr):
        tag = " ※" if sp == "ベイスギ" else ""
        print(f"  {sp}: {r:.2f}{tag}")

    rows = [{"method": n, "rmse": r, "models": str([models[i]["name"] for i in c])} for n, r, c, w in results]
    pd.DataFrame(rows).to_csv(OUT_DIR / "issue84_cycle8_results.csv", index=False)
    print(f"\n保存: {OUT_DIR / 'issue84_cycle8_results.csv'}")
    print(f"時間: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
