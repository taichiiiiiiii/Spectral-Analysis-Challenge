"""Issue #83: サイクル7 - チェリー対策 + 水分吸収帯活用

チェリーfold RMSE=21.71が最大のボトルネック。
PLS成分3のバイアス回避と、水分吸収帯の物理的特徴を活用。

1. PLS成分2モデル（成分3バイアス回避）
2. 水分吸収帯（5200, 6900 cm-1）限定のPLS
3. 多様なGBR組み合わせの追加
4. AdaBoost + PLS
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
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
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
        X_tr_s = apply_snv(X_tr_raw)
        X_te_s = apply_snv(X_te_raw)
        return apply_savgol(X_tr_s, deriv=2, window_length=7), apply_savgol(X_te_s, deriv=2, window_length=7)
    return X_tr_raw.copy(), X_te_raw.copy()


def feature_select(X_tr, X_te, y_train, fs, wavenumbers=None):
    if not fs:
        return X_tr, X_te
    if fs.startswith("siPLS"):
        parts = fs.replace("siPLS(", "").rstrip(")").split(",")
        ni, nc = int(parts[0]), int(parts[1])
        X_tr, X_te, _ = sipls_select(X_tr, y_train, X_te, n_intervals=ni, n_components=3, n_combine=nc)
    elif fs.startswith("iPLS"):
        ni = int(fs.replace("iPLS(", "").rstrip(")"))
        X_tr, X_te, _ = ipls_select(X_tr, y_train, X_te, n_intervals=ni, n_components=3, n_best=1)
    elif fs == "water_bands" and wavenumbers is not None:
        # 水分吸収帯: 5000-5400 cm-1 と 6700-7100 cm-1
        wn = np.array([float(w) for w in wavenumbers])
        mask = ((wn >= 5000) & (wn <= 5400)) | ((wn >= 6700) & (wn <= 7100))
        X_tr = X_tr[:, mask]
        X_te = X_te[:, mask]
    elif fs == "water_bands_extended" and wavenumbers is not None:
        # 水分吸収帯拡張: 4800-5600 cm-1 と 6500-7300 cm-1
        wn = np.array([float(w) for w in wavenumbers])
        mask = ((wn >= 4800) & (wn <= 5600)) | ((wn >= 6500) & (wn <= 7300))
        X_tr = X_tr[:, mask]
        X_te = X_te[:, mask]
    return X_tr, X_te


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


def predict_fold_std(X_tr_raw, X_te_raw, y_train, groups_train, cfg, wn=None):
    pp, fs, tf, nc = cfg["pp"], cfg.get("fs"), cfg["tf"], cfg.get("nc", 4)
    X_tr, X_te = preprocess(X_tr_raw, X_te_raw, groups_train, pp)
    X_tr, X_te = feature_select(X_tr, X_te, y_train, fs, wn)
    return predict_pls(X_tr, X_te, y_train, nc, tf)


def predict_fold_gbr(X_tr_raw, X_te_raw, y_train, groups_train, cfg, wn=None):
    pp, fs, tf = cfg["pp"], cfg.get("fs"), cfg["tf"]
    nc = cfg.get("nc", 4)
    X_tr, X_te = preprocess(X_tr_raw, X_te_raw, groups_train, pp)
    X_tr, X_te = feature_select(X_tr, X_te, y_train, fs, wn)

    nc_pls = min(nc, X_tr.shape[1] - 1)
    nc_pls = max(1, nc_pls)
    y_fit = np.sqrt(y_train) if tf == "sqrt" else y_train.copy()

    pls = PLSRegression(n_components=nc_pls)
    pls.fit(X_tr, y_fit)
    T_tr = pls.transform(X_tr)
    T_te = pls.transform(X_te)

    gbr = GradientBoostingRegressor(
        n_estimators=cfg.get("n_est", 200), max_depth=cfg.get("max_depth", 3),
        learning_rate=cfg.get("lr", 0.05), subsample=cfg.get("subsample", 0.8),
        min_samples_leaf=cfg.get("min_leaf", 5), random_state=42,
        validation_fraction=0.15, n_iter_no_change=50, tol=0.01
    )
    gbr.fit(T_tr, y_fit)
    pred = gbr.predict(T_te)

    if tf == "sqrt":
        pred = np.clip(pred, 0, None) ** 2
    return pred


def predict_fold_ada(X_tr_raw, X_te_raw, y_train, groups_train, cfg, wn=None):
    """AdaBoost + PLS"""
    pp, fs, tf = cfg["pp"], cfg.get("fs"), cfg["tf"]
    nc = cfg.get("nc", 4)
    X_tr, X_te = preprocess(X_tr_raw, X_te_raw, groups_train, pp)
    X_tr, X_te = feature_select(X_tr, X_te, y_train, fs, wn)

    nc_pls = min(nc, X_tr.shape[1] - 1)
    nc_pls = max(1, nc_pls)
    y_fit = np.sqrt(y_train) if tf == "sqrt" else y_train.copy()

    pls = PLSRegression(n_components=nc_pls)
    pls.fit(X_tr, y_fit)
    T_tr = pls.transform(X_tr)
    T_te = pls.transform(X_te)

    base = DecisionTreeRegressor(max_depth=3)
    ada = AdaBoostRegressor(estimator=base, n_estimators=100, learning_rate=0.05, random_state=42)
    ada.fit(T_tr, y_fit)
    pred = ada.predict(T_te)

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
    wavenumbers = sc

    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_raw, y, groups))
    sp_names = [np.unique(groups[te])[0] for _, te in folds]

    print("=" * 70)
    print("Issue #83: サイクル7 - チェリー対策 + 水分吸収帯活用")
    print(f"  前ベスト: 14.2751 (Cycle6 BestCombo8)")
    print("=" * 70)

    models = [
        # サイクル6ベスト8モデル
        {"name": "M1:EPO+PLS4+sqrt", "pp": "EPO(1)", "nc": 4, "tf": "sqrt", "type": "std"},
        {"name": "M3:SNV+iPLS50+PLS4+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt", "fs": "iPLS(50)", "type": "std"},
        {"name": "M4:PMSC+siPLS+PLS4+sqrt", "pp": "PiecewiseMSC(seg=3)", "nc": 4, "tf": "sqrt", "fs": "siPLS(30,3)", "type": "std"},
        {"name": "M5:SNV+AsLS+siPLS+PLS5+sqrt", "pp": "SNV+AsLS(1e6)", "nc": 5, "tf": "sqrt", "fs": "siPLS(30,3)", "type": "std"},
        {"name": "C1:SNV+iPLS50+PLS5+sqrt", "pp": "SNV", "nc": 5, "tf": "sqrt", "fs": "iPLS(50)", "type": "std"},
        {"name": "G9:SG2d+EPO+PLS4+GBR+raw", "pp": "SG2d+EPO(1)", "nc": 4, "tf": "raw", "type": "gbr"},
        {"name": "N10:PMSC+siPLS+PLS4+GBR+sqrt", "pp": "PiecewiseMSC(seg=3)", "nc": 4, "tf": "sqrt", "fs": "siPLS(30,3)", "type": "gbr"},
        {"name": "G11:EPO+PLS4+GBR+raw", "pp": "EPO(1)", "nc": 4, "tf": "raw", "type": "gbr"},
        {"name": "N2:SNV+SG2d+PLS4+GBR+raw", "pp": "SNV+SG2d", "nc": 4, "tf": "raw", "type": "gbr"},

        # 新モデル: PLS成分2（チェリーの成分3バイアス回避）
        {"name": "W1:EPO+PLS2+sqrt", "pp": "EPO(1)", "nc": 2, "tf": "sqrt", "type": "std"},
        {"name": "W2:SNV+PLS2+sqrt", "pp": "SNV", "nc": 2, "tf": "sqrt", "type": "std"},

        # 水分吸収帯限定PLS
        {"name": "W3:SNV+waterBand+PLS4+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt", "fs": "water_bands", "type": "std"},
        {"name": "W4:SNV+waterBandExt+PLS4+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt", "fs": "water_bands_extended", "type": "std"},
        {"name": "W5:EPO+waterBand+PLS3+sqrt", "pp": "EPO(1)", "nc": 3, "tf": "sqrt", "fs": "water_bands", "type": "std"},

        # 水分吸収帯 + GBR
        {"name": "W6:SNV+waterBandExt+PLS4+GBR+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt", "fs": "water_bands_extended", "type": "gbr"},

        # AdaBoost
        {"name": "A1:EPO+PLS4+Ada+raw", "pp": "EPO(1)", "nc": 4, "tf": "raw", "type": "ada"},
        {"name": "A2:SNV+PLS4+Ada+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt", "type": "ada"},

        # GBRバリエーション追加
        {"name": "G_new1:EPO+PLS4+GBR150_3_008+raw", "pp": "EPO(1)", "nc": 4, "tf": "raw",
         "n_est": 150, "max_depth": 3, "lr": 0.08, "type": "gbr"},
        {"name": "G_new2:SNV+PLS4+GBR200_3_005+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt",
         "n_est": 200, "max_depth": 3, "lr": 0.05, "type": "gbr"},
    ]

    n_models = len(models)
    all_preds = [[] for _ in range(n_models)]

    print(f"\n--- 個別モデル評価 ({n_models}モデル) ---\n", flush=True)

    for m_idx, cfg in enumerate(models):
        t1 = time.time()
        fold_rmses = []
        model_type = cfg.get("type", "std")
        for f_idx, (train_idx, test_idx) in enumerate(folds):
            try:
                if model_type == "gbr":
                    pred = predict_fold_gbr(X_raw[train_idx], X_raw[test_idx],
                                            y[train_idx], groups[train_idx], cfg, wavenumbers)
                elif model_type == "ada":
                    pred = predict_fold_ada(X_raw[train_idx], X_raw[test_idx],
                                            y[train_idx], groups[train_idx], cfg, wavenumbers)
                else:
                    pred = predict_fold_std(X_raw[train_idx], X_raw[test_idx],
                                            y[train_idx], groups[train_idx], cfg, wavenumbers)
                all_preds[m_idx].append(pred)
                fold_rmses.append(rmse(y[test_idx], pred))
            except Exception as e:
                fallback = np.full(len(test_idx), y[train_idx].mean())
                all_preds[m_idx].append(fallback)
                fold_rmses.append(999.0)
                print(f"  ERROR {cfg['name']} fold {f_idx}: {e}")
        mean_r = np.mean(fold_rmses)
        print(f"  [{m_idx+1:>2}/{n_models}] {cfg['name']}: {mean_r:.2f} ± {np.std(fold_rmses):.2f} ({time.time()-t1:.1f}s)", flush=True)

    # アンサンブル最適化
    print("\n--- アンサンブル最適化 ---\n", flush=True)

    def eval_ens(indices, weights=None):
        fold_rmses = []
        for f_idx, (_, test_idx) in enumerate(folds):
            preds = [all_preds[m][f_idx] for m in indices]
            if weights is not None:
                w = np.array(weights); w = w / w.sum()
                ens = sum(wi * p for wi, p in zip(w, preds))
            else:
                ens = np.mean(preds, axis=0)
            ens = np.clip(ens, 0, 300)
            fold_rmses.append(rmse(y[test_idx], ens))
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
    print("個別ランキング:")
    for r, i in indiv:
        print(f"  {r:.2f} | {models[i]['name']}")

    # Cycle6ベース（9モデル）
    c6_base = list(range(9))
    w_c6, r_c6 = opt_w(c6_base)
    print(f"\nCycle6-Base(9モデル): {r_c6:.4f}")

    top15 = [i for _, i in indiv[:15]]
    results = [("Cycle6-Base", r_c6, c6_base, w_c6)]

    for k in [6, 7, 8, 9, 10]:
        best_r, best_combo = np.inf, None
        for combo in combinations(top15, k):
            r, _ = eval_ens(list(combo))
            if r < best_r:
                best_r = r
                best_combo = list(combo)
        if best_combo:
            w, r_w = opt_w(best_combo)
            results.append((f"BestCombo{k}-Weighted", r_w, best_combo, w))
            print(f"  BestCombo{k}-Weighted: {r_w:.4f}", flush=True)
            for i, wi in zip(best_combo, w):
                if wi > 0.02:
                    print(f"    {wi:.3f}: {models[i]['name']}")

    results.sort(key=lambda x: x[1])
    best = results[0]
    print(f"\n{'='*70}")
    print(f"BEST: {best[1]:.4f} ({best[0]})")
    print(f"前ベスト: 14.2751")
    print(f"改善幅: {14.2751 - best[1]:+.4f}")
    print(f"{'='*70}")

    _, fold_rmses = eval_ens(best[2], best[3])
    print("\nfold詳細:")
    for sp, fr in zip(sp_names, fold_rmses):
        tag = " ※参考" if sp == "ベイスギ" else ""
        print(f"  {sp}: {fr:.2f}{tag}")
    non_bs = [r for sp, r in zip(sp_names, fold_rmses) if sp != "ベイスギ"]
    print(f"  ベイスギ除外平均: {np.mean(non_bs):.4f}")

    rows = [{"method": n, "rmse": r, "models": str([models[i]["name"] for i in c])} for n, r, c, w in results]
    pd.DataFrame(rows).to_csv(OUT_DIR / "issue83_cycle7_results.csv", index=False)
    print(f"\n結果保存: {OUT_DIR / 'issue83_cycle7_results.csv'}")
    print(f"総実行時間: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
