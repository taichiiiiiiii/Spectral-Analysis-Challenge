"""Issue #81: サイクル5 - GBRパラメータ最適化 + 追加非線形モデル

サイクル4でGBR(PLS4スコア)が有効と判明（BestCombo6で14.8121）。
GBRのハイパーパラメータを最適化し、さらに改善を図る。

1. GBRパラメータグリッド探索
2. 複数の前処理×GBR組み合わせ
3. PLS成分数の多様化（4,6,8）
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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
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


def predict_pls_standard(X_tr, X_te, y_train, nc, tf):
    nc = min(nc, X_tr.shape[1] - 1)
    nc = max(1, nc)
    y_fit = np.sqrt(y_train) if tf == "sqrt" else y_train.copy()
    pls = PLSRegression(n_components=nc)
    pls.fit(X_tr, y_fit)
    pred = pls.predict(X_te).ravel()
    if tf == "sqrt":
        pred = np.clip(pred, 0, None) ** 2
    return pred


def predict_fold_gbr(X_tr_raw, X_te_raw, y_train, groups_train, cfg):
    """PLS + GBR"""
    pp, fs, tf = cfg["pp"], cfg.get("fs"), cfg["tf"]
    nc = cfg.get("nc", 4)
    X_tr, X_te = preprocess(X_tr_raw, X_te_raw, groups_train, pp)
    X_tr, X_te = feature_select(X_tr, X_te, y_train, fs)

    nc_pls = min(nc, X_tr.shape[1] - 1)
    nc_pls = max(1, nc_pls)
    y_fit = np.sqrt(y_train) if tf == "sqrt" else y_train.copy()

    pls = PLSRegression(n_components=nc_pls)
    pls.fit(X_tr, y_fit)
    T_tr = pls.transform(X_tr)
    T_te = pls.transform(X_te)

    gbr = GradientBoostingRegressor(
        n_estimators=cfg.get("n_est", 200),
        max_depth=cfg.get("max_depth", 3),
        learning_rate=cfg.get("lr", 0.05),
        subsample=cfg.get("subsample", 0.8),
        min_samples_leaf=cfg.get("min_leaf", 5),
        random_state=42,
        validation_fraction=0.15,
        n_iter_no_change=50,
        tol=0.01
    )
    gbr.fit(T_tr, y_fit)
    pred = gbr.predict(T_te)

    if tf == "sqrt":
        pred = np.clip(pred, 0, None) ** 2
    return pred


def predict_fold_std(X_tr_raw, X_te_raw, y_train, groups_train, cfg):
    pp, fs, tf = cfg["pp"], cfg.get("fs"), cfg["tf"]
    nc = cfg.get("nc", 4)
    X_tr, X_te = preprocess(X_tr_raw, X_te_raw, groups_train, pp)
    X_tr, X_te = feature_select(X_tr, X_te, y_train, fs)
    return predict_pls_standard(X_tr, X_te, y_train, nc, tf)


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
    print("Issue #81: サイクル5 - GBRパラメータ最適化")
    print(f"  ベースライン: 14.8121 (Cycle4 BestCombo6)")
    print("=" * 70)

    models = [
        # 既存5モデル
        {"name": "M1:EPO(1)+PLS(4)+sqrt", "pp": "EPO(1)", "nc": 4, "tf": "sqrt", "func": predict_fold_std},
        {"name": "M2:SG2d+EPO(1)+PLS(3)+raw", "pp": "SG2d+EPO(1)", "nc": 3, "tf": "raw", "func": predict_fold_std},
        {"name": "M3:SNV+iPLS(50)+PLS(4)+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt", "fs": "iPLS(50)", "func": predict_fold_std},
        {"name": "M4:PMSC+siPLS(30,3)+PLS(4)+sqrt", "pp": "PiecewiseMSC(seg=3)", "nc": 4, "tf": "sqrt", "fs": "siPLS(30,3)", "func": predict_fold_std},
        {"name": "M5:SNV+AsLS+siPLS(30,3)+PLS(5)+sqrt", "pp": "SNV+AsLS(1e6)", "nc": 5, "tf": "sqrt", "fs": "siPLS(30,3)", "func": predict_fold_std},

        # GBRバリエーション - EPO(1)ベース
        {"name": "G1:EPO+PLS4+GBR(200,3,0.05)+sqrt", "pp": "EPO(1)", "nc": 4, "tf": "sqrt",
         "n_est": 200, "max_depth": 3, "lr": 0.05, "subsample": 0.8, "func": predict_fold_gbr},
        {"name": "G2:EPO+PLS4+GBR(300,2,0.03)+sqrt", "pp": "EPO(1)", "nc": 4, "tf": "sqrt",
         "n_est": 300, "max_depth": 2, "lr": 0.03, "subsample": 0.8, "func": predict_fold_gbr},
        {"name": "G3:EPO+PLS4+GBR(500,2,0.01)+sqrt", "pp": "EPO(1)", "nc": 4, "tf": "sqrt",
         "n_est": 500, "max_depth": 2, "lr": 0.01, "subsample": 0.7, "func": predict_fold_gbr},
        {"name": "G4:EPO+PLS6+GBR(200,3,0.05)+sqrt", "pp": "EPO(1)", "nc": 6, "tf": "sqrt",
         "n_est": 200, "max_depth": 3, "lr": 0.05, "subsample": 0.8, "func": predict_fold_gbr},
        {"name": "G5:EPO+PLS8+GBR(200,3,0.05)+sqrt", "pp": "EPO(1)", "nc": 8, "tf": "sqrt",
         "n_est": 200, "max_depth": 3, "lr": 0.05, "subsample": 0.8, "func": predict_fold_gbr},

        # GBR - SNVベース
        {"name": "G6:SNV+PLS4+GBR(200,3,0.05)+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt",
         "n_est": 200, "max_depth": 3, "lr": 0.05, "subsample": 0.8, "func": predict_fold_gbr},
        {"name": "G7:SNV+iPLS(50)+PLS4+GBR(200,3,0.05)+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt",
         "fs": "iPLS(50)", "n_est": 200, "max_depth": 3, "lr": 0.05, "subsample": 0.8, "func": predict_fold_gbr},
        {"name": "G8:SNV+PLS6+GBR(300,2,0.03)+sqrt", "pp": "SNV", "nc": 6, "tf": "sqrt",
         "n_est": 300, "max_depth": 2, "lr": 0.03, "subsample": 0.8, "func": predict_fold_gbr},

        # GBR - SG2d+EPOベース
        {"name": "G9:SG2d+EPO+PLS4+GBR(200,3,0.05)+raw", "pp": "SG2d+EPO(1)", "nc": 4, "tf": "raw",
         "n_est": 200, "max_depth": 3, "lr": 0.05, "subsample": 0.8, "func": predict_fold_gbr},

        # GBR - PMSCベース
        {"name": "G10:PMSC+PLS4+GBR(200,3,0.05)+sqrt", "pp": "PiecewiseMSC(seg=3)", "nc": 4, "tf": "sqrt",
         "n_est": 200, "max_depth": 3, "lr": 0.05, "subsample": 0.8, "func": predict_fold_gbr},

        # GBR raw (no sqrt)
        {"name": "G11:EPO+PLS4+GBR(200,3,0.05)+raw", "pp": "EPO(1)", "nc": 4, "tf": "raw",
         "n_est": 200, "max_depth": 3, "lr": 0.05, "subsample": 0.8, "func": predict_fold_gbr},

        # GBR min_leaf variants
        {"name": "G12:EPO+PLS4+GBR(200,3,0.05,ml10)+sqrt", "pp": "EPO(1)", "nc": 4, "tf": "sqrt",
         "n_est": 200, "max_depth": 3, "lr": 0.05, "subsample": 0.8, "min_leaf": 10, "func": predict_fold_gbr},
    ]

    n_models = len(models)
    all_preds = [[] for _ in range(n_models)]

    print(f"\n--- 個別モデル評価 ({n_models}モデル) ---\n", flush=True)

    for m_idx, cfg in enumerate(models):
        t1 = time.time()
        fold_rmses = []
        for f_idx, (train_idx, test_idx) in enumerate(folds):
            try:
                pred = cfg["func"](X_raw[train_idx], X_raw[test_idx],
                                    y[train_idx], groups[train_idx], cfg)
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

    def opt_w(indices, n_restarts=25):
        n = len(indices)
        def obj(w):
            w_n = np.abs(w) / np.sum(np.abs(w))
            r, _ = eval_ens(indices, w_n)
            return r
        best_r, best_w = np.inf, np.ones(n) / n
        for s in range(n_restarts):
            w0 = np.random.RandomState(s).dirichlet(np.ones(n))
            res = minimize(obj, w0, method="Nelder-Mead", options={"maxiter": 3000})
            if res.fun < best_r:
                best_r = res.fun
                best_w = np.abs(res.x) / np.sum(np.abs(res.x))
        return best_w, best_r

    indiv = sorted([(eval_ens([i])[0], i) for i in range(n_models)])
    print("個別ランキング:")
    for r, i in indiv:
        print(f"  {r:.2f} | {models[i]['name']}")

    base5 = list(range(5))
    w5, r5 = opt_w(base5)
    print(f"\nBaseline(M1-5): {r5:.4f}")

    top12 = [i for _, i in indiv[:12]]
    results = [("Baseline(M1-5)", r5, base5, w5)]

    for k in [5, 6, 7, 8]:
        best_r, best_combo = np.inf, None
        for combo in combinations(top12, k):
            r, _ = eval_ens(list(combo))
            if r < best_r:
                best_r = r
                best_combo = list(combo)
        if best_combo:
            w, r_w = opt_w(best_combo)
            results.append((f"BestCombo{k}-Weighted", r_w, best_combo, w))
            mnames = [models[i]["name"] for i in best_combo]
            print(f"  BestCombo{k}-Weighted: {r_w:.4f}", flush=True)
            for i, wi in zip(best_combo, w):
                if wi > 0.01:
                    print(f"    {wi:.3f}: {models[i]['name']}")

    results.sort(key=lambda x: x[1])
    best = results[0]
    print(f"\n{'='*70}")
    print(f"BEST: {best[1]:.4f} ({best[0]})")
    print(f"前ベスト: 14.8121")
    print(f"改善幅: {14.8121 - best[1]:+.4f}")
    print(f"{'='*70}")

    _, fold_rmses = eval_ens(best[2], best[3])
    print("\nfold詳細:")
    for sp, fr in zip(sp_names, fold_rmses):
        tag = " ※参考" if sp == "ベイスギ" else ""
        print(f"  {sp}: {fr:.2f}{tag}")
    non_bs = [r for sp, r in zip(sp_names, fold_rmses) if sp != "ベイスギ"]
    print(f"  ベイスギ除外平均: {np.mean(non_bs):.4f}")

    rows = [{"method": n, "rmse": r, "models": str([models[i]["name"] for i in c])} for n, r, c, w in results]
    pd.DataFrame(rows).to_csv(OUT_DIR / "issue81_cycle5_results.csv", index=False)
    print(f"\n結果保存: {OUT_DIR / 'issue81_cycle5_results.csv'}")
    print(f"総実行時間: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
