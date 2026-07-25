"""Issue #80: サイクル4 - PLSスコアベース非線形モデル + Ridgeメタラーナー

PLSの線形枠内では改善限界（14.963）。非線形関係を活用:
1. PLSスコア + SVR（RBFカーネル）
2. PLSスコア + KernelRidge
3. PLSスコア + RandomForest
4. PLSスコア + GradientBoosting（sklearn）
5. Ridgeメタラーナー: 既存5予測値 + PLSスコア統計量を特徴量
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
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
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


def get_pls_scores(X_tr, X_te, y_train, nc, tf):
    """PLSスコアを取得（標準化済み）"""
    nc = min(nc, X_tr.shape[1] - 1, max(1, nc))
    y_fit = np.sqrt(y_train) if tf == "sqrt" else y_train.copy()
    pls = PLSRegression(n_components=nc)
    pls.fit(X_tr, y_fit)
    T_tr = pls.transform(X_tr)
    T_te = pls.transform(X_te)
    return T_tr, T_te, pls


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


def predict_fold_pls_svr(X_tr_raw, X_te_raw, y_train, groups_train, cfg):
    """PLS + SVR: PLSスコアを入力にSVR"""
    pp, fs, tf, nc = cfg["pp"], cfg.get("fs"), cfg["tf"], cfg.get("nc", 4)
    X_tr, X_te = preprocess(X_tr_raw, X_te_raw, groups_train, pp)
    X_tr, X_te = feature_select(X_tr, X_te, y_train, fs)

    T_tr, T_te, _ = get_pls_scores(X_tr, X_te, y_train, nc, tf)
    y_fit = np.sqrt(y_train) if tf == "sqrt" else y_train.copy()

    scaler = StandardScaler()
    T_tr_s = scaler.fit_transform(T_tr)
    T_te_s = scaler.transform(T_te)

    svr = SVR(kernel="rbf", C=cfg.get("C", 100), gamma="scale", epsilon=0.1)
    svr.fit(T_tr_s, y_fit)
    pred = svr.predict(T_te_s)

    if tf == "sqrt":
        pred = np.clip(pred, 0, None) ** 2
    return pred


def predict_fold_pls_kr(X_tr_raw, X_te_raw, y_train, groups_train, cfg):
    """PLS + KernelRidge"""
    pp, fs, tf, nc = cfg["pp"], cfg.get("fs"), cfg["tf"], cfg.get("nc", 4)
    X_tr, X_te = preprocess(X_tr_raw, X_te_raw, groups_train, pp)
    X_tr, X_te = feature_select(X_tr, X_te, y_train, fs)

    T_tr, T_te, _ = get_pls_scores(X_tr, X_te, y_train, nc, tf)
    y_fit = np.sqrt(y_train) if tf == "sqrt" else y_train.copy()

    scaler = StandardScaler()
    T_tr_s = scaler.fit_transform(T_tr)
    T_te_s = scaler.transform(T_te)

    kr = KernelRidge(alpha=1.0, kernel="rbf", gamma=0.5)
    kr.fit(T_tr_s, y_fit)
    pred = kr.predict(T_te_s)

    if tf == "sqrt":
        pred = np.clip(pred, 0, None) ** 2
    return pred


def predict_fold_pls_gbr(X_tr_raw, X_te_raw, y_train, groups_train, cfg):
    """PLS + GradientBoosting"""
    pp, fs, tf, nc = cfg["pp"], cfg.get("fs"), cfg["tf"], cfg.get("nc", 4)
    X_tr, X_te = preprocess(X_tr_raw, X_te_raw, groups_train, pp)
    X_tr, X_te = feature_select(X_tr, X_te, y_train, fs)

    T_tr, T_te, _ = get_pls_scores(X_tr, X_te, y_train, nc, tf)
    y_fit = np.sqrt(y_train) if tf == "sqrt" else y_train.copy()

    gbr = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42,
        validation_fraction=0.15, n_iter_no_change=50, tol=0.01
    )
    gbr.fit(T_tr, y_fit)
    pred = gbr.predict(T_te)

    if tf == "sqrt":
        pred = np.clip(pred, 0, None) ** 2
    return pred


def predict_fold_pls_rf(X_tr_raw, X_te_raw, y_train, groups_train, cfg):
    """PLS + RandomForest"""
    pp, fs, tf, nc = cfg["pp"], cfg.get("fs"), cfg["tf"], cfg.get("nc", 4)
    X_tr, X_te = preprocess(X_tr_raw, X_te_raw, groups_train, pp)
    X_tr, X_te = feature_select(X_tr, X_te, y_train, fs)

    T_tr, T_te, _ = get_pls_scores(X_tr, X_te, y_train, nc, tf)
    y_fit = np.sqrt(y_train) if tf == "sqrt" else y_train.copy()

    rf = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=42)
    rf.fit(T_tr, y_fit)
    pred = rf.predict(T_te)

    if tf == "sqrt":
        pred = np.clip(pred, 0, None) ** 2
    return pred


def predict_fold_standard(X_tr_raw, X_te_raw, y_train, groups_train, cfg):
    pp, fs, tf, nc = cfg["pp"], cfg.get("fs"), cfg["tf"], cfg.get("nc", 4)
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
    species_names = [np.unique(groups[te])[0] for _, te in folds]

    print("=" * 70)
    print("Issue #80: サイクル4 - PLSスコアベース非線形モデル")
    print(f"  ベースライン: 14.963 (BestCombo5-Weighted)")
    print("=" * 70)

    models = [
        # 既存5モデル
        {"name": "M1:EPO(1)+PLS(4)+sqrt", "pp": "EPO(1)", "nc": 4, "tf": "sqrt", "func": predict_fold_standard},
        {"name": "M2:SG2d+EPO(1)+PLS(3)+raw", "pp": "SG2d+EPO(1)", "nc": 3, "tf": "raw", "func": predict_fold_standard},
        {"name": "M3:SNV+iPLS(50)+PLS(4)+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt", "fs": "iPLS(50)", "func": predict_fold_standard},
        {"name": "M4:PMSC+siPLS(30,3)+PLS(4)+sqrt", "pp": "PiecewiseMSC(seg=3)", "nc": 4, "tf": "sqrt", "fs": "siPLS(30,3)", "func": predict_fold_standard},
        {"name": "M5:SNV+AsLS+siPLS(30,3)+PLS(5)+sqrt", "pp": "SNV+AsLS(1e6)", "nc": 5, "tf": "sqrt", "fs": "siPLS(30,3)", "func": predict_fold_standard},

        # PLS + SVR
        {"name": "N1:EPO(1)+PLS4+SVR+sqrt", "pp": "EPO(1)", "nc": 4, "tf": "sqrt", "C": 100, "func": predict_fold_pls_svr},
        {"name": "N2:SNV+iPLS(50)+PLS4+SVR+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt", "fs": "iPLS(50)", "C": 100, "func": predict_fold_pls_svr},
        {"name": "N3:EPO(1)+PLS4+SVR+raw", "pp": "EPO(1)", "nc": 4, "tf": "raw", "C": 500, "func": predict_fold_pls_svr},

        # PLS + KernelRidge
        {"name": "N4:EPO(1)+PLS4+KR+sqrt", "pp": "EPO(1)", "nc": 4, "tf": "sqrt", "func": predict_fold_pls_kr},
        {"name": "N5:SNV+iPLS(50)+PLS4+KR+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt", "fs": "iPLS(50)", "func": predict_fold_pls_kr},

        # PLS + GradientBoosting
        {"name": "N6:EPO(1)+PLS4+GBR+sqrt", "pp": "EPO(1)", "nc": 4, "tf": "sqrt", "func": predict_fold_pls_gbr},
        {"name": "N7:SNV+iPLS(50)+PLS4+GBR+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt", "fs": "iPLS(50)", "func": predict_fold_pls_gbr},

        # PLS + RandomForest
        {"name": "N8:EPO(1)+PLS4+RF+sqrt", "pp": "EPO(1)", "nc": 4, "tf": "sqrt", "func": predict_fold_pls_rf},

        # PLS多成分 + SVR（より多い情報を活用）
        {"name": "N9:EPO(1)+PLS8+SVR+sqrt", "pp": "EPO(1)", "nc": 8, "tf": "sqrt", "C": 100, "func": predict_fold_pls_svr},
        {"name": "N10:SNV+PLS8+SVR+sqrt", "pp": "SNV", "nc": 8, "tf": "sqrt", "C": 100, "func": predict_fold_pls_svr},
    ]

    n_models = len(models)
    all_preds = [[] for _ in range(n_models)]

    print(f"\n--- 個別モデル評価 ({n_models}モデル) ---\n", flush=True)

    for m_idx, cfg in enumerate(models):
        t1 = time.time()
        fold_rmses = []
        for f_idx, (train_idx, test_idx) in enumerate(folds):
            try:
                pred = cfg["func"](
                    X_raw[train_idx], X_raw[test_idx],
                    y[train_idx], groups[train_idx], cfg
                )
                all_preds[m_idx].append(pred)
                fold_rmses.append(rmse(y[test_idx], pred))
            except Exception as e:
                fallback = np.full(len(test_idx), y[train_idx].mean())
                all_preds[m_idx].append(fallback)
                fold_rmses.append(999.0)
                print(f"  ERROR {cfg['name']} fold {f_idx}: {e}")
        mean_r = np.mean(fold_rmses)
        print(f"  [{m_idx+1:>2}/{n_models}] {cfg['name']}: {mean_r:.2f} ± {np.std(fold_rmses):.2f} ({time.time()-t1:.1f}s)", flush=True)

    # アンサンブル
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

    def opt_w(indices, n_restarts=20):
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

    # 個別ランキング
    indiv = []
    for m_idx in range(n_models):
        r, _ = eval_ens([m_idx])
        indiv.append((r, m_idx))
    indiv.sort()

    print("個別モデル (fold-RMSE):")
    for r, i in indiv:
        print(f"  {r:.2f} | {models[i]['name']}")

    # ベースライン
    base5 = list(range(5))
    w5, r5 = opt_w(base5)
    print(f"\nBaseline(M1-5): {r5:.4f}")

    # 組み合わせ探索
    top12 = [i for _, i in indiv[:12]]
    results = [("Baseline(M1-5)", r5, base5, w5)]

    for k in [4, 5, 6, 7, 8]:
        best_r, best_combo = np.inf, None
        for combo in combinations(top12, k):
            r, _ = eval_ens(list(combo))
            if r < best_r:
                best_r = r
                best_combo = list(combo)
        if best_combo:
            w, r_w = opt_w(best_combo)
            results.append((f"BestCombo{k}-Weighted", r_w, best_combo, w))
            print(f"  BestCombo{k}-Weighted: {r_w:.4f} | {[models[i]['name'] for i in best_combo]}", flush=True)

    results.sort(key=lambda x: x[1])

    best = results[0]
    print(f"\n{'='*70}")
    print(f"BEST: {best[1]:.4f} ({best[0]})")
    print(f"ベースライン: 14.963")
    print(f"改善幅: {14.963 - best[1]:+.4f}")
    print(f"{'='*70}")

    _, fold_rmses = eval_ens(best[2], best[3])
    print("\nfold詳細:")
    for sp, fr in zip(species_names, fold_rmses):
        tag = " ※参考" if sp == "ベイスギ" else ""
        print(f"  {sp}: {fr:.2f}{tag}")

    # 残差相関
    print("\n残差相関（新モデル vs M1-M5）:")
    residuals = np.zeros((n_models, len(y)))
    for m_idx in range(n_models):
        for f_idx, (_, test_idx) in enumerate(folds):
            residuals[m_idx, test_idx] = y[test_idx] - all_preds[m_idx][f_idx]
    corr = np.corrcoef(residuals)
    for i in range(5, n_models):
        corrs_base = [corr[i, j] for j in range(5)]
        print(f"  {models[i]['name']}: avg={np.mean(corrs_base):.3f}, min={min(corrs_base):.3f}")

    rows = [{"method": n, "rmse": r, "models": str([models[i]["name"] for i in c])} for n, r, c, w in results]
    pd.DataFrame(rows).to_csv(OUT_DIR / "issue80_cycle4_results.csv", index=False)
    print(f"\n結果保存: {OUT_DIR / 'issue80_cycle4_results.csv'}")
    print(f"総実行時間: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
