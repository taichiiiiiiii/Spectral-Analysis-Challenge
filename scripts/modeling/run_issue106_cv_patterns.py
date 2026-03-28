"""Issue #106: CVパターン別RMSE比較 - LBとの相関分析

best11 (run_issue87_final.py) と v9_top12 (run_submit_v9_diverse.py) の
fold別RMSEを計算し、異なるfold選択パターンでCV RMSEを算出。
どのパターンがLBスコアと最も相関するかを分析する。

パターン:
  A (生スペクトル類似): ヒノキ, ナラ, 米ヒバ, スプルース, クリ, ホワイトオーク
  B (SNV前処理後類似): ナラ, 米ヒバ, スプルース, ベイマツ, トチ, クリ
  C (含水率類似): ホワイトオーク, スプルース, チェリー, ウォールナット
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import warnings
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
from src.preprocessing.issue19_msc import compute_msc_reference, apply_msc
from src.preprocessing.issue62_additional_preprocessing import apply_asls, apply_piecewise_msc
from src.modeling.issue65_feature_selection import sipls_select, ipls_select

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CV fold パターン定義
# ============================================================
PATTERN_A = {"ヒノキ", "ナラ", "米ヒバ", "スプルース", "クリ", "ホワイトオーク"}
PATTERN_B = {"ナラ", "米ヒバ", "スプルース", "ベイマツ", "トチ", "クリ"}
PATTERN_C = {"ホワイトオーク", "スプルース", "チェリー", "ウォールナット"}

# LBスコア (既知)
LB_SCORES = {
    "best11": 21.56,
    "v9_top12": 17.14,
    "v12_opt12": 21.56,
    "v10_mega": 18.62,
    "v8_blend": 18.11,
}


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ============================================================
# best11 用の関数群 (run_issue87_final.py から)
# ============================================================
def pp_best11(X_tr, X_te, g, pp_name):
    """best11用前処理"""
    if pp_name == "SNV":
        return apply_snv(X_tr), apply_snv(X_te)
    elif pp_name == "EPO(1)":
        P = compute_epo_projection(X_tr, g, n_components=1)
        return apply_epo(X_tr, P), apply_epo(X_te, P)
    elif pp_name == "SNV+AsLS(1e6)":
        return apply_asls(apply_snv(X_tr), lam=1e6), apply_asls(apply_snv(X_te), lam=1e6)
    elif pp_name == "PMSC":
        ref = compute_msc_reference(X_tr)
        return apply_piecewise_msc(X_tr, ref, 3), apply_piecewise_msc(X_te, ref, 3)
    elif pp_name == "SG2d+EPO(1)":
        xs = apply_savgol(X_tr, deriv=2, window_length=7)
        xst = apply_savgol(X_te, deriv=2, window_length=7)
        P = compute_epo_projection(xs, g, n_components=1)
        return apply_epo(xs, P), apply_epo(xst, P)
    elif pp_name == "SNV+SG2d":
        return (
            apply_savgol(apply_snv(X_tr), deriv=2, window_length=7),
            apply_savgol(apply_snv(X_te), deriv=2, window_length=7),
        )
    return X_tr.copy(), X_te.copy()


def fs_best11(X_tr, X_te, y, fs_name):
    """best11用特徴量選択"""
    if not fs_name:
        return X_tr, X_te
    if fs_name.startswith("siPLS"):
        p = fs_name.replace("siPLS(", "").rstrip(")").split(",")
        X_tr, X_te, _ = sipls_select(X_tr, y, X_te, n_intervals=int(p[0]), n_components=3, n_combine=int(p[1]))
    elif fs_name.startswith("iPLS"):
        n = int(fs_name.replace("iPLS(", "").rstrip(")"))
        X_tr, X_te, _ = ipls_select(X_tr, y, X_te, n_intervals=n, n_components=3, n_best=1)
    return X_tr, X_te


def pred_pls(Xtr, Xte, y, nc, tf):
    nc = max(1, min(nc, Xtr.shape[1] - 1))
    yf = np.sqrt(y) if tf == "sqrt" else y.copy()
    pls = PLSRegression(n_components=nc)
    pls.fit(Xtr, yf)
    p = pls.predict(Xte).ravel()
    return np.clip(p, 0, None) ** 2 if tf == "sqrt" else p


def pred_gbr(Xtr, Xte, y, nc, tf, n_est=200, md=3, lr=0.05):
    nc = max(1, min(nc, Xtr.shape[1] - 1))
    yf = np.sqrt(y) if tf == "sqrt" else y.copy()
    pls = PLSRegression(n_components=nc)
    pls.fit(Xtr, yf)
    Ttr, Tte = pls.transform(Xtr), pls.transform(Xte)
    gbr = GradientBoostingRegressor(
        n_estimators=n_est, max_depth=md, learning_rate=lr,
        subsample=0.8, min_samples_leaf=5, random_state=42,
        validation_fraction=0.15, n_iter_no_change=50, tol=0.01,
    )
    gbr.fit(Ttr, yf)
    p = gbr.predict(Tte)
    return np.clip(p, 0, None) ** 2 if tf == "sqrt" else p


def pred_huber(Xtr, Xte, y, nc, tf, eps=1.35):
    nc = max(1, min(nc, Xtr.shape[1] - 1))
    yf = np.sqrt(y) if tf == "sqrt" else y.copy()
    pls = PLSRegression(n_components=nc)
    pls.fit(Xtr, yf)
    Ttr, Tte = pls.transform(Xtr), pls.transform(Xte)
    sc = StandardScaler()
    Ttr_s, Tte_s = sc.fit_transform(Ttr), sc.transform(Tte)
    h = HuberRegressor(epsilon=eps, max_iter=200, alpha=0.01)
    h.fit(Ttr_s, yf)
    p = h.predict(Tte_s)
    return np.clip(p, 0, None) ** 2 if tf == "sqrt" else p


def run_model_best11(X_tr, X_te, y, g, cfg):
    Xtr, Xte = pp_best11(X_tr, X_te, g, cfg["pp"])
    Xtr, Xte = fs_best11(Xtr, Xte, y, cfg.get("fs"))
    t = cfg["type"]
    if t == "pls":
        return pred_pls(Xtr, Xte, y, cfg.get("nc", 4), cfg["tf"])
    elif t == "gbr":
        return pred_gbr(Xtr, Xte, y, cfg.get("nc", 4), cfg["tf"],
                         cfg.get("ne", 200), cfg.get("md", 3), cfg.get("lr", 0.05))
    elif t == "huber":
        return pred_huber(Xtr, Xte, y, cfg.get("nc", 4), cfg["tf"], cfg.get("eps", 1.35))


# ============================================================
# v9 用の前処理関数群 (run_submit_v9_diverse.py から)
# ============================================================
def v9_pp_raw(Xtr, Xte, g_tr):
    return Xtr.copy(), Xte.copy()

def v9_pp_snv(Xtr, Xte, g_tr):
    return apply_snv(Xtr), apply_snv(Xte)

def v9_pp_epo1(Xtr, Xte, g_tr):
    P = compute_epo_projection(Xtr, g_tr, 1)
    return apply_epo(Xtr, P), apply_epo(Xte, P)

def v9_pp_epo2(Xtr, Xte, g_tr):
    P = compute_epo_projection(Xtr, g_tr, 2)
    return apply_epo(Xtr, P), apply_epo(Xte, P)

def v9_pp_snv_epo1(Xtr, Xte, g_tr):
    Xtr_s, Xte_s = apply_snv(Xtr), apply_snv(Xte)
    P = compute_epo_projection(Xtr_s, g_tr, 1)
    return apply_epo(Xtr_s, P), apply_epo(Xte_s, P)

def v9_pp_sg1d(Xtr, Xte, g_tr):
    return (
        apply_savgol(Xtr, deriv=1, window_length=11),
        apply_savgol(Xte, deriv=1, window_length=11),
    )

def v9_pp_sg2d_epo1(Xtr, Xte, g_tr):
    xs = apply_savgol(Xtr, deriv=2, window_length=7)
    xst = apply_savgol(Xte, deriv=2, window_length=7)
    P = compute_epo_projection(xs, g_tr, 1)
    return apply_epo(xs, P), apply_epo(xst, P)

def v9_pp_msc(Xtr, Xte, g_tr):
    ref = compute_msc_reference(Xtr)
    return apply_msc(Xtr, ref), apply_msc(Xte, ref)

def v9_pp_snv_sg1d(Xtr, Xte, g_tr):
    return (
        apply_savgol(apply_snv(Xtr), deriv=1, window_length=11),
        apply_savgol(apply_snv(Xte), deriv=1, window_length=11),
    )

def v9_pp_snv_asls(Xtr, Xte, g_tr):
    return (
        apply_asls(apply_snv(Xtr), lam=1e6),
        apply_asls(apply_snv(Xte), lam=1e6),
    )

def v9_pp_pmsc(Xtr, Xte, g_tr):
    ref = compute_msc_reference(Xtr)
    return (
        apply_piecewise_msc(Xtr, ref, 3),
        apply_piecewise_msc(Xte, ref, 3),
    )

def v9_pp_msc_epo1(Xtr, Xte, g_tr):
    ref = compute_msc_reference(Xtr)
    Xtr_m, Xte_m = apply_msc(Xtr, ref), apply_msc(Xte, ref)
    P = compute_epo_projection(Xtr_m, g_tr, 1)
    return apply_epo(Xtr_m, P), apply_epo(Xte_m, P)


# ============================================================
# fold別RMSE → パターン別CV RMSE 計算
# ============================================================
def compute_fold_rmses(y, folds, species_list, ensemble_preds):
    """各foldのRMSEを計算して辞書で返す"""
    fold_rmses = {}
    for fi, (_, te) in enumerate(folds):
        sp = species_list[fi]
        fold_rmses[sp] = rmse(y[te], ensemble_preds[te])
    return fold_rmses


def compute_pattern_cv(fold_rmses, pattern_species):
    """指定された樹種のfold RMSEの平均を計算"""
    vals = [fold_rmses[sp] for sp in pattern_species if sp in fold_rmses]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def compute_all_patterns(fold_rmses):
    """全パターンのCV RMSEを計算"""
    all_species = set(fold_rmses.keys())
    no_beisugi = all_species - {"ベイスギ"}

    return {
        "全13種 LOSO-CV": np.mean(list(fold_rmses.values())),
        "除ベイスギ LOSO-CV": compute_pattern_cv(fold_rmses, no_beisugi),
        "Pattern A (生スペクトル類似)": compute_pattern_cv(fold_rmses, PATTERN_A),
        "Pattern B (SNV前処理後類似)": compute_pattern_cv(fold_rmses, PATTERN_B),
        "Pattern C (含水率類似)": compute_pattern_cv(fold_rmses, PATTERN_C),
    }


# ============================================================
# best11 パイプライン
# ============================================================
def run_best11_pipeline(X, y, g, folds, species_list):
    """best11のパイプラインを実行し、fold別RMSEを返す"""
    print("=" * 70)
    print("best11 パイプライン (run_issue87_final.py)")
    print("=" * 70)

    cfgs = [
        # Cycle9ベスト9
        {"name": "M1:EPO+PLS4+sqrt", "pp": "EPO(1)", "nc": 4, "tf": "sqrt", "type": "pls"},
        {"name": "M3:SNV+iPLS50+PLS4+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt", "fs": "iPLS(50)", "type": "pls"},
        {"name": "M5:SNV+AsLS+siPLS+PLS5+sqrt", "pp": "SNV+AsLS(1e6)", "nc": 5, "tf": "sqrt", "fs": "siPLS(30,3)", "type": "pls"},
        {"name": "C1:SNV+iPLS50+PLS5+sqrt", "pp": "SNV", "nc": 5, "tf": "sqrt", "fs": "iPLS(50)", "type": "pls"},
        {"name": "N10:PMSC+siPLS+GBR+sqrt", "pp": "PMSC", "nc": 4, "tf": "sqrt", "fs": "siPLS(30,3)", "type": "gbr"},
        {"name": "G9:SG2d+EPO+GBR+raw", "pp": "SG2d+EPO(1)", "nc": 4, "tf": "raw", "type": "gbr"},
        {"name": "N2:SNV+SG2d+GBR+raw", "pp": "SNV+SG2d", "nc": 4, "tf": "raw", "type": "gbr"},
        {"name": "Gnew1:EPO+GBR150+raw", "pp": "EPO(1)", "nc": 4, "tf": "raw", "ne": 150, "lr": 0.08, "type": "gbr"},
        {"name": "H2:SNV+iPLS50+Huber+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt", "fs": "iPLS(50)", "type": "huber"},
        # 新発見
        {"name": "H5:PMSC+siPLS+Huber+sqrt", "pp": "PMSC", "nc": 4, "tf": "sqrt", "fs": "siPLS(30,3)", "type": "huber"},
        {"name": "H2a:SNV+iPLS50+Huber(1.1)+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt", "fs": "iPLS(50)", "eps": 1.1, "type": "huber"},
        {"name": "H6:SNV+AsLS+siPLS+Huber+sqrt", "pp": "SNV+AsLS(1e6)", "nc": 4, "tf": "sqrt", "fs": "siPLS(30,3)", "type": "huber"},
        # 追加
        {"name": "G11:EPO+GBR+raw", "pp": "EPO(1)", "nc": 4, "tf": "raw", "type": "gbr"},
        {"name": "M4:PMSC+siPLS+PLS4+sqrt", "pp": "PMSC", "nc": 4, "tf": "sqrt", "fs": "siPLS(30,3)", "type": "pls"},
        {"name": "M2:SG2d+EPO+PLS3+raw", "pp": "SG2d+EPO(1)", "nc": 3, "tf": "raw", "type": "pls"},
    ]

    n = len(cfgs)
    all_p = [[] for _ in range(n)]

    print(f"\n--- 個別モデル ({n}) ---\n", flush=True)
    for i, c in enumerate(cfgs):
        t1 = time.time()
        fr = []
        for fi, (tr, te) in enumerate(folds):
            try:
                p = run_model_best11(X[tr], X[te], y[tr], g[tr], c)
                all_p[i].append(p)
                fr.append(rmse(y[te], p))
            except Exception as e:
                all_p[i].append(np.full(len(te), y[tr].mean()))
                fr.append(999.0)
                print(f"  ERR {c['name']} f{fi}: {e}")
        print(f"  [{i+1:>2}/{n}] {c['name']}: {np.mean(fr):.2f} ({time.time()-t1:.1f}s)", flush=True)

    # アンサンブル評価関数
    def ev(idx, w=None):
        fr = []
        for fi, (_, te) in enumerate(folds):
            ps = [all_p[m][fi] for m in idx]
            if w is not None:
                wn = np.array(w)
                wn /= wn.sum()
                e = sum(wi * p for wi, p in zip(wn, ps))
            else:
                e = np.mean(ps, axis=0)
            fr.append(rmse(y[te], np.clip(e, 0, 300)))
        return np.mean(fr), fr

    # 重み最適化
    def ow(idx, nr=40):
        nn = len(idx)
        def obj(w):
            wn = np.abs(w) / np.sum(np.abs(w))
            r, _ = ev(idx, wn)
            return r
        br, bw = np.inf, np.ones(nn) / nn
        for s in range(nr):
            w0 = np.random.RandomState(s).dirichlet(np.ones(nn))
            res = minimize(obj, w0, method="Nelder-Mead", options={"maxiter": 5000})
            if res.fun < br:
                br = res.fun
                bw = np.abs(res.x) / np.sum(np.abs(res.x))
        return bw, br

    # 個別ランキング
    iv = sorted([(ev([i])[0], i) for i in range(n)])
    print("\n個別ランキング:")
    for r, i in iv:
        print(f"  {r:.2f} | {cfgs[i]['name']}")

    # ベストK探索 + 重み最適化
    top = min(14, n)
    ti = [i for _, i in iv[:top]]
    res = []
    for k in [8, 9, 10, 11, 12]:
        if k > top:
            break
        br, bc = np.inf, None
        for combo in combinations(ti, k):
            r, _ = ev(list(combo))
            if r < br:
                br = r
                bc = list(combo)
        if bc:
            w, rw = ow(bc)
            res.append((f"Best{k}", rw, bc, w))
            print(f"  Best{k}: {rw:.4f}", flush=True)

    res.sort(key=lambda x: x[1])
    best_name, best_rmse, best_combo, best_weights = res[0]
    print(f"\nBEST ensemble: {best_name} = {best_rmse:.4f}")

    # fold別予測値を構築
    ensemble_preds = np.zeros_like(y, dtype=float)
    wn = best_weights / best_weights.sum()
    for fi, (_, te) in enumerate(folds):
        ps = [all_p[m][fi] for m in best_combo]
        ensemble_preds[te] = np.clip(sum(wi * p for wi, p in zip(wn, ps)), 0, 300)

    # fold別RMSE
    fold_rmses = compute_fold_rmses(y, folds, species_list, ensemble_preds)
    print("\nbest11 fold別RMSE:")
    for sp in species_list:
        marker = " ※" if sp == "ベイスギ" else ""
        print(f"  {sp}: {fold_rmses[sp]:.2f}{marker}")

    return fold_rmses


# ============================================================
# v9 パイプライン
# ============================================================
def run_v9_pipeline(X, y, g, folds, species_list):
    """v9のパイプラインを実行し、fold別RMSEを返す (top-12均等平均)"""
    print("\n" + "=" * 70)
    print("v9_top12 パイプライン (run_submit_v9_diverse.py)")
    print("=" * 70)

    configs = [
        ("EPO1", v9_pp_epo1, [3, 4], ["sqrt"]),
        ("EPO2", v9_pp_epo2, [3, 4], ["sqrt"]),
        ("SNV", v9_pp_snv, [3, 4], ["sqrt"]),
        ("SNV_EPO1", v9_pp_snv_epo1, [3, 4], ["sqrt"]),
        ("SG2d_EPO1", v9_pp_sg2d_epo1, [3], ["raw", "sqrt"]),
        ("SG1d", v9_pp_sg1d, [3, 4], ["sqrt"]),
        ("MSC", v9_pp_msc, [3, 4], ["sqrt"]),
        ("SNV_SG1d", v9_pp_snv_sg1d, [3, 4], ["sqrt"]),
        ("SNV_AsLS", v9_pp_snv_asls, [3, 4], ["sqrt"]),
        ("PMSC", v9_pp_pmsc, [3, 4], ["sqrt"]),
        ("MSC_EPO1", v9_pp_msc_epo1, [3, 4], ["sqrt"]),
        ("Raw", v9_pp_raw, [3, 4], ["sqrt"]),
    ]

    models = []
    cv_preds = []

    def add_model(name, pp_func, nc, tf):
        cv = np.zeros_like(y)
        for fi, (tr, te) in enumerate(folds):
            try:
                Xtr, Xte2 = pp_func(X[tr], X[te], g[tr])
                ytr = np.sqrt(y[tr]) if tf == "sqrt" else y[tr]
                n_comp = max(1, min(nc, Xtr.shape[1] - 1))
                pls = PLSRegression(n_components=n_comp)
                pls.fit(Xtr, ytr)
                p = pls.predict(Xte2).ravel()
                if tf == "sqrt":
                    p = np.clip(p, 0, None) ** 2
                cv[te] = p
            except Exception as e:
                cv[te] = y[tr].mean()
                print(f"    WARN {name} fold {fi}: {e}")

        cv_rmse = rmse(y, cv)
        models.append(name)
        cv_preds.append(cv.copy())
        print(f"  {name}: CV={cv_rmse:.2f}", flush=True)

    for pp_name, pp_func, ncs, tfs in configs:
        for nc in ncs:
            for tf in tfs:
                add_model(f"{pp_name}_PLS{nc}_{tf}", pp_func, nc, tf)

    n_models = len(models)
    print(f"\n合計 {n_models} モデル")

    # ランキング
    ranked = sorted(range(n_models), key=lambda i: rmse(y, cv_preds[i]))
    print("\nCV RMSEランキング Top-15:")
    for rank, i in enumerate(ranked[:15]):
        print(f"  {rank+1}. {models[i]}: {rmse(y, cv_preds[i]):.2f}")

    # Top-12 均等平均
    k = 12
    top_idx = ranked[:k]
    top12_cv = np.mean([cv_preds[i] for i in top_idx], axis=0)
    print(f"\nTop-{k}均等: CV={rmse(y, top12_cv):.2f}")
    print("Top-12 モデル:")
    for i in top_idx:
        print(f"    {models[i]}: CV={rmse(y, cv_preds[i]):.2f}")

    # fold別RMSE
    fold_rmses = compute_fold_rmses(y, folds, species_list, top12_cv)
    print("\nv9_top12 fold別RMSE:")
    for sp in species_list:
        marker = " ※" if sp == "ベイスギ" else ""
        print(f"  {sp}: {fold_rmses[sp]:.2f}{marker}")

    return fold_rmses


# ============================================================
# メイン
# ============================================================
def main():
    t0 = time.time()
    df = load_train(DATA_DIR)
    sc = get_spectral_columns(df)
    X = df[sc].values
    y = df["含水率"].values
    g = df["樹種"].values

    logo = LeaveOneGroupOut()
    folds = list(logo.split(X, y, g))
    species_list = [np.unique(g[te])[0] for _, te in folds]

    print(f"データ: {X.shape}, 樹種: {species_list}")
    print(f"パターンA: {sorted(PATTERN_A)}")
    print(f"パターンB: {sorted(PATTERN_B)}")
    print(f"パターンC: {sorted(PATTERN_C)}")

    # ---- best11 ----
    best11_fold_rmses = run_best11_pipeline(X, y, g, folds, species_list)

    # ---- v9_top12 ----
    v9_fold_rmses = run_v9_pipeline(X, y, g, folds, species_list)

    # ============================================================
    # パターン別CV RMSE 集計
    # ============================================================
    print("\n" + "=" * 70)
    print("パターン別 CV RMSE 比較")
    print("=" * 70)

    # fold別RMSE一覧
    print("\n### fold別RMSE一覧\n")
    print(f"{'樹種':<14} {'best11':>10} {'v9_top12':>10}")
    print("-" * 36)
    for sp in species_list:
        b11 = best11_fold_rmses.get(sp, float("nan"))
        v9 = v9_fold_rmses.get(sp, float("nan"))
        marker = " ※" if sp == "ベイスギ" else ""
        print(f"{sp:<12} {b11:>10.2f} {v9:>10.2f}{marker}")

    # パターン別CV RMSE
    best11_patterns = compute_all_patterns(best11_fold_rmses)
    v9_patterns = compute_all_patterns(v9_fold_rmses)

    submissions = {
        "best11": {"patterns": best11_patterns, "lb": LB_SCORES["best11"]},
        "v9_top12": {"patterns": v9_patterns, "lb": LB_SCORES["v9_top12"]},
    }

    # Markdown テーブル出力
    print("\n### パターン別CV RMSE vs LB\n")
    pattern_names = list(best11_patterns.keys())
    header = f"| {'提出':>12} | {'LB':>6} |"
    for pn in pattern_names:
        short = pn.replace(" LOSO-CV", "").replace("LOSO-CV", "")
        header += f" {short:>16} |"
    print(header)
    sep = "|" + "-" * 14 + "|" + "-" * 8 + "|"
    for _ in pattern_names:
        sep += "-" * 18 + "|"
    print(sep)

    for sub_name, info in submissions.items():
        row = f"| {sub_name:>12} | {info['lb']:>6.2f} |"
        for pn in pattern_names:
            row += f" {info['patterns'][pn]:>16.2f} |"
        print(row)

    # 追加: 既知LBとの対応表（参考情報）
    print("\n### 既知LBスコア一覧 (参考)\n")
    print(f"| {'提出':>14} | {'LB':>6} |")
    print("|" + "-" * 16 + "|" + "-" * 8 + "|")
    for name, lb in sorted(LB_SCORES.items(), key=lambda x: x[1]):
        print(f"| {name:>14} | {lb:>6.2f} |")

    # 相関分析: 各パターンのCV RMSEとLBの差分
    print("\n### CV RMSE - LB 差分 (正=CVがLBより悪い, 負=CVがLBより良い)\n")
    header2 = f"| {'提出':>12} | {'LB':>6} |"
    for pn in pattern_names:
        short = pn.replace(" LOSO-CV", "").replace("LOSO-CV", "")
        header2 += f" {short:>16} |"
    print(header2)
    sep2 = "|" + "-" * 14 + "|" + "-" * 8 + "|"
    for _ in pattern_names:
        sep2 += "-" * 18 + "|"
    print(sep2)

    for sub_name, info in submissions.items():
        row = f"| {sub_name:>12} | {info['lb']:>6.2f} |"
        for pn in pattern_names:
            diff = info["patterns"][pn] - info["lb"]
            row += f" {diff:>+16.2f} |"
        print(row)

    # パターン間の順序整合性チェック
    print("\n### LBとの順序整合性チェック")
    print("(best11のLBがv9_top12のLBより高い→CVも高いか?)\n")
    lb_order = LB_SCORES["best11"] > LB_SCORES["v9_top12"]  # True: best11がLBで劣る
    print(f"LB: best11({LB_SCORES['best11']:.2f}) > v9_top12({LB_SCORES['v9_top12']:.2f}) → best11がLBで劣る")

    for pn in pattern_names:
        b11_cv = best11_patterns[pn]
        v9_cv = v9_patterns[pn]
        cv_order = b11_cv > v9_cv  # True: best11がCVで劣る
        match = "一致" if cv_order == lb_order else "不一致"
        print(f"  {pn}: best11={b11_cv:.2f}, v9={v9_cv:.2f} → {match}")

    print(f"\n合計時間: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
