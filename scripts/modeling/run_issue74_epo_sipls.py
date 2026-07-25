"""Issue #74: EPO+siPLS/iPLS組み合わせ評価 & v6アンサンブル構築

Step 1: EPO/SNV/PiecewiseMSC/MSC × siPLS/iPLS × PLS の新規組み合わせをLOSO-CVで評価
Step 2: v5オリジナル4モデル + 新規トップ + 追加多様モデルで最適アンサンブル探索
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
from src.preprocessing.issue19_msc import compute_msc_reference, apply_msc
from src.preprocessing.issue62_additional_preprocessing import apply_asls, apply_piecewise_msc
from src.modeling.issue65_feature_selection import sipls_select, ipls_select

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def calc_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ============================================================
# 前処理関数（fold単位キャッシュ用）
# ============================================================

def pp_raw(X_tr, X_te, groups_tr, **kw):
    return X_tr.copy(), X_te.copy()

def pp_snv(X_tr, X_te, groups_tr, **kw):
    return apply_snv(X_tr), apply_snv(X_te)

def pp_epo1(X_tr, X_te, groups_tr, **kw):
    P = compute_epo_projection(X_tr, groups_tr, n_components=1)
    return apply_epo(X_tr, P), apply_epo(X_te, P)

def pp_sg2d_epo1(X_tr, X_te, groups_tr, **kw):
    X_tr_sg = apply_savgol(X_tr, deriv=2, window_length=7)
    X_te_sg = apply_savgol(X_te, deriv=2, window_length=7)
    P = compute_epo_projection(X_tr_sg, groups_tr, n_components=1)
    return apply_epo(X_tr_sg, P), apply_epo(X_te_sg, P)

def pp_snv_asls(X_tr, X_te, groups_tr, **kw):
    return apply_asls(apply_snv(X_tr), lam=1e6), apply_asls(apply_snv(X_te), lam=1e6)

def pp_piecewise_msc(X_tr, X_te, groups_tr, **kw):
    ref = compute_msc_reference(X_tr)
    return apply_piecewise_msc(X_tr, ref, 3), apply_piecewise_msc(X_te, ref, 3)

def pp_msc(X_tr, X_te, groups_tr, **kw):
    ref = compute_msc_reference(X_tr)
    return apply_msc(X_tr, ref), apply_msc(X_te, ref)


PP_REGISTRY = {
    "Raw": pp_raw,
    "SNV": pp_snv,
    "EPO(1)": pp_epo1,
    "SG2d+EPO(1)": pp_sg2d_epo1,
    "SNV+AsLS(1e6)": pp_snv_asls,
    "PiecewiseMSC": pp_piecewise_msc,
    "MSC": pp_msc,
}


# ============================================================
# 特徴量選択
# ============================================================

def fs_none(X_tr, y_tr, X_te, **kw):
    return X_tr, X_te, np.arange(X_tr.shape[1])

def fs_sipls(X_tr, y_tr, X_te, n_intervals=30, n_combine=3, **kw):
    return sipls_select(X_tr, y_tr, X_te, n_intervals=n_intervals, n_components=3, n_combine=n_combine)

def fs_ipls(X_tr, y_tr, X_te, n_intervals=50, **kw):
    return ipls_select(X_tr, y_tr, X_te, n_intervals=n_intervals, n_components=3, n_best=1)


# ============================================================
# モデルフィッティング
# ============================================================

def fit_predict(X_tr, y_fit, X_te, model_name):
    if model_name.startswith("PLS("):
        nc = int(model_name.split("(")[1].rstrip(")"))
        nc = min(nc, X_tr.shape[1] - 1, X_tr.shape[0] - 1)
        nc = max(1, nc)
        m = PLSRegression(n_components=nc)
        m.fit(X_tr, y_fit)
        return m.predict(X_te).ravel()
    elif model_name == "Lasso":
        m = Lasso(alpha=0.1, max_iter=10000)
        m.fit(X_tr, y_fit)
        return m.predict(X_te)
    elif model_name == "ElasticNet":
        m = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
        m.fit(X_tr, y_fit)
        return m.predict(X_te)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ============================================================
# LOSO-CVでfold予測を収集
# ============================================================

def collect_fold_preds(X_raw, y, groups, folds, config):
    """1つのモデル設定でLOSO全foldの予測を収集する。

    Returns: list of np.ndarray (fold predictions), list of float (fold RMSEs)
    """
    pp_name = config["pp"]
    fs_name = config.get("fs", "None")
    model_name = config["model"]
    tf = config["tf"]

    pp_fn = PP_REGISTRY[pp_name]

    fold_preds = []
    fold_rmses = []

    for f_idx, (tr, te) in enumerate(folds):
        # 前処理
        X_tr_pp, X_te_pp = pp_fn(X_raw[tr], X_raw[te], groups[tr])

        # 特徴量選択
        y_for_fs = y[tr]  # raw y for feature selection
        if fs_name == "None":
            X_tr_sel, X_te_sel = X_tr_pp, X_te_pp
        elif fs_name.startswith("siPLS("):
            # Parse siPLS(N,C)
            params = fs_name[6:-1].split(",")
            n_int, n_comb = int(params[0]), int(params[1])
            X_tr_sel, X_te_sel, _ = fs_sipls(X_tr_pp, y_for_fs, X_te_pp,
                                               n_intervals=n_int, n_combine=n_comb)
        elif fs_name.startswith("iPLS("):
            n_int = int(fs_name[5:-1])
            X_tr_sel, X_te_sel, _ = fs_ipls(X_tr_pp, y_for_fs, X_te_pp,
                                              n_intervals=n_int)
        else:
            X_tr_sel, X_te_sel = X_tr_pp, X_te_pp

        # 目的変数変換
        if tf == "sqrt":
            y_fit = np.sqrt(y[tr])
        else:
            y_fit = y[tr].copy()

        # モデル予測
        pred = fit_predict(X_tr_sel, y_fit, X_te_sel, model_name)

        # 逆変換
        if tf == "sqrt":
            pred = np.clip(pred, 0, None) ** 2

        pred = np.clip(pred, 0, 200)
        fold_preds.append(pred)
        fold_rmses.append(calc_rmse(y[te], pred))

    return fold_preds, fold_rmses


# ============================================================
# アンサンブル評価
# ============================================================

def eval_ensemble(all_preds, y, folds, idx, weights=None):
    """idx番目のモデル群でアンサンブル予測を評価。"""
    fold_rmses = []
    for f, (_, te) in enumerate(folds):
        preds = [all_preds[i][f] for i in idx]
        if weights is not None:
            ens = sum(w * p for w, p in zip(weights, preds))
        else:
            ens = np.mean(preds, axis=0)
        fold_rmses.append(calc_rmse(y[te], np.clip(ens, 0, 200)))
    return np.mean(fold_rmses), np.std(fold_rmses), fold_rmses


def opt_weights(all_preds, y, folds, idx, n_restarts=30):
    """Nelder-Meadで重み最適化。"""
    n = len(idx)

    def obj(w):
        wn = np.abs(w) / np.sum(np.abs(w))
        r, _, _ = eval_ensemble(all_preds, y, folds, idx, wn)
        return r

    best_r, best_w = np.inf, np.ones(n) / n
    for s in range(n_restarts):
        w0 = np.random.RandomState(s).dirichlet(np.ones(n))
        res = minimize(obj, w0, method="Nelder-Mead",
                       options={"maxiter": 3000, "xatol": 1e-5, "fatol": 1e-5})
        if res.fun < best_r:
            best_r = res.fun
            best_w = np.abs(res.x) / np.sum(np.abs(res.x))
    return best_w, best_r


# ============================================================
# メイン
# ============================================================

def main():
    t_start = time.time()
    df = load_train(DATA_DIR)
    sc = get_spectral_columns(df)
    X_raw = df[sc].values
    y = df["含水率"].values
    groups = df["樹種"].values

    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_raw, y, groups))
    species = [np.unique(groups[te])[0] for _, te in folds]
    n_folds = len(folds)

    print("=" * 70)
    print("Issue #74: EPO+siPLS/iPLS & v6アンサンブル")
    print(f"  v5ベースライン: fold-RMSE = 15.28")
    print(f"  フォールド数: {n_folds} (LOSO-CV)")
    print("=" * 70)

    # ================================================================
    # Step 1: 新規モデル候補の評価
    # ================================================================
    print("\n" + "=" * 70)
    print("Step 1: 新規モデル候補の評価")
    print("=" * 70)

    step1_configs = [
        {"name": "EPO(1)+siPLS(30,3)+PLS(4)+sqrt", "pp": "EPO(1)", "fs": "siPLS(30,3)", "model": "PLS(4)", "tf": "sqrt"},
        {"name": "EPO(1)+siPLS(20,3)+PLS(4)+sqrt", "pp": "EPO(1)", "fs": "siPLS(20,3)", "model": "PLS(4)", "tf": "sqrt"},
        {"name": "EPO(1)+iPLS(50)+PLS(4)+sqrt", "pp": "EPO(1)", "fs": "iPLS(50)", "model": "PLS(4)", "tf": "sqrt"},
        {"name": "EPO(1)+iPLS(30)+PLS(4)+sqrt", "pp": "EPO(1)", "fs": "iPLS(30)", "model": "PLS(4)", "tf": "sqrt"},
        {"name": "EPO(1)+siPLS(30,3)+PLS(3)+raw", "pp": "EPO(1)", "fs": "siPLS(30,3)", "model": "PLS(3)", "tf": "raw"},
        {"name": "SNV+siPLS(40,3)+PLS(4)+sqrt", "pp": "SNV", "fs": "siPLS(40,3)", "model": "PLS(4)", "tf": "sqrt"},
        {"name": "SNV+siPLS(50,3)+PLS(4)+sqrt", "pp": "SNV", "fs": "siPLS(50,3)", "model": "PLS(4)", "tf": "sqrt"},
        {"name": "SNV+iPLS(40)+PLS(4)+sqrt", "pp": "SNV", "fs": "iPLS(40)", "model": "PLS(4)", "tf": "sqrt"},
        {"name": "PiecewiseMSC+siPLS(30,3)+PLS(4)+sqrt", "pp": "PiecewiseMSC", "fs": "siPLS(30,3)", "model": "PLS(4)", "tf": "sqrt"},
        {"name": "MSC+siPLS(30,3)+PLS(4)+sqrt", "pp": "MSC", "fs": "siPLS(30,3)", "model": "PLS(4)", "tf": "sqrt"},
    ]

    step1_results = []
    for i, cfg in enumerate(step1_configs):
        t1 = time.time()
        fold_preds, fold_rmses = collect_fold_preds(X_raw, y, groups, folds, cfg)
        mean_r = np.mean(fold_rmses)
        std_r = np.std(fold_rmses)
        elapsed = time.time() - t1
        print(f"  [{i+1}/{len(step1_configs)}] {cfg['name']}: "
              f"RMSE={mean_r:.2f} +/- {std_r:.2f} ({elapsed:.1f}s)", flush=True)
        step1_results.append({
            "name": cfg["name"],
            "config": cfg,
            "fold_preds": fold_preds,
            "fold_rmses": fold_rmses,
            "mean_rmse": mean_r,
            "std_rmse": std_r,
        })

    # Step1結果をRMSE順にソート
    step1_results.sort(key=lambda x: x["mean_rmse"])
    print("\nStep 1 結果（RMSE順）:")
    print("-" * 60)
    for r in step1_results:
        print(f"  {r['mean_rmse']:.2f} +/- {r['std_rmse']:.2f} | {r['name']}")

    # ================================================================
    # Step 2: 全モデルのfold予測を収集
    # ================================================================
    print("\n" + "=" * 70)
    print("Step 2: v6アンサンブル候補モデルのfold予測収集")
    print("=" * 70)

    # v5オリジナル4モデル
    v5_configs = [
        {"name": "v5:EPO(1)+PLS(4)+sqrt", "pp": "EPO(1)", "model": "PLS(4)", "tf": "sqrt"},
        {"name": "v5:SG2d+EPO(1)+PLS(3)+raw", "pp": "SG2d+EPO(1)", "model": "PLS(3)", "tf": "raw"},
        {"name": "v5:SNV+AsLS+siPLS(30,3)+PLS(4)+sqrt", "pp": "SNV+AsLS(1e6)", "fs": "siPLS(30,3)", "model": "PLS(4)", "tf": "sqrt"},
        {"name": "v5:SNV+iPLS(50)+PLS(4)+sqrt", "pp": "SNV", "fs": "iPLS(50)", "model": "PLS(4)", "tf": "sqrt"},
    ]

    # 追加多様モデル
    extra_configs = [
        {"name": "extra:SNV+PLS(2)+sqrt", "pp": "SNV", "model": "PLS(2)", "tf": "sqrt"},
        {"name": "extra:SNV+ElasticNet+sqrt", "pp": "SNV", "model": "ElasticNet", "tf": "sqrt"},
        {"name": "extra:PiecewiseMSC+Lasso+raw", "pp": "PiecewiseMSC", "model": "Lasso", "tf": "raw"},
        {"name": "extra:EPO(1)+Lasso+raw", "pp": "EPO(1)", "model": "Lasso", "tf": "raw"},
    ]

    # Step1上位をピックアップ（重複チェック付き）
    top_step1_names = set()
    top_step1_configs = []
    for r in step1_results[:6]:  # 上位6つ
        name = f"new:{r['name']}"
        top_step1_names.add(name)
        top_step1_configs.append({
            "name": name,
            "fold_preds": r["fold_preds"],  # すでに計算済み
        })

    # 全モデル: v5 + extra + step1トップ
    all_model_names = []
    all_preds = []

    # v5モデル（AsLSは遅いので注意）
    for cfg in v5_configs:
        t1 = time.time()
        fold_preds, fold_rmses = collect_fold_preds(X_raw, y, groups, folds, cfg)
        all_model_names.append(cfg["name"])
        all_preds.append(fold_preds)
        mean_r = np.mean(fold_rmses)
        print(f"  {cfg['name']}: RMSE={mean_r:.2f} ({time.time()-t1:.1f}s)", flush=True)

    # Step1トップ（キャッシュ済み）
    for item in top_step1_configs:
        all_model_names.append(item["name"])
        all_preds.append(item["fold_preds"])
        fr = [calc_rmse(y[folds[f][1]], item["fold_preds"][f]) for f in range(n_folds)]
        print(f"  {item['name']}: RMSE={np.mean(fr):.2f} (cached)")

    # 追加多様モデル
    for cfg in extra_configs:
        t1 = time.time()
        fold_preds, fold_rmses = collect_fold_preds(X_raw, y, groups, folds, cfg)
        all_model_names.append(cfg["name"])
        all_preds.append(fold_preds)
        mean_r = np.mean(fold_rmses)
        print(f"  {cfg['name']}: RMSE={mean_r:.2f} ({time.time()-t1:.1f}s)", flush=True)

    n_models = len(all_model_names)
    print(f"\nアンサンブル候補: {n_models} モデル")

    # 個別モデルランキング
    indiv_scores = []
    for i in range(n_models):
        fr = [calc_rmse(y[folds[f][1]], all_preds[i][f]) for f in range(n_folds)]
        indiv_scores.append((np.mean(fr), np.std(fr), i))
    indiv_scores.sort()

    print("\n個別モデルランキング:")
    print("-" * 70)
    for mean_r, std_r, i in indiv_scores:
        fold_detail = " ".join([f"{calc_rmse(y[folds[f][1]], all_preds[i][f]):.1f}" for f in range(n_folds)])
        print(f"  {mean_r:.2f} +/- {std_r:.2f} | {all_model_names[i]}  [{fold_detail}]")

    # ================================================================
    # Step 3: アンサンブル探索
    # ================================================================
    print("\n" + "=" * 70)
    print("Step 3: アンサンブル探索 (exhaustive combo search)")
    print("=" * 70)

    results = []

    # Exhaustive combo search for k=3,4,5,6,7
    for k in [3, 4, 5, 6, 7]:
        if k > n_models:
            break
        t1 = time.time()
        n_combos = 0
        best_r_avg = np.inf
        best_combo_avg = None
        best_r_top = np.inf
        best_combo_top = None

        for combo in combinations(range(n_models), k):
            n_combos += 1
            r, s, _ = eval_ensemble(all_preds, y, folds, list(combo))
            if r < best_r_avg:
                best_r_avg = r
                best_combo_avg = list(combo)

        # Record best SimpleAvg
        if best_combo_avg is not None:
            _, s, fr = eval_ensemble(all_preds, y, folds, best_combo_avg)
            results.append({
                "method": f"BestSimpleAvg-k{k}",
                "rmse": best_r_avg,
                "std": s,
                "idx": best_combo_avg,
                "weights": None,
                "fold_rmses": fr,
            })

            # WeightedAvg for best combo
            w, rw = opt_weights(all_preds, y, folds, best_combo_avg)
            _, sw, frw = eval_ensemble(all_preds, y, folds, best_combo_avg, w)
            results.append({
                "method": f"BestWeightedAvg-k{k}",
                "rmse": rw,
                "std": sw,
                "idx": best_combo_avg,
                "weights": w,
                "fold_rmses": frw,
            })

        elapsed = time.time() - t1
        print(f"  k={k}: {n_combos} combos, "
              f"BestAvg={best_r_avg:.4f}, "
              f"BestWeighted={rw:.4f} ({elapsed:.1f}s)", flush=True)

    # Also try top-N simple avg and weighted
    for k in [3, 4, 5, 6, 7]:
        if k > n_models:
            break
        top_idx = [i for _, _, i in indiv_scores[:k]]
        r, s, fr = eval_ensemble(all_preds, y, folds, top_idx)
        results.append({
            "method": f"TopN-SimpleAvg-k{k}",
            "rmse": r,
            "std": s,
            "idx": top_idx,
            "weights": None,
            "fold_rmses": fr,
        })
        w, rw = opt_weights(all_preds, y, folds, top_idx)
        _, sw, frw = eval_ensemble(all_preds, y, folds, top_idx, w)
        results.append({
            "method": f"TopN-WeightedAvg-k{k}",
            "rmse": rw,
            "std": sw,
            "idx": top_idx,
            "weights": w,
            "fold_rmses": frw,
        })

    # Sort all results
    results.sort(key=lambda x: x["rmse"])

    # ================================================================
    # 結果出力
    # ================================================================
    print("\n" + "=" * 70)
    print("Top 20 アンサンブル結果")
    print("=" * 70)

    for rank, res in enumerate(results[:20], 1):
        print(f"\n  [{rank}] RMSE={res['rmse']:.4f} +/- {res['std']:.2f} | {res['method']}")
        if res["weights"] is not None:
            for i, wi in zip(res["idx"], res["weights"]):
                if wi > 0.01:
                    print(f"       {wi:.3f}: {all_model_names[i]}")
        else:
            for i in res["idx"]:
                print(f"       - {all_model_names[i]}")

        # Fold details
        if res["fold_rmses"]:
            fold_str = " | ".join([f"{sp}:{r:.2f}" for sp, r in zip(species, res["fold_rmses"])])
            print(f"       Folds: {fold_str}")

    # Best result
    best = results[0]
    print("\n" + "=" * 70)
    print(f"BEST: RMSE = {best['rmse']:.4f} ({best['method']})")
    print(f"v5: 15.28 | 改善: {15.28 - best['rmse']:.4f}")
    print("Fold詳細:")
    for sp, r in zip(species, best["fold_rmses"]):
        print(f"  {sp}: {r:.2f}")
    if best["weights"] is not None:
        print("重み:")
        for i, wi in zip(best["idx"], best["weights"]):
            print(f"  {wi:.4f}: {all_model_names[i]}")
    else:
        print("モデル:")
        for i in best["idx"]:
            print(f"  - {all_model_names[i]}")
    print("=" * 70)

    # CSV保存
    rows = []
    for res in results:
        rows.append({
            "method": res["method"],
            "rmse": res["rmse"],
            "std": res["std"],
            "models": str([all_model_names[i] for i in res["idx"]]),
            "weights": str(res["weights"].tolist()) if res["weights"] is not None else "",
        })
    pd.DataFrame(rows).to_csv(OUT_DIR / "issue74_epo_sipls_results.csv", index=False)

    print(f"\n総実行時間: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
