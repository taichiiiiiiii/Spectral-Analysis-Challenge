"""Issue #93 Cycle1: LB最適化提出パイプライン v6

背景:
- Local LOSO-CV RMSE: 13.70 (best11) → Public LB RMSE: 21.56
- Local LOSO-CV RMSE: 13.82 (best8) → Public LB RMSE: 21.29
- best8 (少モデル) > best11 (多モデル) on LB → 重み最適化がCV過学習
- trainの13樹種とtestの6樹種は完全に異なる

戦略:
(A) Top-K均等平均 (K=4,6,8) — Nelder-Mead最適化なし
(B) 正則化付き最適化: obj(w) = RMSE + alpha * ||w - 1/n||^2
(C) PLS系モデルのみの均等平均（GBR/Huber除外）
(D) Shrinkage: w_final = (1-alpha) * w_opt + alpha * w_uniform
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import warnings
import time
from scipy.optimize import minimize
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from src.eda.data_loader import load_train, load_test, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue20_savgol import apply_savgol
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue19_msc import compute_msc_reference
from src.preprocessing.issue62_additional_preprocessing import apply_asls, apply_piecewise_msc
from src.modeling.issue65_feature_selection import sipls_select, ipls_select

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUT = OUT_DIR / "modeling"
MODEL_OUT.mkdir(parents=True, exist_ok=True)

# --- 共通関数 (run_issue87_final.pyから) ---

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def pp(X_tr, X_te, g, pp_name):
    """前処理を適用"""
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
        return (apply_savgol(apply_snv(X_tr), deriv=2, window_length=7),
                apply_savgol(apply_snv(X_te), deriv=2, window_length=7))
    return X_tr.copy(), X_te.copy()


def fs(X_tr, X_te, y, fs_name):
    """特徴量選択"""
    if not fs_name:
        return X_tr, X_te
    if fs_name.startswith("siPLS"):
        p = fs_name.replace("siPLS(", "").rstrip(")").split(",")
        X_tr, X_te, _ = sipls_select(X_tr, y, X_te, n_intervals=int(p[0]),
                                      n_components=3, n_combine=int(p[1]))
    elif fs_name.startswith("iPLS"):
        n = int(fs_name.replace("iPLS(", "").rstrip(")"))
        X_tr, X_te, _ = ipls_select(X_tr, y, X_te, n_intervals=n,
                                     n_components=3, n_best=1)
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
        validation_fraction=0.15, n_iter_no_change=50, tol=0.01)
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


def run_model(X_tr, X_te, y, g, cfg):
    """1つのモデル設定でCV予測を生成"""
    Xtr, Xte = pp(X_tr, X_te, g, cfg["pp"])
    Xtr, Xte = fs(Xtr, Xte, y, cfg.get("fs"))
    t = cfg["type"]
    if t == "pls":
        return pred_pls(Xtr, Xte, y, cfg.get("nc", 4), cfg["tf"])
    elif t == "gbr":
        return pred_gbr(Xtr, Xte, y, cfg.get("nc", 4), cfg["tf"],
                        cfg.get("ne", 200), cfg.get("md", 3), cfg.get("lr", 0.05))
    elif t == "huber":
        return pred_huber(Xtr, Xte, y, cfg.get("nc", 4), cfg["tf"],
                          cfg.get("eps", 1.35))


def predict_test(X_train, y_train, groups_train, X_test, cfg):
    """train全体でfit → test predict"""
    Xtr, Xte = pp(X_train, X_test, groups_train, cfg["pp"])
    Xtr, Xte = fs(Xtr, Xte, y_train, cfg.get("fs"))
    t = cfg["type"]
    if t == "pls":
        return pred_pls(Xtr, Xte, y_train, cfg.get("nc", 4), cfg["tf"])
    elif t == "gbr":
        return pred_gbr(Xtr, Xte, y_train, cfg.get("nc", 4), cfg["tf"],
                        cfg.get("ne", 200), cfg.get("md", 3), cfg.get("lr", 0.05))
    elif t == "huber":
        return pred_huber(Xtr, Xte, y_train, cfg.get("nc", 4), cfg["tf"],
                          cfg.get("eps", 1.35))


# --- 15モデル構成 ---
CFGS = [
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


# --- アンサンブル戦略関数 ---

def ensemble_eval(all_preds, folds, y, idx, w=None):
    """指定モデルのアンサンブルをfold別RMSE評価"""
    fold_rmses = []
    for fi, (_, te) in enumerate(folds):
        ps = [all_preds[m][fi] for m in idx]
        if w is not None:
            wn = np.array(w)
            wn = wn / wn.sum()
            e = sum(wi * p for wi, p in zip(wn, ps))
        else:
            e = np.mean(ps, axis=0)
        fold_rmses.append(rmse(y[te], np.clip(e, 0, 300)))
    return np.mean(fold_rmses), fold_rmses


def optimize_weights_nelder_mead(all_preds, folds, y, idx, n_restarts=40):
    """Nelder-Meadで重み最適化（正則化なし）"""
    nn = len(idx)

    def obj(w):
        wn = np.abs(w) / np.sum(np.abs(w))
        r, _ = ensemble_eval(all_preds, folds, y, idx, wn)
        return r

    best_r, best_w = np.inf, np.ones(nn) / nn
    for s in range(n_restarts):
        w0 = np.random.RandomState(s).dirichlet(np.ones(nn))
        res = minimize(obj, w0, method="Nelder-Mead", options={"maxiter": 5000})
        if res.fun < best_r:
            best_r = res.fun
            best_w = np.abs(res.x) / np.sum(np.abs(res.x))
    return best_w, best_r


def optimize_weights_regularized(all_preds, folds, y, idx, alpha, n_restarts=40):
    """正則化付き最適化: obj(w) = RMSE + alpha * ||w - 1/n||^2"""
    nn = len(idx)
    w_uniform = np.ones(nn) / nn

    def obj(w):
        wn = np.abs(w) / np.sum(np.abs(w))
        r, _ = ensemble_eval(all_preds, folds, y, idx, wn)
        reg = alpha * np.sum((wn - w_uniform) ** 2)
        return r + reg

    best_r, best_w = np.inf, w_uniform.copy()
    for s in range(n_restarts):
        w0 = np.random.RandomState(s).dirichlet(np.ones(nn))
        res = minimize(obj, w0, method="Nelder-Mead", options={"maxiter": 5000})
        if res.fun < best_r:
            best_r = res.fun
            wn = np.abs(res.x) / np.sum(np.abs(res.x))
            best_w = wn
            # 実際のRMSE（正則化項なし）を計算
    actual_rmse, _ = ensemble_eval(all_preds, folds, y, idx, best_w)
    return best_w, actual_rmse


def strategy_a_topk_uniform(all_preds, folds, y, individual_scores, ks=(4, 6, 8)):
    """(A) Top-K均等平均"""
    sorted_idx = [i for _, i in sorted(individual_scores)]
    results = []
    for k in ks:
        top_k = sorted_idx[:k]
        avg_rmse, fold_rmses = ensemble_eval(all_preds, folds, y, top_k)
        results.append({
            "name": f"A:Top{k}_uniform",
            "rmse": avg_rmse,
            "fold_rmses": fold_rmses,
            "idx": top_k,
            "weights": np.ones(k) / k,
        })
    return results


def strategy_b_regularized(all_preds, folds, y, individual_scores, alphas=(0.1, 0.5, 1.0)):
    """(B) 正則化付き最適化"""
    sorted_idx = [i for _, i in sorted(individual_scores)]
    # Top-8を使用
    top_idx = sorted_idx[:8]
    results = []
    for alpha in alphas:
        w, actual_rmse = optimize_weights_regularized(all_preds, folds, y, top_idx, alpha)
        _, fold_rmses = ensemble_eval(all_preds, folds, y, top_idx, w)
        results.append({
            "name": f"B:Reg_alpha{alpha}",
            "rmse": actual_rmse,
            "fold_rmses": fold_rmses,
            "idx": top_idx,
            "weights": w,
        })
    return results


def strategy_c_pls_only(all_preds, folds, y, cfgs):
    """(C) PLS系モデルのみの均等平均"""
    pls_idx = [i for i, c in enumerate(cfgs) if c["type"] == "pls"]
    avg_rmse, fold_rmses = ensemble_eval(all_preds, folds, y, pls_idx)
    return [{
        "name": "C:PLS_only_uniform",
        "rmse": avg_rmse,
        "fold_rmses": fold_rmses,
        "idx": pls_idx,
        "weights": np.ones(len(pls_idx)) / len(pls_idx),
    }]


def strategy_d_shrinkage(all_preds, folds, y, individual_scores, alphas=(0.3, 0.5, 0.7)):
    """(D) Shrinkage: w_final = (1-alpha) * w_opt + alpha * w_uniform"""
    sorted_idx = [i for _, i in sorted(individual_scores)]
    top_idx = sorted_idx[:8]
    # まず通常の最適化
    w_opt, _ = optimize_weights_nelder_mead(all_preds, folds, y, top_idx)
    nn = len(top_idx)
    w_uniform = np.ones(nn) / nn
    results = []
    for alpha in alphas:
        w_shrunk = (1 - alpha) * w_opt + alpha * w_uniform
        w_shrunk = w_shrunk / w_shrunk.sum()
        avg_rmse, fold_rmses = ensemble_eval(all_preds, folds, y, top_idx, w_shrunk)
        results.append({
            "name": f"D:Shrink_alpha{alpha}",
            "rmse": avg_rmse,
            "fold_rmses": fold_rmses,
            "idx": top_idx,
            "weights": w_shrunk,
        })
    return results


def save_submission(test_preds, sample_numbers, filename):
    """提出ファイルを保存（ヘッダーなし）"""
    df = pd.DataFrame({"sample_number": sample_numbers, "prediction": test_preds})
    df.to_csv(filename, index=False, header=False)
    return filename


def main():
    t0 = time.time()

    # データ読み込み
    df_train = load_train(DATA_DIR)
    df_test = load_test(DATA_DIR)
    sc = get_spectral_columns(df_train)
    X = df_train[sc].values
    y = df_train["含水率"].values
    g = df_train["樹種"].values
    X_test = df_test[sc].values
    test_sample_numbers = df_test["sample number"].values

    logo = LeaveOneGroupOut()
    folds = list(logo.split(X, y, g))
    sp = [np.unique(g[te])[0] for _, te in folds]

    print("=" * 70)
    print("Issue #93 Cycle1: LB最適化提出パイプライン v6")
    print("=" * 70)

    n = len(CFGS)
    all_preds = [[] for _ in range(n)]

    # --- 個別モデルCV ---
    print(f"\n--- 個別モデル評価 ({n}モデル) ---\n", flush=True)
    individual_scores = []
    individual_fold_rmses = {}
    for i, c in enumerate(CFGS):
        t1 = time.time()
        fr = []
        for fi, (tr, te) in enumerate(folds):
            try:
                p = run_model(X[tr], X[te], y[tr], g[tr], c)
                all_preds[i].append(p)
                fr.append(rmse(y[te], p))
            except Exception as e:
                all_preds[i].append(np.full(len(te), y[tr].mean()))
                fr.append(999.0)
                print(f"  ERR {c['name']} f{fi}: {e}")
        avg = np.mean(fr)
        individual_scores.append((avg, i))
        individual_fold_rmses[i] = fr
        print(f"  [{i+1:>2}/{n}] {c['name']}: {avg:.2f} ({time.time()-t1:.1f}s)", flush=True)

    # 個別モデルのソート表示
    iv = sorted(individual_scores)
    print("\n--- 個別モデルランキング ---")
    for rank, (r, i) in enumerate(iv, 1):
        print(f"  {rank:>2}. {r:.2f} | {CFGS[i]['name']} (type={CFGS[i]['type']})")

    # --- アンサンブル戦略比較 ---
    print("\n" + "=" * 70)
    print("アンサンブル戦略比較")
    print("=" * 70)

    all_results = []

    # (A) Top-K均等平均
    print("\n--- (A) Top-K均等平均 ---")
    results_a = strategy_a_topk_uniform(all_preds, folds, y, individual_scores)
    for r in results_a:
        print(f"  {r['name']}: RMSE={r['rmse']:.4f}")
    all_results.extend(results_a)

    # (B) 正則化付き最適化
    print("\n--- (B) 正則化付き最適化 ---")
    results_b = strategy_b_regularized(all_preds, folds, y, individual_scores)
    for r in results_b:
        print(f"  {r['name']}: RMSE={r['rmse']:.4f}")
        w = r["weights"]
        idx = r["idx"]
        for ii, wi in zip(idx, w):
            if wi > 0.02:
                print(f"    {wi:.3f}: {CFGS[ii]['name']}")
    all_results.extend(results_b)

    # (C) PLS系のみ
    print("\n--- (C) PLS系のみ均等平均 ---")
    results_c = strategy_c_pls_only(all_preds, folds, y, CFGS)
    for r in results_c:
        print(f"  {r['name']}: RMSE={r['rmse']:.4f}")
        print(f"    モデル: {[CFGS[i]['name'] for i in r['idx']]}")
    all_results.extend(results_c)

    # (D) Shrinkage
    print("\n--- (D) Shrinkage ---")
    results_d = strategy_d_shrinkage(all_preds, folds, y, individual_scores)
    for r in results_d:
        print(f"  {r['name']}: RMSE={r['rmse']:.4f}")
    all_results.extend(results_d)

    # --- 全戦略のソート ---
    all_results.sort(key=lambda x: x["rmse"])
    print("\n" + "=" * 70)
    print("全戦略ランキング")
    print("=" * 70)
    for rank, r in enumerate(all_results, 1):
        print(f"  {rank:>2}. {r['name']}: RMSE={r['rmse']:.4f}")

    # --- fold別RMSE出力 ---
    print("\n" + "=" * 70)
    print("fold別RMSE (上位5戦略)")
    print("=" * 70)
    for r in all_results[:5]:
        print(f"\n  {r['name']} (RMSE={r['rmse']:.4f}):")
        for s, fr in zip(sp, r["fold_rmses"]):
            mark = " ※" if s == "ベイスギ" else ""
            print(f"    {s}: {fr:.2f}{mark}")
        nbs = [fr for s, fr in zip(sp, r["fold_rmses"]) if s != "ベイスギ"]
        print(f"    除ベイスギ: {np.mean(nbs):.4f}")

    # --- テスト予測生成 ---
    print("\n" + "=" * 70)
    print("テスト予測生成")
    print("=" * 70)

    # 各戦略のBESTで提出ファイル生成
    # まず全モデルのテスト予測を生成
    print("\n  全モデルのテスト予測を生成中...", flush=True)
    test_preds_all = []
    for i, c in enumerate(CFGS):
        t1 = time.time()
        try:
            tp = predict_test(X, y, g, X_test, c)
            test_preds_all.append(tp)
            print(f"    [{i+1:>2}/{n}] {c['name']}: done ({time.time()-t1:.1f}s)", flush=True)
        except Exception as e:
            test_preds_all.append(np.full(len(X_test), y.mean()))
            print(f"    [{i+1:>2}/{n}] {c['name']}: ERR {e}", flush=True)

    # 各戦略カテゴリのベストで提出ファイルを生成
    strategy_categories = {
        "A": [r for r in all_results if r["name"].startswith("A:")],
        "B": [r for r in all_results if r["name"].startswith("B:")],
        "C": [r for r in all_results if r["name"].startswith("C:")],
        "D": [r for r in all_results if r["name"].startswith("D:")],
    }

    submissions = []
    for cat, cat_results in strategy_categories.items():
        if not cat_results:
            continue
        best_in_cat = min(cat_results, key=lambda x: x["rmse"])
        idx = best_in_cat["idx"]
        w = best_in_cat["weights"]
        preds = [test_preds_all[m] for m in idx]
        wn = np.array(w) / np.array(w).sum()
        test_pred = sum(wi * p for wi, p in zip(wn, preds))
        test_pred = np.clip(test_pred, 0, 300)

        fname = OUT_DIR / f"submission_v6_{cat}_{best_in_cat['name'].replace(':', '_').replace(' ', '')}.csv"
        save_submission(test_pred, test_sample_numbers, fname)
        submissions.append((best_in_cat["name"], best_in_cat["rmse"], fname))
        print(f"\n  [{cat}] {best_in_cat['name']} (CV={best_in_cat['rmse']:.4f})")
        print(f"    → {fname}")

    # 全体ベストも保存
    overall_best = all_results[0]
    idx = overall_best["idx"]
    w = overall_best["weights"]
    preds = [test_preds_all[m] for m in idx]
    wn = np.array(w) / np.array(w).sum()
    test_pred = sum(wi * p for wi, p in zip(wn, preds))
    test_pred = np.clip(test_pred, 0, 300)

    fname_best = OUT_DIR / f"submission_v6_overall_best.csv"
    save_submission(test_pred, test_sample_numbers, fname_best)
    print(f"\n  [Overall BEST] {overall_best['name']} (CV={overall_best['rmse']:.4f})")
    print(f"    → {fname_best}")

    # --- 結果CSV ---
    rows = []
    for r in all_results:
        rows.append({
            "strategy": r["name"],
            "cv_rmse": r["rmse"],
            "models": str([CFGS[i]["name"] for i in r["idx"]]),
            "weights": str(r["weights"].tolist()),
        })
    pd.DataFrame(rows).to_csv(MODEL_OUT / "issue93_cycle1_results.csv", index=False)
    print(f"\n結果CSV: {MODEL_OUT / 'issue93_cycle1_results.csv'}")

    print(f"\n{'='*70}")
    print(f"完了. 総時間: {time.time()-t0:.0f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
