"""Issue #102: Cycle 10 統合提出パイプライン v7

LB改善を直接狙う構成:
- 均等平均 or 軽い正則化重み（Nelder-Mead最適化をしない or 正則化付き）
- ドメイン適応モデルを必ず含める（TCA, di-PLS, SA）
- 多様な前処理の組み合わせ（相関の低いモデル同士を組む）
- PLS系モデル中心（GBR系はCV過学習リスクが高い）

グループA: PLS系（ドメイン適応なし）
グループB: ドメイン適応モデル（TCA, di-PLS, SA）
グループC: ロバストモデル（Huber, PMSC）

4つのアンサンブル戦略:
  Strategy 1: 全12モデル均等平均
  Strategy 2: グループA(4) + B(5)のドメイン適応重視均等平均
  Strategy 3: Shrinkage: w = 0.5 * w_opt + 0.5 * w_uniform
  Strategy 4: PLS系のみ(A1-A4 + C3)均等平均
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
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut

from src.eda.data_loader import load_train, load_test, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue20_savgol import apply_savgol
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue19_msc import compute_msc_reference
from src.preprocessing.issue62_additional_preprocessing import apply_asls, apply_piecewise_msc
from src.modeling.issue65_feature_selection import sipls_select, ipls_select
from src.preprocessing.issue38_tca import tca_transform
from src.preprocessing.issue43_dipls import fit_predict_dipls
from src.preprocessing.issue44_subspace_alignment import subspace_align

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUT = OUT_DIR / "modeling"
MODEL_OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 共通ユーティリティ
# ============================================================

def calc_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def preprocess(X_tr, X_te, groups_train, pp_name):
    """前処理を適用する。"""
    if pp_name == "SNV":
        return apply_snv(X_tr), apply_snv(X_te)
    elif pp_name == "EPO(1)":
        P = compute_epo_projection(X_tr, groups_train, n_components=1)
        return apply_epo(X_tr, P), apply_epo(X_te, P)
    elif pp_name == "SNV+AsLS(1e6)":
        return apply_asls(apply_snv(X_tr), lam=1e6), apply_asls(apply_snv(X_te), lam=1e6)
    elif pp_name == "PMSC":
        ref = compute_msc_reference(X_tr)
        return apply_piecewise_msc(X_tr, ref, 3), apply_piecewise_msc(X_te, ref, 3)
    elif pp_name == "SG2d+EPO(1)":
        xs = apply_savgol(X_tr, deriv=2, window_length=7)
        xst = apply_savgol(X_te, deriv=2, window_length=7)
        P = compute_epo_projection(xs, groups_train, n_components=1)
        return apply_epo(xs, P), apply_epo(xst, P)
    return X_tr.copy(), X_te.copy()


def feature_select(X_tr, X_te, y_train, fs_name):
    """特徴量選択を適用する。"""
    if not fs_name:
        return X_tr, X_te
    if fs_name.startswith("siPLS"):
        p = fs_name.replace("siPLS(", "").rstrip(")").split(",")
        X_tr, X_te, _ = sipls_select(X_tr, y_train, X_te,
                                      n_intervals=int(p[0]),
                                      n_components=3, n_combine=int(p[1]))
    elif fs_name.startswith("iPLS"):
        n = int(fs_name.replace("iPLS(", "").rstrip(")"))
        X_tr, X_te, _ = ipls_select(X_tr, y_train, X_te,
                                     n_intervals=n, n_components=3, n_best=1)
    return X_tr, X_te


def apply_target_transform(y, tf):
    """目的変数変換を適用する。"""
    if tf == "sqrt":
        return np.sqrt(y)
    return y.copy()


def inverse_target_transform(pred, tf):
    """目的変数逆変換を適用する。"""
    if tf == "sqrt":
        return np.clip(pred, 0, None) ** 2
    return pred


# ============================================================
# グループA: PLS系（ドメイン適応なし）
# ============================================================

def predict_pls(X_tr_raw, X_te_raw, y_train, groups_train, cfg):
    """PLS系の予測。前処理 + 特徴量選択 + PLS。"""
    X_tr, X_te = preprocess(X_tr_raw, X_te_raw, groups_train, cfg["pp"])
    X_tr, X_te = feature_select(X_tr, X_te, y_train, cfg.get("fs"))
    nc = max(1, min(cfg.get("nc", 4), X_tr.shape[1] - 1))
    y_fit = apply_target_transform(y_train, cfg["tf"])
    pls = PLSRegression(n_components=nc)
    pls.fit(X_tr, y_fit)
    pred = pls.predict(X_te).ravel()
    return inverse_target_transform(pred, cfg["tf"])


# ============================================================
# グループB: ドメイン適応モデル
# ============================================================

def _pca_reduce(X_tr, X_te, n_pca=50):
    """PCAで次元削減してからドメイン適応に渡す（高速化）。"""
    pca = PCA(n_components=min(n_pca, X_tr.shape[1], X_tr.shape[0]))
    X_tr_r = pca.fit_transform(X_tr)
    X_te_r = pca.transform(X_te)
    return X_tr_r, X_te_r


def predict_tca_ridge(X_tr_raw, X_te_raw, y_train, groups_train, cfg):
    """TCA + Ridge回帰の予測（PCA前処理で高速化）。"""
    X_tr, X_te = preprocess(X_tr_raw, X_te_raw, groups_train, cfg["pp"])
    kernel = cfg.get("kernel", "linear")
    nc = cfg.get("nc", 10)
    alpha = cfg.get("alpha", 1.0)
    n_pca = cfg.get("n_pca", 50)
    y_fit = apply_target_transform(y_train, cfg["tf"])

    # PCAで次元削減してからTCA適用（高速化）
    X_tr_r, X_te_r = _pca_reduce(X_tr, X_te, n_pca=n_pca)
    Z_tr, Z_te = tca_transform(X_tr_r, X_te_r, n_components=nc, kernel=kernel)
    ridge = Ridge(alpha=alpha)
    ridge.fit(Z_tr, y_fit)
    pred = ridge.predict(Z_te)
    return inverse_target_transform(pred, cfg["tf"])


def predict_dipls(X_tr_raw, X_te_raw, y_train, groups_train, cfg):
    """di-PLS予測（PCA前処理で高速化）。"""
    X_tr, X_te = preprocess(X_tr_raw, X_te_raw, groups_train, cfg["pp"])
    nc = cfg.get("nc", 4)
    dipls_lambda = cfg.get("dipls_lambda", 1.0)
    n_pca = cfg.get("n_pca", 50)
    y_fit = apply_target_transform(y_train, cfg["tf"])

    # PCAで次元削減してからdi-PLS適用（高速化）
    X_tr_r, X_te_r = _pca_reduce(X_tr, X_te, n_pca=n_pca)
    pred = fit_predict_dipls(X_tr_r, y_fit, X_te_r,
                             n_components=nc, dipls_lambda=dipls_lambda)
    return inverse_target_transform(pred, cfg["tf"])


def predict_sa_ridge(X_tr_raw, X_te_raw, y_train, groups_train, cfg):
    """SubspaceAlignment + Ridge回帰の予測（PCA前処理で高速化）。"""
    X_tr, X_te = preprocess(X_tr_raw, X_te_raw, groups_train, cfg["pp"])
    nc = cfg.get("nc", 10)
    alpha = cfg.get("alpha", 1.0)
    n_pca = cfg.get("n_pca", 50)
    y_fit = apply_target_transform(y_train, cfg["tf"])

    # PCAで次元削減してからSA適用（高速化）
    X_tr_r, X_te_r = _pca_reduce(X_tr, X_te, n_pca=n_pca)
    Z_tr, Z_te = subspace_align(X_tr_r, X_te_r, n_components=nc)
    ridge = Ridge(alpha=alpha)
    ridge.fit(Z_tr, y_fit)
    pred = ridge.predict(Z_te)
    return inverse_target_transform(pred, cfg["tf"])


# ============================================================
# グループC: ロバストモデル
# ============================================================

def predict_huber(X_tr_raw, X_te_raw, y_train, groups_train, cfg):
    """PLS + Huber回帰の予測。"""
    X_tr, X_te = preprocess(X_tr_raw, X_te_raw, groups_train, cfg["pp"])
    X_tr, X_te = feature_select(X_tr, X_te, y_train, cfg.get("fs"))
    nc = max(1, min(cfg.get("nc", 4), X_tr.shape[1] - 1))
    eps = cfg.get("eps", 1.35)
    y_fit = apply_target_transform(y_train, cfg["tf"])

    pls = PLSRegression(n_components=nc)
    pls.fit(X_tr, y_fit)
    T_tr = pls.transform(X_tr)
    T_te = pls.transform(X_te)
    sc = StandardScaler()
    T_tr_s = sc.fit_transform(T_tr)
    T_te_s = sc.transform(T_te)
    huber = HuberRegressor(epsilon=eps, max_iter=200, alpha=0.01)
    huber.fit(T_tr_s, y_fit)
    pred = huber.predict(T_te_s)
    return inverse_target_transform(pred, cfg["tf"])


# ============================================================
# モデル設定
# ============================================================

# 予測関数ディスパッチ
PREDICT_FNS = {
    "pls": predict_pls,
    "tca": predict_tca_ridge,
    "dipls": predict_dipls,
    "sa": predict_sa_ridge,
    "huber": predict_huber,
}

# 12モデル構成
MODEL_CFGS = [
    # --- グループA: PLS系（ドメイン適応なし） ---
    # A1: EPO(1) + PLS(4) + sqrt — 現ベスト個別
    {"name": "A1:EPO+PLS4+sqrt", "pp": "EPO(1)", "nc": 4, "tf": "sqrt",
     "type": "pls", "group": "A"},
    # A2: SNV + iPLS(50) + PLS(4) + sqrt
    {"name": "A2:SNV+iPLS50+PLS4+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt",
     "fs": "iPLS(50)", "type": "pls", "group": "A"},
    # A3: SG2d + EPO(1) + PLS(3) + raw
    {"name": "A3:SG2d+EPO+PLS3+raw", "pp": "SG2d+EPO(1)", "nc": 3, "tf": "raw",
     "type": "pls", "group": "A"},
    # A4: SNV + AsLS + siPLS(30,3) + PLS(4) + sqrt
    {"name": "A4:SNV+AsLS+siPLS+PLS4+sqrt", "pp": "SNV+AsLS(1e6)", "nc": 4,
     "tf": "sqrt", "fs": "siPLS(30,3)", "type": "pls", "group": "A"},

    # --- グループB: ドメイン適応モデル ---
    # B1: TCA(linear, nc=10) + Ridge + sqrt
    {"name": "B1:TCA_linear+Ridge+sqrt", "pp": "EPO(1)", "nc": 10,
     "kernel": "linear", "alpha": 1.0, "tf": "sqrt", "type": "tca", "group": "B"},
    # B2: TCA(rbf, nc=5) + Ridge + sqrt
    {"name": "B2:TCA_rbf+Ridge+sqrt", "pp": "EPO(1)", "nc": 5,
     "kernel": "rbf", "alpha": 1.0, "tf": "sqrt", "type": "tca", "group": "B"},
    # B3: di-PLS(lambda=1.0, nc=4) + sqrt
    {"name": "B3:diPLS_l1+sqrt", "pp": "EPO(1)", "nc": 4,
     "dipls_lambda": 1.0, "tf": "sqrt", "type": "dipls", "group": "B"},
    # B4: di-PLS(lambda=10.0, nc=3) + raw
    {"name": "B4:diPLS_l10+raw", "pp": "EPO(1)", "nc": 3,
     "dipls_lambda": 10.0, "tf": "raw", "type": "dipls", "group": "B"},
    # B5: SubspaceAlignment(nc=10) + Ridge + sqrt
    {"name": "B5:SA+Ridge+sqrt", "pp": "EPO(1)", "nc": 10,
     "alpha": 1.0, "tf": "sqrt", "type": "sa", "group": "B"},

    # --- グループC: ロバストモデル ---
    # C1: EPO(1) + PLS(4) + Huber + sqrt
    {"name": "C1:EPO+Huber+sqrt", "pp": "EPO(1)", "nc": 4, "tf": "sqrt",
     "eps": 1.35, "type": "huber", "group": "C"},
    # C2: SNV + iPLS(50) + PLS(4) + Huber + sqrt
    {"name": "C2:SNV+iPLS50+Huber+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt",
     "fs": "iPLS(50)", "eps": 1.35, "type": "huber", "group": "C"},
    # C3: PMSC + siPLS + PLS(4) + sqrt
    {"name": "C3:PMSC+siPLS+PLS4+sqrt", "pp": "PMSC", "nc": 4, "tf": "sqrt",
     "fs": "siPLS(30,3)", "type": "pls", "group": "C"},
]


# ============================================================
# アンサンブル評価関数
# ============================================================

def ensemble_eval(all_preds, folds, y, indices, weights=None):
    """指定モデルのアンサンブルをfold別RMSE評価する。"""
    fold_rmses = []
    for fi, (_, te) in enumerate(folds):
        preds_list = [all_preds[m][fi] for m in indices]
        if weights is not None:
            wn = np.array(weights)
            wn = wn / wn.sum()
            ens = sum(wi * p for wi, p in zip(wn, preds_list))
        else:
            ens = np.mean(preds_list, axis=0)
        fold_rmses.append(calc_rmse(y[te], np.clip(ens, 0, 300)))
    return np.mean(fold_rmses), fold_rmses


def optimize_weights_nelder_mead(all_preds, folds, y, indices, n_restarts=30):
    """Nelder-Meadで重み最適化（正則化なし）。"""
    nn = len(indices)

    def obj(w):
        wn = np.abs(w) / np.sum(np.abs(w))
        r, _ = ensemble_eval(all_preds, folds, y, indices, wn)
        return r

    best_r, best_w = np.inf, np.ones(nn) / nn
    for s in range(n_restarts):
        w0 = np.random.RandomState(s).dirichlet(np.ones(nn))
        res = minimize(obj, w0, method="Nelder-Mead",
                       options={"maxiter": 3000})
        if res.fun < best_r:
            best_r = res.fun
            best_w = np.abs(res.x) / np.sum(np.abs(res.x))
    return best_w, best_r


def save_submission(test_preds, sample_numbers, filepath):
    """提出ファイルを保存（ヘッダーなし）。"""
    df = pd.DataFrame({"id": sample_numbers, "pred": test_preds})
    df.to_csv(filepath, index=False, header=False)
    return filepath


# ============================================================
# メイン実行
# ============================================================

def main():
    t0 = time.time()

    # --- データ読み込み ---
    df_train = load_train(DATA_DIR)
    df_test = load_test(DATA_DIR)
    spec_cols = get_spectral_columns(df_train)
    X_raw = df_train[spec_cols].values
    y = df_train["含水率"].values
    groups = df_train["樹種"].values
    X_test_raw = df_test[spec_cols].values
    test_sample_numbers = df_test["sample number"].values

    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_raw, y, groups))
    sp_names = [np.unique(groups[te])[0] for _, te in folds]

    n_models = len(MODEL_CFGS)

    print("=" * 70)
    print("Issue #102: Cycle 10 統合提出パイプライン v7")
    print(f"  モデル数: {n_models}")
    print(f"  フォールド数: {len(folds)}")
    print("=" * 70)

    # ============================================================
    # Phase 1: 個別モデルのLOSO-CV評価
    # ============================================================
    print(f"\n--- Phase 1: 個別モデルCV ({n_models}モデル) ---\n", flush=True)

    all_cv_preds = [[] for _ in range(n_models)]
    individual_scores = []
    individual_fold_rmses = {}

    for m_idx, cfg in enumerate(MODEL_CFGS):
        t1 = time.time()
        predict_fn = PREDICT_FNS[cfg["type"]]
        fold_rmses = []
        for f_idx, (tr, te) in enumerate(folds):
            try:
                pred = predict_fn(X_raw[tr], X_raw[te], y[tr], groups[tr], cfg)
                all_cv_preds[m_idx].append(pred)
                fold_rmses.append(calc_rmse(y[te], pred))
            except Exception as e:
                fallback = np.full(len(te), y[tr].mean())
                all_cv_preds[m_idx].append(fallback)
                fold_rmses.append(999.0)
                print(f"  ERR {cfg['name']} fold{f_idx}: {e}")

        avg_rmse = np.mean(fold_rmses)
        std_rmse = np.std(fold_rmses)
        individual_scores.append((avg_rmse, m_idx))
        individual_fold_rmses[m_idx] = fold_rmses
        print(f"  [{m_idx+1:>2}/{n_models}] {cfg['name']:40s}: "
              f"RMSE={avg_rmse:.2f} +/- {std_rmse:.2f} "
              f"({time.time()-t1:.1f}s)", flush=True)

    # 個別ランキング
    sorted_scores = sorted(individual_scores)
    print("\n--- 個別モデルランキング ---")
    for rank, (r, i) in enumerate(sorted_scores, 1):
        cfg = MODEL_CFGS[i]
        print(f"  {rank:>2}. {r:.2f} | {cfg['name']} [{cfg['group']}]")

    # fold別RMSE一覧
    print("\n--- fold別RMSE ---")
    header = "  " + "".join(f"{s[:6]:>10s}" for s in sp_names)
    print(header)
    for _, m_idx in sorted_scores:
        fr = individual_fold_rmses[m_idx]
        vals = "".join(f"{r:10.2f}" for r in fr)
        print(f"  {MODEL_CFGS[m_idx]['name']:40s}{vals}")

    # ============================================================
    # Phase 2: アンサンブル戦略
    # ============================================================
    print("\n" + "=" * 70)
    print("Phase 2: アンサンブル戦略")
    print("=" * 70)

    all_strategies = []

    # --- Strategy 1: 全12モデル均等平均 ---
    all_idx = list(range(n_models))
    r1, fr1 = ensemble_eval(all_cv_preds, folds, y, all_idx)
    all_strategies.append({
        "name": "S1:Equal12",
        "rmse": r1,
        "fold_rmses": fr1,
        "idx": all_idx,
        "weights": np.ones(n_models) / n_models,
    })
    print(f"\n  S1: 全12モデル均等平均: RMSE={r1:.4f}")

    # --- Strategy 2: グループA+B ドメイン適応重視均等平均 ---
    ab_idx = [i for i, c in enumerate(MODEL_CFGS) if c["group"] in ("A", "B")]
    r2, fr2 = ensemble_eval(all_cv_preds, folds, y, ab_idx)
    all_strategies.append({
        "name": "S2:DA9_AB",
        "rmse": r2,
        "fold_rmses": fr2,
        "idx": ab_idx,
        "weights": np.ones(len(ab_idx)) / len(ab_idx),
    })
    print(f"  S2: A+B(9)均等平均: RMSE={r2:.4f}")

    # --- Strategy 3: Shrinkage (w = 0.5 * w_opt + 0.5 * w_uniform) ---
    print("  S3: Shrinkage最適化中...", flush=True)
    w_opt, r_opt = optimize_weights_nelder_mead(
        all_cv_preds, folds, y, all_idx, n_restarts=30)
    w_uniform = np.ones(n_models) / n_models
    w_shrink = 0.5 * w_opt + 0.5 * w_uniform
    w_shrink = w_shrink / w_shrink.sum()
    r3, fr3 = ensemble_eval(all_cv_preds, folds, y, all_idx, w_shrink)
    all_strategies.append({
        "name": "S3:Shrink50",
        "rmse": r3,
        "fold_rmses": fr3,
        "idx": all_idx,
        "weights": w_shrink,
    })
    print(f"  S3: Shrinkage(50%): RMSE={r3:.4f}")
    print(f"    (Nelder-Mead最適: {r_opt:.4f})")
    for i, wi in enumerate(w_shrink):
        if wi > 0.02:
            print(f"    {wi:.3f}: {MODEL_CFGS[i]['name']}")

    # --- Strategy 4: PLS系のみ(A1-A4 + C3)均等平均 ---
    pls_idx = [i for i, c in enumerate(MODEL_CFGS)
               if c["type"] == "pls"]
    r4, fr4 = ensemble_eval(all_cv_preds, folds, y, pls_idx)
    all_strategies.append({
        "name": "S4:PLS5",
        "rmse": r4,
        "fold_rmses": fr4,
        "idx": pls_idx,
        "weights": np.ones(len(pls_idx)) / len(pls_idx),
    })
    print(f"  S4: PLS系のみ({len(pls_idx)})均等平均: RMSE={r4:.4f}")

    # --- 全戦略ランキング ---
    all_strategies.sort(key=lambda x: x["rmse"])
    print("\n" + "=" * 70)
    print("全戦略ランキング")
    print("=" * 70)
    for rank, s in enumerate(all_strategies, 1):
        print(f"  {rank}. {s['name']}: RMSE={s['rmse']:.4f}")

    # --- fold別RMSE出力（全戦略） ---
    print("\n" + "=" * 70)
    print("fold別RMSE (全戦略)")
    print("=" * 70)
    for s in all_strategies:
        print(f"\n  {s['name']} (RMSE={s['rmse']:.4f}):")
        for sp, fr in zip(sp_names, s["fold_rmses"]):
            mark = " (ref)" if sp == "ベイスギ" else ""
            print(f"    {sp}: {fr:.2f}{mark}")
        non_beisugi = [fr for sp, fr in zip(sp_names, s["fold_rmses"])
                       if sp != "ベイスギ"]
        if non_beisugi:
            print(f"    除ベイスギ平均: {np.mean(non_beisugi):.4f}")

    # ============================================================
    # Phase 3: テスト予測生成
    # ============================================================
    print("\n" + "=" * 70)
    print("Phase 3: テスト予測生成")
    print("=" * 70)

    test_preds_all = []
    for m_idx, cfg in enumerate(MODEL_CFGS):
        t1 = time.time()
        predict_fn = PREDICT_FNS[cfg["type"]]
        try:
            tp = predict_fn(X_raw, X_test_raw, y, groups, cfg)
            test_preds_all.append(tp)
            print(f"  [{m_idx+1:>2}/{n_models}] {cfg['name']}: done ({time.time()-t1:.1f}s)",
                  flush=True)
        except Exception as e:
            test_preds_all.append(np.full(len(X_test_raw), y.mean()))
            print(f"  [{m_idx+1:>2}/{n_models}] {cfg['name']}: ERR {e}", flush=True)

    # --- 提出ファイル生成 ---
    strategy_filenames = {
        "S1:Equal12": "submission_v7_equal12.csv",
        "S2:DA9_AB": "submission_v7_da9.csv",
        "S3:Shrink50": "submission_v7_shrink.csv",
        "S4:PLS5": "submission_v7_pls5.csv",
    }

    print("\n--- 提出ファイル ---")
    for s in all_strategies:
        fname = strategy_filenames.get(s["name"], f"submission_v7_{s['name']}.csv")
        fpath = OUT_DIR / fname
        idx = s["idx"]
        w = s["weights"]
        preds = [test_preds_all[m] for m in idx]
        wn = np.array(w) / np.array(w).sum()
        test_pred = sum(wi * p for wi, p in zip(wn, preds))
        test_pred = np.clip(test_pred, 0, 300)

        save_submission(test_pred, test_sample_numbers, fpath)
        print(f"  {s['name']} (CV={s['rmse']:.4f}) -> {fpath}")

        # 予測の統計情報
        print(f"    mean={np.mean(test_pred):.1f}, "
              f"std={np.std(test_pred):.1f}, "
              f"min={np.min(test_pred):.1f}, max={np.max(test_pred):.1f}")

    # --- 結果CSV ---
    rows = []
    for s in all_strategies:
        rows.append({
            "strategy": s["name"],
            "cv_rmse": s["rmse"],
            "models": str([MODEL_CFGS[i]["name"] for i in s["idx"]]),
            "weights": str(s["weights"].tolist()),
        })
        for sp, fr in zip(sp_names, s["fold_rmses"]):
            rows[-1][f"fold_{sp}"] = fr
    result_csv = MODEL_OUT / "issue102_cycle10_v7_results.csv"
    pd.DataFrame(rows).to_csv(result_csv, index=False)
    print(f"\n結果CSV: {result_csv}")

    print(f"\n{'='*70}")
    print(f"完了. 総時間: {time.time()-t0:.0f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
