"""Issue #105: Final 18-model ensemble

Best12 (12モデル) + LightGBM (3モデル) + Wavelet/Detrend (2モデル) + LWPLS (1モデル)
= 18モデルのLOSO-CVを実行し、アンサンブル最適化→提出ファイル生成。

最適化: 同一pp+fs組のfeat_selをキャッシュして高速化。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import warnings
import time
warnings.filterwarnings("ignore")

from scipy.optimize import minimize

import lightgbm as lgb
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
from src.preprocessing.issue39_wavelet_transform import wavelet_denoise
from src.preprocessing.issue24_detrending import apply_detrending
from src.modeling.issue33_lwpls import lwpls_predict

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
SUB_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PREV_BEST = 13.40


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ============================================================
# 前処理 (キャッシュ付き)
# ============================================================
_preproc_cache = {}


def preproc(X_tr, X_te, g, name, fold_key=None):
    """前処理を適用。キャッシュでfold間の再計算を回避。"""
    cache_key = (name, fold_key) if fold_key is not None else None
    if cache_key and cache_key in _preproc_cache:
        return _preproc_cache[cache_key]

    if name == "SNV":
        result = apply_snv(X_tr), apply_snv(X_te)
    elif name == "EPO(1)":
        P = compute_epo_projection(X_tr, g, n_components=1)
        result = apply_epo(X_tr, P), apply_epo(X_te, P)
    elif name == "SNV+AsLS(1e6)":
        result = apply_asls(apply_snv(X_tr), lam=1e6), apply_asls(apply_snv(X_te), lam=1e6)
    elif name == "PMSC":
        ref = compute_msc_reference(X_tr)
        result = apply_piecewise_msc(X_tr, ref, 3), apply_piecewise_msc(X_te, ref, 3)
    elif name == "SG2d+EPO(1)":
        xs = apply_savgol(X_tr, deriv=2, window_length=7)
        xst = apply_savgol(X_te, deriv=2, window_length=7)
        P = compute_epo_projection(xs, g, n_components=1)
        result = apply_epo(xs, P), apply_epo(xst, P)
    elif name == "SNV+SG2d":
        result = (apply_savgol(apply_snv(X_tr), deriv=2, window_length=7),
                  apply_savgol(apply_snv(X_te), deriv=2, window_length=7))
    elif name == "Wavelet":
        result = wavelet_denoise(X_tr), wavelet_denoise(X_te)
    elif name == "Detrend":
        result = apply_detrending(X_tr), apply_detrending(X_te)
    else:
        result = X_tr.copy(), X_te.copy()

    if cache_key:
        _preproc_cache[cache_key] = result
    return result


# ============================================================
# 特徴選択 (キャッシュ付き)
# ============================================================
_fs_cache = {}


def feat_sel(X_tr, X_te, y, name, cache_key=None):
    if not name:
        return X_tr, X_te
    full_key = (name, cache_key) if cache_key else None
    if full_key and full_key in _fs_cache:
        selected = _fs_cache[full_key]
        return X_tr[:, selected], X_te[:, selected]

    if name.startswith("siPLS"):
        p = name.replace("siPLS(", "").rstrip(")").split(",")
        X_tr_sel, X_te_sel, sel = sipls_select(
            X_tr, y, X_te, n_intervals=int(p[0]), n_components=3, n_combine=int(p[1])
        )
        if full_key:
            _fs_cache[full_key] = sel
        return X_tr_sel, X_te_sel
    elif name.startswith("iPLS"):
        ni = int(name.replace("iPLS(", "").rstrip(")"))
        X_tr_sel, X_te_sel, sel = ipls_select(
            X_tr, y, X_te, n_intervals=ni, n_components=3, n_best=1
        )
        if full_key:
            _fs_cache[full_key] = sel
        return X_tr_sel, X_te_sel
    return X_tr, X_te


# ============================================================
# PLSスコア・逆変換
# ============================================================
def get_pls_scores(X_tr, X_te, y, nc, tf):
    nc = max(1, min(nc, X_tr.shape[1] - 1))
    yf = np.sqrt(y) if tf == "sqrt" else y.copy()
    pls = PLSRegression(n_components=nc)
    pls.fit(X_tr, yf)
    return pls.transform(X_tr), pls.transform(X_te), pls, yf


def inv_tf(pred, tf):
    if tf == "sqrt":
        return np.clip(pred, 0, None) ** 2
    return pred


# ============================================================
# モデル予測関数
# ============================================================
def pred_pls(Xtr, Xte, y, nc, tf):
    _, _, pls, yf = get_pls_scores(Xtr, Xte, y, nc, tf)
    return inv_tf(pls.predict(Xte).ravel(), tf)


def pred_gbr(Xtr, Xte, y, nc, tf, n_est=200, md=3, lr=0.05):
    Ttr, Tte, _, yf = get_pls_scores(Xtr, Xte, y, nc, tf)
    gbr = GradientBoostingRegressor(
        n_estimators=n_est, max_depth=md, learning_rate=lr,
        subsample=0.8, min_samples_leaf=5, random_state=42,
        validation_fraction=0.15, n_iter_no_change=50, tol=0.01,
    )
    gbr.fit(Ttr, yf)
    return inv_tf(gbr.predict(Tte), tf)


def pred_huber(Xtr, Xte, y, nc, tf, eps=1.35):
    Ttr, Tte, _, yf = get_pls_scores(Xtr, Xte, y, nc, tf)
    sc = StandardScaler()
    Ttr_s, Tte_s = sc.fit_transform(Ttr), sc.transform(Tte)
    h = HuberRegressor(epsilon=eps, max_iter=200, alpha=0.01)
    h.fit(Ttr_s, yf)
    return inv_tf(h.predict(Tte_s), tf)


def pred_lgbm(Xtr, Xte, y, nc, tf):
    Ttr, Tte, _, yf = get_pls_scores(Xtr, Xte, y, nc, tf)
    n_val = max(1, int(len(yf) * 0.15))
    indices = np.random.RandomState(42).permutation(len(yf))
    val_idx, trn_idx = indices[:n_val], indices[n_val:]
    dtrain = lgb.Dataset(Ttr[trn_idx], label=yf[trn_idx])
    dval = lgb.Dataset(Ttr[val_idx], label=yf[val_idx], reference=dtrain)
    params = {
        "objective": "regression", "metric": "rmse",
        "learning_rate": 0.05, "max_depth": 4, "num_leaves": 15,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 5,
        "random_state": 42, "verbose": -1,
    }
    model = lgb.train(params, dtrain, num_boost_round=500, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    return inv_tf(model.predict(Tte), tf)


def pred_lwpls(Xtr, Xte, y, nc, tf, k=500):
    """LWPLS on PLS score space for speed."""
    Ttr, Tte, _, yf = get_pls_scores(Xtr, Xte, y, nc, tf)
    nc_lw = max(1, min(nc, Ttr.shape[1] - 1))
    pred = lwpls_predict(Ttr, yf, Tte, n_components=nc_lw, k=min(k, len(yf)))
    return inv_tf(pred, tf)


# ============================================================
# モデル実行ディスパッチ
# ============================================================
def run_model(X_tr, X_te, y, g, cfg, fold_key=None):
    Xtr, Xte = preproc(X_tr, X_te, g, cfg["pp"], fold_key=fold_key)
    fs_cache_key = (cfg["pp"], fold_key) if fold_key is not None else None
    Xtr, Xte = feat_sel(Xtr, Xte, y, cfg.get("fs"), cache_key=fs_cache_key)
    t = cfg["type"]
    if t == "pls":
        return pred_pls(Xtr, Xte, y, cfg.get("nc", 4), cfg["tf"])
    elif t == "gbr":
        return pred_gbr(Xtr, Xte, y, cfg.get("nc", 4), cfg["tf"],
                         cfg.get("ne", 200), cfg.get("md", 3), cfg.get("lr", 0.05))
    elif t == "huber":
        return pred_huber(Xtr, Xte, y, cfg.get("nc", 4), cfg["tf"],
                          cfg.get("eps", 1.35))
    elif t == "lgbm":
        return pred_lgbm(Xtr, Xte, y, cfg.get("nc", 4), cfg["tf"])
    elif t == "lwpls":
        return pred_lwpls(Xtr, Xte, y, cfg.get("nc", 4), cfg["tf"],
                           cfg.get("k", 500))
    else:
        raise ValueError(f"Unknown model type: {t}")


# ============================================================
# 18モデル設定
# ============================================================
def get_model_configs():
    return [
        # 既存12モデル
        {"name": "M1:EPO+PLS4+sqrt", "pp": "EPO(1)", "nc": 4, "tf": "sqrt", "type": "pls"},
        {"name": "M3:SNV+iPLS50+PLS4+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt", "fs": "iPLS(50)", "type": "pls"},
        {"name": "M5:SNV+AsLS+siPLS+PLS5+sqrt", "pp": "SNV+AsLS(1e6)", "nc": 5, "tf": "sqrt", "fs": "siPLS(30,3)", "type": "pls"},
        {"name": "C1:SNV+iPLS50+PLS5+sqrt", "pp": "SNV", "nc": 5, "tf": "sqrt", "fs": "iPLS(50)", "type": "pls"},
        {"name": "H2:SNV+iPLS50+Huber+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt", "fs": "iPLS(50)", "type": "huber"},
        {"name": "H2a:SNV+iPLS50+Huber(1.1)+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt", "fs": "iPLS(50)", "eps": 1.1, "type": "huber"},
        {"name": "H5:PMSC+siPLS+Huber+sqrt", "pp": "PMSC", "nc": 4, "tf": "sqrt", "fs": "siPLS(30,3)", "type": "huber"},
        {"name": "H6:SNV+AsLS+siPLS+Huber+sqrt", "pp": "SNV+AsLS(1e6)", "nc": 4, "tf": "sqrt", "fs": "siPLS(30,3)", "type": "huber"},
        {"name": "N10:PMSC+siPLS+GBR+sqrt", "pp": "PMSC", "nc": 4, "tf": "sqrt", "fs": "siPLS(30,3)", "type": "gbr"},
        {"name": "G9:SG2d+EPO+GBR+raw", "pp": "SG2d+EPO(1)", "nc": 4, "tf": "raw", "type": "gbr"},
        {"name": "Gnew1:EPO+GBR150+raw", "pp": "EPO(1)", "nc": 4, "tf": "raw", "ne": 150, "lr": 0.08, "type": "gbr"},
        {"name": "N2:SNV+SG2d+GBR+raw", "pp": "SNV+SG2d", "nc": 4, "tf": "raw", "type": "gbr"},
        # LightGBM 3モデル
        {"name": "LGB1:SNV+AsLS+siPLS+PLS5+sqrt", "pp": "SNV+AsLS(1e6)", "nc": 5, "tf": "sqrt", "fs": "siPLS(30,3)", "type": "lgbm"},
        {"name": "LGB2:SNV+AsLS+siPLS+PLS4+sqrt", "pp": "SNV+AsLS(1e6)", "nc": 4, "tf": "sqrt", "fs": "siPLS(30,3)", "type": "lgbm"},
        {"name": "LGB3:EPO+PLS6+raw", "pp": "EPO(1)", "nc": 6, "tf": "raw", "type": "lgbm"},
        # Wavelet/Detrend 2モデル
        {"name": "Wavelet+PLS4+sqrt", "pp": "Wavelet", "nc": 4, "tf": "sqrt", "type": "pls"},
        {"name": "Detrend+PLS4+sqrt", "pp": "Detrend", "nc": 4, "tf": "sqrt", "type": "pls"},
        # LWPLS 1モデル
        {"name": "LWPLS:SNV+nc4+k500+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt", "type": "lwpls", "k": 500},
    ]


# ============================================================
# アンサンブル評価・最適化
# ============================================================
def evaluate(all_preds, folds, y, idx, w=None):
    fold_rmses = []
    for fi, (_, te) in enumerate(folds):
        preds = [all_preds[m][fi] for m in idx]
        if w is not None:
            wn = np.array(w); wn = wn / wn.sum()
            ens = sum(wi * p for wi, p in zip(wn, preds))
        else:
            ens = np.mean(preds, axis=0)
        fold_rmses.append(rmse(y[te], np.clip(ens, 0, 300)))
    return np.mean(fold_rmses), fold_rmses


def opt_weights(all_preds, folds, y, idx, n_restarts=30):
    nn = len(idx)
    def obj(w):
        wn = np.abs(w) / np.sum(np.abs(w))
        r, _ = evaluate(all_preds, folds, y, idx, wn)
        return r
    best_r, best_w = np.inf, np.ones(nn) / nn
    for s in range(n_restarts):
        w0 = np.random.RandomState(s).dirichlet(np.ones(nn))
        res = minimize(obj, w0, method="Nelder-Mead", options={"maxiter": 5000})
        if res.fun < best_r:
            best_r = res.fun
            best_w = np.abs(res.x) / np.sum(np.abs(res.x))
    return best_w, best_r


# ============================================================
# テストデータ予測
# ============================================================
def predict_test(X_train, y_train, g_train, X_test, cfg):
    Xtr, Xte = preproc(X_train, X_test, g_train, cfg["pp"], fold_key="test")
    Xtr, Xte = feat_sel(Xtr, Xte, y_train, cfg.get("fs"),
                         cache_key=(cfg["pp"], "test"))
    t = cfg["type"]
    if t == "pls":
        return pred_pls(Xtr, Xte, y_train, cfg.get("nc", 4), cfg["tf"])
    elif t == "gbr":
        return pred_gbr(Xtr, Xte, y_train, cfg.get("nc", 4), cfg["tf"],
                         cfg.get("ne", 200), cfg.get("md", 3), cfg.get("lr", 0.05))
    elif t == "huber":
        return pred_huber(Xtr, Xte, y_train, cfg.get("nc", 4), cfg["tf"],
                          cfg.get("eps", 1.35))
    elif t == "lgbm":
        return pred_lgbm(Xtr, Xte, y_train, cfg.get("nc", 4), cfg["tf"])
    elif t == "lwpls":
        return pred_lwpls(Xtr, Xte, y_train, cfg.get("nc", 4), cfg["tf"],
                           cfg.get("k", 500))
    else:
        raise ValueError(f"Unknown: {t}")


# ============================================================
# メイン
# ============================================================
def main():
    t0 = time.time()
    df = load_train(DATA_DIR)
    sc_cols = get_spectral_columns(df)
    X = df[sc_cols].values
    y = df["含水率"].values
    g = df["樹種"].values
    logo = LeaveOneGroupOut()
    folds = list(logo.split(X, y, g))
    species_list = [np.unique(g[te])[0] for _, te in folds]

    print("=" * 70)
    print("Issue #105: Final 18-model Ensemble (cached)")
    print("=" * 70)

    cfgs = get_model_configs()
    n = len(cfgs)
    all_preds = [[] for _ in range(n)]

    beisugi_fold_idx = None
    for fi, sp in enumerate(species_list):
        if sp == "ベイスギ":
            beisugi_fold_idx = fi
            break

    # ============================================================
    # Phase 1: 全モデルのLOSO-CV (fold単位でキャッシュ活用)
    # ============================================================
    print(f"\n--- Phase 1: 全{n}モデルのLOSO-CV ---\n", flush=True)
    indiv_rmses = []
    for i, cfg in enumerate(cfgs):
        t1 = time.time()
        fold_rmses = []
        for fi, (tr, te) in enumerate(folds):
            try:
                pred = run_model(X[tr], X[te], y[tr], g[tr], cfg, fold_key=fi)
                all_preds[i].append(pred)
                fold_rmses.append(rmse(y[te], pred))
            except Exception as e:
                fallback = np.full(len(te), y[tr].mean())
                all_preds[i].append(fallback)
                fold_rmses.append(999.0)
                print(f"  ERR {cfg['name']} fold{fi}: {e}")
        r_mean = np.mean(fold_rmses)
        indiv_rmses.append(r_mean)
        r_bei = fold_rmses[beisugi_fold_idx] if beisugi_fold_idx is not None else 0
        non_bei = [r for j, r in enumerate(fold_rmses) if j != beisugi_fold_idx]
        print(f"  [{i+1:>2}/{n}] {cfg['name']}: {r_mean:.2f} "
              f"(bei={r_bei:.1f}, 除bei={np.mean(non_bei):.2f}) [{time.time()-t1:.1f}s]",
              flush=True)

    # 個別ランキング
    indiv = sorted([(r, i) for i, r in enumerate(indiv_rmses)])
    print("\n--- 個別ランキング ---")
    for rank, (r, i) in enumerate(indiv):
        cat = "[B12]" if i < 12 else "[LGB]" if i < 15 else "[NEW]"
        print(f"  {rank+1:>2}. {r:.2f} | {cfgs[i]['name']} {cat}")

    # ============================================================
    # Phase 2: 均等平均アンサンブル
    # ============================================================
    print(f"\n--- Phase 2: 均等平均アンサンブル ---\n", flush=True)
    top_idx = [i for _, i in indiv]
    for k in range(10, n + 1):
        idx_k = top_idx[:k]
        r, _ = evaluate(all_preds, folds, y, idx_k)
        print(f"  Equal-{k}: RMSE={r:.4f}")

    # ============================================================
    # Phase 3: Nelder-Mead重み最適化 (k=12)
    # ============================================================
    print(f"\n--- Phase 3: Nelder-Mead重み最適化 (k=12) ---\n", flush=True)
    k_opt = 12
    best_idx_12 = top_idx[:k_opt]
    r_equal_12, fr_equal_12 = evaluate(all_preds, folds, y, best_idx_12)
    print(f"  Equal-12 baseline: RMSE={r_equal_12:.4f}")

    w_opt, r_opt = opt_weights(all_preds, folds, y, best_idx_12, n_restarts=20)
    _, fr_opt = evaluate(all_preds, folds, y, best_idx_12, w_opt)
    print(f"  Opt-12:   RMSE={r_opt:.4f}")
    print(f"  改善: {r_equal_12:.4f} -> {r_opt:.4f} ({r_equal_12 - r_opt:+.4f})")
    print("\n  重み:")
    for ci, wi in zip(best_idx_12, w_opt):
        print(f"    {wi:.4f}: {cfgs[ci]['name']}")

    # fold別RMSE
    print("\n--- fold別RMSE ---")
    for fi, sp in enumerate(species_list):
        print(f"  {sp}: Equal={fr_equal_12[fi]:.2f}, Opt={fr_opt[fi]:.2f}")

    # ============================================================
    # Phase 4: テスト予測・提出ファイル生成
    # ============================================================
    print(f"\n--- Phase 4: テスト予測 ---\n", flush=True)
    df_test = load_test(DATA_DIR)
    X_test = df_test[sc_cols].values
    test_ids = df_test["sample number"].values

    all_test_preds = []
    for i, cfg in enumerate(cfgs):
        t1 = time.time()
        try:
            pred = predict_test(X, y, g, X_test, cfg)
            all_test_preds.append(pred)
            print(f"  [{i+1:>2}/{n}] {cfg['name']}: mean={pred.mean():.2f} [{time.time()-t1:.1f}s]",
                  flush=True)
        except Exception as e:
            print(f"  ERR {cfg['name']}: {e}")
            all_test_preds.append(np.full(len(X_test), y.mean()))

    # --- 提出ファイル生成 ---
    print(f"\n--- 提出ファイル生成 ---\n", flush=True)

    # 1. Best12均等平均
    preds_equal_12 = np.mean([all_test_preds[ci] for ci in best_idx_12], axis=0)
    preds_equal_12 = np.clip(preds_equal_12, 0, 300)
    sub_equal = pd.DataFrame({"id": test_ids, "pred": preds_equal_12})
    path_equal = SUB_DIR / "submission_v12_final_equal12.csv"
    sub_equal.to_csv(path_equal, index=False, header=False)
    print(f"  {path_equal.name} (mean={preds_equal_12.mean():.2f})")

    # 2. Best12 Nelder-Mead最適化
    wn = np.array(w_opt) / np.sum(w_opt)
    preds_opt_12 = sum(wi * all_test_preds[ci] for wi, ci in zip(wn, best_idx_12))
    preds_opt_12 = np.clip(preds_opt_12, 0, 300)
    sub_opt = pd.DataFrame({"id": test_ids, "pred": preds_opt_12})
    path_best = SUB_DIR / "submission_v12_final_opt12.csv"
    sub_opt.to_csv(path_best, index=False, header=False)
    print(f"  {path_best.name} (mean={preds_opt_12.mean():.2f})")

    # 3. LightGBM重視 (B12均等:LGB3均等 = 0.7:0.3)
    b12_test = np.mean(all_test_preds[:12], axis=0)
    lgb_test = np.mean(all_test_preds[12:15], axis=0)
    lgbm_heavy = np.clip(0.7 * b12_test + 0.3 * lgb_test, 0, 300)
    sub_lgbm = pd.DataFrame({"id": test_ids, "pred": lgbm_heavy})
    path_lgbm = SUB_DIR / "submission_v12_final_blend_lgbm_heavy.csv"
    sub_lgbm.to_csv(path_lgbm, index=False, header=False)
    print(f"  {path_lgbm.name} (mean={lgbm_heavy.mean():.2f})")

    # ============================================================
    # 結果保存
    # ============================================================
    out_path = OUT_DIR / "issue105_final_ensemble_results.csv"
    rows = [
        {"method": "Equal-12", "rmse": r_equal_12,
         "models": str([cfgs[ci]["name"] for ci in best_idx_12])},
        {"method": "Opt-12", "rmse": r_opt,
         "models": str([cfgs[ci]["name"] for ci in best_idx_12]),
         "weights": str([f"{wi:.4f}" for wi in w_opt])},
    ]
    pd.DataFrame(rows).to_csv(out_path, index=False)

    indiv_rows = [{"model": cfgs[i]["name"], "rmse": indiv_rmses[i]} for i in range(n)]
    pd.DataFrame(indiv_rows).to_csv(OUT_DIR / "issue105_individual_model_rmses.csv", index=False)

    print(f"\n結果保存: {out_path}")
    print(f"\n{'='*70}")
    print(f"最終結果:")
    print(f"  Equal-12 LOSO-CV RMSE: {r_equal_12:.4f}")
    print(f"  Opt-12   LOSO-CV RMSE: {r_opt:.4f}")
    print(f"  前ベスト: {PREV_BEST:.4f}")
    print(f"  提出: {path_equal.name}, {path_best.name}, {path_lgbm.name}")
    print(f"  総実行時間: {time.time() - t0:.0f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
