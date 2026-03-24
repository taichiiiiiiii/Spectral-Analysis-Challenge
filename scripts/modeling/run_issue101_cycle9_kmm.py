"""Issue #101: サイクル9 - KMM (Kernel Mean Matching) によるサンプル重み付け

KMMでソースドメイン（train）の重みを最適化し、
重み付きソース分布がターゲットドメイン（test）に一致するようにする。
PCA次元削減後にKMM適用（元の1555次元では不安定）。
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
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut

from src.eda.data_loader import load_train, load_test, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue50_kmm import compute_kmm_weights

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def preprocess(X_tr_raw, X_te_raw, groups_train, pp):
    """前処理を適用する。"""
    if pp == "SNV":
        return apply_snv(X_tr_raw), apply_snv(X_te_raw)
    elif pp == "EPO(1)":
        P = compute_epo_projection(X_tr_raw, groups_train, n_components=1)
        return apply_epo(X_tr_raw, P), apply_epo(X_te_raw, P)
    elif pp == "raw":
        return X_tr_raw.copy(), X_te_raw.copy()
    return X_tr_raw.copy(), X_te_raw.copy()


def compute_kmm_for_fold(X_tr_pp, X_te_pp, pca_dim, kmm_B, kmm_gamma=None):
    """PCA次元削減後にKMM重みを計算する。

    Parameters
    ----------
    X_tr_pp : 前処理済みtrain (n_tr, d)
    X_te_pp : 前処理済みtest (n_te, d)
    pca_dim : PCA次元数
    kmm_B : KMM重み上限
    kmm_gamma : RBFカーネルパラメータ（Noneでmedian heuristic）

    Returns
    -------
    weights : (n_tr,) KMM重みベクトル
    """
    pca = PCA(n_components=min(pca_dim, X_tr_pp.shape[1], X_tr_pp.shape[0]))
    X_tr_pca = pca.fit_transform(X_tr_pp)
    X_te_pca = pca.transform(X_te_pp)
    weights = compute_kmm_weights(X_tr_pca, X_te_pca, gamma=kmm_gamma, B=kmm_B)
    return weights


def predict_fold_kmm_ridge(X_tr_raw, X_te_raw, y_train, groups_train, cfg):
    """KMM重み付きRidge on PLSスコア"""
    pp = cfg["pp"]
    tf = cfg["tf"]
    nc = cfg.get("nc", 4)
    pca_dim = cfg.get("pca_dim", 50)
    kmm_B = cfg.get("kmm_B", 10.0)
    alpha = cfg.get("alpha", 1.0)

    X_tr, X_te = preprocess(X_tr_raw, X_te_raw, groups_train, pp)

    # KMM重み計算
    weights = compute_kmm_for_fold(X_tr, X_te, pca_dim, kmm_B)

    # PLSスコア抽出
    y_fit = np.sqrt(y_train) if tf == "sqrt" else y_train.copy()
    nc_actual = max(1, min(nc, X_tr.shape[1] - 1))
    pls = PLSRegression(n_components=nc_actual)
    pls.fit(X_tr, y_fit)
    T_tr = pls.transform(X_tr)
    T_te = pls.transform(X_te)

    # 重み付きRidge
    ridge = Ridge(alpha=alpha)
    ridge.fit(T_tr, y_fit, sample_weight=weights)
    pred = ridge.predict(T_te).ravel()

    if tf == "sqrt":
        pred = np.clip(pred, 0, None) ** 2
    return pred


def predict_fold_kmm_gbr(X_tr_raw, X_te_raw, y_train, groups_train, cfg):
    """KMM重み付きGBR on PLSスコア"""
    pp = cfg["pp"]
    tf = cfg["tf"]
    nc = cfg.get("nc", 4)
    pca_dim = cfg.get("pca_dim", 50)
    kmm_B = cfg.get("kmm_B", 10.0)
    n_est = cfg.get("n_est", 200)
    max_depth = cfg.get("max_depth", 3)
    lr = cfg.get("lr", 0.05)

    X_tr, X_te = preprocess(X_tr_raw, X_te_raw, groups_train, pp)

    # KMM重み計算
    weights = compute_kmm_for_fold(X_tr, X_te, pca_dim, kmm_B)

    # PLSスコア抽出
    y_fit = np.sqrt(y_train) if tf == "sqrt" else y_train.copy()
    nc_actual = max(1, min(nc, X_tr.shape[1] - 1))
    pls = PLSRegression(n_components=nc_actual)
    pls.fit(X_tr, y_fit)
    T_tr = pls.transform(X_tr)
    T_te = pls.transform(X_te)

    # 重み付きGBR
    gbr = GradientBoostingRegressor(
        n_estimators=n_est, max_depth=max_depth,
        learning_rate=lr, subsample=0.8,
        min_samples_leaf=5, random_state=42,
        validation_fraction=0.15, n_iter_no_change=50, tol=0.01,
    )
    gbr.fit(T_tr, y_fit, sample_weight=weights)
    pred = gbr.predict(T_te)

    if tf == "sqrt":
        pred = np.clip(pred, 0, None) ** 2
    return pred


def predict_fold_kmm_wpls(X_tr_raw, X_te_raw, y_train, groups_train, cfg):
    """KMM重み付きPLS（sqrt(w)で重み付け）"""
    pp = cfg["pp"]
    tf = cfg["tf"]
    nc = cfg.get("nc", 4)
    pca_dim = cfg.get("pca_dim", 50)
    kmm_B = cfg.get("kmm_B", 10.0)

    X_tr, X_te = preprocess(X_tr_raw, X_te_raw, groups_train, pp)

    # KMM重み計算
    weights = compute_kmm_for_fold(X_tr, X_te, pca_dim, kmm_B)

    # 重み付きPLS
    y_fit = np.sqrt(y_train) if tf == "sqrt" else y_train.copy()
    nc_actual = max(1, min(nc, X_tr.shape[1] - 1))
    sqrt_w = np.sqrt(weights)
    X_w = X_tr * sqrt_w[:, None]
    y_w = y_fit * sqrt_w
    pls = PLSRegression(n_components=nc_actual)
    pls.fit(X_w, y_w)
    pred = pls.predict(X_te).ravel()

    if tf == "sqrt":
        pred = np.clip(pred, 0, None) ** 2
    return pred


def predict_fold_nokmm_pls(X_tr_raw, X_te_raw, y_train, groups_train, cfg):
    """ベースライン: KMM無しPLS"""
    pp = cfg["pp"]
    tf = cfg["tf"]
    nc = cfg.get("nc", 4)

    X_tr, X_te = preprocess(X_tr_raw, X_te_raw, groups_train, pp)

    y_fit = np.sqrt(y_train) if tf == "sqrt" else y_train.copy()
    nc_actual = max(1, min(nc, X_tr.shape[1] - 1))
    pls = PLSRegression(n_components=nc_actual)
    pls.fit(X_tr, y_fit)
    pred = pls.predict(X_te).ravel()

    if tf == "sqrt":
        pred = np.clip(pred, 0, None) ** 2
    return pred


def predict_fold_nokmm_gbr(X_tr_raw, X_te_raw, y_train, groups_train, cfg):
    """ベースライン: KMM無しGBR on PLSスコア"""
    pp = cfg["pp"]
    tf = cfg["tf"]
    nc = cfg.get("nc", 4)
    n_est = cfg.get("n_est", 200)
    max_depth = cfg.get("max_depth", 3)
    lr = cfg.get("lr", 0.05)

    X_tr, X_te = preprocess(X_tr_raw, X_te_raw, groups_train, pp)

    y_fit = np.sqrt(y_train) if tf == "sqrt" else y_train.copy()
    nc_actual = max(1, min(nc, X_tr.shape[1] - 1))
    pls = PLSRegression(n_components=nc_actual)
    pls.fit(X_tr, y_fit)
    T_tr = pls.transform(X_tr)
    T_te = pls.transform(X_te)

    gbr = GradientBoostingRegressor(
        n_estimators=n_est, max_depth=max_depth,
        learning_rate=lr, subsample=0.8,
        min_samples_leaf=5, random_state=42,
        validation_fraction=0.15, n_iter_no_change=50, tol=0.01,
    )
    gbr.fit(T_tr, y_fit)
    pred = gbr.predict(T_te)

    if tf == "sqrt":
        pred = np.clip(pred, 0, None) ** 2
    return pred


def main():
    t0 = time.time()
    df_train = load_train(DATA_DIR)
    df_test = load_test(DATA_DIR)
    sc = get_spectral_columns(df_train)
    X_raw = df_train[sc].values
    y = df_train["含水率"].values
    groups = df_train["樹種"].values
    X_test_raw = df_test[sc].values

    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_raw, y, groups))
    sp_names = [np.unique(groups[te])[0] for _, te in folds]

    print("=" * 70)
    print("Issue #101: サイクル9 - KMM (Kernel Mean Matching) サンプル重み付け")
    print(f"  ベストスコア: 17.03")
    print("=" * 70)

    # --- モデル定義 ---
    # KMM重みはfold内で再利用するため、事前にキャッシュする
    # key: (pp, pca_dim, kmm_B, fold_idx) -> weights
    kmm_cache = {}

    def get_kmm_weights_cached(X_tr_raw, X_te_raw, groups_train, pp, pca_dim, kmm_B, fold_idx):
        key = (pp, pca_dim, kmm_B, fold_idx)
        if key not in kmm_cache:
            X_tr, X_te = preprocess(X_tr_raw, X_te_raw, groups_train, pp)
            kmm_cache[key] = compute_kmm_for_fold(X_tr, X_te, pca_dim, kmm_B)
        return kmm_cache[key]

    models = [
        # ベースライン（KMM無し）
        {"name": "BL1:EPO+PLS4+sqrt", "pp": "EPO(1)", "nc": 4, "tf": "sqrt",
         "func": predict_fold_nokmm_pls},
        {"name": "BL2:SNV+PLS4+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt",
         "func": predict_fold_nokmm_pls},
        {"name": "BL3:EPO+GBR+raw", "pp": "EPO(1)", "nc": 4, "tf": "raw",
         "func": predict_fold_nokmm_gbr},

        # KMM重み付きRidge: 代表的な設定
        {"name": "KR1:EPO+KMM20+Ridge+sqrt", "pp": "EPO(1)", "nc": 4, "tf": "sqrt",
         "pca_dim": 20, "kmm_B": 10.0, "alpha": 1.0,
         "func": predict_fold_kmm_ridge},
        {"name": "KR2:SNV+KMM20+Ridge+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt",
         "pca_dim": 20, "kmm_B": 10.0, "alpha": 1.0,
         "func": predict_fold_kmm_ridge},
        {"name": "KR3:EPO+KMM50+Ridge+sqrt", "pp": "EPO(1)", "nc": 4, "tf": "sqrt",
         "pca_dim": 50, "kmm_B": 10.0, "alpha": 1.0,
         "func": predict_fold_kmm_ridge},
        {"name": "KR4:raw+KMM20+Ridge+sqrt", "pp": "raw", "nc": 4, "tf": "sqrt",
         "pca_dim": 20, "kmm_B": 10.0, "alpha": 1.0,
         "func": predict_fold_kmm_ridge},
        {"name": "KR5:EPO+KMM100+Ridge+sqrt", "pp": "EPO(1)", "nc": 4, "tf": "sqrt",
         "pca_dim": 100, "kmm_B": 10.0, "alpha": 1.0,
         "func": predict_fold_kmm_ridge},

        # KMM重み付きGBR: 代表的な設定
        {"name": "KG1:EPO+KMM20+GBR+raw", "pp": "EPO(1)", "nc": 4, "tf": "raw",
         "pca_dim": 20, "kmm_B": 10.0,
         "func": predict_fold_kmm_gbr},
        {"name": "KG2:SNV+KMM20+GBR+raw", "pp": "SNV", "nc": 4, "tf": "raw",
         "pca_dim": 20, "kmm_B": 10.0,
         "func": predict_fold_kmm_gbr},
        {"name": "KG3:EPO+KMM20+GBR+sqrt", "pp": "EPO(1)", "nc": 4, "tf": "sqrt",
         "pca_dim": 20, "kmm_B": 10.0,
         "func": predict_fold_kmm_gbr},

        # KMM重み付きPLS
        {"name": "KP1:EPO+KMM20+wPLS+sqrt", "pp": "EPO(1)", "nc": 4, "tf": "sqrt",
         "pca_dim": 20, "kmm_B": 10.0,
         "func": predict_fold_kmm_wpls},
        {"name": "KP2:SNV+KMM20+wPLS+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt",
         "pca_dim": 20, "kmm_B": 10.0,
         "func": predict_fold_kmm_wpls},
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
        avg_rmse = np.mean(fold_rmses)
        std_rmse = np.std(fold_rmses)
        print(f"  [{m_idx+1:>2}/{n_models}] {cfg['name']}: "
              f"{avg_rmse:.2f} +/- {std_rmse:.2f} ({time.time()-t1:.1f}s)", flush=True)

    # KMM重み統計を表示
    print("\n--- KMM重み統計 ---")
    for m_idx, cfg in enumerate(models):
        if "kmm" in cfg["name"].lower() or "KMM" in cfg["name"]:
            pass  # 重みは各fold内で計算するので個別には表示しない

    # --- アンサンブル ---
    def eval_ens(indices, weights=None):
        fold_rmses = []
        for f_idx, (_, te) in enumerate(folds):
            preds = [all_preds[m][f_idx] for m in indices]
            if weights is not None:
                w = np.array(weights)
                w = w / w.sum()
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

    # 個別ランキング
    indiv = sorted([(eval_ens([i])[0], i) for i in range(n_models)])
    print("\n--- 個別ランキング ---")
    for r, i in indiv:
        print(f"  {r:.2f} | {models[i]['name']}")

    # ベースラインとKMMモデルの比較
    bl_indices = [i for i, m in enumerate(models) if m["name"].startswith("BL")]
    kmm_indices = [i for i, m in enumerate(models) if not m["name"].startswith("BL")]

    print("\n--- ベースライン vs KMM比較 ---")
    for bi in bl_indices:
        bl_r, _ = eval_ens([bi])
        print(f"  {models[bi]['name']}: {bl_r:.2f}")
    print("  ---")
    for ki in kmm_indices:
        k_r, _ = eval_ens([ki])
        print(f"  {models[ki]['name']}: {k_r:.2f}")

    # アンサンブル探索
    top14 = [i for _, i in indiv[:14]]
    results = []
    for k in [3, 4, 5, 6, 7, 8]:
        best_r_eq, best_combo_eq = np.inf, None
        for combo in combinations(top14, k):
            r, _ = eval_ens(list(combo))
            if r < best_r_eq:
                best_r_eq = r
                best_combo_eq = list(combo)
        if best_combo_eq:
            w, r_w = opt_w(best_combo_eq)
            results.append((f"BestCombo{k}-W", r_w, best_combo_eq, w))
            print(f"  BestCombo{k}-W: {r_w:.4f}", flush=True)
            for i, wi in zip(best_combo_eq, w):
                if wi > 0.02:
                    print(f"    {wi:.3f}: {models[i]['name']}")

    # ベスト結果
    if results:
        results.sort(key=lambda x: x[1])
        best = results[0]
        print(f"\n{'='*70}")
        print(f"BEST ENSEMBLE: {best[1]:.4f} ({best[0]})")
        print(f"ベストスコア: 17.03, 差: {17.03 - best[1]:+.4f}")
        print(f"{'='*70}")

        _, fr = eval_ens(best[2], best[3])
        print("\nfold詳細:")
        for sp, r in zip(sp_names, fr):
            tag = " *" if sp == "ベイスギ" else ""
            print(f"  {sp}: {r:.2f}{tag}")

        # --- テスト予測 (ベストアンサンブル) ---
        print("\n--- テスト予測 ---")
        best_indices = best[2]
        best_weights = best[3]
        test_preds_all = []

        for m_idx in best_indices:
            cfg = models[m_idx]
            try:
                pred = cfg["func"](X_raw, X_test_raw, y, groups, cfg)
                test_preds_all.append(pred)
                print(f"  {cfg['name']}: mean={pred.mean():.2f}, std={pred.std():.2f}")
            except Exception as e:
                test_preds_all.append(np.full(len(X_test_raw), y.mean()))
                print(f"  ERR {cfg['name']}: {e}")

        # 重み付きアンサンブル
        w_norm = best_weights / best_weights.sum()
        test_pred = sum(wi * p for wi, p in zip(w_norm, test_preds_all))
        test_pred = np.clip(test_pred, 0, 300)

        print(f"\n  アンサンブル予測: mean={test_pred.mean():.2f}, "
              f"std={test_pred.std():.2f}, "
              f"min={test_pred.min():.2f}, max={test_pred.max():.2f}")
    else:
        # アンサンブルが無い場合、個別ベストで予測
        best_r, best_idx = indiv[0]
        cfg = models[best_idx]
        print(f"\n--- テスト予測（個別ベスト: {cfg['name']}, RMSE={best_r:.2f}）---")
        test_pred = cfg["func"](X_raw, X_test_raw, y, groups, cfg)
        test_pred = np.clip(test_pred, 0, 300)
        print(f"  予測: mean={test_pred.mean():.2f}, std={test_pred.std():.2f}")

    # 提出ファイル生成
    submit = pd.read_csv(DATA_DIR / "sample_submit.csv", header=None)
    submit[1] = test_pred
    submit_path = Path(__file__).resolve().parents[2] / "outputs" / "submission_v6_kmm.csv"
    submit.to_csv(submit_path, index=False, header=False)
    print(f"\n提出ファイル保存: {submit_path}")

    # 結果CSV保存
    rows = []
    for r, i in indiv:
        rows.append({"model": models[i]["name"], "rmse": r})
    if results:
        for name, r, combo, w in results:
            rows.append({"model": name, "rmse": r})
    pd.DataFrame(rows).to_csv(OUT_DIR / "issue101_cycle9_kmm_results.csv", index=False)
    print(f"結果保存: {OUT_DIR / 'issue101_cycle9_kmm_results.csv'}")
    print(f"総時間: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
