"""Issue #97: サイクル5 - Subspace Alignment + CORAL ドメイン適応

1. Subspace Alignment + PLS/Ridge/Huber
   - n_components: [3, 5, 10, 15, 20, 30]
   - 前処理: [raw, SNV, EPO(1)]
   - 回帰: [Ridge(alpha=1.0), PLS(nc=min(nc, sa_nc)), Huber]
   - 目的変数: [raw, sqrt]

2. CORAL + PLS/Ridge
   - PCA次元削減後にCORAL適用
   - PCA次元: [10, 20, 50, 100]
   - 前処理: [raw, SNV]
   - 回帰: [Ridge, PLS]

3. LOSO-CV + テスト予測
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
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut

from src.eda.data_loader import load_train, load_test, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue44_subspace_alignment import subspace_align

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ============================================================
# CORAL (CORrelation ALignment)
# ============================================================

def coral_transform(X_source, X_target, reg=1e-3):
    """CORALドメイン適応: ソースの2次統計量をターゲットに整合。

    Parameters
    ----------
    X_source : (n_s, d) ソースドメイン特徴量
    X_target : (n_t, d) ターゲットドメイン特徴量
    reg : float 正則化パラメータ

    Returns
    -------
    X_source_aligned : (n_s, d)
    X_target_copy : (n_t, d)
    """
    d = X_source.shape[1]
    Cs = np.cov(X_source, rowvar=False) + reg * np.eye(d)
    Ct = np.cov(X_target, rowvar=False) + reg * np.eye(d)

    # Cs^{-1/2}
    Us, Ss, _ = np.linalg.svd(Cs)
    Cs_half_inv = Us @ np.diag(1.0 / np.sqrt(Ss + 1e-10)) @ Us.T

    # Ct^{1/2}
    Ut, St, _ = np.linalg.svd(Ct)
    Ct_half = Ut @ np.diag(np.sqrt(St + 1e-10)) @ Ut.T

    # Transform: ソースをターゲット統計量に合わせる
    A = Cs_half_inv @ Ct_half
    return X_source @ A, X_target.copy()


# ============================================================
# 前処理
# ============================================================

def preprocess(X_tr, X_te, groups_tr, pp_name):
    """前処理を適用する。"""
    if pp_name == "SNV":
        return apply_snv(X_tr), apply_snv(X_te)
    elif pp_name == "EPO(1)":
        P = compute_epo_projection(X_tr, groups_tr, n_components=1)
        return apply_epo(X_tr, P), apply_epo(X_te, P)
    return X_tr.copy(), X_te.copy()


# ============================================================
# Subspace Alignment パイプライン
# ============================================================

def predict_sa(X_tr_raw, X_te_raw, y_train, groups_train, cfg):
    """Subspace Alignment + 回帰モデル"""
    pp_name = cfg["pp"]
    sa_nc = cfg["sa_nc"]
    reg_type = cfg["reg"]
    tf = cfg["tf"]

    # 前処理
    X_tr, X_te = preprocess(X_tr_raw, X_te_raw, groups_train, pp_name)

    # SA適用
    Z_tr, Z_te = subspace_align(X_tr, X_te, n_components=sa_nc)

    # 目的変数変換
    y_fit = np.sqrt(y_train) if tf == "sqrt" else y_train.copy()

    # 回帰
    if reg_type == "PLS":
        nc = min(cfg.get("pls_nc", sa_nc), sa_nc - 1)
        nc = max(1, nc)
        pls = PLSRegression(n_components=nc)
        pls.fit(Z_tr, y_fit)
        pred = pls.predict(Z_te).ravel()
    elif reg_type == "Ridge":
        sc = StandardScaler()
        Z_tr_s = sc.fit_transform(Z_tr)
        Z_te_s = sc.transform(Z_te)
        ridge = Ridge(alpha=cfg.get("alpha", 1.0))
        ridge.fit(Z_tr_s, y_fit)
        pred = ridge.predict(Z_te_s)
    elif reg_type == "Huber":
        sc = StandardScaler()
        Z_tr_s = sc.fit_transform(Z_tr)
        Z_te_s = sc.transform(Z_te)
        huber = HuberRegressor(epsilon=1.35, max_iter=200, alpha=0.01)
        huber.fit(Z_tr_s, y_fit)
        pred = huber.predict(Z_te_s)
    else:
        raise ValueError(f"Unknown reg: {reg_type}")

    if tf == "sqrt":
        pred = np.clip(pred, 0, None) ** 2
    return pred


# ============================================================
# CORAL パイプライン
# ============================================================

def predict_coral(X_tr_raw, X_te_raw, y_train, groups_train, cfg):
    """PCA次元削減 + CORAL + 回帰モデル"""
    pp_name = cfg["pp"]
    pca_dim = cfg["pca_dim"]
    reg_type = cfg["reg"]
    tf = cfg["tf"]

    # 前処理
    X_tr, X_te = preprocess(X_tr_raw, X_te_raw, groups_train, pp_name)

    # PCA次元削減（trainでfit、testはtransformのみ）
    pca = PCA(n_components=pca_dim)
    X_tr_pca = pca.fit_transform(X_tr)
    X_te_pca = pca.transform(X_te)

    # CORAL適用
    X_tr_coral, X_te_coral = coral_transform(X_tr_pca, X_te_pca)

    # 目的変数変換
    y_fit = np.sqrt(y_train) if tf == "sqrt" else y_train.copy()

    # 回帰
    if reg_type == "PLS":
        nc = min(cfg.get("pls_nc", 4), pca_dim - 1)
        nc = max(1, nc)
        pls = PLSRegression(n_components=nc)
        pls.fit(X_tr_coral, y_fit)
        pred = pls.predict(X_te_coral).ravel()
    elif reg_type == "Ridge":
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr_coral)
        X_te_s = sc.transform(X_te_coral)
        ridge = Ridge(alpha=cfg.get("alpha", 1.0))
        ridge.fit(X_tr_s, y_fit)
        pred = ridge.predict(X_te_s)
    else:
        raise ValueError(f"Unknown reg: {reg_type}")

    if tf == "sqrt":
        pred = np.clip(pred, 0, None) ** 2
    return pred


# ============================================================
# メイン
# ============================================================

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
    print("Issue #97: サイクル5 - Subspace Alignment + CORAL")
    print(f"  ベースライン: RMSE = 17.03")
    print("=" * 70)

    # ============================================================
    # SA構成の生成（計算時間を考慮して絞る）
    # ============================================================
    sa_configs = []
    for pp_name in ["raw", "SNV", "EPO(1)"]:
        for sa_nc in [3, 5, 10, 15, 20, 30]:
            for reg_type in ["Ridge", "PLS", "Huber"]:
                for tf in ["raw", "sqrt"]:
                    cfg = {
                        "pp": pp_name,
                        "sa_nc": sa_nc,
                        "reg": reg_type,
                        "tf": tf,
                        "func": predict_sa,
                    }
                    if reg_type == "PLS":
                        cfg["pls_nc"] = min(sa_nc, 4)
                    label = f"SA:{pp_name}+nc{sa_nc}+{reg_type}+{tf}"
                    cfg["name"] = label
                    sa_configs.append(cfg)

    # ============================================================
    # CORAL構成の生成
    # ============================================================
    coral_configs = []
    for pp_name in ["raw", "SNV"]:
        for pca_dim in [10, 20, 50, 100]:
            for reg_type in ["Ridge", "PLS"]:
                for tf in ["raw", "sqrt"]:
                    cfg = {
                        "pp": pp_name,
                        "pca_dim": pca_dim,
                        "reg": reg_type,
                        "tf": tf,
                        "func": predict_coral,
                    }
                    if reg_type == "PLS":
                        cfg["pls_nc"] = min(4, pca_dim - 1)
                    label = f"CORAL:{pp_name}+pca{pca_dim}+{reg_type}+{tf}"
                    cfg["name"] = label
                    coral_configs.append(cfg)

    all_configs = sa_configs + coral_configs
    n_total = len(all_configs)
    print(f"\n総モデル数: {n_total} (SA: {len(sa_configs)}, CORAL: {len(coral_configs)})")

    # ============================================================
    # インクリメンタル評価: SA/CORAL特徴量を1グループずつ計算し回帰評価
    # メモリを節約しつつ前処理のキャッシュを活用
    # ============================================================
    # config_idxを管理するためのマッピング
    all_preds = [[] for _ in range(n_total)]
    all_rmses = [None] * n_total
    best_rmse_so_far = np.inf
    completed = 0

    # 前処理キャッシュ: pp_name -> fold別 (X_tr_pp, X_te_pp)
    pp_cache = {}

    def get_pp_cache(pp_name):
        if pp_name not in pp_cache:
            fold_data = []
            for f_idx, (train_idx, test_idx) in enumerate(folds):
                X_tr_pp, X_te_pp = preprocess(X_raw[train_idx], X_raw[test_idx],
                                               groups[train_idx], pp_name)
                fold_data.append((X_tr_pp, X_te_pp))
            pp_cache[pp_name] = fold_data
        return pp_cache[pp_name]

    def run_regression(Z_tr, Z_te, y_fit, reg_type, cfg):
        """キャッシュ済み特徴量に対して回帰を実行"""
        if reg_type == "PLS":
            dim = Z_tr.shape[1]
            nc = min(cfg.get("pls_nc", 4), dim - 1)
            nc = max(1, nc)
            pls = PLSRegression(n_components=nc)
            pls.fit(Z_tr, y_fit)
            return pls.predict(Z_te).ravel()
        elif reg_type == "Ridge":
            sc_obj = StandardScaler()
            Z_tr_s = sc_obj.fit_transform(Z_tr)
            Z_te_s = sc_obj.transform(Z_te)
            ridge = Ridge(alpha=cfg.get("alpha", 1.0))
            ridge.fit(Z_tr_s, y_fit)
            return ridge.predict(Z_te_s)
        elif reg_type == "Huber":
            sc_obj = StandardScaler()
            Z_tr_s = sc_obj.fit_transform(Z_tr)
            Z_te_s = sc_obj.transform(Z_te)
            huber = HuberRegressor(epsilon=1.35, max_iter=200, alpha=0.01)
            huber.fit(Z_tr_s, y_fit)
            return huber.predict(Z_te_s)
        else:
            raise ValueError(f"Unknown reg: {reg_type}")

    # --- SA評価: (pp, sa_nc) グループごとに処理 ---
    print(f"\n--- SA評価 ---\n", flush=True)

    sa_groups = {}
    for m_idx, cfg in enumerate(sa_configs):
        key = (cfg["pp"], cfg["sa_nc"])
        if key not in sa_groups:
            sa_groups[key] = []
        sa_groups[key].append((m_idx, cfg))

    for grp_idx, (key, cfgs_in_group) in enumerate(sorted(sa_groups.items())):
        pp_name, sa_nc = key
        t1 = time.time()

        # SA特徴量をfold別に計算
        pp_data = get_pp_cache(pp_name)
        sa_fold_data = []
        for f_idx, (X_tr_pp, X_te_pp) in enumerate(pp_data):
            Z_tr, Z_te = subspace_align(X_tr_pp, X_te_pp, n_components=sa_nc)
            sa_fold_data.append((Z_tr, Z_te))

        cache_time = time.time() - t1

        # このグループ内の全回帰モデルを評価
        for m_idx, cfg in cfgs_in_group:
            fold_rmses = []
            for f_idx, (train_idx, test_idx) in enumerate(folds):
                try:
                    y_train = y[train_idx]
                    tf = cfg["tf"]
                    y_fit = np.sqrt(y_train) if tf == "sqrt" else y_train.copy()
                    Z_tr, Z_te = sa_fold_data[f_idx]
                    pred = run_regression(Z_tr, Z_te, y_fit, cfg["reg"], cfg)
                    if tf == "sqrt":
                        pred = np.clip(pred, 0, None) ** 2
                    all_preds[m_idx].append(pred)
                    fold_rmses.append(rmse(y[test_idx], pred))
                except Exception as e:
                    fallback = np.full(len(test_idx), y[train_idx].mean())
                    all_preds[m_idx].append(fallback)
                    fold_rmses.append(999.0)
                    print(f"  ERROR {cfg['name']} fold {f_idx}: {e}")

            mean_r = np.mean(fold_rmses)
            all_rmses[m_idx] = mean_r
            completed += 1
            nb = [r for sp, r in zip(sp_names, fold_rmses) if sp != "ベイスギ"]
            avg_nb = np.mean(nb) if nb else mean_r

            if mean_r < best_rmse_so_far:
                best_rmse_so_far = mean_r
                print(f"  [{completed:>3}/{n_total}] ★BEST★ {cfg['name']}: {mean_r:.2f} (除ベイスギ:{avg_nb:.2f})", flush=True)

        print(f"  SA [{grp_idx+1}/{len(sa_groups)}] pp={pp_name}, nc={sa_nc}: {len(cfgs_in_group)}モデル完了 (cache:{cache_time:.0f}s, best_so_far:{best_rmse_so_far:.2f})", flush=True)

    # --- CORAL評価: (pp, pca_dim) グループごとに処理 ---
    print(f"\n--- CORAL評価 ---\n", flush=True)

    coral_groups = {}
    for m_idx_offset, cfg in enumerate(coral_configs):
        m_idx = len(sa_configs) + m_idx_offset
        key = (cfg["pp"], cfg["pca_dim"])
        if key not in coral_groups:
            coral_groups[key] = []
        coral_groups[key].append((m_idx, cfg))

    for grp_idx, (key, cfgs_in_group) in enumerate(sorted(coral_groups.items())):
        pp_name, pca_dim = key
        t1 = time.time()

        # CORAL特徴量をfold別に計算
        pp_data = get_pp_cache(pp_name)
        coral_fold_data = []
        for f_idx, (X_tr_pp, X_te_pp) in enumerate(pp_data):
            pca = PCA(n_components=pca_dim)
            X_tr_pca = pca.fit_transform(X_tr_pp)
            X_te_pca = pca.transform(X_te_pp)
            X_tr_c, X_te_c = coral_transform(X_tr_pca, X_te_pca)
            coral_fold_data.append((X_tr_c, X_te_c))

        cache_time = time.time() - t1

        for m_idx, cfg in cfgs_in_group:
            fold_rmses = []
            for f_idx, (train_idx, test_idx) in enumerate(folds):
                try:
                    y_train = y[train_idx]
                    tf = cfg["tf"]
                    y_fit = np.sqrt(y_train) if tf == "sqrt" else y_train.copy()
                    X_tr_c, X_te_c = coral_fold_data[f_idx]
                    pred = run_regression(X_tr_c, X_te_c, y_fit, cfg["reg"], cfg)
                    if tf == "sqrt":
                        pred = np.clip(pred, 0, None) ** 2
                    all_preds[m_idx].append(pred)
                    fold_rmses.append(rmse(y[test_idx], pred))
                except Exception as e:
                    fallback = np.full(len(test_idx), y[train_idx].mean())
                    all_preds[m_idx].append(fallback)
                    fold_rmses.append(999.0)
                    print(f"  ERROR {cfg['name']} fold {f_idx}: {e}")

            mean_r = np.mean(fold_rmses)
            all_rmses[m_idx] = mean_r
            completed += 1
            nb = [r for sp, r in zip(sp_names, fold_rmses) if sp != "ベイスギ"]
            avg_nb = np.mean(nb) if nb else mean_r

            if mean_r < best_rmse_so_far:
                best_rmse_so_far = mean_r
                print(f"  [{completed:>3}/{n_total}] ★BEST★ {cfg['name']}: {mean_r:.2f} (除ベイスギ:{avg_nb:.2f})", flush=True)

        print(f"  CORAL [{grp_idx+1}/{len(coral_groups)}] pp={pp_name}, pca={pca_dim}: {len(cfgs_in_group)}モデル完了 (cache:{cache_time:.0f}s, best_so_far:{best_rmse_so_far:.2f})", flush=True)

    # ============================================================
    # 個別モデルランキング
    # ============================================================
    ranking = sorted(range(n_total), key=lambda i: all_rmses[i])

    print(f"\n--- 個別モデル Top 20 ---\n")
    for rank, i in enumerate(ranking[:20]):
        cfg = all_configs[i]
        fold_rmses = []
        for f_idx, (_, test_idx) in enumerate(folds):
            fold_rmses.append(rmse(y[test_idx], all_preds[i][f_idx]))
        nb = [r for sp, r in zip(sp_names, fold_rmses) if sp != "ベイスギ"]
        print(f"  {rank+1:>2}. {all_rmses[i]:.2f} (除ベイスギ:{np.mean(nb):.2f}) | {cfg['name']}")

    # SA/CORAL別ベスト
    sa_ranking = [(i, all_rmses[i]) for i in range(len(sa_configs))]
    sa_ranking.sort(key=lambda x: x[1])
    coral_ranking = [(i, all_rmses[i]) for i in range(len(sa_configs), n_total)]
    coral_ranking.sort(key=lambda x: x[1])

    print(f"\n--- SA Best 5 ---")
    for i, r in sa_ranking[:5]:
        print(f"  {r:.2f} | {all_configs[i]['name']}")

    print(f"\n--- CORAL Best 5 ---")
    for i, r in coral_ranking[:5]:
        print(f"  {r:.2f} | {all_configs[i]['name']}")

    # ============================================================
    # アンサンブル最適化（Top12から）
    # ============================================================
    print(f"\n--- アンサンブル最適化 ---\n", flush=True)

    def eval_ens(indices, weights=None):
        fold_rmses = []
        for f_idx, (_, test_idx) in enumerate(folds):
            preds = [all_preds[m][f_idx] for m in indices]
            if weights is not None:
                w = np.array(weights)
                w = w / w.sum()
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
            res = minimize(obj, w0, method="Nelder-Mead", options={"maxiter": 2000})
            if res.fun < best_r:
                best_r = res.fun
                best_w = np.abs(res.x) / np.sum(np.abs(res.x))
        return best_w, best_r

    top12 = [ranking[i] for i in range(min(12, n_total))]
    results_ens = []

    for k in [3, 5, 7]:
        best_r, best_combo = np.inf, None
        for combo in combinations(top12, k):
            r, _ = eval_ens(list(combo))
            if r < best_r:
                best_r = r
                best_combo = list(combo)
        if best_combo:
            w, r_w = opt_w(best_combo)
            results_ens.append((f"Ensemble{k}-Weighted", r_w, best_combo, w))
            print(f"  Ensemble{k}-Weighted: {r_w:.4f}", flush=True)
            for idx, wi in zip(best_combo, w):
                if wi > 0.01:
                    print(f"    {wi:.3f}: {all_configs[idx]['name']}")

    # ============================================================
    # ベスト結果まとめ
    # ============================================================
    best_indiv_idx = ranking[0]
    best_indiv_rmse = all_rmses[best_indiv_idx]
    best_indiv_cfg = all_configs[best_indiv_idx]

    if results_ens:
        results_ens.sort(key=lambda x: x[1])
        best_ens = results_ens[0]
    else:
        best_ens = None

    print(f"\n{'='*70}")
    print(f"ベスト個別モデル: {best_indiv_rmse:.4f} ({best_indiv_cfg['name']})")
    if best_ens:
        print(f"ベストアンサンブル: {best_ens[1]:.4f} ({best_ens[0]})")
    print(f"現ベスト: 17.03")
    print(f"{'='*70}")

    # fold詳細（ベスト個別）
    print(f"\nベスト個別モデル fold詳細:")
    for f_idx, (_, test_idx) in enumerate(folds):
        r = rmse(y[test_idx], all_preds[best_indiv_idx][f_idx])
        tag = " ※参考" if sp_names[f_idx] == "ベイスギ" else ""
        print(f"  {sp_names[f_idx]}: {r:.2f}{tag}")

    # ============================================================
    # テスト予測（SAベスト・CORALベスト）
    # ============================================================
    print(f"\n--- テスト予測 ---\n", flush=True)

    # SAベスト
    sa_best_idx = sa_ranking[0][0] if sa_ranking else None
    if sa_best_idx is not None:
        sa_cfg = all_configs[sa_best_idx]
        try:
            pred_sa = predict_sa(X_raw, X_test_raw, y, groups, sa_cfg)
            sub_sa = pd.read_csv(DATA_DIR / "sample_submit.csv", header=None)
            sub_sa.iloc[:, 1] = np.clip(pred_sa, 0, 300)
            sa_path = OUT_DIR.parent / "submission_v6_sa.csv"
            sub_sa.to_csv(sa_path, index=False, header=False)
            print(f"  SA提出ファイル: {sa_path}")
            print(f"  SAベスト設定: {sa_cfg['name']} (CV RMSE={sa_ranking[0][1]:.2f})")
        except Exception as e:
            print(f"  SA提出エラー: {e}")

    # CORALベスト
    coral_best_idx = coral_ranking[0][0] if coral_ranking else None
    if coral_best_idx is not None:
        coral_cfg = all_configs[coral_best_idx]
        try:
            pred_coral = predict_coral(X_raw, X_test_raw, y, groups, coral_cfg)
            sub_coral = pd.read_csv(DATA_DIR / "sample_submit.csv", header=None)
            sub_coral.iloc[:, 1] = np.clip(pred_coral, 0, 300)
            coral_path = OUT_DIR.parent / "submission_v6_coral.csv"
            sub_coral.to_csv(coral_path, index=False, header=False)
            print(f"  CORAL提出ファイル: {coral_path}")
            print(f"  CORALベスト設定: {coral_cfg['name']} (CV RMSE={coral_ranking[0][1]:.2f})")
        except Exception as e:
            print(f"  CORAL提出エラー: {e}")

    # ============================================================
    # 結果保存
    # ============================================================
    rows = []
    for i in ranking:
        cfg = all_configs[i]
        fold_rmses = [rmse(y[te], all_preds[i][f]) for f, (_, te) in enumerate(folds)]
        nb = [r for sp, r in zip(sp_names, fold_rmses) if sp != "ベイスギ"]
        rows.append({
            "rank": len(rows) + 1,
            "name": cfg["name"],
            "mean_rmse": all_rmses[i],
            "mean_rmse_excl_beisugi": np.mean(nb) if nb else all_rmses[i],
            **{f"fold_{sp}": r for sp, r in zip(sp_names, fold_rmses)},
        })

    result_path = OUT_DIR / "issue97_cycle5_sa_coral_results.csv"
    pd.DataFrame(rows).to_csv(result_path, index=False)
    print(f"\n結果CSV: {result_path}")
    print(f"総実行時間: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
