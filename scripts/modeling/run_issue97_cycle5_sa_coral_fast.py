"""Issue #97: サイクル5 - SA + CORAL 軽量版（8分以内完了を目標）

SA: pp=[raw, EPO(1)] x nc=[3, 10, 20] x reg=[Ridge, PLS, Huber] x tf=[raw, sqrt] = 36
CORAL: pp=[raw, SNV] x pca=[10, 20, 50, 100] x reg=[Ridge, PLS] x tf=[raw, sqrt] = 32
合計: 68モデル
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


def coral_transform(X_source, X_target, reg=1e-3):
    d = X_source.shape[1]
    Cs = np.cov(X_source, rowvar=False) + reg * np.eye(d)
    Ct = np.cov(X_target, rowvar=False) + reg * np.eye(d)
    Us, Ss, _ = np.linalg.svd(Cs)
    Cs_half_inv = Us @ np.diag(1.0 / np.sqrt(Ss + 1e-10)) @ Us.T
    Ut, St, _ = np.linalg.svd(Ct)
    Ct_half = Ut @ np.diag(np.sqrt(St + 1e-10)) @ Ut.T
    A = Cs_half_inv @ Ct_half
    return X_source @ A, X_target.copy()


def preprocess(X_tr, X_te, groups_tr, pp_name):
    if pp_name == "SNV":
        return apply_snv(X_tr), apply_snv(X_te)
    elif pp_name == "EPO(1)":
        P = compute_epo_projection(X_tr, groups_tr, n_components=1)
        return apply_epo(X_tr, P), apply_epo(X_te, P)
    return X_tr.copy(), X_te.copy()


def predict_sa(X_tr_raw, X_te_raw, y_train, groups_train, cfg):
    pp_name = cfg["pp"]
    sa_nc = cfg["sa_nc"]
    reg_type = cfg["reg"]
    tf = cfg["tf"]
    X_tr, X_te = preprocess(X_tr_raw, X_te_raw, groups_train, pp_name)
    Z_tr, Z_te = subspace_align(X_tr, X_te, n_components=sa_nc)
    y_fit = np.sqrt(y_train) if tf == "sqrt" else y_train.copy()
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


def predict_coral(X_tr_raw, X_te_raw, y_train, groups_train, cfg):
    pp_name = cfg["pp"]
    pca_dim = cfg["pca_dim"]
    reg_type = cfg["reg"]
    tf = cfg["tf"]
    X_tr, X_te = preprocess(X_tr_raw, X_te_raw, groups_train, pp_name)
    pca = PCA(n_components=pca_dim)
    X_tr_pca = pca.fit_transform(X_tr)
    X_te_pca = pca.transform(X_te)
    X_tr_coral, X_te_coral = coral_transform(X_tr_pca, X_te_pca)
    y_fit = np.sqrt(y_train) if tf == "sqrt" else y_train.copy()
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
    print("Issue #97: サイクル5 - SA + CORAL 軽量版")
    print(f"  ベースライン: RMSE = 13.70")
    print("=" * 70)

    # SA構成（絞った版）
    sa_configs = []
    for pp_name in ["raw", "EPO(1)"]:
        for sa_nc in [3, 10, 20]:
            for reg_type in ["Ridge", "PLS", "Huber"]:
                for tf in ["raw", "sqrt"]:
                    cfg = {
                        "pp": pp_name, "sa_nc": sa_nc, "reg": reg_type,
                        "tf": tf, "func": predict_sa,
                    }
                    if reg_type == "PLS":
                        cfg["pls_nc"] = min(sa_nc, 4)
                    cfg["name"] = f"SA:{pp_name}+nc{sa_nc}+{reg_type}+{tf}"
                    sa_configs.append(cfg)

    # CORAL構成
    coral_configs = []
    for pp_name in ["raw", "SNV"]:
        for pca_dim in [10, 20, 50, 100]:
            for reg_type in ["Ridge", "PLS"]:
                for tf in ["raw", "sqrt"]:
                    cfg = {
                        "pp": pp_name, "pca_dim": pca_dim, "reg": reg_type,
                        "tf": tf, "func": predict_coral,
                    }
                    if reg_type == "PLS":
                        cfg["pls_nc"] = min(4, pca_dim - 1)
                    cfg["name"] = f"CORAL:{pp_name}+pca{pca_dim}+{reg_type}+{tf}"
                    coral_configs.append(cfg)

    all_configs = sa_configs + coral_configs
    n_total = len(all_configs)
    print(f"\n総モデル数: {n_total} (SA: {len(sa_configs)}, CORAL: {len(coral_configs)})")

    all_preds = [[] for _ in range(n_total)]
    all_rmses = [None] * n_total
    best_rmse_so_far = np.inf
    completed = 0

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

    # --- SA評価 ---
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
        pp_data = get_pp_cache(pp_name)
        sa_fold_data = []
        for f_idx, (X_tr_pp, X_te_pp) in enumerate(pp_data):
            Z_tr, Z_te = subspace_align(X_tr_pp, X_te_pp, n_components=sa_nc)
            sa_fold_data.append((Z_tr, Z_te))
        cache_time = time.time() - t1

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

            mean_r = np.mean(fold_rmses)
            all_rmses[m_idx] = mean_r
            completed += 1
            nb = [r for sp, r in zip(sp_names, fold_rmses) if sp != "ベイスギ"]
            avg_nb = np.mean(nb) if nb else mean_r
            if mean_r < best_rmse_so_far:
                best_rmse_so_far = mean_r
                print(f"  [{completed:>3}/{n_total}] BEST {cfg['name']}: {mean_r:.2f} (除ベイスギ:{avg_nb:.2f})", flush=True)

        elapsed = time.time() - t0
        print(f"  SA [{grp_idx+1}/{len(sa_groups)}] pp={pp_name}, nc={sa_nc}: {len(cfgs_in_group)}モデル (cache:{cache_time:.0f}s, best:{best_rmse_so_far:.2f}, elapsed:{elapsed:.0f}s)", flush=True)

    # --- CORAL評価 ---
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

            mean_r = np.mean(fold_rmses)
            all_rmses[m_idx] = mean_r
            completed += 1
            nb = [r for sp, r in zip(sp_names, fold_rmses) if sp != "ベイスギ"]
            avg_nb = np.mean(nb) if nb else mean_r
            if mean_r < best_rmse_so_far:
                best_rmse_so_far = mean_r
                print(f"  [{completed:>3}/{n_total}] BEST {cfg['name']}: {mean_r:.2f} (除ベイスギ:{avg_nb:.2f})", flush=True)

        elapsed = time.time() - t0
        print(f"  CORAL [{grp_idx+1}/{len(coral_groups)}] pp={pp_name}, pca={pca_dim}: {len(cfgs_in_group)}モデル (cache:{cache_time:.0f}s, best:{best_rmse_so_far:.2f}, elapsed:{elapsed:.0f}s)", flush=True)

    # ランキング
    ranking = sorted(range(n_total), key=lambda i: all_rmses[i])

    print(f"\n--- 個別モデル Top 20 ---\n")
    for rank, i in enumerate(ranking[:20]):
        cfg = all_configs[i]
        fold_rmses = [rmse(y[te], all_preds[i][f]) for f, (_, te) in enumerate(folds)]
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

    # アンサンブル（軽量版: top8からk=3,5のみ）
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

    def opt_w(indices, n_restarts=10):
        n = len(indices)
        def obj(w):
            w_n = np.abs(w) / np.sum(np.abs(w))
            r, _ = eval_ens(indices, w_n)
            return r
        best_r, best_w = np.inf, np.ones(n) / n
        for s in range(n_restarts):
            w0 = np.random.RandomState(s).dirichlet(np.ones(n))
            res = minimize(obj, w0, method="Nelder-Mead", options={"maxiter": 1000})
            if res.fun < best_r:
                best_r = res.fun
                best_w = np.abs(res.x) / np.sum(np.abs(res.x))
        return best_w, best_r

    top8 = [ranking[i] for i in range(min(8, n_total))]
    results_ens = []
    for k in [3, 5]:
        best_r, best_combo = np.inf, None
        for combo in combinations(top8, k):
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

    # ベスト結果
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
    print(f"ベースライン: 13.70")
    print(f"{'='*70}")

    # fold詳細
    print(f"\nベスト個別モデル fold詳細:")
    for f_idx, (_, test_idx) in enumerate(folds):
        r = rmse(y[test_idx], all_preds[best_indiv_idx][f_idx])
        tag = " ※参考" if sp_names[f_idx] == "ベイスギ" else ""
        print(f"  {sp_names[f_idx]}: {r:.2f}{tag}")

    # 結果保存
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
