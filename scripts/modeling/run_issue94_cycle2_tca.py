"""Issue #94: Cycle 2 - TCAドメイン適応をテスト予測パイプラインに統合

仮説: TCA変換後の特徴空間でPLS/Ridge/Huber回帰を行えば、
train-test間の分布差が縮小し、LBスコアが改善する。

高速化戦略:
- 前処理+PCA結果をfold×前処理ごとにキャッシュ
- サブサンプリングでeighの行列サイズを~70に削減
- eigh(69×69)≈0.1秒、PCA(1300×1555)≈20秒をキャッシュで1回に
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import warnings
import time

warnings.filterwarnings("ignore")

from scipy.linalg import eigh
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from sklearn.decomposition import TruncatedSVD
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.model_selection import LeaveOneGroupOut

from src.eda.data_loader import load_train, load_test, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBSAMPLE_PER_GROUP = 3
MAX_SUBSAMPLE_TARGET = 30
PCA_DIM = 30


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def subsample_by_group(n_samples, groups, n_per_group=SUBSAMPLE_PER_GROUP, seed=42):
    rng = np.random.RandomState(seed)
    indices = []
    for g in np.unique(groups):
        g_idx = np.where(groups == g)[0]
        n = min(n_per_group, len(g_idx))
        selected = rng.choice(g_idx, size=n, replace=False)
        indices.extend(selected.tolist())
    return np.array(sorted(indices))


def pca_reduce(X_tr, X_te, n_components=PCA_DIM):
    """PCA次元削減。trainでfitしtestはtransformのみ。"""
    nc = min(n_components, X_tr.shape[1], X_tr.shape[0])
    svd = TruncatedSVD(n_components=nc, random_state=42)
    X_tr_pca = svd.fit_transform(X_tr)
    X_te_pca = svd.transform(X_te)
    return X_tr_pca, X_te_pca


def tca_on_reduced(X_s_pca, X_t_pca, n_components=10, kernel="rbf",
                   gamma=None, mu=1.0, source_groups=None):
    """PCA済みデータに対するサブサンプリングTCA。"""
    n_s = len(X_s_pca)
    n_t = len(X_t_pca)

    # サブサンプリング
    if source_groups is not None and n_s > 50:
        sub_s_idx = subsample_by_group(n_s, source_groups)
    else:
        sub_s_idx = np.arange(n_s)

    if n_t > MAX_SUBSAMPLE_TARGET:
        rng = np.random.RandomState(42)
        sub_t_idx = np.sort(rng.choice(n_t, size=MAX_SUBSAMPLE_TARGET, replace=False))
    else:
        sub_t_idx = np.arange(n_t)

    X_sub_s = X_s_pca[sub_s_idx]
    X_sub_t = X_t_pca[sub_t_idx]
    n_sub_s = len(X_sub_s)
    n_sub_t = len(X_sub_t)
    n_sub = n_sub_s + n_sub_t

    X_sub_all = np.vstack([X_sub_s, X_sub_t])

    # カーネル行列
    if kernel == "rbf":
        if gamma is None:
            gamma = 1.0 / X_sub_all.shape[1]
        K_sub = rbf_kernel(X_sub_all, gamma=gamma)
    elif kernel == "linear":
        K_sub = linear_kernel(X_sub_all)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # MMDペナルティ行列
    L = np.zeros((n_sub, n_sub))
    L[:n_sub_s, :n_sub_s] = 1.0 / (n_sub_s * n_sub_s)
    L[n_sub_s:, n_sub_s:] = 1.0 / (n_sub_t * n_sub_t)
    L[:n_sub_s, n_sub_s:] = -1.0 / (n_sub_s * n_sub_t)
    L[n_sub_s:, :n_sub_s] = -1.0 / (n_sub_s * n_sub_t)

    H = np.eye(n_sub) - np.ones((n_sub, n_sub)) / n_sub

    A = K_sub @ L @ K_sub + mu * np.eye(n_sub)
    B = K_sub @ H @ K_sub
    B = (B + B.T) / 2
    min_eig = np.real(np.linalg.eigvalsh(B).min())
    reg = max(1e-8, -min_eig + 1e-6) if min_eig < 1e-6 else 1e-8
    B += reg * np.eye(n_sub)

    nc = min(n_components, n_sub - 1)
    eigenvalues, eigenvectors = eigh(A, B)
    W = eigenvectors[:, :nc]

    # 全データのカーネル行列（全データ vs サブサンプル）
    X_all_pca = np.vstack([X_s_pca, X_t_pca])
    if kernel == "rbf":
        K_full = rbf_kernel(X_all_pca, X_sub_all, gamma=gamma)
    else:
        K_full = linear_kernel(X_all_pca, X_sub_all)

    Z = K_full @ W
    return Z[:n_s], Z[n_s:]


def preprocess(X_tr, X_te, groups_tr, method):
    if method == "raw":
        return X_tr.copy(), X_te.copy()
    elif method == "snv":
        return apply_snv(X_tr), apply_snv(X_te)
    elif method == "epo1":
        P = compute_epo_projection(X_tr, groups_tr, n_components=1)
        return apply_epo(X_tr, P), apply_epo(X_te, P)
    else:
        raise ValueError(f"Unknown preprocessing: {method}")


def make_regressor(name):
    if name == "pls3":
        return PLSRegression(n_components=3)
    elif name == "ridge":
        return Ridge(alpha=1.0)
    elif name == "huber":
        return HuberRegressor(max_iter=300)
    else:
        raise ValueError(f"Unknown regressor: {name}")


def fit_predict(model, Z_tr, y_tr, Z_te):
    model.fit(Z_tr, y_tr)
    pred = model.predict(Z_te)
    if hasattr(pred, "ravel"):
        pred = pred.ravel()
    return pred


def main():
    print("=" * 70)
    print("Issue #94 Cycle 2: TCA Domain Adaptation Pipeline")
    print("(Cached PCA + Subsampled TCA)")
    print("=" * 70)

    t0 = time.time()

    # データ読み込み
    df_train = load_train(DATA_DIR)
    df_test = load_test(DATA_DIR)
    spec_cols = get_spectral_columns(df_train)

    X_train = df_train[spec_cols].values.astype(float)
    y_train = df_train["含水率"].values.astype(float)
    groups = df_train["樹種"].values

    X_test = df_test[spec_cols].values.astype(float)
    test_ids = df_test["sample number"].values

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Species (train): {np.unique(groups)}")
    print()

    # ========================================
    # Phase 0: 前処理+PCAキャッシュ構築
    # ========================================
    print("Phase 0: Building preprocessed PCA cache for each fold")
    print("-" * 50)

    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_train, y_train, groups))
    preprocessings = ["raw", "snv", "epo1"]

    # cache[fold_idx][pp_method] = (X_tr_pca, X_te_pca, groups_tr)
    cache = {}
    for fi, (tr_idx, te_idx) in enumerate(folds):
        cache[fi] = {}
        X_tr, X_te = X_train[tr_idx], X_train[te_idx]
        g_tr = groups[tr_idx]
        species = groups[te_idx][0]

        for pp in preprocessings:
            X_tr_pp, X_te_pp = preprocess(X_tr, X_te, g_tr, pp)
            X_tr_pca, X_te_pca = pca_reduce(X_tr_pp, X_te_pp, n_components=PCA_DIM)
            cache[fi][pp] = (X_tr_pca, X_te_pca, g_tr, tr_idx, te_idx)

        elapsed = time.time() - t0
        print(f"  Fold {fi+1}/13 ({species}): {elapsed:.0f}s")

    print(f"Cache built: {time.time()-t0:.0f}s")

    # ========================================
    # Phase 1: LOSO-CV グリッドサーチ（キャッシュ利用）
    # ========================================
    print("\nPhase 1: LOSO-CV Grid Search")
    print("-" * 50)

    kernels = ["linear", "rbf"]
    n_components_list = [3, 5, 10, 15, 20]
    mus = [0.01, 0.1, 1.0, 10.0]
    regressors = ["pls3", "ridge", "huber"]
    target_transforms = ["raw", "sqrt"]

    configs = []
    for kernel in kernels:
        for nc in n_components_list:
            for mu in mus:
                for pp in preprocessings:
                    for reg in regressors:
                        for tt in target_transforms:
                            configs.append((kernel, nc, mu, pp, reg, tt))

    total = len(configs)
    print(f"Total configurations: {total}")

    results = []
    for ci, (kernel, nc, mu, pp, reg, tt) in enumerate(configs):
        if (ci + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = elapsed / (ci + 1)
            eta = rate * (total - ci - 1)
            print(f"  [{ci+1}/{total}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

        all_pred = np.zeros_like(y_train, dtype=float)
        fold_rmses = []
        fold_species = []
        failed = False

        for fi in range(len(folds)):
            X_tr_pca, X_te_pca, g_tr, tr_idx, te_idx = cache[fi][pp]
            y_tr = y_train[tr_idx]
            y_te = y_train[te_idx]

            nc_actual = min(nc, len(te_idx) - 1)
            if nc_actual < 1:
                nc_actual = 1

            try:
                Z_tr, Z_te = tca_on_reduced(
                    X_tr_pca, X_te_pca, n_components=nc_actual,
                    kernel=kernel, mu=mu, source_groups=g_tr
                )
            except Exception:
                failed = True
                break

            y_tr_t = np.sqrt(y_tr) if tt == "sqrt" else y_tr.copy()

            model = make_regressor(reg)
            if reg == "pls3" and nc_actual < 3:
                model = PLSRegression(n_components=max(1, nc_actual))

            try:
                pred = fit_predict(model, Z_tr, y_tr_t, Z_te)
            except Exception:
                failed = True
                break

            if tt == "sqrt":
                pred = np.clip(pred, 0, None) ** 2

            all_pred[te_idx] = pred
            fold_rmses.append(rmse(pred, y_te))
            fold_species.append(groups[te_idx][0])

        if not failed:
            overall = rmse(all_pred, y_train)
            results.append({
                "kernel": kernel, "n_components": nc, "mu": mu,
                "preprocessing": pp, "regressor": reg, "target_transform": tt,
                "rmse": overall, "fold_rmses": fold_rmses, "fold_species": fold_species,
            })

    df_results = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)

    print(f"\nCompleted: {len(df_results)} valid results")
    print(f"Time: {time.time()-t0:.0f}s")
    print(f"\nTop 20 configurations by LOSO-CV RMSE:")
    print("-" * 90)
    print(f"{'Rank':>4} {'Kernel':>7} {'NC':>3} {'Mu':>6} {'PP':>5} {'Reg':>6} {'TT':>5} {'RMSE':>8}")
    print("-" * 90)

    for i, row in df_results.head(20).iterrows():
        print(
            f"{i+1:>4} {row['kernel']:>7} {row['n_components']:>3} "
            f"{row['mu']:>6.2f} {row['preprocessing']:>5} {row['regressor']:>6} "
            f"{row['target_transform']:>5} {row['rmse']:>8.3f}"
        )

    best = df_results.iloc[0]
    print(f"\nBest configuration: RMSE = {best['rmse']:.4f}")
    print(f"  kernel={best['kernel']}, nc={best['n_components']}, mu={best['mu']}")
    print(f"  pp={best['preprocessing']}, reg={best['regressor']}, tt={best['target_transform']}")
    print(f"\nFold-level RMSE (best config):")
    for sp, fr in zip(best["fold_species"], best["fold_rmses"]):
        marker = " (参考: 外挿領域)" if sp == "ベイスギ" else ""
        print(f"  {sp}: {fr:.3f}{marker}")

    # ========================================
    # Phase 2: テスト予測 (ベスト設定)
    # ========================================
    print("\n" + "=" * 70)
    print("Phase 2: Test Prediction (Best Config)")
    print("=" * 70)

    bk, bnc, bmu = best["kernel"], int(best["n_components"]), best["mu"]
    bpp, breg, btt = best["preprocessing"], best["regressor"], best["target_transform"]

    X_tr_pp, X_te_pp = preprocess(X_train, X_test, groups, bpp)
    X_tr_pca, X_te_pca = pca_reduce(X_tr_pp, X_te_pp, n_components=PCA_DIM)
    Z_tr, Z_te = tca_on_reduced(
        X_tr_pca, X_te_pca, n_components=bnc, kernel=bk, mu=bmu, source_groups=groups
    )
    print(f"TCA transformed: train={Z_tr.shape}, test={Z_te.shape}")

    y_fit = np.sqrt(y_train) if btt == "sqrt" else y_train.copy()
    model = make_regressor(breg)
    pred_test = fit_predict(model, Z_tr, y_fit, Z_te)
    if btt == "sqrt":
        pred_test = np.clip(pred_test, 0, None) ** 2

    print(f"Predictions: min={pred_test.min():.2f}, max={pred_test.max():.2f}, "
          f"mean={pred_test.mean():.2f}, std={pred_test.std():.2f}")

    # ========================================
    # Phase 3: Top-5 Ensemble
    # ========================================
    print("\n" + "=" * 70)
    print("Phase 3: Top-5 Ensemble")
    print("=" * 70)

    top5_preds = []
    for i, row in df_results.head(5).iterrows():
        X_tr_pp, X_te_pp = preprocess(X_train, X_test, groups, row["preprocessing"])
        X_tr_pca, X_te_pca = pca_reduce(X_tr_pp, X_te_pp, n_components=PCA_DIM)
        nc = int(row["n_components"])
        Z_tr, Z_te = tca_on_reduced(
            X_tr_pca, X_te_pca, n_components=nc, kernel=row["kernel"], mu=row["mu"],
            source_groups=groups
        )
        y_f = np.sqrt(y_train) if row["target_transform"] == "sqrt" else y_train.copy()
        m = make_regressor(row["regressor"])
        p = fit_predict(m, Z_tr, y_f, Z_te)
        if row["target_transform"] == "sqrt":
            p = np.clip(p, 0, None) ** 2
        top5_preds.append(p)
        print(f"  Config {i+1} (RMSE={row['rmse']:.3f}): pred [{p.min():.1f}, {p.max():.1f}]")

    ensemble_pred = np.mean(top5_preds, axis=0)
    print(f"\nEnsemble: min={ensemble_pred.min():.2f}, max={ensemble_pred.max():.2f}, "
          f"mean={ensemble_pred.mean():.2f}")

    # ========================================
    # Phase 4: 提出ファイル生成
    # ========================================
    print("\n" + "=" * 70)
    print("Phase 4: Submission File")
    print("=" * 70)

    submit_single = pd.DataFrame({0: test_ids.astype(int), 1: pred_test})
    path_single = OUT_DIR / "submission_v6_tca.csv"
    submit_single.to_csv(path_single, index=False, header=False)
    print(f"Single best: {path_single}")

    submit_ensemble = pd.DataFrame({0: test_ids.astype(int), 1: ensemble_pred})
    path_ensemble = OUT_DIR / "submission_v6_tca_ensemble.csv"
    submit_ensemble.to_csv(path_ensemble, index=False, header=False)
    print(f"Ensemble: {path_ensemble}")

    sample_submit = pd.read_csv(DATA_DIR / "sample_submit.csv", header=None)
    assert len(submit_single) == len(sample_submit), \
        f"Row count mismatch: {len(submit_single)} vs {len(sample_submit)}"
    print(f"Validation: row count OK ({len(submit_single)} rows)")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    summary_cols = ["kernel", "n_components", "mu", "preprocessing", "regressor", "target_transform", "rmse"]
    df_results[summary_cols].to_csv(OUT_DIR / "issue94_cycle2_tca_results.csv", index=False)
    print(f"Results saved: {OUT_DIR / 'issue94_cycle2_tca_results.csv'}")


if __name__ == "__main__":
    main()
