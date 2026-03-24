"""Issue #94: Cycle 2 - TCAドメイン適応をテスト予測パイプラインに統合

仮説: TCA変換後の特徴空間でPLS/Ridge/Huber回帰を行えば、
train-test間の分布差が縮小し、LBスコアが改善する。

高速化: サブサンプリングTCA + PCA次元削減でカーネル行列を小さくする。
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
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.model_selection import LeaveOneGroupOut

from src.eda.data_loader import load_train, load_test, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# サブサンプリング設定
SUBSAMPLE_PER_GROUP = 3  # 各樹種3サンプル（~39 source）
MAX_SUBSAMPLE_TARGET = 30  # target最大30サンプル
PCA_DIM = 30  # PCA次元削減（カーネル計算高速化用）


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def subsample_by_group(n_samples, groups, n_per_group=SUBSAMPLE_PER_GROUP, seed=42):
    """各グループから代表サンプルをサブサンプリング"""
    rng = np.random.RandomState(seed)
    indices = []
    for g in np.unique(groups):
        g_idx = np.where(groups == g)[0]
        n = min(n_per_group, len(g_idx))
        selected = rng.choice(g_idx, size=n, replace=False)
        indices.extend(selected.tolist())
    return np.array(sorted(indices))


def tca_transform_fast(X_source, X_target, n_components=10, kernel="rbf",
                       gamma=None, mu=1.0, source_groups=None, pca_dim=PCA_DIM):
    """高速TCA: PCA次元削減 + サブサンプリング → 固有値問題 → 全データ射影"""
    n_s = len(X_source)
    n_t = len(X_target)

    # PCA次元削減（全データで。TCAはラベルなし設定なのでOK）
    X_all = np.vstack([X_source, X_target])
    actual_pca_dim = min(pca_dim, X_all.shape[1], X_all.shape[0])
    pca = PCA(n_components=actual_pca_dim)
    X_all_pca = pca.fit_transform(X_all)
    X_s_pca = X_all_pca[:n_s]
    X_t_pca = X_all_pca[n_s:]

    # サブサンプリング
    if source_groups is not None and n_s > 50:
        sub_s_idx = subsample_by_group(n_s, source_groups,
                                       n_per_group=SUBSAMPLE_PER_GROUP)
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

    # カーネル行列（サブサンプル間）
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

    # センタリング行列
    H = np.eye(n_sub) - np.ones((n_sub, n_sub)) / n_sub

    # 一般化固有値問題
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
    if kernel == "rbf":
        K_full = rbf_kernel(X_all_pca, X_sub_all, gamma=gamma)
    else:
        K_full = linear_kernel(X_all_pca, X_sub_all)

    # 射影
    Z = K_full @ W
    Z_source = Z[:n_s]
    Z_target = Z[n_s:]

    return Z_source, Z_target


def preprocess(X_tr, X_te, groups_tr, method):
    """前処理を適用する。"""
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


def run_loso_cv(X, y, groups, kernel, n_components, mu, pp_method, reg_name, target_transform):
    """LOSO-CVを実行してRMSEを計算する。"""
    logo = LeaveOneGroupOut()
    all_pred = np.zeros_like(y, dtype=float)
    fold_rmses = []
    fold_species = []

    for tr_idx, te_idx in logo.split(X, y, groups):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr = y[tr_idx].copy()
        y_te = y[te_idx]
        g_tr = groups[tr_idx]

        # 前処理
        X_tr_pp, X_te_pp = preprocess(X_tr, X_te, g_tr, pp_method)

        # TCA変換
        nc = min(n_components, len(te_idx) - 1)
        if nc < 1:
            nc = 1
        try:
            Z_tr, Z_te = tca_transform_fast(
                X_tr_pp, X_te_pp, n_components=nc, kernel=kernel, mu=mu,
                source_groups=g_tr
            )
        except Exception as e:
            return None, None, None

        # 目的変数変換
        if target_transform == "sqrt":
            y_tr_t = np.sqrt(y_tr)
        else:
            y_tr_t = y_tr

        # 回帰
        model = make_regressor(reg_name)
        if reg_name == "pls3" and nc < 3:
            model = PLSRegression(n_components=max(1, nc))

        try:
            pred = fit_predict(model, Z_tr, y_tr_t, Z_te)
        except Exception:
            return None, None, None

        # 逆変換
        if target_transform == "sqrt":
            pred = np.clip(pred, 0, None) ** 2

        all_pred[te_idx] = pred
        fold_rmses.append(rmse(pred, y_te))
        fold_species.append(groups[te_idx][0])

    return rmse(all_pred, y), fold_rmses, fold_species


def main():
    print("=" * 70)
    print("Issue #94 Cycle 2: TCA Domain Adaptation Pipeline")
    print("(PCA + Subsampled TCA)")
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
    print(f"Config: PCA_DIM={PCA_DIM}, SUBSAMPLE={SUBSAMPLE_PER_GROUP}/group, MAX_TARGET={MAX_SUBSAMPLE_TARGET}")
    print()

    # ========================================
    # Phase 1: LOSO-CV グリッドサーチ
    # ========================================
    print("Phase 1: LOSO-CV Grid Search")
    print("-" * 50)

    configs = []
    for kernel in ["linear", "rbf"]:
        for nc in [3, 5, 10, 15, 20]:
            for mu in [0.01, 0.1, 1.0, 10.0]:
                for pp in ["raw", "snv", "epo1"]:
                    for reg in ["pls3", "ridge", "huber"]:
                        for tt in ["raw", "sqrt"]:
                            configs.append((kernel, nc, mu, pp, reg, tt))

    total = len(configs)
    print(f"Total configurations: {total}")

    results = []
    for i, (kernel, nc, mu, pp, reg, tt) in enumerate(configs):
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = elapsed / (i + 1)
            eta = rate * (total - i - 1)
            print(f"  [{i+1}/{total}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

        overall, fold_rmses, fold_species = run_loso_cv(
            X_train, y_train, groups, kernel, nc, mu, pp, reg, tt
        )

        if overall is not None:
            results.append({
                "kernel": kernel, "n_components": nc, "mu": mu,
                "preprocessing": pp, "regressor": reg, "target_transform": tt,
                "rmse": overall, "fold_rmses": fold_rmses, "fold_species": fold_species,
            })

    df_results = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)

    print(f"\nCompleted: {len(df_results)} valid results out of {total}")
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
    Z_tr, Z_te = tca_transform_fast(
        X_tr_pp, X_te_pp, n_components=bnc, kernel=bk, mu=bmu, source_groups=groups
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
        nc = int(row["n_components"])
        Z_tr, Z_te = tca_transform_fast(
            X_tr_pp, X_te_pp, n_components=nc, kernel=row["kernel"], mu=row["mu"],
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
