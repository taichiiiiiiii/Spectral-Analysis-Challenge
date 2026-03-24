"""Issue #94: Cycle 2 - TCAドメイン適応をテスト予測パイプラインに統合

仮説: TCA変換後の特徴空間でPLS/Ridge/Huber回帰を行えば、
train-test間の分布差が縮小し、LBスコアが改善する。

パラメータグリッド:
- kernel: ['linear', 'rbf']
- n_components: [3, 5, 10, 15, 20]
- mu: [0.01, 0.1, 1.0, 10.0]
- 前処理: [raw, SNV, EPO(1)]
- 回帰: [PLS(nc=3), Ridge(alpha=1.0), HuberRegressor]
- 目的変数: [raw, sqrt]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import warnings
import time

warnings.filterwarnings("ignore")

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.model_selection import LeaveOneGroupOut

from src.eda.data_loader import load_train, load_test, get_spectral_columns
from src.preprocessing.issue38_tca import tca_transform
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def preprocess(X_tr, X_te, groups_tr, method):
    """前処理を適用する。trainのみでfitし、testはtransformのみ。"""
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
    """回帰モデルを生成する。"""
    if name == "pls3":
        return PLSRegression(n_components=3)
    elif name == "ridge":
        return Ridge(alpha=1.0)
    elif name == "huber":
        return HuberRegressor(max_iter=300)
    else:
        raise ValueError(f"Unknown regressor: {name}")


def fit_predict(model, Z_tr, y_tr, Z_te):
    """モデルの学習と予測を行う。"""
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
        nc = min(n_components, len(tr_idx), len(te_idx))
        try:
            Z_tr, Z_te = tca_transform(X_tr_pp, X_te_pp, n_components=nc, kernel=kernel, mu=mu)
        except Exception:
            return None, None, None

        # 目的変数変換
        if target_transform == "sqrt":
            y_tr_t = np.sqrt(y_tr)
        else:
            y_tr_t = y_tr

        # 回帰
        model = make_regressor(reg_name)
        # PLS の n_components を TCA 次元に合わせる
        if reg_name == "pls3" and nc < 3:
            model = PLSRegression(n_components=nc)

        try:
            pred = fit_predict(model, Z_tr, y_tr_t, Z_te)
        except Exception:
            return None, None, None

        # 逆変換
        if target_transform == "sqrt":
            pred = np.clip(pred, 0, None) ** 2

        all_pred[te_idx] = pred

        fold_rmse = rmse(pred, y_te)
        fold_rmses.append(fold_rmse)
        fold_species.append(groups[te_idx][0])

    overall_rmse = rmse(all_pred, y)
    return overall_rmse, fold_rmses, fold_species


def main():
    print("=" * 70)
    print("Issue #94 Cycle 2: TCA Domain Adaptation Pipeline")
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
    # Phase 1: LOSO-CV グリッドサーチ
    # ========================================
    print("Phase 1: LOSO-CV Grid Search")
    print("-" * 50)

    kernels = ["linear", "rbf"]
    n_components_list = [3, 5, 10, 15, 20]
    mus = [0.01, 0.1, 1.0, 10.0]
    preprocessings = ["raw", "snv", "epo1"]
    regressors = ["pls3", "ridge", "huber"]
    target_transforms = ["raw", "sqrt"]

    results = []
    total = len(kernels) * len(n_components_list) * len(mus) * len(preprocessings) * len(regressors) * len(target_transforms)
    count = 0

    for kernel in kernels:
        for nc in n_components_list:
            for mu in mus:
                for pp in preprocessings:
                    for reg in regressors:
                        for tt in target_transforms:
                            count += 1
                            if count % 50 == 0:
                                print(f"  [{count}/{total}] ...")

                            overall, fold_rmses, fold_species = run_loso_cv(
                                X_train, y_train, groups,
                                kernel, nc, mu, pp, reg, tt
                            )

                            if overall is not None:
                                results.append({
                                    "kernel": kernel,
                                    "n_components": nc,
                                    "mu": mu,
                                    "preprocessing": pp,
                                    "regressor": reg,
                                    "target_transform": tt,
                                    "rmse": overall,
                                    "fold_rmses": fold_rmses,
                                    "fold_species": fold_species,
                                })

    # 結果をDataFrameに
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("rmse").reset_index(drop=True)

    print(f"\nCompleted {count} configurations, {len(df_results)} valid results")
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

    # ベスト設定のfold別RMSE
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

    bk = best["kernel"]
    bnc = int(best["n_components"])
    bmu = best["mu"]
    bpp = best["preprocessing"]
    breg = best["regressor"]
    btt = best["target_transform"]

    # 前処理
    X_tr_pp, X_te_pp = preprocess(X_train, X_test, groups, bpp)

    # TCA変換（train全体 → test全体）
    Z_tr, Z_te = tca_transform(X_tr_pp, X_te_pp, n_components=bnc, kernel=bk, mu=bmu)
    print(f"TCA transformed: train={Z_tr.shape}, test={Z_te.shape}")

    # 目的変数変換
    if btt == "sqrt":
        y_fit = np.sqrt(y_train)
    else:
        y_fit = y_train.copy()

    # 回帰
    model = make_regressor(breg)
    pred_test = fit_predict(model, Z_tr, y_fit, Z_te)

    # 逆変換
    if btt == "sqrt":
        pred_test = np.clip(pred_test, 0, None) ** 2

    print(f"Test predictions: min={pred_test.min():.2f}, max={pred_test.max():.2f}, "
          f"mean={pred_test.mean():.2f}, std={pred_test.std():.2f}")

    # ========================================
    # Phase 3: Top-5設定でのアンサンブルも試す
    # ========================================
    print("\n" + "=" * 70)
    print("Phase 3: Top-5 Ensemble")
    print("=" * 70)

    top5_preds = []
    for i, row in df_results.head(5).iterrows():
        X_tr_pp, X_te_pp = preprocess(X_train, X_test, groups, row["preprocessing"])
        nc = int(row["n_components"])
        Z_tr, Z_te = tca_transform(X_tr_pp, X_te_pp, n_components=nc, kernel=row["kernel"], mu=row["mu"])

        if row["target_transform"] == "sqrt":
            y_f = np.sqrt(y_train)
        else:
            y_f = y_train.copy()

        m = make_regressor(row["regressor"])
        p = fit_predict(m, Z_tr, y_f, Z_te)

        if row["target_transform"] == "sqrt":
            p = np.clip(p, 0, None) ** 2

        top5_preds.append(p)
        print(f"  Config {i+1} (RMSE={row['rmse']:.3f}): pred range [{p.min():.1f}, {p.max():.1f}]")

    ensemble_pred = np.mean(top5_preds, axis=0)
    print(f"\nEnsemble predictions: min={ensemble_pred.min():.2f}, max={ensemble_pred.max():.2f}, "
          f"mean={ensemble_pred.mean():.2f}")

    # ========================================
    # Phase 4: 提出ファイル生成
    # ========================================
    print("\n" + "=" * 70)
    print("Phase 4: Submission File")
    print("=" * 70)

    # ベスト単一モデル
    submit_single = pd.DataFrame({
        0: test_ids.astype(int),
        1: pred_test,
    })
    path_single = OUT_DIR / "submission_v6_tca.csv"
    submit_single.to_csv(path_single, index=False, header=False)
    print(f"Single best submission: {path_single}")

    # Top-5アンサンブル
    submit_ensemble = pd.DataFrame({
        0: test_ids.astype(int),
        1: ensemble_pred,
    })
    path_ensemble = OUT_DIR / "submission_v6_tca_ensemble.csv"
    submit_ensemble.to_csv(path_ensemble, index=False, header=False)
    print(f"Ensemble submission: {path_ensemble}")

    # バリデーション
    sample_submit = pd.read_csv(DATA_DIR / "sample_submit.csv", header=None)
    assert len(submit_single) == len(sample_submit), \
        f"Row count mismatch: {len(submit_single)} vs {len(sample_submit)}"
    print(f"Validation: row count OK ({len(submit_single)} rows)")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    # ========================================
    # 結果サマリーを保存
    # ========================================
    summary_cols = ["kernel", "n_components", "mu", "preprocessing", "regressor", "target_transform", "rmse"]
    df_results[summary_cols].to_csv(OUT_DIR / "issue94_cycle2_tca_results.csv", index=False)
    print(f"Results saved: {OUT_DIR / 'issue94_cycle2_tca_results.csv'}")


if __name__ == "__main__":
    main()
