"""Issue #32: 改良アンサンブル評価

既存4モデル + 追加モデルでアンサンブルの多様性を高め、RMSE改善を狙う。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from scipy.optimize import minimize

from src.eda.data_loader import load_train, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_extended_oof(df, spectral_cols):
    """拡張ベースモデルのOOF予測を生成"""
    X_raw = df[spectral_cols].values
    y = df["含水率"].values
    groups = df["樹種"].values
    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_raw, y, groups))
    n_samples = len(df)
    oof = {}

    # --- 既存4モデル ---
    # Model 1: EPO(1) + PLS(4) — ベスト単体モデル
    preds = np.zeros(n_samples)
    for train_idx, test_idx in folds:
        P = compute_epo_projection(X_raw[train_idx], groups[train_idx], n_components=1)
        pls = PLSRegression(n_components=4)
        pls.fit(apply_epo(X_raw[train_idx], P), y[train_idx])
        preds[test_idx] = pls.predict(apply_epo(X_raw[test_idx], P)).ravel()
    oof["EPO1+PLS4"] = preds.copy()

    # Model 2: Raw + PLS(4)
    preds = np.zeros(n_samples)
    for train_idx, test_idx in folds:
        pls = PLSRegression(n_components=4)
        pls.fit(X_raw[train_idx], y[train_idx])
        preds[test_idx] = pls.predict(X_raw[test_idx]).ravel()
    oof["Raw+PLS4"] = preds.copy()

    # Model 3: EPO(1) + PCA(10) + SVR
    preds = np.zeros(n_samples)
    for train_idx, test_idx in folds:
        P = compute_epo_projection(X_raw[train_idx], groups[train_idx], n_components=1)
        X_tr = apply_epo(X_raw[train_idx], P)
        X_te = apply_epo(X_raw[test_idx], P)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        pca = PCA(n_components=10)
        X_tr_pca = pca.fit_transform(X_tr_s)
        X_te_pca = pca.transform(X_te_s)
        svr = SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1)
        svr.fit(X_tr_pca, y[train_idx])
        preds[test_idx] = svr.predict(X_te_pca)
    oof["EPO1+SVR"] = preds.copy()

    # Model 4: SNV + PLS(4)
    preds = np.zeros(n_samples)
    for train_idx, test_idx in folds:
        X_tr = apply_snv(X_raw[train_idx])
        X_te = apply_snv(X_raw[test_idx])
        pls = PLSRegression(n_components=4)
        pls.fit(X_tr, y[train_idx])
        preds[test_idx] = pls.predict(X_te).ravel()
    oof["SNV+PLS4"] = preds.copy()

    # --- 追加モデル ---
    # Model 5: EPO(1) + PLS(3) — 少し正則化が強い
    preds = np.zeros(n_samples)
    for train_idx, test_idx in folds:
        P = compute_epo_projection(X_raw[train_idx], groups[train_idx], n_components=1)
        pls = PLSRegression(n_components=3)
        pls.fit(apply_epo(X_raw[train_idx], P), y[train_idx])
        preds[test_idx] = pls.predict(apply_epo(X_raw[test_idx], P)).ravel()
    oof["EPO1+PLS3"] = preds.copy()

    # Model 6: EPO(1) + PLS(5) — やや柔軟
    preds = np.zeros(n_samples)
    for train_idx, test_idx in folds:
        P = compute_epo_projection(X_raw[train_idx], groups[train_idx], n_components=1)
        pls = PLSRegression(n_components=5)
        pls.fit(apply_epo(X_raw[train_idx], P), y[train_idx])
        preds[test_idx] = pls.predict(apply_epo(X_raw[test_idx], P)).ravel()
    oof["EPO1+PLS5"] = preds.copy()

    # Model 7: EPO(2) + PLS(4)
    preds = np.zeros(n_samples)
    for train_idx, test_idx in folds:
        P = compute_epo_projection(X_raw[train_idx], groups[train_idx], n_components=2)
        pls = PLSRegression(n_components=4)
        pls.fit(apply_epo(X_raw[train_idx], P), y[train_idx])
        preds[test_idx] = pls.predict(apply_epo(X_raw[test_idx], P)).ravel()
    oof["EPO2+PLS4"] = preds.copy()

    # Model 8: Raw + PLS(3)
    preds = np.zeros(n_samples)
    for train_idx, test_idx in folds:
        pls = PLSRegression(n_components=3)
        pls.fit(X_raw[train_idx], y[train_idx])
        preds[test_idx] = pls.predict(X_raw[test_idx]).ravel()
    oof["Raw+PLS3"] = preds.copy()

    # Model 9: SNV + PLS(3)
    preds = np.zeros(n_samples)
    for train_idx, test_idx in folds:
        X_tr = apply_snv(X_raw[train_idx])
        X_te = apply_snv(X_raw[test_idx])
        pls = PLSRegression(n_components=3)
        pls.fit(X_tr, y[train_idx])
        preds[test_idx] = pls.predict(X_te).ravel()
    oof["SNV+PLS3"] = preds.copy()

    # Model 10: EPO(1) + PCA(20) + SVR（異なるPCA次元数）
    preds = np.zeros(n_samples)
    for train_idx, test_idx in folds:
        P = compute_epo_projection(X_raw[train_idx], groups[train_idx], n_components=1)
        X_tr = apply_epo(X_raw[train_idx], P)
        X_te = apply_epo(X_raw[test_idx], P)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        pca = PCA(n_components=20)
        X_tr_pca = pca.fit_transform(X_tr_s)
        X_te_pca = pca.transform(X_te_s)
        svr = SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1)
        svr.fit(X_tr_pca, y[train_idx])
        preds[test_idx] = svr.predict(X_te_pca)
    oof["EPO1+PCA20+SVR"] = preds.copy()

    return pd.DataFrame(oof)


def optimize_weights(oof_matrix, y):
    """重みの最適化（合計=1, 非負制約）"""
    n_models = oof_matrix.shape[1]
    init_w = np.ones(n_models) / n_models

    def obj(w):
        pred = oof_matrix @ w
        return np.sqrt(np.mean((pred - y) ** 2))

    res = minimize(obj, init_w, method="SLSQP",
                   bounds=[(0, 1)] * n_models,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    return res.x


def main():
    df = load_train(DATA_DIR)
    spectral_cols = get_spectral_columns(df)
    y = df["含水率"].values
    groups = df["樹種"].values
    logo = LeaveOneGroupOut()

    print("=== Issue #32: 改良アンサンブル評価 ===\n")
    print("OOF予測生成中...")
    oof_df = generate_extended_oof(df, spectral_cols)
    oof_matrix = oof_df.values
    folds = list(logo.split(oof_matrix, y, groups))

    results = []

    # 個別モデルのRMSE
    print("\n--- 個別モデル ---")
    for col in oof_df.columns:
        fold_rmses = [
            float(np.sqrt(np.mean((oof_df[col].values[ti] - y[ti]) ** 2)))
            for _, ti in folds
        ]
        rmse = np.mean(fold_rmses)
        results.append({"method": col, "rmse": rmse, "rmse_std": np.std(fold_rmses)})
        print(f"  {col}: RMSE={rmse:.2f}")

    # 既存4モデルのSimpleAvg
    orig_cols = ["EPO1+PLS4", "Raw+PLS4", "EPO1+SVR", "SNV+PLS4"]
    fold_rmses = [
        float(np.sqrt(np.mean((oof_df[orig_cols].values[ti].mean(axis=1) - y[ti]) ** 2)))
        for _, ti in folds
    ]
    rmse = np.mean(fold_rmses)
    results.append({"method": "SimpleAvg(4 orig)", "rmse": rmse, "rmse_std": np.std(fold_rmses)})
    print(f"\n  SimpleAvg(4 orig): RMSE={rmse:.2f}")

    # 全10モデルのSimpleAvg
    fold_rmses = [
        float(np.sqrt(np.mean((oof_matrix[ti].mean(axis=1) - y[ti]) ** 2)))
        for _, ti in folds
    ]
    rmse = np.mean(fold_rmses)
    results.append({"method": "SimpleAvg(10 all)", "rmse": rmse, "rmse_std": np.std(fold_rmses)})
    print(f"  SimpleAvg(10 all): RMSE={rmse:.2f}")

    # 最適重み付け（nested LOSO-CV）
    print("\n--- 重み最適化（nested CV） ---")

    # 全10モデル最適重み
    fold_rmses = []
    all_weights = []
    for train_idx, test_idx in folds:
        w = optimize_weights(oof_matrix[train_idx], y[train_idx])
        all_weights.append(w)
        pred = oof_matrix[test_idx] @ w
        fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
    rmse = np.mean(fold_rmses)
    results.append({"method": "OptWeights(10 all)", "rmse": rmse, "rmse_std": np.std(fold_rmses)})
    print(f"  OptWeights(10 all): RMSE={rmse:.2f}")

    # 平均重みを表示
    avg_w = np.mean(all_weights, axis=0)
    print("  平均重み:")
    for col, w in zip(oof_df.columns, avg_w):
        if w > 0.01:
            print(f"    {col}: {w:.3f}")

    # 既存4モデル最適重み
    fold_rmses = []
    for train_idx, test_idx in folds:
        w = optimize_weights(oof_df[orig_cols].values[train_idx], y[train_idx])
        pred = oof_df[orig_cols].values[test_idx] @ w
        fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
    rmse = np.mean(fold_rmses)
    results.append({"method": "OptWeights(4 orig)", "rmse": rmse, "rmse_std": np.std(fold_rmses)})
    print(f"  OptWeights(4 orig): RMSE={rmse:.2f}")

    # Inverse-variance weighting
    fold_rmses = []
    for train_idx, test_idx in folds:
        vars_ = np.array([np.mean((oof_matrix[train_idx, i] - y[train_idx]) ** 2)
                          for i in range(oof_matrix.shape[1])])
        inv_var = 1.0 / (vars_ + 1e-10)
        w = inv_var / inv_var.sum()
        pred = oof_matrix[test_idx] @ w
        fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
    rmse = np.mean(fold_rmses)
    results.append({"method": "InvVar(10 all)", "rmse": rmse, "rmse_std": np.std(fold_rmses)})
    print(f"  InvVar(10 all): RMSE={rmse:.2f}")

    # ElasticNet stacking
    fold_rmses = []
    for train_idx, test_idx in folds:
        meta = ElasticNet(alpha=0.1, l1_ratio=0.5, positive=True, max_iter=5000)
        meta.fit(oof_matrix[train_idx], y[train_idx])
        pred = meta.predict(oof_matrix[test_idx])
        fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
    rmse = np.mean(fold_rmses)
    results.append({"method": "Stacking(ElasticNet)", "rmse": rmse, "rmse_std": np.std(fold_rmses)})
    print(f"  Stacking(ElasticNet): RMSE={rmse:.2f}")

    # Ridge stacking
    fold_rmses = []
    for train_idx, test_idx in folds:
        meta = Ridge(alpha=1.0)
        meta.fit(oof_matrix[train_idx], y[train_idx])
        pred = meta.predict(oof_matrix[test_idx])
        fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
    rmse = np.mean(fold_rmses)
    results.append({"method": "Stacking(Ridge)", "rmse": rmse, "rmse_std": np.std(fold_rmses)})
    print(f"  Stacking(Ridge): RMSE={rmse:.2f}")

    # 上位モデルのみのサブセットSimpleAvg
    # PLS系のみ（3, 4, 5成分でEPO1）
    pls_cols = ["EPO1+PLS3", "EPO1+PLS4", "EPO1+PLS5"]
    fold_rmses = [
        float(np.sqrt(np.mean((oof_df[pls_cols].values[ti].mean(axis=1) - y[ti]) ** 2)))
        for _, ti in folds
    ]
    rmse = np.mean(fold_rmses)
    results.append({"method": "SimpleAvg(EPO+PLS3,4,5)", "rmse": rmse, "rmse_std": np.std(fold_rmses)})
    print(f"\n  SimpleAvg(EPO+PLS3,4,5): RMSE={rmse:.2f}")

    # 結果まとめ
    result_df = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    print("\n=== 全結果 ===")
    print(result_df.to_string(index=False))

    # プロット
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = []
    for m in result_df["method"]:
        if "orig" in m and "SimpleAvg" in m:
            colors.append("green")
        elif "SimpleAvg" in m or "OptWeights" in m or "Stacking" in m or "InvVar" in m:
            colors.append("steelblue")
        else:
            colors.append("lightgray")
    ax.barh(range(len(result_df)), result_df["rmse"], color=colors)
    ax.errorbar(result_df["rmse"], range(len(result_df)), xerr=result_df["rmse_std"],
                fmt="none", color="black", capsize=3)
    ax.set_yticks(range(len(result_df)))
    ax.set_yticklabels(result_df["method"], fontsize=8)
    ax.set_xlabel("RMSE (LOSO-CV)")
    ax.set_title("Issue #32: Improved Ensemble Evaluation")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "issue32_improved_ensemble.png", dpi=150)
    plt.close()
    print(f"\nSaved: {OUT_DIR / 'issue32_improved_ensemble.png'}")


if __name__ == "__main__":
    main()
