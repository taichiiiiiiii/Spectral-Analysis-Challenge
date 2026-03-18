"""Issue #27 深掘り: 非線形モデル × 前処理 × PLSスコア特徴量 × 正則化線形モデル"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.eda.data_loader import load_train, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue20_savgol import apply_savgol
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue38_tca import tca_transform

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def loso_cv_evaluate(X_raw, y, groups, model_fn, preprocess_fn=None, target_transform=None):
    """LOSO-CV evaluation with flexible preprocessing and target transform."""
    logo = LeaveOneGroupOut()
    fold_rmses = []

    for train_idx, test_idx in logo.split(X_raw, y, groups):
        X_train, X_test = X_raw[train_idx], X_raw[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Preprocessing
        if preprocess_fn:
            X_train, X_test = preprocess_fn(X_train, X_test, y_train, groups[train_idx])

        # Target transform
        if target_transform == "sqrt":
            y_tr = np.sqrt(y_train)
        else:
            y_tr = y_train

        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Model
        model = model_fn()
        model.fit(X_train_s, y_tr)
        pred = model.predict(X_test_s).ravel()

        # Inverse transform
        if target_transform == "sqrt":
            pred = np.clip(pred, 0, None) ** 2

        fold_rmses.append(np.sqrt(mean_squared_error(y_test, pred)))

    return np.mean(fold_rmses), np.std(fold_rmses)


def preprocess_raw(X_train, X_test, y_train, groups):
    return X_train, X_test

def preprocess_snv(X_train, X_test, y_train, groups):
    return apply_snv(X_train), apply_snv(X_test)

def preprocess_epo(X_train, X_test, y_train, groups):
    P = compute_epo_projection(X_train, groups, n_components=1)
    return apply_epo(X_train, P), apply_epo(X_test, P)

def preprocess_snv_epo(X_train, X_test, y_train, groups):
    X_tr_snv, X_te_snv = apply_snv(X_train), apply_snv(X_test)
    P = compute_epo_projection(X_tr_snv, groups, n_components=1)
    return apply_epo(X_tr_snv, P), apply_epo(X_te_snv, P)

def preprocess_tca(X_train, X_test, y_train, groups):
    X_tr_t, X_te_t = tca_transform(X_train, X_test, n_components=10, gamma=0.001)
    return X_tr_t, X_te_t

def make_pls_score_preprocess(n_comp):
    """PLSスコアを特徴量として使う前処理を返す"""
    def preprocess(X_train, X_test, y_train, groups):
        pls = PLSRegression(n_components=n_comp, max_iter=500)
        pls.fit(X_train, y_train)
        return pls.transform(X_train), pls.transform(X_test)
    return preprocess

def make_epo_pls_score_preprocess(n_comp):
    """EPO + PLSスコアを特徴量として使う"""
    def preprocess(X_train, X_test, y_train, groups):
        P = compute_epo_projection(X_train, groups, n_components=1)
        X_tr_epo = apply_epo(X_train, P)
        X_te_epo = apply_epo(X_test, P)
        pls = PLSRegression(n_components=n_comp, max_iter=500)
        pls.fit(X_tr_epo, y_train)
        return pls.transform(X_tr_epo), pls.transform(X_te_epo)
    return preprocess

def make_pca_preprocess(base_preprocess, n_pca):
    """任意の前処理 + PCA圧縮"""
    def preprocess(X_train, X_test, y_train, groups):
        X_tr, X_te = base_preprocess(X_train, X_test, y_train, groups)
        pca = PCA(n_components=n_pca)
        return pca.fit_transform(X_tr), pca.transform(X_te)
    return preprocess


def main():
    df = load_train(DATA_DIR)
    spectral_cols = get_spectral_columns(df)
    X = df[spectral_cols].values
    y = df["含水率"].values
    groups = df["樹種"].values

    results = []

    # === B: 前処理の組み合わせ × 非線形モデル ===
    print("=== B: 前処理 × 非線形モデル ===")

    preprocessings = {
        "Raw+PCA(10)": make_pca_preprocess(preprocess_raw, 10),
        "SNV+PCA(10)": make_pca_preprocess(preprocess_snv, 10),
        "EPO+PCA(10)": make_pca_preprocess(preprocess_epo, 10),
        "SNV+EPO+PCA(10)": make_pca_preprocess(preprocess_snv_epo, 10),
        "TCA(10)": preprocess_tca,
    }

    models_b = {
        "SVR": lambda: SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1),
        "RF": lambda: RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    }

    for pp_name, pp_fn in preprocessings.items():
        for model_name, model_fn in models_b.items():
            for target in [None, "sqrt"]:
                label = f"{pp_name}+{model_name}" + ("+sqrt(y)" if target else "")
                try:
                    rmse, rmse_std = loso_cv_evaluate(X, y, groups, model_fn, pp_fn, target)
                    results.append({"experiment": "B", "method": label, "rmse": round(rmse, 2), "rmse_std": round(rmse_std, 2)})
                    print(f"  {label}: RMSE={rmse:.2f} ± {rmse_std:.2f}")
                except Exception as e:
                    print(f"  {label}: ERROR - {e}")

    # === C: PLSスコアを特徴量に ===
    print("\n=== C: PLSスコア特徴量 × 非線形モデル ===")

    pls_preprocessings = {
        "PLSscore(4)": make_pls_score_preprocess(4),
        "PLSscore(6)": make_pls_score_preprocess(6),
        "PLSscore(8)": make_pls_score_preprocess(8),
        "EPO+PLSscore(4)": make_epo_pls_score_preprocess(4),
        "EPO+PLSscore(6)": make_epo_pls_score_preprocess(6),
    }

    models_c = {
        "SVR": lambda: SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1),
        "RF": lambda: RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
        "Ridge": lambda: Ridge(alpha=1.0),
    }

    for pp_name, pp_fn in pls_preprocessings.items():
        for model_name, model_fn in models_c.items():
            for target in [None, "sqrt"]:
                label = f"{pp_name}+{model_name}" + ("+sqrt(y)" if target else "")
                try:
                    rmse, rmse_std = loso_cv_evaluate(X, y, groups, model_fn, pp_fn, target)
                    results.append({"experiment": "C", "method": label, "rmse": round(rmse, 2), "rmse_std": round(rmse_std, 2)})
                    print(f"  {label}: RMSE={rmse:.2f} ± {rmse_std:.2f}")
                except Exception as e:
                    print(f"  {label}: ERROR - {e}")

    # === D: 正則化線形モデル ===
    print("\n=== D: 正則化線形モデル ===")

    d_preprocessings = {
        "Raw+PCA(10)": make_pca_preprocess(preprocess_raw, 10),
        "SNV+PCA(10)": make_pca_preprocess(preprocess_snv, 10),
        "EPO+PCA(10)": make_pca_preprocess(preprocess_epo, 10),
        "TCA(10)": preprocess_tca,
        "PLSscore(4)": make_pls_score_preprocess(4),
        "EPO+PLSscore(4)": make_epo_pls_score_preprocess(4),
    }

    models_d = {
        "Ridge(0.1)": lambda: Ridge(alpha=0.1),
        "Ridge(1)": lambda: Ridge(alpha=1.0),
        "Ridge(10)": lambda: Ridge(alpha=10.0),
        "Ridge(100)": lambda: Ridge(alpha=100.0),
        "Lasso(0.1)": lambda: Lasso(alpha=0.1, max_iter=5000),
        "Lasso(1)": lambda: Lasso(alpha=1.0, max_iter=5000),
        "ElasticNet(0.1)": lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
        "ElasticNet(1)": lambda: ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000),
    }

    for pp_name, pp_fn in d_preprocessings.items():
        for model_name, model_fn in models_d.items():
            for target in [None, "sqrt"]:
                label = f"{pp_name}+{model_name}" + ("+sqrt(y)" if target else "")
                try:
                    rmse, rmse_std = loso_cv_evaluate(X, y, groups, model_fn, pp_fn, target)
                    results.append({"experiment": "D", "method": label, "rmse": round(rmse, 2), "rmse_std": round(rmse_std, 2)})
                    print(f"  {label}: RMSE={rmse:.2f} ± {rmse_std:.2f}")
                except Exception as e:
                    print(f"  {label}: ERROR - {e}")

    # === ベースライン追加 ===
    print("\n=== Baselines ===")
    baselines = {
        "Raw+PLS(4)": (preprocess_raw, lambda: PLSRegression(n_components=4, max_iter=500), None),
        "EPO+PLS(4)": (preprocess_epo, lambda: PLSRegression(n_components=4, max_iter=500), None),
        "SNV+PLS(2)": (preprocess_snv, lambda: PLSRegression(n_components=2, max_iter=500), None),
        "TCA(10)+PLS(4)+sqrt(y)": (preprocess_tca, lambda: PLSRegression(n_components=4, max_iter=500), "sqrt"),
    }
    for label, (pp_fn, model_fn, target) in baselines.items():
        rmse, rmse_std = loso_cv_evaluate(X, y, groups, model_fn, pp_fn, target)
        results.append({"experiment": "baseline", "method": label, "rmse": round(rmse, 2), "rmse_std": round(rmse_std, 2)})
        print(f"  {label}: RMSE={rmse:.2f} ± {rmse_std:.2f}")

    # === Results summary ===
    results_df = pd.DataFrame(results).sort_values("rmse")
    print("\n=== 全結果 Top 20 ===")
    print(results_df.head(20).to_string(index=False))

    results_df.to_csv(OUT_DIR / "issue27_deep_dive.csv", index=False)

    # === Figure ===
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    for i, (exp, title) in enumerate([("B", "B: Preprocessing x Models"), ("C", "C: PLS Score Features"), ("D", "D: Regularized Linear")]):
        ax = axes[i]
        exp_df = results_df[results_df["experiment"] == exp].head(15)
        if len(exp_df) == 0:
            continue
        colors = ['#e74c3c' if r < 22 else '#3498db' for r in exp_df['rmse']]
        ax.barh(range(len(exp_df)), exp_df["rmse"], color=colors)
        ax.set_yticks(range(len(exp_df)))
        ax.set_yticklabels(exp_df["method"], fontsize=7)
        ax.set_xlabel("RMSE")
        ax.set_title(title)
        ax.axvline(21.94, color='green', linestyle='--', alpha=0.7, label='EPO+PLS(4)=21.94')
        ax.axvline(19.91, color='red', linestyle='--', alpha=0.7, label='TCA+PLS+sqrt=19.91')
        ax.legend(fontsize=7)
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "issue27_deep_dive.png", dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {OUT_DIR / 'issue27_deep_dive.png'}")


if __name__ == "__main__":
    main()
