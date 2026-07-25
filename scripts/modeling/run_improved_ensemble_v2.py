"""改良アンサンブルv2: 深掘りで発見した新モデルを統合"""
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
from sklearn.model_selection import LeaveOneGroupOut, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from scipy.optimize import minimize

from src.eda.data_loader import load_train, load_test, get_spectral_columns, get_wavenumbers
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue20_savgol import apply_savgol
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def cars_select(X_train, y_train, X_test, n_pls=4, n_iterations=30):
    """CARS波長選択"""
    n_samples, n_features = X_train.shape
    remaining = np.arange(n_features)
    ratio = (max(n_pls + 2, 5) / n_features) ** (1.0 / n_iterations)
    best_rmse = np.inf
    best_idx = remaining.copy()

    for i in range(n_iterations):
        n_keep = max(n_pls + 1, int(n_features * ratio ** (i + 1)))
        if n_keep >= len(remaining):
            continue
        n_comp = min(n_pls, len(remaining) - 1, n_samples - 1)
        if n_comp < 1:
            break
        pls = PLSRegression(n_components=n_comp)
        pls.fit(X_train[:, remaining], y_train)
        coefs = np.abs(pls.coef_.ravel())
        top_idx = np.argsort(coefs)[::-1][:n_keep]
        remaining = remaining[np.sort(top_idx)]

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_rmses = []
        for tr, te in kf.split(X_train[:, remaining]):
            n_c = min(n_comp, len(remaining) - 1)
            if n_c < 1:
                break
            p = PLSRegression(n_components=n_c)
            p.fit(X_train[tr][:, remaining], y_train[tr])
            pred = p.predict(X_train[te][:, remaining]).ravel()
            cv_rmses.append(np.sqrt(np.mean((pred - y_train[te]) ** 2)))
        if cv_rmses:
            rmse = np.mean(cv_rmses)
            if rmse < best_rmse:
                best_rmse = rmse
                best_idx = remaining.copy()

    return X_train[:, best_idx], X_test[:, best_idx], best_idx


def main():
    df = load_train(DATA_DIR)
    spectral_cols = get_spectral_columns(df)
    wn = get_wavenumbers(spectral_cols)
    X_raw = df[spectral_cols].values
    y = df["含水率"].values
    groups = df["樹種"].values

    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_raw, y, groups))
    n_samples = len(df)
    np.random.seed(42)

    oof = {}

    # === 既存4モデル ===
    print("OOF生成: 既存モデル + 新モデル")

    # M1: EPO(1)+PLS(4)
    preds = np.zeros(n_samples)
    for train_idx, test_idx in folds:
        P = compute_epo_projection(X_raw[train_idx], groups[train_idx], n_components=1)
        pls = PLSRegression(n_components=4)
        pls.fit(apply_epo(X_raw[train_idx], P), y[train_idx])
        preds[test_idx] = pls.predict(apply_epo(X_raw[test_idx], P)).ravel()
    oof["EPO1+PLS4"] = preds.copy()
    print(f"  EPO1+PLS4: {np.sqrt(np.mean((preds - y)**2)):.2f}")

    # M2: Raw+PLS(4)
    preds = np.zeros(n_samples)
    for train_idx, test_idx in folds:
        pls = PLSRegression(n_components=4)
        pls.fit(X_raw[train_idx], y[train_idx])
        preds[test_idx] = pls.predict(X_raw[test_idx]).ravel()
    oof["Raw+PLS4"] = preds.copy()
    print(f"  Raw+PLS4: {np.sqrt(np.mean((preds - y)**2)):.2f}")

    # M3: EPO(1)+PCA(10)+SVR
    preds = np.zeros(n_samples)
    for train_idx, test_idx in folds:
        P = compute_epo_projection(X_raw[train_idx], groups[train_idx], n_components=1)
        X_tr = apply_epo(X_raw[train_idx], P)
        X_te = apply_epo(X_raw[test_idx], P)
        scaler = StandardScaler()
        pca = PCA(n_components=10)
        X_tr_pca = pca.fit_transform(scaler.fit_transform(X_tr))
        X_te_pca = pca.transform(scaler.transform(X_te))
        svr = SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1)
        svr.fit(X_tr_pca, y[train_idx])
        preds[test_idx] = svr.predict(X_te_pca)
    oof["EPO1+SVR"] = preds.copy()
    print(f"  EPO1+SVR: {np.sqrt(np.mean((preds - y)**2)):.2f}")

    # M4: SNV+PLS(4)
    preds = np.zeros(n_samples)
    for train_idx, test_idx in folds:
        pls = PLSRegression(n_components=4)
        pls.fit(apply_snv(X_raw[train_idx]), y[train_idx])
        preds[test_idx] = pls.predict(apply_snv(X_raw[test_idx])).ravel()
    oof["SNV+PLS4"] = preds.copy()
    print(f"  SNV+PLS4: {np.sqrt(np.mean((preds - y)**2)):.2f}")

    # === 新モデル（深掘りから発見） ===

    # M5: SNV+PLS(2) — 最適PLS成分数
    preds = np.zeros(n_samples)
    for train_idx, test_idx in folds:
        pls = PLSRegression(n_components=2)
        pls.fit(apply_snv(X_raw[train_idx]), y[train_idx])
        preds[test_idx] = pls.predict(apply_snv(X_raw[test_idx])).ravel()
    oof["SNV+PLS2"] = preds.copy()
    print(f"  SNV+PLS2: {np.sqrt(np.mean((preds - y)**2)):.2f}")

    # M6: SNV+PLS(2)+sqrt(y)
    y_sqrt = np.sqrt(y)
    preds = np.zeros(n_samples)
    for train_idx, test_idx in folds:
        pls = PLSRegression(n_components=2)
        pls.fit(apply_snv(X_raw[train_idx]), y_sqrt[train_idx])
        pred_sqrt = pls.predict(apply_snv(X_raw[test_idx])).ravel()
        preds[test_idx] = np.clip(pred_sqrt, 0, None) ** 2
    oof["SNV+PLS2+sqrt"] = preds.copy()
    print(f"  SNV+PLS2+sqrt: {np.sqrt(np.mean((preds - y)**2)):.2f}")

    # M7: EPO(1)+CARS+PLS(3)
    preds = np.zeros(n_samples)
    for train_idx, test_idx in folds:
        P = compute_epo_projection(X_raw[train_idx], groups[train_idx], n_components=1)
        X_tr = apply_epo(X_raw[train_idx], P)
        X_te = apply_epo(X_raw[test_idx], P)
        X_tr_sel, X_te_sel, _ = cars_select(X_tr, y[train_idx], X_te, n_pls=3, n_iterations=30)
        n_c = min(3, X_tr_sel.shape[1] - 1)
        pls = PLSRegression(n_components=max(1, n_c))
        pls.fit(X_tr_sel, y[train_idx])
        preds[test_idx] = pls.predict(X_te_sel).ravel()
    oof["EPO1+CARS+PLS3"] = preds.copy()
    print(f"  EPO1+CARS+PLS3: {np.sqrt(np.mean((preds - y)**2)):.2f}")

    # M8: SG2d(w=7)+EPO(1)+PLS(3)
    preds = np.zeros(n_samples)
    for train_idx, test_idx in folds:
        X_tr = apply_savgol(X_raw[train_idx], deriv=2, window_length=7)
        X_te = apply_savgol(X_raw[test_idx], deriv=2, window_length=7)
        P = compute_epo_projection(X_tr, groups[train_idx], n_components=1)
        X_tr = apply_epo(X_tr, P)
        X_te = apply_epo(X_te, P)
        pls = PLSRegression(n_components=3)
        pls.fit(X_tr, y[train_idx])
        preds[test_idx] = pls.predict(X_te).ravel()
    oof["SG2d+EPO1+PLS3"] = preds.copy()
    print(f"  SG2d+EPO1+PLS3: {np.sqrt(np.mean((preds - y)**2)):.2f}")

    # M9: EPO(1)+PLS(4)+sqrt(y)
    preds = np.zeros(n_samples)
    for train_idx, test_idx in folds:
        P = compute_epo_projection(X_raw[train_idx], groups[train_idx], n_components=1)
        X_tr = apply_epo(X_raw[train_idx], P)
        X_te = apply_epo(X_raw[test_idx], P)
        pls = PLSRegression(n_components=4)
        pls.fit(X_tr, y_sqrt[train_idx])
        pred_sqrt = pls.predict(X_te).ravel()
        preds[test_idx] = np.clip(pred_sqrt, 0, None) ** 2
    oof["EPO1+PLS4+sqrt"] = preds.copy()
    print(f"  EPO1+PLS4+sqrt: {np.sqrt(np.mean((preds - y)**2)):.2f}")

    # M10: SNV+PLS(3)
    preds = np.zeros(n_samples)
    for train_idx, test_idx in folds:
        pls = PLSRegression(n_components=3)
        pls.fit(apply_snv(X_raw[train_idx]), y[train_idx])
        preds[test_idx] = pls.predict(apply_snv(X_raw[test_idx])).ravel()
    oof["SNV+PLS3"] = preds.copy()
    print(f"  SNV+PLS3: {np.sqrt(np.mean((preds - y)**2)):.2f}")

    oof_df = pd.DataFrame(oof)
    oof_matrix = oof_df.values

    # === アンサンブル評価 ===
    print("\n=== アンサンブル評価 ===")
    results = []

    # 個別モデルRMSE
    for col in oof_df.columns:
        fold_rmses = [
            float(np.sqrt(np.mean((oof_df[col].values[ti] - y[ti]) ** 2)))
            for _, ti in folds
        ]
        results.append({"method": col, "rmse": np.mean(fold_rmses), "rmse_std": np.std(fold_rmses)})

    # 既存4モデルSimpleAvg
    orig_cols = ["EPO1+PLS4", "Raw+PLS4", "EPO1+SVR", "SNV+PLS4"]
    fold_rmses = [
        float(np.sqrt(np.mean((oof_df[orig_cols].values[ti].mean(axis=1) - y[ti]) ** 2)))
        for _, ti in folds
    ]
    results.append({"method": "SimpleAvg(4 orig)", "rmse": np.mean(fold_rmses), "rmse_std": np.std(fold_rmses)})

    # 全10モデルSimpleAvg
    fold_rmses = [
        float(np.sqrt(np.mean((oof_matrix[ti].mean(axis=1) - y[ti]) ** 2)))
        for _, ti in folds
    ]
    results.append({"method": "SimpleAvg(10 all)", "rmse": np.mean(fold_rmses), "rmse_std": np.std(fold_rmses)})

    # 新モデル含むサブセットSimpleAvg
    for subset_name, cols in [
        ("Top6", ["EPO1+PLS4", "Raw+PLS4", "EPO1+SVR", "SNV+PLS2", "SNV+PLS2+sqrt", "EPO1+CARS+PLS3"]),
        ("SNV系", ["SNV+PLS2", "SNV+PLS3", "SNV+PLS4", "SNV+PLS2+sqrt"]),
        ("新5モデル", ["SNV+PLS2", "SNV+PLS2+sqrt", "EPO1+CARS+PLS3", "SG2d+EPO1+PLS3", "EPO1+PLS4+sqrt"]),
        ("orig4+新2", ["EPO1+PLS4", "Raw+PLS4", "EPO1+SVR", "SNV+PLS4", "SNV+PLS2+sqrt", "EPO1+CARS+PLS3"]),
        ("多様性重視7", ["EPO1+PLS4", "SNV+PLS2", "SNV+PLS2+sqrt", "EPO1+CARS+PLS3", "SG2d+EPO1+PLS3", "EPO1+SVR", "Raw+PLS4"]),
        ("ベスト8", ["EPO1+PLS4", "Raw+PLS4", "EPO1+SVR", "SNV+PLS4", "SNV+PLS2", "SNV+PLS2+sqrt", "EPO1+CARS+PLS3", "EPO1+PLS4+sqrt"]),
    ]:
        fold_rmses = [
            float(np.sqrt(np.mean((oof_df[cols].values[ti].mean(axis=1) - y[ti]) ** 2)))
            for _, ti in folds
        ]
        results.append({"method": f"SimpleAvg({subset_name})", "rmse": np.mean(fold_rmses), "rmse_std": np.std(fold_rmses)})

    # 最適重み（全10モデル, nested CV）
    fold_rmses = []
    all_weights = []
    for train_idx, test_idx in folds:
        def obj(w):
            return np.sqrt(np.mean((oof_matrix[train_idx] @ w - y[train_idx]) ** 2))
        res = minimize(obj, np.ones(oof_matrix.shape[1]) / oof_matrix.shape[1],
                       method="SLSQP",
                       bounds=[(0, 1)] * oof_matrix.shape[1],
                       constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
        all_weights.append(res.x)
        pred = oof_matrix[test_idx] @ res.x
        fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
    results.append({"method": "OptWeights(10 all)", "rmse": np.mean(fold_rmses), "rmse_std": np.std(fold_rmses)})

    # 結果表示
    result_df = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    print("\n" + result_df.to_string(index=False))

    # 平均最適重み
    avg_w = np.mean(all_weights, axis=0)
    print("\n最適重み（平均）:")
    for col, w in sorted(zip(oof_df.columns, avg_w), key=lambda x: -x[1]):
        if w > 0.01:
            print(f"  {col}: {w:.3f}")

    # プロット
    fig, ax = plt.subplots(figsize=(12, 8))
    ensemble_mask = result_df["method"].str.contains("SimpleAvg|OptWeights")
    colors = ["steelblue" if e else "lightgray" for e in ensemble_mask]
    ax.barh(range(len(result_df)), result_df["rmse"], color=colors)
    ax.errorbar(result_df["rmse"], range(len(result_df)), xerr=result_df["rmse_std"],
                fmt="none", color="black", capsize=3)
    ax.set_yticks(range(len(result_df)))
    ax.set_yticklabels(result_df["method"], fontsize=7)
    ax.set_xlabel("RMSE (LOSO-CV)")
    ax.set_title("Improved Ensemble v2 (with deep-dive discoveries)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "improved_ensemble_v2.png", dpi=150)
    plt.close()
    print(f"\nSaved: {OUT_DIR / 'improved_ensemble_v2.png'}")


if __name__ == "__main__":
    main()
