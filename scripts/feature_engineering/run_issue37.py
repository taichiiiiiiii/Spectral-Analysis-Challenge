"""Issue #37: EDA知見に基づく樹種不変特徴量の評価"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

from src.eda.data_loader import load_train, get_spectral_columns, get_wavenumbers
from src.feature_engineering.issue37_species_invariant import create_species_invariant_features
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "feature_engineering"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def loso_cv_rmse(X, y, groups, model_fn, model_name=""):
    """LOSO-CVでRMSEを計算"""
    logo = LeaveOneGroupOut()
    fold_rmses = []
    for train_idx, test_idx in logo.split(X, y, groups):
        model = model_fn()
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        rmse = np.sqrt(np.mean((pred - y[test_idx]) ** 2))
        fold_rmses.append(rmse)
    return np.mean(fold_rmses), np.std(fold_rmses)


def main():
    # データ読み込み
    df = load_train(DATA_DIR)
    spectral_cols = get_spectral_columns(df)
    wn = get_wavenumbers(spectral_cols)
    X_raw = df[spectral_cols].values
    y = df["含水率"].values
    groups = df["樹種"].values

    print("=== Issue #37: EDA知見に基づく樹種不変特徴量 ===\n")

    # 樹種不変特徴量を計算
    print("特徴量計算中...")
    feat_df = create_species_invariant_features(X_raw, wn)
    print(f"生成された特徴量数: {feat_df.shape[1]}")
    print(f"特徴量名: {list(feat_df.columns)}\n")

    X_feat = feat_df.values

    # EPO前処理
    P_epo = compute_epo_projection(X_raw, groups, n_components=1)
    X_epo = apply_epo(X_raw, P_epo)

    # EPO適用後の樹種不変特徴量
    feat_epo_df = create_species_invariant_features(X_epo, wn)
    X_feat_epo = feat_epo_df.values

    # 結合特徴量: EPO + 樹種不変特徴量
    X_combined = np.hstack([X_epo, X_feat])

    results = []

    # --- 評価 1: 樹種不変特徴量のみ + PLS ---
    print("評価中: 樹種不変特徴量のみ + PLS...")
    for n_comp in [2, 3, 4, 5, 6]:
        rmse, std = loso_cv_rmse(
            X_feat, y, groups,
            lambda nc=n_comp: PLSRegression(n_components=nc),
        )
        results.append({"method": f"InvFeat+PLS({n_comp})", "rmse": rmse, "rmse_std": std})
        print(f"  PLS({n_comp}): RMSE={rmse:.2f} ± {std:.2f}")

    # --- 評価 2: EPO後の樹種不変特徴量 + PLS ---
    print("\n評価中: EPO後の樹種不変特徴量 + PLS...")
    for n_comp in [2, 3, 4, 5, 6]:
        rmse, std = loso_cv_rmse(
            X_feat_epo, y, groups,
            lambda nc=n_comp: PLSRegression(n_components=nc),
        )
        results.append({"method": f"EPO+InvFeat+PLS({n_comp})", "rmse": rmse, "rmse_std": std})
        print(f"  PLS({n_comp}): RMSE={rmse:.2f} ± {std:.2f}")

    # --- 評価 3: 樹種不変特徴量 + SVR ---
    print("\n評価中: 樹種不変特徴量 + SVR...")
    scaler = StandardScaler()
    X_feat_scaled = scaler.fit_transform(X_feat)
    rmse, std = loso_cv_rmse(
        X_feat_scaled, y, groups,
        lambda: SVR(kernel="rbf", C=100, gamma="scale"),
    )
    results.append({"method": "InvFeat+SVR", "rmse": rmse, "rmse_std": std})
    print(f"  SVR: RMSE={rmse:.2f} ± {std:.2f}")

    # --- 評価 4: EPOスペクトル + 樹種不変特徴量の結合 + PLS ---
    print("\n評価中: EPO+InvFeat結合 + PLS...")
    for n_comp in [3, 4, 5, 6, 8]:
        rmse, std = loso_cv_rmse(
            X_combined, y, groups,
            lambda nc=n_comp: PLSRegression(n_components=nc),
        )
        results.append({"method": f"EPO+Raw+InvFeat+PLS({n_comp})", "rmse": rmse, "rmse_std": std})
        print(f"  PLS({n_comp}): RMSE={rmse:.2f} ± {std:.2f}")

    # --- ベースライン ---
    print("\n--- ベースライン ---")
    rmse_base, std_base = loso_cv_rmse(
        X_epo, y, groups,
        lambda: PLSRegression(n_components=4),
    )
    results.append({"method": "EPO+PLS(4) [baseline]", "rmse": rmse_base, "rmse_std": std_base})
    print(f"EPO+PLS(4): RMSE={rmse_base:.2f} ± {std_base:.2f}")

    # 結果まとめ
    result_df = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    print("\n=== 全結果 ===")
    print(result_df.to_string(index=False))

    # プロット
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["green" if "baseline" in m else "steelblue" for m in result_df["method"]]
    ax.barh(range(len(result_df)), result_df["rmse"], color=colors)
    ax.errorbar(result_df["rmse"], range(len(result_df)), xerr=result_df["rmse_std"],
                fmt="none", color="black", capsize=3)
    ax.set_yticks(range(len(result_df)))
    ax.set_yticklabels(result_df["method"])
    ax.set_xlabel("RMSE (LOSO-CV)")
    ax.set_title("Issue #37: Species-Invariant Features Evaluation")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "issue37_species_invariant.png", dpi=150)
    plt.close()
    print(f"\nSaved: {OUT_DIR / 'issue37_species_invariant.png'}")


if __name__ == "__main__":
    main()
