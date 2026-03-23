"""Issue #68: Savitzky-Golay微分パラメータ × EPO × PLS 系統的評価

SG微分パラメータ（deriv, window_length, polyorder）の組み合わせを
EPO+PLS / Direct PLS パイプラインと組み合わせてLOSO-CVで評価する。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut

from src.eda.data_loader import load_train, get_spectral_columns
from src.preprocessing.issue20_savgol import apply_savgol
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main():
    t0 = time.time()

    # --- データ読み込み ---
    print("Loading data...")
    df = load_train(DATA_DIR)
    spectral_cols = get_spectral_columns(df)
    X_raw = df[spectral_cols].values.astype(np.float64)
    y = df["含水率"].values.astype(np.float64)
    groups = df["樹種"].values

    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_raw, y, groups))
    n_folds = len(folds)
    print(f"Data: {X_raw.shape[0]} samples, {X_raw.shape[1]} features, {n_folds} folds")

    # --- SG パラメータ組み合わせ生成 ---
    sg_params = []
    for deriv in [1, 2]:
        for wl in [5, 7, 9, 11, 13, 15]:
            for po in [2, 3]:
                if po < wl and po >= deriv:
                    sg_params.append((deriv, wl, po))

    print(f"SG parameter combinations: {len(sg_params)}")

    # パイプライン定義
    pipelines = [
        ("EPO(1)+PLS(3)+raw", 1, 3, "raw"),
        ("EPO(1)+PLS(4)+sqrt", 1, 4, "sqrt"),
        ("PLS(2)+sqrt", None, 2, "sqrt"),
    ]

    total_combos = len(sg_params) * len(pipelines)
    print(f"Total combinations: {total_combos}")
    print("=" * 80)

    # --- 全SG前処理をサンプル全体に一括適用してキャッシュ ---
    print("Pre-computing SG derivatives (full dataset)...")
    sg_full_cache = {}
    for deriv, wl, po in sg_params:
        sg_full_cache[(deriv, wl, po)] = apply_savgol(X_raw, deriv=deriv, window_length=wl, polyorder=po)
    print(f"SG full cache built: {len(sg_full_cache)} entries ({time.time() - t0:.1f}s)")

    # --- 実験実行: SGパラメータ毎にfoldをループ ---
    results = []
    combo_count = 0

    for sg_idx, (deriv, wl, po) in enumerate(sg_params):
        X_sg_all = sg_full_cache[(deriv, wl, po)]

        # fold毎にEPO投影行列をキャッシュ（同じSGパラメータ内で再利用）
        epo_cache = {}  # fold_idx -> P matrix

        for pipe_name, n_epo, n_pls, y_transform in pipelines:
            combo_count += 1
            fold_rmses = []

            for fold_idx, (train_idx, test_idx) in enumerate(folds):
                X_train_sg = X_sg_all[train_idx]
                X_test_sg = X_sg_all[test_idx]

                # 目的変数変換
                if y_transform == "sqrt":
                    y_train = np.sqrt(y[train_idx])
                else:
                    y_train = y[train_idx]

                # EPO適用（キャッシュ利用）
                if n_epo is not None:
                    cache_key = (fold_idx, n_epo)
                    if cache_key not in epo_cache:
                        P = compute_epo_projection(X_train_sg, groups[train_idx], n_components=n_epo)
                        epo_cache[cache_key] = P
                    else:
                        P = epo_cache[cache_key]
                    X_train_final = apply_epo(X_train_sg, P)
                    X_test_final = apply_epo(X_test_sg, P)
                else:
                    X_train_final = X_train_sg
                    X_test_final = X_test_sg

                # PLS学習・予測
                pls = PLSRegression(n_components=n_pls)
                pls.fit(X_train_final, y_train)
                pred = pls.predict(X_test_final).ravel()

                # 逆変換
                if y_transform == "sqrt":
                    pred = pred ** 2

                fold_rmses.append(rmse(y[test_idx], pred))

            mean_rmse = float(np.mean(fold_rmses))
            std_rmse = float(np.std(fold_rmses))

            results.append({
                "deriv": deriv,
                "window_length": wl,
                "polyorder": po,
                "pipeline": pipe_name,
                "n_epo": n_epo if n_epo else 0,
                "n_pls": n_pls,
                "y_transform": y_transform,
                "mean_rmse": mean_rmse,
                "std_rmse": std_rmse,
            })

            if combo_count % 10 == 0 or combo_count == total_combos:
                elapsed = time.time() - t0
                print(f"  [{combo_count}/{total_combos}] SG(d={deriv},w={wl},p={po}) + {pipe_name}: "
                      f"RMSE={mean_rmse:.4f} ± {std_rmse:.4f} ({elapsed:.1f}s)")

    # --- 結果集計 ---
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("mean_rmse").reset_index(drop=True)

    # CSV保存
    csv_path = OUT_DIR / "issue68_sg_params_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # --- Top 20 表示 ---
    print("\n" + "=" * 80)
    print("TOP 20 RESULTS (sorted by mean RMSE)")
    print("=" * 80)
    print(f"{'Rank':<5} {'Deriv':<6} {'WinLen':<7} {'PolyOrd':<8} {'Pipeline':<22} {'RMSE':<10} {'Std':<10}")
    print("-" * 80)

    for i, row in df_results.head(20).iterrows():
        print(f"{i+1:<5} {int(row['deriv']):<6} {int(row['window_length']):<7} "
              f"{int(row['polyorder']):<8} {row['pipeline']:<22} "
              f"{row['mean_rmse']:<10.4f} {row['std_rmse']:<10.4f}")

    # --- パイプライン別ベスト ---
    print("\n" + "=" * 80)
    print("BEST PER PIPELINE")
    print("=" * 80)
    for pipe_name in [p[0] for p in pipelines]:
        subset = df_results[df_results["pipeline"] == pipe_name]
        best = subset.iloc[0]
        print(f"  {pipe_name:<22}: RMSE={best['mean_rmse']:.4f} "
              f"(d={int(best['deriv'])}, w={int(best['window_length'])}, p={int(best['polyorder'])})")

    # --- 微分次数別ベスト ---
    print("\n" + "=" * 80)
    print("BEST PER DERIVATIVE ORDER")
    print("=" * 80)
    for deriv in [1, 2]:
        subset = df_results[df_results["deriv"] == deriv]
        best = subset.iloc[0]
        print(f"  deriv={deriv}: RMSE={best['mean_rmse']:.4f} "
              f"(w={int(best['window_length'])}, p={int(best['polyorder'])}, {best['pipeline']})")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Baseline RMSE: 17.03")
    print(f"Best RMSE: {df_results.iloc[0]['mean_rmse']:.4f}")
    improvement = 17.03 - df_results.iloc[0]["mean_rmse"]
    print(f"Improvement: {improvement:+.4f}")


if __name__ == "__main__":
    main()
