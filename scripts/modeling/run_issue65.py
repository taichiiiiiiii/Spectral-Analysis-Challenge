"""Issue #65: 高度な特徴量選択手法 (VIP, CARS, iPLS, siPLS) 実験スクリプト

特徴量選択手法 × 前処理 × モデル × 目的変数変換のグリッド評価をLOSO-CVで実行。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.eda.data_loader import load_train, get_spectral_columns, get_wavenumbers
from src.modeling.issue65_feature_selection import run_feature_selection_experiment

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"


def main():
    start = time.time()

    # データ読み込み
    df_train = load_train(DATA_DIR)
    spectral_cols = get_spectral_columns(df_train)
    X_raw = df_train[spectral_cols].values
    y = df_train["含水率"].values
    groups = df_train["樹種"].values
    wavenumbers = get_wavenumbers(spectral_cols)

    print("=" * 80)
    print("Issue #65: 高度な特徴量選択手法 (VIP, CARS, iPLS, siPLS) 実験")
    print("=" * 80)
    print(f"データ: {X_raw.shape[0]}サンプル × {X_raw.shape[1]}特徴量")
    print(f"樹種数: {len(np.unique(groups))} ({np.unique(groups)})")
    print(f"波数範囲: {wavenumbers.min():.0f} - {wavenumbers.max():.0f} cm-1")
    print(f"目的変数: mean={y.mean():.1f}, std={y.std():.1f}, "
          f"min={y.min():.1f}, max={y.max():.1f}")
    print("=" * 80)
    print()

    # 実験実行
    results_df = run_feature_selection_experiment(X_raw, y, groups)

    # 有効な結果のみ
    valid = results_df[results_df["rmse"] < 900].copy()
    valid = valid.sort_values("rmse").reset_index(drop=True)

    # ========================================
    # Top 20 表示
    # ========================================
    print("\n" + "=" * 80)
    print("TOP 20 COMBINATIONS (sorted by RMSE)")
    print("=" * 80)
    top20 = valid.head(20)[[
        "preprocessing", "feature_selection", "model", "transform",
        "rmse", "rmse_std", "n_features_avg"
    ]]
    print(top20.to_string(index=True))

    # ========================================
    # 特徴量選択手法別ベスト
    # ========================================
    print("\n" + "=" * 80)
    print("BEST per FEATURE SELECTION METHOD")
    print("=" * 80)
    for fs in valid["feature_selection"].unique():
        fs_df = valid[valid["feature_selection"] == fs]
        if len(fs_df) == 0:
            continue
        best = fs_df.iloc[0]
        print(f"  {fs:30s} → RMSE={best['rmse']:.2f} ± {best['rmse_std']:.2f} "
              f"({best['preprocessing']}, {best['model']}, {best['transform']}, "
              f"n_feat={best['n_features_avg']:.0f})")

    # ========================================
    # 前処理別ベスト
    # ========================================
    print("\n" + "=" * 80)
    print("BEST per PREPROCESSING")
    print("=" * 80)
    for pp in valid["preprocessing"].unique():
        pp_df = valid[valid["preprocessing"] == pp]
        if len(pp_df) == 0:
            continue
        best = pp_df.iloc[0]
        print(f"  {pp:20s} → RMSE={best['rmse']:.2f} ± {best['rmse_std']:.2f} "
              f"({best['feature_selection']}, {best['model']}, {best['transform']})")

    # ========================================
    # モデル別ベスト
    # ========================================
    print("\n" + "=" * 80)
    print("BEST per MODEL")
    print("=" * 80)
    for mdl in valid["model"].unique():
        mdl_df = valid[valid["model"] == mdl]
        if len(mdl_df) == 0:
            continue
        best = mdl_df.iloc[0]
        print(f"  {mdl:20s} → RMSE={best['rmse']:.2f} ± {best['rmse_std']:.2f} "
              f"({best['preprocessing']}, {best['feature_selection']}, {best['transform']})")

    # ========================================
    # ベースライン(None)との比較
    # ========================================
    print("\n" + "=" * 80)
    print("FEATURE SELECTION IMPROVEMENT vs BASELINE (None)")
    print("=" * 80)
    baseline = valid[valid["feature_selection"] == "None"]
    if len(baseline) > 0:
        base_best = baseline.iloc[0]["rmse"]
        print(f"  Baseline best RMSE: {base_best:.2f}")
        for fs in valid["feature_selection"].unique():
            if fs == "None":
                continue
            fs_best = valid[valid["feature_selection"] == fs].iloc[0]["rmse"]
            diff = fs_best - base_best
            symbol = "↓" if diff < 0 else "↑"
            print(f"  {fs:30s} → RMSE={fs_best:.2f} ({symbol}{abs(diff):.2f})")

    # ========================================
    # CSV保存
    # ========================================
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "issue65_feature_selection_results.csv"
    valid.to_csv(csv_path, index=False)
    print(f"\n結果をCSVに保存: {csv_path}")

    elapsed = time.time() - start
    print(f"\n総実行時間: {elapsed:.1f}秒 ({elapsed / 60:.1f}分)")


if __name__ == "__main__":
    main()
