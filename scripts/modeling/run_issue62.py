"""Issue #62: 前処理×線形モデル統合グリッド評価 実行スクリプト

前処理10種 × モデル5種 × 変換2種 = 100パターンをLOSO-CVで評価。
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.eda.data_loader import load_train, get_spectral_columns
from src.modeling.issue62_grid_evaluation import run_grid_evaluation


def main():
    start = time.time()
    data_dir = Path("Input_data")
    train_df = load_train(data_dir)
    spectral_cols = get_spectral_columns(train_df)

    X_raw = train_df[spectral_cols].values
    y = train_df["含水率"].values
    groups = train_df["樹種"].values

    print("=" * 70)
    print("Issue #62: 前処理×線形モデル統合グリッド評価")
    print(f"データ: {X_raw.shape[0]}サンプル × {X_raw.shape[1]}特徴量")
    print(f"樹種数: {len(np.unique(groups))}")
    print("=" * 70)

    results_df = run_grid_evaluation(X_raw, y, groups, spectral_cols)

    # 有効な結果のみ（エラーを除く）
    valid = results_df[results_df["rmse"] < 900].copy()
    valid = valid.sort_values("rmse").reset_index(drop=True)

    # サマリー表示
    print("\n" + "=" * 70)
    print("SUMMARY: Top 20 combinations (sorted by RMSE)")
    print("=" * 70)
    top20 = valid.head(20)[["preprocessing", "model", "transform", "rmse", "rmse_std"]]
    print(top20.to_string(index=False))

    # 前処理別ベストモデル
    print("\n" + "=" * 70)
    print("BEST MODEL per preprocessing")
    print("=" * 70)
    for pp in valid["preprocessing"].unique():
        pp_best = valid[valid["preprocessing"] == pp].iloc[0]
        print(f"  {pp}: {pp_best['model']}+{pp_best['transform']} → RMSE={pp_best['rmse']:.2f} ± {pp_best['rmse_std']:.2f}")

    # モデル別ベスト前処理
    print("\n" + "=" * 70)
    print("BEST PREPROCESSING per model")
    print("=" * 70)
    for mdl in valid["model"].unique():
        mdl_df = valid[valid["model"] == mdl]
        mdl_best = mdl_df.iloc[0]
        print(f"  {mdl}: {mdl_best['preprocessing']}+{mdl_best['transform']} → RMSE={mdl_best['rmse']:.2f} ± {mdl_best['rmse_std']:.2f}")

    # 交互作用ヒートマップ用データ（raw変換のみ）
    print("\n" + "=" * 70)
    print("INTERACTION: preprocessing × model (raw transform)")
    print("=" * 70)
    raw_only = valid[valid["transform"] == "raw"]
    if len(raw_only) > 0:
        pivot = raw_only.pivot_table(index="preprocessing", columns="model", values="rmse")
        print(pivot.to_string())

    print("\n" + "=" * 70)
    print("INTERACTION: preprocessing × model (sqrt transform)")
    print("=" * 70)
    sqrt_only = valid[valid["transform"] == "sqrt"]
    if len(sqrt_only) > 0:
        pivot = sqrt_only.pivot_table(index="preprocessing", columns="model", values="rmse")
        print(pivot.to_string())

    # CSV保存
    out_dir = Path("outputs/modeling")
    out_dir.mkdir(parents=True, exist_ok=True)
    valid.to_csv(out_dir / "issue62_grid_results.csv", index=False)
    print(f"\nSaved to {out_dir / 'issue62_grid_results.csv'}")

    elapsed = time.time() - start
    print(f"\n総実行時間: {elapsed:.1f}秒")


if __name__ == "__main__":
    main()
