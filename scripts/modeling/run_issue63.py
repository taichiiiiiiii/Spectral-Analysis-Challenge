"""Issue #63: 非線形モデル（LightGBM, XGBoost, SVR）Optuna最適化実験

前処理4種 × モデル3種 × PLS成分数4種 × 目的変数変換2種 = 96パターン
各パターンでOptuna 20試行 → LOSO-CV評価
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eda.data_loader import load_train, load_test, get_spectral_columns, get_wavenumbers
from src.modeling.issue63_nonlinear_optuna import run_full_experiment

import numpy as np
import pandas as pd


def main():
    print("=" * 70)
    print("Issue #63: 非線形モデル Optuna最適化 LOSO-CV実験")
    print("=" * 70)

    # データ読み込み
    DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
    df_train = load_train(DATA_DIR)
    spectral_cols = get_spectral_columns(df_train)
    X_raw = df_train[spectral_cols].values
    y = df_train["含水率"].values
    groups = df_train["樹種"].values

    print(f"Train shape: {X_raw.shape}")
    print(f"Target range: {y.min():.1f} - {y.max():.1f}")
    print(f"Species: {np.unique(groups)}")
    print(f"Spectral columns: {len(spectral_cols)}")

    # 樹種ごとのサンプル数
    unique_sp, sp_counts = np.unique(groups, return_counts=True)
    print("\n樹種別サンプル数:")
    for sp, cnt in sorted(zip(unique_sp, sp_counts), key=lambda x: -x[1]):
        print(f"  {sp}: {cnt}")

    # 実験パラメータ（実行時間を抑えるためPLS成分数を絞る）
    preprocessings = ["SNV", "EPO(1)", "SNV+AsLS(1e6)", "PiecewiseMSC(seg=3)"]
    model_types = ["lgb", "xgb", "svr"]
    pls_components_list = [3, 5, 10]
    n_trials = 15

    output_dir = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n実験設定:")
    print(f"  前処理: {preprocessings}")
    print(f"  モデル: {model_types}")
    print(f"  PLS成分数: {pls_components_list}")
    print(f"  Optuna試行回数: {n_trials}")
    print(f"  合計パターン: {len(preprocessings) * len(model_types) * len(pls_components_list) * 2}")

    t_start = time.time()

    # 実験実行
    df_results = run_full_experiment(
        X_raw, y, groups, spectral_cols,
        preprocessings=preprocessings,
        model_types=model_types,
        pls_components_list=pls_components_list,
        n_trials=n_trials,
        output_dir=output_dir,
    )

    total_time = time.time() - t_start

    # 結果をCSV保存
    csv_path = output_dir / "issue63_nonlinear_optuna_results.csv"
    df_save = df_results.drop(columns=["fold_rmses", "fold_species"], errors="ignore")
    df_save.to_csv(csv_path, index=False)
    print(f"\n結果CSV保存: {csv_path}")

    # ----------- サマリー表示 -----------
    print("\n" + "=" * 70)
    print("全結果サマリー（RMSE昇順）")
    print("=" * 70)

    df_sorted = df_results.sort_values("rmse").reset_index(drop=True)
    for i, row in df_sorted.iterrows():
        print(f"  {i+1:3d}. RMSE={row['rmse']:6.2f} ± {row['rmse_std']:5.2f} | "
              f"{row['preprocessing']:20s} | {row['model']:4s} | PLS({row['n_pls']:2d}) | {row['transform']}")

    # Top 10
    print("\n" + "=" * 70)
    print("TOP 10")
    print("=" * 70)
    for i, row in df_sorted.head(10).iterrows():
        print(f"  {i+1:3d}. RMSE={row['rmse']:6.2f} ± {row['rmse_std']:5.2f} | "
              f"{row['preprocessing']:20s} | {row['model']:4s} | PLS({row['n_pls']:2d}) | {row['transform']}")
        if isinstance(row.get("fold_rmses"), list) and row["fold_rmses"]:
            species = row.get("fold_species", [])
            rmses = row["fold_rmses"]
            fold_str = ", ".join(f"{s}:{r:.1f}" for s, r in zip(species, rmses))
            print(f"       Folds: {fold_str}")

    # モデル別ベスト
    print("\n" + "=" * 70)
    print("モデル別ベスト")
    print("=" * 70)
    for mt in ["LGB", "XGB", "SVR"]:
        df_mt = df_sorted[df_sorted["model"] == mt]
        if len(df_mt) > 0:
            best = df_mt.iloc[0]
            print(f"  {mt}: RMSE={best['rmse']:.2f} ± {best['rmse_std']:.2f} | "
                  f"{best['preprocessing']} | PLS({best['n_pls']}) | {best['transform']}")
            if best.get("best_params"):
                print(f"       Params: {best['best_params']}")

    # 前処理別ベスト
    print("\n" + "=" * 70)
    print("前処理別ベスト")
    print("=" * 70)
    for pp in preprocessings:
        df_pp = df_sorted[df_sorted["preprocessing"] == pp]
        if len(df_pp) > 0:
            best = df_pp.iloc[0]
            print(f"  {pp:20s}: RMSE={best['rmse']:.2f} ± {best['rmse_std']:.2f} | "
                  f"{best['model']} | PLS({best['n_pls']}) | {best['transform']}")

    # ベストスコア vs ベースライン比較
    print("\n" + "=" * 70)
    best_overall = df_sorted.iloc[0]
    print(f"ベストRMSE: {best_overall['rmse']:.2f} ± {best_overall['rmse_std']:.2f}")
    print(f"  設定: {best_overall['preprocessing']} | {best_overall['model']} | PLS({best_overall['n_pls']}) | {best_overall['transform']}")
    print(f"  現在のベストスコア（CLAUDE.md）: RMSE = 17.03")
    improvement = 17.03 - best_overall['rmse']
    print(f"  改善幅: {improvement:+.2f}" if improvement > 0 else f"  差分: {improvement:+.2f}")

    print(f"\n合計実行時間: {total_time:.0f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    main()
