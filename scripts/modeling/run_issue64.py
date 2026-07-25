"""MLP実験実行スクリプト

対応Issue: #64
scikit-learn MLPRegressorでPLSスコア特徴量のLOSO-CV評価を実行する。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eda.data_loader import load_train, get_spectral_columns
from src.modeling.issue64_mlp_experiment import run_experiment

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("Issue #64: MLP (MLPRegressor) LOSO-CV 実験")
    print("=" * 70)

    # データ読み込み
    print("\nデータ読み込み中...")
    df_train = load_train(DATA_DIR)
    spectral_cols = get_spectral_columns(df_train)
    X_raw = df_train[spectral_cols].values
    y = df_train["含水率"].values
    groups = df_train["樹種"].values

    print(f"  サンプル数: {len(y)}")
    print(f"  特徴量数: {X_raw.shape[1]}")
    print(f"  樹種数: {len(set(groups))}")
    print(f"  含水率: mean={y.mean():.2f}, std={y.std():.2f}, "
          f"min={y.min():.2f}, max={y.max():.2f}")

    # 実験実行
    start = time.time()
    df_results, fold_data = run_experiment(X_raw, y, groups, verbose=True)
    elapsed = time.time() - start

    # 結果表示
    print("\n" + "=" * 70)
    print(f"全 {len(df_results)} パターン完了 (所要時間: {elapsed:.1f}秒)")
    print("=" * 70)

    # Top 20 表示
    print("\n--- Top 20 RMSE ---")
    display_cols = ["preprocess", "n_pls", "hidden_layers", "activation",
                    "alpha", "lr_init", "target_transform", "rmse"]
    print(df_results[display_cols].head(20).to_string(index=True))

    # 軸別集計
    print("\n--- 前処理別の平均RMSE ---")
    print(df_results.groupby("preprocess")["rmse"].agg(["mean", "min", "count"])
          .sort_values("min").to_string())

    print("\n--- PLS成分数別の平均RMSE ---")
    print(df_results.groupby("n_pls")["rmse"].agg(["mean", "min", "count"])
          .sort_values("min").to_string())

    print("\n--- MLP構造別の平均RMSE ---")
    print(df_results.groupby("hidden_layers")["rmse"].agg(["mean", "min", "count"])
          .sort_values("min").to_string())

    print("\n--- 活性化関数別の平均RMSE ---")
    print(df_results.groupby("activation")["rmse"].agg(["mean", "min", "count"])
          .sort_values("min").to_string())

    print("\n--- 正則化(alpha)別の平均RMSE ---")
    print(df_results.groupby("alpha")["rmse"].agg(["mean", "min", "count"])
          .sort_values("min").to_string())

    print("\n--- 目的変数変換別の平均RMSE ---")
    print(df_results.groupby("target_transform")["rmse"].agg(["mean", "min", "count"])
          .sort_values("min").to_string())

    # ベストモデルのfold別結果
    best_idx = df_results.index[0]
    print(f"\n--- ベストモデル (RMSE={df_results.loc[best_idx, 'rmse']:.4f}) のfold別結果 ---")
    # fold_dataのキーは元のインデックスだが、sortでずれている可能性
    # df_resultsはsort済みなので、元indexを使う
    # run_experimentでindex resetしているので、fold_dataの元indexを探す必要がある
    # 簡易的にベスト設定を再特定
    best_row = df_results.iloc[0]
    print(f"  設定: {best_row['preprocess']} | PLS={best_row['n_pls']} | "
          f"MLP{best_row['hidden_layers']} | {best_row['activation']} | "
          f"alpha={best_row['alpha']} | lr={best_row['lr_init']} | "
          f"target={best_row['target_transform']}")

    # CSV保存
    csv_path = OUTPUT_DIR / "issue64_mlp_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n結果をCSV保存: {csv_path}")

    # ベストスコアのサマリ
    print(f"\n{'=' * 70}")
    print(f"ベストRMSE: {df_results['rmse'].min():.4f}")
    print(f"現在のベストスコア(参考): 17.03")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
