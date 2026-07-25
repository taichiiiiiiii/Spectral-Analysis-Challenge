"""Issue #66: 目的変数変換の包括的比較実験

9種の目的変数変換 × 4種の前処理 × 4種のモデルをLOSO-CVで評価。
log1pはPLS(1-3)のみで評価。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.eda.data_loader import load_train, get_spectral_columns, get_wavenumbers
from src.modeling.issue66_target_transforms import run_target_transform_evaluation

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# データ読み込み
df_train = load_train(DATA_DIR)
spectral_cols = get_spectral_columns(df_train)
X_raw = df_train[spectral_cols].values
y = df_train["含水率"].values
groups = df_train["樹種"].values
wavenumbers = get_wavenumbers(spectral_cols)

print(f"データ: {X_raw.shape[0]} samples, {X_raw.shape[1]} features", flush=True)
print(f"樹種: {sorted(set(groups))}", flush=True)
print(f"含水率: min={y.min():.1f}, max={y.max():.1f}, mean={y.mean():.1f}", flush=True)

# 実験実行
t_start = time.time()
df_results = run_target_transform_evaluation(X_raw, y, groups, spectral_cols)
elapsed = time.time() - t_start

# CSV保存
csv_path = OUTPUT_DIR / "issue66_target_transforms.csv"
df_save = df_results.drop(columns=["fold_rmses", "fold_species"], errors="ignore")
df_save = df_save.sort_values("rmse").reset_index(drop=True)
df_save.to_csv(csv_path, index=False)
print(f"\nResults saved to {csv_path}", flush=True)

# Top 20 表示
print(f"\n{'='*80}")
print(f"Top 20 combinations (total: {len(df_results)}, elapsed: {elapsed:.0f}s)")
print(f"{'='*80}")
top20 = df_save.head(20)
print(
    top20.to_string(
        index=False,
        columns=["preprocessing", "model", "transform", "rmse", "rmse_std"],
        float_format="%.2f",
    )
)

# 変換ごとの最良スコア
print(f"\n{'='*80}")
print("Best RMSE per transform")
print(f"{'='*80}")
best_per_tf = df_save.groupby("transform").first().reset_index()
best_per_tf = best_per_tf.sort_values("rmse")
print(
    best_per_tf.to_string(
        index=False,
        columns=["transform", "preprocessing", "model", "rmse", "rmse_std"],
        float_format="%.2f",
    )
)

# 前処理ごとの最良スコア
print(f"\n{'='*80}")
print("Best RMSE per preprocessing")
print(f"{'='*80}")
best_per_pp = df_save.groupby("preprocessing").first().reset_index()
best_per_pp = best_per_pp.sort_values("rmse")
print(
    best_per_pp.to_string(
        index=False,
        columns=["preprocessing", "transform", "model", "rmse", "rmse_std"],
        float_format="%.2f",
    )
)

# モデルごとの最良スコア
print(f"\n{'='*80}")
print("Best RMSE per model")
print(f"{'='*80}")
best_per_mdl = df_save.groupby("model").first().reset_index()
best_per_mdl = best_per_mdl.sort_values("rmse")
print(
    best_per_mdl.to_string(
        index=False,
        columns=["model", "transform", "preprocessing", "rmse", "rmse_std"],
        float_format="%.2f",
    )
)

print(f"\nDone! Total elapsed: {elapsed:.0f}s", flush=True)
