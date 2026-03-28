"""Issue #103: 既存ベストとドメイン適応手法のクロスアンサンブル

既存ベスト(RMSE=13.70)の予測とドメイン適応手法の予測をブレンドし、
多様性による改善を試みる。
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs"

# 利用可能な提出ファイルをロード
files = {
    "best": OUT_DIR / "submission_best11_rmse13.70.csv",
    "advval": OUT_DIR / "submission_v8_advval.csv",
    "pseudo": OUT_DIR / "submission_v8_pseudo.csv",
    "tta": OUT_DIR / "submission_v8_tta_ensemble.csv",
}

preds = {}
for name, path in files.items():
    if path.exists():
        df = pd.read_csv(path, header=None, names=["id", "pred"])
        preds[name] = df["pred"].values
        print(f"Loaded {name}: {path.name}, mean={df['pred'].mean():.2f}, std={df['pred'].std():.2f}")
    else:
        print(f"MISSING: {name}: {path}")

ids = pd.read_csv(files["best"], header=None, names=["id", "pred"])["id"].values
best = preds["best"]

# 予測値の差異を分析
print("\n=== 予測の差異分析 ===")
for name, pred in preds.items():
    if name == "best":
        continue
    diff = pred - best
    corr = np.corrcoef(best, pred)[0, 1]
    print(f"{name}: mean_diff={diff.mean():.2f}, std_diff={diff.std():.2f}, corr={corr:.4f}")

print("\n=== ブレンド提出ファイル生成 ===")

def save_submission(ids, preds_arr, filename):
    df = pd.DataFrame({"id": ids, "pred": preds_arr})
    path = OUT_DIR / filename
    df.to_csv(path, index=False, header=False)
    print(f"  Saved: {filename} (mean={preds_arr.mean():.2f})")
    return path

# 1. Best + AdvVal ブレンド
if "advval" in preds:
    print("\n--- AdvVal ブレンド ---")
    for alpha in [0.05, 0.10, 0.15, 0.20, 0.30]:
        blend = (1 - alpha) * best + alpha * preds["advval"]
        save_submission(ids, blend, f"submission_v11_blend_advval_a{alpha:.2f}.csv")

# 2. Best + Pseudo ブレンド
if "pseudo" in preds:
    print("\n--- Pseudo ブレンド ---")
    for alpha in [0.05, 0.10, 0.15]:
        blend = (1 - alpha) * best + alpha * preds["pseudo"]
        save_submission(ids, blend, f"submission_v11_blend_pseudo_a{alpha:.2f}.csv")

# 3. Best + AdvVal + TTA (均等)
if "advval" in preds and "tta" in preds:
    print("\n--- Best + AdvVal + TTA コンボ ---")
    for alpha in [0.10, 0.15, 0.20]:
        da_avg = (preds["advval"] + preds["tta"]) / 2
        blend = (1 - alpha) * best + alpha * da_avg
        save_submission(ids, blend, f"submission_v11_blend_best_combo_a{alpha:.2f}.csv")

# 4. 全手法マルチブレンド
da_methods = {k: v for k, v in preds.items() if k != "best"}
if len(da_methods) >= 2:
    print("\n--- マルチブレンド (全手法均等) ---")
    da_mean = np.mean(list(da_methods.values()), axis=0)
    for alpha in [0.10, 0.15, 0.20, 0.30]:
        blend = (1 - alpha) * best + alpha * da_mean
        save_submission(ids, blend, f"submission_v11_blend_multi_a{alpha:.2f}.csv")

# 5. Rank-based ブレンド（順位平均）
print("\n--- Rank-based ブレンド ---")
all_preds = list(preds.values())
ranks = np.array([np.argsort(np.argsort(p)) for p in all_preds])
rank_mean = ranks.mean(axis=0)
sorted_best = np.sort(best)
rank_blend = sorted_best[(rank_mean).astype(int).clip(0, len(best)-1)]
save_submission(ids, rank_blend, "submission_v11_rank_blend.csv")

print("\n=== 完了 ===")
print(f"生成したブレンド数: {len(list(OUT_DIR.glob('submission_v11_*.csv')))} files")
