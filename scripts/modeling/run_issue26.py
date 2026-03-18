"""Issue #26: PLS + EPO グリッドサーチ実行"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.eda.data_loader import load_train, get_spectral_columns
from src.modeling.issue26_pls_epo_optimization import grid_search_pls_epo

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = load_train(DATA_DIR)
spectral_cols = get_spectral_columns(df)

print("=== Issue #26: PLS + EPO Grid Search ===")
result = grid_search_pls_epo(df, spectral_cols)

print("\nTop 10 results:")
print(result.head(10).to_string(index=False))

# Heatmap
pivot = result.pivot(index="n_epo", columns="n_pls", values="rmse")
fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r")
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)
ax.set_xlabel("PLS Components")
ax.set_ylabel("EPO Components")
ax.set_title("LOSO-CV RMSE: PLS × EPO Grid Search")
plt.colorbar(im, label="RMSE")

# Annotate
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7,
                color="white" if val > 30 else "black")

plt.tight_layout()
plt.savefig(OUT_DIR / "issue26_pls_epo_heatmap.png", dpi=150)
plt.close()
print(f"\nSaved: {OUT_DIR / 'issue26_pls_epo_heatmap.png'}")

# Best result
best = result.iloc[0]
print(f"\nBest: n_pls={best['n_pls']}, n_epo={best['n_epo']}, RMSE={best['rmse']:.4f} ± {best['rmse_std']:.4f}")
