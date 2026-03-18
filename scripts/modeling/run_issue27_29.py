"""Issue #27 & #29 実行スクリプト"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.eda.data_loader import load_train, get_spectral_columns

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = load_train(DATA_DIR)
spectral_cols = get_spectral_columns(df)

# ── Issue #27: 非線形モデル ──
print("=== Issue #27: Non-linear Models ===")
from src.modeling.issue27_nonlinear_models import evaluate_nonlinear_models
result_27 = evaluate_nonlinear_models(df, spectral_cols)
print("\nResults:")
print(result_27.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))
labels = [f"{r['model']}(pca={r['n_pca']})" for _, r in result_27.iterrows()]
ax.barh(range(len(result_27)), result_27["rmse"], color="steelblue")
ax.set_yticks(range(len(result_27)))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("RMSE (fold average)")
ax.set_title("Issue #27: EPO + Non-linear Models")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(OUT_DIR / "issue27_nonlinear.png", dpi=150)
plt.close()

# ── Issue #29: 前処理+EPO組み合わせ ──
print("\n=== Issue #29: Preprocessing + EPO Combinations ===")
from src.modeling.issue29_epo_preprocessing_combo import evaluate_epo_preprocessing_combos
result_29 = evaluate_epo_preprocessing_combos(df, spectral_cols)
print("\nResults:")
print(result_29.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(len(result_29)), result_29["rmse"], color="forestgreen")
ax.set_yticks(range(len(result_29)))
ax.set_yticklabels(result_29["method"], fontsize=9)
ax.set_xlabel("RMSE (fold average)")
ax.set_title("Issue #29: EPO + Preprocessing Combinations")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(OUT_DIR / "issue29_epo_combos.png", dpi=150)
plt.close()

print(f"\nSaved: issue27_nonlinear.png, issue29_epo_combos.png")
print("=== ALL DONE ===")
