"""Issue #22-#25 前処理の一括実行スクリプト"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.eda.data_loader import load_train, get_spectral_columns

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "preprocessing"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = load_train(DATA_DIR)
spectral_cols = get_spectral_columns(df)
print(f"Data loaded: {df.shape}, spectral cols: {len(spectral_cols)}")

# ── Issue #22: EPO ──
print("\n=== Issue #22: EPO ===")
from src.preprocessing.issue22_epo import evaluate_epo_effect
result_epo = evaluate_epo_effect(df, spectral_cols, n_epo_components_list=[1, 2, 3, 4, 5, 6, 7, 8])
print(result_epo.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(result_epo["method"], result_epo["rmse"], color="steelblue")
ax.errorbar(range(len(result_epo)), result_epo["rmse"], yerr=result_epo["rmse_std"],
            fmt="none", color="black", capsize=3)
ax.set_ylabel("RMSE (fold average)")
ax.set_xlabel("Method")
ax.set_title("Issue #22: EPO Effect on PLS LOSO-CV RMSE")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / "issue22_epo.png", dpi=150)
plt.close()
print(f"Saved: {OUT_DIR / 'issue22_epo.png'}")

# ── Issue #23: OSC ──
print("\n=== Issue #23: OSC ===")
from src.preprocessing.issue23_osc import evaluate_osc_effect
result_osc = evaluate_osc_effect(df, spectral_cols, n_osc_components_list=[1, 2, 3, 4, 5])
print(result_osc.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(result_osc["method"], result_osc["rmse"], color="darkorange")
ax.errorbar(range(len(result_osc)), result_osc["rmse"], yerr=result_osc["rmse_std"],
            fmt="none", color="black", capsize=3)
ax.set_ylabel("RMSE (fold average)")
ax.set_xlabel("Method")
ax.set_title("Issue #23: OSC Effect on PLS LOSO-CV RMSE")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / "issue23_osc.png", dpi=150)
plt.close()
print(f"Saved: {OUT_DIR / 'issue23_osc.png'}")

# ── Issue #24: SNV+DT ──
print("\n=== Issue #24: SNV+De-trending ===")
from src.preprocessing.issue24_detrending import evaluate_detrending_effect
result_dt = evaluate_detrending_effect(df, spectral_cols)
print(result_dt.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(result_dt["method"], result_dt["rmse"], color="forestgreen")
ax.errorbar(range(len(result_dt)), result_dt["rmse"], yerr=result_dt["rmse_std"],
            fmt="none", color="black", capsize=3)
ax.set_ylabel("RMSE (fold average)")
ax.set_xlabel("Method")
ax.set_title("Issue #24: SNV+De-trending Effect on PLS LOSO-CV RMSE")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / "issue24_detrending.png", dpi=150)
plt.close()
print(f"Saved: {OUT_DIR / 'issue24_detrending.png'}")

# ── Issue #25: EMSC ──
print("\n=== Issue #25: EMSC ===")
from src.preprocessing.issue25_emsc import evaluate_emsc_effect
result_emsc = evaluate_emsc_effect(df, spectral_cols)
print(result_emsc.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(result_emsc["method"], result_emsc["rmse"], color="purple")
ax.errorbar(range(len(result_emsc)), result_emsc["rmse"], yerr=result_emsc["rmse_std"],
            fmt="none", color="black", capsize=3)
ax.set_ylabel("RMSE (fold average)")
ax.set_xlabel("Method")
ax.set_title("Issue #25: EMSC Effect on PLS LOSO-CV RMSE")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / "issue25_emsc.png", dpi=150)
plt.close()
print(f"Saved: {OUT_DIR / 'issue25_emsc.png'}")

print("\n=== ALL DONE ===")
