"""Issue #28: ドメイン適応の評価実行"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.eda.data_loader import load_train, get_spectral_columns
from src.modeling.issue28_domain_adaptation import evaluate_domain_adaptation

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = load_train(DATA_DIR)
spectral_cols = get_spectral_columns(df)

print("=== Issue #28: Domain Adaptation ===")
result = evaluate_domain_adaptation(df, spectral_cols)
print(result.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(range(len(result)), result["rmse"], color="coral")
ax.errorbar(result["rmse"], range(len(result)), xerr=result["rmse_std"],
            fmt="none", color="black", capsize=3)
ax.set_yticks(range(len(result)))
ax.set_yticklabels(result["method"])
ax.set_xlabel("RMSE (fold average)")
ax.set_title("Issue #28: Domain Adaptation Methods")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(OUT_DIR / "issue28_domain_adaptation.png", dpi=150)
plt.close()
print(f"Saved: {OUT_DIR / 'issue28_domain_adaptation.png'}")
