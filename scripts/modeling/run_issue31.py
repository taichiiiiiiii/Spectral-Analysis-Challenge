"""Issue #31: 提出ファイル生成"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eda.data_loader import load_train, load_test, get_spectral_columns
from src.modeling.issue31_submission_pipeline import generate_submission
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

train_df = load_train(DATA_DIR)
test_df = load_test(DATA_DIR)
spectral_cols = get_spectral_columns(train_df)

print(f"Train: {train_df.shape}, Test: {test_df.shape}")
print(f"Spectral columns: {len(spectral_cols)}")

output_path = OUT_DIR / "submission.csv"
generate_submission(train_df, test_df, spectral_cols, output_path)

# Verify
sub = pd.read_csv(output_path, header=None)
print(f"\nSubmission shape: {sub.shape}")
print(f"Prediction range: {sub[1].min():.2f} - {sub[1].max():.2f}")
print(f"Prediction mean: {sub[1].mean():.2f}")
print(f"\nFirst 5 rows:")
print(sub.head())
print(f"\nSaved: {output_path}")
