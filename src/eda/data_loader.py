"""データ読み込みモジュール"""
import numpy as np
import pandas as pd
from pathlib import Path


def load_train(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "train.csv", encoding="shift-jis")


def load_test(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "test.csv", encoding="shift-jis")


def get_spectral_columns(df: pd.DataFrame) -> list[str]:
    non_spectral = {"sample number", "species number", "樹種", "含水率"}
    return [c for c in df.columns if c not in non_spectral]


def get_wavenumbers(spectral_cols: list[str]) -> np.ndarray:
    wn = np.array([float(c) for c in spectral_cols])
    return np.sort(wn)
