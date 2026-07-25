"""近赤外吸収帯照合モジュール

対応Issue: #14 近赤外吸収帯の分光学的照合
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/14
"""
import numpy as np
import pandas as pd


THEORETICAL_BANDS = {
    "water_combination": 5155,
    "water_first_overtone": 6900,
    "oh_stretch_bend": 5200,
    "ch_stretch": 8300,
    "ch_combination": 4300,
}


def compute_correlation_map(df: pd.DataFrame, spectral_cols: list[str]) -> pd.Series:
    corr = df[spectral_cols].corrwith(df["含水率"])
    corr.index = pd.Index([float(c) for c in spectral_cols], dtype=float)
    return corr


def identify_theoretical_bands() -> dict:
    return THEORETICAL_BANDS.copy()


def compare_theory_vs_data(
    corr: pd.Series, wavenumbers: np.ndarray, window: float = 100.0
) -> pd.DataFrame:
    rows = []
    for band_name, theory_wn in THEORETICAL_BANDS.items():
        nearest_wn = wavenumbers[np.argmin(np.abs(wavenumbers - theory_wn))]
        corr_at_theory = float(corr.iloc[np.argmin(np.abs(corr.index - theory_wn))])

        mask = (corr.index >= theory_wn - window) & (corr.index <= theory_wn + window)
        local_corr = corr[mask]
        if len(local_corr) > 0:
            peak_idx = local_corr.abs().idxmax()
            peak_corr = float(local_corr[peak_idx])
            peak_wn = float(peak_idx)
        else:
            peak_corr = corr_at_theory
            peak_wn = float(nearest_wn)

        rows.append({
            "band_name": band_name,
            "theoretical_wn": theory_wn,
            "nearest_actual_wn": float(nearest_wn),
            "correlation_at_theory": corr_at_theory,
            "peak_correlation": peak_corr,
            "peak_wavenumber": peak_wn,
        })
    return pd.DataFrame(rows)
