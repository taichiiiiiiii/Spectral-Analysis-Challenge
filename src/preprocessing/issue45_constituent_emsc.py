"""Constituent-Informed EMSC (Extended Multiplicative Signal Correction)

成分スペクトル（セルロース/リグニン/水など）を明示的にモデル化し、
樹種依存の化学変動と散乱を除去する。

参考: Solheim et al., Applied Spectroscopy 2022
"""
import numpy as np


def _generate_approximate_constituent_spectra(wavenumbers: np.ndarray) -> np.ndarray:
    """NIR領域の近似成分スペクトルを生成する。

    4300-10000 cm⁻¹における水、セルロース、リグニンの
    主要吸収帯をガウシアンで近似。
    """
    wn = wavenumbers
    constituents = []

    # 水: ~5200 cm⁻¹ (OH bend+stretch combo), ~6900 cm⁻¹ (OH 1st overtone)
    water = (np.exp(-0.5 * ((wn - 5200) / 100) ** 2) +
             0.8 * np.exp(-0.5 * ((wn - 6900) / 120) ** 2))
    constituents.append(water / (water.max() + 1e-10))

    # セルロース: ~4400 cm⁻¹ (CH combo), ~5600 cm⁻¹, ~6700 cm⁻¹
    cellulose = (np.exp(-0.5 * ((wn - 4400) / 80) ** 2) +
                 0.6 * np.exp(-0.5 * ((wn - 5600) / 100) ** 2) +
                 0.4 * np.exp(-0.5 * ((wn - 6700) / 90) ** 2))
    constituents.append(cellulose / (cellulose.max() + 1e-10))

    # リグニン: ~4700 cm⁻¹ (aromatic CH), ~5900 cm⁻¹
    lignin = (np.exp(-0.5 * ((wn - 4700) / 90) ** 2) +
              0.7 * np.exp(-0.5 * ((wn - 5900) / 110) ** 2))
    constituents.append(lignin / (lignin.max() + 1e-10))

    return np.array(constituents)


def constituent_emsc(
    spectra: np.ndarray,
    wavenumbers: np.ndarray | None = None,
    constituent_spectra: np.ndarray | None = None,
    poly_order: int = 2,
    reference: np.ndarray | None = None,
) -> np.ndarray:
    """Constituent-informed EMSCを適用する。

    Parameters
    ----------
    spectra : (n_samples, n_features)
    wavenumbers : (n_features,) 波数軸。Noneの場合はインデックスを使用
    constituent_spectra : (n_constituents, n_features) 成分スペクトル。
        Noneの場合は近似スペクトルを自動生成
    poly_order : ベースライン多項式の次数
    reference : (n_features,) 参照スペクトル。Noneの場合は平均スペクトル

    Returns
    -------
    (n_samples, n_features) 補正後スペクトル
    """
    n, p = spectra.shape

    if reference is None:
        reference = spectra.mean(axis=0)

    if wavenumbers is None:
        wavenumbers = np.arange(p, dtype=float)

    if constituent_spectra is None:
        constituent_spectra = _generate_approximate_constituent_spectra(wavenumbers)

    # 正規化波数軸
    wn_norm = (wavenumbers - wavenumbers.min()) / (wavenumbers.max() - wavenumbers.min() + 1e-10)

    # モデル行列: [intercept, reference, constituents, polynomials]
    cols = [np.ones(p), reference]
    for cs in constituent_spectra:
        cols.append(cs)
    for order in range(1, poly_order + 1):
        cols.append(wn_norm ** order)
    M = np.column_stack(cols)  # (p, n_terms)

    corrected = np.zeros_like(spectra)
    n_const = len(constituent_spectra)

    for i in range(n):
        coefs, _, _, _ = np.linalg.lstsq(M, spectra[i], rcond=None)
        # 乗算係数 (reference)
        b = coefs[1]
        if abs(b) < 1e-10:
            b = 1.0
        # 除去: intercept + constituents + polynomials
        interferent = coefs[0] * np.ones(p)
        for j in range(n_const):
            interferent += coefs[2 + j] * constituent_spectra[j]
        for order in range(1, poly_order + 1):
            interferent += coefs[2 + n_const + order - 1] * (wn_norm ** order)
        corrected[i] = (spectra[i] - interferent) / b

    return corrected
