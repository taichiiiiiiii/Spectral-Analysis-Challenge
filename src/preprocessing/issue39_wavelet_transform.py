"""ウェーブレット変換による前処理と特徴量抽出

DWT (Discrete Wavelet Transform) を用いて:
1. スペクトルのデノイズ（高周波ノイズ除去）
2. マルチレゾリューション特徴量（各レベルのエネルギー等）を抽出する。
"""
import numpy as np
import pandas as pd
import pywt


def wavelet_denoise(
    X: np.ndarray,
    wavelet: str = "db4",
    level: int = 3,
    threshold_mode: str = "soft",
) -> np.ndarray:
    """ウェーブレットデノイズを適用する。

    Parameters
    ----------
    X : (n_samples, n_features)
    wavelet : ウェーブレット名
    level : 分解レベル
    threshold_mode : "soft" or "hard"

    Returns
    -------
    (n_samples, n_features) デノイズされたスペクトル
    """
    X_denoised = np.zeros_like(X)
    for i in range(len(X)):
        coeffs = pywt.wavedec(X[i], wavelet, level=level)
        # ユニバーサル閾値 (VisuShrink)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(X[i])))
        # 近似係数(coeffs[0])は閾値処理しない
        new_coeffs = [coeffs[0]]
        for c in coeffs[1:]:
            new_coeffs.append(pywt.threshold(c, threshold, mode=threshold_mode))
        X_denoised[i] = pywt.waverec(new_coeffs, wavelet)[:X.shape[1]]
    return X_denoised


def extract_wavelet_features(
    X: np.ndarray,
    wavelet: str = "db4",
    level: int = 3,
) -> pd.DataFrame:
    """ウェーブレット係数からマルチレゾリューション特徴量を抽出する。

    各レベルのエネルギー（係数の二乗和）と統計量を返す。

    Parameters
    ----------
    X : (n_samples, n_features)
    wavelet : ウェーブレット名
    level : 分解レベル

    Returns
    -------
    pd.DataFrame of shape (n_samples, n_features_out)
    """
    all_features = []
    for i in range(len(X)):
        coeffs = pywt.wavedec(X[i], wavelet, level=level)
        row = {}
        # 近似係数のエネルギー
        row["energy_approx"] = np.sum(coeffs[0] ** 2)
        # 各詳細レベルのエネルギー
        for lv, c in enumerate(coeffs[1:], 1):
            row[f"energy_detail_{lv}"] = np.sum(c ** 2)
            row[f"std_detail_{lv}"] = np.std(c)
            row[f"max_abs_detail_{lv}"] = np.max(np.abs(c))
        # エネルギー比
        total_energy = sum(np.sum(c ** 2) for c in coeffs)
        if total_energy > 1e-12:
            row["energy_ratio_approx"] = np.sum(coeffs[0] ** 2) / total_energy
            for lv, c in enumerate(coeffs[1:], 1):
                row[f"energy_ratio_detail_{lv}"] = np.sum(c ** 2) / total_energy
        else:
            row["energy_ratio_approx"] = 0.0
            for lv in range(1, level + 1):
                row[f"energy_ratio_detail_{lv}"] = 0.0
        all_features.append(row)
    return pd.DataFrame(all_features)
