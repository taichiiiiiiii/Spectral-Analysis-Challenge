"""EDA知見に基づく樹種不変特徴量

対応Issue: #37
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/37

EDAで得た知見:
- Issue #14: 水の吸収帯 ~5200, ~6900 cm⁻¹ が含水率と最も相関
- Issue #9: 針葉樹/広葉樹でベースラインに差 → 比で正規化
- Issue #11: ベースラインドリフト → 面積比で樹種差を吸収
- Issue #4: ドメインシフト → 物理化学的に意味のある指標で樹種依存性を低減
"""
import numpy as np
import pandas as pd


def _wn_idx(wavenumbers: np.ndarray, target_wn: float) -> int:
    """最も近い波数のインデックスを返す"""
    return int(np.argmin(np.abs(wavenumbers - target_wn)))


def _wn_range_idx(wavenumbers: np.ndarray, wn_low: float, wn_high: float) -> slice:
    """波数範囲のスライスを返す"""
    idx_low = _wn_idx(wavenumbers, wn_low)
    idx_high = _wn_idx(wavenumbers, wn_high)
    if idx_low > idx_high:
        idx_low, idx_high = idx_high, idx_low
    return slice(idx_low, idx_high + 1)


def compute_band_ratios(X: np.ndarray, wavenumbers: np.ndarray) -> pd.DataFrame:
    """バンド比特徴量を計算する。

    EDA知見: 水の吸収帯(5155, 6900) / 構造帯(8300, 6000, 7500)の比は
    ベースラインのスケーリング差を相殺し、樹種に依存しにくい。

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    wavenumbers : np.ndarray of shape (n_features,)

    Returns
    -------
    pd.DataFrame of shape (n_samples, n_ratios)
    """
    eps = 1e-10

    # 主要波数での吸光度
    w5155 = X[:, _wn_idx(wavenumbers, 5155)]
    w5200 = X[:, _wn_idx(wavenumbers, 5200)]
    w6900 = X[:, _wn_idx(wavenumbers, 6900)]
    w8300 = X[:, _wn_idx(wavenumbers, 8300)]
    w6000 = X[:, _wn_idx(wavenumbers, 6000)]
    w7500 = X[:, _wn_idx(wavenumbers, 7500)]
    w4600 = X[:, _wn_idx(wavenumbers, 4600)]

    features = {
        # 水帯 / CH伸縮帯 (Issue #14: 水バンドが含水率と最も相関)
        "ratio_w5155_ch8300": w5155 / (w8300 + eps),
        "ratio_w6900_ch8300": w6900 / (w8300 + eps),
        # 水帯間の比 (水の結合状態の指標)
        "ratio_w5155_w6900": w5155 / (w6900 + eps),
        # 水帯 / 参照点 (Issue #11: ベースライン差を吸収)
        "ratio_w5200_ref6000": w5200 / (w6000 + eps),
        "ratio_w5155_ref7500": w5155 / (w7500 + eps),
        "ratio_w6900_ref6000": w6900 / (w6000 + eps),
        # セルロース関連 / CH帯 (Issue #9: 針葉樹/広葉樹の差を考慮)
        "ratio_w4600_ch8300": w4600 / (w8300 + eps),
    }
    return pd.DataFrame(features)


def compute_ndmi(X: np.ndarray, wavenumbers: np.ndarray) -> pd.DataFrame:
    """正規化差分水分指標（NDMI）を計算する。

    (A_water - A_ref) / (A_water + A_ref) の形式。
    値域が[-1, 1]に正規化されるため、モデルの安定性が向上。

    Returns
    -------
    pd.DataFrame of shape (n_samples, n_indices)
    """
    eps = 1e-10
    w5155 = X[:, _wn_idx(wavenumbers, 5155)]
    w6900 = X[:, _wn_idx(wavenumbers, 6900)]
    w8300 = X[:, _wn_idx(wavenumbers, 8300)]
    w7500 = X[:, _wn_idx(wavenumbers, 7500)]
    w6000 = X[:, _wn_idx(wavenumbers, 6000)]

    features = {
        # 水第1倍音 vs CH帯
        "ndmi_6900_8300": (w6900 - w8300) / (w6900 + w8300 + eps),
        # 水コンビネーション vs CH帯
        "ndmi_5155_8300": (w5155 - w8300) / (w5155 + w8300 + eps),
        # 水コンビネーション vs 中立領域
        "ndmi_5155_7500": (w5155 - w7500) / (w5155 + w7500 + eps),
        # 水コンビネーション vs 参照点
        "ndmi_5155_6000": (w5155 - w6000) / (w5155 + w6000 + eps),
        # 水第1倍音 vs 参照点
        "ndmi_6900_6000": (w6900 - w6000) / (w6900 + w6000 + eps),
    }
    return pd.DataFrame(features)


def compute_area_ratios(X: np.ndarray, wavenumbers: np.ndarray) -> pd.DataFrame:
    """領域面積比を計算する。

    Issue #11: ベースラインドリフトが大きい → 面積比で正規化。
    台形積分で各領域の面積を計算し、比を取る。

    Returns
    -------
    pd.DataFrame of shape (n_samples, n_areas)
    """
    eps = 1e-10

    # 水コンビネーションバンド領域
    water_comb_slice = _wn_range_idx(wavenumbers, 5000, 5350)
    water_comb_area = np.trapezoid(X[:, water_comb_slice], wavenumbers[water_comb_slice], axis=1)

    # 水第1倍音領域
    water_1st_slice = _wn_range_idx(wavenumbers, 6600, 7200)
    water_1st_area = np.trapezoid(X[:, water_1st_slice], wavenumbers[water_1st_slice], axis=1)

    # CH伸縮領域（構造参照）
    ch_slice = _wn_range_idx(wavenumbers, 8100, 8500)
    ch_area = np.trapezoid(X[:, ch_slice], wavenumbers[ch_slice], axis=1)

    # 全域
    total_area = np.trapezoid(X, wavenumbers, axis=1)

    features = {
        # 水帯 / CH帯比 (Issue #14 + #9: 水/構造の比)
        "area_water_comb_ch": water_comb_area / (ch_area + eps),
        # 水帯 / 全域比
        "area_water_comb_total": water_comb_area / (total_area + eps),
        # 第1倍音 / CH帯比
        "area_water_1st_ch": water_1st_area / (ch_area + eps),
        # 2つの水帯の比
        "area_water_comb_1st": water_comb_area / (water_1st_area + eps),
    }
    return pd.DataFrame(features)


def continuum_removal(X: np.ndarray, wavenumbers: np.ndarray) -> np.ndarray:
    """コンティニュアム除去を適用する。

    凸包（上側エンベロープ）で正規化し、吸収バンドの深さを強調する。
    Issue #4: ドメインシフトの影響を低減。

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    wavenumbers : np.ndarray of shape (n_features,)

    Returns
    -------
    np.ndarray of shape (n_samples, n_features), CR値 (0〜1)
    """
    n_samples, n_features = X.shape
    cr = np.ones_like(X)

    for i in range(n_samples):
        spectrum = X[i]
        # 上側凸包の点を見つける
        # 左端と右端は必ず含める
        hull_x = [0]
        hull_y = [spectrum[0]]

        for j in range(1, n_features):
            while len(hull_x) >= 2:
                # 直前の2点と新しい点で凸性をチェック
                x0, y0 = hull_x[-2], hull_y[-2]
                x1, y1 = hull_x[-1], hull_y[-1]
                x2, y2 = j, spectrum[j]
                # 外積で判定（上側凸包なので右回りを除去）
                cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
                if cross >= 0:  # 左回りまたは直線 → 凸包に追加
                    break
                hull_x.pop()
                hull_y.pop()
            hull_x.append(j)
            hull_y.append(spectrum[j])

        # 凸包の各セグメントで線形補間してコンティニュアムを計算
        continuum = np.zeros(n_features)
        seg = 0
        for j in range(n_features):
            while seg < len(hull_x) - 2 and j > hull_x[seg + 1]:
                seg += 1
            x0, y0 = hull_x[seg], hull_y[seg]
            x1, y1 = hull_x[seg + 1], hull_y[seg + 1]
            if x1 == x0:
                continuum[j] = y0
            else:
                t = (j - x0) / (x1 - x0)
                continuum[j] = y0 + t * (y1 - y0)

        # CR = spectrum / continuum
        cr[i] = spectrum / (continuum + 1e-10)

    return np.clip(cr, 0, 1)


def compute_band_depths(X: np.ndarray, wavenumbers: np.ndarray) -> pd.DataFrame:
    """コンティニュアム除去後のバンド深さ特徴量。

    Band Depth = 1 - CR値。水の吸収帯での深さが含水率の直接指標。

    Returns
    -------
    pd.DataFrame of shape (n_samples, n_bands)
    """
    cr = continuum_removal(X, wavenumbers)

    features = {
        # 水コンビネーションバンドの深さ (Issue #14)
        "bd_5155": 1 - cr[:, _wn_idx(wavenumbers, 5155)],
        "bd_5200": 1 - cr[:, _wn_idx(wavenumbers, 5200)],
        # 水第1倍音の深さ
        "bd_6900": 1 - cr[:, _wn_idx(wavenumbers, 6900)],
        # 水帯領域の平均深さ
        "bd_water_comb_mean": 1 - cr[:, _wn_range_idx(wavenumbers, 5000, 5350)].mean(axis=1),
        "bd_water_1st_mean": 1 - cr[:, _wn_range_idx(wavenumbers, 6600, 7200)].mean(axis=1),
    }
    return pd.DataFrame(features)


def compute_spectral_moments(X: np.ndarray, wavenumbers: np.ndarray) -> pd.DataFrame:
    """水帯周辺のスペクトルモーメント。

    バンド幅や形状の統計量。低次モーメントは樹種汎化性が高い。

    Returns
    -------
    pd.DataFrame of shape (n_samples, n_moments)
    """
    features = {}

    for name, wn_low, wn_high in [
        ("water_comb", 5000, 5350),
        ("water_1st", 6600, 7200),
    ]:
        s = _wn_range_idx(wavenumbers, wn_low, wn_high)
        region = X[:, s]
        wn_region = wavenumbers[s]

        # 正規化して重み付きモーメントを計算
        region_norm = region / (region.sum(axis=1, keepdims=True) + 1e-10)

        # 1次モーメント: 重心波数
        centroid = (region_norm * wn_region).sum(axis=1)
        features[f"moment1_{name}"] = centroid

        # 2次モーメント: 分散（バンド幅に対応）
        variance = (region_norm * (wn_region - centroid[:, None]) ** 2).sum(axis=1)
        features[f"moment2_{name}"] = variance

    return pd.DataFrame(features)


def create_species_invariant_features(
    X: np.ndarray, wavenumbers: np.ndarray
) -> pd.DataFrame:
    """全ての樹種不変特徴量を統合して返す。

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    wavenumbers : np.ndarray of shape (n_features,)

    Returns
    -------
    pd.DataFrame of shape (n_samples, n_total_features)
    """
    dfs = [
        compute_band_ratios(X, wavenumbers),
        compute_ndmi(X, wavenumbers),
        compute_area_ratios(X, wavenumbers),
        compute_band_depths(X, wavenumbers),
        compute_spectral_moments(X, wavenumbers),
    ]
    return pd.concat(dfs, axis=1)
