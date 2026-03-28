"""追加前処理手法: AsLS, PiecewiseMSC, Whittaker, 水吸収帯レンジ選択

対応Issue: #62（前処理×線形モデル統合評価の前提）
Issue #60, #61で使用された前処理のうち、未実装の手法を追加する。
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from src.preprocessing.issue19_msc import compute_msc_reference, apply_msc


def apply_asls(X: np.ndarray, lam: float = 1e5, p: float = 0.01, n_iter: int = 10) -> np.ndarray:
    """Asymmetric Least Squares (AsLS) ベースライン補正を適用する。

    各サンプルのベースラインを推定し、差し引く。
    Eilers & Boelens (2005) のアルゴリズム。

    Parameters
    ----------
    X : (n_samples, n_features)
    lam : 平滑化パラメータ（大きいほど滑らか）
    p : 非対称パラメータ（0 < p < 1, 小さいほどベースライン寄り）
    n_iter : 反復回数

    Returns
    -------
    np.ndarray of shape (n_samples, n_features), ベースライン補正後
    """
    X = np.array(X, dtype=float)
    n_samples, n_features = X.shape
    X_corrected = np.zeros_like(X)

    # 2次差分行列 D
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(n_features - 2, n_features), dtype=float)
    DTD = (lam * D.T @ D).tocsc()

    for i in range(n_samples):
        y = X[i]
        w = np.ones(n_features)
        for _ in range(n_iter):
            W = sparse.diags(w, dtype=float, format="csc")
            z = spsolve(W + DTD, w * y)
            w = p * (y > z) + (1 - p) * (y <= z)
        X_corrected[i] = y - z

    return X_corrected


def apply_whittaker(X: np.ndarray, lam: float = 1e3, d: int = 2) -> np.ndarray:
    """Whittaker smoother を適用する。

    Eilers (2003) のペナルティ付き最小二乗法による平滑化。

    Parameters
    ----------
    X : (n_samples, n_features)
    lam : 平滑化パラメータ
    d : 差分の次数

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)
    """
    X = np.array(X, dtype=float)
    n_samples, n_features = X.shape
    X_smooth = np.zeros_like(X)

    I = sparse.eye(n_features, dtype=float)
    D = I
    for _ in range(d):
        n = D.shape[0]
        D = sparse.diags([1, -1], [0, 1], shape=(n - 1, n), dtype=float) @ D

    penalty = (lam * D.T @ D).tocsc()

    A = (I + penalty).tocsc()
    for i in range(n_samples):
        X_smooth[i] = spsolve(A, X[i])

    return X_smooth


def apply_piecewise_msc(X: np.ndarray, reference: np.ndarray, n_segments: int = 3) -> np.ndarray:
    """Piecewise MSC: スペクトルをセグメントに分割して各セグメントごとにMSC適用。

    異なるスペクトル領域で散乱特性が異なる場合に有効。

    Parameters
    ----------
    X : (n_samples, n_features)
    reference : (n_features,) リファレンススペクトル
    n_segments : セグメント数

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)
    """
    X = np.array(X, dtype=float)
    reference = np.array(reference, dtype=float)
    n_samples, n_features = X.shape

    segment_size = n_features // n_segments
    X_corrected = np.zeros_like(X)

    for seg in range(n_segments):
        start = seg * segment_size
        end = n_features if seg == n_segments - 1 else (seg + 1) * segment_size

        X_seg = X[:, start:end]
        ref_seg = reference[start:end]

        ref_with_bias = np.column_stack([ref_seg, np.ones(len(ref_seg))])
        for i in range(n_samples):
            coeffs, _, _, _ = np.linalg.lstsq(ref_with_bias, X_seg[i], rcond=None)
            a, b = coeffs[0], coeffs[1]
            X_corrected[i, start:end] = (X_seg[i] - b) / a

    return X_corrected


def select_water_bands(X: np.ndarray, wavenumbers: np.ndarray) -> np.ndarray:
    """水吸収帯のみを選択する（water_bands）。

    水の第一倍音（~6900 cm⁻¹）と結合音（~5200 cm⁻¹）の2帯域。

    Parameters
    ----------
    X : (n_samples, n_features)
    wavenumbers : (n_features,) 波数

    Returns
    -------
    np.ndarray, 選択された列のみ
    """
    mask = ((wavenumbers >= 6600) & (wavenumbers <= 7200)) | \
           ((wavenumbers >= 5000) & (wavenumbers <= 5350))
    return X[:, mask]


def select_water_wide(X: np.ndarray, wavenumbers: np.ndarray) -> np.ndarray:
    """水吸収帯の広域選択（water_wide）。

    water_bandsより広い範囲を選択。

    Parameters
    ----------
    X : (n_samples, n_features)
    wavenumbers : (n_features,) 波数

    Returns
    -------
    np.ndarray, 選択された列のみ
    """
    mask = ((wavenumbers >= 6400) & (wavenumbers <= 7400)) | \
           ((wavenumbers >= 4800) & (wavenumbers <= 5600))
    return X[:, mask]
