"""BDA (Balanced Distribution Adaptation)

JDAを拡張し、周辺分布と条件付き分布のバランスを適応的に調整する。
バランスファクター mu_b で周辺分布MMDと条件付き分布MMDの重みを制御:
  L = mu_b * L_marginal + (1 - mu_b) * L_conditional

mu_b が大きいほど周辺分布整合を重視、小さいほど条件付き分布整合を重視。

参考: Wang et al., "Balanced Distribution Adaptation for Transfer Learning", ICDM 2017
"""
import numpy as np
from scipy.linalg import eigh
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from sklearn.cross_decomposition import PLSRegression


def _compute_kernel_matrix(
    X_source: np.ndarray,
    X_target: np.ndarray,
    kernel: str = "rbf",
    gamma: float | None = None,
) -> np.ndarray:
    """ソース+ターゲット結合データのカーネル行列を計算する。"""
    X_all = np.vstack([X_source, X_target])
    if kernel == "rbf":
        if gamma is None:
            gamma = 1.0 / X_all.shape[1]
        K = rbf_kernel(X_all, gamma=gamma)
    elif kernel == "linear":
        K = linear_kernel(X_all)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    return K


def _compute_marginal_mmd_matrix(n_source: int, n_target: int) -> np.ndarray:
    """周辺分布のMMDペナルティ行列を計算する。"""
    n = n_source + n_target
    L = np.zeros((n, n))
    L[:n_source, :n_source] = 1.0 / (n_source * n_source)
    L[n_source:, n_source:] = 1.0 / (n_target * n_target)
    L[:n_source, n_source:] = -1.0 / (n_source * n_target)
    L[n_source:, :n_source] = -1.0 / (n_source * n_target)
    return L


def _compute_conditional_mmd_matrix(
    y_source_bins: np.ndarray,
    y_target_bins: np.ndarray,
    n_bins: int,
    n_source: int,
) -> np.ndarray:
    """条件付き分布のMMDペナルティ行列を計算する。"""
    n = n_source + len(y_target_bins)
    Lc = np.zeros((n, n))

    for c in range(n_bins):
        src_idx = np.where(y_source_bins == c)[0]
        tgt_idx = np.where(y_target_bins == c)[0] + n_source

        n_sc = len(src_idx)
        n_tc = len(tgt_idx)

        if n_sc == 0 or n_tc == 0:
            continue

        # ベクトル化して高速化
        Lc[np.ix_(src_idx, src_idx)] += 1.0 / (n_sc * n_sc)
        Lc[np.ix_(tgt_idx, tgt_idx)] += 1.0 / (n_tc * n_tc)
        Lc[np.ix_(src_idx, tgt_idx)] -= 1.0 / (n_sc * n_tc)
        Lc[np.ix_(tgt_idx, src_idx)] -= 1.0 / (n_sc * n_tc)

    return Lc


def _bin_continuous_values(y: np.ndarray, n_bins: int) -> np.ndarray:
    """連続値をn_bins個の等頻度ビンに離散化する。"""
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(y, percentiles)
    bins = np.digitize(y, bin_edges[1:-1], right=False)
    return bins


def _predict_target_bins(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """ソースデータでPLSモデルを学習し、ターゲットの擬似ラベル（ビン）を予測する。"""
    n_comp = min(3, X_source.shape[1], X_source.shape[0])
    pls = PLSRegression(n_components=n_comp)
    pls.fit(X_source, y_source)
    y_pred = pls.predict(X_target).ravel()

    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(y_source, percentiles)
    bins = np.digitize(y_pred, bin_edges[1:-1], right=False)
    return bins


def bda_transform(
    X_source: np.ndarray,
    X_target: np.ndarray,
    y_source: np.ndarray,
    n_components: int = 10,
    kernel: str = "rbf",
    gamma: float | None = None,
    mu: float = 1.0,
    mu_b: float = 0.5,
    n_iterations: int = 3,
    n_bins: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """BDAによるドメイン適応変換を行う。

    Parameters
    ----------
    X_source : (n_source, d)
    X_target : (n_target, d)
    y_source : (n_source,) ソースの連続ラベル
    n_components : 射影後の次元数
    kernel : "rbf" or "linear"
    gamma : RBFカーネルのパラメータ
    mu : 正則化パラメータ
    mu_b : バランスファクター (0~1)。大きいほど周辺分布整合を重視
    n_iterations : 反復回数
    n_bins : 回帰用y離散化のビン数

    Returns
    -------
    Z_source : (n_source, n_components)
    Z_target : (n_target, n_components)
    """
    n_s = len(X_source)
    n_t = len(X_target)
    n = n_s + n_t

    K = _compute_kernel_matrix(X_source, X_target, kernel=kernel, gamma=gamma)
    L0 = _compute_marginal_mmd_matrix(n_s, n_t)
    H = np.eye(n) - np.ones((n, n)) / n

    y_source_bins = _bin_continuous_values(y_source, n_bins)

    X_s_current = X_source.copy()
    X_t_current = X_target.copy()

    W = None
    for iteration in range(n_iterations):
        y_target_bins = _predict_target_bins(
            X_s_current, y_source, X_t_current, n_bins
        )

        Lc = _compute_conditional_mmd_matrix(
            y_source_bins, y_target_bins, n_bins, n_s
        )

        # BDAのバランス: L = mu_b * L0 + (1 - mu_b) * Lc
        L = mu_b * L0 + (1.0 - mu_b) * Lc

        A = K @ L @ K + mu * np.eye(n)
        B = K @ H @ K

        B = (B + B.T) / 2
        min_eig = np.real(np.linalg.eigvalsh(B).min())
        reg = max(1e-8, -min_eig + 1e-6) if min_eig < 1e-6 else 1e-8
        B += reg * np.eye(n)

        eigenvalues, eigenvectors = eigh(A, B)
        W = eigenvectors[:, :n_components]

        Z = K @ W
        X_s_current = Z[:n_s]
        X_t_current = Z[n_s:]

    Z = K @ W
    Z_source = Z[:n_s]
    Z_target = Z[n_s:]

    return Z_source, Z_target
