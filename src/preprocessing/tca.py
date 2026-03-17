"""Transfer Component Analysis (TCA)

MMD (Maximum Mean Discrepancy) を最小化するカーネル空間への射影を学習し、
ソースドメインとターゲットドメインの分布差を縮小する。

参考: Pan et al., "Domain Adaptation via Transfer Component Analysis", IEEE TNN 2011
"""
import numpy as np
from scipy.linalg import eigh
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel


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


def _compute_mmd_matrix(n_source: int, n_target: int) -> np.ndarray:
    """MMDペナルティ行列 L を計算する。"""
    n = n_source + n_target
    L = np.zeros((n, n))
    L[:n_source, :n_source] = 1.0 / (n_source * n_source)
    L[n_source:, n_source:] = 1.0 / (n_target * n_target)
    L[:n_source, n_source:] = -1.0 / (n_source * n_target)
    L[n_source:, :n_source] = -1.0 / (n_source * n_target)
    return L


def tca_transform(
    X_source: np.ndarray,
    X_target: np.ndarray,
    n_components: int = 10,
    kernel: str = "rbf",
    gamma: float | None = None,
    mu: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """TCAによるドメイン適応変換を行う。

    Parameters
    ----------
    X_source : (n_source, d)
    X_target : (n_target, d)
    n_components : 射影後の次元数
    kernel : "rbf" or "linear"
    gamma : RBFカーネルのパラメータ (Noneならデフォルト=1/d)
    mu : 正則化パラメータ

    Returns
    -------
    Z_source : (n_source, n_components)
    Z_target : (n_target, n_components)
    """
    n_s = len(X_source)
    n_t = len(X_target)
    n = n_s + n_t

    K = _compute_kernel_matrix(X_source, X_target, kernel=kernel, gamma=gamma)
    L = _compute_mmd_matrix(n_s, n_t)

    # 正則化付きMMD最小化: min tr(W^T K L K W) s.t. W^T K H K W = I
    # H = I - (1/n) * 11^T (centering matrix)
    H = np.eye(n) - np.ones((n, n)) / n

    # 一般化固有値問題: (KLK + mu*I) W = KHK W Lambda
    A = K @ L @ K + mu * np.eye(n)
    B = K @ H @ K

    # B を正定値に安定化
    B = (B + B.T) / 2
    # 固有値を確認して十分な正則化を追加
    min_eig = np.real(np.linalg.eigvalsh(B).min())
    reg = max(1e-8, -min_eig + 1e-6) if min_eig < 1e-6 else 1e-8
    B += reg * np.eye(n)

    # 最小固有値に対応する固有ベクトルを取得
    eigenvalues, eigenvectors = eigh(A, B)

    # 最小のn_components個を選択
    W = eigenvectors[:, :n_components]

    # 射影
    Z = K @ W
    Z_source = Z[:n_s]
    Z_target = Z[n_s:]

    return Z_source, Z_target
