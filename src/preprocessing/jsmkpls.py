"""JSMKPLS (Joint Statistical and Manifold alignment in Kernel PLS subspace)

カーネル空間でMMD（統計的整列）+ ラプラシアン正則化（マニフォルド保存）を
同時に最小化する射影を学習する。TCAの拡張版。

参考: Yang et al., Chemometrics and Intelligent Laboratory Systems 2025
"""
import numpy as np
from scipy.linalg import eigh
from sklearn.metrics.pairwise import rbf_kernel


def jsmkpls_transform(
    X_source: np.ndarray,
    X_target: np.ndarray,
    n_components: int = 10,
    gamma: float | None = None,
    gamma_local: float | None = None,
    mu: float = 1.0,
    lam: float = 1.0,
    eta: float = 1.0,
    k_neighbors: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """JSMKPLSによるドメイン適応変換。

    Parameters
    ----------
    X_source : (n_s, d)
    X_target : (n_t, d)
    n_components : 射影後の次元数
    gamma : カーネルパラメータ (None=auto)
    gamma_local : マニフォルド用カーネルパラメータ (None=auto)
    mu : MMD正則化の重み
    lam : マニフォルド正則化の重み
    eta : 単位行列正則化の重み
    k_neighbors : マニフォルドのk-NN数

    Returns
    -------
    Z_source : (n_s, n_components)
    Z_target : (n_t, n_components)
    """
    n_s = len(X_source)
    n_t = len(X_target)
    n = n_s + n_t
    X_all = np.vstack([X_source, X_target])

    if gamma is None:
        gamma = 1.0 / X_all.shape[1]
    if gamma_local is None:
        gamma_local = gamma

    # カーネル行列
    K = rbf_kernel(X_all, gamma=gamma)

    # MMD行列
    M_mmd = np.zeros((n, n))
    M_mmd[:n_s, :n_s] = 1.0 / (n_s * n_s)
    M_mmd[n_s:, n_s:] = 1.0 / (n_t * n_t)
    M_mmd[:n_s, n_s:] = -1.0 / (n_s * n_t)
    M_mmd[n_s:, :n_s] = -1.0 / (n_s * n_t)

    # ラプラシアン行列 (マニフォルド正則化)
    W_knn = rbf_kernel(X_all, gamma=gamma_local)
    # k-NNスパース化
    for i in range(n):
        idx = np.argsort(W_knn[i])[:-(k_neighbors + 1)]
        W_knn[i, idx] = 0
    W_knn = (W_knn + W_knn.T) / 2
    D_lap = np.diag(W_knn.sum(axis=1))
    L = D_lap - W_knn

    # センタリング行列
    H = np.eye(n) - np.ones((n, n)) / n

    # 一般化固有値問題
    # max: alpha^T K H K alpha
    # min: alpha^T K (mu*M_mmd + lam*L + eta*I) K alpha
    A = K @ H @ K
    B = K @ (mu * M_mmd + lam * L + eta * np.eye(n)) @ K

    # Bを正定値に安定化
    B = (B + B.T) / 2
    min_eig = np.real(np.linalg.eigvalsh(B).min())
    reg = max(1e-6, -min_eig + 1e-4) if min_eig < 1e-4 else 1e-6
    B += reg * np.eye(n)

    A = (A + A.T) / 2

    eigenvalues, eigenvectors = eigh(A, B)

    # 最大固有値に対応するn_components個を選択
    W = eigenvectors[:, -n_components:]

    # 射影
    Z = K @ W
    Z_source = Z[:n_s]
    Z_target = Z[n_s:]

    return Z_source, Z_target
