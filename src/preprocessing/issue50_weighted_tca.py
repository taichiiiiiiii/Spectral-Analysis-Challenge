"""Weighted TCA

TCAのMMD計算にソースサンプルの重みを導入する。
ターゲットに近いソースサンプルを重視し、遠いサンプルの影響を抑える。
重みはKMM (Kernel Mean Matching) で推定するか、外部から指定できる。

参考: Huang et al., "Correcting Sample Selection Bias by Unlabeled Data", NIPS 2006 (KMM)
      Pan et al., "Domain Adaptation via Transfer Component Analysis", IEEE TNN 2011 (TCA)
"""
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
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


def _estimate_kmm_weights(
    X_source: np.ndarray,
    X_target: np.ndarray,
    gamma: float | None = None,
    B: float = 10.0,
) -> np.ndarray:
    """KMM (Kernel Mean Matching) でソースサンプルの重みを推定する。

    min  0.5 * beta^T K_ss beta - kappa^T beta
    s.t. 0 <= beta_i <= B, |sum(beta) - n_s| <= n_s * epsilon
    """
    n_s = len(X_source)
    n_t = len(X_target)

    if gamma is None:
        gamma = 1.0 / X_source.shape[1]

    K_ss = rbf_kernel(X_source, gamma=gamma)
    K_st = rbf_kernel(X_source, X_target, gamma=gamma)

    kappa = (n_s / n_t) * K_st.sum(axis=1)

    # 簡易QP: scipy.optimize.minimize (L-BFGS-B)
    def objective(beta):
        return 0.5 * beta @ K_ss @ beta - kappa @ beta

    def gradient(beta):
        return K_ss @ beta - kappa

    # 初期値: 均一重み
    beta0 = np.ones(n_s)
    bounds = [(0.0, B)] * n_s

    result = minimize(
        objective,
        beta0,
        jac=gradient,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500},
    )

    weights = result.x
    # 正規化: 平均1に
    weights = weights / weights.mean()
    return weights


def weighted_tca_transform(
    X_source: np.ndarray,
    X_target: np.ndarray,
    y_source: np.ndarray,
    n_components: int = 10,
    kernel: str = "rbf",
    gamma: float | None = None,
    mu: float = 1.0,
    source_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted TCAによるドメイン適応変換を行う。

    Parameters
    ----------
    X_source : (n_source, d)
    X_target : (n_target, d)
    y_source : (n_source,) ソースの連続ラベル（KMM重み推定には使わないが、
               インターフェース統一のため受け取る）
    n_components : 射影後の次元数
    kernel : "rbf" or "linear"
    gamma : RBFカーネルのパラメータ
    mu : 正則化パラメータ
    source_weights : ソースサンプルの重み。Noneの場合KMMで推定

    Returns
    -------
    Z_source : (n_source, n_components)
    Z_target : (n_target, n_components)
    """
    n_s = len(X_source)
    n_t = len(X_target)
    n = n_s + n_t

    # 重みの推定または使用
    if source_weights is None:
        source_weights = _estimate_kmm_weights(X_source, X_target, gamma=gamma)
    else:
        source_weights = source_weights / source_weights.mean()

    K = _compute_kernel_matrix(X_source, X_target, kernel=kernel, gamma=gamma)

    # 重み付きMMD行列（ベクトル化）
    w = source_weights
    L = np.zeros((n, n))
    L[:n_s, :n_s] = np.outer(w, w) / (n_s * n_s)
    L[n_s:, n_s:] = 1.0 / (n_t * n_t)
    cross = w[:, None] / (n_s * n_t)  # (n_s, 1)
    L[:n_s, n_s:] = -np.tile(cross, (1, n_t))
    L[n_s:, :n_s] = L[:n_s, n_s:].T

    H = np.eye(n) - np.ones((n, n)) / n

    A = K @ L @ K + mu * np.eye(n)
    B = K @ H @ K

    B = (B + B.T) / 2
    min_eig = np.real(np.linalg.eigvalsh(B).min())
    reg = max(1e-8, -min_eig + 1e-6) if min_eig < 1e-6 else 1e-8
    B += reg * np.eye(n)

    eigenvalues, eigenvectors = eigh(A, B)
    W = eigenvectors[:, :n_components]

    Z = K @ W
    Z_source = Z[:n_s]
    Z_target = Z[n_s:]

    return Z_source, Z_target
