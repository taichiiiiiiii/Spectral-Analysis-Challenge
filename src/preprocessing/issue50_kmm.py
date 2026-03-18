"""KMM (Kernel Mean Matching) インスタンス重み付け

ターゲットドメインに近い訓練サンプルを重み付けすることで
covariate shiftに対処する。

参考: Huang et al., NIPS 2006
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cross_decomposition import PLSRegression


def compute_kmm_weights(
    X_source: np.ndarray,
    X_target: np.ndarray,
    gamma: float | None = None,
    B: float = 10.0,
    eps: float = 0.1,
) -> np.ndarray:
    """KMMでソースサンプルの重みを計算する。

    Parameters
    ----------
    X_source : (n_s, d)
    X_target : (n_t, d)
    gamma : RBFカーネルパラメータ (None=median heuristic)
    B : 重みの上限
    eps : 制約の緩和パラメータ

    Returns
    -------
    (n_s,) 重みベクトル
    """
    n_s = len(X_source)
    n_t = len(X_target)

    if gamma is None:
        from sklearn.metrics.pairwise import euclidean_distances
        dists = euclidean_distances(X_source, X_target)
        gamma = 1.0 / (np.median(dists.ravel()) ** 2 + 1e-10)

    K_ss = rbf_kernel(X_source, X_source, gamma=gamma)
    K_st = rbf_kernel(X_source, X_target, gamma=gamma)
    kappa = K_st.mean(axis=1) * (n_s / n_t)

    # 正則化でK_ssを安定化
    K_ss += 1e-6 * np.eye(n_s)

    def objective(beta):
        return 0.5 * beta @ K_ss @ beta - kappa @ beta

    def jac(beta):
        return K_ss @ beta - kappa

    bounds = [(0, B)] * n_s
    constraints = [
        {"type": "ineq", "fun": lambda b: n_s * eps - abs(b.sum() - n_s)}
    ]

    result = minimize(
        objective, x0=np.ones(n_s), jac=jac,
        method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 500},
    )
    return np.maximum(result.x, 0)


def weighted_pls_predict(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    weights: np.ndarray,
    n_components: int = 4,
) -> np.ndarray:
    """重み付きPLSで予測する。"""
    sqrt_w = np.sqrt(weights)
    X_w = X_source * sqrt_w[:, None]
    y_w = y_source * sqrt_w
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_w, y_w)
    return pls.predict(X_target).ravel()
