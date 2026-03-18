"""di-PLS (Domain-Invariant PLS)

PLS重みベクトルにドメイン正則化項を追加し、ソース・ターゲット間の
分布差を抑制しながら予測モデルを構築する。

diPLSlibがsklearn新版と非互換のため自前実装。

参考: Nikzad-Langerodi et al., Analytical Chemistry 2018
"""
import numpy as np
from sklearn.preprocessing import StandardScaler


def _nipals_dipls(X_s, y_s, X_t, n_components, dipls_lambda):
    """di-PLS NIPALS algorithm.

    Modified PLS1 where weight vector minimizes:
        w = argmax [cov(Xw, y)^2 - lambda * ||mean(X_s @ w) - mean(X_t @ w)||^2]
    Solved via: w = (X_s^T X_s + lambda * D)^{-1} X_s^T y_s
    where D captures the domain discrepancy.
    """
    n_s, p = X_s.shape
    n_t = X_t.shape[0]

    # Center
    x_mean = X_s.mean(axis=0)
    y_mean = y_s.mean()
    X_sc = X_s - x_mean
    X_tc = X_t - x_mean
    y_c = y_s - y_mean

    W = np.zeros((p, n_components))
    P_load = np.zeros((p, n_components))
    Q = np.zeros(n_components)
    T = np.zeros((n_s, n_components))

    # Domain discrepancy matrix
    mean_diff = X_sc.mean(axis=0) - X_tc.mean(axis=0)
    D = np.outer(mean_diff, mean_diff)

    for a in range(n_components):
        # Modified weight: (X^T X + lambda * D) w = X^T y
        XtX = X_sc.T @ X_sc
        Xty = X_sc.T @ y_c

        A_mat = XtX + dipls_lambda * D
        A_mat += 1e-8 * np.eye(p)  # 正則化

        w = np.linalg.solve(A_mat, Xty)
        w = w / (np.linalg.norm(w) + 1e-12)

        # Score
        t = X_sc @ w
        # Loadings
        p_load = X_sc.T @ t / (t @ t + 1e-12)
        q = y_c @ t / (t @ t + 1e-12)

        W[:, a] = w
        P_load[:, a] = p_load
        Q[a] = q
        T[:, a] = t

        # Deflate
        X_sc = X_sc - np.outer(t, p_load)
        X_tc = X_tc - X_tc @ np.outer(w, p_load)
        y_c = y_c - t * q

        # Update D for deflated space
        mean_diff = X_sc.mean(axis=0) - X_tc.mean(axis=0)
        D = np.outer(mean_diff, mean_diff)

    return W, P_load, Q, x_mean, y_mean


def fit_predict_dipls(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    n_components: int = 4,
    dipls_lambda: float = 1.0,
) -> np.ndarray:
    """di-PLSで学習し、ターゲットの予測を返す。

    Parameters
    ----------
    X_source : (n_source, d)
    y_source : (n_source,)
    X_target : (n_target, d)
    n_components : PLS成分数
    dipls_lambda : ドメイン正則化パラメータ (0=通常PLS)

    Returns
    -------
    (n_target,) 予測値
    """
    W, P, Q, x_mean, y_mean = _nipals_dipls(
        X_source, y_source, X_target, n_components, dipls_lambda
    )

    # PLS regression coefficients: B = W (P^T W)^{-1} Q
    R = W @ np.linalg.inv(P.T @ W + 1e-10 * np.eye(n_components))
    B = R @ Q

    X_tc = X_target - x_mean
    return X_tc @ B + y_mean
