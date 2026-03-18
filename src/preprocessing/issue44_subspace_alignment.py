"""Subspace Alignment (SA)

ソース/ターゲットのPCA部分空間を閉形式で整列させる。

参考: Fernando et al., ICCV 2013
"""
import numpy as np
from sklearn.decomposition import PCA


def subspace_align(
    X_source: np.ndarray,
    X_target: np.ndarray,
    n_components: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Subspace Alignmentでドメイン適応する。

    Parameters
    ----------
    X_source : (n_s, d)
    X_target : (n_t, d)
    n_components : 部分空間の次元数

    Returns
    -------
    Z_source : (n_s, n_components)
    Z_target : (n_t, n_components)
    """
    pca_s = PCA(n_components=n_components)
    pca_t = PCA(n_components=n_components)
    pca_s.fit(X_source)
    pca_t.fit(X_target)

    Xs = pca_s.components_.T  # (d, n_components)
    Xt = pca_t.components_.T  # (d, n_components)

    # 整列行列 (閉形式)
    M = Xs.T @ Xt  # (n_components, n_components)

    # 射影
    Z_source = X_source @ Xs @ M
    Z_target = X_target @ Xt

    return Z_source, Z_target
