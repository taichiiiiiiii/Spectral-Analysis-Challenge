"""Locally Weighted PLS (LWPLS)

対応Issue: #33
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/33

テストサンプルごとに近傍の訓練データに重み付けしてPLSを適合する。
樹種間のドメインシフトに対して、局所的な適応が可能。
"""
import numpy as np
from sklearn.cross_decomposition import PLSRegression


def _gaussian_weights(distances: np.ndarray, sigma: float) -> np.ndarray:
    """ガウシアンカーネルで重みを計算"""
    weights = np.exp(-0.5 * (distances / sigma) ** 2)
    return weights / weights.sum()


def lwpls_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 4,
    k: int = 50,
    sigma_factor: float = 1.0,
) -> np.ndarray:
    """LWPLSで予測する。

    各テストサンプルに対して:
    1. k近傍の訓練サンプルを選択
    2. 距離ベースのガウシアン重みを計算
    3. 重み付きPLSを適合して予測

    Parameters
    ----------
    X_train : np.ndarray of shape (n_train, n_features)
    y_train : np.ndarray of shape (n_train,)
    X_test : np.ndarray of shape (n_test, n_features)
    n_components : int, PLS成分数
    k : int, 近傍数
    sigma_factor : float, ガウシアンカーネルの幅パラメータ（中央値距離に対する倍率）

    Returns
    -------
    np.ndarray of shape (n_test,)
    """
    n_test = X_test.shape[0]
    predictions = np.zeros(n_test)

    # n_componentsはk以下かつ特徴量数以下に制限
    max_comp = min(n_components, X_train.shape[1])

    for i in range(n_test):
        # テストサンプルとの距離を計算
        diffs = X_train - X_test[i]
        distances = np.sqrt(np.sum(diffs ** 2, axis=1))

        # k近傍を選択
        k_actual = min(k, len(X_train))
        nn_idx = np.argsort(distances)[:k_actual]

        X_nn = X_train[nn_idx]
        y_nn = y_train[nn_idx]
        d_nn = distances[nn_idx]

        # ガウシアン重み
        sigma = np.median(d_nn) * sigma_factor + 1e-10
        weights = _gaussian_weights(d_nn, sigma)

        # 重み付きPLS: サンプルを重みのsqrtでスケーリング
        sqrt_w = np.sqrt(weights)
        X_weighted = X_nn * sqrt_w[:, None]
        y_weighted = y_nn * sqrt_w

        n_comp = min(max_comp, k_actual)
        pls = PLSRegression(n_components=n_comp)
        pls.fit(X_weighted, y_weighted)

        # 予測（テストサンプルも同様にスケーリング不要 — 回帰係数をそのまま適用）
        predictions[i] = pls.predict(X_test[i:i+1].copy()).ravel()[0]

    return predictions
