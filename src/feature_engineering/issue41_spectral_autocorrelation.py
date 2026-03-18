"""スペクトル自己相関特徴量

各サンプルのスペクトルに対してlag-k自己相関関数(ACF)を計算し、
スペクトル形状の滑らかさや周期性を捉える特徴量を生成する。
樹種に依存しにくい形状特徴として、ドメインシフトにロバストな予測に寄与する。
"""
import numpy as np
import pandas as pd


def _acf_single(x: np.ndarray, lag: int) -> float:
    """1サンプルのlag-k自己相関を計算する。"""
    n = len(x)
    if lag >= n:
        return 0.0
    mean = x.mean()
    var = np.sum((x - mean) ** 2)
    if var < 1e-12:
        return 0.0
    cov = np.sum((x[:n - lag] - mean) * (x[lag:] - mean))
    return cov / var


def compute_spectral_autocorrelation(
    X: np.ndarray,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """スペクトル自己相関特徴量を計算する。

    Parameters
    ----------
    X : (n_samples, n_features) スペクトルデータ
    lags : 計算するラグのリスト (デフォルト: [1, 5, 10, 20, 50])

    Returns
    -------
    pd.DataFrame of shape (n_samples, len(lags))
    """
    if lags is None:
        lags = [1, 5, 10, 20, 50]

    n_samples = X.shape[0]
    result = np.zeros((n_samples, len(lags)))

    for i in range(n_samples):
        for j, lag in enumerate(lags):
            result[i, j] = _acf_single(X[i], lag)

    columns = [f"acf_lag{lag}" for lag in lags]
    return pd.DataFrame(result, columns=columns)
