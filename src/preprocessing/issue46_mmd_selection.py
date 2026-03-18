"""MMD-based Preprocessing Selection

予測値分布のMMDを最小化する前処理を自動選択するメタ戦略。

参考: Unsupervised optimization of spectral pre-processing selection, Measurement 2025
"""
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics.pairwise import rbf_kernel


def compute_prediction_mmd(
    y_source: np.ndarray,
    y_target: np.ndarray,
    gamma: float | None = None,
) -> float:
    """予測値間のMMD^2を計算する。"""
    ys = y_source.reshape(-1, 1)
    yt = y_target.reshape(-1, 1)

    if gamma is None:
        all_y = np.vstack([ys, yt])
        from scipy.spatial.distance import cdist
        dists = cdist(all_y, all_y)
        gamma = 1.0 / (np.median(dists[dists > 0]) ** 2 + 1e-10)

    K_ss = rbf_kernel(ys, ys, gamma=gamma).mean()
    K_tt = rbf_kernel(yt, yt, gamma=gamma).mean()
    K_st = rbf_kernel(ys, yt, gamma=gamma).mean()
    return float(K_ss + K_tt - 2 * K_st)


def select_best_preprocessing(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    preprocessings: dict,
    n_pls_components: int = 4,
) -> tuple[str, float]:
    """MMD最小の前処理を選択する。

    Parameters
    ----------
    X_source, y_source : 訓練データ
    X_target : テストデータ (ラベルなし)
    preprocessings : {name: callable(X) -> X_preprocessed}
    n_pls_components : PLS成分数

    Returns
    -------
    (best_name, best_mmd)
    """
    results = select_best_preprocessing_detailed(
        X_source, y_source, X_target, preprocessings, n_pls_components
    )
    best = min(results, key=lambda r: r["mmd"])
    return best["name"], best["mmd"]


def select_best_preprocessing_detailed(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    preprocessings: dict,
    n_pls_components: int = 4,
) -> list[dict]:
    """全前処理のMMDスコアを返す。"""
    results = []
    for name, preproc in preprocessings.items():
        try:
            X_s_pp = preproc(X_source)
            X_t_pp = preproc(X_target)
            pls = PLSRegression(n_components=n_pls_components)
            pls.fit(X_s_pp, y_source)
            y_pred_s = pls.predict(X_s_pp).ravel()
            y_pred_t = pls.predict(X_t_pp).ravel()
            mmd = compute_prediction_mmd(y_pred_s, y_pred_t)
            results.append({"name": name, "mmd": mmd})
        except Exception as e:
            results.append({"name": name, "mmd": float("inf"), "error": str(e)})
    return results
