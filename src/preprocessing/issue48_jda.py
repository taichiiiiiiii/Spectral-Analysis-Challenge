"""JDA (Joint Distribution Adaptation)

TCAを拡張し、周辺分布 P(X) だけでなく条件付き分布 P(Y|X) も同時に整合させる。
回帰問題では y を離散ビンに分割して擬似クラスラベルとして扱い、
各ビンごとのMMDも最小化する反復アルゴリズム。

参考: Long et al., "Transfer Feature Learning with Joint Distribution Adaptation", ICCV 2013
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
    """周辺分布のMMDペナルティ行列 L0 を計算する。"""
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
) -> np.ndarray:
    """条件付き分布のMMDペナルティ行列 Lc を計算する。

    各ビンcについて、ソース側のビンcサンプルとターゲット側のビンcサンプルの
    MMDを合算する。
    """
    n_s = len(y_source_bins)
    n_t = len(y_target_bins)
    n = n_s + n_t
    Lc = np.zeros((n, n))

    for c in range(n_bins):
        src_idx = np.where(y_source_bins == c)[0]
        tgt_idx = np.where(y_target_bins == c)[0] + n_s  # オフセット

        n_sc = len(src_idx)
        n_tc = len(tgt_idx) if len(tgt_idx) > 0 else 0

        if n_sc == 0 or n_tc == 0:
            continue

        # ベクトル化
        Lc[np.ix_(src_idx, src_idx)] += 1.0 / (n_sc * n_sc)
        Lc[np.ix_(tgt_idx, tgt_idx)] += 1.0 / (n_tc * n_tc)
        Lc[np.ix_(src_idx, tgt_idx)] -= 1.0 / (n_sc * n_tc)
        Lc[np.ix_(tgt_idx, src_idx)] -= 1.0 / (n_sc * n_tc)

    return Lc


def _bin_continuous_values(y: np.ndarray, n_bins: int) -> np.ndarray:
    """連続値をn_bins個の等頻度ビンに離散化する。"""
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(y, percentiles)
    # digitizeで振り分け（右端を含むように調整）
    bins = np.digitize(y, bin_edges[1:-1], right=False)
    return bins


def _predict_target_bins(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """ソースデータで簡易PLSモデルを学習し、ターゲットのy擬似ラベル（ビン）を予測する。"""
    n_comp = min(3, X_source.shape[1], X_source.shape[0])
    pls = PLSRegression(n_components=n_comp)
    pls.fit(X_source, y_source)
    y_pred = pls.predict(X_target).ravel()

    # ソースのビン境界を使ってターゲットのビンを割り当て
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(y_source, percentiles)
    bins = np.digitize(y_pred, bin_edges[1:-1], right=False)
    return bins


def jda_transform(
    X_source: np.ndarray,
    X_target: np.ndarray,
    y_source: np.ndarray,
    n_components: int = 10,
    kernel: str = "rbf",
    gamma: float | None = None,
    mu: float = 1.0,
    n_iterations: int = 3,
    n_bins: int = 5,
    lam: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """JDAによるドメイン適応変換を行う。

    Parameters
    ----------
    X_source : (n_source, d)
    X_target : (n_target, d)
    y_source : (n_source,) ソースの連続ラベル
    n_components : 射影後の次元数
    kernel : "rbf" or "linear"
    gamma : RBFカーネルのパラメータ
    mu : 正則化パラメータ
    n_iterations : 反復回数
    n_bins : 回帰用y離散化のビン数
    lam : 条件付き分布MMDの重み

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

    # ソースのyをビン化
    y_source_bins = _bin_continuous_values(y_source, n_bins)

    # 現在の射影で変換されたデータ（初回はオリジナル）
    X_s_current = X_source.copy()
    X_t_current = X_target.copy()

    W = None
    for iteration in range(n_iterations):
        # ターゲットの擬似ラベルを予測
        y_target_bins = _predict_target_bins(
            X_s_current, y_source, X_t_current, n_bins
        )

        # 条件付き分布MMD行列を計算
        Lc = _compute_conditional_mmd_matrix(y_source_bins, y_target_bins, n_bins)

        # 統合MMD行列: L = L0 + lam * Lc
        L = L0 + lam * Lc

        # 一般化固有値問題: (KLK + mu*I) W = KHK W Lambda
        A = K @ L @ K + mu * np.eye(n)
        B = K @ H @ K

        # Bを正定値に安定化
        B = (B + B.T) / 2
        min_eig = np.real(np.linalg.eigvalsh(B).min())
        reg = max(1e-8, -min_eig + 1e-6) if min_eig < 1e-6 else 1e-8
        B += reg * np.eye(n)

        eigenvalues, eigenvectors = eigh(A, B)
        W = eigenvectors[:, :n_components]

        # 射影して次の反復の入力にする
        Z = K @ W
        X_s_current = Z[:n_s]
        X_t_current = Z[n_s:]

    # 最終射影
    Z = K @ W
    Z_source = Z[:n_s]
    Z_target = Z[n_s:]

    return Z_source, Z_target
