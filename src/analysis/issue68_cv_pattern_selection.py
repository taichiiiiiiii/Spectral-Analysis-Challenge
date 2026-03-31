"""Issue #68: CVパターン D-O の樹種選定ロジック

パターン定義:
  D: PCA空間マハラノビス距離が近い樹種
  E: サンプル数が極端に少ない樹種を除外
  F: 指定された含水率レンジをカバーする樹種のみ
  G: fold RMSEのばらつきが小さい安定樹種
  H: 針葉樹/広葉樹分類でtest樹種に近いグループ
  I: MMD距離が小さい樹種
  J: CORAL距離（共分散行列のFrobenius距離）
  K: スペクトル同質性（樹種内分散が小さい）
  L: Wasserstein距離（含水率分布の最適輸送距離）
  M: スペクトル-含水率相関安定性
  N: 水分吸収帯のみでのtest類似度
  O: EPO残差でのtest類似度
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

from src.eda.issue9_wood_type_analysis import SOFTWOOD_SPECIES


# ============================================================
# Pattern D: PCA空間マハラノビス距離
# ============================================================
def select_pattern_d_mahalanobis(
    df_train: pd.DataFrame,
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 10,
    top_k: int = 6,
) -> set[str]:
    """train各樹種のPCA重心とtest重心のマハラノビス距離で近い樹種を選択。"""
    n_comp = min(n_components, X_train.shape[1], X_train.shape[0])
    pca = PCA(n_components=n_comp)
    Z_train = pca.fit_transform(X_train)
    Z_test = pca.transform(X_test)

    test_centroid = Z_test.mean(axis=0)

    # 全trainデータの共分散行列（正則化付き）
    cov = np.cov(Z_train, rowvar=False)
    cov += np.eye(n_comp) * 1e-6
    cov_inv = np.linalg.inv(cov)

    species = df_train["樹種"].values
    unique_species = sorted(set(species))

    distances = {}
    for sp in unique_species:
        mask = species == sp
        centroid = Z_train[mask].mean(axis=0)
        diff = centroid - test_centroid
        dist = float(np.sqrt(diff @ cov_inv @ diff))
        distances[sp] = dist

    sorted_species = sorted(distances, key=distances.get)
    return set(sorted_species[:top_k])


# ============================================================
# Pattern E: サンプル数フィルタ
# ============================================================
def select_pattern_e_sample_count(
    df_train: pd.DataFrame,
    min_samples: int = 20,
) -> set[str]:
    """サンプル数が閾値以上の樹種を選択。"""
    counts = df_train["樹種"].value_counts()
    return set(counts[counts >= min_samples].index)


# ============================================================
# Pattern F: 含水率レンジカバレッジ
# ============================================================
def select_pattern_f_moisture_coverage(
    df_train: pd.DataFrame,
    target_min: float,
    target_max: float,
    coverage_threshold: float = 0.5,
) -> set[str]:
    """各樹種の含水率レンジがターゲット範囲をどれだけカバーするかで選択。"""
    target_range = target_max - target_min
    if target_range <= 0:
        return set()

    selected = set()
    for sp, group in df_train.groupby("樹種"):
        sp_min = group["含水率"].min()
        sp_max = group["含水率"].max()
        # ターゲット範囲との重なりを計算
        overlap_min = max(sp_min, target_min)
        overlap_max = min(sp_max, target_max)
        overlap = max(0, overlap_max - overlap_min)
        coverage = overlap / target_range
        if coverage >= coverage_threshold:
            selected.add(sp)

    return selected


# ============================================================
# Pattern G: fold RMSE安定樹種
# ============================================================
def select_pattern_g_stable_folds(
    fold_rmses: dict[str, float],
    max_rmse: float | None = None,
    top_k: int | None = None,
) -> set[str]:
    """fold RMSEが安定（低い）樹種を選択。

    max_rmse指定時: RMSE <= max_rmse の樹種を選択
    top_k指定時: RMSE昇順でtop_k件を選択
    両方指定時: top_kを優先
    """
    sorted_species = sorted(fold_rmses, key=fold_rmses.get)

    if top_k is not None:
        return set(sorted_species[:top_k])

    if max_rmse is not None:
        return {sp for sp, r in fold_rmses.items() if r <= max_rmse}

    return set(fold_rmses.keys())


# ============================================================
# Pattern H: 針葉樹/広葉樹分類
# ============================================================
# train樹種リスト
TRAIN_SPECIES = {
    "イチョウ", "ウエンジ", "ウォールナット", "クリ", "スプルース",
    "チェリー", "トチ", "ナラ", "ヒノキ", "ベイスギ", "ベイマツ",
    "ホワイトオーク", "米ヒバ",
}

TRAIN_SOFTWOOD = TRAIN_SPECIES & SOFTWOOD_SPECIES
TRAIN_HARDWOOD = TRAIN_SPECIES - SOFTWOOD_SPECIES


def select_pattern_h_wood_type(
    test_species: set[str],
) -> set[str]:
    """test樹種の針葉樹/広葉樹比率に合わせてtrain樹種を選択。

    test樹種中の針葉樹割合に応じて、trainの針葉樹/広葉樹を選択。
    """
    n_test = len(test_species)
    if n_test == 0:
        return set()

    n_softwood_test = len(test_species & SOFTWOOD_SPECIES)
    softwood_ratio = n_softwood_test / n_test
    hardwood_ratio = 1 - softwood_ratio

    # test樹種の針葉樹/広葉樹比率に基づき、trainから同じ比率で選択
    # 少なくとも各タイプ1種は含める
    total_select = min(8, len(TRAIN_SOFTWOOD) + len(TRAIN_HARDWOOD))
    n_train_softwood = max(1, round(total_select * softwood_ratio))
    n_train_hardwood = max(1, total_select - n_train_softwood)

    n_train_softwood = min(n_train_softwood, len(TRAIN_SOFTWOOD))
    n_train_hardwood = min(n_train_hardwood, len(TRAIN_HARDWOOD))

    selected = set()
    selected.update(sorted(TRAIN_SOFTWOOD)[:n_train_softwood])
    selected.update(sorted(TRAIN_HARDWOOD)[:n_train_hardwood])

    return selected


# ============================================================
# Pattern I: MMD距離
# ============================================================
def _compute_spectral_mmd(
    X_species: np.ndarray,
    X_test: np.ndarray,
    gamma: float | None = None,
) -> float:
    """スペクトル空間でのMMD^2を計算。"""
    if gamma is None:
        all_X = np.vstack([X_species, X_test])
        dists = cdist(all_X, all_X)
        median_dist = np.median(dists[dists > 0])
        gamma = 1.0 / (median_dist ** 2 + 1e-10)

    K_ss = rbf_kernel(X_species, X_species, gamma=gamma).mean()
    K_tt = rbf_kernel(X_test, X_test, gamma=gamma).mean()
    K_st = rbf_kernel(X_species, X_test, gamma=gamma).mean()
    return float(K_ss + K_tt - 2 * K_st)


def select_pattern_i_mmd(
    df_train: pd.DataFrame,
    X_train: np.ndarray,
    X_test: np.ndarray,
    top_k: int = 6,
) -> set[str]:
    """train各樹種とtestのスペクトルMMD距離で近い樹種を選択。"""
    species = df_train["樹種"].values
    unique_species = sorted(set(species))

    # 全データから共通gammaを事前計算（樹種間の公平な比較のため）
    all_X = np.vstack([X_train, X_test])
    dists = cdist(all_X, all_X)
    median_dist = np.median(dists[dists > 0])
    gamma = 1.0 / (median_dist ** 2 + 1e-10)

    mmd_scores = {}
    for sp in unique_species:
        mask = species == sp
        mmd = _compute_spectral_mmd(X_train[mask], X_test, gamma=gamma)
        mmd_scores[sp] = mmd

    sorted_species = sorted(mmd_scores, key=mmd_scores.get)
    return set(sorted_species[:top_k])


# ============================================================
# Pattern J: 共分散Frobenius距離（PCA空間）
# ============================================================
def select_pattern_j_coral(
    df_train: pd.DataFrame,
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 20,
    top_k: int = 6,
) -> set[str]:
    """PCA空間での共分散行列のFrobenius距離で近い樹種を選択。

    高次元（1555次元）での共分散行列はランク不足になるため、
    PCA次元削減後に共分散を比較する。
    """
    species = df_train["樹種"].values
    unique_species = sorted(set(species))

    n_comp = min(n_components, X_train.shape[1], X_train.shape[0])
    pca = PCA(n_components=n_comp)
    Z_train = pca.fit_transform(X_train)
    Z_test = pca.transform(X_test)

    cov_test = np.cov(Z_test, rowvar=False)

    distances = {}
    for sp in unique_species:
        mask = species == sp
        Z_sp = Z_train[mask]
        if Z_sp.shape[0] < 2:
            distances[sp] = float("inf")
            continue
        cov_sp = np.cov(Z_sp, rowvar=False)
        dist = np.linalg.norm(cov_sp - cov_test, "fro")
        distances[sp] = float(dist)

    sorted_species = sorted(distances, key=distances.get)
    return set(sorted_species[:top_k])


# ============================================================
# Pattern K: スペクトル同質性（樹種内分散が小さい）
# ============================================================
def select_pattern_k_spectral_homogeneity(
    df_train: pd.DataFrame,
    X_train: np.ndarray,
    top_k: int = 6,
) -> set[str]:
    """樹種内スペクトル分散が小さい（同質的な）樹種を選択。"""
    species = df_train["樹種"].values
    unique_species = sorted(set(species))

    variances = {}
    for sp in unique_species:
        mask = species == sp
        X_sp = X_train[mask]
        centroid = X_sp.mean(axis=0)
        var = float(np.mean(np.sum((X_sp - centroid) ** 2, axis=1)))
        variances[sp] = var

    sorted_species = sorted(variances, key=variances.get)
    return set(sorted_species[:top_k])


# ============================================================
# Pattern L: Wasserstein距離（含水率分布）
# ============================================================
def select_pattern_l_wasserstein(
    df_train: pd.DataFrame,
    target_distribution: np.ndarray,
    top_k: int = 6,
) -> set[str]:
    """各樹種の含水率分布とターゲット分布のWasserstein距離で選択。

    target_distribution: 比較対象の含水率分布。testラベルは不明なため、
    train全体の含水率分布を使用する（train全体に近い分布の樹種を選択）。
    """
    distances = {}
    for sp, group in df_train.groupby("樹種"):
        sp_moisture = group["含水率"].values
        dist = wasserstein_distance(sp_moisture, target_distribution)
        distances[sp] = float(dist)

    sorted_species = sorted(distances, key=distances.get)
    return set(sorted_species[:top_k])


# ============================================================
# Pattern M: スペクトル-含水率相関安定性
# ============================================================
def select_pattern_m_correlation_stability(
    df_train: pd.DataFrame,
    X_train: np.ndarray,
    top_k: int = 6,
) -> set[str]:
    """波数-含水率相関が強く安定している樹種を選択。"""
    species = df_train["樹種"].values
    y = df_train["含水率"].values
    unique_species = sorted(set(species))

    scores = {}
    for sp in unique_species:
        mask = species == sp
        X_sp = X_train[mask]
        y_sp = y[mask]
        if len(y_sp) < 5 or np.std(y_sp) < 1e-6:
            scores[sp] = 0.0
            continue
        # 各波数と含水率の相関係数
        corrs = np.array([
            np.corrcoef(X_sp[:, j], y_sp)[0, 1]
            for j in range(X_sp.shape[1])
        ])
        corrs = np.nan_to_num(corrs)
        mean_abs_corr = float(np.mean(np.abs(corrs)))
        std_corr = float(np.std(np.abs(corrs))) + 0.01
        scores[sp] = mean_abs_corr / std_corr

    sorted_species = sorted(scores, key=scores.get, reverse=True)
    return set(sorted_species[:top_k])


# ============================================================
# Pattern N: 水分吸収帯フォーカス
# ============================================================
def select_pattern_n_water_band(
    df_train: pd.DataFrame,
    X_train: np.ndarray,
    X_test: np.ndarray,
    wave_indices: list[int] | None = None,
    top_k: int = 6,
) -> set[str]:
    """水分吸収帯のみでtestとのユークリッド距離が近い樹種を選択。"""
    if wave_indices is None:
        wave_indices = list(range(X_train.shape[1]))

    X_train_wb = X_train[:, wave_indices]
    X_test_wb = X_test[:, wave_indices]
    test_centroid = X_test_wb.mean(axis=0)

    species = df_train["樹種"].values
    unique_species = sorted(set(species))

    distances = {}
    for sp in unique_species:
        mask = species == sp
        sp_centroid = X_train_wb[mask].mean(axis=0)
        dist = float(np.linalg.norm(sp_centroid - test_centroid))
        distances[sp] = dist

    sorted_species = sorted(distances, key=distances.get)
    return set(sorted_species[:top_k])


# ============================================================
# Pattern O: EPO残差類似度
# ============================================================
def select_pattern_o_epo_residual(
    df_train: pd.DataFrame,
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 1,
    top_k: int = 6,
) -> set[str]:
    """EPO適用後のスペクトルでtestとの距離が近い樹種を選択。"""
    species = df_train["樹種"].values
    unique_species = sorted(set(species))

    # EPO射影行列を計算（樹種変動を除去）
    group_means = np.array([
        X_train[species == sp].mean(axis=0) for sp in unique_species
    ])
    group_means_centered = group_means - group_means.mean(axis=0)
    _, _, Vt = np.linalg.svd(group_means_centered, full_matrices=False)
    n_comp = min(n_components, len(unique_species) - 1, Vt.shape[0])
    D = Vt[:n_comp].T
    P = np.eye(X_train.shape[1]) - D @ D.T

    # EPO適用
    X_train_epo = X_train @ P
    X_test_epo = X_test @ P
    test_centroid = X_test_epo.mean(axis=0)

    distances = {}
    for sp in unique_species:
        mask = species == sp
        sp_centroid = X_train_epo[mask].mean(axis=0)
        dist = float(np.linalg.norm(sp_centroid - test_centroid))
        distances[sp] = dist

    sorted_species = sorted(distances, key=distances.get)
    return set(sorted_species[:top_k])
