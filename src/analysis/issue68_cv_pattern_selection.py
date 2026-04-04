"""Issue #68: CVパターンの樹種選定ロジック

trainデータのみを使用するパターン（ルール準拠）:
  E: サンプル数が極端に少ない樹種を除外
  F: 指定された含水率レンジをカバーする樹種のみ
  G: fold RMSEのばらつきが小さい安定樹種
  H: 針葉樹/広葉樹分類でtest樹種に近いグループ
  K: スペクトル同質性（樹種内分散が小さい）
  L: Wasserstein距離（含水率分布の最適輸送距離）
  M: スペクトル-含水率相関安定性

削除済み（ルール違反: testデータの集団統計量を使用）:
  D, I, J, N, O
"""
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from src.eda.issue9_wood_type_analysis import SOFTWOOD_SPECIES


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
    """fold RMSEが安定（低い）樹種を選択。"""
    sorted_species = sorted(fold_rmses, key=fold_rmses.get)

    if top_k is not None:
        return set(sorted_species[:top_k])

    if max_rmse is not None:
        return {sp for sp, r in fold_rmses.items() if r <= max_rmse}

    return set(fold_rmses.keys())


# ============================================================
# Pattern H: 針葉樹/広葉樹分類
# ============================================================
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
    """test樹種の針葉樹/広葉樹比率に合わせてtrain樹種を選択。"""
    n_test = len(test_species)
    if n_test == 0:
        return set()

    n_softwood_test = len(test_species & SOFTWOOD_SPECIES)
    softwood_ratio = n_softwood_test / n_test

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
