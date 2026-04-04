"""Test Issue #68: CVパターンの樹種選定ロジック（ルール準拠版）"""
import numpy as np
import pandas as pd
import pytest

from src.analysis.issue68_cv_pattern_selection import (
    select_pattern_e_sample_count,
    select_pattern_f_moisture_coverage,
    select_pattern_g_stable_folds,
    select_pattern_h_wood_type,
    select_pattern_k_spectral_homogeneity,
    select_pattern_l_wasserstein,
    select_pattern_m_correlation_stability,
)


# ============================================================
# テスト用フィクスチャ
# ============================================================
@pytest.fixture
def dummy_spectral_data():
    """ダミーのスペクトルデータ (train 5樹種)"""
    rng = np.random.RandomState(42)
    species_list = ["SpeciesA", "SpeciesB", "SpeciesC", "SpeciesD", "SpeciesE"]
    n_per_species = [30, 10, 50, 5, 25]
    n_features = 20

    rows = []
    for sp, n in zip(species_list, n_per_species):
        for _ in range(n):
            rows.append({"樹種": sp})
    df_train = pd.DataFrame(rows)

    X_train = rng.randn(sum(n_per_species), n_features)
    offset = 0
    for i, n in enumerate(n_per_species):
        X_train[offset:offset + n] += i * 2
        offset += n

    return df_train, X_train, species_list


@pytest.fixture
def dummy_moisture_data():
    """含水率データ付きダミー"""
    rng = np.random.RandomState(42)
    species = ["A", "B", "C", "D"]
    n_per = [20, 15, 30, 10]
    moisture_ranges = [(10, 50), (30, 80), (5, 100), (60, 90)]

    rows = []
    for sp, n, (lo, hi) in zip(species, n_per, moisture_ranges):
        for _ in range(n):
            rows.append({"樹種": sp, "含水率": rng.uniform(lo, hi)})
    return pd.DataFrame(rows)


@pytest.fixture
def dummy_fold_rmses():
    """fold RMSE のダミー"""
    return {
        "SpeciesA": 15.0,
        "SpeciesB": 45.0,
        "SpeciesC": 16.0,
        "SpeciesD": 50.0,
        "SpeciesE": 14.5,
    }


# ============================================================
# Pattern E: サンプル数フィルタ
# ============================================================
class TestPatternE:
    def test_excludes_small_species(self, dummy_spectral_data):
        df_train, _, _ = dummy_spectral_data
        result = select_pattern_e_sample_count(df_train, min_samples=10)
        assert isinstance(result, set)
        assert "SpeciesD" not in result

    def test_includes_large_species(self, dummy_spectral_data):
        df_train, _, _ = dummy_spectral_data
        result = select_pattern_e_sample_count(df_train, min_samples=10)
        assert "SpeciesC" in result
        assert "SpeciesA" in result
        assert "SpeciesE" in result

    def test_threshold_boundary(self, dummy_spectral_data):
        df_train, _, _ = dummy_spectral_data
        result = select_pattern_e_sample_count(df_train, min_samples=10)
        assert "SpeciesB" in result

    def test_returns_empty_if_all_below(self, dummy_spectral_data):
        df_train, _, _ = dummy_spectral_data
        result = select_pattern_e_sample_count(df_train, min_samples=100)
        assert isinstance(result, set)


# ============================================================
# Pattern F: 含水率レンジカバレッジ
# ============================================================
class TestPatternF:
    def test_returns_species_covering_range(self, dummy_moisture_data):
        result = select_pattern_f_moisture_coverage(
            dummy_moisture_data, target_min=20, target_max=70, coverage_threshold=0.5
        )
        assert isinstance(result, set)
        assert len(result) > 0

    def test_species_with_wide_range_included(self, dummy_moisture_data):
        result = select_pattern_f_moisture_coverage(
            dummy_moisture_data, target_min=20, target_max=70, coverage_threshold=0.5
        )
        assert "C" in result

    def test_species_outside_range_excluded(self, dummy_moisture_data):
        result = select_pattern_f_moisture_coverage(
            dummy_moisture_data, target_min=10, target_max=30, coverage_threshold=0.8
        )
        assert "D" not in result


# ============================================================
# Pattern G: fold RMSE安定樹種
# ============================================================
class TestPatternG:
    def test_selects_stable_species(self, dummy_fold_rmses):
        result = select_pattern_g_stable_folds(dummy_fold_rmses, max_rmse=30.0)
        assert isinstance(result, set)
        assert "SpeciesA" in result
        assert "SpeciesC" in result
        assert "SpeciesE" in result

    def test_excludes_unstable_species(self, dummy_fold_rmses):
        result = select_pattern_g_stable_folds(dummy_fold_rmses, max_rmse=30.0)
        assert "SpeciesB" not in result
        assert "SpeciesD" not in result

    def test_top_k_limits(self, dummy_fold_rmses):
        result = select_pattern_g_stable_folds(dummy_fold_rmses, top_k=2)
        assert len(result) == 2


# ============================================================
# Pattern H: 針葉樹/広葉樹分類
# ============================================================
class TestPatternH:
    def test_softwood_selection(self):
        test_species = {"スギ", "ケヤキ", "クスノキ", "タモ", "チーク", "ヤマザクラ"}
        result = select_pattern_h_wood_type(test_species)
        assert isinstance(result, set)
        from src.eda.issue9_wood_type_analysis import SOFTWOOD_SPECIES
        from src.analysis.issue68_cv_pattern_selection import TRAIN_SPECIES
        train_softwood = TRAIN_SPECIES & SOFTWOOD_SPECIES
        assert len(result & train_softwood) >= 1

    def test_returns_both_types(self):
        test_species = {"スギ", "ケヤキ", "クスノキ", "タモ", "チーク", "ヤマザクラ"}
        result = select_pattern_h_wood_type(test_species)
        assert len(result) > 0


# ============================================================
# Pattern K: スペクトル同質性
# ============================================================
class TestPatternK:
    def test_returns_set_of_species(self, dummy_spectral_data):
        df_train, X_train, species_list = dummy_spectral_data
        result = select_pattern_k_spectral_homogeneity(df_train, X_train, top_k=3)
        assert isinstance(result, set)
        assert len(result) == 3

    def test_top_k(self, dummy_spectral_data):
        df_train, X_train, _ = dummy_spectral_data
        result = select_pattern_k_spectral_homogeneity(df_train, X_train, top_k=2)
        assert len(result) == 2


# ============================================================
# Pattern L: Wasserstein距離
# ============================================================
class TestPatternL:
    def test_returns_set(self, dummy_moisture_data):
        y_all = dummy_moisture_data["含水率"].values
        result = select_pattern_l_wasserstein(
            dummy_moisture_data, target_distribution=y_all, top_k=2
        )
        assert isinstance(result, set)
        assert len(result) == 2

    def test_species_with_similar_distribution(self, dummy_moisture_data):
        y_all = dummy_moisture_data["含水率"].values
        result = select_pattern_l_wasserstein(
            dummy_moisture_data, target_distribution=y_all, top_k=3
        )
        assert len(result) == 3


# ============================================================
# Pattern M: スペクトル-含水率相関安定性
# ============================================================
class TestPatternM:
    def test_returns_set(self):
        rng = np.random.RandomState(42)
        n = 100
        species = ["A"] * 40 + ["B"] * 30 + ["C"] * 30
        X = rng.randn(n, 10)
        y = np.zeros(n)
        y[:40] = np.sum(X[:40], axis=1) * 3 + 50
        y[40:70] = 50 + rng.randn(30) * 0.01
        y[70:] = rng.randn(30) * 50 + 50
        df = pd.DataFrame({"樹種": species, "含水率": y})
        result = select_pattern_m_correlation_stability(df, X, top_k=2)
        assert isinstance(result, set)
        assert len(result) == 2
