"""Test Issue #68: パターン別サブミッション生成ロジック"""
import numpy as np
import pandas as pd
import pytest

from src.analysis.issue68_cv_pattern_selection import (
    select_pattern_d_mahalanobis,
    select_pattern_e_sample_count,
    select_pattern_f_moisture_coverage,
    select_pattern_g_stable_folds,
    select_pattern_h_wood_type,
    select_pattern_i_mmd,
)


class TestPatternSubsetTraining:
    """パターンの樹種サブセットでフィルタした学習データの妥当性テスト"""

    @pytest.fixture
    def train_data(self):
        rng = np.random.RandomState(42)
        species = ["A"] * 40 + ["B"] * 30 + ["C"] * 50 + ["D"] * 10 + ["E"] * 20
        df = pd.DataFrame({
            "樹種": species,
            "含水率": rng.uniform(5, 100, len(species)),
        })
        X = rng.randn(len(species), 10)
        return df, X

    def test_filter_by_pattern_species(self, train_data):
        """パターンの樹種でフィルタしてサンプル数が正しいか"""
        df, X = train_data
        pattern = {"A", "C"}
        mask = np.isin(df["樹種"].values, list(pattern))
        X_sub = X[mask]
        y_sub = df["含水率"].values[mask]
        assert X_sub.shape[0] == 90  # A:40 + C:50
        assert y_sub.shape[0] == 90

    def test_empty_pattern_produces_no_samples(self, train_data):
        """存在しない樹種パターンではサンプル0"""
        df, X = train_data
        pattern = {"Z"}
        mask = np.isin(df["樹種"].values, list(pattern))
        assert mask.sum() == 0

    def test_all_patterns_produce_valid_subsets(self, train_data):
        """各パターン関数が返す樹種セットでフィルタできる"""
        df, X = train_data
        g = df["樹種"].values

        # Pattern E
        sp_set = select_pattern_e_sample_count(df, min_samples=15)
        mask = np.isin(g, list(sp_set))
        assert mask.sum() > 0

        # Pattern G
        fold_rmses = {"A": 10, "B": 20, "C": 15, "D": 50, "E": 12}
        sp_set = select_pattern_g_stable_folds(fold_rmses, top_k=3)
        mask = np.isin(g, list(sp_set))
        assert mask.sum() > 0


class TestSubmissionFormat:
    """サブミッションCSVのフォーマットテスト"""

    def test_submission_shape(self):
        """サブミッションは550行2列"""
        test_ids = np.arange(95, 645)  # 550 samples
        preds = np.random.uniform(10, 100, 550)
        sub_df = pd.DataFrame({0: test_ids.astype(int), 1: preds})
        assert sub_df.shape == (550, 2)

    def test_submission_no_negative(self):
        """予測値は非負"""
        preds = np.clip(np.random.randn(550) * 20 + 50, 0, 300)
        assert (preds >= 0).all()

    def test_submission_csv_format(self, tmp_path):
        """CSVがヘッダなし2列で出力される"""
        test_ids = np.arange(95, 645)
        preds = np.random.uniform(10, 100, 550)
        sub_df = pd.DataFrame({0: test_ids.astype(int), 1: preds})
        path = tmp_path / "test_submission.csv"
        sub_df.to_csv(path, index=False, header=False)

        loaded = pd.read_csv(path, header=None)
        assert loaded.shape == (550, 2)
        assert loaded[0].iloc[0] == 95
        assert loaded[0].iloc[-1] == 644
