"""Issue #32: Weighted Ensembleのテスト"""
import numpy as np
import pandas as pd
import pytest


class TestWeightedEnsemble:

    @pytest.fixture
    def dummy_oof(self):
        np.random.seed(42)
        n = 100
        y = np.random.randn(n) * 10 + 50
        oof = pd.DataFrame({
            "model_a": y + np.random.randn(n) * 5,
            "model_b": y + np.random.randn(n) * 8,
            "model_c": y + np.random.randn(n) * 6,
        })
        groups = np.random.choice(["A", "B", "C", "D"], n)
        return oof, y, groups

    def test_optimize_weights_sum_to_one(self, dummy_oof):
        """最適化された重みの合計が1であること"""
        from src.modeling.issue32_weighted_ensemble import optimize_weights
        oof, y, groups = dummy_oof
        weights = optimize_weights(oof.values, y)
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_optimize_weights_non_negative(self, dummy_oof):
        """重みが非負であること"""
        from src.modeling.issue32_weighted_ensemble import optimize_weights
        oof, y, groups = dummy_oof
        weights = optimize_weights(oof.values, y)
        assert all(w >= -1e-10 for w in weights)

    def test_weighted_avg_better_than_worst(self, dummy_oof):
        """重み付け平均が最悪の単体モデルより良いこと"""
        from src.modeling.issue32_weighted_ensemble import optimize_weights
        oof, y, groups = dummy_oof
        weights = optimize_weights(oof.values, y)
        weighted_pred = oof.values @ weights
        rmse_weighted = np.sqrt(np.mean((weighted_pred - y) ** 2))

        worst_rmse = max(
            np.sqrt(np.mean((oof[col].values - y) ** 2))
            for col in oof.columns
        )
        assert rmse_weighted < worst_rmse

    def test_evaluate_returns_dataframe(self, dummy_oof):
        """評価関数がDataFrameを返すこと"""
        from src.modeling.issue32_weighted_ensemble import evaluate_weighted_ensemble
        oof, y, groups = dummy_oof
        result = evaluate_weighted_ensemble(oof, y, groups)
        assert isinstance(result, pd.DataFrame)
        assert "method" in result.columns
        assert "rmse" in result.columns
        assert any("Weighted" in m for m in result["method"].tolist())
