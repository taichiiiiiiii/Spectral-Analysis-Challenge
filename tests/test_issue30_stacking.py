"""Issue #30: スタッキングアンサンブルのテスト"""
import numpy as np
import pandas as pd
import pytest


class TestStackingEnsemble:
    """スタッキングアンサンブルのユニットテスト"""

    @pytest.fixture
    def dummy_df(self):
        np.random.seed(42)
        n_samples = 100
        n_features = 50
        df = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[str(i) for i in range(n_features)],
        )
        df["含水率"] = np.random.randn(n_samples) * 10 + 50
        df["樹種"] = np.random.choice(["A", "B", "C", "D"], n_samples)
        return df, [str(i) for i in range(n_features)]

    def test_generate_oof_predictions_shape(self, dummy_df):
        """OOF予測の形状が正しいこと"""
        from src.modeling.issue30_stacking import generate_oof_predictions
        df, spectral_cols = dummy_df
        oof = generate_oof_predictions(df, spectral_cols, n_pls_components=2)
        assert isinstance(oof, pd.DataFrame)
        assert len(oof) == len(df)
        assert oof.shape[1] >= 2  # At least 2 base models

    def test_oof_no_nan(self, dummy_df):
        """OOF予測にNaNがないこと"""
        from src.modeling.issue30_stacking import generate_oof_predictions
        df, spectral_cols = dummy_df
        oof = generate_oof_predictions(df, spectral_cols, n_pls_components=2)
        assert not oof.isnull().any().any()

    def test_stacking_returns_dataframe(self, dummy_df):
        """スタッキング評価がDataFrameを返すこと"""
        from src.modeling.issue30_stacking import evaluate_stacking
        df, spectral_cols = dummy_df
        result = evaluate_stacking(df, spectral_cols, n_pls_components=2)
        assert isinstance(result, pd.DataFrame)
        assert "method" in result.columns
        assert "rmse" in result.columns

    def test_stacking_includes_ensemble(self, dummy_df):
        """結果にStackingモデルが含まれること"""
        from src.modeling.issue30_stacking import evaluate_stacking
        df, spectral_cols = dummy_df
        result = evaluate_stacking(df, spectral_cols, n_pls_components=2)
        methods = result["method"].tolist()
        assert any("Stacking" in m for m in methods)
