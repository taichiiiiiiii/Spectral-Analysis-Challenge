"""Issue #31: 最終提出パイプラインのテスト"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path


class TestSubmissionPipeline:
    """提出パイプラインのテスト"""

    @pytest.fixture
    def dummy_data(self):
        np.random.seed(42)
        n_train, n_test, n_features = 100, 30, 50
        cols = [str(i) for i in range(n_features)]

        train_df = pd.DataFrame(np.random.randn(n_train, n_features), columns=cols)
        train_df["含水率"] = np.random.randn(n_train) * 10 + 50
        train_df["樹種"] = np.random.choice(["A", "B", "C", "D"], n_train)
        train_df["sample number"] = range(n_train)

        test_df = pd.DataFrame(np.random.randn(n_test, n_features), columns=cols)
        test_df["sample number"] = range(n_test)

        return train_df, test_df, cols

    def test_predict_returns_array(self, dummy_data):
        """predict関数がndarrayを返すこと"""
        from src.modeling.issue31_submission_pipeline import predict_ensemble
        train_df, test_df, spectral_cols = dummy_data
        preds = predict_ensemble(train_df, test_df, spectral_cols, n_pls_components=2)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(test_df)

    def test_predict_no_nan(self, dummy_data):
        """予測にNaNがないこと"""
        from src.modeling.issue31_submission_pipeline import predict_ensemble
        train_df, test_df, spectral_cols = dummy_data
        preds = predict_ensemble(train_df, test_df, spectral_cols, n_pls_components=2)
        assert not np.any(np.isnan(preds))

    def test_predict_reasonable_range(self, dummy_data):
        """予測値が物理的に妥当な範囲であること（含水率は0-200%程度）"""
        from src.modeling.issue31_submission_pipeline import predict_ensemble
        train_df, test_df, spectral_cols = dummy_data
        preds = predict_ensemble(train_df, test_df, spectral_cols, n_pls_components=2)
        assert np.all(preds >= 0)
        assert np.all(preds <= 200)

    def test_generate_submission_format(self, dummy_data, tmp_path):
        """提出ファイルのフォーマットが正しいこと"""
        from src.modeling.issue31_submission_pipeline import generate_submission
        train_df, test_df, spectral_cols = dummy_data
        output_path = tmp_path / "submission.csv"
        generate_submission(train_df, test_df, spectral_cols, output_path, n_pls_components=2)

        sub = pd.read_csv(output_path, header=None)
        assert len(sub) == len(test_df)
        assert sub.shape[1] == 2  # sample_number, prediction
