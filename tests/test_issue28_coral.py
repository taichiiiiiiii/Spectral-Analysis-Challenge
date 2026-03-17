"""Issue #28: CORAL（Correlation Alignment）のテスト"""
import numpy as np
import pandas as pd
import pytest


class TestCORAL:
    """CORAL変換のユニットテスト"""

    def test_coral_transform_shape(self):
        """CORAL変換後のデータ形状が入力と同じであること"""
        from src.modeling.issue28_domain_adaptation import coral_transform

        np.random.seed(42)
        X_source = np.random.randn(100, 50)
        X_target = np.random.randn(30, 50)
        X_aligned = coral_transform(X_source, X_target)
        assert X_aligned.shape == X_source.shape

    def test_coral_covariance_alignment(self):
        """CORAL変換後のソース共分散がターゲットに近づくこと"""
        from src.modeling.issue28_domain_adaptation import coral_transform

        np.random.seed(42)
        X_source = np.random.randn(100, 10) @ np.random.randn(10, 10)
        X_target = np.random.randn(50, 10) @ np.random.randn(10, 10)

        X_aligned = coral_transform(X_source, X_target)

        cov_aligned = np.cov(X_aligned.T)
        cov_target = np.cov(X_target.T)

        # 変換後の共分散がターゲットに近づくこと（完全一致は不要）
        diff_before = np.linalg.norm(np.cov(X_source.T) - cov_target, "fro")
        diff_after = np.linalg.norm(cov_aligned - cov_target, "fro")
        assert diff_after < diff_before

    def test_coral_no_nan(self):
        """CORAL変換後にNaNが含まれないこと"""
        from src.modeling.issue28_domain_adaptation import coral_transform

        np.random.seed(42)
        X_source = np.random.randn(50, 20)
        X_target = np.random.randn(30, 20)
        X_aligned = coral_transform(X_source, X_target)
        assert not np.any(np.isnan(X_aligned))

    def test_coral_identity_case(self):
        """ソースとターゲットが同じ分布の場合、変換が大きく変わらないこと"""
        from src.modeling.issue28_domain_adaptation import coral_transform

        np.random.seed(42)
        X = np.random.randn(200, 10)
        X_source = X[:100]
        X_target = X[100:]
        X_aligned = coral_transform(X_source, X_target)
        # 同じ分布から来ているので、変換前後で大きく変わらない
        assert np.corrcoef(X_source.ravel(), X_aligned.ravel())[0, 1] > 0.8


class TestPseudoLabeling:
    """擬似ラベルのテスト"""

    def test_pseudo_label_shape(self):
        """擬似ラベルの形状が正しいこと"""
        from src.modeling.issue28_domain_adaptation import generate_pseudo_labels
        from sklearn.cross_decomposition import PLSRegression

        np.random.seed(42)
        X_train = np.random.randn(100, 50)
        y_train = np.random.randn(100)
        X_test = np.random.randn(30, 50)

        pseudo_labels = generate_pseudo_labels(X_train, y_train, X_test)
        assert pseudo_labels.shape == (30,)

    def test_pseudo_label_reasonable_range(self):
        """擬似ラベルが妥当な範囲にあること"""
        from src.modeling.issue28_domain_adaptation import generate_pseudo_labels

        np.random.seed(42)
        X_train = np.random.randn(100, 50)
        y_train = np.random.randn(100) * 10 + 50  # mean=50
        X_test = np.random.randn(30, 50)

        pseudo_labels = generate_pseudo_labels(X_train, y_train, X_test)
        # 予測値がy_trainの範囲から大きく外れないこと
        assert pseudo_labels.min() > y_train.min() - 3 * y_train.std()
        assert pseudo_labels.max() < y_train.max() + 3 * y_train.std()


class TestEvaluateDomainAdaptation:
    """ドメイン適応の評価関数テスト"""

    def test_evaluate_returns_dataframe(self):
        """評価関数がDataFrameを返すこと"""
        from src.modeling.issue28_domain_adaptation import evaluate_domain_adaptation

        np.random.seed(42)
        n_samples = 100
        n_features = 50
        df = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[str(i) for i in range(n_features)],
        )
        df["含水率"] = np.random.randn(n_samples) * 10 + 50
        df["樹種"] = np.random.choice(["A", "B", "C", "D"], n_samples)

        spectral_cols = [str(i) for i in range(n_features)]
        result = evaluate_domain_adaptation(df, spectral_cols, n_pls_components=2)

        assert isinstance(result, pd.DataFrame)
        assert "method" in result.columns
        assert "rmse" in result.columns
        assert len(result) > 0
