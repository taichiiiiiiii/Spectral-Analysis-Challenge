"""JDA (Joint Distribution Adaptation) のテスト"""
import numpy as np
import pytest


@pytest.fixture
def domain_data():
    """ソース・ターゲットドメインのダミーデータ（回帰用）"""
    np.random.seed(42)
    X_source = np.random.randn(80, 50)
    X_target = np.random.randn(30, 50) + 2.0
    y_source = X_source[:, 0] * 2 + np.random.randn(80) * 0.5
    return X_source, X_target, y_source


class TestJDA:

    def test_output_shape(self, domain_data):
        """変換後の形状が正しいこと"""
        from src.preprocessing.issue48_jda import jda_transform
        X_s, X_t, y_s = domain_data
        Z_s, Z_t = jda_transform(X_s, X_t, y_s, n_components=5)
        assert Z_s.shape == (80, 5)
        assert Z_t.shape == (30, 5)

    def test_no_nan(self, domain_data):
        """変換後にNaNがないこと"""
        from src.preprocessing.issue48_jda import jda_transform
        X_s, X_t, y_s = domain_data
        Z_s, Z_t = jda_transform(X_s, X_t, y_s, n_components=5)
        assert not np.any(np.isnan(Z_s))
        assert not np.any(np.isnan(Z_t))

    def test_reduces_domain_distance(self, domain_data):
        """変換後にドメイン間の平均距離が縮まること"""
        from src.preprocessing.issue48_jda import jda_transform
        X_s, X_t, y_s = domain_data
        dist_before = np.linalg.norm(X_s.mean(axis=0) - X_t.mean(axis=0))
        Z_s, Z_t = jda_transform(X_s, X_t, y_s, n_components=10)
        dist_after = np.linalg.norm(Z_s.mean(axis=0) - Z_t.mean(axis=0))
        assert dist_after < dist_before

    def test_iterations(self, domain_data):
        """反復回数を変えても動作すること"""
        from src.preprocessing.issue48_jda import jda_transform
        X_s, X_t, y_s = domain_data
        Z_s1, _ = jda_transform(X_s, X_t, y_s, n_components=5, n_iterations=1)
        Z_s3, _ = jda_transform(X_s, X_t, y_s, n_components=5, n_iterations=3)
        # 反復回数が異なると結果が異なる
        assert not np.allclose(Z_s1, Z_s3)

    def test_n_bins(self, domain_data):
        """異なるビン数で動作すること"""
        from src.preprocessing.issue48_jda import jda_transform
        X_s, X_t, y_s = domain_data
        for n_bins in [3, 5, 10]:
            Z_s, Z_t = jda_transform(X_s, X_t, y_s, n_components=5, n_bins=n_bins)
            assert Z_s.shape == (80, 5)
