"""TCA (Transfer Component Analysis) のテスト"""
import numpy as np
import pytest


@pytest.fixture
def domain_data():
    """ソース・ターゲットドメインのダミーデータ"""
    np.random.seed(42)
    # ソース: mean=0, ターゲット: mean=2 (ドメインシフトあり)
    X_source = np.random.randn(80, 50)
    X_target = np.random.randn(30, 50) + 2.0
    return X_source, X_target


class TestTCA:

    def test_output_shape(self, domain_data):
        """変換後の形状が正しいこと"""
        from src.preprocessing.issue38_tca import tca_transform
        X_s, X_t = domain_data
        Z_s, Z_t = tca_transform(X_s, X_t, n_components=5)
        assert Z_s.shape == (80, 5)
        assert Z_t.shape == (30, 5)

    def test_no_nan(self, domain_data):
        """変換後にNaNがないこと"""
        from src.preprocessing.issue38_tca import tca_transform
        X_s, X_t = domain_data
        Z_s, Z_t = tca_transform(X_s, X_t, n_components=5)
        assert not np.any(np.isnan(Z_s))
        assert not np.any(np.isnan(Z_t))

    def test_different_n_components(self, domain_data):
        """n_componentsを変えても動作すること"""
        from src.preprocessing.issue38_tca import tca_transform
        X_s, X_t = domain_data
        for n in [2, 5, 10]:
            Z_s, Z_t = tca_transform(X_s, X_t, n_components=n)
            assert Z_s.shape[1] == n
            assert Z_t.shape[1] == n

    def test_reduces_domain_distance(self, domain_data):
        """変換後にドメイン間の平均距離が縮まること"""
        from src.preprocessing.issue38_tca import tca_transform
        X_s, X_t = domain_data
        # 変換前のドメイン間距離
        dist_before = np.linalg.norm(X_s.mean(axis=0) - X_t.mean(axis=0))
        Z_s, Z_t = tca_transform(X_s, X_t, n_components=10)
        # 変換後のドメイン間距離
        dist_after = np.linalg.norm(Z_s.mean(axis=0) - Z_t.mean(axis=0))
        assert dist_after < dist_before

    def test_kernel_parameter(self, domain_data):
        """異なるカーネルパラメータで動作すること"""
        from src.preprocessing.issue38_tca import tca_transform
        X_s, X_t = domain_data
        Z_s1, Z_t1 = tca_transform(X_s, X_t, n_components=5, kernel="rbf", gamma=0.1)
        Z_s2, Z_t2 = tca_transform(X_s, X_t, n_components=5, kernel="rbf", gamma=1.0)
        # 異なるgammaで異なる結果になること
        assert not np.allclose(Z_s1, Z_s2)

    def test_linear_kernel(self, domain_data):
        """線形カーネルでも動作すること"""
        from src.preprocessing.issue38_tca import tca_transform
        X_s, X_t = domain_data
        Z_s, Z_t = tca_transform(X_s, X_t, n_components=5, kernel="linear")
        assert Z_s.shape == (80, 5)
        assert Z_t.shape == (30, 5)
