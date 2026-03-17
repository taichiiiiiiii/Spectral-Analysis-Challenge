"""MMD-based Preprocessing Selection のテスト"""
import numpy as np
import pytest


@pytest.fixture
def domain_data():
    np.random.seed(42)
    X_source = np.random.randn(80, 50)
    X_target = np.random.randn(30, 50) + 1.0
    y_source = X_source[:, 0] * 2 + np.random.randn(80) * 0.5
    return X_source, X_target, y_source


class TestMMDSelection:

    def test_compute_mmd(self):
        """MMD計算が正しいこと"""
        from src.preprocessing.mmd_selection import compute_prediction_mmd
        np.random.seed(42)
        y1 = np.random.randn(50)
        y2 = np.random.randn(50) + 5.0  # 大きなシフト
        y3 = np.random.randn(50)  # シフトなし
        mmd_shift = compute_prediction_mmd(y1, y2)
        mmd_same = compute_prediction_mmd(y1, y3)
        assert mmd_shift > mmd_same  # シフトありの方がMMD大

    def test_select_best_preprocessing(self, domain_data):
        """最適前処理が選択されること"""
        from src.preprocessing.mmd_selection import select_best_preprocessing
        X_s, X_t, y_s = domain_data
        # ダミー前処理: 恒等変換とスケーリング
        preprocessings = {
            "identity": lambda X: X,
            "scale2": lambda X: X * 2,
            "scale05": lambda X: X * 0.5,
        }
        best_name, best_mmd = select_best_preprocessing(
            X_s, y_s, X_t, preprocessings, n_pls_components=2
        )
        assert best_name in preprocessings
        assert isinstance(best_mmd, float)
        assert best_mmd >= 0

    def test_returns_all_results(self, domain_data):
        """全前処理のMMDスコアが返されること"""
        from src.preprocessing.mmd_selection import select_best_preprocessing_detailed
        X_s, X_t, y_s = domain_data
        preprocessings = {
            "identity": lambda X: X,
            "scale2": lambda X: X * 2,
        }
        results = select_best_preprocessing_detailed(
            X_s, y_s, X_t, preprocessings, n_pls_components=2
        )
        assert len(results) == 2
        assert all("name" in r and "mmd" in r for r in results)
