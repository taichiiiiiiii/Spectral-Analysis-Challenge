"""Issue #97: Subspace Alignment + CORAL ドメイン適応のユニットテスト"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest


# ============================================================
# CORAL transform テスト
# ============================================================

class TestCoralTransform:
    """coral_transform のユニットテスト"""

    def _import_coral(self):
        from scripts.modeling.run_issue97_cycle5_sa_coral import coral_transform
        return coral_transform

    def test_output_shapes(self):
        """出力の形状がソース/ターゲットと同じであること"""
        coral_transform = self._import_coral()
        rng = np.random.RandomState(42)
        X_s = rng.randn(50, 10)
        X_t = rng.randn(30, 10)
        X_s_out, X_t_out = coral_transform(X_s, X_t)
        assert X_s_out.shape == X_s.shape
        assert X_t_out.shape == X_t.shape

    def test_target_unchanged(self):
        """ターゲットはコピーされるだけで変換されないこと"""
        coral_transform = self._import_coral()
        rng = np.random.RandomState(42)
        X_s = rng.randn(50, 10)
        X_t = rng.randn(30, 10)
        _, X_t_out = coral_transform(X_s, X_t)
        np.testing.assert_array_almost_equal(X_t, X_t_out)

    def test_source_is_transformed(self):
        """ソースが実際に変換されていること（同一でないこと）"""
        coral_transform = self._import_coral()
        rng = np.random.RandomState(42)
        X_s = rng.randn(50, 10)
        X_t = rng.randn(30, 10) + 5  # ドメインシフトを追加
        X_s_out, _ = coral_transform(X_s, X_t)
        assert not np.allclose(X_s, X_s_out)

    def test_identical_domains_identity(self):
        """同じ分布の場合、変換は近似的に恒等変換であること"""
        coral_transform = self._import_coral()
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        X_s = X[:50]
        X_t = X[50:]
        X_s_out, _ = coral_transform(X_s, X_t, reg=1e-3)
        # 同じ分布なら変換後も大きく変わらないはず
        ratio = np.std(X_s_out) / np.std(X_s)
        assert 0.3 < ratio < 3.0, f"ratio={ratio}"

    def test_no_nan_inf(self):
        """NaN/Infが出力に含まれないこと"""
        coral_transform = self._import_coral()
        rng = np.random.RandomState(42)
        X_s = rng.randn(40, 8)
        X_t = rng.randn(20, 8) * 0.1 + 3
        X_s_out, X_t_out = coral_transform(X_s, X_t)
        assert not np.any(np.isnan(X_s_out))
        assert not np.any(np.isinf(X_s_out))
        assert not np.any(np.isnan(X_t_out))
        assert not np.any(np.isinf(X_t_out))

    def test_regularization_effect(self):
        """正則化パラメータが機能すること（小さいサンプルでも安定）"""
        coral_transform = self._import_coral()
        rng = np.random.RandomState(42)
        # サンプル数 < 次元数（rank deficient）
        X_s = rng.randn(5, 10)
        X_t = rng.randn(3, 10)
        X_s_out, X_t_out = coral_transform(X_s, X_t, reg=1.0)
        assert not np.any(np.isnan(X_s_out))
        assert not np.any(np.isinf(X_s_out))


# ============================================================
# Subspace Alignment テスト
# ============================================================

class TestSubspaceAlign:
    """subspace_align のユニットテスト"""

    def test_output_shapes(self):
        from src.preprocessing.issue44_subspace_alignment import subspace_align
        rng = np.random.RandomState(42)
        X_s = rng.randn(50, 20)
        X_t = rng.randn(30, 20)
        Z_s, Z_t = subspace_align(X_s, X_t, n_components=5)
        assert Z_s.shape == (50, 5)
        assert Z_t.shape == (30, 5)

    def test_no_nan_inf(self):
        from src.preprocessing.issue44_subspace_alignment import subspace_align
        rng = np.random.RandomState(42)
        X_s = rng.randn(50, 20)
        X_t = rng.randn(30, 20)
        Z_s, Z_t = subspace_align(X_s, X_t, n_components=5)
        assert not np.any(np.isnan(Z_s))
        assert not np.any(np.isinf(Z_s))


# ============================================================
# predict_sa / predict_coral パイプライン テスト
# ============================================================

class TestPredictSA:
    """predict_sa のパイプラインテスト"""

    def test_predict_sa_ridge(self):
        from scripts.modeling.run_issue97_cycle5_sa_coral import predict_sa
        rng = np.random.RandomState(42)
        X_tr = rng.randn(60, 30)
        X_te = rng.randn(10, 30)
        y_tr = rng.rand(60) * 100
        groups_tr = np.array(["A"] * 20 + ["B"] * 20 + ["C"] * 20)
        cfg = {"pp": "raw", "sa_nc": 5, "reg": "Ridge", "tf": "raw", "alpha": 1.0}
        pred = predict_sa(X_tr, X_te, y_tr, groups_tr, cfg)
        assert pred.shape == (10,)
        assert not np.any(np.isnan(pred))

    def test_predict_sa_pls_sqrt(self):
        from scripts.modeling.run_issue97_cycle5_sa_coral import predict_sa
        rng = np.random.RandomState(42)
        X_tr = rng.randn(60, 30)
        X_te = rng.randn(10, 30)
        y_tr = rng.rand(60) * 100 + 1
        groups_tr = np.array(["A"] * 20 + ["B"] * 20 + ["C"] * 20)
        cfg = {"pp": "raw", "sa_nc": 5, "reg": "PLS", "tf": "sqrt", "pls_nc": 3}
        pred = predict_sa(X_tr, X_te, y_tr, groups_tr, cfg)
        assert pred.shape == (10,)
        assert np.all(pred >= 0), "sqrt変換後の予測は非負であるべき"

    def test_predict_sa_huber(self):
        from scripts.modeling.run_issue97_cycle5_sa_coral import predict_sa
        rng = np.random.RandomState(42)
        X_tr = rng.randn(60, 30)
        X_te = rng.randn(10, 30)
        y_tr = rng.rand(60) * 100
        groups_tr = np.array(["A"] * 20 + ["B"] * 20 + ["C"] * 20)
        cfg = {"pp": "raw", "sa_nc": 5, "reg": "Huber", "tf": "raw"}
        pred = predict_sa(X_tr, X_te, y_tr, groups_tr, cfg)
        assert pred.shape == (10,)
        assert not np.any(np.isnan(pred))


class TestPredictCORAL:
    """predict_coral のパイプラインテスト"""

    def test_predict_coral_ridge(self):
        from scripts.modeling.run_issue97_cycle5_sa_coral import predict_coral
        rng = np.random.RandomState(42)
        X_tr = rng.randn(60, 50)
        X_te = rng.randn(10, 50)
        y_tr = rng.rand(60) * 100
        groups_tr = np.array(["A"] * 20 + ["B"] * 20 + ["C"] * 20)
        cfg = {"pp": "raw", "pca_dim": 10, "reg": "Ridge", "tf": "raw", "alpha": 1.0}
        pred = predict_coral(X_tr, X_te, y_tr, groups_tr, cfg)
        assert pred.shape == (10,)
        assert not np.any(np.isnan(pred))

    def test_predict_coral_pls_sqrt(self):
        from scripts.modeling.run_issue97_cycle5_sa_coral import predict_coral
        rng = np.random.RandomState(42)
        X_tr = rng.randn(60, 50)
        X_te = rng.randn(10, 50)
        y_tr = rng.rand(60) * 100 + 1
        groups_tr = np.array(["A"] * 20 + ["B"] * 20 + ["C"] * 20)
        cfg = {"pp": "raw", "pca_dim": 10, "reg": "PLS", "tf": "sqrt", "pls_nc": 4}
        pred = predict_coral(X_tr, X_te, y_tr, groups_tr, cfg)
        assert pred.shape == (10,)
        assert np.all(pred >= 0)

    def test_predict_coral_snv(self):
        from scripts.modeling.run_issue97_cycle5_sa_coral import predict_coral
        rng = np.random.RandomState(42)
        X_tr = rng.randn(60, 50) + 1  # SNVが0除算しないように
        X_te = rng.randn(10, 50) + 1
        y_tr = rng.rand(60) * 100
        groups_tr = np.array(["A"] * 20 + ["B"] * 20 + ["C"] * 20)
        cfg = {"pp": "SNV", "pca_dim": 10, "reg": "Ridge", "tf": "raw"}
        pred = predict_coral(X_tr, X_te, y_tr, groups_tr, cfg)
        assert pred.shape == (10,)
        assert not np.any(np.isnan(pred))
