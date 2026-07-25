"""Issue #91: サイクル28 GLSW実装のユニットテスト"""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestGLSW:
    """GLSW (Generalized Least Squares Weighting) のテスト"""

    def setup_method(self):
        np.random.seed(42)
        self.X = np.random.randn(50, 20)
        self.y = np.random.randn(50) * 10 + 50

    def test_compute_glsw_shape(self):
        from scripts.modeling.run_issue91_cycle28 import compute_glsw
        G = compute_glsw(self.X, self.y, alpha=0.001, threshold=5.0)
        assert G.shape == (20, 20)

    def test_compute_glsw_symmetric(self):
        from scripts.modeling.run_issue91_cycle28 import compute_glsw
        G = compute_glsw(self.X, self.y, alpha=0.001, threshold=5.0)
        assert np.allclose(G, G.T, atol=1e-10)

    def test_compute_glsw_finite(self):
        from scripts.modeling.run_issue91_cycle28 import compute_glsw
        G = compute_glsw(self.X, self.y, alpha=0.001, threshold=5.0)
        assert np.all(np.isfinite(G))

    def test_apply_glsw_shape(self):
        from scripts.modeling.run_issue91_cycle28 import compute_glsw, apply_glsw
        G = compute_glsw(self.X, self.y, alpha=0.001, threshold=5.0)
        X_w = apply_glsw(self.X, G)
        assert X_w.shape == self.X.shape

    def test_apply_glsw_finite(self):
        from scripts.modeling.run_issue91_cycle28 import compute_glsw, apply_glsw
        G = compute_glsw(self.X, self.y, alpha=0.001, threshold=5.0)
        X_w = apply_glsw(self.X, G)
        assert np.all(np.isfinite(X_w))

    def test_glsw_no_pairs_returns_identity(self):
        """含水率差が大きくペアが見つからない場合、単位行列を返す"""
        from scripts.modeling.run_issue91_cycle28 import compute_glsw
        y_spread = np.arange(50) * 100.0  # 差が全て大きい
        G = compute_glsw(self.X, y_spread, alpha=0.001, threshold=0.01)
        assert np.allclose(G, np.eye(20))

    def test_glsw_different_thresholds(self):
        """閾値を変えると異なる重み行列になる"""
        from scripts.modeling.run_issue91_cycle28 import compute_glsw
        G1 = compute_glsw(self.X, self.y, threshold=5.0)
        G2 = compute_glsw(self.X, self.y, threshold=10.0)
        assert not np.allclose(G1, G2)


class TestPP:
    """前処理パイプライン pp() のテスト"""

    def setup_method(self):
        np.random.seed(42)
        self.X_tr = np.random.randn(40, 20) + 1.0
        self.X_te = np.random.randn(10, 20) + 1.0
        self.y_tr = np.random.randn(40) * 10 + 50
        self.g = np.array(["A"] * 20 + ["B"] * 20)

    def test_pp_osc1(self):
        from scripts.modeling.run_issue91_cycle28 import pp
        Xtr, Xte = pp(self.X_tr, self.X_te, self.g, self.y_tr, "OSC(1)")
        assert Xtr.shape == self.X_tr.shape
        assert Xte.shape == self.X_te.shape
        assert np.all(np.isfinite(Xtr))
        assert np.all(np.isfinite(Xte))

    def test_pp_osc2(self):
        from scripts.modeling.run_issue91_cycle28 import pp
        Xtr, Xte = pp(self.X_tr, self.X_te, self.g, self.y_tr, "OSC(2)")
        assert Xtr.shape == self.X_tr.shape
        assert np.all(np.isfinite(Xtr))

    def test_pp_glsw(self):
        from scripts.modeling.run_issue91_cycle28 import pp
        Xtr, Xte = pp(self.X_tr, self.X_te, self.g, self.y_tr, "GLSW")
        assert Xtr.shape == self.X_tr.shape
        assert Xte.shape == self.X_te.shape

    def test_pp_epo2(self):
        from scripts.modeling.run_issue91_cycle28 import pp
        Xtr, Xte = pp(self.X_tr, self.X_te, self.g, self.y_tr, "EPO(2)")
        assert Xtr.shape == self.X_tr.shape

    def test_pp_osc1_snv(self):
        from scripts.modeling.run_issue91_cycle28 import pp
        Xtr, Xte = pp(self.X_tr, self.X_te, self.g, self.y_tr, "OSC(1)+SNV")
        assert Xtr.shape == self.X_tr.shape
        assert np.all(np.isfinite(Xtr))
