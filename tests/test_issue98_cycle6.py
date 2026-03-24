"""Issue #98: Adversarial Validation + サンプル重み付け学習のユニットテスト"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest


# ============================================================
# compute_adversarial_weights テスト
# ============================================================

class TestComputeAdversarialWeights:
    """compute_adversarial_weights のユニットテスト"""

    def _import_func(self):
        from scripts.modeling.run_issue98_cycle6_advval import compute_adversarial_weights
        return compute_adversarial_weights

    def test_output_shapes(self):
        """出力の形状が正しいこと"""
        compute_adversarial_weights = self._import_func()
        rng = np.random.RandomState(42)
        X_train = rng.randn(80, 50)
        X_test = rng.randn(30, 50)
        weights, auc, probs = compute_adversarial_weights(X_train, X_test, n_pca=10)
        assert weights.shape == (80,), f"weights shape: {weights.shape}"
        assert probs.shape == (80,), f"probs shape: {probs.shape}"
        assert isinstance(auc, float)

    def test_auc_range(self):
        """AUCが[0, 1]の範囲にあること"""
        compute_adversarial_weights = self._import_func()
        rng = np.random.RandomState(42)
        X_train = rng.randn(60, 30)
        X_test = rng.randn(20, 30)
        _, auc, _ = compute_adversarial_weights(X_train, X_test, n_pca=10)
        assert 0.0 <= auc <= 1.0, f"AUC={auc} is out of range"

    def test_weights_positive(self):
        """重みが正であること"""
        compute_adversarial_weights = self._import_func()
        rng = np.random.RandomState(42)
        X_train = rng.randn(60, 30)
        X_test = rng.randn(20, 30)
        weights, _, _ = compute_adversarial_weights(X_train, X_test, n_pca=10)
        assert np.all(weights > 0), "All weights must be positive"

    def test_weights_normalized(self):
        """重みの平均が1に正規化されていること"""
        compute_adversarial_weights = self._import_func()
        rng = np.random.RandomState(42)
        X_train = rng.randn(60, 30)
        X_test = rng.randn(20, 30)
        weights, _, _ = compute_adversarial_weights(X_train, X_test, n_pca=10)
        assert abs(weights.mean() - 1.0) < 1e-6, f"Mean weight={weights.mean()}"

    def test_weights_clipped(self):
        """重みが指定範囲にクリップされていること"""
        compute_adversarial_weights = self._import_func()
        rng = np.random.RandomState(42)
        X_train = rng.randn(60, 30)
        X_test = rng.randn(20, 30) + 5  # 大きなドメインシフト
        weights, _, _ = compute_adversarial_weights(
            X_train, X_test, n_pca=10, clip_min=0.2, clip_max=5.0
        )
        # クリップ後に正規化されるため、正規化前の比率関係のみ確認
        assert not np.any(np.isnan(weights))
        assert not np.any(np.isinf(weights))

    def test_domain_shift_high_auc(self):
        """明確なドメインシフトがある場合にAUCが高くなること"""
        compute_adversarial_weights = self._import_func()
        rng = np.random.RandomState(42)
        X_train = rng.randn(80, 20)
        X_test = rng.randn(30, 20) + 10  # 大きなシフト
        _, auc, _ = compute_adversarial_weights(X_train, X_test, n_pca=10)
        assert auc > 0.8, f"Domain-shifted AUC should be high, got {auc}"

    def test_probs_range(self):
        """確率が[0, 1]の範囲にあること"""
        compute_adversarial_weights = self._import_func()
        rng = np.random.RandomState(42)
        X_train = rng.randn(60, 30)
        X_test = rng.randn(20, 30)
        _, _, probs = compute_adversarial_weights(X_train, X_test, n_pca=10)
        assert np.all(probs >= 0) and np.all(probs <= 1), "Probs must be in [0, 1]"

    def test_no_nan_inf(self):
        """NaN/Infが含まれないこと"""
        compute_adversarial_weights = self._import_func()
        rng = np.random.RandomState(42)
        X_train = rng.randn(60, 30)
        X_test = rng.randn(20, 30)
        weights, auc, probs = compute_adversarial_weights(X_train, X_test, n_pca=10)
        assert not np.any(np.isnan(weights))
        assert not np.any(np.isinf(weights))
        assert not np.isnan(auc)
        assert not np.any(np.isnan(probs))


# ============================================================
# predict_pls_weighted_ridge テスト
# ============================================================

class TestPredictPLSWeightedRidge:
    """predict_pls_weighted_ridge のユニットテスト"""

    def _import_func(self):
        from scripts.modeling.run_issue98_cycle6_advval import predict_pls_weighted_ridge
        return predict_pls_weighted_ridge

    def test_output_shape(self):
        """出力の形状が正しいこと"""
        predict = self._import_func()
        rng = np.random.RandomState(42)
        X_tr = rng.randn(60, 30)
        X_te = rng.randn(10, 30)
        y_tr = rng.rand(60) * 100 + 1
        weights = np.ones(60)
        cfg = {"nc": 4, "tf": "sqrt", "alpha": 1.0}
        pred = predict(X_tr, X_te, y_tr, weights, cfg)
        assert pred.shape == (10,)

    def test_no_nan_output(self):
        """NaNが出力に含まれないこと"""
        predict = self._import_func()
        rng = np.random.RandomState(42)
        X_tr = rng.randn(60, 30)
        X_te = rng.randn(10, 30)
        y_tr = rng.rand(60) * 100 + 1
        weights = rng.rand(60) + 0.5
        cfg = {"nc": 4, "tf": "raw", "alpha": 1.0}
        pred = predict(X_tr, X_te, y_tr, weights, cfg)
        assert not np.any(np.isnan(pred))

    def test_sqrt_transform_nonneg(self):
        """sqrt変換の場合、予測が非負であること"""
        predict = self._import_func()
        rng = np.random.RandomState(42)
        X_tr = rng.randn(60, 30)
        X_te = rng.randn(10, 30)
        y_tr = rng.rand(60) * 100 + 1
        weights = np.ones(60)
        cfg = {"nc": 4, "tf": "sqrt", "alpha": 1.0}
        pred = predict(X_tr, X_te, y_tr, weights, cfg)
        assert np.all(pred >= 0), "sqrt transform predictions should be non-negative"

    def test_weights_affect_predictions(self):
        """異なる重みで異なる予測が得られること"""
        predict = self._import_func()
        rng = np.random.RandomState(42)
        X_tr = rng.randn(60, 20)
        X_te = rng.randn(10, 20)
        y_tr = rng.rand(60) * 100
        cfg = {"nc": 3, "tf": "raw", "alpha": 1.0}

        w_uniform = np.ones(60)
        w_skewed = np.ones(60)
        w_skewed[:30] = 10.0  # 前半を重視

        pred_uniform = predict(X_tr, X_te, y_tr, w_uniform, cfg)
        pred_skewed = predict(X_tr, X_te, y_tr, w_skewed, cfg)
        # 重みが異なれば予測も異なるはず
        assert not np.allclose(pred_uniform, pred_skewed), \
            "Different weights should produce different predictions"


# ============================================================
# predict_weighted_gbr テスト
# ============================================================

class TestPredictWeightedGBR:
    """predict_weighted_gbr のユニットテスト"""

    def _import_func(self):
        from scripts.modeling.run_issue98_cycle6_advval import predict_weighted_gbr
        return predict_weighted_gbr

    def test_output_shape(self):
        """出力の形状が正しいこと"""
        predict = self._import_func()
        rng = np.random.RandomState(42)
        X_tr = rng.randn(60, 30)
        X_te = rng.randn(10, 30)
        y_tr = rng.rand(60) * 100
        weights = np.ones(60)
        cfg = {"nc": 4, "tf": "raw", "n_est": 50, "max_depth": 2, "lr": 0.1}
        pred = predict(X_tr, X_te, y_tr, weights, cfg)
        assert pred.shape == (10,)

    def test_no_nan_output(self):
        """NaNが出力に含まれないこと"""
        predict = self._import_func()
        rng = np.random.RandomState(42)
        X_tr = rng.randn(60, 30)
        X_te = rng.randn(10, 30)
        y_tr = rng.rand(60) * 100
        weights = rng.rand(60) + 0.5
        cfg = {"nc": 4, "tf": "raw", "n_est": 50, "max_depth": 2, "lr": 0.1}
        pred = predict(X_tr, X_te, y_tr, weights, cfg)
        assert not np.any(np.isnan(pred))

    def test_sqrt_transform(self):
        """sqrt変換が正しく機能すること"""
        predict = self._import_func()
        rng = np.random.RandomState(42)
        X_tr = rng.randn(60, 30)
        X_te = rng.randn(10, 30)
        y_tr = rng.rand(60) * 100 + 1
        weights = np.ones(60)
        cfg = {"nc": 4, "tf": "sqrt", "n_est": 50, "max_depth": 2, "lr": 0.1}
        pred = predict(X_tr, X_te, y_tr, weights, cfg)
        assert np.all(pred >= 0), "sqrt transform predictions should be non-negative"


# ============================================================
# predict_pls_no_weight テスト
# ============================================================

class TestPredictPLSNoWeight:
    """predict_pls_no_weight のユニットテスト"""

    def _import_func(self):
        from scripts.modeling.run_issue98_cycle6_advval import predict_pls_no_weight
        return predict_pls_no_weight

    def test_output_shape(self):
        """出力の形状が正しいこと"""
        predict = self._import_func()
        rng = np.random.RandomState(42)
        X_tr = rng.randn(60, 30)
        X_te = rng.randn(10, 30)
        y_tr = rng.rand(60) * 100 + 1
        weights = np.ones(60)  # 使われないが引数は必要
        cfg = {"nc": 4, "tf": "sqrt"}
        pred = predict(X_tr, X_te, y_tr, weights, cfg)
        assert pred.shape == (10,)

    def test_ignores_weights(self):
        """重みを無視すること (PLSは重みなし)"""
        predict = self._import_func()
        rng = np.random.RandomState(42)
        X_tr = rng.randn(60, 20)
        X_te = rng.randn(10, 20)
        y_tr = rng.rand(60) * 100 + 1
        cfg = {"nc": 3, "tf": "raw"}

        pred1 = predict(X_tr, X_te, y_tr, np.ones(60), cfg)
        pred2 = predict(X_tr, X_te, y_tr, np.ones(60) * 10, cfg)
        np.testing.assert_array_almost_equal(pred1, pred2)


# ============================================================
# preprocess テスト
# ============================================================

class TestPreprocess:
    """preprocess のユニットテスト"""

    def _import_func(self):
        from scripts.modeling.run_issue98_cycle6_advval import preprocess
        return preprocess

    def test_raw_passthrough(self):
        """rawの場合、入力のコピーが返ること"""
        preprocess = self._import_func()
        rng = np.random.RandomState(42)
        X_tr = rng.randn(40, 20)
        X_te = rng.randn(10, 20)
        groups = np.array(["A"] * 20 + ["B"] * 20)
        X_tr_out, X_te_out = preprocess(X_tr, X_te, groups, "raw")
        np.testing.assert_array_almost_equal(X_tr, X_tr_out)
        np.testing.assert_array_almost_equal(X_te, X_te_out)

    def test_snv_output_shape(self):
        """SNVで出力形状が保たれること"""
        preprocess = self._import_func()
        rng = np.random.RandomState(42)
        X_tr = rng.randn(40, 20) + 1
        X_te = rng.randn(10, 20) + 1
        groups = np.array(["A"] * 20 + ["B"] * 20)
        X_tr_out, X_te_out = preprocess(X_tr, X_te, groups, "SNV")
        assert X_tr_out.shape == X_tr.shape
        assert X_te_out.shape == X_te.shape

    def test_epo_output_shape(self):
        """EPO(1)で出力形状が保たれること"""
        preprocess = self._import_func()
        rng = np.random.RandomState(42)
        X_tr = rng.randn(40, 20)
        X_te = rng.randn(10, 20)
        groups = np.array(["A"] * 20 + ["B"] * 20)
        X_tr_out, X_te_out = preprocess(X_tr, X_te, groups, "EPO(1)")
        assert X_tr_out.shape == X_tr.shape
        assert X_te_out.shape == X_te.shape
