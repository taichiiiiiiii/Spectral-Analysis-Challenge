"""Issue #99: Cycle7 反復的Pseudo Labeling テスト"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest


# ============================================================
# pseudo_label_iteration のインポート
# ============================================================

def _import_module():
    """run_issue99 モジュールから関数をインポート"""
    import importlib.util
    mod_path = Path(__file__).resolve().parents[1] / "scripts" / "modeling" / "run_issue99_cycle7_pseudo.py"
    spec = importlib.util.spec_from_file_location("run_issue99", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _import_module()


# ============================================================
# テストデータ生成
# ============================================================

@pytest.fixture(scope="module")
def synth_data():
    """合成データ: 3グループ、50次元"""
    rng = np.random.RandomState(42)
    n_per_group = 30
    n_features = 50

    groups_list = ["A"] * n_per_group + ["B"] * n_per_group + ["C"] * n_per_group
    groups = np.array(groups_list)
    n_total = len(groups)

    X = rng.randn(n_total, n_features)
    # グループ固有のオフセット
    for i, g in enumerate(["A", "B", "C"]):
        mask = groups == g
        X[mask] += i * 0.5

    # 目的変数: 最初の3次元の線形結合 + ノイズ
    coef = rng.randn(3)
    y = X[:, :3] @ coef + rng.randn(n_total) * 0.5
    y = np.abs(y) * 10 + 5  # 正値に変換

    X_test = rng.randn(20, n_features) + 1.0
    return X, y, groups, X_test


# ============================================================
# pseudo_label_iteration 基本テスト
# ============================================================

class TestPseudoLabelIteration:
    def test_output_shape(self, mod, synth_data):
        """出力の形状が正しい"""
        X, y, groups, X_test = synth_data
        X_aug, y_aug, g_aug = mod.pseudo_label_iteration(
            X, y, groups, X_test,
            n_iterations=1, confidence_ratio=0.3,
        )
        n_pseudo = int(len(X_test) * 0.3)
        assert X_aug.shape[0] == len(X) + n_pseudo
        assert len(y_aug) == len(X) + n_pseudo
        assert len(g_aug) == len(X) + n_pseudo
        assert X_aug.shape[1] == X.shape[1]

    def test_original_data_preserved(self, mod, synth_data):
        """元のtrainデータが先頭に保持される"""
        X, y, groups, X_test = synth_data
        X_aug, y_aug, g_aug = mod.pseudo_label_iteration(
            X, y, groups, X_test,
            n_iterations=1, confidence_ratio=0.2,
        )
        np.testing.assert_array_equal(X_aug[:len(X)], X)
        np.testing.assert_array_equal(y_aug[:len(y)], y)
        np.testing.assert_array_equal(g_aug[:len(groups)], groups)

    def test_pseudo_group_label(self, mod, synth_data):
        """疑似サンプルのグループは 'pseudo'"""
        X, y, groups, X_test = synth_data
        _, _, g_aug = mod.pseudo_label_iteration(
            X, y, groups, X_test,
            n_iterations=1, confidence_ratio=0.5,
        )
        pseudo_groups = g_aug[len(groups):]
        assert all(g == "pseudo" for g in pseudo_groups)

    def test_multiple_iterations_reset(self, mod, synth_data):
        """反復ごとに元trainからリセットされる(サイズが元+pseudo)"""
        X, y, groups, X_test = synth_data
        ratio = 0.3
        X_aug, y_aug, _ = mod.pseudo_label_iteration(
            X, y, groups, X_test,
            n_iterations=3, confidence_ratio=ratio,
        )
        n_pseudo = int(len(X_test) * ratio)
        # 最終的なサイズは元train + 最終反復の疑似サンプル
        assert X_aug.shape[0] == len(X) + n_pseudo

    def test_confidence_ratio_bounds(self, mod, synth_data):
        """confidence_ratio=0で疑似サンプルなし"""
        X, y, groups, X_test = synth_data
        X_aug, y_aug, g_aug = mod.pseudo_label_iteration(
            X, y, groups, X_test,
            n_iterations=1, confidence_ratio=0.0,
        )
        # ratio=0 → n_select=0 → 元データのみ
        assert X_aug.shape[0] == len(X)

    def test_pseudo_labels_are_finite(self, mod, synth_data):
        """疑似ラベルが有限値"""
        X, y, groups, X_test = synth_data
        _, y_aug, _ = mod.pseudo_label_iteration(
            X, y, groups, X_test,
            n_iterations=2, confidence_ratio=0.3,
        )
        assert np.all(np.isfinite(y_aug))


# ============================================================
# preprocess 関数テスト
# ============================================================

class TestPreprocess:
    def test_raw(self, mod, synth_data):
        X, y, groups, _ = synth_data
        Xtr, Xte = mod.preprocess(X[:60], X[60:], groups[:60], "raw")
        np.testing.assert_array_equal(Xtr, X[:60])

    def test_snv_shape(self, mod, synth_data):
        X, y, groups, _ = synth_data
        Xtr, Xte = mod.preprocess(X[:60], X[60:], groups[:60], "SNV")
        assert Xtr.shape == X[:60].shape
        assert Xte.shape == X[60:].shape


# ============================================================
# train_and_predict テスト
# ============================================================

class TestTrainAndPredict:
    def test_pls_prediction_shape(self, mod, synth_data):
        """PLSベース予測の出力形状"""
        X, y, groups, X_test = synth_data
        cfg = {"pp": "raw", "nc": 3, "tf": "sqrt", "type": "pls"}
        pred = mod.train_and_predict(X, y, groups, X_test, cfg)
        assert pred.shape == (len(X_test),)
        assert np.all(np.isfinite(pred))

    def test_sqrt_transform_nonneg(self, mod, synth_data):
        """sqrt変換で予測値が非負"""
        X, y, groups, X_test = synth_data
        cfg = {"pp": "raw", "nc": 3, "tf": "sqrt", "type": "pls"}
        pred = mod.train_and_predict(X, y, groups, X_test, cfg)
        assert np.all(pred >= 0)


# ============================================================
# rmse 関数テスト
# ============================================================

class TestRmse:
    def test_perfect(self, mod):
        y = np.array([1.0, 2.0, 3.0])
        assert mod.rmse(y, y) == pytest.approx(0.0)

    def test_known_value(self, mod):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        expected = np.sqrt(1.0 / 3.0)
        assert mod.rmse(y_true, y_pred) == pytest.approx(expected)
