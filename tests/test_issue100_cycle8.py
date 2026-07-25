"""Issue #100: Cycle 8 - Test-Time Augmentation (TTA) のテスト

TTA関数の正しさ、拡張スペクトルの統計的性質、
モデル実行パイプラインの整合性を検証する。
"""
import pytest
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.modeling.run_issue100_cycle8_tta import (
    generate_augmented_spectra,
    predict_with_tta,
    predict_without_tta,
    run_model_full,
    run_model_cv,
    pp,
    fs,
    rmse,
)
from src.eda.data_loader import load_train, get_spectral_columns

DATA_DIR = Path("Input_data")


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def train_data():
    """実データを読み込む。"""
    df = load_train(DATA_DIR)
    sc = get_spectral_columns(df)
    X = df[sc].values
    y = df["含水率"].values
    groups = df["樹種"].values
    return X, y, groups


@pytest.fixture
def small_spectra():
    """テスト用の小さなスペクトルデータ。"""
    rng = np.random.RandomState(42)
    X = rng.rand(20, 100)
    return X


# ============================================================
# generate_augmented_spectra のテスト
# ============================================================

class TestGenerateAugmentedSpectra:
    def test_returns_correct_count(self, small_spectra):
        """n_aug+1個の拡張スペクトルが返される（元のスペクトル含む）。"""
        result = generate_augmented_spectra(small_spectra, n_aug=5, seed=42)
        assert len(result) == 6  # 元 + 5拡張

    def test_first_element_is_original(self, small_spectra):
        """最初の要素は元のスペクトルのコピー。"""
        result = generate_augmented_spectra(small_spectra, n_aug=3, seed=42)
        np.testing.assert_array_equal(result[0], small_spectra)

    def test_augmented_shape_preserved(self, small_spectra):
        """全拡張バージョンのshapeが元と同じ。"""
        result = generate_augmented_spectra(small_spectra, n_aug=5, seed=42)
        for aug in result:
            assert aug.shape == small_spectra.shape

    def test_augmented_differs_from_original(self, small_spectra):
        """拡張バージョンは元のスペクトルと異なる。"""
        result = generate_augmented_spectra(small_spectra, n_aug=3, seed=42)
        for i in range(1, len(result)):
            assert not np.allclose(result[i], small_spectra)

    def test_augmented_close_to_original(self, small_spectra):
        """拡張バージョンは元のスペクトルに近い（微小摂動）。"""
        result = generate_augmented_spectra(
            small_spectra, n_aug=5, seed=42,
            offset_std=0.001, slope_std=1e-6,
            scale_std=0.002, noise_std=0.0005,
        )
        for i in range(1, len(result)):
            diff = np.abs(result[i] - small_spectra)
            # 全体の差分は小さいはず（最大でも数%以内）
            assert np.mean(diff) < 0.05

    def test_reproducibility(self, small_spectra):
        """同じseedで同じ結果が得られる。"""
        r1 = generate_augmented_spectra(small_spectra, n_aug=3, seed=123)
        r2 = generate_augmented_spectra(small_spectra, n_aug=3, seed=123)
        for a, b in zip(r1, r2):
            np.testing.assert_array_equal(a, b)

    def test_different_seeds_give_different_results(self, small_spectra):
        """異なるseedで異なる結果が得られる。"""
        r1 = generate_augmented_spectra(small_spectra, n_aug=3, seed=42)
        r2 = generate_augmented_spectra(small_spectra, n_aug=3, seed=99)
        assert not np.allclose(r1[1], r2[1])

    def test_zero_noise_returns_original(self, small_spectra):
        """全ノイズパラメータが0なら元のスペクトルと同じ。"""
        result = generate_augmented_spectra(
            small_spectra, n_aug=3, seed=42,
            offset_std=0, slope_std=0, scale_std=0, noise_std=0,
        )
        for aug in result:
            np.testing.assert_array_almost_equal(aug, small_spectra)

    def test_n_aug_zero(self, small_spectra):
        """n_aug=0の場合、元のスペクトルのみ返す。"""
        result = generate_augmented_spectra(small_spectra, n_aug=0, seed=42)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], small_spectra)

    def test_mean_of_augmented_close_to_original(self, small_spectra):
        """多数の拡張版の平均が元のスペクトルに近い。"""
        result = generate_augmented_spectra(small_spectra, n_aug=100, seed=42)
        mean_aug = np.mean(result[1:], axis=0)  # 拡張版のみの平均
        # scale_std=0.002で乗算されるので完全一致はしないが近い
        np.testing.assert_allclose(mean_aug, small_spectra, atol=0.01)


# ============================================================
# rmse のテスト
# ============================================================

class TestRMSE:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        expected = np.sqrt(1.0 / 3.0)
        assert rmse(y_true, y_pred) == pytest.approx(expected)


# ============================================================
# 前処理のテスト
# ============================================================

class TestPreprocessing:
    def test_pp_returns_correct_shape(self, train_data):
        """前処理後のshapeが正しい。"""
        X, y, groups = train_data
        # SNV
        Xtr, Xte = pp(X[:50], X[50:60], groups[:50], "SNV")
        assert Xtr.shape == (50, X.shape[1])
        assert Xte.shape == (10, X.shape[1])

    def test_pp_epo(self, train_data):
        """EPO前処理が動作する。"""
        X, y, groups = train_data
        Xtr, Xte = pp(X[:50], X[50:60], groups[:50], "EPO(1)")
        assert Xtr.shape[0] == 50
        assert Xte.shape[0] == 10

    def test_pp_unknown_returns_copy(self, train_data):
        """未知の前処理名ではコピーが返される。"""
        X, y, groups = train_data
        Xtr, Xte = pp(X[:10], X[10:15], groups[:10], "unknown")
        np.testing.assert_array_equal(Xtr, X[:10])

    def test_fs_none(self, train_data):
        """fs_name=NoneではX_tr, X_teがそのまま返される。"""
        X, y, groups = train_data
        Xtr, Xte = fs(X[:50], X[50:60], y[:50], None)
        np.testing.assert_array_equal(Xtr, X[:50])


# ============================================================
# run_model_full / run_model_cv のテスト
# ============================================================

class TestRunModel:
    def test_run_model_full_pls_sqrt(self, train_data):
        """PLS+sqrt変換で予測が返される。"""
        X, y, groups = train_data
        cfg = {"pp": "EPO(1)", "nc": 4, "tf": "sqrt"}
        pred = run_model_full(X, y, groups, X[:10], cfg)
        assert pred.shape == (10,)
        assert np.all(np.isfinite(pred))

    def test_run_model_full_pls_raw(self, train_data):
        """PLS+raw変換で予測が返される。"""
        X, y, groups = train_data
        cfg = {"pp": "SNV", "nc": 3, "tf": "raw"}
        pred = run_model_full(X, y, groups, X[:10], cfg)
        assert pred.shape == (10,)

    def test_run_model_cv(self, train_data):
        """CVモードで予測が返される。"""
        X, y, groups = train_data
        cfg = {"pp": "SNV", "nc": 4, "tf": "sqrt"}
        pred = run_model_cv(X[:100], X[100:110], y[:100], groups[:100], cfg)
        assert pred.shape == (10,)
        assert np.all(np.isfinite(pred))


# ============================================================
# predict_with_tta / predict_without_tta のテスト
# ============================================================

class TestPredictTTA:
    def test_predict_without_tta_returns_correct_shape(self, train_data):
        """TTAなし予測が正しいshapeを返す。"""
        X, y, groups = train_data
        cfgs = [
            {"name": "M1", "pp": "EPO(1)", "nc": 4, "tf": "sqrt"},
        ]
        pred = predict_without_tta(X, y, groups, X[:5], cfgs)
        assert pred.shape == (5,)
        assert np.all(pred >= 0)
        assert np.all(pred <= 300)

    def test_predict_with_tta_returns_correct_shape(self, train_data):
        """TTAあり予測が正しいshapeを返す。"""
        X, y, groups = train_data
        cfgs = [
            {"name": "M1", "pp": "EPO(1)", "nc": 4, "tf": "sqrt"},
        ]
        pred = predict_with_tta(X, y, groups, X[:5], cfgs, n_aug=3, seed=42)
        assert pred.shape == (5,)
        assert np.all(pred >= 0)
        assert np.all(pred <= 300)

    def test_tta_with_zero_aug_equals_no_tta(self, train_data):
        """n_aug=0のTTAはTTAなしと同じ結果。"""
        X, y, groups = train_data
        cfgs = [
            {"name": "M1", "pp": "EPO(1)", "nc": 4, "tf": "sqrt"},
        ]
        pred_notta = predict_without_tta(X, y, groups, X[:5], cfgs)
        pred_tta0 = predict_with_tta(
            X, y, groups, X[:5], cfgs, n_aug=0, seed=42
        )
        np.testing.assert_allclose(pred_tta0, pred_notta, atol=1e-10)

    def test_tta_with_zero_noise_equals_no_tta(self, train_data):
        """全ノイズ0のTTAはTTAなしと同じ結果。"""
        X, y, groups = train_data
        cfgs = [
            {"name": "M1", "pp": "EPO(1)", "nc": 4, "tf": "sqrt"},
        ]
        zero_params = {
            "offset_std": 0, "slope_std": 0,
            "scale_std": 0, "noise_std": 0,
        }
        pred_notta = predict_without_tta(X, y, groups, X[:5], cfgs)
        pred_tta = predict_with_tta(
            X, y, groups, X[:5], cfgs, n_aug=5, seed=42,
            tta_params=zero_params
        )
        np.testing.assert_allclose(pred_tta, pred_notta, atol=1e-10)

    def test_tta_produces_different_result_with_noise(self, train_data):
        """ノイズありTTAはTTAなしとわずかに異なる。"""
        X, y, groups = train_data
        cfgs = [
            {"name": "M1", "pp": "EPO(1)", "nc": 4, "tf": "sqrt"},
        ]
        pred_notta = predict_without_tta(X, y, groups, X[:5], cfgs)
        pred_tta = predict_with_tta(
            X, y, groups, X[:5], cfgs, n_aug=10, seed=42
        )
        # ノイズが小さいので差は小さいが、完全一致ではない
        assert not np.allclose(pred_tta, pred_notta, atol=1e-10)
        # ただし差は小さい
        np.testing.assert_allclose(pred_tta, pred_notta, atol=5.0)
