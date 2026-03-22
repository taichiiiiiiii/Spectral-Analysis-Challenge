"""Issue #62: 前処理×線形モデル統合グリッド評価のテスト"""
import numpy as np
import pytest

from src.preprocessing.issue62_additional_preprocessing import (
    apply_asls,
    apply_whittaker,
    apply_piecewise_msc,
    select_water_bands,
    select_water_wide,
)
from src.preprocessing.issue19_msc import compute_msc_reference
from src.modeling.issue62_grid_evaluation import (
    _apply_preprocessing,
    _build_model,
    evaluate_single_combination,
)


@pytest.fixture
def sample_data():
    """テスト用の小規模データ"""
    rng = np.random.RandomState(42)
    n_samples, n_features = 60, 100
    X = rng.randn(n_samples, n_features) + 1000
    y = rng.rand(n_samples) * 100 + 10
    groups = np.array(["A"] * 20 + ["B"] * 20 + ["C"] * 20)
    wavenumbers = np.linspace(4000, 10000, n_features)
    return X, y, groups, wavenumbers


class TestAsLS:
    def test_shape_preserved(self, sample_data):
        X, _, _, _ = sample_data
        result = apply_asls(X, lam=1e5)
        assert result.shape == X.shape

    def test_baseline_removed(self, sample_data):
        X, _, _, _ = sample_data
        result = apply_asls(X, lam=1e5)
        # ベースライン補正後は平均がゼロ近傍に移動
        assert np.abs(result.mean()) < np.abs(X.mean())


class TestWhittaker:
    def test_shape_preserved(self, sample_data):
        X, _, _, _ = sample_data
        result = apply_whittaker(X, lam=1e3)
        assert result.shape == X.shape

    def test_smoothing_reduces_variance(self, sample_data):
        X, _, _, _ = sample_data
        # ノイズの多いデータに適用
        X_noisy = X + np.random.randn(*X.shape) * 10
        result = apply_whittaker(X_noisy, lam=1e3)
        # 行ごとの差分の分散が減少
        diff_orig = np.diff(X_noisy, axis=1)
        diff_smooth = np.diff(result, axis=1)
        assert diff_smooth.var() < diff_orig.var()


class TestPiecewiseMSC:
    def test_shape_preserved(self, sample_data):
        X, _, _, _ = sample_data
        ref = compute_msc_reference(X)
        result = apply_piecewise_msc(X, ref, n_segments=3)
        assert result.shape == X.shape

    def test_no_nan(self, sample_data):
        X, _, _, _ = sample_data
        ref = compute_msc_reference(X)
        result = apply_piecewise_msc(X, ref, n_segments=3)
        assert not np.any(np.isnan(result))


class TestWaterBandSelection:
    def test_water_bands_reduces_features(self, sample_data):
        X, _, _, wavenumbers = sample_data
        result = select_water_bands(X, wavenumbers)
        assert result.shape[0] == X.shape[0]
        assert result.shape[1] < X.shape[1]

    def test_water_wide_reduces_features(self, sample_data):
        X, _, _, wavenumbers = sample_data
        result = select_water_wide(X, wavenumbers)
        assert result.shape[0] == X.shape[0]
        assert result.shape[1] < X.shape[1]

    def test_water_wide_more_features_than_bands(self, sample_data):
        X, _, _, wavenumbers = sample_data
        bands = select_water_bands(X, wavenumbers)
        wide = select_water_wide(X, wavenumbers)
        assert wide.shape[1] >= bands.shape[1]


class TestPreprocessingDispatch:
    def test_raw(self, sample_data):
        X, _, groups, wavenumbers = sample_data
        X_tr, X_te = _apply_preprocessing(X[:40], X[40:], groups[:40], wavenumbers, "Raw")
        assert X_tr.shape == (40, 100)
        assert X_te.shape == (20, 100)

    def test_snv(self, sample_data):
        X, _, groups, wavenumbers = sample_data
        X_tr, X_te = _apply_preprocessing(X[:40], X[40:], groups[:40], wavenumbers, "SNV")
        assert X_tr.shape == (40, 100)

    def test_msc(self, sample_data):
        X, _, groups, wavenumbers = sample_data
        X_tr, X_te = _apply_preprocessing(X[:40], X[40:], groups[:40], wavenumbers, "MSC")
        assert X_tr.shape == (40, 100)


class TestBuildModel:
    def test_pls4(self):
        model = _build_model("PLS(4)")
        assert model.n_components == 4

    def test_lasso(self):
        model = _build_model("Lasso(0.1)")
        assert model.alpha == 0.1

    def test_elasticnet(self):
        model = _build_model("ElasticNet(0.1)")
        assert model.alpha == 0.1

    def test_ridge(self):
        model = _build_model("Ridge(100)")
        assert model.alpha == 100


class TestEvaluateSingleCombination:
    def test_returns_valid_result(self, sample_data):
        X, y, groups, wavenumbers = sample_data
        from sklearn.model_selection import LeaveOneGroupOut
        logo = LeaveOneGroupOut()
        folds = list(logo.split(X, y, groups))

        result = evaluate_single_combination(
            X, y, groups, wavenumbers, folds,
            "SNV", "Ridge(100)", "raw",
        )
        assert "rmse" in result
        assert "rmse_std" in result
        assert result["rmse"] > 0
        assert result["rmse"] < 900

    def test_sqrt_transform(self, sample_data):
        X, y, groups, wavenumbers = sample_data
        from sklearn.model_selection import LeaveOneGroupOut
        logo = LeaveOneGroupOut()
        folds = list(logo.split(X, y, groups))

        result = evaluate_single_combination(
            X, y, groups, wavenumbers, folds,
            "SNV", "PLS(2)", "sqrt",
        )
        assert result["rmse"] > 0
        assert result["rmse"] < 900
