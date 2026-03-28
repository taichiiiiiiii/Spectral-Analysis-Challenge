"""Issue #96: サイクル4 水分吸収帯特徴量のユニットテスト"""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.modeling.run_issue96_cycle4_waterband import (
    _wn_idx,
    _wn_range_mask,
    compute_waterband_features,
    run_strategy,
)


class TestWnIdx:
    """波数インデックス検索のテスト"""

    def test_exact_match(self):
        wn = np.array([7000, 6000, 5000, 4000])
        assert _wn_idx(wn, 5000) == 2

    def test_closest_match(self):
        wn = np.array([7000, 6500, 6000, 5500, 5000])
        # 5200と5000の距離=200, 5200と5500の距離=300 → 5000(idx=4)が最も近い
        assert _wn_idx(wn, 5200) == 4

    def test_boundary(self):
        wn = np.array([9000, 8000, 7000])
        assert _wn_idx(wn, 9500) == 0  # 左端が最も近い
        assert _wn_idx(wn, 6000) == 2  # 右端が最も近い


class TestWnRangeMask:
    """波数範囲マスクのテスト"""

    def test_basic_range(self):
        wn = np.array([5300, 5250, 5200, 5150, 5100])
        mask = _wn_range_mask(wn, 5200, 50)
        # 5150-5250の範囲 → idx 1, 2, 3
        assert mask.sum() == 3
        assert mask[0] is np.bool_(False)  # 5300は範囲外
        assert mask[4] is np.bool_(False)  # 5100は範囲外

    def test_all_in_range(self):
        wn = np.array([5210, 5200, 5190])
        mask = _wn_range_mask(wn, 5200, 50)
        assert mask.sum() == 3

    def test_none_in_range(self):
        wn = np.array([9000, 8000, 7000])
        mask = _wn_range_mask(wn, 5200, 50)
        assert mask.sum() == 0


class TestComputeWaterbandFeatures:
    """水分帯特徴量計算のテスト"""

    def setup_method(self):
        np.random.seed(42)
        # 実際のデータに近い波数配列（降順、4000-10000 cm⁻¹）
        self.wavenumbers = np.linspace(10000, 4000, 1555)
        self.X = np.random.rand(30, 1555) * 0.5 + 0.3

    def test_output_shape_no_snv(self):
        feat, names = compute_waterband_features(self.X, self.wavenumbers, use_snv=False)
        assert feat.shape[0] == 30
        assert feat.shape[1] == len(names)
        # バンド比3 + バンド差2 + 面積2 + SG2d 2 + NDMI 1 = 10
        assert feat.shape[1] == 10

    def test_output_shape_with_snv(self):
        feat, names = compute_waterband_features(self.X, self.wavenumbers, use_snv=True)
        assert feat.shape[0] == 30
        assert feat.shape[1] == len(names)
        # 基本10 + SNV: ratio3 + diff1 + ndmi1 + sg2d 2 = 17
        assert feat.shape[1] == 17

    def test_all_finite(self):
        feat, _ = compute_waterband_features(self.X, self.wavenumbers, use_snv=True)
        assert np.all(np.isfinite(feat))

    def test_feature_names(self):
        _, names = compute_waterband_features(self.X, self.wavenumbers, use_snv=False)
        assert "ratio_5200_6900" in names
        assert "ratio_5200_5800" in names
        assert "ratio_6900_5800" in names
        assert "diff_5200_6900" in names
        assert "diff_5200_5800" in names
        assert "area_5200" in names
        assert "area_6900" in names
        assert "sg2d_5200" in names
        assert "sg2d_6900" in names
        assert "ndmi" in names

    def test_snv_feature_names(self):
        _, names = compute_waterband_features(self.X, self.wavenumbers, use_snv=True)
        assert "snv_ratio_5200_6900" in names
        assert "snv_ndmi" in names
        assert "snv_sg2d_5200" in names

    def test_ndmi_range(self):
        """NDMIは[-1, 1]の範囲内であるべき"""
        feat, names = compute_waterband_features(self.X, self.wavenumbers, use_snv=False)
        ndmi_idx = names.index("ndmi")
        ndmi_vals = feat[:, ndmi_idx]
        assert np.all(ndmi_vals >= -1.0 - 1e-10)
        assert np.all(ndmi_vals <= 1.0 + 1e-10)

    def test_single_sample(self):
        """1サンプルでもエラーなく動作"""
        feat, names = compute_waterband_features(
            self.X[:1], self.wavenumbers, use_snv=False)
        assert feat.shape == (1, len(names))
        assert np.all(np.isfinite(feat))


class TestRunStrategy:
    """各戦略の実行テスト"""

    def setup_method(self):
        np.random.seed(42)
        self.wavenumbers = np.linspace(10000, 4000, 1555)
        self.X_tr = np.random.rand(50, 1555) * 0.5 + 0.3
        self.X_te = np.random.rand(10, 1555) * 0.5 + 0.3
        self.y_tr = np.random.rand(50) * 100 + 50
        self.groups_tr = np.array(["A"] * 25 + ["B"] * 25)

    def test_strategy_A_sqrt(self):
        pred = run_strategy(
            self.X_tr, self.X_te, self.y_tr, self.groups_tr,
            self.wavenumbers, "A", "sqrt")
        assert pred.shape == (10,)
        assert np.all(np.isfinite(pred))
        assert np.all(pred >= 0)  # sqrt変換後は非負

    def test_strategy_A_raw(self):
        pred = run_strategy(
            self.X_tr, self.X_te, self.y_tr, self.groups_tr,
            self.wavenumbers, "A", "raw")
        assert pred.shape == (10,)
        assert np.all(np.isfinite(pred))

    def test_strategy_B_sqrt(self):
        pred = run_strategy(
            self.X_tr, self.X_te, self.y_tr, self.groups_tr,
            self.wavenumbers, "B", "sqrt")
        assert pred.shape == (10,)
        assert np.all(np.isfinite(pred))

    def test_strategy_C_sqrt(self):
        pred = run_strategy(
            self.X_tr, self.X_te, self.y_tr, self.groups_tr,
            self.wavenumbers, "C", "sqrt")
        assert pred.shape == (10,)
        assert np.all(np.isfinite(pred))

    def test_strategy_D_sqrt(self):
        pred = run_strategy(
            self.X_tr, self.X_te, self.y_tr, self.groups_tr,
            self.wavenumbers, "D", "sqrt")
        assert pred.shape == (10,)
        assert np.all(np.isfinite(pred))

    def test_strategy_E_sqrt(self):
        pred = run_strategy(
            self.X_tr, self.X_te, self.y_tr, self.groups_tr,
            self.wavenumbers, "E", "sqrt")
        assert pred.shape == (10,)
        assert np.all(np.isfinite(pred))

    def test_invalid_strategy(self):
        with pytest.raises(ValueError):
            run_strategy(
                self.X_tr, self.X_te, self.y_tr, self.groups_tr,
                self.wavenumbers, "Z", "sqrt")

    def test_strategy_B_raw(self):
        pred = run_strategy(
            self.X_tr, self.X_te, self.y_tr, self.groups_tr,
            self.wavenumbers, "B", "raw")
        assert pred.shape == (10,)
        assert np.all(np.isfinite(pred))
