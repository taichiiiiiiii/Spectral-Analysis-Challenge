"""Issue #94 Cycle 2: TCAドメイン適応パイプラインのテスト"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.model_selection import LeaveOneGroupOut

from src.preprocessing.issue38_tca import tca_transform
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo


class TestTCATransform:
    """TCA変換の基本テスト"""

    def setup_method(self):
        np.random.seed(42)
        self.X_source = np.random.randn(50, 20)
        self.X_target = np.random.randn(30, 20)

    @pytest.mark.parametrize("n_components", [3, 5, 10])
    def test_output_dimensions(self, n_components):
        """TCA変換後の次元数が正しいことを確認"""
        Z_s, Z_t = tca_transform(
            self.X_source, self.X_target, n_components=n_components, kernel="linear"
        )
        assert Z_s.shape == (50, n_components)
        assert Z_t.shape == (30, n_components)

    def test_rbf_kernel(self):
        """RBFカーネルで正常動作"""
        Z_s, Z_t = tca_transform(
            self.X_source, self.X_target, n_components=5, kernel="rbf"
        )
        assert Z_s.shape == (50, 5)
        assert Z_t.shape == (30, 5)
        assert np.all(np.isfinite(Z_s))
        assert np.all(np.isfinite(Z_t))

    def test_linear_kernel(self):
        """Linearカーネルで正常動作"""
        Z_s, Z_t = tca_transform(
            self.X_source, self.X_target, n_components=5, kernel="linear"
        )
        assert np.all(np.isfinite(Z_s))
        assert np.all(np.isfinite(Z_t))

    def test_mu_parameter(self):
        """muパラメータによって結果が変わることを確認"""
        Z_s1, _ = tca_transform(
            self.X_source, self.X_target, n_components=5, kernel="linear", mu=0.01
        )
        Z_s2, _ = tca_transform(
            self.X_source, self.X_target, n_components=5, kernel="linear", mu=10.0
        )
        assert not np.allclose(Z_s1, Z_s2)


class TestLOSOCVLoop:
    """LOSO-CVループの正しさをテスト"""

    def setup_method(self):
        np.random.seed(42)
        n_species = 4
        samples_per_species = 15
        n_features = 20
        self.n_total = n_species * samples_per_species

        self.X = np.random.randn(self.n_total, n_features)
        self.y = np.random.rand(self.n_total) * 100 + 50
        self.groups = np.repeat([f"sp_{i}" for i in range(n_species)], samples_per_species)

    def test_loso_cv_no_leakage(self):
        """各foldでtrain/testの樹種が重複しないことを確認"""
        logo = LeaveOneGroupOut()
        for tr_idx, te_idx in logo.split(self.X, self.y, self.groups):
            train_species = set(self.groups[tr_idx])
            test_species = set(self.groups[te_idx])
            assert len(train_species & test_species) == 0

    def test_loso_cv_with_tca(self):
        """TCA + 回帰のLOSO-CVが正常に実行できることを確認"""
        logo = LeaveOneGroupOut()
        predictions = []
        actuals = []

        for tr_idx, te_idx in logo.split(self.X, self.y, self.groups):
            X_tr, X_te = self.X[tr_idx], self.X[te_idx]
            y_tr = self.y[tr_idx]

            Z_tr, Z_te = tca_transform(X_tr, X_te, n_components=3, kernel="linear", mu=1.0)
            model = Ridge(alpha=1.0)
            model.fit(Z_tr, y_tr)
            pred = model.predict(Z_te)

            predictions.extend(pred.tolist())
            actuals.extend(self.y[te_idx].tolist())

        assert len(predictions) == self.n_total
        assert len(actuals) == self.n_total

    def test_loso_cv_all_samples_predicted(self):
        """全サンプルが1回ずつ予測されることを確認"""
        logo = LeaveOneGroupOut()
        predicted_indices = []
        for _, te_idx in logo.split(self.X, self.y, self.groups):
            predicted_indices.extend(te_idx.tolist())
        assert sorted(predicted_indices) == list(range(self.n_total))


class TestPredictionRange:
    """予測値の範囲チェック"""

    def setup_method(self):
        np.random.seed(42)
        self.X_source = np.random.randn(60, 20)
        self.X_target = np.random.randn(20, 20)
        self.y = np.abs(np.random.randn(60)) * 50 + 30  # 含水率は正の値

    def test_predictions_finite(self):
        """予測値が有限であることを確認"""
        Z_s, Z_t = tca_transform(self.X_source, self.X_target, n_components=5, kernel="linear")
        model = Ridge(alpha=1.0)
        model.fit(Z_s, self.y)
        pred = model.predict(Z_t)
        assert np.all(np.isfinite(pred))

    def test_predictions_reasonable_range(self):
        """予測値が極端な値にならないことを確認（含水率: 典型的に0~300%）"""
        Z_s, Z_t = tca_transform(self.X_source, self.X_target, n_components=5, kernel="linear")
        model = Ridge(alpha=1.0)
        model.fit(Z_s, self.y)
        pred = model.predict(Z_t)
        # 極端な外挿でなければ±1000以内に収まるはず
        assert np.all(pred > -1000)
        assert np.all(pred < 1000)

    def test_sqrt_transform_inverse(self):
        """sqrt変換→逆変換の整合性"""
        y_sqrt = np.sqrt(self.y)
        Z_s, Z_t = tca_transform(self.X_source, self.X_target, n_components=5, kernel="linear")
        model = Ridge(alpha=1.0)
        model.fit(Z_s, y_sqrt)
        pred_sqrt = model.predict(Z_t)
        pred = pred_sqrt ** 2
        assert np.all(np.isfinite(pred))


class TestPreprocessingIntegration:
    """前処理との統合テスト"""

    def setup_method(self):
        np.random.seed(42)
        self.X = np.random.rand(40, 20) + 0.1  # 正の値（スペクトルデータ想定）
        self.groups = np.repeat(["A", "B", "C", "D"], 10)

    def test_snv_then_tca(self):
        """SNV → TCA の連携"""
        X_snv = apply_snv(self.X)
        X_source = X_snv[:30]
        X_target = X_snv[30:]
        Z_s, Z_t = tca_transform(X_source, X_target, n_components=5, kernel="rbf")
        assert Z_s.shape == (30, 5)
        assert Z_t.shape == (10, 5)
        assert np.all(np.isfinite(Z_s))

    def test_epo_then_tca(self):
        """EPO → TCA の連携"""
        P = compute_epo_projection(self.X[:30], self.groups[:30], n_components=1)
        X_epo_source = apply_epo(self.X[:30], P)
        X_epo_target = apply_epo(self.X[30:], P)
        Z_s, Z_t = tca_transform(X_epo_source, X_epo_target, n_components=5, kernel="linear")
        assert Z_s.shape == (30, 5)
        assert Z_t.shape == (10, 5)
        assert np.all(np.isfinite(Z_s))
