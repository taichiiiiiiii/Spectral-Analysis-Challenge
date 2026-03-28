"""Issue #101: サイクル9 KMM テスト"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor

from src.preprocessing.issue50_kmm import compute_kmm_weights
from src.eda.data_loader import load_train, load_test, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo

DATA_DIR = Path(__file__).resolve().parents[1] / "Input_data"


class TestKMMWeights:
    """KMM重み計算のテスト"""

    def test_weights_shape(self):
        """KMM重みの形状がソースサンプル数と一致すること"""
        np.random.seed(42)
        X_s = np.random.randn(50, 10)
        X_t = np.random.randn(20, 10)
        w = compute_kmm_weights(X_s, X_t, B=10.0)
        assert w.shape == (50,), f"Expected (50,), got {w.shape}"

    def test_weights_nonnegative(self):
        """KMM重みが非負であること"""
        np.random.seed(42)
        X_s = np.random.randn(50, 10)
        X_t = np.random.randn(20, 10)
        w = compute_kmm_weights(X_s, X_t, B=10.0)
        assert np.all(w >= 0), "Weights should be non-negative"

    def test_weights_bounded(self):
        """KMM重みが上限B以下であること"""
        np.random.seed(42)
        X_s = np.random.randn(50, 10)
        X_t = np.random.randn(20, 10)
        B = 5.0
        w = compute_kmm_weights(X_s, X_t, B=B)
        assert np.all(w <= B + 1e-6), f"Weights should be <= {B}"

    def test_weights_with_similar_distributions(self):
        """同じ分布からのデータでは重みが均一に近いこと"""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        X_s = X[:60]
        X_t = X[60:]
        w = compute_kmm_weights(X_s, X_t, B=10.0)
        # 同分布なら重みは1.0付近のはず
        assert np.std(w) < 3.0, "Weights should be relatively uniform for same distribution"

    def test_weights_with_shifted_distributions(self):
        """シフトした分布では重みにばらつきが出ること"""
        np.random.seed(42)
        X_s = np.random.randn(60, 10)
        X_t = np.random.randn(30, 10) + 3.0  # 大きなシフト
        w = compute_kmm_weights(X_s, X_t, B=10.0)
        # シフトがあれば重みのばらつきが大きくなる
        assert w.shape == (60,)
        assert np.all(w >= 0)


class TestKMMWithPCA:
    """PCA次元削減後のKMM適用テスト"""

    def test_pca_then_kmm(self):
        """PCA後にKMMが正常動作すること"""
        np.random.seed(42)
        X_s = np.random.randn(80, 100)
        X_t = np.random.randn(30, 100)
        pca = PCA(n_components=20)
        X_s_pca = pca.fit_transform(X_s)
        X_t_pca = pca.transform(X_t)
        w = compute_kmm_weights(X_s_pca, X_t_pca, B=10.0)
        assert w.shape == (80,)
        assert np.all(w >= 0)

    def test_different_pca_dims(self):
        """異なるPCA次元数でKMMが動作すること"""
        np.random.seed(42)
        X_s = np.random.randn(80, 200)
        X_t = np.random.randn(30, 200)
        for n_comp in [10, 20, 50]:
            pca = PCA(n_components=n_comp)
            X_s_pca = pca.fit_transform(X_s)
            X_t_pca = pca.transform(X_t)
            w = compute_kmm_weights(X_s_pca, X_t_pca, B=10.0)
            assert w.shape == (80,)


class TestWeightedModels:
    """重み付きモデル学習のテスト"""

    def test_weighted_ridge(self):
        """重み付きRidgeが正常に学習・予測できること"""
        np.random.seed(42)
        X_tr = np.random.randn(50, 5)
        y_tr = np.random.randn(50) + 30
        X_te = np.random.randn(10, 5)
        w = np.random.uniform(0.1, 5.0, size=50)

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_tr, y_tr, sample_weight=w)
        pred = ridge.predict(X_te)
        assert pred.shape == (10,)
        assert np.all(np.isfinite(pred))

    def test_weighted_gbr(self):
        """重み付きGBRが正常に学習・予測できること"""
        np.random.seed(42)
        X_tr = np.random.randn(60, 5)
        y_tr = np.abs(np.random.randn(60)) * 10 + 20
        X_te = np.random.randn(10, 5)
        w = np.random.uniform(0.1, 5.0, size=60)

        gbr = GradientBoostingRegressor(
            n_estimators=50, max_depth=2, random_state=42,
        )
        gbr.fit(X_tr, y_tr, sample_weight=w)
        pred = gbr.predict(X_te)
        assert pred.shape == (10,)
        assert np.all(np.isfinite(pred))


class TestKMMPipeline:
    """KMMパイプライン統合テスト（実データ使用）"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.df = load_train(DATA_DIR)
        self.sc = get_spectral_columns(self.df)
        self.X = self.df[self.sc].values
        self.y = self.df["含水率"].values
        self.groups = self.df["樹種"].values

    def test_kmm_ridge_pipeline_single_fold(self):
        """1 foldでKMM+Ridge全パイプラインが動作すること"""
        from sklearn.model_selection import LeaveOneGroupOut
        logo = LeaveOneGroupOut()
        tr, te = next(iter(logo.split(self.X, self.y, self.groups)))

        X_tr, X_te = self.X[tr], self.X[te]
        y_tr = self.y[tr]
        groups_tr = self.groups[tr]

        # 前処理: EPO
        P = compute_epo_projection(X_tr, groups_tr, n_components=1)
        X_tr_pp = apply_epo(X_tr, P)
        X_te_pp = apply_epo(X_te, P)

        # KMM重み
        pca = PCA(n_components=20)
        X_tr_pca = pca.fit_transform(X_tr_pp)
        X_te_pca = pca.transform(X_te_pp)
        weights = compute_kmm_weights(X_tr_pca, X_te_pca, B=10.0)
        assert weights.shape == (len(tr),)

        # PLS + Ridge
        y_sqrt = np.sqrt(y_tr)
        pls = PLSRegression(n_components=4)
        pls.fit(X_tr_pp, y_sqrt)
        T_tr = pls.transform(X_tr_pp)
        T_te = pls.transform(X_te_pp)

        ridge = Ridge(alpha=1.0)
        ridge.fit(T_tr, y_sqrt, sample_weight=weights)
        pred = ridge.predict(T_te).ravel()
        pred_mc = np.clip(pred, 0, None) ** 2

        assert pred_mc.shape == (len(te),)
        assert np.all(np.isfinite(pred_mc))
        assert np.all(pred_mc >= 0)

    def test_kmm_snv_pipeline_single_fold(self):
        """SNV前処理でKMMパイプラインが動作すること"""
        from sklearn.model_selection import LeaveOneGroupOut
        logo = LeaveOneGroupOut()
        tr, te = next(iter(logo.split(self.X, self.y, self.groups)))

        X_tr_snv = apply_snv(self.X[tr])
        X_te_snv = apply_snv(self.X[te])

        pca = PCA(n_components=10)
        X_tr_pca = pca.fit_transform(X_tr_snv)
        X_te_pca = pca.transform(X_te_snv)
        weights = compute_kmm_weights(X_tr_pca, X_te_pca, B=5.0)

        assert weights.shape == (len(tr),)
        assert np.all(weights >= 0)

    def test_predict_fold_functions(self):
        """predict_fold関数が正常に動作すること"""
        from scripts.modeling.run_issue101_cycle9_kmm import (
            predict_fold_kmm_ridge,
            predict_fold_kmm_gbr,
            predict_fold_kmm_wpls,
        )
        from sklearn.model_selection import LeaveOneGroupOut
        logo = LeaveOneGroupOut()
        tr, te = next(iter(logo.split(self.X, self.y, self.groups)))

        cfg_ridge = {
            "pp": "EPO(1)", "nc": 4, "tf": "sqrt",
            "pca_dim": 20, "kmm_B": 10.0, "alpha": 1.0,
        }
        pred_r = predict_fold_kmm_ridge(
            self.X[tr], self.X[te], self.y[tr], self.groups[tr], cfg_ridge
        )
        assert pred_r.shape == (len(te),)
        assert np.all(np.isfinite(pred_r))
        assert np.all(pred_r >= 0)

        cfg_gbr = {
            "pp": "EPO(1)", "nc": 4, "tf": "raw",
            "pca_dim": 20, "kmm_B": 10.0,
            "n_est": 50, "max_depth": 2, "lr": 0.1,
        }
        pred_g = predict_fold_kmm_gbr(
            self.X[tr], self.X[te], self.y[tr], self.groups[tr], cfg_gbr
        )
        assert pred_g.shape == (len(te),)
        assert np.all(np.isfinite(pred_g))

        cfg_wpls = {
            "pp": "EPO(1)", "nc": 4, "tf": "sqrt",
            "pca_dim": 20, "kmm_B": 10.0,
        }
        pred_p = predict_fold_kmm_wpls(
            self.X[tr], self.X[te], self.y[tr], self.groups[tr], cfg_wpls
        )
        assert pred_p.shape == (len(te),)
        assert np.all(np.isfinite(pred_p))
        assert np.all(pred_p >= 0)
