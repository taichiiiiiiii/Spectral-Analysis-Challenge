"""di-PLSのテスト

Issue #95 Cycle 3: di-PLSドメイン適応のアンサンブル統合
"""
import numpy as np
import pytest
from pathlib import Path

from src.preprocessing.issue43_dipls import fit_predict_dipls
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue20_savgol import apply_savgol


@pytest.fixture
def synthetic_data():
    """合成データを生成する。"""
    rng = np.random.RandomState(42)
    n_s, n_t, p = 50, 20, 30
    # 真の係数
    beta = rng.randn(p) * 0.5
    X_s = rng.randn(n_s, p)
    X_t = rng.randn(n_t, p) + 0.5  # ドメインシフト
    y_s = X_s @ beta + rng.randn(n_s) * 0.1
    return X_s, y_s, X_t, beta


class TestFitPredictDipls:
    """fit_predict_diplsの基本動作確認。"""

    def test_output_shape(self, synthetic_data):
        X_s, y_s, X_t, _ = synthetic_data
        pred = fit_predict_dipls(X_s, y_s, X_t, n_components=3, dipls_lambda=1.0)
        assert pred.shape == (X_t.shape[0],)

    def test_no_nan(self, synthetic_data):
        X_s, y_s, X_t, _ = synthetic_data
        pred = fit_predict_dipls(X_s, y_s, X_t, n_components=3, dipls_lambda=1.0)
        assert not np.any(np.isnan(pred))

    def test_lambda_zero_close_to_pls(self, synthetic_data):
        """lambda=0ではstandard PLSに近い結果になるか確認。"""
        from sklearn.cross_decomposition import PLSRegression

        X_s, y_s, X_t, _ = synthetic_data
        n_comp = 3

        # di-PLS with lambda=0
        pred_dipls = fit_predict_dipls(X_s, y_s, X_t, n_components=n_comp, dipls_lambda=0.0)

        # Standard PLS
        pls = PLSRegression(n_components=n_comp)
        pls.fit(X_s, y_s)
        pred_pls = pls.predict(X_t).ravel()

        # 完全一致ではないが相関は高いはず
        corr = np.corrcoef(pred_dipls, pred_pls)[0, 1]
        assert corr > 0.9, f"Correlation between di-PLS(lambda=0) and PLS: {corr:.4f}"

    def test_different_lambda_different_results(self, synthetic_data):
        X_s, y_s, X_t, _ = synthetic_data
        pred1 = fit_predict_dipls(X_s, y_s, X_t, n_components=3, dipls_lambda=0.1)
        pred2 = fit_predict_dipls(X_s, y_s, X_t, n_components=3, dipls_lambda=100.0)
        assert not np.allclose(pred1, pred2), "Different lambdas should give different results"

    def test_different_components(self, synthetic_data):
        X_s, y_s, X_t, _ = synthetic_data
        pred2 = fit_predict_dipls(X_s, y_s, X_t, n_components=2, dipls_lambda=1.0)
        pred5 = fit_predict_dipls(X_s, y_s, X_t, n_components=5, dipls_lambda=1.0)
        assert pred2.shape == pred5.shape
        assert not np.allclose(pred2, pred5)


class TestPreprocessingCombinations:
    """各前処理との組み合わせが動作するか確認。"""

    def test_snv_dipls(self, synthetic_data):
        X_s, y_s, X_t, _ = synthetic_data
        X_s_snv = apply_snv(X_s)
        X_t_snv = apply_snv(X_t)
        pred = fit_predict_dipls(X_s_snv, y_s, X_t_snv, n_components=3, dipls_lambda=1.0)
        assert pred.shape == (X_t.shape[0],)
        assert not np.any(np.isnan(pred))

    def test_sg2d_dipls(self, synthetic_data):
        X_s, y_s, X_t, _ = synthetic_data
        # SG2dにはある程度の特徴量数が必要（window_length=11）
        # synthetic_dataはp=30なので問題ない
        X_s_sg = apply_savgol(X_s, deriv=2, window_length=11, polyorder=2)
        X_t_sg = apply_savgol(X_t, deriv=2, window_length=11, polyorder=2)
        pred = fit_predict_dipls(X_s_sg, y_s, X_t_sg, n_components=3, dipls_lambda=1.0)
        assert pred.shape == (X_t.shape[0],)
        assert not np.any(np.isnan(pred))

    def test_epo_dipls(self, synthetic_data):
        X_s, y_s, X_t, _ = synthetic_data
        # EPOにはグループが必要 → 合成グループを作る
        groups = np.array(["A"] * 25 + ["B"] * 25)
        P = compute_epo_projection(X_s, groups, n_components=1)
        X_s_epo = apply_epo(X_s, P)
        X_t_epo = apply_epo(X_t, P)
        pred = fit_predict_dipls(X_s_epo, y_s, X_t_epo, n_components=3, dipls_lambda=1.0)
        assert pred.shape == (X_t.shape[0],)
        assert not np.any(np.isnan(pred))

    def test_sqrt_transform(self, synthetic_data):
        X_s, y_s, X_t, _ = synthetic_data
        # y_sを正にする
        y_pos = np.abs(y_s) + 1.0
        y_sqrt = np.sqrt(y_pos)
        pred_sqrt = fit_predict_dipls(X_s, y_sqrt, X_t, n_components=3, dipls_lambda=1.0)
        pred = pred_sqrt ** 2
        assert pred.shape == (X_t.shape[0],)
        assert not np.any(np.isnan(pred))
        assert np.all(pred >= 0)
