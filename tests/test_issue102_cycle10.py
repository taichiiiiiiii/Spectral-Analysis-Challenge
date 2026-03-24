"""Issue #102: Cycle 10 統合提出パイプライン v7 のテスト

各モデルタイプの予測関数、前処理、アンサンブル評価をテストする。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from scripts.modeling.run_issue102_cycle10_v7 import (
    calc_rmse,
    preprocess,
    feature_select,
    apply_target_transform,
    inverse_target_transform,
    predict_pls,
    predict_tca_ridge,
    predict_dipls,
    predict_sa_ridge,
    predict_huber,
    ensemble_eval,
    save_submission,
    MODEL_CFGS,
    PREDICT_FNS,
)


# ============================================================
# フィクスチャ
# ============================================================

@pytest.fixture
def synthetic_data():
    """テスト用の合成スペクトルデータ。"""
    rng = np.random.RandomState(42)
    n_train, n_test, n_feat = 60, 15, 50
    X_tr = rng.randn(n_train, n_feat)
    X_te = rng.randn(n_test, n_feat)
    y_tr = rng.uniform(10, 200, n_train)
    # 3グループ
    groups = np.array(["sp_A"] * 20 + ["sp_B"] * 20 + ["sp_C"] * 20)
    return X_tr, X_te, y_tr, groups


@pytest.fixture
def real_data():
    """実データの読み込み（存在する場合のみ）。"""
    data_dir = Path(__file__).resolve().parents[1] / "Input_data"
    if not (data_dir / "train.csv").exists():
        pytest.skip("train.csv not found")
    from src.eda.data_loader import load_train, load_test, get_spectral_columns
    df_train = load_train(data_dir)
    df_test = load_test(data_dir)
    spec_cols = get_spectral_columns(df_train)
    return (
        df_train[spec_cols].values,
        df_test[spec_cols].values,
        df_train["含水率"].values,
        df_train["樹種"].values,
        df_test["sample number"].values,
    )


# ============================================================
# ユーティリティテスト
# ============================================================

class TestCalcRmse:
    def test_zero_error(self):
        y = np.array([1.0, 2.0, 3.0])
        assert calc_rmse(y, y) == pytest.approx(0.0, abs=1e-10)

    def test_known_error(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        assert calc_rmse(y_true, y_pred) == pytest.approx(1.0, abs=1e-10)


class TestTargetTransform:
    def test_raw_identity(self):
        y = np.array([10.0, 50.0, 100.0])
        np.testing.assert_array_equal(apply_target_transform(y, "raw"), y)

    def test_sqrt_and_inverse(self):
        y = np.array([4.0, 16.0, 100.0])
        y_t = apply_target_transform(y, "sqrt")
        np.testing.assert_array_almost_equal(y_t, np.array([2.0, 4.0, 10.0]))
        y_back = inverse_target_transform(y_t, "sqrt")
        np.testing.assert_array_almost_equal(y_back, y)

    def test_sqrt_inverse_clips_negative(self):
        pred = np.array([-1.0, 0.0, 3.0])
        result = inverse_target_transform(pred, "sqrt")
        assert result[0] == 0.0  # clipped
        assert result[1] == 0.0
        assert result[2] == pytest.approx(9.0)


# ============================================================
# 前処理テスト
# ============================================================

class TestPreprocess:
    def test_snv_output_shape(self, synthetic_data):
        X_tr, X_te, y, groups = synthetic_data
        X_tr_pp, X_te_pp = preprocess(X_tr, X_te, groups, "SNV")
        assert X_tr_pp.shape == X_tr.shape
        assert X_te_pp.shape == X_te.shape

    def test_epo_output_shape(self, synthetic_data):
        X_tr, X_te, y, groups = synthetic_data
        X_tr_pp, X_te_pp = preprocess(X_tr, X_te, groups, "EPO(1)")
        assert X_tr_pp.shape == X_tr.shape
        assert X_te_pp.shape == X_te.shape

    def test_unknown_pp_returns_copy(self, synthetic_data):
        X_tr, X_te, y, groups = synthetic_data
        X_tr_pp, X_te_pp = preprocess(X_tr, X_te, groups, "UNKNOWN")
        np.testing.assert_array_equal(X_tr_pp, X_tr)
        np.testing.assert_array_equal(X_te_pp, X_te)

    def test_copy_independence(self, synthetic_data):
        X_tr, X_te, y, groups = synthetic_data
        X_tr_pp, _ = preprocess(X_tr, X_te, groups, "UNKNOWN")
        X_tr_pp[0, 0] = 999.0
        assert X_tr[0, 0] != 999.0  # 元データは変更されない


# ============================================================
# 予測関数テスト
# ============================================================

class TestPredictPls:
    def test_output_shape(self, synthetic_data):
        X_tr, X_te, y, groups = synthetic_data
        cfg = {"pp": "SNV", "nc": 3, "tf": "sqrt", "type": "pls"}
        pred = predict_pls(X_tr, X_te, y, groups, cfg)
        assert pred.shape == (X_te.shape[0],)

    def test_raw_transform(self, synthetic_data):
        X_tr, X_te, y, groups = synthetic_data
        cfg = {"pp": "SNV", "nc": 3, "tf": "raw", "type": "pls"}
        pred = predict_pls(X_tr, X_te, y, groups, cfg)
        assert pred.shape == (X_te.shape[0],)


class TestPredictTcaRidge:
    def test_output_shape(self, synthetic_data):
        X_tr, X_te, y, groups = synthetic_data
        cfg = {"pp": "SNV", "nc": 5, "kernel": "linear",
               "alpha": 1.0, "tf": "sqrt", "type": "tca"}
        pred = predict_tca_ridge(X_tr, X_te, y, groups, cfg)
        assert pred.shape == (X_te.shape[0],)


class TestPredictDipls:
    def test_output_shape(self, synthetic_data):
        X_tr, X_te, y, groups = synthetic_data
        cfg = {"pp": "SNV", "nc": 3, "dipls_lambda": 1.0,
               "tf": "sqrt", "type": "dipls"}
        pred = predict_dipls(X_tr, X_te, y, groups, cfg)
        assert pred.shape == (X_te.shape[0],)


class TestPredictSaRidge:
    def test_output_shape(self, synthetic_data):
        X_tr, X_te, y, groups = synthetic_data
        cfg = {"pp": "SNV", "nc": 5, "alpha": 1.0,
               "tf": "sqrt", "type": "sa"}
        pred = predict_sa_ridge(X_tr, X_te, y, groups, cfg)
        assert pred.shape == (X_te.shape[0],)


class TestPredictHuber:
    def test_output_shape(self, synthetic_data):
        X_tr, X_te, y, groups = synthetic_data
        cfg = {"pp": "SNV", "nc": 3, "tf": "sqrt",
               "eps": 1.35, "type": "huber"}
        pred = predict_huber(X_tr, X_te, y, groups, cfg)
        assert pred.shape == (X_te.shape[0],)


# ============================================================
# モデル設定テスト
# ============================================================

class TestModelCfgs:
    def test_12_models(self):
        assert len(MODEL_CFGS) == 12

    def test_all_have_required_keys(self):
        required = {"name", "pp", "nc", "tf", "type", "group"}
        for cfg in MODEL_CFGS:
            missing = required - set(cfg.keys())
            assert not missing, f"{cfg['name']} missing keys: {missing}"

    def test_all_types_in_dispatch(self):
        for cfg in MODEL_CFGS:
            assert cfg["type"] in PREDICT_FNS, \
                f"{cfg['name']} type={cfg['type']} not in PREDICT_FNS"

    def test_group_counts(self):
        groups = {}
        for cfg in MODEL_CFGS:
            g = cfg["group"]
            groups[g] = groups.get(g, 0) + 1
        assert groups["A"] == 4
        assert groups["B"] == 5
        assert groups["C"] == 3

    def test_unique_names(self):
        names = [c["name"] for c in MODEL_CFGS]
        assert len(names) == len(set(names))


# ============================================================
# アンサンブル評価テスト
# ============================================================

class TestEnsembleEval:
    def test_single_model(self):
        """1モデルのアンサンブル = そのモデル自身。"""
        rng = np.random.RandomState(0)
        y_true = rng.uniform(10, 100, 20)
        pred = y_true + rng.randn(20) * 5
        # folds: 2つ
        folds = [(np.arange(10), np.arange(10, 20)),
                 (np.arange(10, 20), np.arange(10))]
        all_preds = [[pred[10:], pred[:10]]]
        y_full = y_true
        r, fr = ensemble_eval(all_preds, folds, y_full, [0])
        assert r > 0
        assert len(fr) == 2

    def test_uniform_weights_same_as_none(self):
        """均等重み = weights=None と同じ結果。"""
        rng = np.random.RandomState(1)
        y = rng.uniform(10, 100, 20)
        p1 = y + rng.randn(20) * 5
        p2 = y + rng.randn(20) * 8
        folds = [(np.arange(10), np.arange(10, 20))]
        all_preds = [[p1[10:]], [p2[10:]]]

        r_none, _ = ensemble_eval(all_preds, folds, y, [0, 1])
        r_uniform, _ = ensemble_eval(all_preds, folds, y, [0, 1],
                                      weights=[0.5, 0.5])
        assert r_none == pytest.approx(r_uniform, abs=1e-10)


# ============================================================
# 提出ファイルテスト
# ============================================================

class TestSaveSubmission:
    def test_csv_format(self, tmp_path):
        preds = np.array([10.5, 20.3, 30.1])
        ids = np.array([95, 96, 97])
        fpath = tmp_path / "test_sub.csv"
        save_submission(preds, ids, fpath)

        import csv
        with open(fpath) as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) == 3
        assert rows[0][0] == "95"
        assert float(rows[0][1]) == pytest.approx(10.5)


# ============================================================
# 実データ統合テスト（実データがある場合のみ）
# ============================================================

class TestIntegrationWithRealData:
    def test_all_models_run_on_real_data(self, real_data):
        """実データで全12モデルが動作することを確認する。"""
        X_tr_full, X_te_full, y, groups, _ = real_data
        # サブセット: 最初の2種のみでテスト（高速化）
        unique_sp = np.unique(groups)
        sp_a, sp_b = unique_sp[0], unique_sp[1]
        mask_tr = groups == sp_a
        mask_te = groups == sp_b
        X_tr = X_tr_full[mask_tr]
        X_te = X_tr_full[mask_te]
        y_tr = y[mask_tr]
        g_tr = groups[mask_tr]

        for cfg in MODEL_CFGS:
            predict_fn = PREDICT_FNS[cfg["type"]]
            try:
                pred = predict_fn(X_tr, X_te, y_tr, g_tr, cfg)
                assert pred.shape[0] == X_te.shape[0], \
                    f"{cfg['name']}: wrong shape"
                assert np.all(np.isfinite(pred)), \
                    f"{cfg['name']}: non-finite values"
            except Exception as e:
                pytest.fail(f"{cfg['name']} failed: {e}")
