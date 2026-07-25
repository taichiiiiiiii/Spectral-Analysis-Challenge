"""Issue #93 Cycle1: LB最適化パイプライン v6 ユニットテスト"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
import tempfile
import os

from scripts.modeling.run_issue93_cycle1_lb import (
    rmse,
    ensemble_eval,
    optimize_weights_regularized,
    strategy_a_topk_uniform,
    strategy_c_pls_only,
    save_submission,
    CFGS,
)


# --- テストデータ生成 ---

def make_dummy_folds_and_preds(n_models=5, n_folds=3, n_samples_per_fold=10):
    """ダミーのfold・予測データを生成"""
    np.random.seed(42)
    y_all = np.random.uniform(10, 200, n_folds * n_samples_per_fold)
    folds = []
    offset = 0
    for fi in range(n_folds):
        te = np.arange(offset, offset + n_samples_per_fold)
        tr = np.array([j for j in range(len(y_all)) if j not in te])
        folds.append((tr, te))
        offset += n_samples_per_fold

    all_preds = []
    for m in range(n_models):
        preds = []
        for fi in range(n_folds):
            _, te = folds[fi]
            p = y_all[te] + np.random.normal(0, 5, len(te))
            preds.append(p)
        all_preds.append(preds)

    return all_preds, folds, y_all


# --- テストケース ---

class TestRmse:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        assert rmse(y_true, y_pred) == pytest.approx(1.0)


class TestEnsembleEval:
    def test_uniform_weights(self):
        all_preds, folds, y = make_dummy_folds_and_preds(n_models=3)
        avg_rmse, fold_rmses = ensemble_eval(all_preds, folds, y, [0, 1, 2])
        assert isinstance(avg_rmse, float)
        assert len(fold_rmses) == 3
        assert all(r >= 0 for r in fold_rmses)

    def test_explicit_weights(self):
        all_preds, folds, y = make_dummy_folds_and_preds(n_models=3)
        w = [0.5, 0.3, 0.2]
        avg_rmse, fold_rmses = ensemble_eval(all_preds, folds, y, [0, 1, 2], w)
        assert isinstance(avg_rmse, float)
        assert len(fold_rmses) == 3

    def test_single_model(self):
        all_preds, folds, y = make_dummy_folds_and_preds(n_models=3)
        avg_rmse, fold_rmses = ensemble_eval(all_preds, folds, y, [0])
        assert isinstance(avg_rmse, float)
        # 単一モデルは重みなしと同じ
        avg_rmse_w, _ = ensemble_eval(all_preds, folds, y, [0], [1.0])
        assert avg_rmse == pytest.approx(avg_rmse_w)

    def test_weight_normalization(self):
        """重みが正規化されることを確認"""
        all_preds, folds, y = make_dummy_folds_and_preds(n_models=2)
        r1, _ = ensemble_eval(all_preds, folds, y, [0, 1], [1.0, 1.0])
        r2, _ = ensemble_eval(all_preds, folds, y, [0, 1], [0.5, 0.5])
        assert r1 == pytest.approx(r2)


class TestRegularization:
    def test_regularized_optimization(self):
        """正則化項が正しく動作: alpha大 → 均等重みに近づく"""
        all_preds, folds, y = make_dummy_folds_and_preds(n_models=4)
        idx = list(range(4))
        nn = len(idx)
        w_uniform = np.ones(nn) / nn

        # alpha=0 (正則化なし)
        w_low, _ = optimize_weights_regularized(all_preds, folds, y, idx, alpha=0.0, n_restarts=5)
        # alpha=10 (強い正則化)
        w_high, _ = optimize_weights_regularized(all_preds, folds, y, idx, alpha=10.0, n_restarts=5)

        # 強い正則化 → 均等重みに近い
        dist_low = np.sum((w_low - w_uniform) ** 2)
        dist_high = np.sum((w_high - w_uniform) ** 2)
        assert dist_high <= dist_low + 1e-6, \
            f"強い正則化でも均等重みに近づかない: dist_high={dist_high:.4f}, dist_low={dist_low:.4f}"


class TestStrategyA:
    def test_topk_returns_correct_k(self):
        all_preds, folds, y = make_dummy_folds_and_preds(n_models=5)
        scores = [(np.random.uniform(5, 20), i) for i in range(5)]
        results = strategy_a_topk_uniform(all_preds, folds, y, scores, ks=(2, 3))
        assert len(results) == 2
        assert len(results[0]["idx"]) == 2
        assert len(results[1]["idx"]) == 3

    def test_uniform_weights_assigned(self):
        all_preds, folds, y = make_dummy_folds_and_preds(n_models=5)
        scores = [(np.random.uniform(5, 20), i) for i in range(5)]
        results = strategy_a_topk_uniform(all_preds, folds, y, scores, ks=(3,))
        w = results[0]["weights"]
        assert len(w) == 3
        assert np.allclose(w, 1.0 / 3.0)


class TestStrategyC:
    def test_pls_only_filter(self):
        """PLS系モデルのみがフィルタされるか"""
        results = strategy_c_pls_only(
            [[] for _ in range(len(CFGS))],  # ダミー（実際に評価はしない）
            [],  # ダミー
            np.array([]),
            CFGS,
        )
        # PLS系のインデックスが正しいか
        pls_idx = results[0]["idx"]
        for i in pls_idx:
            assert CFGS[i]["type"] == "pls", f"Index {i} ({CFGS[i]['name']}) is not PLS"
        # GBR/Huberが含まれないか
        non_pls = [i for i in range(len(CFGS)) if CFGS[i]["type"] != "pls"]
        for i in non_pls:
            assert i not in pls_idx


class TestSubmission:
    def test_format(self):
        """提出ファイルのフォーマット検証"""
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = Path(tmpdir) / "test_sub.csv"
            preds = np.array([100.5, 200.3, 50.0])
            samples = np.array([95, 96, 97])
            save_submission(preds, samples, fname)

            # ファイルが存在
            assert fname.exists()

            # ヘッダーなし、2列
            content = fname.read_text().strip().split("\n")
            assert len(content) == 3
            for line in content:
                parts = line.split(",")
                assert len(parts) == 2
                int(parts[0])  # sample_numberは整数
                float(parts[1])  # predictionは数値

    def test_values_match(self):
        """値が正しく保存されるか"""
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = Path(tmpdir) / "test_sub.csv"
            preds = np.array([100.5, 200.3, 50.0])
            samples = np.array([95, 96, 97])
            save_submission(preds, samples, fname)

            df = pd.read_csv(fname, header=None, names=["sample_number", "prediction"])
            assert list(df["sample_number"]) == [95, 96, 97]
            np.testing.assert_array_almost_equal(df["prediction"].values, preds)


class TestModelConfig:
    def test_15_models_defined(self):
        """15モデルが定義されているか"""
        assert len(CFGS) == 15

    def test_model_types(self):
        """モデルタイプが正しいか"""
        types = {c["type"] for c in CFGS}
        assert types == {"pls", "gbr", "huber"}

    def test_pls_count(self):
        """PLS系モデル数"""
        pls_count = sum(1 for c in CFGS if c["type"] == "pls")
        assert pls_count == 6

    def test_required_keys(self):
        """必須キーが存在するか"""
        for c in CFGS:
            assert "name" in c
            assert "pp" in c
            assert "tf" in c
            assert "type" in c
