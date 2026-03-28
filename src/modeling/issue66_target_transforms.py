"""目的変数変換の包括的比較実験

対応Issue: #66
9種の目的変数変換 × 4種の前処理 × 4種のモデル = 144パターンをLOSO-CVで評価。
ただしlog1pはPLS(1-3)のみで評価（高成分数でRMSE発散のため）。
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

from src.eda.data_loader import get_wavenumbers
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue19_msc import compute_msc_reference
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue62_additional_preprocessing import (
    apply_asls,
    apply_piecewise_msc,
)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ---------------------------------------------------------------------------
# 前処理
# ---------------------------------------------------------------------------

def _apply_preprocessing(X_train_raw, X_test_raw, groups_train, wavenumbers, method):
    """前処理を適用する。data leakage防止のためtrain側のみでfit。"""
    if method == "SNV":
        return apply_snv(X_train_raw), apply_snv(X_test_raw)

    elif method == "EPO(1)":
        P = compute_epo_projection(X_train_raw, groups_train, n_components=1)
        return apply_epo(X_train_raw, P), apply_epo(X_test_raw, P)

    elif method == "SNV+AsLS(1e6)":
        X_tr_snv = apply_snv(X_train_raw)
        X_te_snv = apply_snv(X_test_raw)
        return apply_asls(X_tr_snv, lam=1e6), apply_asls(X_te_snv, lam=1e6)

    elif method == "PiecewiseMSC(seg=3)":
        ref = compute_msc_reference(X_train_raw)
        return (
            apply_piecewise_msc(X_train_raw, ref, n_segments=3),
            apply_piecewise_msc(X_test_raw, ref, n_segments=3),
        )

    else:
        raise ValueError(f"Unknown preprocessing: {method}")


# ---------------------------------------------------------------------------
# モデル
# ---------------------------------------------------------------------------

def _build_model(model_name):
    """モデルを構築する。"""
    if model_name == "PLS(2)":
        return PLSRegression(n_components=2)
    elif model_name == "PLS(4)":
        return PLSRegression(n_components=4)
    elif model_name == "PLS(1)":
        return PLSRegression(n_components=1)
    elif model_name == "PLS(3)":
        return PLSRegression(n_components=3)
    elif model_name == "Lasso(0.1)":
        return Lasso(alpha=0.1, max_iter=10000)
    elif model_name == "Ridge(100)":
        return Ridge(alpha=100)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _is_pls(model_name):
    return model_name.startswith("PLS")


# ---------------------------------------------------------------------------
# 目的変数変換クラス
# ---------------------------------------------------------------------------

class RawTransform:
    """変換なし（ベースライン）"""
    name = "raw"

    def fit_transform(self, y_train):
        return y_train.copy()

    def inverse_transform(self, y_pred):
        return y_pred.copy()


class SqrtTransform:
    """平方根変換"""
    name = "sqrt"

    def fit_transform(self, y_train):
        return np.sqrt(np.clip(y_train, 0, None))

    def inverse_transform(self, y_pred):
        return np.clip(y_pred, 0, None) ** 2


class Log1pTransform:
    """log(1+y)変換 — PLSの低成分数(1-3)のみ使用"""
    name = "log1p"

    def fit_transform(self, y_train):
        return np.log1p(np.clip(y_train, 0, None))

    def inverse_transform(self, y_pred):
        return np.expm1(y_pred)


class BoxCoxTransform:
    """Box-Cox変換（y > 0 必須）"""
    name = "boxcox"

    def __init__(self):
        self._lmbda = None

    def fit_transform(self, y_train):
        y_pos = np.clip(y_train, 1e-6, None)  # Box-Cox requires y > 0
        y_transformed, self._lmbda = stats.boxcox(y_pos)
        return y_transformed

    def inverse_transform(self, y_pred):
        if self._lmbda == 0:
            result = np.exp(y_pred)
        else:
            # inverse: y = ((y_pred * lmbda) + 1) ^ (1/lmbda)
            inner = y_pred * self._lmbda + 1
            inner = np.clip(inner, 1e-10, None)  # 安全のためclip
            result = inner ** (1.0 / self._lmbda)
        return np.clip(result, 0, None)


class YeoJohnsonTransform:
    """Yeo-Johnson変換（y >= 0 対応）"""
    name = "yeo_johnson"

    def __init__(self):
        self._pt = PowerTransformer(method="yeo-johnson", standardize=True)

    def fit_transform(self, y_train):
        return self._pt.fit_transform(y_train.reshape(-1, 1)).ravel()

    def inverse_transform(self, y_pred):
        result = self._pt.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        return np.clip(result, 0, None)


class PowerTransform025:
    """y^0.25変換"""
    name = "power_0.25"

    def fit_transform(self, y_train):
        return np.clip(y_train, 0, None) ** 0.25

    def inverse_transform(self, y_pred):
        return np.clip(y_pred, 0, None) ** 4


class PowerTransform033:
    """y^(1/3) 立方根変換"""
    name = "power_0.33"

    def fit_transform(self, y_train):
        return np.cbrt(y_train)  # cbrt handles negative values

    def inverse_transform(self, y_pred):
        return np.clip(y_pred, 0, None) ** 3


class QuantileTransformWrap:
    """QuantileTransformer (uniform output)"""
    name = "quantile_uniform"

    def __init__(self):
        self._qt = QuantileTransformer(
            n_quantiles=100, output_distribution="uniform", random_state=42
        )

    def fit_transform(self, y_train):
        return self._qt.fit_transform(y_train.reshape(-1, 1)).ravel()

    def inverse_transform(self, y_pred):
        # clip to [0, 1] for uniform distribution before inverse
        y_clipped = np.clip(y_pred, 0, 1)
        result = self._qt.inverse_transform(y_clipped.reshape(-1, 1)).ravel()
        return np.clip(result, 0, None)


class RankGaussTransform:
    """RankGauss: QuantileTransformer with output_distribution='normal'"""
    name = "rank_gauss"

    def __init__(self):
        self._qt = QuantileTransformer(
            n_quantiles=100, output_distribution="normal", random_state=42
        )

    def fit_transform(self, y_train):
        return self._qt.fit_transform(y_train.reshape(-1, 1)).ravel()

    def inverse_transform(self, y_pred):
        result = self._qt.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        return np.clip(result, 0, None)


# ---------------------------------------------------------------------------
# 変換一覧
# ---------------------------------------------------------------------------

ALL_TRANSFORMS = [
    RawTransform,
    SqrtTransform,
    Log1pTransform,
    BoxCoxTransform,
    YeoJohnsonTransform,
    PowerTransform025,
    PowerTransform033,
    QuantileTransformWrap,
    RankGaussTransform,
]


# ---------------------------------------------------------------------------
# グリッド評価
# ---------------------------------------------------------------------------

def run_target_transform_evaluation(X_raw, y, groups, spectral_cols):
    """目的変数変換の包括的グリッド評価を実行する。

    Parameters
    ----------
    X_raw : スペクトルデータ (n_samples, n_features)
    y : 含水率 (n_samples,)
    groups : 樹種 (n_samples,)
    spectral_cols : スペクトル列名リスト

    Returns
    -------
    pd.DataFrame : 全結果
    """
    wavenumbers = get_wavenumbers(spectral_cols)
    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_raw, y, groups))

    preprocessings = ["SNV", "EPO(1)", "SNV+AsLS(1e6)", "PiecewiseMSC(seg=3)"]
    models = ["PLS(2)", "PLS(4)", "Lasso(0.1)", "Ridge(100)"]
    # log1p用の低成分数PLS
    log1p_models = ["PLS(1)", "PLS(2)", "PLS(3)"]

    results = []
    count = 0

    for pp in preprocessings:
        import time
        t0 = time.time()
        print(f"\n{'='*60}", flush=True)
        print(f"Preprocessing: {pp}", flush=True)
        print(f"{'='*60}", flush=True)

        # 全foldの前処理結果をキャッシュ
        fold_data = []
        for train_idx, test_idx in folds:
            try:
                X_tr, X_te = _apply_preprocessing(
                    X_raw[train_idx], X_raw[test_idx],
                    groups[train_idx], wavenumbers, pp,
                )
                fold_data.append((X_tr, X_te))
            except Exception as e:
                print(f"  Error in fold: {e}", flush=True)
                fold_data.append((None, None))

        t1 = time.time()
        print(f"  Preprocessing cached in {t1 - t0:.1f}s", flush=True)

        # 全変換 × モデルを評価
        for TransformClass in ALL_TRANSFORMS:
            tf_name = TransformClass.name

            # log1pはPLS(1-3)のみ
            if tf_name == "log1p":
                current_models = log1p_models
            else:
                current_models = models

            for mdl_name in current_models:
                count += 1
                fold_rmses = []
                fold_species = []

                for fold_idx, (train_idx, test_idx) in enumerate(folds):
                    X_train, X_test = fold_data[fold_idx]
                    if X_train is None:
                        fold_rmses.append(999.0)
                        sp = np.unique(groups[test_idx])
                        fold_species.append(sp[0] if len(sp) > 0 else "?")
                        continue

                    y_train = y[train_idx].copy()
                    y_test = y[test_idx]

                    # 変換オブジェクトをfold毎に新規作成（fitをleak防止）
                    transformer = TransformClass()

                    try:
                        y_fit = transformer.fit_transform(y_train)
                    except Exception as e:
                        fold_rmses.append(999.0)
                        sp = np.unique(groups[test_idx])
                        fold_species.append(sp[0] if len(sp) > 0 else "?")
                        continue

                    # モデル構築
                    n_features = X_train.shape[1]
                    model = _build_model(mdl_name)
                    if _is_pls(mdl_name):
                        if model.n_components > n_features:
                            model.n_components = max(1, n_features - 1)

                    # 学習・予測
                    try:
                        model.fit(X_train, y_fit)
                        pred = model.predict(X_test).ravel()
                    except Exception:
                        fold_rmses.append(999.0)
                        sp = np.unique(groups[test_idx])
                        fold_species.append(sp[0] if len(sp) > 0 else "?")
                        continue

                    # 逆変換
                    try:
                        pred = transformer.inverse_transform(pred)
                    except Exception:
                        fold_rmses.append(999.0)
                        sp = np.unique(groups[test_idx])
                        fold_species.append(sp[0] if len(sp) > 0 else "?")
                        continue

                    # 負の予測値をclip
                    pred = np.clip(pred, 0, None)

                    fold_rmses.append(rmse(y_test, pred))
                    sp = np.unique(groups[test_idx])
                    fold_species.append(sp[0] if len(sp) > 0 else "?")

                result = {
                    "preprocessing": pp,
                    "model": mdl_name,
                    "transform": tf_name,
                    "rmse": float(np.mean(fold_rmses)),
                    "rmse_std": float(np.std(fold_rmses)),
                    "fold_rmses": fold_rmses,
                    "fold_species": fold_species,
                }
                results.append(result)
                print(
                    f"  [{count:3d}] {pp:20s} × {mdl_name:10s} × {tf_name:18s} "
                    f"→ RMSE={result['rmse']:.2f} ± {result['rmse_std']:.2f}",
                    flush=True,
                )

    return pd.DataFrame(results)
