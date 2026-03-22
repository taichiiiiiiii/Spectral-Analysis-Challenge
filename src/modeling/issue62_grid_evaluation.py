"""前処理×線形モデル統合グリッド評価

対応Issue: #62
前処理10種 × モデル5種 × 目的変数変換2種 = 100パターンをLOSO-CVで評価。
"""
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.model_selection import LeaveOneGroupOut

from src.eda.data_loader import get_wavenumbers
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue19_msc import compute_msc_reference, apply_msc
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue38_tca import tca_transform
from src.preprocessing.issue62_additional_preprocessing import (
    apply_asls,
    apply_whittaker,
    apply_piecewise_msc,
    select_water_bands,
    select_water_wide,
)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _apply_preprocessing(X_train_raw, X_test_raw, groups_train, wavenumbers, method):
    """前処理を適用する。data leakage防止のためtrain側のみでfit。

    Returns
    -------
    X_train, X_test : 前処理済みデータ
    """
    if method == "Raw":
        return X_train_raw.copy(), X_test_raw.copy()

    elif method == "SNV":
        return apply_snv(X_train_raw), apply_snv(X_test_raw)

    elif method == "MSC":
        ref = compute_msc_reference(X_train_raw)
        return apply_msc(X_train_raw, ref), apply_msc(X_test_raw, ref)

    elif method == "PiecewiseMSC(seg=3)":
        ref = compute_msc_reference(X_train_raw)
        return (
            apply_piecewise_msc(X_train_raw, ref, n_segments=3),
            apply_piecewise_msc(X_test_raw, ref, n_segments=3),
        )

    elif method == "AsLS(lam=1e5)":
        return apply_asls(X_train_raw, lam=1e5), apply_asls(X_test_raw, lam=1e5)

    elif method == "SNV+AsLS(1e6)":
        X_tr_snv = apply_snv(X_train_raw)
        X_te_snv = apply_snv(X_test_raw)
        return apply_asls(X_tr_snv, lam=1e6), apply_asls(X_te_snv, lam=1e6)

    elif method == "EPO(1)":
        P = compute_epo_projection(X_train_raw, groups_train, n_components=1)
        return apply_epo(X_train_raw, P), apply_epo(X_test_raw, P)

    elif method == "TCA(10)":
        Z_tr, Z_te = tca_transform(X_train_raw, X_test_raw, n_components=10)
        return Z_tr, Z_te

    elif method == "Whittaker(lam=1e3)":
        return apply_whittaker(X_train_raw, lam=1e3), apply_whittaker(X_test_raw, lam=1e3)

    elif method == "Range:water_wide":
        return (
            select_water_wide(X_train_raw, wavenumbers),
            select_water_wide(X_test_raw, wavenumbers),
        )

    elif method == "Range:water_bands":
        return (
            select_water_bands(X_train_raw, wavenumbers),
            select_water_bands(X_test_raw, wavenumbers),
        )

    else:
        raise ValueError(f"Unknown preprocessing: {method}")


def _build_model(model_name):
    """モデルを構築する。"""
    if model_name == "PLS(4)":
        return PLSRegression(n_components=4)
    elif model_name == "PLS(2)":
        return PLSRegression(n_components=2)
    elif model_name == "Lasso(0.1)":
        return Lasso(alpha=0.1, max_iter=10000)
    elif model_name == "ElasticNet(0.1)":
        return ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
    elif model_name == "Ridge(100)":
        return Ridge(alpha=100)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _is_pls(model_name):
    return model_name.startswith("PLS")


def evaluate_single_combination(
    X_raw, y, groups, wavenumbers, folds,
    preprocess_name, model_name, transform_name,
):
    """1つの組み合わせを評価してRMSE/stdを返す。"""
    fold_rmses = []
    fold_species = []

    for train_idx, test_idx in folds:
        X_train_raw = X_raw[train_idx]
        X_test_raw = X_raw[test_idx]
        groups_train = groups[train_idx]
        y_train = y[train_idx].copy()
        y_test = y[test_idx]

        # 前処理適用
        try:
            X_train, X_test = _apply_preprocessing(
                X_train_raw, X_test_raw, groups_train, wavenumbers, preprocess_name
            )
        except Exception as e:
            fold_rmses.append(999.0)
            fold_species.append(np.unique(groups[test_idx])[0] if len(test_idx) > 0 else "?")
            continue

        # TCA/レンジ選択後はPLSの成分数を調整
        n_features = X_train.shape[1]
        model = _build_model(model_name)
        if _is_pls(model_name):
            nc = model.n_components
            if nc > n_features:
                model.n_components = max(1, n_features - 1)

        # 目的変数変換
        if transform_name == "sqrt":
            y_fit = np.sqrt(y_train)
        else:
            y_fit = y_train

        # 学習・予測
        try:
            model.fit(X_train, y_fit)
            pred = model.predict(X_test).ravel()
        except Exception:
            fold_rmses.append(999.0)
            fold_species.append(np.unique(groups[test_idx])[0] if len(test_idx) > 0 else "?")
            continue

        # 逆変換
        if transform_name == "sqrt":
            pred = np.clip(pred, 0, None) ** 2

        fold_rmses.append(rmse(y_test, pred))
        fold_species.append(np.unique(groups[test_idx])[0] if len(test_idx) > 0 else "?")

    return {
        "preprocessing": preprocess_name,
        "model": model_name,
        "transform": transform_name,
        "rmse": float(np.mean(fold_rmses)),
        "rmse_std": float(np.std(fold_rmses)),
        "fold_rmses": fold_rmses,
        "fold_species": fold_species,
    }


def _evaluate_with_cached_preprocessing(
    fold_data, y, groups, model_name, transform_name, folds,
):
    """キャッシュ済み前処理データでモデル評価する。"""
    fold_rmses = []
    fold_species = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        X_train, X_test = fold_data[fold_idx]
        if X_train is None:
            fold_rmses.append(999.0)
            fold_species.append(np.unique(groups[test_idx])[0] if len(test_idx) > 0 else "?")
            continue

        y_train = y[train_idx].copy()
        y_test = y[test_idx]

        n_features = X_train.shape[1]
        model = _build_model(model_name)
        if _is_pls(model_name):
            if model.n_components > n_features:
                model.n_components = max(1, n_features - 1)

        y_fit = np.sqrt(y_train) if transform_name == "sqrt" else y_train

        try:
            model.fit(X_train, y_fit)
            pred = model.predict(X_test).ravel()
        except Exception:
            fold_rmses.append(999.0)
            fold_species.append(np.unique(groups[test_idx])[0] if len(test_idx) > 0 else "?")
            continue

        if transform_name == "sqrt":
            pred = np.clip(pred, 0, None) ** 2

        fold_rmses.append(rmse(y_test, pred))
        fold_species.append(np.unique(groups[test_idx])[0] if len(test_idx) > 0 else "?")

    return fold_rmses, fold_species


def run_grid_evaluation(X_raw, y, groups, spectral_cols):
    """全100パターンのグリッド評価を実行する。前処理結果をキャッシュして高速化。"""
    wavenumbers = get_wavenumbers(spectral_cols)
    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_raw, y, groups))

    preprocessings = [
        "SNV", "PiecewiseMSC(seg=3)", "Range:water_wide", "Range:water_bands",
        "AsLS(lam=1e5)", "SNV+AsLS(1e6)", "MSC", "EPO(1)", "TCA(10)", "Whittaker(lam=1e3)",
    ]
    models = ["Lasso(0.1)", "ElasticNet(0.1)", "Ridge(100)", "PLS(4)", "PLS(2)"]
    transforms = ["raw", "sqrt"]

    results = []
    total = len(preprocessings) * len(models) * len(transforms)
    count = 0

    for pp in preprocessings:
        import time
        t0 = time.time()
        print(f"\n--- Preprocessing: {pp} ---", flush=True)

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
        print(f"  Preprocessing done in {t1-t0:.1f}s", flush=True)

        # キャッシュ済みデータで全モデル×変換を評価
        for mdl in models:
            for tf in transforms:
                count += 1
                fold_rmses, fold_species = _evaluate_with_cached_preprocessing(
                    fold_data, y, groups, mdl, tf, folds,
                )
                result = {
                    "preprocessing": pp,
                    "model": mdl,
                    "transform": tf,
                    "rmse": float(np.mean(fold_rmses)),
                    "rmse_std": float(np.std(fold_rmses)),
                    "fold_rmses": fold_rmses,
                    "fold_species": fold_species,
                }
                results.append(result)
                print(f"  [{count}/{total}] {pp} × {mdl} × {tf} → RMSE={result['rmse']:.2f} ± {result['rmse_std']:.2f}", flush=True)

    return pd.DataFrame(results)
