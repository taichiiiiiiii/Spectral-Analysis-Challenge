"""非線形モデル（LightGBM, XGBoost, SVR）のOptuna最適化 + LOSO-CV評価

対応Issue: #63
前処理→PLSスコア抽出→非線形モデルのパイプラインをOptuna HPOで最適化。
LOSO-CVでドメインシフトに対するロバスト性を検証。
"""
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import lightgbm as lgb
import xgboost as xgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.eda.data_loader import get_wavenumbers
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue19_msc import compute_msc_reference
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue62_additional_preprocessing import (
    apply_asls,
    apply_piecewise_msc,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_sample_weights(groups_train):
    """樹種サンプル数の不均衡に対するサンプル重みを計算。
    少数樹種のサンプルに高い重みを割り当てる。
    """
    unique, counts = np.unique(groups_train, return_counts=True)
    freq = dict(zip(unique, counts))
    median_count = np.median(counts)
    weights = np.array([median_count / freq[g] for g in groups_train])
    return weights


# ---------------------------------------------------------------------------
# 前処理
# ---------------------------------------------------------------------------

def apply_preprocessing(X_train_raw, X_test_raw, groups_train, wavenumbers, method):
    """前処理適用（data leakage防止: trainのみでfit）。"""
    if method == "SNV":
        return apply_snv(X_train_raw), apply_snv(X_test_raw)
    elif method == "EPO(1)":
        P = compute_epo_projection(X_train_raw, groups_train, n_components=1)
        return apply_epo(X_train_raw, P), apply_epo(X_test_raw, P)
    elif method == "SNV+AsLS(1e6)":
        X_tr = apply_asls(apply_snv(X_train_raw), lam=1e6)
        X_te = apply_asls(apply_snv(X_test_raw), lam=1e6)
        return X_tr, X_te
    elif method == "PiecewiseMSC(seg=3)":
        ref = compute_msc_reference(X_train_raw)
        return (
            apply_piecewise_msc(X_train_raw, ref, n_segments=3),
            apply_piecewise_msc(X_test_raw, ref, n_segments=3),
        )
    else:
        raise ValueError(f"Unknown preprocessing: {method}")


def extract_pls_scores(X_train, y_train, X_test, n_components):
    """PLSスコアを抽出する（trainでfit、testはtransformのみ）。

    Returns
    -------
    T_train, T_test : PLSスコア（n_components次元）
    """
    n_comp = min(n_components, X_train.shape[1] - 1, X_train.shape[0] - 1)
    pls = PLSRegression(n_components=n_comp, scale=False)
    pls.fit(X_train, y_train)
    T_train = pls.transform(X_train)
    T_test = pls.transform(X_test)
    return T_train, T_test


# ---------------------------------------------------------------------------
# Optunaオブジェクティブ関数
# ---------------------------------------------------------------------------

def _lgb_objective(trial, fold_data_list, n_pls):
    """LightGBMのOptuna目的関数。LOSO-CV全foldの平均RMSEを最小化。"""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "num_leaves": trial.suggest_int("num_leaves", 4, 48),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }

    fold_rmses = []
    for fd in fold_data_list:
        X_tr_pp, X_te_pp = fd["X_train_pp"], fd["X_test_pp"]
        y_tr, y_te = fd["y_train"], fd["y_test"]
        weights_tr = fd["weights"]
        use_log = fd["use_log"]

        # PLSスコア抽出
        y_fit = np.log1p(y_tr) if use_log else y_tr
        T_tr, T_te = extract_pls_scores(X_tr_pp, y_fit, X_te_pp, n_pls)

        # LightGBM学習
        model = lgb.LGBMRegressor(
            **params, verbose=-1, random_state=42, n_jobs=1,
        )
        model.fit(
            T_tr, y_fit,
            sample_weight=weights_tr,
            eval_set=[(T_te, np.log1p(y_te) if use_log else y_te)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        pred = model.predict(T_te)
        if use_log:
            pred = np.expm1(pred)
        pred = np.clip(pred, 0, None)
        fold_rmses.append(rmse(y_te, pred))

    return float(np.mean(fold_rmses))


def _xgb_objective(trial, fold_data_list, n_pls):
    """XGBoostのOptuna目的関数。"""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 5.0, log=True),
    }

    fold_rmses = []
    for fd in fold_data_list:
        X_tr_pp, X_te_pp = fd["X_train_pp"], fd["X_test_pp"]
        y_tr, y_te = fd["y_train"], fd["y_test"]
        weights_tr = fd["weights"]
        use_log = fd["use_log"]

        y_fit = np.log1p(y_tr) if use_log else y_tr
        T_tr, T_te = extract_pls_scores(X_tr_pp, y_fit, X_te_pp, n_pls)

        model = xgb.XGBRegressor(
            **params, verbosity=0, random_state=42, n_jobs=1,
            early_stopping_rounds=50,
        )
        model.fit(
            T_tr, y_fit,
            sample_weight=weights_tr,
            eval_set=[(T_te, np.log1p(y_te) if use_log else y_te)],
            verbose=False,
        )
        pred = model.predict(T_te)
        if use_log:
            pred = np.expm1(pred)
        pred = np.clip(pred, 0, None)
        fold_rmses.append(rmse(y_te, pred))

    return float(np.mean(fold_rmses))


def _svr_objective(trial, fold_data_list, n_pls):
    """SVRのOptuna目的関数。"""
    C = trial.suggest_float("C", 0.1, 1000.0, log=True)
    epsilon = trial.suggest_float("epsilon", 0.001, 1.0, log=True)
    gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
    kernel = trial.suggest_categorical("kernel", ["rbf", "linear"])

    fold_rmses = []
    for fd in fold_data_list:
        X_tr_pp, X_te_pp = fd["X_train_pp"], fd["X_test_pp"]
        y_tr, y_te = fd["y_train"], fd["y_test"]
        use_log = fd["use_log"]

        y_fit = np.log1p(y_tr) if use_log else y_tr
        T_tr, T_te = extract_pls_scores(X_tr_pp, y_fit, X_te_pp, n_pls)

        # SVRではスケーリングが必要
        scaler = StandardScaler()
        T_tr_s = scaler.fit_transform(T_tr)
        T_te_s = scaler.transform(T_te)

        model = SVR(C=C, epsilon=epsilon, gamma=gamma, kernel=kernel)
        model.fit(T_tr_s, y_fit)
        pred = model.predict(T_te_s)
        if use_log:
            pred = np.expm1(pred)
        pred = np.clip(pred, 0, None)
        fold_rmses.append(rmse(y_te, pred))

    return float(np.mean(fold_rmses))


# ---------------------------------------------------------------------------
# メイン評価関数
# ---------------------------------------------------------------------------

def prepare_fold_data(X_raw, y, groups, wavenumbers, preprocess_name, use_log):
    """全LOSO-CVフォールドの前処理済みデータを事前準備する。"""
    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_raw, y, groups))
    fold_data_list = []

    for train_idx, test_idx in folds:
        X_tr_raw = X_raw[train_idx]
        X_te_raw = X_raw[test_idx]
        g_tr = groups[train_idx]

        X_tr_pp, X_te_pp = apply_preprocessing(
            X_tr_raw, X_te_raw, g_tr, wavenumbers, preprocess_name
        )
        weights = compute_sample_weights(g_tr)

        fold_data_list.append({
            "X_train_pp": X_tr_pp,
            "X_test_pp": X_te_pp,
            "y_train": y[train_idx],
            "y_test": y[test_idx],
            "groups_train": g_tr,
            "groups_test": groups[test_idx],
            "weights": weights,
            "use_log": use_log,
            "train_idx": train_idx,
            "test_idx": test_idx,
        })

    return fold_data_list, folds


def optimize_model(
    model_type, fold_data_list, n_pls, n_trials=20
):
    """Optunaでモデルを最適化する。

    Parameters
    ----------
    model_type : "lgb", "xgb", "svr"
    fold_data_list : prepare_fold_dataの出力
    n_pls : PLSスコア次元数
    n_trials : Optuna試行回数

    Returns
    -------
    study : Optuna study
    """
    if model_type == "lgb":
        obj_fn = lambda trial: _lgb_objective(trial, fold_data_list, n_pls)
    elif model_type == "xgb":
        obj_fn = lambda trial: _xgb_objective(trial, fold_data_list, n_pls)
    elif model_type == "svr":
        obj_fn = lambda trial: _svr_objective(trial, fold_data_list, n_pls)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    study = optuna.create_study(direction="minimize")
    study.optimize(obj_fn, n_trials=n_trials, show_progress_bar=False)
    return study


def evaluate_best_model(model_type, fold_data_list, best_params, n_pls):
    """最適パラメータで各foldのRMSEと学習曲線データを取得。

    Returns
    -------
    dict : fold別RMSE、species、学習曲線
    """
    fold_rmses = []
    fold_species = []
    learning_curves = []

    for fd in fold_data_list:
        X_tr_pp, X_te_pp = fd["X_train_pp"], fd["X_test_pp"]
        y_tr, y_te = fd["y_train"], fd["y_test"]
        weights_tr = fd["weights"]
        use_log = fd["use_log"]
        sp = np.unique(fd["groups_test"])[0]

        y_fit = np.log1p(y_tr) if use_log else y_tr
        y_eval = np.log1p(y_te) if use_log else y_te
        T_tr, T_te = extract_pls_scores(X_tr_pp, y_fit, X_te_pp, n_pls)

        if model_type == "lgb":
            model = lgb.LGBMRegressor(
                **best_params, verbose=-1, random_state=42, n_jobs=1,
            )
            eval_result = {}
            model.fit(
                T_tr, y_fit,
                sample_weight=weights_tr,
                eval_set=[(T_tr, y_fit), (T_te, y_eval)],
                eval_metric="rmse",
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(-1),
                    lgb.record_evaluation(eval_result),
                ],
            )
            lc = {
                "train_loss": eval_result.get("training", {}).get("rmse", []),
                "val_loss": eval_result.get("valid_1", {}).get("rmse", []),
            }

        elif model_type == "xgb":
            model = xgb.XGBRegressor(
                **best_params, verbosity=0, random_state=42, n_jobs=1,
                early_stopping_rounds=50,
            )
            model.fit(
                T_tr, y_fit,
                sample_weight=weights_tr,
                eval_set=[(T_tr, y_fit), (T_te, y_eval)],
                verbose=False,
            )
            evals = model.evals_result()
            lc = {
                "train_loss": evals.get("validation_0", {}).get("rmse", []),
                "val_loss": evals.get("validation_1", {}).get("rmse", []),
            }

        elif model_type == "svr":
            scaler = StandardScaler()
            T_tr_s = scaler.fit_transform(T_tr)
            T_te_s = scaler.transform(T_te)
            model = SVR(**best_params)
            model.fit(T_tr_s, y_fit)
            pred = model.predict(T_te_s)
            if use_log:
                pred = np.expm1(pred)
            pred = np.clip(pred, 0, None)
            fold_rmses.append(rmse(y_te, pred))
            fold_species.append(sp)
            learning_curves.append({"train_loss": [], "val_loss": []})
            continue

        pred = model.predict(T_te)
        if use_log:
            pred = np.expm1(pred)
        pred = np.clip(pred, 0, None)
        fold_rmses.append(rmse(y_te, pred))
        fold_species.append(sp)
        learning_curves.append(lc)

    return {
        "fold_rmses": fold_rmses,
        "fold_species": fold_species,
        "mean_rmse": float(np.mean(fold_rmses)),
        "std_rmse": float(np.std(fold_rmses)),
        "learning_curves": learning_curves,
    }


def plot_learning_curves(eval_result, title, save_path):
    """学習曲線をプロットして保存。"""
    fig, axes = plt.subplots(1, min(len(eval_result["learning_curves"]), 6),
                              figsize=(4 * min(len(eval_result["learning_curves"]), 6), 3.5))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for i, (lc, sp) in enumerate(zip(eval_result["learning_curves"],
                                      eval_result["fold_species"])):
        if i >= len(axes):
            break
        ax = axes[i]
        if lc["train_loss"] and lc["val_loss"]:
            ax.plot(lc["train_loss"], label="Train", alpha=0.8, linewidth=0.8)
            ax.plot(lc["val_loss"], label="Val", alpha=0.8, linewidth=0.8)
            ax.set_title(f"{sp}\nRMSE={eval_result['fold_rmses'][i]:.1f}", fontsize=8)
            ax.set_xlabel("Iteration", fontsize=7)
            ax.set_ylabel("RMSE", fontsize=7)
            ax.legend(fontsize=6)
            ax.tick_params(labelsize=6)
        else:
            ax.text(0.5, 0.5, f"{sp}\nRMSE={eval_result['fold_rmses'][i]:.1f}",
                    ha="center", va="center", fontsize=8, transform=ax.transAxes)
            ax.set_title("No learning curve (SVR)", fontsize=8)

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_full_experiment(
    X_raw, y, groups, spectral_cols,
    preprocessings=None,
    model_types=None,
    pls_components_list=None,
    n_trials=20,
    output_dir=None,
):
    """全組み合わせの実験を実行。

    Parameters
    ----------
    X_raw : (n_samples, n_features)
    y : (n_samples,)
    groups : (n_samples,) 樹種
    spectral_cols : list[str]
    preprocessings : list[str]
    model_types : list[str]
    pls_components_list : list[int]
    n_trials : int, Optuna試行回数
    output_dir : Path

    Returns
    -------
    pd.DataFrame : 全結果
    """
    if preprocessings is None:
        preprocessings = ["SNV", "EPO(1)", "SNV+AsLS(1e6)", "PiecewiseMSC(seg=3)"]
    if model_types is None:
        model_types = ["lgb", "xgb", "svr"]
    if pls_components_list is None:
        pls_components_list = [3, 5, 8, 10]
    if output_dir is None:
        output_dir = Path("outputs/modeling")

    output_dir.mkdir(parents=True, exist_ok=True)
    wavenumbers = get_wavenumbers(spectral_cols)

    transforms = ["raw", "log1p"]
    all_results = []
    total_combos = len(preprocessings) * len(model_types) * len(transforms) * len(pls_components_list)
    combo_idx = 0

    for pp in preprocessings:
        print(f"\n{'='*70}")
        print(f"前処理: {pp}")
        print(f"{'='*70}")

        for tf in transforms:
            use_log = tf == "log1p"
            print(f"\n  目的変数変換: {tf}")

            # 全fold前処理データを事前計算
            t0 = time.time()
            fold_data_list, folds = prepare_fold_data(
                X_raw, y, groups, wavenumbers, pp, use_log
            )
            t_pp = time.time() - t0
            print(f"  前処理完了: {t_pp:.1f}s")

            for n_pls in pls_components_list:
                for mt in model_types:
                    combo_idx += 1
                    label = f"{pp} | {mt.upper()} | PLS({n_pls}) | {tf}"
                    print(f"\n  [{combo_idx}/{total_combos}] {label}")
                    t0 = time.time()

                    try:
                        # Optuna最適化
                        study = optimize_model(mt, fold_data_list, n_pls, n_trials)
                        best_params = study.best_params
                        best_val = study.best_value

                        # 最適パラメータで詳細評価
                        eval_result = evaluate_best_model(
                            mt, fold_data_list, best_params, n_pls
                        )

                        elapsed = time.time() - t0
                        print(f"    → RMSE={eval_result['mean_rmse']:.2f} ± {eval_result['std_rmse']:.2f} ({elapsed:.1f}s)")
                        print(f"    → Fold RMSE: {', '.join(f'{r:.1f}' for r in eval_result['fold_rmses'])}")
                        print(f"    → Best params: {best_params}")

                        # 学習曲線保存（ベスト結果のみ）
                        plot_title = f"{pp} | {mt.upper()} | PLS({n_pls}) | {tf}\nRMSE={eval_result['mean_rmse']:.2f}"
                        plot_path = output_dir / f"issue63_lc_{pp}_{mt}_pls{n_pls}_{tf}.png"
                        try:
                            plot_learning_curves(eval_result, plot_title, plot_path)
                        except Exception as e:
                            print(f"    [WARN] 学習曲線保存失敗: {e}")

                        all_results.append({
                            "preprocessing": pp,
                            "model": mt.upper(),
                            "n_pls": n_pls,
                            "transform": tf,
                            "rmse": eval_result["mean_rmse"],
                            "rmse_std": eval_result["std_rmse"],
                            "fold_rmses": eval_result["fold_rmses"],
                            "fold_species": eval_result["fold_species"],
                            "best_params": str(best_params),
                            "optuna_best_value": best_val,
                            "elapsed_sec": elapsed,
                        })

                    except Exception as e:
                        elapsed = time.time() - t0
                        print(f"    → ERROR: {e} ({elapsed:.1f}s)")
                        all_results.append({
                            "preprocessing": pp,
                            "model": mt.upper(),
                            "n_pls": n_pls,
                            "transform": tf,
                            "rmse": 999.0,
                            "rmse_std": 0.0,
                            "fold_rmses": [],
                            "fold_species": [],
                            "best_params": "",
                            "optuna_best_value": 999.0,
                            "elapsed_sec": elapsed,
                        })

    df_results = pd.DataFrame(all_results)
    return df_results
