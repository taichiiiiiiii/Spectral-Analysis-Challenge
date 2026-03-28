"""Issue #63 Lite: 非線形モデル Optuna最適化 - 高速版

部分結果から有望な組み合わせに絞って実行。
SNV+XGB PLS(3) raw = 18.77 が最有望 → 前処理×PLS成分数を網羅、Optuna 15試行。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import time

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from src.eda.data_loader import load_train, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue19_msc import compute_msc_reference
from src.preprocessing.issue62_additional_preprocessing import apply_asls, apply_piecewise_msc

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def apply_pp(X_train, X_test, groups_train, method):
    if method == "SNV":
        return apply_snv(X_train), apply_snv(X_test)
    elif method == "EPO(1)":
        P = compute_epo_projection(X_train, groups_train, n_components=1)
        return apply_epo(X_train, P), apply_epo(X_test, P)
    elif method == "SNV+AsLS(1e6)":
        return apply_asls(apply_snv(X_train), lam=1e6), apply_asls(apply_snv(X_test), lam=1e6)
    elif method == "PiecewiseMSC(seg=3)":
        ref = compute_msc_reference(X_train)
        return apply_piecewise_msc(X_train, ref, 3), apply_piecewise_msc(X_test, ref, 3)
    raise ValueError(method)


def evaluate_loso(X_raw, y, groups, folds, pp_name, model_type, n_pls, target_tf, n_trials=15):
    """1つの組み合わせをLOSO-CVで評価"""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # 全foldの前処理+PLS次元削減をキャッシュ
    fold_cache = []
    for train_idx, test_idx in folds:
        try:
            X_tr_pp, X_te_pp = apply_pp(X_raw[train_idx], X_raw[test_idx], groups[train_idx], pp_name)
            nc = min(n_pls, X_tr_pp.shape[1] - 1, X_tr_pp.shape[0] - 1)
            pls = PLSRegression(n_components=nc)
            pls.fit(X_tr_pp, y[train_idx])
            X_tr_s = pls.transform(X_tr_pp)
            X_te_s = pls.transform(X_te_pp)
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr_s)
            X_te_s = sc.transform(X_te_s)
            fold_cache.append((X_tr_s, X_te_s, train_idx, test_idx))
        except Exception:
            fold_cache.append(None)

    def objective(trial):
        fold_rmses = []
        for fc in fold_cache:
            if fc is None:
                fold_rmses.append(999.0)
                continue
            X_tr, X_te, tr_idx, te_idx = fc
            y_tr = y[tr_idx].copy()
            y_te = y[te_idx]

            if target_tf == "sqrt":
                y_fit = np.sqrt(y_tr)
            else:
                y_fit = y_tr

            if model_type == "lgb":
                import lightgbm as lgb
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "max_depth": trial.suggest_int("max_depth", 2, 6),
                    "num_leaves": trial.suggest_int("num_leaves", 8, 31),
                    "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                    "verbose": -1,
                    "random_state": 42,
                }
                model = lgb.LGBMRegressor(**params)
                model.fit(X_tr, y_fit, eval_set=[(X_te, y_fit[:1])],  # dummy, won't use
                          callbacks=[lgb.log_evaluation(0)])
                # Actually, just fit without eval for speed
                model = lgb.LGBMRegressor(**params)
                model.fit(X_tr, y_fit)

            elif model_type == "xgb":
                import xgboost as xgb
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "max_depth": trial.suggest_int("max_depth", 2, 6),
                    "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                    "random_state": 42,
                    "verbosity": 0,
                }
                model = xgb.XGBRegressor(**params)
                model.fit(X_tr, y_fit)

            elif model_type == "svr":
                from sklearn.svm import SVR
                params = {
                    "C": trial.suggest_float("C", 0.1, 100.0, log=True),
                    "epsilon": trial.suggest_float("epsilon", 0.001, 1.0, log=True),
                    "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
                    "kernel": "rbf",
                }
                model = SVR(**params)
                model.fit(X_tr, y_fit)

            pred = model.predict(X_te)
            if target_tf == "sqrt":
                pred = np.clip(pred, 0, None) ** 2
            fold_rmses.append(rmse(y_te, pred))

        return np.mean(fold_rmses)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return {
        "preprocessing": pp_name,
        "model": model_type,
        "n_pls": n_pls,
        "transform": target_tf,
        "rmse": study.best_value,
        "best_params": study.best_params,
    }


def main():
    t_start = time.time()
    df_train = load_train(DATA_DIR)
    spectral_cols = get_spectral_columns(df_train)
    X_raw = df_train[spectral_cols].values
    y = df_train["含水率"].values
    groups = df_train["樹種"].values

    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_raw, y, groups))

    print("=" * 70)
    print("Issue #63 Lite: 非線形モデル Optuna最適化 (高速版)")
    print("=" * 70)

    # 有望な組み合わせに絞る
    configs = [
        # (pp, model, n_pls, transform, n_trials)
        ("SNV", "xgb", 3, "raw", 15),
        ("SNV", "xgb", 5, "raw", 15),
        ("SNV", "xgb", 3, "sqrt", 15),
        ("SNV", "lgb", 3, "raw", 15),
        ("SNV", "lgb", 5, "raw", 15),
        ("SNV", "lgb", 3, "sqrt", 15),
        ("EPO(1)", "xgb", 3, "raw", 15),
        ("EPO(1)", "lgb", 3, "raw", 15),
        ("EPO(1)", "xgb", 5, "raw", 15),
        ("SNV+AsLS(1e6)", "xgb", 3, "raw", 15),
        ("SNV+AsLS(1e6)", "xgb", 3, "sqrt", 15),
        ("SNV+AsLS(1e6)", "lgb", 3, "raw", 15),
        ("PiecewiseMSC(seg=3)", "xgb", 3, "raw", 15),
        ("PiecewiseMSC(seg=3)", "lgb", 3, "raw", 15),
        ("SNV", "svr", 3, "raw", 10),
        ("SNV", "svr", 5, "raw", 10),
        ("EPO(1)", "svr", 3, "raw", 10),
        ("SNV", "svr", 3, "sqrt", 10),
    ]

    results = []
    for idx, (pp, mdl, n_pls, tf, n_trials) in enumerate(configs, 1):
        t0 = time.time()
        print(f"\n[{idx}/{len(configs)}] {pp} | {mdl.upper()} | PLS({n_pls}) | {tf} (trials={n_trials})", flush=True)
        try:
            res = evaluate_loso(X_raw, y, groups, folds, pp, mdl, n_pls, tf, n_trials)
            results.append(res)
            elapsed = time.time() - t0
            print(f"  → RMSE={res['rmse']:.2f} ({elapsed:.0f}s)", flush=True)
        except Exception as e:
            print(f"  → ERROR: {e}", flush=True)
            results.append({"preprocessing": pp, "model": mdl, "n_pls": n_pls,
                           "transform": tf, "rmse": 999.0, "best_params": {}})

    # 結果表示
    df = pd.DataFrame(results).sort_values("rmse")
    print("\n" + "=" * 70)
    print("TOP 18 RESULTS (sorted by RMSE):")
    print("=" * 70)
    for _, row in df.iterrows():
        print(f"  RMSE={row['rmse']:.2f} | {row['preprocessing']} | {row['model'].upper()} | PLS({row['n_pls']}) | {row['transform']}")

    df.to_csv(OUT_DIR / "issue63_nonlinear_results.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'issue63_nonlinear_results.csv'}")
    print(f"総実行時間: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
