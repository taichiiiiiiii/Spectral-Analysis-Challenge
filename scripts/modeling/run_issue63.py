"""Issue #63: 非線形モデル（LightGBM, XGBoost, SVR）Optuna最適化実験

2段階方式:
  Stage1: 前処理×モデル×PLS成分数の組み合わせを少ないOptuna試行で高速スクリーニング
  Stage2: Top結果のみフル評価（学習曲線出力）

前処理4種 × モデル3種 × PLS成分数2種 × 目的変数変換2種 = 48パターン
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import optuna
import lightgbm as lgb
import xgboost as xgb
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.eda.data_loader import load_train, get_spectral_columns, get_wavenumbers
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue19_msc import compute_msc_reference
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue62_additional_preprocessing import (
    apply_asls,
    apply_piecewise_msc,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_sample_weights(groups_train):
    unique, counts = np.unique(groups_train, return_counts=True)
    freq = dict(zip(unique, counts))
    median_count = np.median(counts)
    return np.array([median_count / freq[g] for g in groups_train])


def apply_preprocessing(X_train_raw, X_test_raw, groups_train, method):
    if method == "SNV":
        return apply_snv(X_train_raw), apply_snv(X_test_raw)
    elif method == "EPO(1)":
        P = compute_epo_projection(X_train_raw, groups_train, n_components=1)
        return apply_epo(X_train_raw, P), apply_epo(X_test_raw, P)
    elif method == "SNV+AsLS(1e6)":
        return apply_asls(apply_snv(X_train_raw), lam=1e6), apply_asls(apply_snv(X_test_raw), lam=1e6)
    elif method == "PiecewiseMSC(seg=3)":
        ref = compute_msc_reference(X_train_raw)
        return (apply_piecewise_msc(X_train_raw, ref, n_segments=3),
                apply_piecewise_msc(X_test_raw, ref, n_segments=3))
    else:
        raise ValueError(f"Unknown: {method}")


def extract_pls_scores(X_train, y_train, X_test, n_components):
    n_comp = min(n_components, X_train.shape[1] - 1, X_train.shape[0] - 1)
    pls = PLSRegression(n_components=n_comp, scale=False)
    pls.fit(X_train, y_train)
    return pls.transform(X_train), pls.transform(X_test)


def prepare_all_folds(X_raw, y, groups, preprocessings):
    """全前処理×全foldのデータを事前キャッシュ。"""
    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_raw, y, groups))
    cache = {}
    for pp in preprocessings:
        t0 = time.time()
        fold_data = []
        for train_idx, test_idx in folds:
            try:
                X_tr_pp, X_te_pp = apply_preprocessing(
                    X_raw[train_idx], X_raw[test_idx], groups[train_idx], pp
                )
            except Exception as e:
                print(f"  [WARN] {pp} fold error: {e}")
                X_tr_pp, X_te_pp = None, None
            fold_data.append({
                "X_train_pp": X_tr_pp,
                "X_test_pp": X_te_pp,
                "y_train": y[train_idx],
                "y_test": y[test_idx],
                "groups_train": groups[train_idx],
                "groups_test": groups[test_idx],
                "weights": compute_sample_weights(groups[train_idx]),
                "train_idx": train_idx,
                "test_idx": test_idx,
            })
        cache[pp] = fold_data
        print(f"  前処理キャッシュ完了: {pp} ({time.time()-t0:.1f}s)")
    return cache, folds


def run_optuna_lgb(fold_data, n_pls, use_log, n_trials):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 800),
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
        for fd in fold_data:
            if fd["X_train_pp"] is None:
                fold_rmses.append(999.0)
                continue
            y_tr = np.log1p(fd["y_train"]) if use_log else fd["y_train"]
            y_te_eval = np.log1p(fd["y_test"]) if use_log else fd["y_test"]
            T_tr, T_te = extract_pls_scores(fd["X_train_pp"], y_tr, fd["X_test_pp"], n_pls)
            model = lgb.LGBMRegressor(**params, verbose=-1, random_state=42, n_jobs=1)
            model.fit(T_tr, y_tr, sample_weight=fd["weights"],
                      eval_set=[(T_te, y_te_eval)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
            pred = model.predict(T_te)
            if use_log:
                pred = np.expm1(pred)
            pred = np.clip(pred, 0, None)
            fold_rmses.append(rmse(fd["y_test"], pred))
        return float(np.mean(fold_rmses))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def run_optuna_xgb(fold_data, n_pls, use_log, n_trials):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 800),
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
        for fd in fold_data:
            if fd["X_train_pp"] is None:
                fold_rmses.append(999.0)
                continue
            y_tr = np.log1p(fd["y_train"]) if use_log else fd["y_train"]
            y_te_eval = np.log1p(fd["y_test"]) if use_log else fd["y_test"]
            T_tr, T_te = extract_pls_scores(fd["X_train_pp"], y_tr, fd["X_test_pp"], n_pls)
            model = xgb.XGBRegressor(**params, verbosity=0, random_state=42, n_jobs=1,
                                      early_stopping_rounds=50)
            model.fit(T_tr, y_tr, sample_weight=fd["weights"],
                      eval_set=[(T_te, y_te_eval)], verbose=False)
            pred = model.predict(T_te)
            if use_log:
                pred = np.expm1(pred)
            pred = np.clip(pred, 0, None)
            fold_rmses.append(rmse(fd["y_test"], pred))
        return float(np.mean(fold_rmses))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def run_optuna_svr(fold_data, n_pls, use_log, n_trials):
    def objective(trial):
        C = trial.suggest_float("C", 0.1, 1000.0, log=True)
        epsilon = trial.suggest_float("epsilon", 0.001, 1.0, log=True)
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        kernel = trial.suggest_categorical("kernel", ["rbf", "linear"])
        fold_rmses = []
        for fd in fold_data:
            if fd["X_train_pp"] is None:
                fold_rmses.append(999.0)
                continue
            y_tr = np.log1p(fd["y_train"]) if use_log else fd["y_train"]
            T_tr, T_te = extract_pls_scores(fd["X_train_pp"], y_tr, fd["X_test_pp"], n_pls)
            scaler = StandardScaler()
            T_tr_s = scaler.fit_transform(T_tr)
            T_te_s = scaler.transform(T_te)
            model = SVR(C=C, epsilon=epsilon, gamma=gamma, kernel=kernel)
            model.fit(T_tr_s, y_tr)
            pred = model.predict(T_te_s)
            if use_log:
                pred = np.expm1(pred)
            pred = np.clip(pred, 0, None)
            fold_rmses.append(rmse(fd["y_test"], pred))
        return float(np.mean(fold_rmses))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def evaluate_best_with_curves(model_type, fold_data, best_params, n_pls, use_log):
    """最適パラメータでfold別RMSE + 学習曲線取得。"""
    fold_rmses = []
    fold_species = []
    learning_curves = []

    for fd in fold_data:
        if fd["X_train_pp"] is None:
            fold_rmses.append(999.0)
            fold_species.append("?")
            learning_curves.append({"train_loss": [], "val_loss": []})
            continue

        sp = np.unique(fd["groups_test"])[0]
        y_tr = np.log1p(fd["y_train"]) if use_log else fd["y_train"]
        y_te_eval = np.log1p(fd["y_test"]) if use_log else fd["y_test"]
        T_tr, T_te = extract_pls_scores(fd["X_train_pp"], y_tr, fd["X_test_pp"], n_pls)

        if model_type == "LGB":
            eval_result = {}
            model = lgb.LGBMRegressor(**best_params, verbose=-1, random_state=42, n_jobs=1)
            model.fit(T_tr, y_tr, sample_weight=fd["weights"],
                      eval_set=[(T_tr, y_tr), (T_te, y_te_eval)], eval_metric="rmse",
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(-1), lgb.record_evaluation(eval_result)])
            lc = {"train_loss": eval_result.get("training", {}).get("rmse", []),
                  "val_loss": eval_result.get("valid_1", {}).get("rmse", [])}
            pred = model.predict(T_te)

        elif model_type == "XGB":
            model = xgb.XGBRegressor(**best_params, verbosity=0, random_state=42, n_jobs=1,
                                      early_stopping_rounds=50)
            model.fit(T_tr, y_tr, sample_weight=fd["weights"],
                      eval_set=[(T_tr, y_tr), (T_te, y_te_eval)], verbose=False)
            evals = model.evals_result()
            lc = {"train_loss": evals.get("validation_0", {}).get("rmse", []),
                  "val_loss": evals.get("validation_1", {}).get("rmse", [])}
            pred = model.predict(T_te)

        elif model_type == "SVR":
            scaler = StandardScaler()
            T_tr_s = scaler.fit_transform(T_tr)
            T_te_s = scaler.transform(T_te)
            model = SVR(**best_params)
            model.fit(T_tr_s, y_tr)
            pred = model.predict(T_te_s)
            lc = {"train_loss": [], "val_loss": []}

        if use_log:
            pred = np.expm1(pred)
        pred = np.clip(pred, 0, None)
        fold_rmses.append(rmse(fd["y_test"], pred))
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
    n_plots = min(len(eval_result["learning_curves"]), 13)
    ncols = min(n_plots, 5)
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for i in range(n_plots):
        ax = axes[i]
        lc = eval_result["learning_curves"][i]
        sp = eval_result["fold_species"][i]
        r = eval_result["fold_rmses"][i]
        if lc["train_loss"] and lc["val_loss"]:
            ax.plot(lc["train_loss"], label="Train", alpha=0.8, linewidth=0.7)
            ax.plot(lc["val_loss"], label="Val", alpha=0.8, linewidth=0.7)
            ax.legend(fontsize=5)
        ax.set_title(f"{sp} RMSE={r:.1f}", fontsize=7)
        ax.tick_params(labelsize=5)

    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 70)
    print("Issue #63: 非線形モデル Optuna最適化 LOSO-CV実験")
    print("=" * 70)

    DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
    df_train = load_train(DATA_DIR)
    spectral_cols = get_spectral_columns(df_train)
    X_raw = df_train[spectral_cols].values
    y = df_train["含水率"].values
    groups = df_train["樹種"].values

    print(f"Train shape: {X_raw.shape}")
    print(f"Target range: {y.min():.1f} - {y.max():.1f}")
    unique_sp, sp_counts = np.unique(groups, return_counts=True)
    print("樹種別サンプル数:")
    for sp, cnt in sorted(zip(unique_sp, sp_counts), key=lambda x: -x[1]):
        print(f"  {sp}: {cnt}")

    preprocessings = ["SNV", "EPO(1)", "SNV+AsLS(1e6)", "PiecewiseMSC(seg=3)"]
    model_types = ["LGB", "XGB", "SVR"]
    pls_list = [5, 10]
    transforms = ["raw", "log1p"]
    n_trials_map = {"LGB": 10, "XGB": 10, "SVR": 8}

    output_dir = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(preprocessings) * len(model_types) * len(pls_list) * len(transforms)
    print(f"\n実験設定: {total}パターン")
    print(f"  前処理: {preprocessings}")
    print(f"  モデル: {model_types}")
    print(f"  PLS成分数: {pls_list}")
    print(f"  目的変数変換: {transforms}")
    print(f"  Optuna試行回数: {n_trials_map}")

    # 全前処理を事前キャッシュ
    print("\n--- 前処理キャッシュ ---")
    cache, folds = prepare_all_folds(X_raw, y, groups, preprocessings)

    all_results = []
    combo_idx = 0
    t_start = time.time()

    for pp in preprocessings:
        fold_data = cache[pp]
        print(f"\n{'='*60}")
        print(f"前処理: {pp}")
        print(f"{'='*60}")

        for tf in transforms:
            use_log = tf == "log1p"
            for n_pls in pls_list:
                for mt in model_types:
                    combo_idx += 1
                    n_trials = n_trials_map[mt]
                    label = f"{pp} | {mt} | PLS({n_pls}) | {tf}"
                    t0 = time.time()

                    try:
                        if mt == "LGB":
                            study = run_optuna_lgb(fold_data, n_pls, use_log, n_trials)
                        elif mt == "XGB":
                            study = run_optuna_xgb(fold_data, n_pls, use_log, n_trials)
                        elif mt == "SVR":
                            study = run_optuna_svr(fold_data, n_pls, use_log, n_trials)

                        best_params = study.best_params
                        eval_result = evaluate_best_with_curves(mt, fold_data, best_params, n_pls, use_log)
                        elapsed = time.time() - t0

                        print(f"  [{combo_idx:2d}/{total}] {label} → RMSE={eval_result['mean_rmse']:.2f} ± {eval_result['std_rmse']:.2f} ({elapsed:.0f}s)")
                        fold_str = ", ".join(f"{s}:{r:.1f}" for s, r in
                                             zip(eval_result["fold_species"], eval_result["fold_rmses"]))
                        print(f"         Folds: {fold_str}")

                        # 学習曲線保存
                        try:
                            safe_pp = pp.replace("(", "").replace(")", "").replace("+", "_")
                            plot_path = output_dir / f"issue63_lc_{safe_pp}_{mt}_pls{n_pls}_{tf}.png"
                            plot_title = f"{label}\nRMSE={eval_result['mean_rmse']:.2f} +/- {eval_result['std_rmse']:.2f}"
                            plot_learning_curves(eval_result, plot_title, plot_path)
                        except Exception as e:
                            print(f"         [WARN] 学習曲線保存失敗: {e}")

                        all_results.append({
                            "preprocessing": pp,
                            "model": mt,
                            "n_pls": n_pls,
                            "transform": tf,
                            "rmse": eval_result["mean_rmse"],
                            "rmse_std": eval_result["std_rmse"],
                            "fold_rmses_str": fold_str,
                            "best_params": str(best_params),
                            "elapsed_sec": elapsed,
                        })

                    except Exception as e:
                        elapsed = time.time() - t0
                        print(f"  [{combo_idx:2d}/{total}] {label} → ERROR: {e} ({elapsed:.0f}s)")
                        all_results.append({
                            "preprocessing": pp,
                            "model": mt,
                            "n_pls": n_pls,
                            "transform": tf,
                            "rmse": 999.0,
                            "rmse_std": 0.0,
                            "fold_rmses_str": "",
                            "best_params": "",
                            "elapsed_sec": elapsed,
                        })

    total_time = time.time() - t_start
    df_results = pd.DataFrame(all_results)

    # CSV保存
    csv_path = output_dir / "issue63_nonlinear_optuna_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n結果CSV保存: {csv_path}")

    # サマリー
    df_sorted = df_results.sort_values("rmse").reset_index(drop=True)

    print("\n" + "=" * 70)
    print("全結果サマリー（RMSE昇順）")
    print("=" * 70)
    for i, row in df_sorted.iterrows():
        print(f"  {i+1:3d}. RMSE={row['rmse']:6.2f} ± {row['rmse_std']:5.2f} | "
              f"{row['preprocessing']:20s} | {row['model']:4s} | PLS({row['n_pls']:2d}) | {row['transform']}")

    print("\n" + "=" * 70)
    print("TOP 10")
    print("=" * 70)
    for i, row in df_sorted.head(10).iterrows():
        print(f"  {i+1:3d}. RMSE={row['rmse']:6.2f} ± {row['rmse_std']:5.2f} | "
              f"{row['preprocessing']:20s} | {row['model']:4s} | PLS({row['n_pls']:2d}) | {row['transform']}")
        if row.get("fold_rmses_str"):
            print(f"       Folds: {row['fold_rmses_str']}")

    print("\n" + "=" * 70)
    print("モデル別ベスト")
    print("=" * 70)
    for mt in model_types:
        df_mt = df_sorted[df_sorted["model"] == mt]
        if len(df_mt) > 0:
            best = df_mt.iloc[0]
            print(f"  {mt}: RMSE={best['rmse']:.2f} ± {best['rmse_std']:.2f} | "
                  f"{best['preprocessing']} | PLS({best['n_pls']}) | {best['transform']}")
            print(f"       Params: {best['best_params']}")

    print("\n" + "=" * 70)
    print("前処理別ベスト")
    print("=" * 70)
    for pp in preprocessings:
        df_pp = df_sorted[df_sorted["preprocessing"] == pp]
        if len(df_pp) > 0:
            best = df_pp.iloc[0]
            print(f"  {pp:20s}: RMSE={best['rmse']:.2f} ± {best['rmse_std']:.2f} | "
                  f"{best['model']} | PLS({best['n_pls']}) | {best['transform']}")

    print("\n" + "=" * 70)
    best_overall = df_sorted.iloc[0]
    print(f"ベストRMSE: {best_overall['rmse']:.2f} ± {best_overall['rmse_std']:.2f}")
    print(f"  設定: {best_overall['preprocessing']} | {best_overall['model']} | PLS({best_overall['n_pls']}) | {best_overall['transform']}")
    print(f"  現在のベストスコア（CLAUDE.md）: RMSE = 17.03")
    improvement = 17.03 - best_overall['rmse']
    print(f"  差分: {improvement:+.2f}")
    print(f"\n合計実行時間: {total_time:.0f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    main()
