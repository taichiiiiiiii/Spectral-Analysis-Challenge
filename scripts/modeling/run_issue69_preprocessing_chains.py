"""Issue #69: 2段階前処理チェーン × PLS/Lasso LOSO-CV評価

前処理チェーン（1段目→2段目）とモデル(PLS/Lasso)の組み合わせを
LOSO-CVで評価し、RMSE上位20件を表示・CSV保存する。

fold単位で前処理結果をキャッシュし、高速化を図る。
AsLS/Whittakerは30秒タイムアウト付き。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import LeaveOneGroupOut

from src.eda.data_loader import load_train, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue19_msc import compute_msc_reference, apply_msc
from src.preprocessing.issue20_savgol import apply_savgol
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue62_additional_preprocessing import apply_asls, apply_whittaker

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# === Preprocessing step functions ===
# Each: (X_train, X_test, groups_train) -> (X_train_out, X_test_out)

def step_snv(X_train, X_test, groups_train):
    return apply_snv(X_train), apply_snv(X_test)

def step_sg1d_w11(X_train, X_test, groups_train):
    return (apply_savgol(X_train, deriv=1, window_length=11, polyorder=2),
            apply_savgol(X_test, deriv=1, window_length=11, polyorder=2))

def step_sg2d_w7(X_train, X_test, groups_train):
    return (apply_savgol(X_train, deriv=2, window_length=7, polyorder=2),
            apply_savgol(X_test, deriv=2, window_length=7, polyorder=2))

def step_msc(X_train, X_test, groups_train):
    ref = compute_msc_reference(X_train)
    return apply_msc(X_train, ref), apply_msc(X_test, ref)

def step_epo1(X_train, X_test, groups_train):
    P = compute_epo_projection(X_train, groups_train, n_components=1)
    return apply_epo(X_train, P), apply_epo(X_test, P)

def step_asls_1e6(X_train, X_test, groups_train):
    return apply_asls(X_train, lam=1e6), apply_asls(X_test, lam=1e6)

def step_whittaker_1e3(X_train, X_test, groups_train):
    return apply_whittaker(X_train, lam=1e3), apply_whittaker(X_test, lam=1e3)


# === Chain definitions ===
# (name, [step_functions], is_slow)
CHAINS = [
    ("SNV->SG1d(w=11)",      [step_snv, step_sg1d_w11],      False),
    ("SNV->SG2d(w=7)",       [step_snv, step_sg2d_w7],       False),
    ("MSC->SG1d(w=11)",      [step_msc, step_sg1d_w11],      False),
    ("MSC->SG2d(w=7)",       [step_msc, step_sg2d_w7],       False),
    ("SNV->EPO(1)",           [step_snv, step_epo1],          False),
    ("MSC->EPO(1)",           [step_msc, step_epo1],          False),
    ("SG1d(w=11)->EPO(1)",    [step_sg1d_w11, step_epo1],    False),
    ("EPO(1)->SNV",           [step_epo1, step_snv],          False),
    ("EPO(1)->AsLS(1e6)",     [step_epo1, step_asls_1e6],    True),
    ("SNV->Whittaker(1e3)",   [step_snv, step_whittaker_1e3], True),
]

# === Model definitions ===
# (name, model_factory, y_transform, y_inverse)
MODELS = [
    ("PLS(2)+sqrt", lambda: PLSRegression(n_components=2), np.sqrt, lambda y: y**2),
    ("PLS(3)+raw",  lambda: PLSRegression(n_components=3), None, None),
    ("PLS(4)+sqrt", lambda: PLSRegression(n_components=4), np.sqrt, lambda y: y**2),
    ("Lasso(0.1)+raw", lambda: Lasso(alpha=0.1, max_iter=5000), None, None),
]


def apply_chain_fold(steps, X_train_raw, X_test_raw, groups_train):
    """Apply a chain of preprocessing steps for one fold."""
    X_tr, X_te = X_train_raw.copy(), X_test_raw.copy()
    for step_fn in steps:
        X_tr, X_te = step_fn(X_tr, X_te, groups_train)
    return X_tr, X_te


def preprocess_all_folds(chain_name, steps, folds, X_raw, groups, is_slow, timeout_sec=30):
    """Preprocess all folds for a chain. Returns dict of fold_data or None if timed out."""
    fold_data = {}
    chain_t0 = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        # For slow chains, check elapsed time before each fold
        if is_slow and fold_idx > 0:
            elapsed = time.time() - chain_t0
            avg_per_fold = elapsed / fold_idx
            if avg_per_fold > timeout_sec / len(folds):
                # Extrapolate: will exceed timeout
                est_total = avg_per_fold * len(folds)
                if est_total > timeout_sec * 2:
                    print(f"  [SKIP] fold {fold_idx}: 推定所要時間 {est_total:.0f}s > タイムアウト {timeout_sec}s")
                    return None

        fold_t0 = time.time()
        try:
            X_tr_pp, X_te_pp = apply_chain_fold(
                steps, X_raw[train_idx], X_raw[test_idx], groups[train_idx]
            )
            fold_data[fold_idx] = (X_tr_pp, X_te_pp, train_idx, test_idx)
            fold_time = time.time() - fold_t0

            # For slow chains, skip immediately if first fold is too slow
            if is_slow and fold_idx == 0 and fold_time > timeout_sec / 2:
                print(f"  [SKIP] fold 0 took {fold_time:.1f}s - chain too slow")
                return None

        except Exception as e:
            print(f"  [SKIP] fold {fold_idx} エラー: {e}")
            return None

    return fold_data


def main():
    print("=" * 70)
    print("Issue #69: 2段階前処理チェーン × PLS/Lasso LOSO-CV評価")
    print("=" * 70)

    # Load data
    t0 = time.time()
    df = load_train(DATA_DIR)
    spectral_cols = get_spectral_columns(df)
    X_raw = df[spectral_cols].values.astype(np.float64)
    y = df["含水率"].values.astype(np.float64)
    groups = df["樹種"].values

    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_raw, y, groups))
    n_folds = len(folds)
    print(f"データ読み込み完了: {X_raw.shape[0]}サンプル, {X_raw.shape[1]}特徴量, {n_folds}フォールド")
    print(f"樹種: {np.unique(groups)}")
    print()

    results = []

    for chain_name, steps, is_slow in CHAINS:
        print(f"--- 前処理チェーン: {chain_name} {'[slow]' if is_slow else ''} ---")
        chain_t0 = time.time()

        # Preprocess all folds
        fold_data = preprocess_all_folds(
            chain_name, steps, folds, X_raw, groups,
            is_slow=is_slow, timeout_sec=30
        )

        if fold_data is None:
            print(f"  => チェーン {chain_name} をスキップ")
            print()
            continue

        pp_time = time.time() - chain_t0
        print(f"  前処理完了: {pp_time:.1f}s ({n_folds}フォールド)")

        # Evaluate each model using cached preprocessing
        for model_name, model_factory, y_transform, y_inverse in MODELS:
            fold_rmses = []

            for fold_idx in range(n_folds):
                X_tr_pp, X_te_pp, train_idx, test_idx = fold_data[fold_idx]
                y_train = y[train_idx]
                y_test = y[test_idx]

                # Transform target if needed
                y_train_t = y_transform(y_train) if y_transform is not None else y_train

                # Fit & predict
                model = model_factory()
                try:
                    model.fit(X_tr_pp, y_train_t)
                    pred = model.predict(X_te_pp)
                    if hasattr(pred, 'ravel'):
                        pred = pred.ravel()

                    # Inverse transform
                    if y_inverse is not None:
                        pred = y_inverse(pred)

                    # Clip negative predictions
                    pred = np.clip(pred, 0, None)

                    fold_rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
                    fold_rmses.append(fold_rmse)
                except Exception as e:
                    print(f"  [WARN] {model_name} fold {fold_idx}: {e}")
                    fold_rmses.append(np.nan)

            mean_rmse = float(np.nanmean(fold_rmses))
            std_rmse = float(np.nanstd(fold_rmses))

            results.append({
                "chain": chain_name,
                "model": model_name,
                "rmse_mean": round(mean_rmse, 4),
                "rmse_std": round(std_rmse, 4),
                "n_folds": int(np.sum(~np.isnan(fold_rmses))),
            })
            print(f"  {model_name}: RMSE={mean_rmse:.4f} (std={std_rmse:.4f})")

        print(f"  合計: {time.time()-chain_t0:.1f}s")
        print()

    # Results
    df_results = pd.DataFrame(results).sort_values("rmse_mean").reset_index(drop=True)

    print("=" * 70)
    print("Top 20 結果 (RMSE昇順)")
    print("=" * 70)
    top20 = df_results.head(20)
    for i, row in top20.iterrows():
        print(f"  {i+1:2d}. {row['chain']:25s} + {row['model']:18s} -> RMSE={row['rmse_mean']:.4f} (std={row['rmse_std']:.4f})")

    # Save CSV
    csv_path = OUT_DIR / "issue69_preprocessing_chains_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nCSV保存: {csv_path}")
    print(f"\n全体実行時間: {time.time()-t0:.1f}s")

    return df_results


if __name__ == "__main__":
    main()
