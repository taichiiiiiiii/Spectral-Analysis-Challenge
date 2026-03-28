"""Issue #64: MLP (MLPRegressor) LOSO-CV 実験 - 軽量版

PLS次元削減 + StandardScaler + MLPRegressor の組み合わせを評価。
組み合わせ数を絞って高速実行。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut

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


def apply_preprocessing(X_train, X_test, groups_train, method):
    if method == "SNV":
        return apply_snv(X_train), apply_snv(X_test)
    elif method == "EPO(1)":
        P = compute_epo_projection(X_train, groups_train, n_components=1)
        return apply_epo(X_train, P), apply_epo(X_test, P)
    elif method == "SNV+AsLS(1e6)":
        X_tr = apply_asls(apply_snv(X_train), lam=1e6)
        X_te = apply_asls(apply_snv(X_test), lam=1e6)
        return X_tr, X_te
    elif method == "PiecewiseMSC(seg=3)":
        ref = compute_msc_reference(X_train)
        return apply_piecewise_msc(X_train, ref, n_segments=3), apply_piecewise_msc(X_test, ref, n_segments=3)
    raise ValueError(f"Unknown: {method}")


def transform_target(y, method):
    if method == "raw":
        return y.copy()
    elif method == "sqrt":
        return np.sqrt(y)
    elif method == "log1p":
        return np.log1p(y)
    raise ValueError(f"Unknown: {method}")


def inverse_transform(pred, method):
    if method == "raw":
        return pred
    elif method == "sqrt":
        return np.clip(pred, 0, None) ** 2
    elif method == "log1p":
        return np.expm1(np.clip(pred, 0, 20))
    raise ValueError(f"Unknown: {method}")


def main():
    df_train = load_train(DATA_DIR)
    spectral_cols = get_spectral_columns(df_train)
    X_raw = df_train[spectral_cols].values
    y = df_train["含水率"].values
    groups = df_train["樹種"].values

    print("=" * 70)
    print("Issue #64: MLP (MLPRegressor) LOSO-CV 実験")
    print("=" * 70)
    print(f"  サンプル数: {len(y)}, 特徴量数: {X_raw.shape[1]}, 樹種数: {len(np.unique(groups))}")

    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_raw, y, groups))

    # 絞った組み合わせ
    configs = [
        # (preprocessing, n_pls, hidden_layers, activation, alpha, lr, target_transform)
        ("SNV", 4, (100, 50), "relu", 0.01, 0.001, "sqrt"),
        ("SNV", 6, (100, 50), "relu", 0.01, 0.001, "sqrt"),
        ("SNV", 8, (100, 50), "relu", 0.01, 0.001, "sqrt"),
        ("SNV", 4, (50,), "relu", 0.01, 0.001, "sqrt"),
        ("SNV", 4, (100, 50, 25), "relu", 0.01, 0.001, "sqrt"),
        ("SNV", 4, (100, 50), "tanh", 0.01, 0.001, "sqrt"),
        ("SNV", 4, (100, 50), "relu", 0.1, 0.001, "sqrt"),
        ("SNV", 4, (100, 50), "relu", 1.0, 0.001, "sqrt"),
        ("SNV", 4, (100, 50), "relu", 0.01, 0.01, "sqrt"),
        ("SNV", 4, (100, 50), "relu", 0.01, 0.001, "raw"),
        ("EPO(1)", 4, (100, 50), "relu", 0.01, 0.001, "sqrt"),
        ("EPO(1)", 6, (100, 50), "relu", 0.01, 0.001, "sqrt"),
        ("EPO(1)", 4, (100, 50), "relu", 0.1, 0.001, "sqrt"),
        ("EPO(1)", 4, (100, 50), "relu", 0.01, 0.001, "raw"),
        ("SNV+AsLS(1e6)", 4, (100, 50), "relu", 0.01, 0.001, "sqrt"),
        ("SNV+AsLS(1e6)", 6, (100, 50), "relu", 0.01, 0.001, "sqrt"),
        ("PiecewiseMSC(seg=3)", 4, (100, 50), "relu", 0.01, 0.001, "sqrt"),
        ("PiecewiseMSC(seg=3)", 4, (100, 50), "relu", 0.01, 0.001, "raw"),
    ]

    results = []
    total = len(configs)

    for idx, (pp, n_pls, hidden, act, alpha, lr, tf) in enumerate(configs, 1):
        fold_rmses = []
        for train_idx, test_idx in folds:
            try:
                X_tr_pp, X_te_pp = apply_preprocessing(
                    X_raw[train_idx], X_raw[test_idx], groups[train_idx], pp
                )
                # PLS次元削減
                n_comp = min(n_pls, X_tr_pp.shape[1] - 1, X_tr_pp.shape[0] - 1)
                pls = PLSRegression(n_components=n_comp)
                pls.fit(X_tr_pp, y[train_idx])
                X_tr_scores = pls.transform(X_tr_pp)
                X_te_scores = pls.transform(X_te_pp)

                # StandardScaler
                scaler = StandardScaler()
                X_tr_scaled = scaler.fit_transform(X_tr_scores)
                X_te_scaled = scaler.transform(X_te_scores)

                # Target transform
                y_tr = transform_target(y[train_idx], tf)

                # MLP
                mlp = MLPRegressor(
                    hidden_layer_sizes=hidden,
                    activation=act,
                    alpha=alpha,
                    learning_rate_init=lr,
                    max_iter=2000,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=42,
                )
                mlp.fit(X_tr_scaled, y_tr)
                pred = mlp.predict(X_te_scaled)
                pred = inverse_transform(pred, tf)
                fold_rmses.append(rmse(y[test_idx], pred))
            except Exception as e:
                fold_rmses.append(999.0)

        mean_rmse = np.mean(fold_rmses)
        std_rmse = np.std(fold_rmses)
        desc = f"{pp} | PLS={n_pls} | MLP{hidden} | {act} | a={alpha} | lr={lr} | {tf}"
        results.append({
            "description": desc,
            "preprocessing": pp,
            "n_pls": n_pls,
            "hidden": str(hidden),
            "activation": act,
            "alpha": alpha,
            "lr": lr,
            "transform": tf,
            "rmse": mean_rmse,
            "rmse_std": std_rmse,
        })
        print(f"[{idx}/{total}] {desc} → RMSE={mean_rmse:.2f} ± {std_rmse:.2f}", flush=True)

    # 結果表示
    df = pd.DataFrame(results).sort_values("rmse")
    print("\n" + "=" * 70)
    print("TOP 10 RESULTS:")
    print("=" * 70)
    for i, row in df.head(10).iterrows():
        print(f"  RMSE={row['rmse']:.2f} ± {row['rmse_std']:.2f} | {row['description']}")

    df.to_csv(OUT_DIR / "issue64_mlp_results.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'issue64_mlp_results.csv'}")


if __name__ == "__main__":
    main()
