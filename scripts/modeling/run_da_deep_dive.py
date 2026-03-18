"""ドメイン適応深掘り: JDA, BDA, Weighted TCA のLOSO-CV評価

TCAの拡張手法3つを複数のハイパーパラメータ設定で評価し、
TCA(10)+PLS(4)+sqrt(y)=19.91 を超えるかを検証する。
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut

from src.eda.data_loader import load_train, load_test, get_spectral_columns
from src.preprocessing.issue48_jda import jda_transform
from src.preprocessing.issue49_bda import bda_transform
from src.preprocessing.issue50_weighted_tca import weighted_tca_transform
from src.preprocessing.issue38_tca import tca_transform


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def p(msg):
    print(msg, flush=True)


def eval_da_method(method_name, transform_fn, X_train, y, folds, nc_pls_list, sqrt_y=False):
    """ドメイン適応手法をLOSO-CVで評価する。

    transform_fn: (X_train_fold, X_test_fold, y_train_fold) -> (Z_train, Z_test)
    """
    results = []
    for nc_pls in nc_pls_list:
        fold_rmses = []
        for train_idx, test_idx in folds:
            X_tr = X_train[train_idx]
            X_te = X_train[test_idx]
            y_tr = y[train_idx]
            y_te = y[test_idx]
            try:
                Z_tr, Z_te = transform_fn(X_tr, X_te, y_tr)
                pls = PLSRegression(n_components=nc_pls)
                if sqrt_y:
                    pls.fit(Z_tr, np.sqrt(y_tr))
                    preds = pls.predict(Z_te).ravel() ** 2
                else:
                    pls.fit(Z_tr, y_tr)
                    preds = pls.predict(Z_te).ravel()
                fold_rmses.append(rmse(y_te, preds))
            except Exception as e:
                p(f"    Error in {method_name}: {e}")
                fold_rmses.append(999.0)
        mean_r = np.mean(fold_rmses)
        std_r = np.std(fold_rmses)
        suffix = "+sqrt(y)" if sqrt_y else ""
        label = f"{method_name}+PLS({nc_pls}){suffix}"
        p(f"  {label}: RMSE={mean_r:.2f} ± {std_r:.2f}")
        results.append({"method": label, "rmse": mean_r, "rmse_std": std_r})
    return results


def main():
    data_dir = Path("Input_data")
    train_df = load_train(data_dir)
    spectral_cols = get_spectral_columns(train_df)

    X_train = train_df[spectral_cols].values
    y = train_df["含水率"].values
    groups = train_df["樹種"].values
    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_train, y, groups))

    all_results = []

    # ===== ベースライン: TCA (再確認) =====
    p("=== TCA (ベースライン) ===")
    for n_comp in [10]:
        for sqrt_y in [False, True]:
            res = eval_da_method(
                f"TCA({n_comp})",
                lambda Xs, Xt, ys, nc=n_comp: tca_transform(Xs, Xt, n_components=nc),
                X_train, y, folds, [4], sqrt_y=sqrt_y
            )
            all_results.extend(res)

    # ===== 1. JDA =====
    p("\n=== JDA (Joint Distribution Adaptation) ===")
    for n_comp in [5, 10, 15]:
        for n_iter in [1, 3, 5]:
            for n_bins in [5]:
                res = eval_da_method(
                    f"JDA({n_comp},iter={n_iter})",
                    lambda Xs, Xt, ys, nc=n_comp, ni=n_iter, nb=n_bins:
                        jda_transform(Xs, Xt, ys, n_components=nc,
                                      n_iterations=ni, n_bins=nb),
                    X_train, y, folds, [4], sqrt_y=False
                )
                all_results.extend(res)

    # JDA best settings + sqrt(y)
    p("\n=== JDA + sqrt(y) ===")
    for n_comp in [10, 15]:
        res = eval_da_method(
            f"JDA({n_comp},iter=3)",
            lambda Xs, Xt, ys, nc=n_comp:
                jda_transform(Xs, Xt, ys, n_components=nc, n_iterations=3),
            X_train, y, folds, [4], sqrt_y=True
        )
        all_results.extend(res)

    # ===== 2. BDA =====
    p("\n=== BDA (Balanced Distribution Adaptation) ===")
    for n_comp in [10]:
        for mu_b in [0.1, 0.3, 0.5, 0.7, 0.9]:
            res = eval_da_method(
                f"BDA({n_comp},μb={mu_b})",
                lambda Xs, Xt, ys, nc=n_comp, mb=mu_b:
                    bda_transform(Xs, Xt, ys, n_components=nc, mu_b=mb,
                                  n_iterations=3),
                X_train, y, folds, [4], sqrt_y=False
            )
            all_results.extend(res)

    # BDA + sqrt(y)
    p("\n=== BDA + sqrt(y) ===")
    for mu_b in [0.3, 0.5, 0.7]:
        res = eval_da_method(
            f"BDA(10,μb={mu_b})",
            lambda Xs, Xt, ys, mb=mu_b:
                bda_transform(Xs, Xt, ys, n_components=10, mu_b=mb, n_iterations=3),
            X_train, y, folds, [4], sqrt_y=True
        )
        all_results.extend(res)

    # ===== 3. Weighted TCA =====
    p("\n=== Weighted TCA ===")
    for n_comp in [10, 15]:
        res = eval_da_method(
            f"WTCA({n_comp})",
            lambda Xs, Xt, ys, nc=n_comp:
                weighted_tca_transform(Xs, Xt, ys, n_components=nc),
            X_train, y, folds, [4], sqrt_y=False
        )
        all_results.extend(res)

    # Weighted TCA + sqrt(y)
    p("\n=== Weighted TCA + sqrt(y) ===")
    for n_comp in [10, 15]:
        res = eval_da_method(
            f"WTCA({n_comp})",
            lambda Xs, Xt, ys, nc=n_comp:
                weighted_tca_transform(Xs, Xt, ys, n_components=nc),
            X_train, y, folds, [4], sqrt_y=True
        )
        all_results.extend(res)

    # ===== Summary =====
    p("\n" + "=" * 60)
    p("SUMMARY (sorted by RMSE)")
    p("=" * 60)
    results_df = pd.DataFrame(all_results)
    results_df = results_df[results_df["rmse"] < 900].sort_values("rmse").reset_index(drop=True)
    p(results_df[["method", "rmse", "rmse_std"]].to_string(index=False))

    output_path = Path("outputs/da_deep_dive.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    p(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
