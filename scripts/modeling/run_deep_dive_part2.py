"""前処理深掘り Part2: CARS波長選択 + 目的変数変換（Part1の続き）"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut, KFold
from scipy import stats
from scipy.special import inv_boxcox

from src.eda.data_loader import load_train, get_spectral_columns, get_wavenumbers
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def cars_select(X_train, y_train, X_test, n_pls=4, n_iterations=50):
    """CARS波長選択"""
    n_samples, n_features = X_train.shape
    remaining = np.arange(n_features)
    ratio = (max(n_pls + 2, 5) / n_features) ** (1.0 / n_iterations)

    best_rmse = np.inf
    best_idx = remaining.copy()

    for i in range(n_iterations):
        n_keep = max(n_pls + 1, int(n_features * ratio ** (i + 1)))
        if n_keep >= len(remaining):
            continue

        n_comp = min(n_pls, len(remaining) - 1, n_samples - 1)
        if n_comp < 1:
            break

        pls = PLSRegression(n_components=n_comp)
        pls.fit(X_train[:, remaining], y_train)
        coefs = np.abs(pls.coef_.ravel())

        # 上位n_keep個を選択
        top_idx = np.argsort(coefs)[::-1][:n_keep]
        remaining = remaining[np.sort(top_idx)]

        # 5-fold CVでRMSE評価
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_rmses = []
        for tr, te in kf.split(X_train[:, remaining]):
            n_c = min(n_comp, len(remaining) - 1)
            if n_c < 1:
                break
            p = PLSRegression(n_components=n_c)
            p.fit(X_train[tr][:, remaining], y_train[tr])
            pred = p.predict(X_train[te][:, remaining]).ravel()
            cv_rmses.append(np.sqrt(np.mean((pred - y_train[te]) ** 2)))

        if cv_rmses:
            rmse = np.mean(cv_rmses)
            if rmse < best_rmse:
                best_rmse = rmse
                best_idx = remaining.copy()

    return X_train[:, best_idx], X_test[:, best_idx], best_idx


def main():
    df = load_train(DATA_DIR)
    spectral_cols = get_spectral_columns(df)
    wn = get_wavenumbers(spectral_cols)
    X_raw = df[spectral_cols].values
    y = df["含水率"].values
    groups = df["樹種"].values

    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_raw, y, groups))
    np.random.seed(42)
    results = []

    # ============================================================
    # CARS波長選択 + EPO
    # ============================================================
    print("=" * 60)
    print("CARS波長選択 + EPO (nested CV)")
    print("=" * 60)

    for n_epo in [0, 1]:
        for n_comp in [3, 4, 5]:
            fold_rmses = []
            sel_sizes = []
            for train_idx, test_idx in folds:
                X_tr = X_raw[train_idx]
                X_te = X_raw[test_idx]

                if n_epo > 0:
                    P = compute_epo_projection(X_tr, groups[train_idx], n_components=n_epo)
                    X_tr = apply_epo(X_tr, P)
                    X_te = apply_epo(X_te, P)

                X_tr_sel, X_te_sel, sel_idx = cars_select(
                    X_tr, y[train_idx], X_te, n_pls=n_comp, n_iterations=30
                )
                sel_sizes.append(len(sel_idx))

                n_c = min(n_comp, X_tr_sel.shape[1] - 1)
                if n_c < 1:
                    n_c = 1
                pls = PLSRegression(n_components=n_c)
                pls.fit(X_tr_sel, y[train_idx])
                pred = pls.predict(X_te_sel).ravel()
                fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))

            rmse, std = np.mean(fold_rmses), np.std(fold_rmses)
            epo_str = f"EPO({n_epo})+" if n_epo > 0 else ""
            name = f"{epo_str}CARS+PLS({n_comp})"
            results.append({"category": "CARS", "method": name, "rmse": rmse, "rmse_std": std})
            print(f"  {name}: RMSE={rmse:.2f} ± {std:.2f} (avg {np.mean(sel_sizes):.0f} features)")

    # ============================================================
    # 目的変数変換
    # ============================================================
    print("\n" + "=" * 60)
    print("目的変数変換")
    print("=" * 60)

    # log(y+1)
    print("\n--- log(y+1) 変換 ---")
    y_log = np.log1p(y)
    for n_comp in [3, 4, 5, 6]:
        fold_rmses = []
        for train_idx, test_idx in folds:
            P = compute_epo_projection(X_raw[train_idx], groups[train_idx], n_components=1)
            X_tr = apply_epo(X_raw[train_idx], P)
            X_te = apply_epo(X_raw[test_idx], P)
            pls = PLSRegression(n_components=n_comp)
            pls.fit(X_tr, y_log[train_idx])
            pred_log = pls.predict(X_te).ravel()
            pred = np.expm1(pred_log)
            fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
        rmse, std = np.mean(fold_rmses), np.std(fold_rmses)
        results.append({"category": "目的変数変換", "method": f"EPO+PLS({n_comp})+log(y)", "rmse": rmse, "rmse_std": std})
        print(f"  EPO+PLS({n_comp})+log(y): RMSE={rmse:.2f} ± {std:.2f}")

    # sqrt(y)
    print("\n--- sqrt(y) 変換 ---")
    y_sqrt = np.sqrt(y)
    for n_comp in [3, 4, 5, 6]:
        fold_rmses = []
        for train_idx, test_idx in folds:
            P = compute_epo_projection(X_raw[train_idx], groups[train_idx], n_components=1)
            X_tr = apply_epo(X_raw[train_idx], P)
            X_te = apply_epo(X_raw[test_idx], P)
            pls = PLSRegression(n_components=n_comp)
            pls.fit(X_tr, y_sqrt[train_idx])
            pred_sqrt = pls.predict(X_te).ravel()
            pred = np.clip(pred_sqrt, 0, None) ** 2
            fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
        rmse, std = np.mean(fold_rmses), np.std(fold_rmses)
        results.append({"category": "目的変数変換", "method": f"EPO+PLS({n_comp})+sqrt(y)", "rmse": rmse, "rmse_std": std})
        print(f"  EPO+PLS({n_comp})+sqrt(y): RMSE={rmse:.2f} ± {std:.2f}")

    # Box-Cox
    print("\n--- Box-Cox変換 ---")
    y_bc, lam = stats.boxcox(y + 1)
    print(f"  Box-Cox lambda = {lam:.4f}")
    for n_comp in [3, 4, 5, 6]:
        fold_rmses = []
        for train_idx, test_idx in folds:
            P = compute_epo_projection(X_raw[train_idx], groups[train_idx], n_components=1)
            X_tr = apply_epo(X_raw[train_idx], P)
            X_te = apply_epo(X_raw[test_idx], P)
            pls = PLSRegression(n_components=n_comp)
            pls.fit(X_tr, y_bc[train_idx])
            pred_bc = pls.predict(X_te).ravel()
            pred = inv_boxcox(pred_bc, lam) - 1
            pred = np.clip(pred, 0, None)
            fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
        rmse, std = np.mean(fold_rmses), np.std(fold_rmses)
        results.append({"category": "目的変数変換", "method": f"EPO+PLS({n_comp})+BoxCox", "rmse": rmse, "rmse_std": std})
        print(f"  EPO+PLS({n_comp})+BoxCox: RMSE={rmse:.2f} ± {std:.2f}")

    # SNV + 目的変数変換
    print("\n--- SNV + 目的変数変換 ---")
    for transform_name, y_t, inv_fn in [
        ("log", np.log1p(y), np.expm1),
        ("sqrt", np.sqrt(y), lambda x: np.clip(x, 0, None) ** 2),
    ]:
        for n_comp in [2, 3, 4]:
            fold_rmses = []
            for train_idx, test_idx in folds:
                X_tr = apply_snv(X_raw[train_idx])
                X_te = apply_snv(X_raw[test_idx])
                pls = PLSRegression(n_components=n_comp)
                pls.fit(X_tr, y_t[train_idx])
                pred_t = pls.predict(X_te).ravel()
                pred = inv_fn(pred_t)
                fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
            rmse, std = np.mean(fold_rmses), np.std(fold_rmses)
            results.append({"category": "目的変数変換", "method": f"SNV+PLS({n_comp})+{transform_name}(y)", "rmse": rmse, "rmse_std": std})
            print(f"  SNV+PLS({n_comp})+{transform_name}(y): RMSE={rmse:.2f} ± {std:.2f}")

    # Raw + 目的変数変換
    print("\n--- Raw + 目的変数変換 ---")
    for transform_name, y_t, inv_fn in [
        ("log", np.log1p(y), np.expm1),
        ("sqrt", np.sqrt(y), lambda x: np.clip(x, 0, None) ** 2),
    ]:
        for n_comp in [3, 4, 5]:
            fold_rmses = []
            for train_idx, test_idx in folds:
                pls = PLSRegression(n_components=n_comp)
                pls.fit(X_raw[train_idx], y_t[train_idx])
                pred_t = pls.predict(X_raw[test_idx]).ravel()
                pred = inv_fn(pred_t)
                fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
            rmse, std = np.mean(fold_rmses), np.std(fold_rmses)
            results.append({"category": "目的変数変換", "method": f"Raw+PLS({n_comp})+{transform_name}(y)", "rmse": rmse, "rmse_std": std})
            print(f"  Raw+PLS({n_comp})+{transform_name}(y): RMSE={rmse:.2f} ± {std:.2f}")

    # ベースライン
    results.append({"category": "ベースライン", "method": "EPO(1)+PLS(4)", "rmse": 21.94, "rmse_std": 12.23})
    results.append({"category": "ベースライン", "method": "SNV+PLS(2) [best]", "rmse": 20.83, "rmse_std": 17.94})
    results.append({"category": "ベースライン", "method": "SimpleAvg(4) [best ensemble]", "rmse": 18.58, "rmse_std": 14.84})

    # 結果まとめ
    result_df = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    print("\n" + "=" * 60)
    print("Part2 全結果")
    print("=" * 60)
    print(result_df.to_string(index=False))

    # 保存
    result_df.to_csv(OUT_DIR / "deep_dive_part2_results.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'deep_dive_part2_results.csv'}")


if __name__ == "__main__":
    main()
