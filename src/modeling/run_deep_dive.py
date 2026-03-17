"""前処理アプローチの深掘り

1. SNV深掘り: SNV+PLS成分数最適化、RNV（Robust Normal Variate）
2. SG微分+EPO: 1st/2nd deriv + EPO成分数同時最適化
3. CARS波長選択 + EPO
4. 目的変数変換: log, sqrt, Box-Cox
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut
from scipy import stats

from src.eda.data_loader import load_train, get_spectral_columns, get_wavenumbers
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue20_savgol import apply_savgol
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def loso_cv(X, y, groups, model_fn, inverse_fn=None):
    """LOSO-CVでRMSEを計算"""
    logo = LeaveOneGroupOut()
    fold_rmses = []
    for train_idx, test_idx in logo.split(X, y, groups):
        model = model_fn()
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx]).ravel()
        if inverse_fn is not None:
            pred = inverse_fn(pred)
        fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
    return np.mean(fold_rmses), np.std(fold_rmses)


def loso_cv_with_transform(X, y_orig, y_transformed, groups, model_fn, inverse_fn):
    """目的変数変換付きLOSO-CV（RMSEは元スケールで計算）"""
    logo = LeaveOneGroupOut()
    fold_rmses = []
    for train_idx, test_idx in logo.split(X, y_transformed, groups):
        model = model_fn()
        model.fit(X[train_idx], y_transformed[train_idx])
        pred_transformed = model.predict(X[test_idx]).ravel()
        pred_orig = inverse_fn(pred_transformed)
        fold_rmses.append(float(np.sqrt(np.mean((pred_orig - y_orig[test_idx]) ** 2))))
    return np.mean(fold_rmses), np.std(fold_rmses)


def loso_cv_preprocess(X_raw, y, groups, preprocess_fn, model_fn):
    """前処理をfold内で適用するLOSO-CV"""
    logo = LeaveOneGroupOut()
    fold_rmses = []
    for train_idx, test_idx in logo.split(X_raw, y, groups):
        X_tr, X_te = preprocess_fn(X_raw, train_idx, test_idx, groups)
        model = model_fn()
        model.fit(X_tr, y[train_idx])
        pred = model.predict(X_te).ravel()
        fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
    return np.mean(fold_rmses), np.std(fold_rmses)


def cars_select(X_train, y_train, X_test, wavenumbers, n_pls=4, n_iterations=50, n_select=None):
    """CARS (Competitive Adaptive Reweighted Sampling) 波長選択

    Parameters
    ----------
    X_train, X_test : arrays
    y_train : array
    wavenumbers : array
    n_pls : int
    n_iterations : int
    n_select : int, optional. 最終的に選択する波長数。Noneなら自動

    Returns
    -------
    X_train_sel, X_test_sel, selected_idx
    """
    n_samples, n_features = X_train.shape
    # 初期: 全波長の重み
    weights = np.ones(n_features)
    remaining = np.arange(n_features)

    # 指数関数的に波長数を減少
    ratio = (2 / n_features) ** (1.0 / n_iterations)

    best_rmse = np.inf
    best_idx = remaining.copy()

    for i in range(n_iterations):
        # 現在の波長数
        n_keep = max(n_pls + 1, int(n_features * ratio ** (i + 1)))
        if n_keep >= len(remaining):
            continue

        # PLSで回帰係数を取得
        n_comp = min(n_pls, len(remaining) - 1, n_samples - 1)
        if n_comp < 1:
            break
        pls = PLSRegression(n_components=n_comp)
        pls.fit(X_train[:, remaining], y_train)
        coefs = np.abs(pls.coef_.ravel())

        # ARS (Adaptive Reweighted Sampling): 重みに基づくランダム選択
        probs = coefs / (coefs.sum() + 1e-10)

        # 上位n_keep個を選択（確定的 + ランダム要素）
        # 確定的に上位半分 + ランダムに残り
        n_deterministic = n_keep // 2
        n_random = n_keep - n_deterministic
        top_idx = np.argsort(coefs)[::-1][:n_deterministic]
        rest_idx = np.argsort(coefs)[::-1][n_deterministic:]
        if len(rest_idx) > 0 and n_random > 0:
            rest_probs = probs[rest_idx]
            rest_probs = rest_probs / (rest_probs.sum() + 1e-10)
            n_random = min(n_random, len(rest_idx))
            random_idx = np.random.choice(rest_idx, size=n_random, replace=False, p=rest_probs)
            selected = np.sort(np.concatenate([top_idx, random_idx]))
        else:
            selected = np.sort(top_idx)

        remaining = remaining[selected]

        # 5-fold CVでRMSE評価
        from sklearn.model_selection import KFold
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

    all_results = []

    # ============================================================
    # 1. SNV深掘り
    # ============================================================
    print("=" * 60)
    print("1. SNV深掘り")
    print("=" * 60)

    X_snv = apply_snv(X_raw)

    # SNV + PLS成分数スキャン
    print("\n--- SNV + PLS成分数スキャン ---")
    for n_comp in range(1, 11):
        rmse, std = loso_cv(X_snv, y, groups, lambda nc=n_comp: PLSRegression(n_components=nc))
        all_results.append({"category": "SNV深掘り", "method": f"SNV+PLS({n_comp})", "rmse": rmse, "rmse_std": std})
        print(f"  SNV+PLS({n_comp}): RMSE={rmse:.2f} ± {std:.2f}")

    # RNV (Robust Normal Variate): median/MADを使用
    print("\n--- RNV (Robust Normal Variate) ---")
    medians = np.median(X_raw, axis=1, keepdims=True)
    mads = np.median(np.abs(X_raw - medians), axis=1, keepdims=True) * 1.4826
    X_rnv = (X_raw - medians) / (mads + 1e-10)
    for n_comp in [3, 4, 5]:
        rmse, std = loso_cv(X_rnv, y, groups, lambda nc=n_comp: PLSRegression(n_components=nc))
        all_results.append({"category": "SNV深掘り", "method": f"RNV+PLS({n_comp})", "rmse": rmse, "rmse_std": std})
        print(f"  RNV+PLS({n_comp}): RMSE={rmse:.2f} ± {std:.2f}")

    # SNV + EPO
    print("\n--- SNV + EPO ---")
    def snv_epo_preprocess(X_raw, train_idx, test_idx, groups, n_epo=1):
        X_tr_snv = apply_snv(X_raw[train_idx])
        X_te_snv = apply_snv(X_raw[test_idx])
        P = compute_epo_projection(X_tr_snv, groups[train_idx], n_components=n_epo)
        return apply_epo(X_tr_snv, P), apply_epo(X_te_snv, P)

    for n_epo in [1, 2]:
        for n_comp in [3, 4, 5]:
            rmse, std = loso_cv_preprocess(
                X_raw, y, groups,
                lambda X, tri, tei, g, ne=n_epo: snv_epo_preprocess(X, tri, tei, g, ne),
                lambda nc=n_comp: PLSRegression(n_components=nc),
            )
            all_results.append({"category": "SNV深掘り", "method": f"SNV+EPO({n_epo})+PLS({n_comp})", "rmse": rmse, "rmse_std": std})
            print(f"  SNV+EPO({n_epo})+PLS({n_comp}): RMSE={rmse:.2f} ± {std:.2f}")

    # ============================================================
    # 2. SG微分 + EPO
    # ============================================================
    print("\n" + "=" * 60)
    print("2. SG微分 + EPO 同時最適化")
    print("=" * 60)

    for deriv in [1, 2]:
        for window in [7, 11, 15, 21]:
            for n_epo in [0, 1, 2]:
                for n_comp in [3, 4, 5]:
                    def sg_epo_preprocess(X_raw, train_idx, test_idx, groups, d=deriv, w=window, ne=n_epo):
                        X_tr = apply_savgol(X_raw[train_idx], deriv=d, window_length=w)
                        X_te = apply_savgol(X_raw[test_idx], deriv=d, window_length=w)
                        if ne > 0:
                            P = compute_epo_projection(X_tr, groups[train_idx], n_components=ne)
                            X_tr = apply_epo(X_tr, P)
                            X_te = apply_epo(X_te, P)
                        return X_tr, X_te

                    rmse, std = loso_cv_preprocess(
                        X_raw, y, groups,
                        sg_epo_preprocess,
                        lambda nc=n_comp: PLSRegression(n_components=nc),
                    )
                    epo_str = f"+EPO({n_epo})" if n_epo > 0 else ""
                    name = f"SG{deriv}d(w={window}){epo_str}+PLS({n_comp})"
                    all_results.append({"category": "SG+EPO", "method": name, "rmse": rmse, "rmse_std": std})
                    print(f"  {name}: RMSE={rmse:.2f} ± {std:.2f}")

    # SNV + SG + EPO
    print("\n--- SNV + SG微分 + EPO ---")
    for deriv in [1, 2]:
        for n_epo in [0, 1]:
            for n_comp in [3, 4, 5]:
                def snv_sg_epo(X_raw, train_idx, test_idx, groups, d=deriv, ne=n_epo):
                    X_tr = apply_savgol(apply_snv(X_raw[train_idx]), deriv=d)
                    X_te = apply_savgol(apply_snv(X_raw[test_idx]), deriv=d)
                    if ne > 0:
                        P = compute_epo_projection(X_tr, groups[train_idx], n_components=ne)
                        X_tr = apply_epo(X_tr, P)
                        X_te = apply_epo(X_te, P)
                    return X_tr, X_te

                rmse, std = loso_cv_preprocess(
                    X_raw, y, groups, snv_sg_epo,
                    lambda nc=n_comp: PLSRegression(n_components=nc),
                )
                epo_str = f"+EPO({n_epo})" if n_epo > 0 else ""
                name = f"SNV+SG{deriv}d{epo_str}+PLS({n_comp})"
                all_results.append({"category": "SG+EPO", "method": name, "rmse": rmse, "rmse_std": std})
                print(f"  {name}: RMSE={rmse:.2f} ± {std:.2f}")

    # ============================================================
    # 3. CARS波長選択 + EPO
    # ============================================================
    print("\n" + "=" * 60)
    print("3. CARS波長選択 + EPO")
    print("=" * 60)

    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_raw, y, groups))
    np.random.seed(42)

    # 水吸収帯領域のみに絞る（手動選択）
    print("\n--- 水吸収帯領域選択 ---")
    # 水コンビネーション帯: 4800-5400 cm⁻¹, 水第1倍音: 6400-7400 cm⁻¹
    water_mask = ((wn >= 4800) & (wn <= 5400)) | ((wn >= 6400) & (wn <= 7400))
    X_water = X_raw[:, water_mask]
    wn_water = wn[water_mask]
    print(f"  水帯選択波数: {water_mask.sum()}/{len(wn)}点")

    for n_epo in [0, 1]:
        for n_comp in [2, 3, 4]:
            def water_epo(X_raw, train_idx, test_idx, groups, ne=n_epo):
                X_tr = X_raw[train_idx][:, water_mask]
                X_te = X_raw[test_idx][:, water_mask]
                if ne > 0:
                    P = compute_epo_projection(X_tr, groups[train_idx], n_components=ne)
                    X_tr = apply_epo(X_tr, P)
                    X_te = apply_epo(X_te, P)
                return X_tr, X_te

            rmse, std = loso_cv_preprocess(
                X_raw, y, groups, water_epo,
                lambda nc=n_comp: PLSRegression(n_components=nc),
            )
            epo_str = f"+EPO({n_epo})" if n_epo > 0 else ""
            name = f"WaterBands{epo_str}+PLS({n_comp})"
            all_results.append({"category": "CARS/波長選択", "method": name, "rmse": rmse, "rmse_std": std})
            print(f"  {name}: RMSE={rmse:.2f} ± {std:.2f}")

    # CARS + EPO (nested CV)
    print("\n--- CARS + EPO (nested CV) ---")
    for n_epo in [0, 1]:
        for n_comp in [3, 4, 5]:
            fold_rmses = []
            for train_idx, test_idx in folds:
                X_tr = X_raw[train_idx]
                X_te = X_raw[test_idx]

                # EPO適用
                if n_epo > 0:
                    P = compute_epo_projection(X_tr, groups[train_idx], n_components=n_epo)
                    X_tr = apply_epo(X_tr, P)
                    X_te = apply_epo(X_te, P)

                # CARS波長選択（train内で実行）
                X_tr_sel, X_te_sel, sel_idx = cars_select(
                    X_tr, y[train_idx], X_te, wn, n_pls=n_comp, n_iterations=30
                )

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
            all_results.append({"category": "CARS/波長選択", "method": name, "rmse": rmse, "rmse_std": std})
            print(f"  {name}: RMSE={rmse:.2f} ± {std:.2f}")

    # ============================================================
    # 4. 目的変数変換
    # ============================================================
    print("\n" + "=" * 60)
    print("4. 目的変数変換")
    print("=" * 60)

    # EPO前処理（fold内で適用）
    def epo_preprocess_arrays(X_raw, train_idx, test_idx, groups, n_epo=1):
        P = compute_epo_projection(X_raw[train_idx], groups[train_idx], n_components=n_epo)
        return apply_epo(X_raw[train_idx], P), apply_epo(X_raw[test_idx], P)

    # log変換
    print("\n--- log(y+1) 変換 ---")
    y_log = np.log1p(y)
    for n_comp in [3, 4, 5, 6]:
        logo = LeaveOneGroupOut()
        fold_rmses = []
        for train_idx, test_idx in logo.split(X_raw, y, groups):
            X_tr, X_te = epo_preprocess_arrays(X_raw, train_idx, test_idx, groups, n_epo=1)
            pls = PLSRegression(n_components=n_comp)
            pls.fit(X_tr, y_log[train_idx])
            pred_log = pls.predict(X_te).ravel()
            pred = np.expm1(pred_log)
            fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
        rmse, std = np.mean(fold_rmses), np.std(fold_rmses)
        all_results.append({"category": "目的変数変換", "method": f"EPO+PLS({n_comp})+log(y)", "rmse": rmse, "rmse_std": std})
        print(f"  EPO+PLS({n_comp})+log(y): RMSE={rmse:.2f} ± {std:.2f}")

    # sqrt変換
    print("\n--- sqrt(y) 変換 ---")
    y_sqrt = np.sqrt(y)
    for n_comp in [3, 4, 5, 6]:
        logo = LeaveOneGroupOut()
        fold_rmses = []
        for train_idx, test_idx in logo.split(X_raw, y, groups):
            X_tr, X_te = epo_preprocess_arrays(X_raw, train_idx, test_idx, groups, n_epo=1)
            pls = PLSRegression(n_components=n_comp)
            pls.fit(X_tr, y_sqrt[train_idx])
            pred_sqrt = pls.predict(X_te).ravel()
            pred = np.clip(pred_sqrt, 0, None) ** 2
            fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
        rmse, std = np.mean(fold_rmses), np.std(fold_rmses)
        all_results.append({"category": "目的変数変換", "method": f"EPO+PLS({n_comp})+sqrt(y)", "rmse": rmse, "rmse_std": std})
        print(f"  EPO+PLS({n_comp})+sqrt(y): RMSE={rmse:.2f} ± {std:.2f}")

    # Box-Cox変換
    print("\n--- Box-Cox変換 ---")
    y_bc, lam = stats.boxcox(y + 1)  # +1 for strictly positive
    print(f"  Box-Cox lambda = {lam:.4f}")
    for n_comp in [3, 4, 5, 6]:
        logo = LeaveOneGroupOut()
        fold_rmses = []
        for train_idx, test_idx in logo.split(X_raw, y, groups):
            X_tr, X_te = epo_preprocess_arrays(X_raw, train_idx, test_idx, groups, n_epo=1)
            pls = PLSRegression(n_components=n_comp)
            pls.fit(X_tr, y_bc[train_idx])
            pred_bc = pls.predict(X_te).ravel()
            # 逆変換
            pred = stats.special.inv_boxcox(pred_bc, lam) - 1
            pred = np.clip(pred, 0, None)
            fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
        rmse, std = np.mean(fold_rmses), np.std(fold_rmses)
        all_results.append({"category": "目的変数変換", "method": f"EPO+PLS({n_comp})+BoxCox", "rmse": rmse, "rmse_std": std})
        print(f"  EPO+PLS({n_comp})+BoxCox: RMSE={rmse:.2f} ± {std:.2f}")

    # SNV + log変換（SNV+PLS3がベスト個別モデルなので）
    print("\n--- SNV + 目的変数変換 ---")
    for transform_name, y_t, inv_fn in [
        ("log", np.log1p(y), np.expm1),
        ("sqrt", np.sqrt(y), lambda x: np.clip(x, 0, None) ** 2),
    ]:
        for n_comp in [3, 4, 5]:
            logo = LeaveOneGroupOut()
            fold_rmses = []
            for train_idx, test_idx in logo.split(X_raw, y, groups):
                X_tr = apply_snv(X_raw[train_idx])
                X_te = apply_snv(X_raw[test_idx])
                pls = PLSRegression(n_components=n_comp)
                pls.fit(X_tr, y_t[train_idx])
                pred_t = pls.predict(X_te).ravel()
                pred = inv_fn(pred_t)
                fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
            rmse, std = np.mean(fold_rmses), np.std(fold_rmses)
            all_results.append({"category": "目的変数変換", "method": f"SNV+PLS({n_comp})+{transform_name}(y)", "rmse": rmse, "rmse_std": std})
            print(f"  SNV+PLS({n_comp})+{transform_name}(y): RMSE={rmse:.2f} ± {std:.2f}")

    # ============================================================
    # ベースライン
    # ============================================================
    print("\n--- ベースライン ---")
    # EPO(1)+PLS(4) baseline
    rmse_base, std_base = loso_cv_preprocess(
        X_raw, y, groups,
        lambda X, tri, tei, g: epo_preprocess_arrays(X, tri, tei, g, 1),
        lambda: PLSRegression(n_components=4),
    )
    all_results.append({"category": "ベースライン", "method": "EPO(1)+PLS(4)", "rmse": rmse_base, "rmse_std": std_base})
    print(f"  EPO(1)+PLS(4): RMSE={rmse_base:.2f} ± {std_base:.2f}")

    # Raw+PLS(4)
    rmse_raw, std_raw = loso_cv(X_raw, y, groups, lambda: PLSRegression(n_components=4))
    all_results.append({"category": "ベースライン", "method": "Raw+PLS(4)", "rmse": rmse_raw, "rmse_std": std_raw})
    print(f"  Raw+PLS(4): RMSE={rmse_raw:.2f} ± {std_raw:.2f}")

    # ============================================================
    # 全結果まとめ
    # ============================================================
    result_df = pd.DataFrame(all_results).sort_values("rmse").reset_index(drop=True)
    print("\n" + "=" * 60)
    print("全結果（上位20）")
    print("=" * 60)
    print(result_df.head(20).to_string(index=False))

    # カテゴリ別ベスト
    print("\n--- カテゴリ別ベスト ---")
    for cat in result_df["category"].unique():
        best = result_df[result_df["category"] == cat].iloc[0]
        print(f"  [{cat}] {best['method']}: RMSE={best['rmse']:.2f}")

    # CSVに保存
    result_df.to_csv(OUT_DIR / "deep_dive_results.csv", index=False)

    # プロット: 上位25
    top25 = result_df.head(25)
    fig, ax = plt.subplots(figsize=(12, 10))
    cat_colors = {
        "SNV深掘り": "skyblue",
        "SG+EPO": "salmon",
        "CARS/波長選択": "lightgreen",
        "目的変数変換": "plum",
        "ベースライン": "gold",
    }
    colors = [cat_colors.get(c, "gray") for c in top25["category"]]
    ax.barh(range(len(top25)), top25["rmse"], color=colors)
    ax.errorbar(top25["rmse"], range(len(top25)), xerr=top25["rmse_std"],
                fmt="none", color="black", capsize=3)
    ax.set_yticks(range(len(top25)))
    ax.set_yticklabels(top25["method"], fontsize=7)
    ax.set_xlabel("RMSE (LOSO-CV)")
    ax.set_title("Deep Dive: Preprocessing Approach Comparison (Top 25)")
    ax.invert_yaxis()

    # 凡例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=v, label=k) for k, v in cat_colors.items()]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "deep_dive_results.png", dpi=150)
    plt.close()
    print(f"\nSaved: {OUT_DIR / 'deep_dive_results.png'}")
    print(f"Saved: {OUT_DIR / 'deep_dive_results.csv'}")


if __name__ == "__main__":
    main()
