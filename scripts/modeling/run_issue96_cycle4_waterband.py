"""Issue #96: サイクル4 - 水分吸収帯の樹種不変特徴量（改良版）

Cycle 27で水分帯特徴量を試したが単独ではRMSE 20-40と悪い。
PLSスコアとバンド比を適切に結合し、GBR/Ridgeで回帰する。

戦略:
(A) PLSスコア(4) + バンド比特徴量 → Ridge
(B) PLSスコア(4) + バンド比特徴量 → GBR(depth=2, n_est=100)
(C) EPO+PLSスコア(4) + SNV後バンド比 → Ridge
(D) バンド比特徴量のみ → Ridge（ベースライン）
(E) PLSスコア(4)のみ → Ridge（ベースライン）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import warnings
import time

warnings.filterwarnings("ignore")

from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut

from src.eda.data_loader import load_train, load_test, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _wn_idx(wavenumbers, target):
    """波数配列から目標波数に最も近いインデックスを返す"""
    return int(np.argmin(np.abs(wavenumbers - target)))


def _wn_range_mask(wavenumbers, center, half_width=50):
    """center ± half_width の範囲のブールマスクを返す"""
    return (wavenumbers >= center - half_width) & (wavenumbers <= center + half_width)


def compute_waterband_features(X, wavenumbers, use_snv=False):
    """水分吸収帯ベースのバンド比・差分・面積特徴量を計算する。

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        スペクトルデータ
    wavenumbers : np.ndarray of shape (n_features,)
        波数配列
    use_snv : bool
        TrueならSNV適用後のスペクトルからも特徴量を計算して追加

    Returns
    -------
    np.ndarray of shape (n_samples, n_features_out)
    list[str] : 特徴量名
    """
    eps = 1e-10
    features = {}

    # --- 主要波数のインデックス ---
    idx_5200 = _wn_idx(wavenumbers, 5200)
    idx_6900 = _wn_idx(wavenumbers, 6900)
    idx_5800 = _wn_idx(wavenumbers, 5800)

    # --- 近傍平均 (±50 cm⁻¹) ---
    mask_5200 = _wn_range_mask(wavenumbers, 5200, 50)
    mask_6900 = _wn_range_mask(wavenumbers, 6900, 50)
    mask_5800 = _wn_range_mask(wavenumbers, 5800, 50)

    A5200 = np.mean(X[:, mask_5200], axis=1)
    A6900 = np.mean(X[:, mask_6900], axis=1)
    A5800 = np.mean(X[:, mask_5800], axis=1)

    # 1. バンド比
    features["ratio_5200_6900"] = A5200 / (A6900 + eps)
    features["ratio_5200_5800"] = A5200 / (A5800 + eps)
    features["ratio_6900_5800"] = A6900 / (A5800 + eps)

    # 2. バンド差
    features["diff_5200_6900"] = A5200 - A6900
    features["diff_5200_5800"] = A5200 - A5800

    # 3. 局所面積 (±100 cm⁻¹の台形積分)
    mask_5200_wide = _wn_range_mask(wavenumbers, 5200, 100)
    mask_6900_wide = _wn_range_mask(wavenumbers, 6900, 100)
    # 台形積分: 波数でソートしてから積分
    for name, mask in [("area_5200", mask_5200_wide), ("area_6900", mask_6900_wide)]:
        if np.sum(mask) > 1:
            wn_sub = wavenumbers[mask]
            sort_idx = np.argsort(wn_sub)
            features[name] = np.trapezoid(X[:, mask][:, sort_idx], wn_sub[sort_idx], axis=1)
        else:
            features[name] = X[:, _wn_idx(wavenumbers, int(name.split("_")[1]))]

    # 4. 2次微分ピーク (Savitzky-Golay)
    X_d2 = savgol_filter(X, window_length=11, polyorder=2, deriv=2, axis=1)
    features["sg2d_5200"] = X_d2[:, idx_5200]
    features["sg2d_6900"] = X_d2[:, idx_6900]

    # 5. 正規化差分指標 (NDMI)
    features["ndmi"] = (A5200 - A6900) / (A5200 + A6900 + eps)

    # --- SNV後の特徴量 ---
    if use_snv:
        X_snv = apply_snv(X)
        A5200_s = np.mean(X_snv[:, mask_5200], axis=1)
        A6900_s = np.mean(X_snv[:, mask_6900], axis=1)
        A5800_s = np.mean(X_snv[:, mask_5800], axis=1)

        features["snv_ratio_5200_6900"] = A5200_s / (A6900_s + eps)
        features["snv_ratio_5200_5800"] = A5200_s / (A5800_s + eps)
        features["snv_ratio_6900_5800"] = A6900_s / (A5800_s + eps)
        features["snv_diff_5200_6900"] = A5200_s - A6900_s
        features["snv_ndmi"] = (A5200_s - A6900_s) / (A5200_s + A6900_s + eps)

        X_snv_d2 = savgol_filter(X_snv, window_length=11, polyorder=2, deriv=2, axis=1)
        features["snv_sg2d_5200"] = X_snv_d2[:, idx_5200]
        features["snv_sg2d_6900"] = X_snv_d2[:, idx_6900]

    names = list(features.keys())
    result = np.column_stack([features[k] for k in names])
    return result, names


def run_strategy(X_tr, X_te, y_tr, groups_tr, wavenumbers,
                 strategy, target_transform):
    """各戦略でtrain/test予測を行う。

    Parameters
    ----------
    strategy : str
        "A", "B", "C", "D", "E"のいずれか
    target_transform : str
        "sqrt" or "raw"

    Returns
    -------
    pred : np.ndarray, テスト予測
    """
    n_pls = 4
    yf = np.sqrt(y_tr) if target_transform == "sqrt" else y_tr.copy()

    if strategy == "A":
        # PLSスコア(4) + バンド比特徴量 → Ridge
        pls = PLSRegression(n_components=n_pls)
        pls.fit(X_tr, yf)
        T_tr = pls.transform(X_tr)
        T_te = pls.transform(X_te)
        wb_tr, _ = compute_waterband_features(X_tr, wavenumbers, use_snv=False)
        wb_te, _ = compute_waterband_features(X_te, wavenumbers, use_snv=False)
        F_tr = np.hstack([T_tr, wb_tr])
        F_te = np.hstack([T_te, wb_te])
        sc = StandardScaler()
        F_tr_s = sc.fit_transform(F_tr)
        F_te_s = sc.transform(F_te)
        m = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        m.fit(F_tr_s, yf)
        pred = m.predict(F_te_s)

    elif strategy == "B":
        # PLSスコア(4) + バンド比特徴量 → GBR(depth=2, n_est=100)
        pls = PLSRegression(n_components=n_pls)
        pls.fit(X_tr, yf)
        T_tr = pls.transform(X_tr)
        T_te = pls.transform(X_te)
        wb_tr, _ = compute_waterband_features(X_tr, wavenumbers, use_snv=False)
        wb_te, _ = compute_waterband_features(X_te, wavenumbers, use_snv=False)
        F_tr = np.hstack([T_tr, wb_tr])
        F_te = np.hstack([T_te, wb_te])
        sc = StandardScaler()
        F_tr_s = sc.fit_transform(F_tr)
        F_te_s = sc.transform(F_te)
        m = GradientBoostingRegressor(
            n_estimators=100, max_depth=2, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5, random_state=42,
            validation_fraction=0.15, n_iter_no_change=50, tol=0.01)
        m.fit(F_tr_s, yf)
        pred = m.predict(F_te_s)

    elif strategy == "C":
        # EPO+PLSスコア(4) + SNV後バンド比 → Ridge
        P = compute_epo_projection(X_tr, groups_tr, n_components=1)
        X_tr_epo = apply_epo(X_tr, P)
        X_te_epo = apply_epo(X_te, P)
        pls = PLSRegression(n_components=n_pls)
        pls.fit(X_tr_epo, yf)
        T_tr = pls.transform(X_tr_epo)
        T_te = pls.transform(X_te_epo)
        # SNV後のバンド比特徴量
        wb_tr, _ = compute_waterband_features(X_tr, wavenumbers, use_snv=True)
        wb_te, _ = compute_waterband_features(X_te, wavenumbers, use_snv=True)
        F_tr = np.hstack([T_tr, wb_tr])
        F_te = np.hstack([T_te, wb_te])
        sc = StandardScaler()
        F_tr_s = sc.fit_transform(F_tr)
        F_te_s = sc.transform(F_te)
        m = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        m.fit(F_tr_s, yf)
        pred = m.predict(F_te_s)

    elif strategy == "D":
        # バンド比特徴量のみ → Ridge（ベースライン）
        wb_tr, _ = compute_waterband_features(X_tr, wavenumbers, use_snv=True)
        wb_te, _ = compute_waterband_features(X_te, wavenumbers, use_snv=True)
        sc = StandardScaler()
        F_tr_s = sc.fit_transform(wb_tr)
        F_te_s = sc.transform(wb_te)
        m = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        m.fit(F_tr_s, yf)
        pred = m.predict(F_te_s)

    elif strategy == "E":
        # PLSスコア(4)のみ → Ridge（ベースライン）
        pls = PLSRegression(n_components=n_pls)
        pls.fit(X_tr, yf)
        T_tr = pls.transform(X_tr)
        T_te = pls.transform(X_te)
        sc = StandardScaler()
        F_tr_s = sc.fit_transform(T_tr)
        F_te_s = sc.transform(T_te)
        m = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        m.fit(F_tr_s, yf)
        pred = m.predict(F_te_s)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if target_transform == "sqrt":
        pred = np.clip(pred, 0, None) ** 2

    return pred


def main():
    t0 = time.time()
    df = load_train(DATA_DIR)
    spectral_cols = get_spectral_columns(df)
    X = df[spectral_cols].values
    y = df["含水率"].values
    groups = df["樹種"].values
    wavenumbers = np.array([float(c) for c in spectral_cols])

    logo = LeaveOneGroupOut()
    folds = list(logo.split(X, y, groups))
    species_per_fold = [np.unique(groups[te])[0] for _, te in folds]

    print("=" * 70)
    print("Issue #96: サイクル4 - 水分吸収帯の樹種不変特徴量（改良版）")
    print("=" * 70)

    # --- 水分帯特徴量の確認 ---
    wb_feat, wb_names = compute_waterband_features(X, wavenumbers, use_snv=True)
    print(f"\n水分帯特徴量数: {len(wb_names)}")
    print(f"特徴量名: {wb_names}")

    # --- 各戦略のLOSO-CV ---
    strategies = ["A", "B", "C", "D", "E"]
    transforms = ["sqrt", "raw"]
    strategy_labels = {
        "A": "PLS4+WB → Ridge",
        "B": "PLS4+WB → GBR(d=2,n=100)",
        "C": "EPO+PLS4+SNV_WB → Ridge",
        "D": "WBのみ → Ridge（ベースライン）",
        "E": "PLS4のみ → Ridge（ベースライン）",
    }

    results = []

    print(f"\n--- LOSO-CV 評価 ({len(strategies)} 戦略 × {len(transforms)} 変換) ---\n")

    for strat in strategies:
        for tf in transforms:
            fold_rmses = []
            fold_preds = []
            for fi, (tr, te) in enumerate(folds):
                try:
                    pred = run_strategy(
                        X[tr], X[te], y[tr], groups[tr],
                        wavenumbers, strat, tf)
                    fold_rmses.append(rmse(y[te], pred))
                    fold_preds.append(pred)
                except Exception as e:
                    fold_rmses.append(999.0)
                    fold_preds.append(np.full(len(te), y[tr].mean()))
                    print(f"  ERR {strat}/{tf} fold{fi}: {e}")

            avg_rmse = np.mean(fold_rmses)
            bei_idx = [j for j, s in enumerate(species_per_fold) if s == "ベイスギ"]
            bei_rmse = fold_rmses[bei_idx[0]] if bei_idx else 0
            non_bei = [r for j, r in enumerate(fold_rmses) if j not in bei_idx]
            avg_non_bei = np.mean(non_bei) if non_bei else avg_rmse

            label = f"({strat}) {strategy_labels[strat]} [{tf}]"
            print(f"  {label}: RMSE={avg_rmse:.2f} "
                  f"(ベイスギ={bei_rmse:.1f}, 除ベイスギ={avg_non_bei:.2f})")

            # fold別RMSE
            for j, (s, r) in enumerate(zip(species_per_fold, fold_rmses)):
                marker = " ※参考" if s == "ベイスギ" else ""
                print(f"    {s}: {r:.2f}{marker}")

            results.append({
                "strategy": strat,
                "label": strategy_labels[strat],
                "transform": tf,
                "rmse": avg_rmse,
                "rmse_excl_beisugi": avg_non_bei,
                "bei_rmse": bei_rmse,
                "fold_rmses": fold_rmses,
                "fold_preds": fold_preds,
            })

    # --- ベスト戦略の特定 ---
    results_df = pd.DataFrame([{
        "strategy": r["strategy"],
        "label": r["label"],
        "transform": r["transform"],
        "rmse": r["rmse"],
        "rmse_excl_beisugi": r["rmse_excl_beisugi"],
        "bei_rmse": r["bei_rmse"],
        "fold_rmses": str([f"{v:.2f}" for v in r["fold_rmses"]]),
    } for r in results])

    results_df = results_df.sort_values("rmse").reset_index(drop=True)
    print("\n--- ランキング ---")
    for _, row in results_df.iterrows():
        print(f"  RMSE={row['rmse']:.2f} | ({row['strategy']}) {row['label']} [{row['transform']}]")

    best = min(results, key=lambda r: r["rmse"])
    print(f"\n{'=' * 70}")
    print(f"BEST: ({best['strategy']}) {best['label']} [{best['transform']}] "
          f"= RMSE {best['rmse']:.4f}")
    print(f"{'=' * 70}")

    # --- テスト予測 ---
    print("\n--- テスト予測生成 ---")
    df_test = load_test(DATA_DIR)
    test_spectral_cols = get_spectral_columns(df_test)
    X_test = df_test[test_spectral_cols].values

    pred_test = run_strategy(
        X, X_test, y, groups,
        wavenumbers, best["strategy"], best["transform"])

    # 提出ファイル作成
    submit = pd.read_csv(DATA_DIR / "sample_submit.csv", header=None)
    submit[1] = pred_test
    submit_path = OUT_DIR.parent / "submission_v6_waterband.csv"
    submit.to_csv(submit_path, index=False, header=False)
    print(f"提出ファイル保存: {submit_path}")
    print(f"予測統計: mean={pred_test.mean():.2f}, std={pred_test.std():.2f}, "
          f"min={pred_test.min():.2f}, max={pred_test.max():.2f}")

    # 結果CSV保存
    results_df.to_csv(OUT_DIR / "issue96_cycle4_results.csv", index=False)
    print(f"結果保存: {OUT_DIR / 'issue96_cycle4_results.csv'}")
    print(f"\n総実行時間: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
