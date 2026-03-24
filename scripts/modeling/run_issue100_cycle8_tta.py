"""Issue #100: Cycle 8 - Test-Time Augmentation (TTA)

テスト時にスペクトルに微小な摂動を加えた複数バージョンで予測し、
平均化することで予測を安定化する。

既存ベスト4モデル（submission_v5相当のPLSモデル）にTTAを適用:
1. EPO(1)+PLS(4)+sqrt
2. SG2d+EPO(1)+PLS(3)+raw
3. SNV+AsLS+siPLS(30,3)+PLS(5)+sqrt
4. SNV+iPLS(50)+PLS(4)+sqrt

最適化: train依存の前処理（EPO projection, PLSモデル, 特徴量選択index）を
事前にキャッシュし、TTA拡張版ごとにテスト側の前処理と予測のみ実行。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import warnings
import time
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut

from src.eda.data_loader import load_train, load_test, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue20_savgol import apply_savgol
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue19_msc import compute_msc_reference
from src.preprocessing.issue62_additional_preprocessing import apply_asls, apply_piecewise_msc
from src.modeling.issue65_feature_selection import sipls_select, ipls_select

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUT = OUT_DIR / "modeling"
MODEL_OUT.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ============================================================
# 前処理・特徴量選択（run_issue87_final.pyと同じ実装）
# ============================================================

def pp(X_tr, X_te, g, pp_name):
    """前処理を適用する。"""
    if pp_name == "SNV":
        return apply_snv(X_tr), apply_snv(X_te)
    elif pp_name == "EPO(1)":
        P = compute_epo_projection(X_tr, g, n_components=1)
        return apply_epo(X_tr, P), apply_epo(X_te, P)
    elif pp_name == "SNV+AsLS(1e6)":
        return apply_asls(apply_snv(X_tr), lam=1e6), apply_asls(apply_snv(X_te), lam=1e6)
    elif pp_name == "PMSC":
        ref = compute_msc_reference(X_tr)
        return apply_piecewise_msc(X_tr, ref, 3), apply_piecewise_msc(X_te, ref, 3)
    elif pp_name == "SG2d+EPO(1)":
        xs = apply_savgol(X_tr, deriv=2, window_length=7)
        xst = apply_savgol(X_te, deriv=2, window_length=7)
        P = compute_epo_projection(xs, g, n_components=1)
        return apply_epo(xs, P), apply_epo(xst, P)
    elif pp_name == "SNV+SG2d":
        return (apply_savgol(apply_snv(X_tr), deriv=2, window_length=7),
                apply_savgol(apply_snv(X_te), deriv=2, window_length=7))
    return X_tr.copy(), X_te.copy()


def fs(X_tr, X_te, y, fs_name):
    """特徴量選択を適用する。"""
    if not fs_name:
        return X_tr, X_te
    if fs_name.startswith("siPLS"):
        p = fs_name.replace("siPLS(", "").rstrip(")").split(",")
        X_tr, X_te, _ = sipls_select(
            X_tr, y, X_te, n_intervals=int(p[0]),
            n_components=3, n_combine=int(p[1])
        )
    elif fs_name.startswith("iPLS"):
        n = int(fs_name.replace("iPLS(", "").rstrip(")"))
        X_tr, X_te, _ = ipls_select(
            X_tr, y, X_te, n_intervals=n, n_components=3, n_best=1
        )
    return X_tr, X_te


def fs_get_indices(X_tr, y, fs_name):
    """特徴量選択のインデックスのみを取得する。"""
    if not fs_name:
        return None
    dummy_te = X_tr[:1]  # ダミー
    if fs_name.startswith("siPLS"):
        p = fs_name.replace("siPLS(", "").rstrip(")").split(",")
        _, _, idx = sipls_select(
            X_tr, y, dummy_te, n_intervals=int(p[0]),
            n_components=3, n_combine=int(p[1])
        )
        return idx
    elif fs_name.startswith("iPLS"):
        n = int(fs_name.replace("iPLS(", "").rstrip(")"))
        _, _, idx = ipls_select(
            X_tr, y, dummy_te, n_intervals=n, n_components=3, n_best=1
        )
        return idx
    return None


# ============================================================
# TTA: Test-Time Augmentation
# ============================================================

def generate_augmented_spectra(X, n_aug=10, seed=42,
                                offset_std=0.001, slope_std=1e-6,
                                scale_std=0.002, noise_std=0.0005):
    """スペクトルの拡張バージョンを生成する。

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        元のスペクトル行列
    n_aug : int
        生成する拡張版の数（元のスペクトルを除く）
    seed : int
        乱数シード
    offset_std, slope_std, scale_std, noise_std : float
        各種ノイズの標準偏差

    Returns
    -------
    list of ndarray
        元のスペクトル + n_aug個の拡張版（合計 n_aug+1 個）
    """
    rng = np.random.RandomState(seed)
    augmented = [X.copy()]  # 元のスペクトルも含む

    for i in range(n_aug):
        X_aug = X.copy()
        # 1. ベースラインオフセット
        offset = rng.normal(0, offset_std, size=(X.shape[0], 1))
        X_aug += offset
        # 2. スロープ変動
        slope = rng.normal(0, slope_std, size=(X.shape[0], 1))
        X_aug += slope * np.arange(X.shape[1]).reshape(1, -1)
        # 3. 強度スケーリング
        scale = rng.normal(1.0, scale_std, size=(X.shape[0], 1))
        X_aug *= scale
        # 4. ガウシアンノイズ
        noise = rng.normal(0, noise_std, size=X.shape)
        X_aug += noise
        augmented.append(X_aug)

    return augmented


# ============================================================
# 高速TTA: 前処理キャッシュ付き
# ============================================================

def pp_test_only(X_te_raw, pp_name, cache):
    """テスト側の前処理のみ適用（train依存パラメータはcacheから取得）。"""
    if pp_name == "SNV":
        return apply_snv(X_te_raw)
    elif pp_name == "EPO(1)":
        P = cache["epo_P"]
        return apply_epo(X_te_raw, P)
    elif pp_name == "SNV+AsLS(1e6)":
        return apply_asls(apply_snv(X_te_raw), lam=1e6)
    elif pp_name == "PMSC":
        ref = cache["msc_ref"]
        return apply_piecewise_msc(X_te_raw, ref, 3)
    elif pp_name == "SG2d+EPO(1)":
        xst = apply_savgol(X_te_raw, deriv=2, window_length=7)
        P = cache["epo_P"]
        return apply_epo(xst, P)
    elif pp_name == "SNV+SG2d":
        return apply_savgol(apply_snv(X_te_raw), deriv=2, window_length=7)
    return X_te_raw.copy()


def build_train_cache(X_tr_raw, y_tr, g_tr, cfg):
    """train側の前処理・特徴量選択・PLSモデルを事前計算してキャッシュする。

    Returns
    -------
    cache : dict
        epo_P, msc_ref, fs_idx, pls_model, tf, pp_name 等
    """
    pp_name = cfg["pp"]
    cache = {"pp_name": pp_name, "tf": cfg["tf"]}

    # 前処理（train側）+ train依存パラメータ
    if pp_name == "SNV":
        Xtr_pp = apply_snv(X_tr_raw)
    elif pp_name == "EPO(1)":
        P = compute_epo_projection(X_tr_raw, g_tr, n_components=1)
        cache["epo_P"] = P
        Xtr_pp = apply_epo(X_tr_raw, P)
    elif pp_name == "SNV+AsLS(1e6)":
        Xtr_pp = apply_asls(apply_snv(X_tr_raw), lam=1e6)
    elif pp_name == "PMSC":
        ref = compute_msc_reference(X_tr_raw)
        cache["msc_ref"] = ref
        Xtr_pp = apply_piecewise_msc(X_tr_raw, ref, 3)
    elif pp_name == "SG2d+EPO(1)":
        xs = apply_savgol(X_tr_raw, deriv=2, window_length=7)
        P = compute_epo_projection(xs, g_tr, n_components=1)
        cache["epo_P"] = P
        Xtr_pp = apply_epo(xs, P)
    elif pp_name == "SNV+SG2d":
        Xtr_pp = apply_savgol(apply_snv(X_tr_raw), deriv=2, window_length=7)
    else:
        Xtr_pp = X_tr_raw.copy()

    # 特徴量選択
    fs_name = cfg.get("fs")
    fs_idx = fs_get_indices(Xtr_pp, y_tr, fs_name)
    cache["fs_idx"] = fs_idx

    if fs_idx is not None:
        Xtr_sel = Xtr_pp[:, fs_idx]
    else:
        Xtr_sel = Xtr_pp

    # PLSモデル
    nc = cfg.get("nc", 4)
    tf = cfg["tf"]
    nc = max(1, min(nc, Xtr_sel.shape[1] - 1))
    yf = np.sqrt(y_tr) if tf == "sqrt" else y_tr.copy()

    pls = PLSRegression(n_components=nc)
    pls.fit(Xtr_sel, yf)
    cache["pls_model"] = pls

    return cache


def predict_from_cache(X_te_raw, cache):
    """キャッシュされたモデルでテストデータを予測する。"""
    # テスト側の前処理
    Xte_pp = pp_test_only(X_te_raw, cache["pp_name"], cache)

    # 特徴量選択
    fs_idx = cache["fs_idx"]
    if fs_idx is not None:
        Xte_sel = Xte_pp[:, fs_idx]
    else:
        Xte_sel = Xte_pp

    # PLS予測
    pls = cache["pls_model"]
    pred = pls.predict(Xte_sel).ravel()

    if cache["tf"] == "sqrt":
        return np.clip(pred, 0, None) ** 2
    return pred


# ============================================================
# 非キャッシュ版（テスト互換）
# ============================================================

def run_model_full(X_train, y_train, groups_train, X_test, cfg):
    """train全体でfit -> test predict（提出用、非キャッシュ版）。"""
    Xtr, Xte = pp(X_train, X_test, groups_train, cfg["pp"])
    Xtr, Xte = fs(Xtr, Xte, y_train, cfg.get("fs"))

    nc = cfg.get("nc", 4)
    tf = cfg["tf"]
    nc = max(1, min(nc, Xtr.shape[1] - 1))

    yf = np.sqrt(y_train) if tf == "sqrt" else y_train.copy()
    pls = PLSRegression(n_components=nc)
    pls.fit(Xtr, yf)
    pred = pls.predict(Xte).ravel()

    if tf == "sqrt":
        return np.clip(pred, 0, None) ** 2
    return pred


def run_model_cv(X_tr, X_te, y_tr, g_tr, cfg):
    """CVフォールド内でのモデル実行。"""
    return run_model_full(X_tr, y_tr, g_tr, X_te, cfg)


def predict_with_tta(X_train, y_train, groups_train, X_test, model_cfgs,
                     n_aug=10, seed=42, tta_params=None):
    """TTAで予測する（キャッシュ使用版）。"""
    if tta_params is None:
        tta_params = {}

    augmented_tests = generate_augmented_spectra(
        X_test, n_aug=n_aug, seed=seed, **tta_params
    )

    # 各モデルのキャッシュを事前構築
    caches = []
    for cfg in model_cfgs:
        cache = build_train_cache(X_train, y_train, groups_train, cfg)
        caches.append(cache)

    all_preds = []
    for X_test_aug in augmented_tests:
        preds_for_this_aug = []
        for cache in caches:
            pred = predict_from_cache(X_test_aug, cache)
            preds_for_this_aug.append(pred)
        avg_pred = np.mean(preds_for_this_aug, axis=0)
        all_preds.append(avg_pred)

    tta_pred = np.mean(all_preds, axis=0)
    return np.clip(tta_pred, 0, 300)


def predict_without_tta(X_train, y_train, groups_train, X_test, model_cfgs):
    """TTA無しで予測する（比較用）。"""
    preds = []
    for cfg in model_cfgs:
        pred = run_model_full(X_train, y_train, groups_train, X_test, cfg)
        preds.append(pred)
    avg_pred = np.mean(preds, axis=0)
    return np.clip(avg_pred, 0, 300)


# ============================================================
# LOSO-CV評価（高速版）
# ============================================================

def build_fold_caches(X, y, groups, model_cfgs):
    """全LOSOフォールドの全モデルのキャッシュを事前構築する。

    Returns
    -------
    fold_caches : list of list of dict
        fold_caches[fold_idx][model_idx] = cache dict
    folds : list of (train_idx, test_idx)
    species_list : list of str
    """
    logo = LeaveOneGroupOut()
    folds = list(logo.split(X, y, groups))
    species_list = [np.unique(groups[te])[0] for _, te in folds]

    fold_caches = []
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        X_tr = X[train_idx]
        y_tr = y[train_idx]
        g_tr = groups[train_idx]
        caches = []
        for cfg in model_cfgs:
            cache = build_train_cache(X_tr, y_tr, g_tr, cfg)
            caches.append(cache)
        fold_caches.append(caches)
        print(f"  キャッシュ構築 Fold {fold_idx+1}/{len(folds)} [{species_list[fold_idx]}]",
              flush=True)

    return fold_caches, folds, species_list


def evaluate_tta_cv_with_cache(X, y, groups, fold_caches, folds, species_list,
                                n_aug=10, seed=42, tta_params=None, verbose=True):
    """事前構築済みキャッシュを使ってTTAあり/なしを評価する。"""
    if tta_params is None:
        tta_params_use = {}
    else:
        tta_params_use = tta_params

    results_tta = []
    results_notta = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        X_te = X[test_idx]
        y_te = y[test_idx]
        sp = species_list[fold_idx]
        caches = fold_caches[fold_idx]

        # TTAなし予測
        preds_notta = []
        for cache in caches:
            pred = predict_from_cache(X_te, cache)
            preds_notta.append(pred)
        avg_notta = np.clip(np.mean(preds_notta, axis=0), 0, 300)
        r_notta = rmse(y_te, avg_notta)
        results_notta.append({"species": sp, "rmse": r_notta})

        # TTAあり予測
        if n_aug > 0:
            augmented_tests = generate_augmented_spectra(
                X_te, n_aug=n_aug, seed=seed, **tta_params_use
            )
            all_tta_preds = []
            for X_te_aug in augmented_tests:
                preds_for_aug = []
                for cache in caches:
                    pred = predict_from_cache(X_te_aug, cache)
                    preds_for_aug.append(pred)
                avg_pred = np.mean(preds_for_aug, axis=0)
                all_tta_preds.append(avg_pred)
            tta_pred = np.clip(np.mean(all_tta_preds, axis=0), 0, 300)
            r_tta = rmse(y_te, tta_pred)
        else:
            r_tta = r_notta

        results_tta.append({"species": sp, "rmse": r_tta})

        if verbose:
            print(f"  Fold {fold_idx+1}/{len(folds)} [{sp}]: "
                  f"NoTTA={r_notta:.2f}, TTA={r_tta:.2f}, "
                  f"diff={r_tta - r_notta:+.2f}")

    mean_notta = np.mean([r["rmse"] for r in results_notta])
    mean_tta = np.mean([r["rmse"] for r in results_tta])

    notta_nobs = [r["rmse"] for r in results_notta if r["species"] != "ベイスギ"]
    tta_nobs = [r["rmse"] for r in results_tta if r["species"] != "ベイスギ"]
    mean_notta_nobs = np.mean(notta_nobs) if notta_nobs else mean_notta
    mean_tta_nobs = np.mean(tta_nobs) if tta_nobs else mean_tta

    return {
        "tta": {"folds": results_tta, "mean_rmse": mean_tta, "mean_rmse_no_beisugi": mean_tta_nobs},
        "notta": {"folds": results_notta, "mean_rmse": mean_notta, "mean_rmse_no_beisugi": mean_notta_nobs},
    }


def evaluate_tta_cv(X, y, groups, model_cfgs, n_aug=10, seed=42, tta_params=None):
    """LOSO-CVでTTAあり/なしを評価する（後方互換版）。"""
    fold_caches, folds, species_list = build_fold_caches(X, y, groups, model_cfgs)
    return evaluate_tta_cv_with_cache(
        X, y, groups, fold_caches, folds, species_list,
        n_aug=n_aug, seed=seed, tta_params=tta_params
    )


# ============================================================
# メイン
# ============================================================

def main():
    t0 = time.time()

    # データ読み込み
    df_train = load_train(DATA_DIR)
    df_test = load_test(DATA_DIR)
    sc = get_spectral_columns(df_train)

    X_train = df_train[sc].values
    y_train = df_train["含水率"].values
    groups_train = df_train["樹種"].values
    X_test = df_test[sc].values
    test_ids = df_test["sample number"].values

    print("=" * 70)
    print("Issue #100: Cycle 8 - Test-Time Augmentation (TTA)")
    print("=" * 70)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # モデル設定（submission_v5相当のベスト4 PLSモデル）
    model_cfgs = [
        {"name": "M1:EPO+PLS4+sqrt", "pp": "EPO(1)", "nc": 4, "tf": "sqrt"},
        {"name": "M2:SG2d+EPO+PLS3+raw", "pp": "SG2d+EPO(1)", "nc": 3, "tf": "raw"},
        {"name": "M5:SNV+AsLS+siPLS+PLS5+sqrt", "pp": "SNV+AsLS(1e6)", "nc": 5,
         "tf": "sqrt", "fs": "siPLS(30,3)"},
        {"name": "M3:SNV+iPLS50+PLS4+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt",
         "fs": "iPLS(50)"},
    ]

    # ========================================
    # キャッシュ構築（1回のみ、全フェーズで再利用）
    # ========================================
    print("\n--- キャッシュ構築（全フォールド x 全モデル）---")
    t_cache = time.time()
    fold_caches, folds, species_list = build_fold_caches(
        X_train, y_train, groups_train, model_cfgs
    )
    print(f"  キャッシュ構築完了: {time.time() - t_cache:.1f}s")

    # ========================================
    # Phase 1: LOSO-CVベースライン
    # ========================================
    print("\n--- Phase 1: ベースライン（TTAなし）LOSO-CV ---")
    cv_base = evaluate_tta_cv_with_cache(
        X_train, y_train, groups_train, fold_caches, folds, species_list,
        n_aug=0, seed=42, tta_params=None
    )
    baseline_rmse = cv_base['notta']['mean_rmse']
    print(f"\nベースライン RMSE: {baseline_rmse:.4f}")

    # ========================================
    # Phase 2: デフォルトTTA
    # ========================================
    print("\n--- Phase 2: デフォルトTTA (n_aug=10) ---")
    cv_default = evaluate_tta_cv_with_cache(
        X_train, y_train, groups_train, fold_caches, folds, species_list,
        n_aug=10, seed=42, tta_params=None
    )
    print(f"\nデフォルトTTA RMSE: {cv_default['tta']['mean_rmse']:.4f}")
    print(f"ベースライン RMSE: {cv_default['notta']['mean_rmse']:.4f}")
    print(f"差分: {cv_default['tta']['mean_rmse'] - cv_default['notta']['mean_rmse']:+.4f}")

    # ========================================
    # Phase 3: ノイズレベル最適化
    # ========================================
    print("\n--- Phase 3: ノイズレベル最適化 ---")

    param_grids = {
        "offset_std": [0.0005, 0.001, 0.002, 0.005],
        "slope_std": [1e-7, 5e-7, 1e-6, 5e-6],
        "scale_std": [0.001, 0.002, 0.005, 0.01],
        "noise_std": [0.0002, 0.0005, 0.001, 0.002],
    }

    default_params = {
        "offset_std": 0.001,
        "slope_std": 1e-6,
        "scale_std": 0.002,
        "noise_std": 0.0005,
    }

    best_params = default_params.copy()
    best_rmse_overall = cv_default['tta']['mean_rmse']
    grid_results = []

    for param_name, values in param_grids.items():
        print(f"\n  Tuning {param_name}...")
        for val in values:
            params = best_params.copy()
            params[param_name] = val

            cv_result = evaluate_tta_cv_with_cache(
                X_train, y_train, groups_train, fold_caches, folds, species_list,
                n_aug=10, seed=42, tta_params=params, verbose=False
            )
            r = cv_result['tta']['mean_rmse']
            grid_results.append({
                "param": param_name, "value": val,
                "rmse_tta": r,
                "rmse_notta": cv_result['notta']['mean_rmse'],
            })
            print(f"    {param_name}={val:.1e}: TTA RMSE={r:.4f}")

            if r < best_rmse_overall:
                best_rmse_overall = r
                best_params[param_name] = val

    print(f"\n  ベストパラメータ: {best_params}")
    print(f"  ベストRMSE: {best_rmse_overall:.4f}")

    # ========================================
    # Phase 4: n_aug最適化
    # ========================================
    print("\n--- Phase 4: n_aug最適化 ---")
    n_aug_values = [5, 10, 20, 50]
    n_aug_results = []

    for n_aug in n_aug_values:
        cv_result = evaluate_tta_cv_with_cache(
            X_train, y_train, groups_train, fold_caches, folds, species_list,
            n_aug=n_aug, seed=42, tta_params=best_params, verbose=False
        )
        r = cv_result['tta']['mean_rmse']
        n_aug_results.append({"n_aug": n_aug, "rmse_tta": r})
        print(f"  n_aug={n_aug}: TTA RMSE={r:.4f}")

    best_n_aug_entry = min(n_aug_results, key=lambda x: x["rmse_tta"])
    best_n_aug = best_n_aug_entry["n_aug"]
    print(f"\n  ベストn_aug: {best_n_aug} (RMSE={best_n_aug_entry['rmse_tta']:.4f})")

    # ========================================
    # Phase 5: 最終評価
    # ========================================
    print("\n--- Phase 5: 最終評価 ---")
    cv_final = evaluate_tta_cv_with_cache(
        X_train, y_train, groups_train, fold_caches, folds, species_list,
        n_aug=best_n_aug, seed=42, tta_params=best_params
    )
    print(f"\n最終TTA RMSE: {cv_final['tta']['mean_rmse']:.4f}")
    print(f"ベースライン RMSE: {cv_final['notta']['mean_rmse']:.4f}")
    print(f"差分: {cv_final['tta']['mean_rmse'] - cv_final['notta']['mean_rmse']:+.4f}")

    print("\nFold別結果:")
    for tta_f, notta_f in zip(cv_final['tta']['folds'], cv_final['notta']['folds']):
        sp = tta_f['species']
        mark = " ※" if sp == "ベイスギ" else ""
        print(f"  {sp}: NoTTA={notta_f['rmse']:.2f}, TTA={tta_f['rmse']:.2f}, "
              f"diff={tta_f['rmse'] - notta_f['rmse']:+.2f}{mark}")

    notta_nobs = cv_final['notta']['mean_rmse_no_beisugi']
    tta_nobs = cv_final['tta']['mean_rmse_no_beisugi']
    print(f"\n除ベイスギ: NoTTA={notta_nobs:.4f}, TTA={tta_nobs:.4f}")

    # ========================================
    # Phase 6: 提出ファイル生成
    # ========================================
    print("\n--- Phase 6: 提出ファイル生成 ---")

    # TTAあり
    pred_tta = predict_with_tta(
        X_train, y_train, groups_train, X_test, model_cfgs,
        n_aug=best_n_aug, seed=42, tta_params=best_params
    )
    sub_tta = pd.DataFrame({
        "sample number": test_ids.astype(int),
        "含水率": pred_tta,
    })
    sub_tta.to_csv(OUT_DIR / "submission_v6_tta.csv", index=False, header=False)
    print(f"  TTA提出ファイル: {OUT_DIR / 'submission_v6_tta.csv'}")
    print(f"  予測統計: mean={pred_tta.mean():.2f}, std={pred_tta.std():.2f}, "
          f"min={pred_tta.min():.2f}, max={pred_tta.max():.2f}")

    # TTAなし
    pred_notta = predict_without_tta(
        X_train, y_train, groups_train, X_test, model_cfgs
    )
    sub_notta = pd.DataFrame({
        "sample number": test_ids.astype(int),
        "含水率": pred_notta,
    })
    sub_notta.to_csv(OUT_DIR / "submission_v6_notta.csv", index=False, header=False)
    print(f"  NoTTA提出ファイル: {OUT_DIR / 'submission_v6_notta.csv'}")
    print(f"  予測統計: mean={pred_notta.mean():.2f}, std={pred_notta.std():.2f}, "
          f"min={pred_notta.min():.2f}, max={pred_notta.max():.2f}")

    # 差分
    diff = pred_tta - pred_notta
    print(f"\n  TTA - NoTTA差分: mean={diff.mean():.4f}, std={diff.std():.4f}, "
          f"max_abs={np.max(np.abs(diff)):.4f}")

    # ========================================
    # 結果保存
    # ========================================
    pd.DataFrame(grid_results).to_csv(
        MODEL_OUT / "issue100_tta_grid_results.csv", index=False
    )
    pd.DataFrame(n_aug_results).to_csv(
        MODEL_OUT / "issue100_tta_naug_results.csv", index=False
    )

    print(f"\n{'='*70}")
    print("サマリー:")
    print(f"  ベースライン RMSE: {baseline_rmse:.4f}")
    print(f"  デフォルトTTA RMSE: {cv_default['tta']['mean_rmse']:.4f}")
    print(f"  最適化TTA RMSE: {cv_final['tta']['mean_rmse']:.4f}")
    print(f"  改善: {baseline_rmse - cv_final['tta']['mean_rmse']:+.4f}")
    print(f"  ベストパラメータ: {best_params}")
    print(f"  ベストn_aug: {best_n_aug}")
    print(f"  経過時間: {time.time() - t0:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
