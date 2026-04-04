"""Issue #68: パターン別Nelder-Mead重み最適化 & サブミッションCSV生成

方式:
  1. 標準LOSO-CV（13 fold: 12種で学習 → 1種を推定）を全24モデルで実行
  2. 各パターンが指定する樹種foldのRMSEを目的関数として、
     全24モデルのアンサンブル重みをNelder-Meadで最適化
  3. パターンごとに異なる重み → 異なるtest予測 → submission CSV

学習は全パターン共通（全13種）、重み配分のみパターンごとに最適化。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import warnings
import time
from scipy.optimize import minimize
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut
from src.eda.data_loader import load_train, load_test, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue20_savgol import apply_savgol
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue19_msc import compute_msc_reference, apply_msc
from src.preprocessing.issue62_additional_preprocessing import apply_asls, apply_piecewise_msc
from src.analysis.issue68_cv_pattern_selection import (
    select_pattern_e_sample_count,
    select_pattern_f_moisture_coverage,
    select_pattern_g_stable_folds,
    select_pattern_h_wood_type,
    select_pattern_k_spectral_homogeneity,
    select_pattern_l_wasserstein,
    select_pattern_m_correlation_stability,
)

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "submissions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 既存パターン A-C
PATTERN_A = {"ヒノキ", "ナラ", "米ヒバ", "スプルース", "クリ", "ホワイトオーク"}
PATTERN_B = {"ナラ", "米ヒバ", "スプルース", "ベイマツ", "トチ", "クリ"}
PATTERN_C = {"ホワイトオーク", "スプルース", "チェリー", "ウォールナット"}
TEST_SPECIES = {"クスノキ", "ケヤキ", "スギ", "タモ", "チーク", "ヤマザクラ"}


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ============================================================
# 前処理関数群
# ============================================================
def pp_raw(Xtr, Xte, g_tr):
    return Xtr.copy(), Xte.copy()

def pp_snv(Xtr, Xte, g_tr):
    return apply_snv(Xtr), apply_snv(Xte)

def pp_epo1(Xtr, Xte, g_tr):
    P = compute_epo_projection(Xtr, g_tr, 1)
    return apply_epo(Xtr, P), apply_epo(Xte, P)

def pp_epo2(Xtr, Xte, g_tr):
    P = compute_epo_projection(Xtr, g_tr, 2)
    return apply_epo(Xtr, P), apply_epo(Xte, P)

def pp_snv_epo1(Xtr, Xte, g_tr):
    Xtr_s, Xte_s = apply_snv(Xtr), apply_snv(Xte)
    P = compute_epo_projection(Xtr_s, g_tr, 1)
    return apply_epo(Xtr_s, P), apply_epo(Xte_s, P)

def pp_sg1d(Xtr, Xte, g_tr):
    return (
        apply_savgol(Xtr, deriv=1, window_length=11),
        apply_savgol(Xte, deriv=1, window_length=11),
    )

def pp_sg2d_epo1(Xtr, Xte, g_tr):
    xs = apply_savgol(Xtr, deriv=2, window_length=7)
    xst = apply_savgol(Xte, deriv=2, window_length=7)
    P = compute_epo_projection(xs, g_tr, 1)
    return apply_epo(xs, P), apply_epo(xst, P)

def pp_msc(Xtr, Xte, g_tr):
    ref = compute_msc_reference(Xtr)
    return apply_msc(Xtr, ref), apply_msc(Xte, ref)

def pp_snv_sg1d(Xtr, Xte, g_tr):
    return (
        apply_savgol(apply_snv(Xtr), deriv=1, window_length=11),
        apply_savgol(apply_snv(Xte), deriv=1, window_length=11),
    )

def pp_snv_asls(Xtr, Xte, g_tr):
    return (
        apply_asls(apply_snv(Xtr), lam=1e6),
        apply_asls(apply_snv(Xte), lam=1e6),
    )

def pp_pmsc(Xtr, Xte, g_tr):
    ref = compute_msc_reference(Xtr)
    return (
        apply_piecewise_msc(Xtr, ref, 3),
        apply_piecewise_msc(Xte, ref, 3),
    )

def pp_msc_epo1(Xtr, Xte, g_tr):
    ref = compute_msc_reference(Xtr)
    Xtr_m, Xte_m = apply_msc(Xtr, ref), apply_msc(Xte, ref)
    P = compute_epo_projection(Xtr_m, g_tr, 1)
    return apply_epo(Xtr_m, P), apply_epo(Xte_m, P)


PP_CONFIGS = [
    ("EPO1", pp_epo1, [3, 4], ["sqrt"]),
    ("EPO2", pp_epo2, [3, 4], ["sqrt"]),
    ("SNV", pp_snv, [3, 4], ["sqrt"]),
    ("SNV_EPO1", pp_snv_epo1, [3, 4], ["sqrt"]),
    ("SG2d_EPO1", pp_sg2d_epo1, [3], ["raw", "sqrt"]),
    ("SG1d", pp_sg1d, [3, 4], ["sqrt"]),
    ("MSC", pp_msc, [3, 4], ["sqrt"]),
    ("SNV_SG1d", pp_snv_sg1d, [3, 4], ["sqrt"]),
    ("SNV_AsLS", pp_snv_asls, [3, 4], ["sqrt"]),
    ("PMSC", pp_pmsc, [3, 4], ["sqrt"]),
    ("MSC_EPO1", pp_msc_epo1, [3, 4], ["sqrt"]),
    ("Raw", pp_raw, [3, 4], ["sqrt"]),
]


# ============================================================
# Step 1: 標準13-fold LOSO-CVを全モデルで実行
# ============================================================
def run_full_loso_cv(X_all, y_all, g_all, X_test):
    """全13 foldのLOSO-CVを実行し、各モデルのCV予測とtest予測を返す。

    Returns:
        models: list[str] — モデル名リスト
        cv_preds: list[np.ndarray] — 各モデルのCV予測 (shape: n_train,)
        test_preds: list[np.ndarray] — 各モデルのtest予測 (shape: n_test,)
        folds: list — LOSO-CVのfold情報
        species_list: list[str] — 各foldの樹種名
    """
    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_all, y_all, g_all))
    species_list = [np.unique(g_all[te])[0] for _, te in folds]

    models = []
    cv_preds = []
    test_preds = []

    total_configs = sum(len(ncs) * len(tfs) for _, _, ncs, tfs in PP_CONFIGS)
    count = 0

    for pp_name, pp_func, ncs, tfs in PP_CONFIGS:
        for nc in ncs:
            for tf in tfs:
                count += 1
                name = f"{pp_name}_PLS{nc}_{tf}"

                # LOSO-CV: 12種で学習 → 1種を推定
                cv = np.zeros_like(y_all)
                for fi, (tr, te) in enumerate(folds):
                    try:
                        Xtr, Xte = pp_func(X_all[tr], X_all[te], g_all[tr])
                        ytr = np.sqrt(y_all[tr]) if tf == "sqrt" else y_all[tr]
                        n_comp = max(1, min(nc, Xtr.shape[1] - 1))
                        pls = PLSRegression(n_components=n_comp)
                        pls.fit(Xtr, ytr)
                        p = pls.predict(Xte).ravel()
                        if tf == "sqrt":
                            p = np.clip(p, 0, None) ** 2
                        cv[te] = p
                    except Exception as e:
                        cv[te] = y_all[tr].mean()
                        print(f"    WARN {name} fold {fi}: {e}")

                # 全13種で学習 → test予測
                try:
                    Xtr_full, Xte_full = pp_func(X_all, X_test, g_all)
                    ytr_full = np.sqrt(y_all) if tf == "sqrt" else y_all
                    n_comp = max(1, min(nc, Xtr_full.shape[1] - 1))
                    pls = PLSRegression(n_components=n_comp)
                    pls.fit(Xtr_full, ytr_full)
                    tp = pls.predict(Xte_full).ravel()
                    if tf == "sqrt":
                        tp = np.clip(tp, 0, None) ** 2
                except Exception:
                    tp = np.full(X_test.shape[0], y_all.mean())

                models.append(name)
                cv_preds.append(cv.copy())
                test_preds.append(tp.copy())
                print(f"  [{count:>2}/{total_configs}] {name}: CV(全13種)={rmse(y_all, cv):.2f}", flush=True)

    return models, cv_preds, test_preds, folds, species_list


# ============================================================
# Step 2: パターン別モデルランキング & test予測
# ============================================================
def compute_pattern_cv_rmse(model_cv_preds, y_all, folds, species_list, pattern_species):
    """パターンの樹種foldのみでCV RMSEを計算"""
    pattern_fold_indices = [
        fi for fi, sp in enumerate(species_list) if sp in pattern_species
    ]
    if not pattern_fold_indices:
        return float("inf")

    fold_rmses = []
    for fi in pattern_fold_indices:
        _, te = folds[fi]
        fold_rmses.append(rmse(y_all[te], model_cv_preds[te]))
    return float(np.mean(fold_rmses))


def optimize_weights_for_pattern(
    models, cv_preds, test_preds, y_all, folds, species_list,
    pattern_species, n_restarts=40,
):
    """パターンのfold RMSEを最小化する重みをNelder-Meadで最適化。

    Returns:
        test_ensemble: np.ndarray (n_test,)
        pattern_cv: float — 最適化後のパターンfold CV RMSE
        full_cv: float — 全13種でのCV RMSE
        weights: np.ndarray — 各モデルの重み
        top_models: list[(str, float)] — 重みが大きい上位モデル
    """
    n_models = len(models)
    cv_arr = np.array(cv_preds)    # (n_models, n_train)
    test_arr = np.array(test_preds)  # (n_models, n_test)

    # パターンのfoldインデックス
    pattern_fold_indices = [
        fi for fi, sp in enumerate(species_list) if sp in pattern_species
    ]

    # L2正則化の強さ（過学習防止: fold数が少ないほど強く正則化）
    n_folds_pattern = len(pattern_fold_indices)
    lambda_l2 = 0.1 / max(n_folds_pattern, 1)

    def pattern_rmse(weights):
        """パターンfoldのみでRMSEを計算（L2正則化付き）"""
        w = np.abs(weights)
        w = w / (w.sum() + 1e-10)
        ensemble = w @ cv_arr  # (n_train,)
        fold_rmses = []
        for fi in pattern_fold_indices:
            _, te = folds[fi]
            fold_rmses.append(rmse(y_all[te], ensemble[te]))
        if not fold_rmses:
            return float("inf")
        # L2正則化: 重みの偏りにペナルティ（均等配分からの乖離を抑制）
        l2_penalty = lambda_l2 * float(np.sum((w - 1.0 / n_models) ** 2))
        return float(np.mean(fold_rmses)) + l2_penalty

    # Nelder-Mead最適化（複数初期点）
    best_rmse = np.inf
    best_weights = np.ones(n_models) / n_models

    for seed in range(n_restarts):
        w0 = np.random.RandomState(seed).dirichlet(np.ones(n_models))
        res = minimize(pattern_rmse, w0, method="Nelder-Mead",
                       options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6})
        if res.fun < best_rmse:
            best_rmse = res.fun
            best_weights = np.abs(res.x) / (np.sum(np.abs(res.x)) + 1e-10)

    # 最適化された重みでアンサンブル
    cv_ensemble = best_weights @ cv_arr
    test_ensemble = np.clip(best_weights @ test_arr, 0, 300)

    # パターンfold CV RMSE
    pattern_cv = best_rmse

    # 全13種 CV RMSE
    full_cv = rmse(y_all, cv_ensemble)

    # 重みが大きいモデル一覧
    top_models = sorted(
        [(models[i], best_weights[i]) for i in range(n_models)],
        key=lambda x: -x[1]
    )

    return test_ensemble, pattern_cv, full_cv, best_weights, top_models


# ============================================================
# メイン
# ============================================================
def main():
    t0 = time.time()

    # データ読み込み
    df_train = load_train(DATA_DIR)
    df_test = load_test(DATA_DIR)
    sc = get_spectral_columns(df_train)
    X_all = df_train[sc].values
    X_test = df_test[sc].values
    y_all = df_train["含水率"].values
    g_all = df_train["樹種"].values
    test_ids = df_test["sample number"].values

    print(f"Train: {X_all.shape}, Test: {X_test.shape}")

    # ============================================================
    # Step 1: 全モデルの13-fold LOSO-CV & test予測（1回だけ実行）
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 1: 標準13-fold LOSO-CV（12種学習→1種推定）")
    print("=" * 70)
    t1 = time.time()

    models, cv_preds, test_preds, folds, species_list = run_full_loso_cv(
        X_all, y_all, g_all, X_test
    )
    print(f"\nStep 1完了: {len(models)}モデル, {time.time()-t1:.0f}s")

    # Pattern G用: 全13種Top-12均等アンサンブルのfold別RMSE
    full_top12_idx = sorted(
        range(len(models)),
        key=lambda i: rmse(y_all, cv_preds[i])
    )[:12]
    full_ensemble_cv = np.mean([cv_preds[i] for i in full_top12_idx], axis=0)
    fold_rmses = {}
    for fi, (_, te) in enumerate(folds):
        sp = species_list[fi]
        fold_rmses[sp] = rmse(y_all[te], full_ensemble_cv[te])

    print("\nfold別RMSE（全13種Top-12アンサンブル）:")
    for sp in species_list:
        marker = " ※" if sp == "ベイスギ" else ""
        print(f"  {sp}: {fold_rmses[sp]:.2f}{marker}")

    # ============================================================
    # Step 2: パターンD-I 樹種セット計算
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 2: パターン樹種セット計算")
    print("=" * 70)

    pattern_e = select_pattern_e_sample_count(df_train, min_samples=90)
    q25, q75 = np.percentile(y_all, [25, 75])
    pattern_f = select_pattern_f_moisture_coverage(df_train, target_min=q25, target_max=q75, coverage_threshold=1.0)
    pattern_g = select_pattern_g_stable_folds(fold_rmses, top_k=6)
    pattern_h = select_pattern_h_wood_type(TEST_SPECIES)
    pattern_k = select_pattern_k_spectral_homogeneity(df_train, X_all, top_k=6)
    pattern_l = select_pattern_l_wasserstein(df_train, target_distribution=y_all, top_k=6)
    pattern_m = select_pattern_m_correlation_stability(df_train, X_all, top_k=6)

    all_patterns = {
        "A": ("生スペクトル類似", PATTERN_A),
        "B": ("SNV前処理後類似", PATTERN_B),
        "C": ("含水率類似", PATTERN_C),
        "E": ("サンプル数≥90", pattern_e),
        "F": ("含水率レンジ", pattern_f),
        "G": ("RMSE安定top6", pattern_g),
        "H": ("針葉樹_広葉樹", pattern_h),
        "K": ("スペクトル同質性", pattern_k),
        "L": ("Wasserstein距離", pattern_l),
        "M": ("相関安定性", pattern_m),
    }

    # 完全重複チェック
    seen = {}
    to_remove = []
    for key, (desc, sp_set) in all_patterns.items():
        frozen = frozenset(sp_set)
        if frozen in seen:
            print(f"  ⚠ Pattern {key} は Pattern {seen[frozen]} と完全重複 → 除外")
            to_remove.append(key)
        else:
            seen[frozen] = key
    for key in to_remove:
        del all_patterns[key]

    for key, (desc, sp_set) in all_patterns.items():
        print(f"  {key} ({desc}): {len(sp_set)}種 — {sorted(sp_set)}")

    # ============================================================
    # Step 3: パターン別Nelder-Mead重み最適化 & サブミッション生成
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 3: パターン別Nelder-Mead重み最適化 & サブミッション生成")
    print("=" * 70)

    results = []
    for key, (desc, sp_set) in all_patterns.items():
        print(f"\n--- Pattern {key} ({desc}) ---", flush=True)
        t2 = time.time()

        test_ensemble, pattern_cv, full_cv, weights, top_models = optimize_weights_for_pattern(
            models, cv_preds, test_preds, y_all, folds, species_list,
            sp_set, n_restarts=40,
        )

        # サブミッションCSV保存
        safe_desc = desc.replace("/", "_")
        filename = f"submission_issue68_{key}_{safe_desc}.csv"
        out_path = OUT_DIR / filename
        sub_df = pd.DataFrame({0: test_ids.astype(int), 1: test_ensemble})
        sub_df.to_csv(out_path, index=False, header=False)

        # 有効モデル（重み >= 1%）
        active_models = [(name, w) for name, w in top_models if w >= 0.01]

        print(f"  パターンCV RMSE ({len(sp_set)}種): {pattern_cv:.2f}")
        print(f"  全13種CV RMSE: {full_cv:.2f}")
        print(f"  test予測: mean={test_ensemble.mean():.1f}, "
              f"range=[{test_ensemble.min():.1f}, {test_ensemble.max():.1f}]")
        print(f"  有効モデル数: {len(active_models)}/{len(models)}")
        for name, w in active_models[:5]:
            print(f"    {name}: {w:.3f}")
        print(f"  最適化時間: {time.time()-t2:.0f}s")
        print(f"  保存: {filename}")

        results.append({
            "パターン": key,
            "説明": desc,
            "評価樹種数": len(sp_set),
            "評価樹種": ", ".join(sorted(sp_set)),
            "パターンCV": pattern_cv,
            "全13種CV": full_cv,
            "test平均": test_ensemble.mean(),
            "test_min": test_ensemble.min(),
            "test_max": test_ensemble.max(),
            "有効モデル数": len(active_models),
            "ファイル": filename,
        })

    # ============================================================
    # 結果一覧
    # ============================================================
    print("\n" + "=" * 70)
    print("結果一覧")
    print("=" * 70)

    df_results = pd.DataFrame(results)
    print(f"\n{'パターン':<4} {'説明':<18} {'評価種数':>5} {'パターンCV':>9} {'全13種CV':>8} {'test平均':>7} {'有効モデル':>6}")
    print("-" * 66)
    for _, row in df_results.iterrows():
        print(
            f"{row['パターン']:<4} {row['説明']:<16} {row['評価樹種数']:>5} "
            f"{row['パターンCV']:>9.2f} {row['全13種CV']:>8.2f} "
            f"{row['test平均']:>7.1f} {row['有効モデル数']:>6}"
        )

    # CSV保存
    summary_path = OUT_DIR / "issue68_pattern_submission_summary.csv"
    df_results.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\nサマリ保存: {summary_path}")
    print(f"合計時間: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
