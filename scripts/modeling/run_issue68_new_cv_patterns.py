"""Issue #68: 新CVパターン D-I の樹種選定・CV RMSE評価

既存パターンA-Cに加え、データ駆動型の新パターンD-Iを計算し、
各パターンのCV RMSEとLBスコアの差分を比較する。

パターン:
  D: PCA空間マハラノビス距離が近い樹種
  E: サンプル数が極端に少ない樹種を除外
  F: 含水率レンジがtest全体をカバーする樹種
  G: fold RMSEのばらつきが小さい安定樹種
  H: 針葉樹/広葉樹分類でtest樹種に近いグループ
  I: MMD距離が小さい樹種
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
from src.preprocessing.issue19_msc import compute_msc_reference, apply_msc
from src.preprocessing.issue62_additional_preprocessing import apply_asls, apply_piecewise_msc
from src.analysis.issue68_cv_pattern_selection import (
    select_pattern_d_mahalanobis,
    select_pattern_e_sample_count,
    select_pattern_f_moisture_coverage,
    select_pattern_g_stable_folds,
    select_pattern_h_wood_type,
    select_pattern_i_mmd,
)

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 既存パターン A-C
# ============================================================
PATTERN_A = {"ヒノキ", "ナラ", "米ヒバ", "スプルース", "クリ", "ホワイトオーク"}
PATTERN_B = {"ナラ", "米ヒバ", "スプルース", "ベイマツ", "トチ", "クリ"}
PATTERN_C = {"ホワイトオーク", "スプルース", "チェリー", "ウォールナット"}

TEST_SPECIES = {"クスノキ", "ケヤキ", "スギ", "タモ", "チーク", "ヤマザクラ"}

LB_SCORES = {
    "best11": 21.56,
    "v9_top12": 17.14,
}


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ============================================================
# v9 前処理関数群
# ============================================================
def v9_pp_raw(Xtr, Xte, g_tr):
    return Xtr.copy(), Xte.copy()

def v9_pp_snv(Xtr, Xte, g_tr):
    return apply_snv(Xtr), apply_snv(Xte)

def v9_pp_epo1(Xtr, Xte, g_tr):
    P = compute_epo_projection(Xtr, g_tr, 1)
    return apply_epo(Xtr, P), apply_epo(Xte, P)

def v9_pp_epo2(Xtr, Xte, g_tr):
    P = compute_epo_projection(Xtr, g_tr, 2)
    return apply_epo(Xtr, P), apply_epo(Xte, P)

def v9_pp_snv_epo1(Xtr, Xte, g_tr):
    Xtr_s, Xte_s = apply_snv(Xtr), apply_snv(Xte)
    P = compute_epo_projection(Xtr_s, g_tr, 1)
    return apply_epo(Xtr_s, P), apply_epo(Xte_s, P)

def v9_pp_sg1d(Xtr, Xte, g_tr):
    return (
        apply_savgol(Xtr, deriv=1, window_length=11),
        apply_savgol(Xte, deriv=1, window_length=11),
    )

def v9_pp_sg2d_epo1(Xtr, Xte, g_tr):
    xs = apply_savgol(Xtr, deriv=2, window_length=7)
    xst = apply_savgol(Xte, deriv=2, window_length=7)
    P = compute_epo_projection(xs, g_tr, 1)
    return apply_epo(xs, P), apply_epo(xst, P)

def v9_pp_msc(Xtr, Xte, g_tr):
    ref = compute_msc_reference(Xtr)
    return apply_msc(Xtr, ref), apply_msc(Xte, ref)

def v9_pp_snv_sg1d(Xtr, Xte, g_tr):
    return (
        apply_savgol(apply_snv(Xtr), deriv=1, window_length=11),
        apply_savgol(apply_snv(Xte), deriv=1, window_length=11),
    )

def v9_pp_snv_asls(Xtr, Xte, g_tr):
    return (
        apply_asls(apply_snv(Xtr), lam=1e6),
        apply_asls(apply_snv(Xte), lam=1e6),
    )

def v9_pp_pmsc(Xtr, Xte, g_tr):
    ref = compute_msc_reference(Xtr)
    return (
        apply_piecewise_msc(Xtr, ref, 3),
        apply_piecewise_msc(Xte, ref, 3),
    )

def v9_pp_msc_epo1(Xtr, Xte, g_tr):
    ref = compute_msc_reference(Xtr)
    Xtr_m, Xte_m = apply_msc(Xtr, ref), apply_msc(Xte, ref)
    P = compute_epo_projection(Xtr_m, g_tr, 1)
    return apply_epo(Xtr_m, P), apply_epo(Xte_m, P)


# ============================================================
# fold別RMSE → パターン別CV RMSE
# ============================================================
def compute_fold_rmses(y, folds, species_list, ensemble_preds):
    fold_rmses = {}
    for fi, (_, te) in enumerate(folds):
        sp = species_list[fi]
        fold_rmses[sp] = rmse(y[te], ensemble_preds[te])
    return fold_rmses


def compute_pattern_cv(fold_rmses, pattern_species):
    vals = [fold_rmses[sp] for sp in pattern_species if sp in fold_rmses]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


# ============================================================
# v9パイプライン（Top-12均等平均）
# ============================================================
def run_v9_pipeline(X, y, g, folds, species_list):
    print("=" * 70)
    print("v9_top12 パイプライン")
    print("=" * 70)

    configs = [
        ("EPO1", v9_pp_epo1, [3, 4], ["sqrt"]),
        ("EPO2", v9_pp_epo2, [3, 4], ["sqrt"]),
        ("SNV", v9_pp_snv, [3, 4], ["sqrt"]),
        ("SNV_EPO1", v9_pp_snv_epo1, [3, 4], ["sqrt"]),
        ("SG2d_EPO1", v9_pp_sg2d_epo1, [3], ["raw", "sqrt"]),
        ("SG1d", v9_pp_sg1d, [3, 4], ["sqrt"]),
        ("MSC", v9_pp_msc, [3, 4], ["sqrt"]),
        ("SNV_SG1d", v9_pp_snv_sg1d, [3, 4], ["sqrt"]),
        ("SNV_AsLS", v9_pp_snv_asls, [3, 4], ["sqrt"]),
        ("PMSC", v9_pp_pmsc, [3, 4], ["sqrt"]),
        ("MSC_EPO1", v9_pp_msc_epo1, [3, 4], ["sqrt"]),
        ("Raw", v9_pp_raw, [3, 4], ["sqrt"]),
    ]

    models = []
    cv_preds = []

    for pp_name, pp_func, ncs, tfs in configs:
        for nc in ncs:
            for tf in tfs:
                name = f"{pp_name}_PLS{nc}_{tf}"
                cv = np.zeros_like(y)
                for fi, (tr, te) in enumerate(folds):
                    try:
                        Xtr, Xte2 = pp_func(X[tr], X[te], g[tr])
                        ytr = np.sqrt(y[tr]) if tf == "sqrt" else y[tr]
                        n_comp = max(1, min(nc, Xtr.shape[1] - 1))
                        pls = PLSRegression(n_components=n_comp)
                        pls.fit(Xtr, ytr)
                        p = pls.predict(Xte2).ravel()
                        if tf == "sqrt":
                            p = np.clip(p, 0, None) ** 2
                        cv[te] = p
                    except Exception as e:
                        cv[te] = y[tr].mean()
                        print(f"  WARN {name} fold {fi}: {e}")
                models.append(name)
                cv_preds.append(cv.copy())
                print(f"  {name}: CV={rmse(y, cv):.2f}", flush=True)

    ranked = sorted(range(len(models)), key=lambda i: rmse(y, cv_preds[i]))
    top_idx = ranked[:12]
    top12_cv = np.mean([cv_preds[i] for i in top_idx], axis=0)
    print(f"\nTop-12均等: CV={rmse(y, top12_cv):.2f}")

    fold_rmses = compute_fold_rmses(y, folds, species_list, top12_cv)
    return fold_rmses


# ============================================================
# 新パターン D-I の自動計算
# ============================================================
def compute_new_patterns(df_train, X_train, X_test, fold_rmses):
    """パターンD-Iの樹種セットを計算して返す"""
    patterns = {}

    # Pattern D: PCA空間マハラノビス距離
    patterns["D (PCAマハラノビス距離)"] = select_pattern_d_mahalanobis(
        df_train, X_train, X_test, n_components=10, top_k=6,
    )

    # Pattern E: サンプル数フィルタ (最小50件)
    patterns["E (サンプル数≥90)"] = select_pattern_e_sample_count(
        df_train, min_samples=90,
    )

    # Pattern F: 含水率レンジカバレッジ
    y_train = df_train["含水率"].values
    q25, q75 = np.percentile(y_train, [25, 75])
    patterns["F (含水率レンジ)"] = select_pattern_f_moisture_coverage(
        df_train, target_min=q25, target_max=q75, coverage_threshold=1.0,
    )

    # Pattern G: fold RMSE安定樹種 (top-6)
    patterns["G (RMSE安定 top6)"] = select_pattern_g_stable_folds(
        fold_rmses, top_k=6,
    )

    # Pattern H: 針葉樹/広葉樹分類
    patterns["H (針葉樹/広葉樹)"] = select_pattern_h_wood_type(TEST_SPECIES)

    # Pattern I: MMD距離
    patterns["I (MMD距離)"] = select_pattern_i_mmd(
        df_train, X_train, X_test, top_k=6,
    )

    return patterns


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
    X_test = df_test[sc].values
    y = df_train["含水率"].values
    g = df_train["樹種"].values

    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_train, y, g))
    species_list = [np.unique(g[te])[0] for _, te in folds]

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train樹種: {species_list}")
    print(f"Test樹種: {sorted(TEST_SPECIES)}")

    # ---- v9パイプラインでfold別RMSE算出 ----
    v9_fold_rmses = run_v9_pipeline(X_train, y, g, folds, species_list)

    # ---- fold別RMSE一覧 ----
    print("\n" + "=" * 70)
    print("fold別RMSE一覧")
    print("=" * 70)
    for sp in species_list:
        marker = " ※" if sp == "ベイスギ" else ""
        print(f"  {sp}: {v9_fold_rmses[sp]:.2f}{marker}")

    # ---- 新パターン D-I の樹種セット計算 ----
    print("\n" + "=" * 70)
    print("新パターン D-I の樹種選定")
    print("=" * 70)

    new_patterns = compute_new_patterns(df_train, X_train, X_test, v9_fold_rmses)

    # 全パターン統合
    all_patterns = {
        "A (生スペクトル類似)": PATTERN_A,
        "B (SNV前処理後類似)": PATTERN_B,
        "C (含水率類似)": PATTERN_C,
    }
    all_patterns.update(new_patterns)

    # 各パターンの樹種セット表示
    for name, species_set in all_patterns.items():
        print(f"\n  Pattern {name}:")
        print(f"    樹種数: {len(species_set)}")
        print(f"    樹種: {sorted(species_set)}")

    # ---- パターン別CV RMSE比較 ----
    print("\n" + "=" * 70)
    print("パターン別 CV RMSE 比較 (v9_top12, LB=17.14)")
    print("=" * 70)

    lb = LB_SCORES["v9_top12"]
    results = []
    for name, species_set in all_patterns.items():
        cv = compute_pattern_cv(v9_fold_rmses, species_set)
        diff = cv - lb
        results.append({
            "パターン": name,
            "樹種数": len(species_set),
            "CV RMSE": cv,
            "LB": lb,
            "CV-LB差": diff,
            "|CV-LB|": abs(diff),
        })

    # 全13種とベイスギ除外も追加
    all_sp = set(v9_fold_rmses.keys())
    for name, sp_set in [("全13種", all_sp), ("除ベイスギ", all_sp - {"ベイスギ"})]:
        cv = compute_pattern_cv(v9_fold_rmses, sp_set)
        results.append({
            "パターン": name,
            "樹種数": len(sp_set),
            "CV RMSE": cv,
            "LB": lb,
            "CV-LB差": cv - lb,
            "|CV-LB|": abs(cv - lb),
        })

    df_results = pd.DataFrame(results).sort_values("|CV-LB|")

    print(f"\n{'パターン':<24} {'樹種数':>4} {'CV RMSE':>8} {'LB':>6} {'CV-LB差':>8} {'|差|':>6}")
    print("-" * 62)
    for _, row in df_results.iterrows():
        marker = " ★" if row["|CV-LB|"] == df_results["|CV-LB|"].min() else ""
        print(
            f"{row['パターン']:<22} {row['樹種数']:>4} "
            f"{row['CV RMSE']:>8.2f} {row['LB']:>6.2f} "
            f"{row['CV-LB差']:>+8.2f} {row['|CV-LB|']:>6.2f}{marker}"
        )

    # CSV出力
    out_path = OUT_DIR / "issue68_new_cv_patterns_results.csv"
    df_results.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n結果保存: {out_path}")
    print(f"合計時間: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
