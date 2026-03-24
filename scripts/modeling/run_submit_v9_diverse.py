"""v9: 多様な前処理 × PLS 均等平均アンサンブル（LB最適化版）

戦略: 12種の前処理 × PLS(nc=3,4) × sqrt変換 = 多様なPLSモデル群を均等平均
GBR/Huber系を排除し、PLS系のみで最大の多様性を確保
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import warnings
import time

warnings.filterwarnings("ignore")

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut
from src.eda.data_loader import load_train, load_test, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue20_savgol import apply_savgol
from src.preprocessing.issue19_msc import compute_msc_reference, apply_msc
from src.preprocessing.issue62_additional_preprocessing import apply_asls, apply_piecewise_msc

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    t0 = time.time()
    df_train = load_train(DATA_DIR)
    df_test = load_test(DATA_DIR)
    spec_cols = get_spectral_columns(df_train)
    X = df_train[spec_cols].values
    y = df_train["含水率"].values
    g = df_train["樹種"].values
    X_test = df_test[spec_cols].values
    test_ids = df_test["sample number"].values

    logo = LeaveOneGroupOut()
    folds = list(logo.split(X, y, g))
    sp = [np.unique(g[te])[0] for _, te in folds]

    models = []
    cv_preds = []
    test_preds = []

    def add_model(name, pp_func, nc, tf):
        cv = np.zeros_like(y)
        for fi, (tr, te) in enumerate(folds):
            try:
                Xtr, Xte2 = pp_func(X[tr], X[te], g[tr])
                ytr = np.sqrt(y[tr]) if tf == "sqrt" else y[tr]
                n_comp = min(nc, Xtr.shape[1] - 1)
                if n_comp < 1:
                    n_comp = 1
                pls = PLSRegression(n_components=n_comp)
                pls.fit(Xtr, ytr)
                p = pls.predict(Xte2).ravel()
                if tf == "sqrt":
                    p = np.clip(p, 0, None) ** 2
                cv[te] = p
            except Exception as e:
                cv[te] = y[tr].mean()
                print(f"    WARN {name} fold {fi}: {e}")

        cv_rmse = rmse(y, cv)

        # テスト予測
        try:
            Xtr, Xte2 = pp_func(X, X_test, g)
            ytr = np.sqrt(y) if tf == "sqrt" else y
            n_comp = min(nc, Xtr.shape[1] - 1)
            if n_comp < 1:
                n_comp = 1
            pls = PLSRegression(n_components=n_comp)
            pls.fit(Xtr, ytr)
            p = pls.predict(Xte2).ravel()
            if tf == "sqrt":
                p = np.clip(p, 0, None) ** 2
        except Exception:
            p = np.full(len(X_test), y.mean())

        models.append(name)
        cv_preds.append(cv)
        test_preds.append(p)
        print(f"  {name}: CV={cv_rmse:.2f}, test_mean={p.mean():.1f} ({time.time()-t0:.0f}s)")

    # 前処理関数群
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

    print("=" * 60)
    print("v9: 多様PLS アンサンブル構築")
    print("=" * 60)

    configs = [
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

    for pp_name, pp_func, ncs, tfs in configs:
        for nc in ncs:
            for tf in tfs:
                add_model(f"{pp_name}_PLS{nc}_{tf}", pp_func, nc, tf)

    n = len(models)
    print(f"\n合計 {n} モデル")

    # ソート
    ranked = sorted(range(n), key=lambda i: rmse(y, cv_preds[i]))
    print("\n=== CV RMSEランキング Top-15 ===")
    for rank, i in enumerate(ranked[:15]):
        print(f"  {rank+1}. {models[i]}: {rmse(y, cv_preds[i]):.2f}")

    # アンサンブル戦略
    print("\n=== アンサンブル戦略 ===")

    strategies = {}

    # 全モデル均等
    all_cv = np.mean(cv_preds, axis=0)
    all_test = np.clip(np.mean(test_preds, axis=0), 0, 300)
    strategies["all_equal"] = (all_cv, all_test)
    print(f"全{n}モデル均等: CV={rmse(y, all_cv):.2f}")

    # Top-K均等
    for k in [5, 8, 10, 12, 15]:
        if k > n:
            break
        idx = ranked[:k]
        cv_avg = np.mean([cv_preds[i] for i in idx], axis=0)
        te_avg = np.clip(np.mean([test_preds[i] for i in idx], axis=0), 0, 300)
        strategies[f"top{k}"] = (cv_avg, te_avg)
        print(f"Top-{k}均等: CV={rmse(y, cv_avg):.2f}")

    # 前処理多様性を重視（各前処理から1つずつ）
    diverse_idx = []
    used_pp = set()
    for i in ranked:
        pp = models[i].rsplit("_PLS", 1)[0]
        if pp not in used_pp:
            diverse_idx.append(i)
            used_pp.add(pp)
        if len(diverse_idx) >= 12:
            break
    cv_diverse = np.mean([cv_preds[i] for i in diverse_idx], axis=0)
    te_diverse = np.clip(
        np.mean([test_preds[i] for i in diverse_idx], axis=0), 0, 300
    )
    strategies["diverse"] = (cv_diverse, te_diverse)
    print(f"多様性重視{len(diverse_idx)}モデル: CV={rmse(y, cv_diverse):.2f}")
    for i in diverse_idx:
        print(f"    {models[i]}: CV={rmse(y, cv_preds[i]):.2f}")

    # EPO系のみ（樹種効果除去に特化）
    epo_idx = [i for i in range(n) if "EPO" in models[i]]
    if epo_idx:
        cv_epo = np.mean([cv_preds[i] for i in epo_idx], axis=0)
        te_epo = np.clip(np.mean([test_preds[i] for i in epo_idx], axis=0), 0, 300)
        strategies["epo_only"] = (cv_epo, te_epo)
        print(f"EPO系{len(epo_idx)}モデル: CV={rmse(y, cv_epo):.2f}")

    # ベストランキング
    print("\n=== 戦略ランキング ===")
    strat_ranked = sorted(strategies.items(), key=lambda x: rmse(y, x[1][0]))
    for rank, (name, (cv, te)) in enumerate(strat_ranked):
        print(f"  {rank+1}. {name}: CV={rmse(y, cv):.2f}, test_mean={te.mean():.1f}")

    # 提出ファイル（ベスト5戦略）
    print("\n=== 提出ファイル ===")
    for name, (cv, te) in strat_ranked[:5]:
        path = OUT_DIR / f"submission_v9_{name}.csv"
        sub = pd.DataFrame({0: test_ids.astype(int), 1: te})
        sub.to_csv(path, index=False, header=False)
        print(f"  {path}: CV={rmse(y, cv):.2f}, mean={te.mean():.1f}, range=[{te.min():.1f},{te.max():.1f}]")

    # fold別RMSE
    best_name, (best_cv, best_te) = strat_ranked[0]
    print(f"\n=== {best_name} fold別RMSE ===")
    for fi, (_, te) in enumerate(folds):
        marker = " ※外挿" if sp[fi] == "ベイスギ" else ""
        print(f"  {sp[fi]}: {rmse(y[te], best_cv[te]):.2f}{marker}")
    nbs = [rmse(y[te], best_cv[te]) for fi, (_, te) in enumerate(folds) if sp[fi] != "ベイスギ"]
    print(f"  除ベイスギ: {np.mean(nbs):.2f}")

    print(f"\nTotal: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
