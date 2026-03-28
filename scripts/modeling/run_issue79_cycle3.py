"""Issue #79: サイクル3 - PLS成分数最適化+バイアス補正+成分数多様性

チェリー分析の知見:
- PLS成分3にバイアスが存在（cherry mean=-2.78, train mean=0）
- 低含水率帯で系統的に過大予測
- バイアス成分がRMSEの73%を占める

仮説:
A. fold内CVでPLS成分数を動的に最適化
B. テストデータのPLSスコア分布から予測バイアスを推定・補正
C. PLS成分数(2,3,4,5)を混ぜたアンサンブル
D. 予測値のスケーリング補正（テスト分布ベース）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import time
from itertools import combinations
from scipy.optimize import minimize

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut, KFold
from sklearn.linear_model import Ridge

from src.eda.data_loader import load_train, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue20_savgol import apply_savgol
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue19_msc import compute_msc_reference
from src.preprocessing.issue62_additional_preprocessing import apply_asls, apply_piecewise_msc
from src.modeling.issue65_feature_selection import sipls_select, ipls_select

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def preprocess(X_tr_raw, X_te_raw, groups_train, pp):
    """前処理を適用"""
    if pp == "SNV":
        return apply_snv(X_tr_raw), apply_snv(X_te_raw)
    elif pp == "EPO(1)":
        P = compute_epo_projection(X_tr_raw, groups_train, n_components=1)
        return apply_epo(X_tr_raw, P), apply_epo(X_te_raw, P)
    elif pp == "SNV+AsLS(1e6)":
        return apply_asls(apply_snv(X_tr_raw), lam=1e6), apply_asls(apply_snv(X_te_raw), lam=1e6)
    elif pp == "PiecewiseMSC(seg=3)":
        ref = compute_msc_reference(X_tr_raw)
        return apply_piecewise_msc(X_tr_raw, ref, 3), apply_piecewise_msc(X_te_raw, ref, 3)
    elif pp == "SG2d+EPO(1)":
        X_sg = apply_savgol(X_tr_raw, deriv=2, window_length=7)
        X_sg_te = apply_savgol(X_te_raw, deriv=2, window_length=7)
        P = compute_epo_projection(X_sg, groups_train, n_components=1)
        return apply_epo(X_sg, P), apply_epo(X_sg_te, P)
    else:
        return X_tr_raw.copy(), X_te_raw.copy()


def feature_select(X_tr, X_te, y_train, fs):
    """特徴量選択"""
    if not fs:
        return X_tr, X_te
    if fs.startswith("siPLS"):
        parts = fs.replace("siPLS(", "").rstrip(")").split(",")
        ni, nc = int(parts[0]), int(parts[1])
        X_tr, X_te, _ = sipls_select(X_tr, y_train, X_te,
                                      n_intervals=ni, n_components=3, n_combine=nc)
    elif fs.startswith("iPLS"):
        ni = int(fs.replace("iPLS(", "").rstrip(")"))
        X_tr, X_te, _ = ipls_select(X_tr, y_train, X_te,
                                     n_intervals=ni, n_components=3, n_best=1)
    return X_tr, X_te


def predict_pls(X_tr, X_te, y_train, nc, tf):
    """PLS予測"""
    nc = min(nc, X_tr.shape[1] - 1)
    nc = max(1, nc)

    if tf == "sqrt":
        y_fit = np.sqrt(y_train)
    else:
        y_fit = y_train.copy()

    pls = PLSRegression(n_components=nc)
    pls.fit(X_tr, y_fit)
    pred = pls.predict(X_te).ravel()

    if tf == "sqrt":
        pred = np.clip(pred, 0, None) ** 2

    return pred


def predict_fold_adaptive_nc(X_tr_raw, X_te_raw, y_train, groups_train, cfg):
    """仮説A: fold内CVでPLS成分数を動的に最適化"""
    pp = cfg["pp"]
    fs = cfg.get("fs")
    tf = cfg["tf"]

    X_tr, X_te = preprocess(X_tr_raw, X_te_raw, groups_train, pp)
    X_tr, X_te = feature_select(X_tr, X_te, y_train, fs)

    # 内側CVでPLS成分数を選択（KFold 5分割）
    best_nc = 4
    best_cv_rmse = np.inf
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for nc_try in [2, 3, 4, 5, 6]:
        if nc_try >= X_tr.shape[1]:
            continue
        cv_rmses = []
        for tr_idx, val_idx in kf.split(X_tr):
            try:
                pred_val = predict_pls(X_tr[tr_idx], X_tr[val_idx],
                                       y_train[tr_idx], nc_try, tf)
                cv_rmses.append(rmse(y_train[val_idx], pred_val))
            except Exception:
                cv_rmses.append(999.0)
        mean_cv = np.mean(cv_rmses)
        if mean_cv < best_cv_rmse:
            best_cv_rmse = mean_cv
            best_nc = nc_try

    return predict_pls(X_tr, X_te, y_train, best_nc, tf)


def predict_fold_bias_corrected(X_tr_raw, X_te_raw, y_train, groups_train, cfg):
    """仮説B: テストPLSスコアからバイアス推定・補正"""
    pp = cfg["pp"]
    fs = cfg.get("fs")
    tf = cfg["tf"]
    nc = cfg.get("nc", 4)

    X_tr, X_te = preprocess(X_tr_raw, X_te_raw, groups_train, pp)
    X_tr, X_te = feature_select(X_tr, X_te, y_train, fs)

    nc = max(1, min(nc, X_tr.shape[1] - 1))
    if tf == "sqrt":
        y_fit = np.sqrt(y_train)
    else:
        y_fit = y_train.copy()

    pls = PLSRegression(n_components=nc)
    pls.fit(X_tr, y_fit)

    # 標準予測
    pred_standard = pls.predict(X_te).ravel()

    # PLSスコア空間でのバイアス推定
    T_train = pls.transform(X_tr)
    T_test = pls.transform(X_te)
    score_offset = T_test.mean(axis=0) - T_train.mean(axis=0)

    # スコアオフセットをY空間に変換してバイアス補正
    offset_y = (score_offset @ pls.y_loadings_.T).ravel()
    if hasattr(pls, '_y_std'):
        offset_y = offset_y * pls._y_std.ravel()
    pred_corrected = pred_standard - offset_y

    if tf == "sqrt":
        pred_corrected = np.clip(pred_corrected, 0, None) ** 2

    return pred_corrected


def predict_fold_standard(X_tr_raw, X_te_raw, y_train, groups_train, cfg):
    """標準的なpredict_fold"""
    pp = cfg["pp"]
    fs = cfg.get("fs")
    tf = cfg["tf"]
    nc = cfg.get("nc", 4)

    X_tr, X_te = preprocess(X_tr_raw, X_te_raw, groups_train, pp)
    X_tr, X_te = feature_select(X_tr, X_te, y_train, fs)
    return predict_pls(X_tr, X_te, y_train, nc, tf)


def predict_fold_scale_corrected(X_tr_raw, X_te_raw, y_train, groups_train, cfg):
    """仮説D: 予測値のスケーリング補正
    テストデータの予測値分布をtrainのfold CV予測分布に近づける"""
    pp = cfg["pp"]
    fs = cfg.get("fs")
    tf = cfg["tf"]
    nc = cfg.get("nc", 4)

    X_tr, X_te = preprocess(X_tr_raw, X_te_raw, groups_train, pp)
    X_tr, X_te = feature_select(X_tr, X_te, y_train, fs)

    # まず標準予測
    pred = predict_pls(X_tr, X_te, y_train, nc, tf)

    # train内CVで予測を生成し、残差の統計を計算
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_preds = np.zeros(len(y_train))
    for tr_idx, val_idx in kf.split(X_tr):
        try:
            train_preds[val_idx] = predict_pls(X_tr[tr_idx], X_tr[val_idx],
                                                y_train[tr_idx], nc, tf)
        except Exception:
            train_preds[val_idx] = y_train.mean()

    # train CV予測のバイアスを推定
    train_bias = np.mean(train_preds - y_train)

    # テスト予測をバイアス補正
    pred_corrected = pred - train_bias

    return np.clip(pred_corrected, 0, 300)


def main():
    t0 = time.time()
    df = load_train(DATA_DIR)
    sc = get_spectral_columns(df)
    X_raw = df[sc].values
    y = df["含水率"].values
    groups = df["樹種"].values

    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_raw, y, groups))
    species_names = [np.unique(groups[te])[0] for _, te in folds]

    print("=" * 70)
    print("Issue #79: サイクル3 - PLS成分数最適化+バイアス補正+成分数多様性")
    print(f"  ベースライン: 14.963 (BestCombo5-Weighted)")
    print("=" * 70)

    # モデル定義
    models = [
        # 既存5モデル (M1-M5)
        {"name": "M1:EPO(1)+PLS(4)+sqrt", "pp": "EPO(1)", "nc": 4, "tf": "sqrt",
         "func": predict_fold_standard},
        {"name": "M2:SG2d+EPO(1)+PLS(3)+raw", "pp": "SG2d+EPO(1)", "nc": 3, "tf": "raw",
         "func": predict_fold_standard},
        {"name": "M3:SNV+iPLS(50)+PLS(4)+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt",
         "fs": "iPLS(50)", "func": predict_fold_standard},
        {"name": "M4:PMSC+siPLS(30,3)+PLS(4)+sqrt", "pp": "PiecewiseMSC(seg=3)", "nc": 4,
         "tf": "sqrt", "fs": "siPLS(30,3)", "func": predict_fold_standard},
        {"name": "M5:SNV+AsLS+siPLS(30,3)+PLS(5)+sqrt", "pp": "SNV+AsLS(1e6)", "nc": 5,
         "tf": "sqrt", "fs": "siPLS(30,3)", "func": predict_fold_standard},

        # 仮説A: 成分数自動最適化
        {"name": "A1:EPO(1)+autoNC+sqrt", "pp": "EPO(1)", "tf": "sqrt",
         "func": predict_fold_adaptive_nc},
        {"name": "A2:SNV+iPLS(50)+autoNC+sqrt", "pp": "SNV", "tf": "sqrt",
         "fs": "iPLS(50)", "func": predict_fold_adaptive_nc},
        {"name": "A3:SNV+autoNC+sqrt", "pp": "SNV", "tf": "sqrt",
         "func": predict_fold_adaptive_nc},

        # 仮説B: バイアス補正
        {"name": "B1:EPO(1)+PLS(4)+sqrt+biasCorr", "pp": "EPO(1)", "nc": 4, "tf": "sqrt",
         "func": predict_fold_bias_corrected},
        {"name": "B2:SNV+iPLS(50)+PLS(4)+sqrt+biasCorr", "pp": "SNV", "nc": 4, "tf": "sqrt",
         "fs": "iPLS(50)", "func": predict_fold_bias_corrected},
        {"name": "B3:SNV+PLS(4)+sqrt+biasCorr", "pp": "SNV", "nc": 4, "tf": "sqrt",
         "func": predict_fold_bias_corrected},

        # 仮説C: 成分数多様性
        {"name": "C1:EPO(1)+PLS(2)+sqrt", "pp": "EPO(1)", "nc": 2, "tf": "sqrt",
         "func": predict_fold_standard},
        {"name": "C2:EPO(1)+PLS(3)+sqrt", "pp": "EPO(1)", "nc": 3, "tf": "sqrt",
         "func": predict_fold_standard},
        {"name": "C3:EPO(1)+PLS(5)+sqrt", "pp": "EPO(1)", "nc": 5, "tf": "sqrt",
         "func": predict_fold_standard},
        {"name": "C4:SNV+iPLS(50)+PLS(2)+sqrt", "pp": "SNV", "nc": 2, "tf": "sqrt",
         "fs": "iPLS(50)", "func": predict_fold_standard},
        {"name": "C5:SNV+iPLS(50)+PLS(5)+sqrt", "pp": "SNV", "nc": 5, "tf": "sqrt",
         "fs": "iPLS(50)", "func": predict_fold_standard},

        # 仮説D: スケーリング補正
        {"name": "D1:EPO(1)+PLS(4)+sqrt+scalCorr", "pp": "EPO(1)", "nc": 4, "tf": "sqrt",
         "func": predict_fold_scale_corrected},
        {"name": "D2:SNV+iPLS(50)+PLS(4)+sqrt+scalCorr", "pp": "SNV", "nc": 4, "tf": "sqrt",
         "fs": "iPLS(50)", "func": predict_fold_scale_corrected},
    ]

    n_models = len(models)
    all_preds = [[] for _ in range(n_models)]

    print(f"\n--- 個別モデル評価 ({n_models}モデル) ---\n", flush=True)

    for m_idx, cfg in enumerate(models):
        t1 = time.time()
        fold_rmses = []
        for f_idx, (train_idx, test_idx) in enumerate(folds):
            try:
                pred = cfg["func"](
                    X_raw[train_idx], X_raw[test_idx],
                    y[train_idx], groups[train_idx], cfg
                )
                all_preds[m_idx].append(pred)
                fold_rmses.append(rmse(y[test_idx], pred))
            except Exception as e:
                fallback = np.full(len(test_idx), y[train_idx].mean())
                all_preds[m_idx].append(fallback)
                fold_rmses.append(999.0)
                print(f"  ERROR {cfg['name']} fold {f_idx}: {e}")

        mean_r = np.mean(fold_rmses)
        print(f"  [{m_idx+1:>2}/{n_models}] {cfg['name']}: {mean_r:.2f} ± {np.std(fold_rmses):.2f} ({time.time()-t1:.1f}s)", flush=True)

    # アンサンブル最適化
    print("\n--- アンサンブル最適化 ---\n", flush=True)

    def eval_ens(indices, weights=None):
        fold_rmses = []
        for f_idx, (_, test_idx) in enumerate(folds):
            preds = [all_preds[m][f_idx] for m in indices]
            if weights is not None:
                w = np.array(weights)
                w = w / w.sum()
                ens = sum(wi * p for wi, p in zip(w, preds))
            else:
                ens = np.mean(preds, axis=0)
            ens = np.clip(ens, 0, 300)
            fold_rmses.append(rmse(y[test_idx], ens))
        return np.mean(fold_rmses), fold_rmses

    def opt_w(indices, n_restarts=20):
        n = len(indices)
        def obj(w):
            w_n = np.abs(w) / np.sum(np.abs(w))
            r, _ = eval_ens(indices, w_n)
            return r
        best_r, best_w = np.inf, np.ones(n) / n
        for s in range(n_restarts):
            w0 = np.random.RandomState(s).dirichlet(np.ones(n))
            res = minimize(obj, w0, method="Nelder-Mead", options={"maxiter": 3000})
            if res.fun < best_r:
                best_r = res.fun
                best_w = np.abs(res.x) / np.sum(np.abs(res.x))
        return best_w, best_r

    # 個別ランキング
    indiv = []
    for m_idx in range(n_models):
        r, _ = eval_ens([m_idx])
        indiv.append((r, m_idx))
    indiv.sort()

    print("個別モデル (fold-RMSE平均):")
    for r, i in indiv:
        print(f"  {r:.2f} | {models[i]['name']}")

    # ベースライン再現
    base5 = list(range(5))
    w5, r5 = opt_w(base5)
    print(f"\nBaseline(M1-5)-Weighted: {r5:.4f}")

    # Top-12から組み合わせ探索
    top12 = [i for _, i in indiv[:12]]
    results = []
    results.append(("Baseline(M1-5)", r5, base5, w5))

    for k in [3, 4, 5, 6, 7, 8]:
        best_r, best_combo = np.inf, None
        for combo in combinations(top12, k):
            r, _ = eval_ens(list(combo))
            if r < best_r:
                best_r = r
                best_combo = list(combo)
        if best_combo:
            # Weighted
            w, r_w = opt_w(best_combo)
            results.append((f"BestCombo{k}-Weighted", r_w, best_combo, w))
            print(f"  BestCombo{k}-Weighted: {r_w:.4f} | {[models[i]['name'] for i in best_combo]}", flush=True)

    results.sort(key=lambda x: x[1])

    # ベスト表示
    best = results[0]
    print(f"\n{'='*70}")
    print(f"BEST: {best[1]:.4f} ({best[0]})")
    print(f"ベースライン: 14.963")
    print(f"改善幅: {14.963 - best[1]:+.4f}")
    print(f"{'='*70}")

    # fold詳細
    _, fold_rmses = eval_ens(best[2], best[3])
    print("\nfold詳細:")
    for sp, fr in zip(species_names, fold_rmses):
        tag = " ※参考" if sp == "ベイスギ" else ""
        print(f"  {sp}: {fr:.2f}{tag}")
    non_bs = [r for sp, r in zip(species_names, fold_rmses) if sp != "ベイスギ"]
    print(f"  ベイスギ除外平均: {np.mean(non_bs):.4f}")

    # 残差相関
    print("\n残差相関（既存M1-M5 vs 新モデル）:")
    residuals = np.zeros((n_models, len(y)))
    for m_idx in range(n_models):
        for f_idx, (_, test_idx) in enumerate(folds):
            residuals[m_idx, test_idx] = y[test_idx] - all_preds[m_idx][f_idx]
    corr = np.corrcoef(residuals)
    for i in range(5, n_models):
        corrs_with_base = [corr[i, j] for j in range(5)]
        print(f"  {models[i]['name']}: avg_corr_with_M1-5={np.mean(corrs_with_base):.3f}, min={min(corrs_with_base):.3f}")

    # CSV保存
    rows = []
    for name, r, combo, w in results:
        rows.append({"method": name, "rmse": r, "models": str([models[i]["name"] for i in combo])})
    pd.DataFrame(rows).to_csv(OUT_DIR / "issue79_cycle3_results.csv", index=False)
    print(f"\n結果保存: {OUT_DIR / 'issue79_cycle3_results.csv'}")
    print(f"総実行時間: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
