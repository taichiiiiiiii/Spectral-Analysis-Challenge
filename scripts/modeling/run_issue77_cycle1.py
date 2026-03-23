"""Issue #77: サイクル1 - 4仮説統合検証

仮説1: CORAL+PLS（多様性向上）
仮説2: Box-Cox変換（高含水率帯バイアス削減）
仮説3: スタッキングメタラーナー（柔軟なモデル結合）
仮説4: LW-PLS（局所ドメイン適応）
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
from scipy.stats import boxcox
from scipy.special import inv_boxcox

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import LeaveOneGroupOut, KFold

from src.eda.data_loader import load_train, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue20_savgol import apply_savgol
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue19_msc import compute_msc_reference
from src.preprocessing.issue62_additional_preprocessing import apply_asls, apply_piecewise_msc
from src.modeling.issue65_feature_selection import sipls_select, ipls_select
from src.modeling.issue33_lwpls import lwpls_predict

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# ユーティリティ関数
# ==============================================================================

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def coral_transform(X_source, X_target):
    """CORAL: ソースの共分散をターゲットに合わせる"""
    Cs = np.cov(X_source.T) + np.eye(X_source.shape[1]) * 1e-6
    Ct = np.cov(X_target.T) + np.eye(X_target.shape[1]) * 1e-6

    Ds, Vs = np.linalg.eigh(Cs)
    Ds = np.maximum(Ds, 1e-8)
    Cs_inv_sqrt = Vs @ np.diag(1.0 / np.sqrt(Ds)) @ Vs.T

    Dt, Vt = np.linalg.eigh(Ct)
    Dt = np.maximum(Dt, 1e-8)
    Ct_sqrt = Vt @ np.diag(np.sqrt(Dt)) @ Vt.T

    X_source_aligned = X_source @ Cs_inv_sqrt @ Ct_sqrt
    return X_source_aligned


def eval_ensemble_fold_rmse(all_preds, y, folds, model_indices, weights=None):
    """fold-RMSE平均でアンサンブルを評価"""
    fold_rmses = []
    for f_idx, (train_idx, test_idx) in enumerate(folds):
        preds = []
        for m_idx in model_indices:
            preds.append(all_preds[m_idx][f_idx])
        if weights is not None:
            w = np.array(weights)
            w = w / w.sum()
            ensemble_pred = sum(w_i * p for w_i, p in zip(w, preds))
        else:
            ensemble_pred = np.mean(preds, axis=0)
        ensemble_pred = np.clip(ensemble_pred, 0, 300)
        fold_rmses.append(rmse(y[test_idx], ensemble_pred))
    return np.mean(fold_rmses), np.std(fold_rmses), fold_rmses


def optimize_weights(all_preds, y, folds, model_indices, n_restarts=30):
    n = len(model_indices)

    def obj(w):
        w_n = np.abs(w) / np.sum(np.abs(w))
        r, _, _ = eval_ensemble_fold_rmse(all_preds, y, folds, model_indices, w_n)
        return r

    best_r, best_w = np.inf, np.ones(n) / n
    for s in range(n_restarts):
        w0 = np.random.RandomState(s).dirichlet(np.ones(n))
        res = minimize(obj, w0, method="Nelder-Mead", options={"maxiter": 3000})
        if res.fun < best_r:
            best_r = res.fun
            best_w = np.abs(res.x) / np.sum(np.abs(res.x))
    return best_w, best_r


def stacking_loso(all_preds, y, folds, model_indices, alpha=1.0):
    """LOSOのネストされたスタッキング"""
    fold_rmses = []
    all_stacking_preds = []

    for f_idx, (train_idx, test_idx) in enumerate(folds):
        # メタ特徴量: 他のfoldの予測値を結合してtrain
        meta_X_train = []
        meta_y_train = []
        for other_f_idx, (_, other_test_idx) in enumerate(folds):
            if other_f_idx == f_idx:
                continue
            row_preds = np.column_stack([all_preds[m_idx][other_f_idx] for m_idx in model_indices])
            meta_X_train.append(row_preds)
            meta_y_train.append(y[other_test_idx])

        meta_X_train = np.vstack(meta_X_train)
        meta_y_train = np.concatenate(meta_y_train)

        # テスト用メタ特徴量
        meta_X_test = np.column_stack([all_preds[m_idx][f_idx] for m_idx in model_indices])

        # メタモデル学習・予測
        meta = Ridge(alpha=alpha)
        meta.fit(meta_X_train, meta_y_train)
        stacking_pred = np.clip(meta.predict(meta_X_test), 0, 300)

        fold_rmses.append(rmse(y[test_idx], stacking_pred))
        all_stacking_preds.append(stacking_pred)

    return np.mean(fold_rmses), np.std(fold_rmses), fold_rmses, all_stacking_preds


# ==============================================================================
# predict_fold: 既存5モデル(M1-M5) + 新4モデル(M6-M9)
# ==============================================================================

def predict_fold(X_tr_raw, X_te_raw, y_train, groups_train, cfg):
    """configベースでfold予測を生成"""
    pp = cfg["pp"]

    # --- 前処理 ---
    if pp == "SNV":
        X_tr, X_te = apply_snv(X_tr_raw), apply_snv(X_te_raw)
    elif pp == "EPO(1)":
        P = compute_epo_projection(X_tr_raw, groups_train, n_components=1)
        X_tr, X_te = apply_epo(X_tr_raw, P), apply_epo(X_te_raw, P)
    elif pp == "SNV+AsLS(1e6)":
        X_tr = apply_asls(apply_snv(X_tr_raw), lam=1e6)
        X_te = apply_asls(apply_snv(X_te_raw), lam=1e6)
    elif pp == "PiecewiseMSC(seg=3)":
        ref = compute_msc_reference(X_tr_raw)
        X_tr = apply_piecewise_msc(X_tr_raw, ref, 3)
        X_te = apply_piecewise_msc(X_te_raw, ref, 3)
    elif pp == "SG2d+EPO(1)":
        X_sg = apply_savgol(X_tr_raw, deriv=2, window_length=7)
        X_sg_te = apply_savgol(X_te_raw, deriv=2, window_length=7)
        P = compute_epo_projection(X_sg, groups_train, n_components=1)
        X_tr, X_te = apply_epo(X_sg, P), apply_epo(X_sg_te, P)
    elif pp == "SNV+CORAL":
        X_tr_snv = apply_snv(X_tr_raw)
        X_te_snv = apply_snv(X_te_raw)
        X_tr = coral_transform(X_tr_snv, X_te_snv)
        X_te = X_te_snv
    elif pp == "EPO(1)+CORAL":
        P = compute_epo_projection(X_tr_raw, groups_train, n_components=1)
        X_tr_epo = apply_epo(X_tr_raw, P)
        X_te_epo = apply_epo(X_te_raw, P)
        X_tr = coral_transform(X_tr_epo, X_te_epo)
        X_te = X_te_epo
    else:
        X_tr, X_te = X_tr_raw.copy(), X_te_raw.copy()

    # --- 特徴量選択 ---
    fs = cfg.get("fs")
    if fs and fs.startswith("siPLS"):
        parts = fs.replace("siPLS(", "").rstrip(")").split(",")
        ni, nc = int(parts[0]), int(parts[1])
        X_tr, X_te, _ = sipls_select(X_tr, y_train, X_te, n_intervals=ni, n_components=3, n_combine=nc)
    elif fs and fs.startswith("iPLS"):
        ni = int(fs.replace("iPLS(", "").rstrip(")"))
        X_tr, X_te, _ = ipls_select(X_tr, y_train, X_te, n_intervals=ni, n_components=3, n_best=1)

    # --- 目的変数変換 ---
    tf = cfg["tf"]
    if tf == "sqrt":
        y_fit = np.sqrt(y_train)
    elif tf == "boxcox":
        y_positive = y_train + 1  # Box-CoxはPositive値のみ
        y_fit, lam = boxcox(y_positive)
    else:
        y_fit = y_train.copy()

    # --- モデル ---
    mdl = cfg["model"]
    if mdl.startswith("PLS("):
        nc = int(mdl.split("(")[1].rstrip(")"))
        nc = min(nc, X_tr.shape[1] - 1)
        m = PLSRegression(n_components=max(1, nc))
        m.fit(X_tr, y_fit)
        pred = m.predict(X_te).ravel()
    elif mdl.startswith("LWPLS"):
        # LWPLS(n_components, k)
        params = mdl.replace("LWPLS(", "").rstrip(")").split(",")
        n_comp = int(params[0])
        k = int(params[1])
        pred = lwpls_predict(X_tr, y_fit, X_te, n_components=n_comp, k=k, sigma_factor=1.0)
    else:
        raise ValueError(f"Unknown model: {mdl}")

    # --- 逆変換 ---
    if tf == "sqrt":
        pred = np.clip(pred, 0, None) ** 2
    elif tf == "boxcox":
        pred = np.clip(pred, -50, 50)  # inv_boxcoxのオーバーフロー防止
        pred = inv_boxcox(pred, lam) - 1

    return pred


# ==============================================================================
# メイン
# ==============================================================================

def main():
    t0 = time.time()

    # データ読み込み
    df = load_train(DATA_DIR)
    sc = get_spectral_columns(df)
    X = df[sc].values
    y = df["含水率"].values
    g = df["樹種"].values
    logo = LeaveOneGroupOut()
    folds = list(logo.split(X, y, g))
    sp_names = [np.unique(g[te])[0] for _, te in folds]
    n_folds = len(folds)

    print("=" * 80)
    print("Issue #77: サイクル1 - 4仮説統合検証")
    print("  ベースライン: BestCombo5-Weighted RMSE=14.963")
    print("=" * 80)

    # ===== モデル定義 =====
    configs = [
        # 既存5モデル (M1-M5)
        {"name": "M1: EPO(1)+PLS(4)+sqrt",
         "pp": "EPO(1)", "model": "PLS(4)", "tf": "sqrt"},
        {"name": "M2: SG2d+EPO(1)+PLS(3)+raw",
         "pp": "SG2d+EPO(1)", "model": "PLS(3)", "tf": "raw"},
        {"name": "M3: SNV+iPLS(50)+PLS(4)+sqrt",
         "pp": "SNV", "model": "PLS(4)", "tf": "sqrt", "fs": "iPLS(50)"},
        {"name": "M4: PMSC+siPLS(30,3)+PLS(4)+sqrt",
         "pp": "PiecewiseMSC(seg=3)", "model": "PLS(4)", "tf": "sqrt", "fs": "siPLS(30,3)"},
        {"name": "M5: SNV+AsLS(1e6)+siPLS(30,3)+PLS(5)+sqrt",
         "pp": "SNV+AsLS(1e6)", "model": "PLS(5)", "tf": "sqrt", "fs": "siPLS(30,3)"},
        # 仮説1: CORAL+PLS (M6)
        {"name": "M6: SNV+CORAL+PLS(4)+sqrt",
         "pp": "SNV+CORAL", "model": "PLS(4)", "tf": "sqrt"},
        # 仮説2: Box-Cox変換 (M7, M8)
        {"name": "M7: EPO(1)+PLS(4)+boxcox",
         "pp": "EPO(1)", "model": "PLS(4)", "tf": "boxcox"},
        {"name": "M8: SNV+iPLS(50)+PLS(4)+boxcox",
         "pp": "SNV", "model": "PLS(4)", "tf": "boxcox", "fs": "iPLS(50)"},
        # 仮説4: LW-PLS (M9, M10)
        {"name": "M9: SNV+LWPLS(4,100)+sqrt",
         "pp": "SNV", "model": "LWPLS(4,100)", "tf": "sqrt"},
        {"name": "M10: EPO(1)+LWPLS(4,100)+sqrt",
         "pp": "EPO(1)", "model": "LWPLS(4,100)", "tf": "sqrt"},
        # 仮説1追加: EPO+CORAL (M11)
        {"name": "M11: EPO(1)+CORAL+PLS(4)+sqrt",
         "pp": "EPO(1)+CORAL", "model": "PLS(4)", "tf": "sqrt"},
    ]

    n_models = len(configs)
    model_names = [c["name"] for c in configs]
    all_preds = []  # all_preds[model_idx][fold_idx] = np.array

    # ===== 各モデルのfold別予測を収集 =====
    print("\n--- 個別モデル評価 (fold-RMSE平均) ---\n")
    for m_idx, cfg in enumerate(configs):
        t1 = time.time()
        fold_preds = []
        fold_rmses = []
        for f_idx, (tr, te) in enumerate(folds):
            try:
                p = predict_fold(X[tr], X[te], y[tr], g[tr], cfg)
                fold_preds.append(p)
                fold_rmses.append(rmse(y[te], p))
            except Exception as e:
                print(f"    [ERROR] {cfg['name']} fold {f_idx} ({sp_names[f_idx]}): {e}")
                fold_preds.append(np.full(len(te), y[tr].mean()))
                fold_rmses.append(999.0)
        all_preds.append(fold_preds)
        elapsed = time.time() - t1
        print(f"  {cfg['name']}: {np.mean(fold_rmses):.2f} ± {np.std(fold_rmses):.2f} ({elapsed:.1f}s)",
              flush=True)

    # ===== 仮説3: スタッキング評価 =====
    print("\n--- 仮説3: スタッキング評価 ---\n")

    stacking_configs = [
        ("Stacking(M1-M5, Ridge α=1.0)", list(range(5)), 1.0),
        ("Stacking(M1-M5, Ridge α=10.0)", list(range(5)), 10.0),
        ("Stacking(M1-M5, Ridge α=100.0)", list(range(5)), 100.0),
        ("Stacking(M1-M9, Ridge α=1.0)", list(range(n_models)), 1.0),
        ("Stacking(M1-M9, Ridge α=10.0)", list(range(n_models)), 10.0),
        ("Stacking(M1-M9, Ridge α=100.0)", list(range(n_models)), 100.0),
    ]

    stacking_results = []
    for name, model_indices, alpha in stacking_configs:
        sr, ss, sfr, spreds = stacking_loso(all_preds, y, folds, model_indices, alpha=alpha)
        stacking_results.append((name, sr, ss, sfr, spreds))
        print(f"  {name}: {sr:.2f} ± {ss:.2f}")

    # ===== アンサンブル最適化 (全9モデル) =====
    print("\n--- アンサンブル最適化 (全9モデル) ---\n")

    ensemble_results = []

    # BestCombo探索 (3〜8)
    for k in range(3, min(n_models + 1, 9)):
        best_r_avg, best_c_avg = np.inf, None
        for combo in combinations(range(n_models), k):
            r, _, _ = eval_ensemble_fold_rmse(all_preds, y, folds, list(combo))
            if r < best_r_avg:
                best_r_avg = r
                best_c_avg = combo

        if best_c_avg:
            idx = list(best_c_avg)
            combo_names = [model_names[i] for i in idx]

            # 均等平均
            ensemble_results.append({
                "method": f"BestCombo{k}-Avg",
                "rmse": best_r_avg,
                "models": combo_names,
                "indices": idx,
                "weights": None,
            })

            # 重み最適化
            w, rw = optimize_weights(all_preds, y, folds, idx, n_restarts=30)
            ensemble_results.append({
                "method": f"BestCombo{k}-Weighted",
                "rmse": rw,
                "models": combo_names,
                "indices": idx,
                "weights": w,
            })

            print(f"  BestCombo{k}-Avg: {best_r_avg:.4f} | {[model_names[i].split(': ')[1] for i in idx]}")
            print(f"  BestCombo{k}-Weighted: {rw:.4f}")
            if w is not None:
                for i, wi in zip(idx, w):
                    if wi > 0.01:
                        print(f"    {wi:.3f}: {model_names[i]}")
            print()

    # ソートして最良を特定
    ensemble_results.sort(key=lambda x: x["rmse"])

    # ===== ベストアンサンブルのfold詳細 =====
    print("\n--- ベストアンサンブルのfold詳細 ---\n")

    best_ens = ensemble_results[0]
    best_idx = best_ens["indices"]
    best_w = best_ens["weights"]

    _, _, best_fold_rmses = eval_ensemble_fold_rmse(all_preds, y, folds, best_idx, best_w)

    print(f"  ベスト: {best_ens['method']} (RMSE={best_ens['rmse']:.4f})")
    print(f"  モデル: {[model_names[i].split(': ')[1] for i in best_idx]}")
    if best_w is not None:
        print(f"  重み: {[f'{w:.3f}' for w in best_w]}")
    print()
    for f_idx, (_, te) in enumerate(folds):
        sp = sp_names[f_idx]
        tag = " ※参考値" if sp == "ベイスギ" else ""
        print(f"  {sp}: {best_fold_rmses[f_idx]:.2f}{tag}")

    # ベイスギ除外RMSE
    beisugi_idx_list = [i for i, sp in enumerate(sp_names) if sp == "ベイスギ"]
    non_beisugi_rmses = [r for i, r in enumerate(best_fold_rmses) if i not in beisugi_idx_list]
    print(f"\n  ベイスギ除外 fold-RMSE平均: {np.mean(non_beisugi_rmses):.4f}")

    # ===== 残差相関行列 =====
    print("\n--- 残差相関行列 ---\n")

    model_residuals = np.zeros((len(y), n_models))
    for m_idx in range(n_models):
        for f_idx, (_, te) in enumerate(folds):
            model_residuals[te, m_idx] = y[te] - all_preds[m_idx][f_idx]

    corr = np.corrcoef(model_residuals.T)
    short_names = [f"M{i+1}" for i in range(n_models)]

    # ヘッダー
    print(f"{'':>4}", end="")
    for sn in short_names:
        print(f" | {sn:>6}", end="")
    print()
    for i, sn in enumerate(short_names):
        print(f"{sn:>4}", end="")
        for j in range(n_models):
            print(f" | {corr[i, j]:>6.3f}", end="")
        print()

    # ===== 結果サマリー =====
    print("\n" + "=" * 80)
    print(f"BEST RESULT: {best_ens['rmse']:.4f} ({best_ens['method']})")
    print(f"ベースライン: 14.963 (既存BestCombo5-Weighted)")
    diff = 14.963 - best_ens["rmse"]
    print(f"改善幅: {diff:+.4f}")
    print("=" * 80)

    # ===== スタッキングのベスト結果も表示 =====
    best_stack = min(stacking_results, key=lambda x: x[1])
    print(f"\nスタッキングベスト: {best_stack[0]}: {best_stack[1]:.4f}")

    # ===== CSV保存 =====
    rows = []

    # 個別モデル
    for m_idx, cfg in enumerate(configs):
        fold_rmse_vals = [rmse(y[folds[f][1]], all_preds[m_idx][f]) for f in range(n_folds)]
        rows.append({
            "method": cfg["name"],
            "rmse_mean": np.mean(fold_rmse_vals),
            "rmse_std": np.std(fold_rmse_vals),
            "type": "individual",
        })

    # スタッキング
    for name, sr, ss, sfr, spreds in stacking_results:
        rows.append({
            "method": name,
            "rmse_mean": sr,
            "rmse_std": ss,
            "type": "stacking",
        })

    # アンサンブル
    for ens in ensemble_results:
        rows.append({
            "method": ens["method"],
            "rmse_mean": ens["rmse"],
            "rmse_std": 0,
            "type": "ensemble",
            "models": str([m.split(": ")[1] if ": " in m else m for m in ens["models"]]),
        })

    result_df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "issue77_cycle1_results.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"\n結果保存: {csv_path}")

    elapsed = time.time() - t0
    print(f"総実行時間: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
