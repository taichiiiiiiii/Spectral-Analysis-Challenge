"""Issue #78 Cycle 2: 新アプローチによる精度向上探索

サイクル1（CORAL/Box-Cox/スタッキング/LW-PLS）が失敗したため、
前処理パラメータ・組み合わせの多様化で精度向上を試みる。
現在のベスト: BestCombo5-Weighted RMSE=14.963（5モデル）

仮説:
  A: EPO成分数の変更 (n_components=2, 3)
  B: 異なるsiPLS/iPLSパラメータ
  C: MSC前処理（SNVの代替）
  D: SG1d（1次微分）の活用
  E: Detrending前処理
  F: EPO+AsLS組み合わせ
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
from sklearn.model_selection import LeaveOneGroupOut

from src.eda.data_loader import load_train, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue20_savgol import apply_savgol
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue19_msc import compute_msc_reference, apply_msc
from src.preprocessing.issue62_additional_preprocessing import apply_asls, apply_piecewise_msc
from src.preprocessing.issue24_detrending import apply_detrending, apply_snv_detrending
from src.modeling.issue65_feature_selection import sipls_select, ipls_select

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def predict_fold(X_tr_raw, X_te_raw, y_train, groups_train, cfg):
    """前処理 -> 特徴量選択 -> PLS予測を実行する。"""
    pp = cfg["pp"]

    # === 前処理 ===
    if pp == "SNV":
        X_tr, X_te = apply_snv(X_tr_raw), apply_snv(X_te_raw)
    elif pp == "EPO(1)":
        P = compute_epo_projection(X_tr_raw, groups_train, n_components=1)
        X_tr, X_te = apply_epo(X_tr_raw, P), apply_epo(X_te_raw, P)
    elif pp == "EPO(2)":
        P = compute_epo_projection(X_tr_raw, groups_train, n_components=2)
        X_tr, X_te = apply_epo(X_tr_raw, P), apply_epo(X_te_raw, P)
    elif pp == "EPO(3)":
        P = compute_epo_projection(X_tr_raw, groups_train, n_components=3)
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
    elif pp == "MSC":
        ref = compute_msc_reference(X_tr_raw)
        X_tr = apply_msc(X_tr_raw, ref)
        X_te = apply_msc(X_te_raw, ref)
    elif pp == "SG1d+EPO(1)":
        X_sg = apply_savgol(X_tr_raw, deriv=1, window_length=7)
        X_sg_te = apply_savgol(X_te_raw, deriv=1, window_length=7)
        P = compute_epo_projection(X_sg, groups_train, n_components=1)
        X_tr, X_te = apply_epo(X_sg, P), apply_epo(X_sg_te, P)
    elif pp == "SNV+SG1d":
        X_tr = apply_savgol(apply_snv(X_tr_raw), deriv=1, window_length=7)
        X_te = apply_savgol(apply_snv(X_te_raw), deriv=1, window_length=7)
    elif pp == "EPO(1)+AsLS(1e6)":
        P = compute_epo_projection(X_tr_raw, groups_train, n_components=1)
        X_tr = apply_asls(apply_epo(X_tr_raw, P), lam=1e6)
        X_te = apply_asls(apply_epo(X_te_raw, P), lam=1e6)
    elif pp == "MSC+AsLS(1e6)":
        ref = compute_msc_reference(X_tr_raw)
        X_tr = apply_asls(apply_msc(X_tr_raw, ref), lam=1e6)
        X_te = apply_asls(apply_msc(X_te_raw, ref), lam=1e6)
    elif pp == "Detrend":
        X_tr = apply_detrending(X_tr_raw, poly_order=2)
        X_te = apply_detrending(X_te_raw, poly_order=2)
    elif pp == "SNV+Detrend":
        X_tr = apply_snv_detrending(X_tr_raw, poly_order=2)
        X_te = apply_snv_detrending(X_te_raw, poly_order=2)
    else:
        X_tr, X_te = X_tr_raw.copy(), X_te_raw.copy()

    # === 特徴量選択 ===
    fs = cfg.get("fs")
    if fs and fs.startswith("siPLS"):
        parts = fs.replace("siPLS(", "").rstrip(")").split(",")
        ni, nc = int(parts[0]), int(parts[1])
        X_tr, X_te, _ = sipls_select(X_tr, y_train, X_te, n_intervals=ni, n_components=3, n_combine=nc)
    elif fs and fs.startswith("iPLS"):
        ni = int(fs.replace("iPLS(", "").rstrip(")"))
        X_tr, X_te, _ = ipls_select(X_tr, y_train, X_te, n_intervals=ni, n_components=3, n_best=1)

    # === 目的変数変換 ===
    tf = cfg["tf"]
    y_fit = np.sqrt(y_train) if tf == "sqrt" else y_train.copy()

    # === モデル ===
    mdl = cfg["model"]
    if mdl.startswith("PLS("):
        nc = int(mdl.split("(")[1].rstrip(")"))
        nc = min(nc, X_tr.shape[1] - 1)
        m = PLSRegression(n_components=max(1, nc))
        m.fit(X_tr, y_fit)
        pred = m.predict(X_te).ravel()
    else:
        raise ValueError(mdl)

    if tf == "sqrt":
        pred = np.clip(pred, 0, None) ** 2
    return pred


def opt_weights(fold_preds_list, y, folds, n_restarts=30):
    """Nelder-Meadで最適重みを求める。"""
    n_models = len(fold_preds_list)

    def objective(w):
        wn = np.abs(w) / np.sum(np.abs(w))
        fold_rmses = []
        for f_idx, (_, te) in enumerate(folds):
            ens = sum(wn[m] * fold_preds_list[m][f_idx] for m in range(n_models))
            fold_rmses.append(rmse(y[te], np.clip(ens, 0, 300)))
        return np.mean(fold_rmses)

    best_r, best_w = np.inf, np.ones(n_models) / n_models
    for s in range(n_restarts):
        w0 = np.random.RandomState(s).dirichlet(np.ones(n_models))
        res = minimize(objective, w0, method="Nelder-Mead",
                       options={"maxiter": 3000, "xatol": 1e-5, "fatol": 1e-5})
        if res.fun < best_r:
            best_r = res.fun
            best_w = np.abs(res.x) / np.sum(np.abs(res.x))
    return best_w, best_r


def main():
    t0 = time.time()
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
    print("Issue #78 Cycle 2: 新アプローチによる精度向上探索")
    print("=" * 80)
    print(f"データ: {len(y)} samples, {X.shape[1]} features, {n_folds} folds")
    print(f"現在のベスト: BestCombo5-Weighted RMSE=14.963")

    # ===== 20モデル構成 =====
    configs = [
        # --- 既存5モデル (M1-M5) ---
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

        # --- 仮説A: EPO成分数 ---
        {"name": "M6: EPO(2)+PLS(4)+sqrt",
         "pp": "EPO(2)", "model": "PLS(4)", "tf": "sqrt"},
        {"name": "M7: EPO(3)+PLS(4)+sqrt",
         "pp": "EPO(3)", "model": "PLS(4)", "tf": "sqrt"},

        # --- 仮説B: 異なるsiPLS/iPLS ---
        {"name": "M8: SNV+siPLS(20,2)+PLS(4)+sqrt",
         "pp": "SNV", "model": "PLS(4)", "tf": "sqrt", "fs": "siPLS(20,2)"},
        {"name": "M9: SNV+siPLS(40,3)+PLS(4)+sqrt",
         "pp": "SNV", "model": "PLS(4)", "tf": "sqrt", "fs": "siPLS(40,3)"},
        {"name": "M10: SNV+iPLS(30)+PLS(4)+sqrt",
         "pp": "SNV", "model": "PLS(4)", "tf": "sqrt", "fs": "iPLS(30)"},
        {"name": "M11: SNV+iPLS(40)+PLS(4)+sqrt",
         "pp": "SNV", "model": "PLS(4)", "tf": "sqrt", "fs": "iPLS(40)"},

        # --- 仮説C: MSC ---
        {"name": "M12: MSC+iPLS(50)+PLS(4)+sqrt",
         "pp": "MSC", "model": "PLS(4)", "tf": "sqrt", "fs": "iPLS(50)"},
        {"name": "M13: MSC+siPLS(30,3)+PLS(4)+sqrt",
         "pp": "MSC", "model": "PLS(4)", "tf": "sqrt", "fs": "siPLS(30,3)"},

        # --- 仮説D: SG1d ---
        {"name": "M14: SG1d+EPO(1)+PLS(3)+raw",
         "pp": "SG1d+EPO(1)", "model": "PLS(3)", "tf": "raw"},
        {"name": "M15: SNV+SG1d+PLS(4)+sqrt",
         "pp": "SNV+SG1d", "model": "PLS(4)", "tf": "sqrt"},

        # --- 仮説E: AsLS組み合わせ ---
        {"name": "M16: EPO(1)+AsLS(1e6)+PLS(4)+sqrt",
         "pp": "EPO(1)+AsLS(1e6)", "model": "PLS(4)", "tf": "sqrt"},
        {"name": "M17: MSC+AsLS(1e6)+PLS(4)+sqrt",
         "pp": "MSC+AsLS(1e6)", "model": "PLS(4)", "tf": "sqrt"},

        # --- 仮説F: PLS成分数バリエーション ---
        {"name": "M18: EPO(1)+PLS(3)+sqrt",
         "pp": "EPO(1)", "model": "PLS(3)", "tf": "sqrt"},
        {"name": "M19: SNV+iPLS(50)+PLS(3)+sqrt",
         "pp": "SNV", "model": "PLS(3)", "tf": "sqrt", "fs": "iPLS(50)"},
        {"name": "M20: SNV+AsLS(1e6)+siPLS(30,3)+PLS(4)+sqrt",
         "pp": "SNV+AsLS(1e6)", "model": "PLS(4)", "tf": "sqrt", "fs": "siPLS(30,3)"},
    ]

    n_models = len(configs)
    all_preds = []  # all_preds[model_idx][fold_idx] = np.array
    model_rmses_list = []  # 各モデルの平均RMSE

    # ===== 各モデルのfold別予測を収集 =====
    print(f"\n--- 個別モデル評価 ({n_models}モデル x {n_folds}フォールド) ---\n")
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
                print(f"  [ERROR] {cfg['name']} fold {f_idx} ({sp_names[f_idx]}): {e}")
                fold_preds.append(np.full(len(te), y[tr].mean()))
                fold_rmses.append(999.0)
        all_preds.append(fold_preds)
        mean_r = np.mean(fold_rmses)
        model_rmses_list.append(mean_r)
        elapsed = time.time() - t1
        print(f"  [{m_idx+1:>2}/{n_models}] {cfg['name']:<50s} RMSE={mean_r:.3f} +/- {np.std(fold_rmses):.2f}  ({elapsed:.1f}s)",
              flush=True)

    # ===== 個別モデルRMSEランキング =====
    print("\n" + "=" * 80)
    print("個別モデルRMSEランキング")
    print("=" * 80)
    ranking = sorted(enumerate(model_rmses_list), key=lambda x: x[1])
    for rank, (m_idx, r) in enumerate(ranking):
        tag = " *既存" if m_idx < 5 else " NEW"
        print(f"  #{rank+1:>2}: {configs[m_idx]['name']:<50s} RMSE={r:.3f}{tag}")

    # ===== アンサンブル最適化: BestCombo3-8の全組み合わせ探索 =====
    print("\n" + "=" * 80)
    print("アンサンブル最適化: BestCombo3-8の全組み合わせ探索")
    print("=" * 80)

    best_overall_rmse = np.inf
    best_overall_combo = None
    best_overall_weights = None

    ensemble_results_rows = []

    for combo_size in range(3, 9):
        t_combo = time.time()
        # 上位モデルから候補を絞る（全組み合わせは多すぎるので上位12モデルから選択）
        top_k = min(12, n_models)
        top_indices = [idx for idx, _ in ranking[:top_k]]

        best_rmse_this_size = np.inf
        best_combo_this_size = None
        best_w_this_size = None
        n_combos = 0

        for combo in combinations(top_indices, combo_size):
            n_combos += 1
            combo_preds = [all_preds[m] for m in combo]
            w, r = opt_weights(combo_preds, y, folds, n_restarts=15)
            if r < best_rmse_this_size:
                best_rmse_this_size = r
                best_combo_this_size = combo
                best_w_this_size = w

        elapsed_c = time.time() - t_combo
        print(f"\n  BestCombo{combo_size} (探索: {n_combos}組合せ, {elapsed_c:.0f}s):")
        print(f"    RMSE = {best_rmse_this_size:.4f}")
        for i, m_idx in enumerate(best_combo_this_size):
            print(f"    {configs[m_idx]['name']:<50s} w={best_w_this_size[i]:.4f}")

        if best_rmse_this_size < best_overall_rmse:
            best_overall_rmse = best_rmse_this_size
            best_overall_combo = best_combo_this_size
            best_overall_weights = best_w_this_size

        ensemble_results_rows.append({
            "combo_size": combo_size,
            "rmse": best_rmse_this_size,
            "models": "+".join([f"M{m+1}" for m in best_combo_this_size]),
            "weights": ",".join([f"{w:.4f}" for w in best_w_this_size]),
        })

    # ===== ベスト結果のfold詳細 =====
    print("\n" + "=" * 80)
    print(f"ベストアンサンブル詳細: RMSE={best_overall_rmse:.4f}")
    print("=" * 80)

    combo_indices = best_overall_combo
    combo_weights = best_overall_weights
    combo_preds_list = [all_preds[m] for m in combo_indices]

    print(f"\n構成モデル ({len(combo_indices)}モデル):")
    for i, m_idx in enumerate(combo_indices):
        print(f"  {configs[m_idx]['name']}: w={combo_weights[i]:.4f}")

    print(f"\n{'樹種':<12} | {'N':>4}", end="")
    for m in combo_indices:
        short = f"M{m+1}"
        print(f" | {short:>8}", end="")
    print(f" | {'Ens最適':>8}")
    print("-" * (22 + 11 * len(combo_indices) + 11))

    fold_ens_rmses = []
    for f_idx, (_, te) in enumerate(folds):
        sp = sp_names[f_idx]
        row_str = f"{sp:<12} | {len(te):>4}"
        for i, m in enumerate(combo_indices):
            r = rmse(y[te], all_preds[m][f_idx])
            row_str += f" | {r:>8.2f}"
        ens_pred = sum(combo_weights[i] * combo_preds_list[i][f_idx]
                       for i in range(len(combo_indices)))
        ens_pred = np.clip(ens_pred, 0, 300)
        r_ens = rmse(y[te], ens_pred)
        fold_ens_rmses.append(r_ens)
        row_str += f" | {r_ens:>8.2f}"
        tag = " *参考" if sp == "ベイスギ" else ""
        print(row_str + tag)

    print("-" * (22 + 11 * len(combo_indices) + 11))
    print(f"{'平均':<12} | {'':>4}", end="")
    for i, m in enumerate(combo_indices):
        avg_r = np.mean([rmse(y[te], all_preds[m][f_idx]) for f_idx, (_, te) in enumerate(folds)])
        print(f" | {avg_r:>8.2f}", end="")
    print(f" | {np.mean(fold_ens_rmses):>8.2f}")

    # ===== M1-M5ベースラインとの比較 =====
    m1_5_preds = [all_preds[m] for m in range(5)]
    w_m1_5, rmse_m1_5 = opt_weights(m1_5_preds, y, folds, n_restarts=15)
    print(f"\nM1-M5ベースライン (Weighted): RMSE={rmse_m1_5:.4f}")

    # ===== 残差相関行列 =====
    print("\n" + "=" * 80)
    print("残差相関行列（全モデル）")
    print("=" * 80)

    model_residuals = np.zeros((len(y), n_models))
    for m_idx in range(n_models):
        for f_idx, (_, te) in enumerate(folds):
            model_residuals[te, m_idx] = y[te] - all_preds[m_idx][f_idx]

    corr = np.corrcoef(model_residuals.T)

    # 低相関ペア
    print("\n低相関ペア（相関 < 0.85）:")
    low_corr_pairs = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            if corr[i, j] < 0.85:
                low_corr_pairs.append((i, j, corr[i, j]))
    low_corr_pairs.sort(key=lambda x: x[2])

    if low_corr_pairs:
        for i, j, c in low_corr_pairs[:20]:
            ni = configs[i]['name'].split(': ')[1] if ': ' in configs[i]['name'] else configs[i]['name']
            nj = configs[j]['name'].split(': ')[1] if ': ' in configs[j]['name'] else configs[j]['name']
            print(f"  M{i+1} vs M{j+1}: r={c:.3f}  ({ni} vs {nj})")
    else:
        print("  なし（全ペアで相関 >= 0.85）")

    # 相関行列の要約統計
    upper_tri = corr[np.triu_indices(n_models, k=1)]
    print(f"\n残差相関の要約: min={upper_tri.min():.3f}, max={upper_tri.max():.3f}, "
          f"mean={upper_tri.mean():.3f}, median={np.median(upper_tri):.3f}")

    # ===== ベースラインとの比較 =====
    print("\n" + "=" * 80)
    print("ベースラインとの比較")
    print("=" * 80)
    print(f"  既存ベスト (BestCombo5-Weighted): RMSE = 14.963")
    print(f"  M1-M5再計算 (Weighted): RMSE = {rmse_m1_5:.4f}")
    print(f"  今回のベスト: RMSE = {best_overall_rmse:.4f}")
    diff = rmse_m1_5 - best_overall_rmse
    if diff > 0:
        print(f"  改善: {diff:+.4f} ({diff/rmse_m1_5*100:.2f}%)")
    else:
        print(f"  差分: {-diff:+.4f} 悪化 ({diff/rmse_m1_5*100:.2f}%)")

    # ===== CSV保存 =====
    # 個別モデル結果
    individual_rows = []
    for m_idx, cfg in enumerate(configs):
        fold_rmses = [rmse(y[te], all_preds[m_idx][f_idx]) for f_idx, (_, te) in enumerate(folds)]
        individual_rows.append({
            "model_id": f"M{m_idx+1}",
            "name": cfg["name"],
            "pp": cfg["pp"],
            "fs": cfg.get("fs", "None"),
            "model": cfg["model"],
            "tf": cfg["tf"],
            "rmse_mean": np.mean(fold_rmses),
            "rmse_std": np.std(fold_rmses),
            **{f"fold_{sp_names[f]}": fold_rmses[f] for f in range(n_folds)},
        })

    df_individual = pd.DataFrame(individual_rows)
    df_individual.to_csv(OUT_DIR / "issue78_cycle2_results.csv", index=False)

    # アンサンブル結果
    df_ensemble = pd.DataFrame(ensemble_results_rows)
    df_ensemble.to_csv(OUT_DIR / "issue78_cycle2_ensemble.csv", index=False)

    print(f"\n結果保存:")
    print(f"  個別モデル: {OUT_DIR / 'issue78_cycle2_results.csv'}")
    print(f"  アンサンブル: {OUT_DIR / 'issue78_cycle2_ensemble.csv'}")
    print(f"\n総実行時間: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
