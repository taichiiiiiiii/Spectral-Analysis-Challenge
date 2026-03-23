"""Issue #76: Fold別（樹種別）RMSE診断

ベストモデル5つのfold別RMSEを出力し、どの樹種で予測が悪いかを特定する。
残差分析により、予測が大きく外れたサンプルの特徴を分析する。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import warnings
import time
warnings.filterwarnings("ignore")

from src.eda.data_loader import load_train, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue20_savgol import apply_savgol
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue19_msc import compute_msc_reference
from src.preprocessing.issue62_additional_preprocessing import apply_asls, apply_piecewise_msc
from src.modeling.issue65_feature_selection import sipls_select, ipls_select
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut
from scipy.optimize import minimize

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def predict_fold(X_tr_raw, X_te_raw, y_train, groups_train, cfg):
    """predict_fold from run_issue75_extended_search.py"""
    pp = cfg["pp"]
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
    else:
        X_tr, X_te = X_tr_raw.copy(), X_te_raw.copy()

    fs = cfg.get("fs")
    if fs and fs.startswith("siPLS"):
        parts = fs.replace("siPLS(", "").rstrip(")").split(",")
        ni, nc = int(parts[0]), int(parts[1])
        X_tr, X_te, _ = sipls_select(X_tr, y_train, X_te, n_intervals=ni, n_components=3, n_combine=nc)
    elif fs and fs.startswith("iPLS"):
        ni = int(fs.replace("iPLS(", "").rstrip(")"))
        X_tr, X_te, _ = ipls_select(X_tr, y_train, X_te, n_intervals=ni, n_components=3, n_best=1)

    tf = cfg["tf"]
    y_fit = np.sqrt(y_train) if tf == "sqrt" else y_train.copy()

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
    """Nelder-Meadで最適重みを求める。

    fold_preds_list: list of list, [model_idx][fold_idx] = predictions
    """
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
        res = minimize(objective, w0, method="Nelder-Mead", options={"maxiter": 3000})
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
    print("Issue #76: Fold別（樹種別）RMSE診断")
    print("=" * 80)

    # ===== 5モデル構成 =====
    configs = [
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
    ]

    n_models = len(configs)
    # all_preds[model_idx][fold_idx] = np.array of predictions
    all_preds = []

    # ===== 各モデルのfold別予測を収集 =====
    print("\n--- モデル評価 ---\n")
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
        elapsed = time.time() - t1
        print(f"  {cfg['name']}: RMSE={np.mean(fold_rmses):.2f} ± {np.std(fold_rmses):.2f} ({elapsed:.1f}s)",
              flush=True)

    # ===== 重み最適化 =====
    print("\n--- 重み最適化 (Nelder-Mead) ---\n")
    opt_w, opt_rmse = opt_weights(all_preds, y, folds, n_restarts=30)
    print(f"  最適重み:")
    for m_idx, cfg in enumerate(configs):
        print(f"    {cfg['name']}: {opt_w[m_idx]:.4f}")
    print(f"  最適化後RMSE: {opt_rmse:.4f}")

    # ===== Fold別RMSE表 =====
    print("\n" + "=" * 80)
    print("Fold別（樹種別）RMSE一覧")
    print("=" * 80)

    # ヘッダー
    header = f"{'樹種':<12}"
    for cfg in configs:
        short = cfg['name'].split(': ')[1] if ': ' in cfg['name'] else cfg['name']
        header += f" | {short[:18]:>18}"
    header += f" | {'Ens(均等)':>10} | {'Ens(最適)':>10}"
    print(header)
    print("-" * len(header))

    fold_rmse_table = []
    for f_idx, (_, te) in enumerate(folds):
        sp = sp_names[f_idx]
        row = {"樹種": sp, "n_samples": len(te)}

        # 各モデル単体
        model_rmses = []
        row_str = f"{sp:<12}"
        for m_idx, cfg in enumerate(configs):
            r = rmse(y[te], all_preds[m_idx][f_idx])
            short = cfg['name'].split(': ')[1] if ': ' in cfg['name'] else cfg['name']
            row[f"M{m_idx+1}"] = r
            model_rmses.append(r)
            row_str += f" | {r:>18.2f}"

        # 均等アンサンブル
        ens_equal = np.mean([all_preds[m][f_idx] for m in range(n_models)], axis=0)
        r_equal = rmse(y[te], np.clip(ens_equal, 0, 300))
        row["Ens_均等"] = r_equal
        row_str += f" | {r_equal:>10.2f}"

        # 最適重みアンサンブル
        ens_opt = sum(opt_w[m] * all_preds[m][f_idx] for m in range(n_models))
        r_opt = rmse(y[te], np.clip(ens_opt, 0, 300))
        row["Ens_最適"] = r_opt
        row_str += f" | {r_opt:>10.2f}"

        tag = " ※参考" if sp == "ベイスギ" else ""
        print(row_str + tag)

        fold_rmse_table.append(row)

    # 平均行
    print("-" * len(header))
    avg_str = f"{'平均':<12}"
    for m_idx in range(n_models):
        avg_r = np.mean([row[f"M{m_idx+1}"] for row in fold_rmse_table])
        avg_str += f" | {avg_r:>18.2f}"
    avg_eq = np.mean([row["Ens_均等"] for row in fold_rmse_table])
    avg_opt = np.mean([row["Ens_最適"] for row in fold_rmse_table])
    avg_str += f" | {avg_eq:>10.2f} | {avg_opt:>10.2f}"
    print(avg_str)

    # ベイスギ除外平均
    non_bei = [row for row in fold_rmse_table if row["樹種"] != "ベイスギ"]
    avg2_str = f"{'平均(除ベイスギ)':<12}"
    for m_idx in range(n_models):
        avg_r = np.mean([row[f"M{m_idx+1}"] for row in non_bei])
        avg2_str += f" | {avg_r:>18.2f}"
    avg_eq2 = np.mean([row["Ens_均等"] for row in non_bei])
    avg_opt2 = np.mean([row["Ens_最適"] for row in non_bei])
    avg2_str += f" | {avg_eq2:>10.2f} | {avg_opt2:>10.2f}"
    print(avg2_str)

    # ===== 各foldの含水率統計 =====
    print("\n" + "=" * 80)
    print("各Foldの含水率統計")
    print("=" * 80)
    print(f"{'樹種':<12} | {'N':>4} | {'Min':>8} | {'Max':>8} | {'Mean':>8} | {'Std':>8} | {'Range':>8}")
    print("-" * 72)

    mc_stats = []
    for f_idx, (_, te) in enumerate(folds):
        sp = sp_names[f_idx]
        yf = y[te]
        stats = {
            "樹種": sp,
            "N": len(te),
            "Min": float(np.min(yf)),
            "Max": float(np.max(yf)),
            "Mean": float(np.mean(yf)),
            "Std": float(np.std(yf)),
            "Range": float(np.max(yf) - np.min(yf)),
        }
        mc_stats.append(stats)
        tag = " ※参考" if sp == "ベイスギ" else ""
        print(f"{sp:<12} | {stats['N']:>4} | {stats['Min']:>8.1f} | {stats['Max']:>8.1f} | "
              f"{stats['Mean']:>8.1f} | {stats['Std']:>8.1f} | {stats['Range']:>8.1f}{tag}")

    # 全体
    print("-" * 72)
    print(f"{'全体':<12} | {len(y):>4} | {np.min(y):>8.1f} | {np.max(y):>8.1f} | "
          f"{np.mean(y):>8.1f} | {np.std(y):>8.1f} | {np.max(y)-np.min(y):>8.1f}")

    # ===== 残差分析 =====
    print("\n" + "=" * 80)
    print("残差分析: 最適重みアンサンブルの各foldの外れサンプル")
    print("=" * 80)

    all_residuals = []
    for f_idx, (_, te) in enumerate(folds):
        sp = sp_names[f_idx]
        ens_pred = sum(opt_w[m] * all_preds[m][f_idx] for m in range(n_models))
        ens_pred = np.clip(ens_pred, 0, 300)
        residuals = y[te] - ens_pred
        abs_res = np.abs(residuals)

        # 各サンプルの残差を記録
        for i, idx in enumerate(te):
            all_residuals.append({
                "sample_idx": int(idx),
                "樹種": sp,
                "含水率_actual": float(y[idx]),
                "含水率_pred": float(ens_pred[i]),
                "residual": float(residuals[i]),
                "abs_residual": float(abs_res[i]),
            })

        # fold内で大きな残差のサンプルを表示
        n_show = min(3, len(te))
        worst_idx = np.argsort(abs_res)[::-1][:n_show]

        print(f"\n{sp} (N={len(te)}, fold RMSE={rmse(y[te], ens_pred):.2f}):")
        print(f"  残差統計: mean={np.mean(residuals):.2f}, std={np.std(residuals):.2f}, "
              f"median_abs={np.median(abs_res):.2f}, max_abs={np.max(abs_res):.2f}")

        for rank, wi in enumerate(worst_idx):
            actual = y[te[wi]]
            pred_val = ens_pred[wi]
            res_val = residuals[wi]
            # 各モデル単体の予測
            model_preds_str = ", ".join(
                [f"M{m+1}={all_preds[m][f_idx][wi]:.1f}" for m in range(n_models)]
            )
            print(f"  #{rank+1}: sample[{te[wi]}] actual={actual:.1f}, pred={pred_val:.1f}, "
                  f"residual={res_val:+.1f} ({model_preds_str})")

    # ===== 残差の全体傾向分析 =====
    print("\n" + "=" * 80)
    print("残差の全体傾向分析")
    print("=" * 80)

    res_df = pd.DataFrame(all_residuals)

    # 含水率帯ごとの残差
    bins = [0, 20, 40, 60, 80, 100, 150, 200, 300]
    res_df["mc_bin"] = pd.cut(res_df["含水率_actual"], bins=bins)
    bin_stats = res_df.groupby("mc_bin", observed=True).agg(
        N=("abs_residual", "count"),
        mean_abs_res=("abs_residual", "mean"),
        median_abs_res=("abs_residual", "median"),
        max_abs_res=("abs_residual", "max"),
        mean_res=("residual", "mean"),
    ).reset_index()

    print("\n含水率帯ごとの残差:")
    print(f"{'含水率帯':<16} | {'N':>4} | {'MAE':>8} | {'MedianAE':>10} | {'MaxAE':>8} | {'Bias':>8}")
    print("-" * 70)
    for _, row in bin_stats.iterrows():
        print(f"{str(row['mc_bin']):<16} | {row['N']:>4.0f} | {row['mean_abs_res']:>8.2f} | "
              f"{row['median_abs_res']:>10.2f} | {row['max_abs_res']:>8.2f} | {row['mean_res']:>+8.2f}")

    # 外れ値サンプル（|residual| > 30）
    outliers = res_df[res_df["abs_residual"] > 30].sort_values("abs_residual", ascending=False)
    print(f"\n外れ値サンプル（|残差| > 30）: {len(outliers)}件")
    if len(outliers) > 0:
        print(f"{'idx':>6} | {'樹種':<12} | {'実測':>8} | {'予測':>8} | {'残差':>8}")
        print("-" * 55)
        for _, row in outliers.head(20).iterrows():
            print(f"{row['sample_idx']:>6.0f} | {row['樹種']:<12} | {row['含水率_actual']:>8.1f} | "
                  f"{row['含水率_pred']:>8.1f} | {row['residual']:>+8.1f}")

    # ===== RMSE悪化要因のまとめ =====
    print("\n" + "=" * 80)
    print("まとめ: RMSE悪化の主要因")
    print("=" * 80)

    # fold RMSEでソート
    fold_ranking = sorted(fold_rmse_table, key=lambda x: x["Ens_最適"], reverse=True)
    print("\nRMSEワースト5 fold:")
    for i, row in enumerate(fold_ranking[:5]):
        sp = row["樹種"]
        tag = " ※参考値" if sp == "ベイスギ" else ""
        mc_stat = [s for s in mc_stats if s["樹種"] == sp][0]
        print(f"  {i+1}. {sp}: Ens_最適 RMSE={row['Ens_最適']:.2f}, "
              f"含水率 range=[{mc_stat['Min']:.1f}-{mc_stat['Max']:.1f}], N={mc_stat['N']}{tag}")

    # モデル間の相関（多様性チェック）
    print("\n各モデルの残差相関行列（多様性チェック）:")
    model_residuals = np.zeros((len(y), n_models))
    for m_idx in range(n_models):
        for f_idx, (_, te) in enumerate(folds):
            model_residuals[te, m_idx] = y[te] - all_preds[m_idx][f_idx]

    corr = np.corrcoef(model_residuals.T)
    short_names = [f"M{i+1}" for i in range(n_models)]
    print(f"{'':>4}", end="")
    for sn in short_names:
        print(f" | {sn:>6}", end="")
    print()
    for i, sn in enumerate(short_names):
        print(f"{sn:>4}", end="")
        for j in range(n_models):
            print(f" | {corr[i, j]:>6.3f}", end="")
        print()

    # CSVに保存
    fold_df = pd.DataFrame(fold_rmse_table)
    fold_df.to_csv(OUT_DIR / "issue76_fold_diagnosis.csv", index=False)
    res_df.to_csv(OUT_DIR / "issue76_residuals.csv", index=False)
    print(f"\n結果保存: {OUT_DIR / 'issue76_fold_diagnosis.csv'}")
    print(f"残差保存: {OUT_DIR / 'issue76_residuals.csv'}")
    print(f"\n総実行時間: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
