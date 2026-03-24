"""Issue #104: LWPLS最適化実験

LWPLSのハイパーパラメータグリッドサーチ:
- 前処理: SNV, EPO(1)（AsLSは計算コスト高のため除外）
- PLS成分数: 4, 5
- 目的変数変換: sqrt, raw
- k (近傍数): 50, 100, 200, 500
- 特徴選択: なし, siPLS(30,3)

高速化: 前処理キャッシュ + PLSスコア空間でweighted線形回帰
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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut
from scipy.spatial.distance import cdist
from src.eda.data_loader import load_train, load_test, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.modeling.issue65_feature_selection import sipls_select

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PREV_BEST = 13.70


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ============================================================
# 前処理
# ============================================================
def preproc(X_tr, X_te, g_tr, name):
    if name == "SNV":
        return apply_snv(X_tr), apply_snv(X_te)
    elif name == "EPO(1)":
        P = compute_epo_projection(X_tr, g_tr, n_components=1)
        return apply_epo(X_tr, P), apply_epo(X_te, P)
    return X_tr.copy(), X_te.copy()


# ============================================================
# 特徴選択
# ============================================================
def feat_sel(X_tr, X_te, y, name):
    if not name:
        return X_tr, X_te
    if name.startswith("siPLS"):
        p = name.replace("siPLS(", "").rstrip(")").split(",")
        X_tr, X_te, _ = sipls_select(
            X_tr, y, X_te, n_intervals=int(p[0]), n_components=3, n_combine=int(p[1])
        )
    return X_tr, X_te


# ============================================================
# LWPLS: PLSスコア空間でweighted線形回帰
# ============================================================
def lwpls_predict(T_train, y_train, T_test, k=200, sigma_factor=1.0):
    """PLSスコア空間でのLWPLS。"""
    n_test = T_test.shape[0]
    predictions = np.zeros(n_test)

    dist_matrix = cdist(T_test, T_train, metric='euclidean')
    k_actual = min(k, T_train.shape[0])

    for i in range(n_test):
        distances = dist_matrix[i]
        nn_idx = np.argpartition(distances, k_actual)[:k_actual]

        T_nn = T_train[nn_idx]
        y_nn = y_train[nn_idx]
        d_nn = distances[nn_idx]

        sigma = np.median(d_nn) * sigma_factor + 1e-10
        weights = np.exp(-0.5 * (d_nn / sigma) ** 2)
        weights = weights / (weights.sum() + 1e-10)

        sqrt_w = np.sqrt(weights)
        T_w = T_nn * sqrt_w[:, None]
        y_w = y_nn * sqrt_w

        try:
            lr = LinearRegression()
            lr.fit(T_w, y_w)
            predictions[i] = lr.predict(T_test[i:i+1])[0]
        except Exception:
            predictions[i] = y_train.mean()

    return predictions


# ============================================================
# LWPLS完全版: raw空間でweighted PLS fit
# ============================================================
def lwpls_predict_full(X_train, y_train, X_test, T_train, T_test,
                       n_components=4, k=200, sigma_factor=1.0):
    """完全版LWPLS。PLS空間で距離、raw空間でweighted PLS fit。"""
    n_test = X_test.shape[0]
    predictions = np.zeros(n_test)

    dist_matrix = cdist(T_test, T_train, metric='euclidean')
    k_actual = min(k, X_train.shape[0])
    max_comp = min(n_components, X_train.shape[1])

    for i in range(n_test):
        distances = dist_matrix[i]
        nn_idx = np.argpartition(distances, k_actual)[:k_actual]

        X_nn = X_train[nn_idx]
        y_nn = y_train[nn_idx]
        d_nn = distances[nn_idx]

        sigma = np.median(d_nn) * sigma_factor + 1e-10
        weights = np.exp(-0.5 * (d_nn / sigma) ** 2)
        weights = weights / (weights.sum() + 1e-10)

        sqrt_w = np.sqrt(weights)
        X_weighted = X_nn * sqrt_w[:, None]
        y_weighted = y_nn * sqrt_w

        n_comp = min(max_comp, k_actual, X_nn.shape[1])
        n_comp = max(1, n_comp)
        pls = PLSRegression(n_components=n_comp)
        try:
            pls.fit(X_weighted, y_weighted)
            predictions[i] = pls.predict(X_test[i:i+1].copy()).ravel()[0]
        except Exception:
            predictions[i] = y_train.mean()

    return predictions


# ============================================================
# 目的変数変換
# ============================================================
def transform_y(y, tf):
    if tf == "sqrt":
        return np.sqrt(y)
    return y.copy()


def inv_transform(pred, tf):
    if tf == "sqrt":
        return np.clip(pred, 0, None) ** 2
    return pred


# ============================================================
# メイン
# ============================================================
def main():
    t0 = time.time()
    df = load_train(DATA_DIR)
    sc_cols = get_spectral_columns(df)
    X = df[sc_cols].values
    y = df["含水率"].values
    g = df["樹種"].values
    logo = LeaveOneGroupOut()
    folds = list(logo.split(X, y, g))
    species_list = [np.unique(g[te])[0] for _, te in folds]

    print("=" * 70)
    print("Issue #104: LWPLS最適化実験")
    print("=" * 70)

    # === 前処理キャッシュ ===
    pp_names = ["SNV", "EPO(1)"]
    print("\n--- 前処理キャッシュ構築 ---", flush=True)
    cache = {}
    for pp in pp_names:
        t1 = time.time()
        for fi, (tr, te) in enumerate(folds):
            X_tr_pp, X_te_pp = preproc(X[tr], X[te], g[tr], pp)
            cache[(pp, None, fi)] = (X_tr_pp, X_te_pp)
            X_tr_fs, X_te_fs = feat_sel(X_tr_pp, X_te_pp, y[tr], "siPLS(30,3)")
            cache[(pp, "siPLS(30,3)", fi)] = (X_tr_fs, X_te_fs)
        print(f"  {pp}: {time.time()-t1:.1f}s", flush=True)
    print(f"  キャッシュ構築完了: {time.time()-t0:.0f}s", flush=True)

    # === グリッド ===
    configs = []
    for pp in pp_names:
        for nc in [4, 5]:
            for tf in ["sqrt", "raw"]:
                for kk in [50, 100, 200, 500]:
                    configs.append({
                        "pp": pp, "nc": nc, "tf": tf, "k": kk,
                        "fs": None, "mode": "ultra",
                        "name": f"LW:{pp}/nc{nc}/{tf}/k{kk}",
                    })
    # siPLS
    for pp in pp_names:
        for nc in [4, 5]:
            for kk in [100, 200]:
                configs.append({
                    "pp": pp, "nc": nc, "tf": "sqrt", "k": kk,
                    "fs": "siPLS(30,3)", "mode": "ultra",
                    "name": f"LW:{pp}/siPLS/nc{nc}/sqrt/k{kk}",
                })

    print(f"\nPhase 1: 超高速スクリーニング ({len(configs)} 設定)")

    results = []
    for ci, cfg in enumerate(configs):
        t1 = time.time()
        fold_rmses = []
        fold_preds = {}

        for fi, (tr, te) in enumerate(folds):
            try:
                X_tr_fs, X_te_fs = cache[(cfg["pp"], cfg["fs"], fi)]
                y_tr = transform_y(y[tr], cfg["tf"])

                nc = min(cfg["nc"], X_tr_fs.shape[1] - 1, X_tr_fs.shape[0] - 1)
                nc = max(1, nc)
                pls = PLSRegression(n_components=nc)
                pls.fit(X_tr_fs, y_tr)
                T_tr = pls.transform(X_tr_fs)
                T_te = pls.transform(X_te_fs)

                pred = lwpls_predict(T_tr, y_tr, T_te, k=cfg["k"])
                pred = inv_transform(pred, cfg["tf"])
                pred = np.clip(pred, 0, 300)
                fold_preds[fi] = pred
                fold_rmses.append(rmse(y[te], pred))
            except Exception as e:
                fold_preds[fi] = np.full(len(te), y[tr].mean())
                fold_rmses.append(999.0)
                print(f"  ERR {cfg['name']} fold{fi}: {e}")

        r_mean = np.mean(fold_rmses)
        elapsed = time.time() - t1
        results.append({
            "name": cfg["name"],
            "rmse": r_mean,
            "fold_rmses": fold_rmses,
            "fold_preds": fold_preds,
            "config": cfg,
        })

        marker = " ***" if r_mean < PREV_BEST else ""
        print(
            f"  [{ci+1:>3}/{len(configs)}] {r_mean:.2f} | {cfg['name']} [{elapsed:.1f}s]{marker}",
            flush=True,
        )

        if time.time() - t0 > 420:
            print(f"\n  *** Phase1 7分制限: {ci+1}/{len(configs)}設定完了 ***")
            break

    # ============================================================
    # 結果まとめ
    # ============================================================
    results.sort(key=lambda x: x["rmse"])

    print(f"\n{'='*70}")
    print("LWPLS結果ランキング (Top 20)")
    print(f"{'='*70}")
    for i, r in enumerate(results[:20]):
        fold_str = " ".join([f"{fr:.1f}" for fr in r["fold_rmses"]])
        marker = " *** BETTER" if r["rmse"] < PREV_BEST else ""
        print(f"  {i+1:>2}. {r['rmse']:.2f} | {r['name']} ({fold_str}){marker}")

    print(f"\n既存Best11: {PREV_BEST:.2f}")

    best = results[0]
    print(f"\nベスト設定: {best['name']} (RMSE={best['rmse']:.2f})")
    print("  fold別RMSE:")
    non_bei = []
    for fi, sp in enumerate(species_list):
        if fi < len(best["fold_rmses"]):
            r = best["fold_rmses"][fi]
            marker = " (外挿)" if sp == "ベイスギ" else ""
            print(f"    {sp}: {r:.2f}{marker}")
            if sp != "ベイスギ":
                non_bei.append(r)
    if non_bei:
        print(f"  除ベイスギ平均RMSE: {np.mean(non_bei):.2f}")

    # ============================================================
    # ベスト設定で提出ファイル生成
    # ============================================================
    print("\n--- 提出ファイル生成 ---")
    bcfg = best["config"]

    df_test = load_test(DATA_DIR)
    sc_cols_test = get_spectral_columns(df_test)
    X_test_raw = df_test[sc_cols_test].values

    X_tr_pp, X_te_pp = preproc(X, X_test_raw, g, bcfg["pp"])
    X_tr_fs, X_te_fs = feat_sel(X_tr_pp, X_te_pp, y, bcfg["fs"])
    y_tr = transform_y(y, bcfg["tf"])

    nc = min(bcfg["nc"], X_tr_fs.shape[1] - 1, X_tr_fs.shape[0] - 1)
    nc = max(1, nc)
    pls_final = PLSRegression(n_components=nc)
    pls_final.fit(X_tr_fs, y_tr)
    T_tr_final = pls_final.transform(X_tr_fs)
    T_te_final = pls_final.transform(X_te_fs)

    if bcfg.get("mode") == "full":
        test_pred = lwpls_predict_full(
            X_tr_fs, y_tr, X_te_fs, T_tr_final, T_te_final,
            n_components=bcfg["nc"], k=bcfg["k"],
        )
    else:
        test_pred = lwpls_predict(T_tr_final, y_tr, T_te_final, k=bcfg["k"])

    test_pred = inv_transform(test_pred, bcfg["tf"])
    test_pred = np.clip(test_pred, 0, 300)

    sample_sub = pd.read_csv(DATA_DIR / "sample_submit.csv", header=None)
    sub = sample_sub.copy()
    sub[1] = sub[1].astype(float)
    sub.iloc[:, 1] = test_pred
    out_path = OUT_DIR.parent / "submission_v12_lwpls_best.csv"
    sub.to_csv(out_path, index=False, header=False)
    print(f"提出ファイル保存: {out_path}")
    print(f"予測値統計: mean={test_pred.mean():.1f}, std={test_pred.std():.1f}, "
          f"min={test_pred.min():.1f}, max={test_pred.max():.1f}")

    # ============================================================
    # 結果CSV保存
    # ============================================================
    rows = []
    for r in results:
        row = {
            "name": r["name"],
            "rmse": r["rmse"],
            "mode": r["config"].get("mode", ""),
            "pp": r["config"]["pp"],
            "nc": r["config"]["nc"],
            "tf": r["config"]["tf"],
            "k": r["config"]["k"],
            "fs": r["config"].get("fs", ""),
        }
        for fi, sp in enumerate(species_list):
            if fi < len(r["fold_rmses"]):
                row[f"fold_{sp}"] = r["fold_rmses"][fi]
        rows.append(row)
    df_results = pd.DataFrame(rows)
    result_path = OUT_DIR / "issue104_lwpls_results.csv"
    df_results.to_csv(result_path, index=False)
    print(f"結果CSV保存: {result_path}")

    print(f"\n総実行時間: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
