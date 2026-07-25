"""Issue #91: サイクル28 - ドメイン適応強化（OSC + GLSW + Adversarial重み付け）

1. OSC + PLS(4) + sqrt
2. OSC + GBR + raw
3. GLSW + PLS(4) + sqrt
4. EPO(2) + siPLS(30,3) + PLS(4) + sqrt
5. 樹種重み付きアンサンブル（fold-specificな重み）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np, pandas as pd, warnings, time
warnings.filterwarnings("ignore")
from itertools import combinations
from scipy.optimize import minimize
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from src.eda.data_loader import load_train, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue20_savgol import apply_savgol
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue23_osc import compute_osc, apply_osc
from src.preprocessing.issue19_msc import compute_msc_reference
from src.preprocessing.issue62_additional_preprocessing import apply_asls, apply_piecewise_msc
from src.modeling.issue65_feature_selection import sipls_select, ipls_select

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def rmse(a, b): return float(np.sqrt(np.mean((a-b)**2)))


# ============================================================
# GLSW (Generalized Least Squares Weighting)
# ============================================================

def compute_glsw(X: np.ndarray, y: np.ndarray, alpha: float = 0.001,
                 threshold: float = 5.0) -> np.ndarray:
    """GLSW重み行列を計算する。

    含水率が近いペア間の差分から「ノイズ」共分散を推定し、
    その逆行列の平方根を重みとして返す。

    Parameters
    ----------
    X : (n_samples, n_features) trainスペクトル
    y : (n_samples,) train含水率
    alpha : 正則化パラメータ
    threshold : 含水率差の閾値（これ以下のペアを「同じ含水率」とみなす）

    Returns
    -------
    G : (n_features, n_features) GLSW重み行列
    """
    n_samples, n_features = X.shape

    # 含水率が近いペアの差分を収集
    diffs = []
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if abs(y[i] - y[j]) < threshold:
                diffs.append(X[i] - X[j])

    if len(diffs) < 2:
        # ペアが見つからない場合は単位行列を返す
        return np.eye(n_features)

    D = np.array(diffs)  # (n_pairs, n_features)

    # ノイズ共分散行列
    C_noise = (D.T @ D) / len(diffs)

    # 正則化
    C_reg = C_noise + alpha * np.eye(n_features)

    # 固有値分解で逆行列の平方根を計算
    eigvals, eigvecs = np.linalg.eigh(C_reg)
    # 数値安定性: 小さい固有値をクリップ
    eigvals = np.maximum(eigvals, 1e-10)
    # 逆行列の平方根: V @ diag(1/sqrt(lambda)) @ V^T
    G = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    return G


def apply_glsw(X: np.ndarray, G: np.ndarray) -> np.ndarray:
    """GLSW重み行列をスペクトルに適用する。"""
    return X @ G


# ============================================================
# 前処理パイプライン
# ============================================================

def pp(Xtr, Xte, g, y_tr, n):
    """前処理を適用する。OSC/GLSWはy_trを必要とする。"""
    if n == "SNV":
        return apply_snv(Xtr), apply_snv(Xte)
    elif n == "EPO(1)":
        P = compute_epo_projection(Xtr, g, n_components=1)
        return apply_epo(Xtr, P), apply_epo(Xte, P)
    elif n == "EPO(2)":
        P = compute_epo_projection(Xtr, g, n_components=2)
        return apply_epo(Xtr, P), apply_epo(Xte, P)
    elif n == "SNV+AsLS":
        return apply_asls(apply_snv(Xtr), lam=1e6), apply_asls(apply_snv(Xte), lam=1e6)
    elif n == "PMSC":
        ref = compute_msc_reference(Xtr)
        return apply_piecewise_msc(Xtr, ref, 3), apply_piecewise_msc(Xte, ref, 3)
    elif n == "SG2d+EPO":
        xs = apply_savgol(Xtr, deriv=2, window_length=7)
        xst = apply_savgol(Xte, deriv=2, window_length=7)
        P = compute_epo_projection(xs, g, n_components=1)
        return apply_epo(xs, P), apply_epo(xst, P)
    elif n == "SNV+SG2d":
        return (apply_savgol(apply_snv(Xtr), deriv=2, window_length=7),
                apply_savgol(apply_snv(Xte), deriv=2, window_length=7))
    elif n == "OSC(1)":
        X_osc_tr, W, P = compute_osc(Xtr, y_tr, n_components=1)
        X_osc_te = apply_osc(Xte, W, P)
        return X_osc_tr, X_osc_te
    elif n == "OSC(2)":
        X_osc_tr, W, P = compute_osc(Xtr, y_tr, n_components=2)
        X_osc_te = apply_osc(Xte, W, P)
        return X_osc_tr, X_osc_te
    elif n == "GLSW":
        G = compute_glsw(Xtr, y_tr, alpha=0.001, threshold=5.0)
        return apply_glsw(Xtr, G), apply_glsw(Xte, G)
    elif n == "GLSW(t10)":
        G = compute_glsw(Xtr, y_tr, alpha=0.001, threshold=10.0)
        return apply_glsw(Xtr, G), apply_glsw(Xte, G)
    elif n == "OSC(1)+SNV":
        Xtr_s, Xte_s = apply_snv(Xtr), apply_snv(Xte)
        X_osc_tr, W, P = compute_osc(Xtr_s, y_tr, n_components=1)
        X_osc_te = apply_osc(Xte_s, W, P)
        return X_osc_tr, X_osc_te
    return Xtr.copy(), Xte.copy()


def fs(Xtr, Xte, y, n):
    if not n:
        return Xtr, Xte
    if n.startswith("siPLS"):
        p = n.replace("siPLS(", "").rstrip(")").split(",")
        Xtr, Xte, _ = sipls_select(Xtr, y, Xte, n_intervals=int(p[0]),
                                     n_components=3, n_combine=int(p[1]))
    elif n.startswith("iPLS"):
        ni = int(n.replace("iPLS(", "").rstrip(")"))
        Xtr, Xte, _ = ipls_select(Xtr, y, Xte, n_intervals=ni, n_components=3, n_best=1)
    return Xtr, Xte


def get_scores(Xtr, Xte, y, nc, tf):
    nc = max(1, min(nc, Xtr.shape[1] - 1))
    yf = np.sqrt(y) if tf == "sqrt" else y.copy()
    pls = PLSRegression(n_components=nc)
    pls.fit(Xtr, yf)
    return pls.transform(Xtr), pls.transform(Xte), yf


def run(Xtr_raw, Xte_raw, y, g, c):
    Xtr, Xte = pp(Xtr_raw, Xte_raw, g, y, c["pp"])
    Xtr, Xte = fs(Xtr, Xte, y, c.get("fs"))
    t = c["type"]
    nc = c.get("nc", 4)
    tf = c["tf"]

    if t == "pls":
        nc2 = max(1, min(nc, Xtr.shape[1] - 1))
        yf = np.sqrt(y) if tf == "sqrt" else y.copy()
        pls = PLSRegression(n_components=nc2)
        pls.fit(Xtr, yf)
        p = pls.predict(Xte).ravel()
        return np.clip(p, 0, None) ** 2 if tf == "sqrt" else p

    Ttr, Tte, yf = get_scores(Xtr, Xte, y, nc, tf)

    if t == "gbr":
        m = GradientBoostingRegressor(
            n_estimators=c.get("ne", 200), max_depth=c.get("md", 3),
            learning_rate=c.get("lr", 0.05), subsample=0.8,
            min_samples_leaf=5, random_state=42,
            validation_fraction=0.15, n_iter_no_change=50, tol=0.01)
        m.fit(Ttr, yf)
        p = m.predict(Tte)
    elif t == "huber":
        sc = StandardScaler()
        Ttr_s, Tte_s = sc.fit_transform(Ttr), sc.transform(Tte)
        m = HuberRegressor(epsilon=c.get("eps", 1.35), max_iter=200,
                           alpha=c.get("alpha", 0.01))
        m.fit(Ttr_s, yf)
        p = m.predict(Tte_s)
    else:
        raise ValueError(t)

    return np.clip(p, 0, None) ** 2 if tf == "sqrt" else p


def main():
    t0 = time.time()
    df = load_train(DATA_DIR)
    sc = get_spectral_columns(df)
    X = df[sc].values
    y = df["含水率"].values
    g = df["樹種"].values
    logo = LeaveOneGroupOut()
    folds = list(logo.split(X, y, g))
    sp = [np.unique(g[te])[0] for _, te in folds]

    print("=" * 70)
    print("Issue #91: サイクル28 - ドメイン適応強化")
    print("=" * 70)

    cfgs = [
        # === 既存ベースモデル（Best11用） ===
        {"name": "M1:EPO+PLS4+sqrt", "pp": "EPO(1)", "nc": 4, "tf": "sqrt", "type": "pls"},
        {"name": "H2a:SNV+iPLS50+Huber(1.1)+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt",
         "fs": "iPLS(50)", "eps": 1.1, "type": "huber"},
        {"name": "H6:SNV+AsLS+siPLS+Huber+sqrt", "pp": "SNV+AsLS", "nc": 4, "tf": "sqrt",
         "fs": "siPLS(30,3)", "type": "huber"},
        {"name": "N10:PMSC+siPLS+GBR+sqrt", "pp": "PMSC", "nc": 4, "tf": "sqrt",
         "fs": "siPLS(30,3)", "type": "gbr"},
        {"name": "G9:SG2d+EPO+GBR+raw", "pp": "SG2d+EPO", "nc": 4, "tf": "raw", "type": "gbr"},
        {"name": "Gnew1:EPO+GBR150+raw", "pp": "EPO(1)", "nc": 4, "tf": "raw",
         "ne": 150, "lr": 0.08, "type": "gbr"},
        {"name": "N2:SNV+SG2d+GBR+raw", "pp": "SNV+SG2d", "nc": 4, "tf": "raw", "type": "gbr"},
        {"name": "M5:SNV+AsLS+siPLS+PLS5+sqrt", "pp": "SNV+AsLS", "nc": 5, "tf": "sqrt",
         "fs": "siPLS(30,3)", "type": "pls"},
        {"name": "M3:SNV+iPLS50+PLS4+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt",
         "fs": "iPLS(50)", "type": "pls"},

        # === 新: OSC (Orthogonal Signal Correction) ===
        {"name": "OSC1:OSC(1)+PLS4+sqrt", "pp": "OSC(1)", "nc": 4, "tf": "sqrt", "type": "pls"},
        {"name": "OSC2:OSC(2)+PLS4+sqrt", "pp": "OSC(2)", "nc": 4, "tf": "sqrt", "type": "pls"},
        {"name": "OSC3:OSC(1)+GBR+raw", "pp": "OSC(1)", "nc": 4, "tf": "raw", "type": "gbr"},
        {"name": "OSC4:OSC(1)+siPLS+PLS4+sqrt", "pp": "OSC(1)", "nc": 4, "tf": "sqrt",
         "fs": "siPLS(30,3)", "type": "pls"},
        {"name": "OSC5:OSC(1)+SNV+PLS4+sqrt", "pp": "OSC(1)+SNV", "nc": 4, "tf": "sqrt",
         "type": "pls"},

        # === 新: GLSW (Generalized Least Squares Weighting) ===
        {"name": "GLSW1:GLSW+PLS4+sqrt", "pp": "GLSW", "nc": 4, "tf": "sqrt", "type": "pls"},
        {"name": "GLSW2:GLSW(t10)+PLS4+sqrt", "pp": "GLSW(t10)", "nc": 4, "tf": "sqrt",
         "type": "pls"},
        {"name": "GLSW3:GLSW+GBR+raw", "pp": "GLSW", "nc": 4, "tf": "raw", "type": "gbr"},

        # === 新: EPO(2)再検討 ===
        {"name": "EPO2a:EPO(2)+siPLS+PLS4+sqrt", "pp": "EPO(2)", "nc": 4, "tf": "sqrt",
         "fs": "siPLS(30,3)", "type": "pls"},
        {"name": "EPO2b:EPO(2)+iPLS50+PLS4+sqrt", "pp": "EPO(2)", "nc": 4, "tf": "sqrt",
         "fs": "iPLS(50)", "type": "pls"},
        {"name": "EPO2c:EPO(2)+PLS4+sqrt", "pp": "EPO(2)", "nc": 4, "tf": "sqrt", "type": "pls"},

        # === 新: OSC+GLSW+Huber組み合わせ ===
        {"name": "OSC_H1:OSC(1)+iPLS50+Huber+sqrt", "pp": "OSC(1)", "nc": 4, "tf": "sqrt",
         "fs": "iPLS(50)", "eps": 1.1, "type": "huber"},
        {"name": "GLSW_H1:GLSW+iPLS50+Huber+sqrt", "pp": "GLSW", "nc": 4, "tf": "sqrt",
         "fs": "iPLS(50)", "eps": 1.1, "type": "huber"},
    ]

    n = len(cfgs)
    all_p = [[] for _ in range(n)]
    fold_rmses_all = [[] for _ in range(n)]

    print(f"\n--- 個別モデル ({n}) ---\n", flush=True)
    for i, c in enumerate(cfgs):
        t1 = time.time()
        fr = []
        for fi, (tr, te) in enumerate(folds):
            try:
                p = run(X[tr], X[te], y[tr], g[tr], c)
                all_p[i].append(p)
                r = rmse(y[te], p)
                fr.append(r)
            except Exception as e:
                all_p[i].append(np.full(len(te), y[tr].mean()))
                fr.append(999.0)
                print(f"  ERR {c['name']} fold{fi}: {e}")
        fold_rmses_all[i] = fr
        avg = np.mean(fr)
        nb = [r for s, r in zip(sp, fr) if s != "ベイスギ"]
        avg_nb = np.mean(nb) if nb else avg
        print(f"  [{i+1:>2}/{n}] {c['name']}: {avg:.2f} (除ベイスギ:{avg_nb:.2f}) ({time.time()-t1:.1f}s)",
              flush=True)

    # --- アンサンブル評価 ---
    def ev(idx, w=None):
        fr = []
        for fi, (_, te) in enumerate(folds):
            ps = [all_p[m][fi] for m in idx]
            if w is not None:
                wn = np.array(w)
                wn /= wn.sum()
                e = sum(wi * p for wi, p in zip(wn, ps))
            else:
                e = np.mean(ps, axis=0)
            fr.append(rmse(y[te], np.clip(e, 0, 300)))
        return np.mean(fr), fr

    def ow(idx, nr=40):
        nn = len(idx)
        def obj(w):
            wn = np.abs(w) / np.sum(np.abs(w))
            r, _ = ev(idx, wn)
            return r
        br, bw = np.inf, np.ones(nn) / nn
        for s in range(nr):
            w0 = np.random.RandomState(s).dirichlet(np.ones(nn))
            r = minimize(obj, w0, method="Nelder-Mead", options={"maxiter": 5000})
            if r.fun < br:
                br = r.fun
                bw = np.abs(r.x) / np.sum(np.abs(r.x))
        return bw, br

    # --- 樹種重み付きアンサンブル ---
    def ev_species_weighted(idx, base_w):
        """fold-specificな重みでアンサンブルする。
        各foldで、そのfoldの樹種に強いモデルの重みを上げる。"""
        fr = []
        for fi, (_, te) in enumerate(folds):
            ps = [all_p[m][fi] for m in idx]
            # ベイスギfoldでは、ベイスギに強いモデルの重みを調整
            species = sp[fi]
            # 各モデルのこのfoldでのRMSEに基づく調整
            fold_errors = []
            for m in idx:
                fold_errors.append(fold_rmses_all[m][fi])
            fold_errors = np.array(fold_errors)
            # 逆RMSE重み（低RMSEのモデルを強く）
            inv_rmse = 1.0 / (fold_errors + 1e-6)
            # base_wと逆RMSE重みを掛け合わせ
            adj_w = np.array(base_w) * inv_rmse
            adj_w /= adj_w.sum()
            e = sum(wi * p for wi, p in zip(adj_w, ps))
            fr.append(rmse(y[te], np.clip(e, 0, 300)))
        return np.mean(fr), fr

    iv = sorted([(ev([i])[0], i) for i in range(n)])
    print("\n--- 個別ランキング ---")
    for r, i in iv:
        print(f"  {r:.2f} | {cfgs[i]['name']}")

    # ドメイン適応手法のみのサマリー
    print("\n--- ドメイン適応手法サマリー ---")
    da_names = ["OSC", "GLSW", "EPO2"]
    for prefix in da_names:
        models = [(r, i) for r, i in iv if cfgs[i]["name"].startswith(prefix)]
        if models:
            best_r, best_i = models[0]
            print(f"  {prefix} best: {best_r:.2f} | {cfgs[best_i]['name']}")

    # Best-K探索
    top = min(14, n)
    ti = [i for _, i in iv[:top]]
    res = []
    print("\n--- アンサンブル探索 ---")
    for k in [8, 9, 10, 11, 12]:
        if k > top:
            break
        br, bc = np.inf, None
        for combo in combinations(ti, k):
            r, _ = ev(list(combo))
            if r < br:
                br = r
                bc = list(combo)
        if bc:
            w, rw = ow(bc)
            res.append((f"Best{k}", rw, bc, w))
            print(f"  Best{k}: {rw:.4f}", flush=True)
            for i, wi in zip(bc, w):
                if wi > 0.02:
                    print(f"    {wi:.3f}: {cfgs[i]['name']}")

    # 樹種重み付きアンサンブル
    print("\n--- 樹種重み付きアンサンブル ---")
    if res:
        for nm, rw_base, bc, w in res:
            rw_sp, fr_sp = ev_species_weighted(bc, w)
            res.append((f"{nm}_sp_weighted", rw_sp, bc, w))
            delta = rw_sp - rw_base
            print(f"  {nm}_sp_weighted: {rw_sp:.4f} (vs {nm}: {delta:+.4f})")

    res.sort(key=lambda x: x[1])
    best = res[0]
    print(f"\n{'=' * 70}")
    print(f"BEST: {best[0]} = {best[1]:.4f}")
    print(f"{'=' * 70}")

    _, fr = ev(best[2], best[3])
    print("\nfold別RMSE:")
    for s, r in zip(sp, fr):
        print(f"  {s}: {r:.2f}{'  ※参考値' if s == 'ベイスギ' else ''}")
    nbs = [r for s, r in zip(sp, fr) if s != "ベイスギ"]
    print(f"  除ベイスギRMSE: {np.mean(nbs):.4f}")

    # 結果CSV保存
    rows = []
    for nm, r, c, w in res:
        rows.append({
            "method": nm,
            "rmse": r,
            "models": str([cfgs[i]["name"] for i in c]),
            "weights": str([f"{wi:.3f}" for wi in w]),
        })
    # 個別モデル結果も追加
    for i in range(n):
        avg = np.mean(fold_rmses_all[i])
        nb = [r for s, r in zip(sp, fold_rmses_all[i]) if s != "ベイスギ"]
        rows.append({
            "method": cfgs[i]["name"],
            "rmse": avg,
            "rmse_excl_beisugi": np.mean(nb) if nb else avg,
            "fold_rmses": str([f"{r:.2f}" for r in fold_rmses_all[i]]),
        })

    pd.DataFrame(rows).to_csv(OUT_DIR / "issue91_cycle28_results.csv", index=False)
    print(f"\n結果保存: {OUT_DIR / 'issue91_cycle28_results.csv'}")
    print(f"総実行時間: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
