"""Issue #89: サイクル26 - ベイスギ外挿対策 + 全体RMSE改善ベースライン構築

1. 既存Best11の完全再現（issue87_finalベース）
2. 新モデル候補: ElasticNet, KernelRidge, BayesianRidge, QuantileRegressor, GBR_deep
3. ベイスギ対策: 予測値クリッピング（train含水率min-10%~max+10%）
   + 外挿信頼度ベース重み調整
4. LOSO-CV + アンサンブル最適化（k=8~14）
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
from sklearn.linear_model import (
    HuberRegressor, ElasticNet, BayesianRidge, QuantileRegressor,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
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

PREV_BEST = 13.70  # 現在のベストRMSE


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ============================================================
# 前処理
# ============================================================
def preproc(X_tr, X_te, g, name):
    """前処理を適用。MSCリファレンスはtrainのみで計算。"""
    if name == "SNV":
        return apply_snv(X_tr), apply_snv(X_te)
    elif name == "EPO(1)":
        P = compute_epo_projection(X_tr, g, n_components=1)
        return apply_epo(X_tr, P), apply_epo(X_te, P)
    elif name == "SNV+AsLS(1e6)":
        return apply_asls(apply_snv(X_tr), lam=1e6), apply_asls(apply_snv(X_te), lam=1e6)
    elif name == "PMSC":
        ref = compute_msc_reference(X_tr)
        return apply_piecewise_msc(X_tr, ref, 3), apply_piecewise_msc(X_te, ref, 3)
    elif name == "SG2d+EPO(1)":
        xs = apply_savgol(X_tr, deriv=2, window_length=7)
        xst = apply_savgol(X_te, deriv=2, window_length=7)
        P = compute_epo_projection(xs, g, n_components=1)
        return apply_epo(xs, P), apply_epo(xst, P)
    elif name == "SNV+SG2d":
        return (apply_savgol(apply_snv(X_tr), deriv=2, window_length=7),
                apply_savgol(apply_snv(X_te), deriv=2, window_length=7))
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
    elif name.startswith("iPLS"):
        ni = int(name.replace("iPLS(", "").rstrip(")"))
        X_tr, X_te, _ = ipls_select(X_tr, y, X_te, n_intervals=ni, n_components=3, n_best=1)
    return X_tr, X_te


# ============================================================
# PLSスコア取得
# ============================================================
def get_pls_scores(X_tr, X_te, y, nc, tf):
    """PLSスコアを取得。PLSはfold内trainでfit。"""
    nc = max(1, min(nc, X_tr.shape[1] - 1))
    yf = np.sqrt(y) if tf == "sqrt" else y.copy()
    pls = PLSRegression(n_components=nc)
    pls.fit(X_tr, yf)
    return pls.transform(X_tr), pls.transform(X_te), pls, yf


# ============================================================
# 逆変換
# ============================================================
def inv_tf(pred, tf):
    if tf == "sqrt":
        return np.clip(pred, 0, None) ** 2
    return pred


# ============================================================
# モデル予測関数
# ============================================================
def pred_pls(Xtr, Xte, y, nc, tf):
    _, _, pls, yf = get_pls_scores(Xtr, Xte, y, nc, tf)
    return inv_tf(pls.predict(Xte).ravel(), tf)


def pred_gbr(Xtr, Xte, y, nc, tf, n_est=200, md=3, lr=0.05):
    """GBR: early stopping (patience=50) 必須。"""
    Ttr, Tte, _, yf = get_pls_scores(Xtr, Xte, y, nc, tf)
    gbr = GradientBoostingRegressor(
        n_estimators=n_est, max_depth=md, learning_rate=lr,
        subsample=0.8, min_samples_leaf=5, random_state=42,
        validation_fraction=0.15, n_iter_no_change=50, tol=0.01,
    )
    gbr.fit(Ttr, yf)
    return inv_tf(gbr.predict(Tte), tf)


def pred_huber(Xtr, Xte, y, nc, tf, eps=1.35, alpha=0.01):
    Ttr, Tte, _, yf = get_pls_scores(Xtr, Xte, y, nc, tf)
    sc = StandardScaler()
    Ttr_s, Tte_s = sc.fit_transform(Ttr), sc.transform(Tte)
    h = HuberRegressor(epsilon=eps, max_iter=200, alpha=alpha)
    h.fit(Ttr_s, yf)
    return inv_tf(h.predict(Tte_s), tf)


def pred_elasticnet(Xtr, Xte, y, nc, tf, alpha=0.01, l1_ratio=0.5):
    Ttr, Tte, _, yf = get_pls_scores(Xtr, Xte, y, nc, tf)
    sc = StandardScaler()
    Ttr_s, Tte_s = sc.fit_transform(Ttr), sc.transform(Tte)
    en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000, random_state=42)
    en.fit(Ttr_s, yf)
    return inv_tf(en.predict(Tte_s), tf)


def pred_kernel_ridge(Xtr, Xte, y, nc, tf, alpha=1.0, gamma=None):
    Ttr, Tte, _, yf = get_pls_scores(Xtr, Xte, y, nc, tf)
    sc = StandardScaler()
    Ttr_s, Tte_s = sc.fit_transform(Ttr), sc.transform(Tte)
    kr = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
    kr.fit(Ttr_s, yf)
    return inv_tf(kr.predict(Tte_s), tf)


def pred_bayesian_ridge(Xtr, Xte, y, nc, tf):
    Ttr, Tte, _, yf = get_pls_scores(Xtr, Xte, y, nc, tf)
    sc = StandardScaler()
    Ttr_s, Tte_s = sc.fit_transform(Ttr), sc.transform(Tte)
    br = BayesianRidge(max_iter=300)
    br.fit(Ttr_s, yf)
    return inv_tf(br.predict(Tte_s), tf)


def pred_quantile(Xtr, Xte, y, nc, tf, quantile=0.5, alpha=0.01):
    Ttr, Tte, _, yf = get_pls_scores(Xtr, Xte, y, nc, tf)
    sc = StandardScaler()
    Ttr_s, Tte_s = sc.fit_transform(Ttr), sc.transform(Tte)
    qr = QuantileRegressor(quantile=quantile, alpha=alpha, solver="highs")
    qr.fit(Ttr_s, yf)
    return inv_tf(qr.predict(Tte_s), tf)


# ============================================================
# モデル実行ディスパッチ
# ============================================================
def run_model(X_tr, X_te, y, g, cfg):
    Xtr, Xte = preproc(X_tr, X_te, g, cfg["pp"])
    Xtr, Xte = feat_sel(Xtr, Xte, y, cfg.get("fs"))
    t = cfg["type"]
    if t == "pls":
        return pred_pls(Xtr, Xte, y, cfg.get("nc", 4), cfg["tf"])
    elif t == "gbr":
        return pred_gbr(Xtr, Xte, y, cfg.get("nc", 4), cfg["tf"],
                         cfg.get("ne", 200), cfg.get("md", 3), cfg.get("lr", 0.05))
    elif t == "huber":
        return pred_huber(Xtr, Xte, y, cfg.get("nc", 4), cfg["tf"],
                          cfg.get("eps", 1.35), cfg.get("alpha", 0.01))
    elif t == "elasticnet":
        return pred_elasticnet(Xtr, Xte, y, cfg.get("nc", 4), cfg["tf"],
                                cfg.get("alpha", 0.01), cfg.get("l1_ratio", 0.5))
    elif t == "kernelridge":
        return pred_kernel_ridge(Xtr, Xte, y, cfg.get("nc", 4), cfg["tf"],
                                  cfg.get("kr_alpha", 1.0), cfg.get("gamma", None))
    elif t == "bayesianridge":
        return pred_bayesian_ridge(Xtr, Xte, y, cfg.get("nc", 4), cfg["tf"])
    elif t == "quantile":
        return pred_quantile(Xtr, Xte, y, cfg.get("nc", 4), cfg["tf"],
                              cfg.get("quantile", 0.5), cfg.get("alpha", 0.01))
    else:
        raise ValueError(f"Unknown model type: {t}")


# ============================================================
# ベイスギ対策: 予測値クリッピング
# ============================================================
def clip_to_train_range(preds, y_train, margin=0.10):
    """train含水率のmin-margin ~ max+margin の範囲にクリップ。

    ベイスギのような外挿foldで極端な予測を抑制する。
    """
    y_min = y_train.min()
    y_max = y_train.max()
    lower = y_min * (1 - margin)
    upper = y_max * (1 + margin)
    return np.clip(preds, lower, upper)


# ============================================================
# 外挿信頼度の推定（PLSスコア空間でのマハラノビス距離）
# ============================================================
def compute_extrapolation_score(T_train, T_test):
    """テストサンプルの外挿度（正規化マハラノビス距離）を計算。

    Returns: 各テストサンプルの外挿スコア（>1なら外挿、大きいほど外挿度が高い）
    """
    if T_train.ndim == 1:
        T_train = T_train.reshape(-1, 1)
        T_test = T_test.reshape(-1, 1)
    mean = T_train.mean(axis=0)
    cov = np.cov(T_train, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[cov]])
    cov += np.eye(cov.shape[0]) * 1e-6
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return np.ones(T_test.shape[0])
    diff = T_test - mean
    dist = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
    diff_tr = T_train - mean
    dist_tr = np.sqrt(np.sum(diff_tr @ cov_inv * diff_tr, axis=1))
    p95 = np.percentile(dist_tr, 95)
    return dist / max(p95, 1e-6)


# ============================================================
# モデル設定
# ============================================================
def get_model_configs():
    """既存Best11 + 新モデル候補を定義。"""
    cfgs = [
        # ============================================================
        # 既存Best11（issue87_final Best11構成）
        # ============================================================
        {"name": "M1:EPO+PLS4+sqrt",
         "pp": "EPO(1)", "nc": 4, "tf": "sqrt", "type": "pls"},
        {"name": "M3:SNV+iPLS50+PLS4+sqrt",
         "pp": "SNV", "nc": 4, "tf": "sqrt", "fs": "iPLS(50)", "type": "pls"},
        {"name": "M5:SNV+AsLS+siPLS+PLS5+sqrt",
         "pp": "SNV+AsLS(1e6)", "nc": 5, "tf": "sqrt", "fs": "siPLS(30,3)", "type": "pls"},
        {"name": "C1:SNV+iPLS50+PLS5+sqrt",
         "pp": "SNV", "nc": 5, "tf": "sqrt", "fs": "iPLS(50)", "type": "pls"},
        {"name": "H2:SNV+iPLS50+Huber+sqrt",
         "pp": "SNV", "nc": 4, "tf": "sqrt", "fs": "iPLS(50)", "type": "huber"},
        {"name": "H2a:SNV+iPLS50+Huber(1.1)+sqrt",
         "pp": "SNV", "nc": 4, "tf": "sqrt", "fs": "iPLS(50)", "eps": 1.1, "type": "huber"},
        {"name": "H5:PMSC+siPLS+Huber+sqrt",
         "pp": "PMSC", "nc": 4, "tf": "sqrt", "fs": "siPLS(30,3)", "type": "huber"},
        {"name": "H6:SNV+AsLS+siPLS+Huber+sqrt",
         "pp": "SNV+AsLS(1e6)", "nc": 4, "tf": "sqrt", "fs": "siPLS(30,3)", "type": "huber"},
        {"name": "N10:PMSC+siPLS+GBR+sqrt",
         "pp": "PMSC", "nc": 4, "tf": "sqrt", "fs": "siPLS(30,3)", "type": "gbr"},
        {"name": "G9:SG2d+EPO+GBR+raw",
         "pp": "SG2d+EPO(1)", "nc": 4, "tf": "raw", "type": "gbr"},
        {"name": "Gnew1:EPO+GBR150+raw",
         "pp": "EPO(1)", "nc": 4, "tf": "raw", "ne": 150, "lr": 0.08, "type": "gbr"},
        {"name": "N2:SNV+SG2d+GBR+raw",
         "pp": "SNV+SG2d", "nc": 4, "tf": "raw", "type": "gbr"},

        # ============================================================
        # 新モデル候補
        # ============================================================
        # (a) ElasticNet: SNV+iPLS(50)+PLS(4~6)+ElasticNet + sqrt
        {"name": "EN1:SNV+iPLS50+PLS4+ElasticNet+sqrt",
         "pp": "SNV", "nc": 4, "tf": "sqrt", "fs": "iPLS(50)",
         "alpha": 0.01, "l1_ratio": 0.5, "type": "elasticnet"},
        {"name": "EN2:SNV+iPLS50+PLS5+ElasticNet+sqrt",
         "pp": "SNV", "nc": 5, "tf": "sqrt", "fs": "iPLS(50)",
         "alpha": 0.05, "l1_ratio": 0.3, "type": "elasticnet"},
        {"name": "EN3:SNV+iPLS50+PLS6+ElasticNet+sqrt",
         "pp": "SNV", "nc": 6, "tf": "sqrt", "fs": "iPLS(50)",
         "alpha": 0.1, "l1_ratio": 0.7, "type": "elasticnet"},

        # (b) KernelRidge: EPO+PLS(4)+KernelRidge(RBF) + sqrt
        {"name": "KR1:EPO+PLS4+KernelRidge+sqrt",
         "pp": "EPO(1)", "nc": 4, "tf": "sqrt",
         "kr_alpha": 1.0, "gamma": None, "type": "kernelridge"},
        {"name": "KR2:EPO+PLS4+KernelRidge(g0.1)+sqrt",
         "pp": "EPO(1)", "nc": 4, "tf": "sqrt",
         "kr_alpha": 1.0, "gamma": 0.1, "type": "kernelridge"},

        # (c) BayesianRidge: SNV+siPLS(30,3)+PLS(4)+BayesianRidge + sqrt
        {"name": "BR1:SNV+siPLS30_3+PLS4+BayesianRidge+sqrt",
         "pp": "SNV", "nc": 4, "tf": "sqrt", "fs": "siPLS(30,3)",
         "type": "bayesianridge"},
        {"name": "BR2:EPO+PLS4+BayesianRidge+sqrt",
         "pp": "EPO(1)", "nc": 4, "tf": "sqrt",
         "type": "bayesianridge"},

        # (d) QuantileRegressor: SNV+PLS(4)+QuantileRegressor(quantile=0.5) + sqrt
        {"name": "QR1:SNV+PLS4+Quantile(0.5)+sqrt",
         "pp": "SNV", "nc": 4, "tf": "sqrt",
         "quantile": 0.5, "alpha": 0.01, "type": "quantile"},

        # (e) GBR_deep: EPO+PLS(6)+GBR(n_estimators=500, max_depth=4, lr=0.03) + raw
        {"name": "GBR_deep:EPO+PLS6+GBR500d4+raw",
         "pp": "EPO(1)", "nc": 6, "tf": "raw",
         "ne": 500, "md": 4, "lr": 0.03, "type": "gbr"},
    ]
    return cfgs


# ============================================================
# アンサンブル評価
# ============================================================
def evaluate(all_preds, folds, y, idx, w=None):
    """指定モデルインデックスのアンサンブルRMSEをfold別に計算。"""
    fold_rmses = []
    for fi, (_, te) in enumerate(folds):
        preds = [all_preds[m][fi] for m in idx]
        if w is not None:
            wn = np.array(w)
            wn = wn / wn.sum()
            ens = sum(wi * p for wi, p in zip(wn, preds))
        else:
            ens = np.mean(preds, axis=0)
        fold_rmses.append(rmse(y[te], np.clip(ens, 0, 300)))
    return np.mean(fold_rmses), fold_rmses


def opt_weights(all_preds, folds, y, idx, n_restarts=40):
    """Nelder-Mead重み最適化。"""
    nn = len(idx)

    def obj(w):
        wn = np.abs(w) / np.sum(np.abs(w))
        r, _ = evaluate(all_preds, folds, y, idx, wn)
        return r

    best_r, best_w = np.inf, np.ones(nn) / nn
    for s in range(n_restarts):
        w0 = np.random.RandomState(s).dirichlet(np.ones(nn))
        res = minimize(obj, w0, method="Nelder-Mead", options={"maxiter": 5000})
        if res.fun < best_r:
            best_r = res.fun
            best_w = np.abs(res.x) / np.sum(np.abs(res.x))
    return best_w, best_r


# ============================================================
# ベイスギ対策付きアンサンブル評価
# ============================================================
def evaluate_with_clipping(all_preds, all_preds_clipped, folds, y, species,
                           idx, w=None):
    """ベイスギfoldではクリップ済み予測を使用するアンサンブル評価。"""
    fold_rmses = []
    for fi, (_, te) in enumerate(folds):
        sp = np.unique(species[te])[0]
        src = all_preds_clipped if sp == "ベイスギ" else all_preds
        preds = [src[m][fi] for m in idx]
        if w is not None:
            wn = np.array(w)
            wn = wn / wn.sum()
            ens = sum(wi * p for wi, p in zip(wn, preds))
        else:
            ens = np.mean(preds, axis=0)
        fold_rmses.append(rmse(y[te], np.clip(ens, 0, 300)))
    return np.mean(fold_rmses), fold_rmses


def opt_weights_with_clipping(all_preds, all_preds_clipped, folds, y, species,
                               idx, n_restarts=40):
    """ベイスギ対策付きNelder-Mead重み最適化。"""
    nn = len(idx)

    def obj(w):
        wn = np.abs(w) / np.sum(np.abs(w))
        r, _ = evaluate_with_clipping(
            all_preds, all_preds_clipped, folds, y, species, idx, wn
        )
        return r

    best_r, best_w = np.inf, np.ones(nn) / nn
    for s in range(n_restarts):
        w0 = np.random.RandomState(s).dirichlet(np.ones(nn))
        res = minimize(obj, w0, method="Nelder-Mead", options={"maxiter": 5000})
        if res.fun < best_r:
            best_r = res.fun
            best_w = np.abs(res.x) / np.sum(np.abs(res.x))
    return best_w, best_r


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
    print("Issue #89: サイクル26 - ベイスギ外挿対策 + 全体RMSE改善")
    print("=" * 70)

    cfgs = get_model_configs()
    n = len(cfgs)
    all_preds = [[] for _ in range(n)]
    all_preds_clipped = [[] for _ in range(n)]

    # --- ベイスギfoldインデックスを特定 ---
    beisugi_fold_idx = None
    for fi, sp in enumerate(species_list):
        if sp == "ベイスギ":
            beisugi_fold_idx = fi
            break

    print(f"\n--- 個別モデル評価 ({n}) ---\n", flush=True)
    for i, cfg in enumerate(cfgs):
        t1 = time.time()
        fold_rmses = []
        fold_rmses_clipped = []
        for fi, (tr, te) in enumerate(folds):
            try:
                pred = run_model(X[tr], X[te], y[tr], g[tr], cfg)
                all_preds[i].append(pred)
                fold_rmses.append(rmse(y[te], pred))

                # ベイスギ対策: train含水率の min*(1-0.1) ~ max*(1+0.1) にクリップ
                pred_clipped = clip_to_train_range(pred, y[tr], margin=0.10)
                all_preds_clipped[i].append(pred_clipped)
                fold_rmses_clipped.append(rmse(y[te], pred_clipped))
            except Exception as e:
                fallback = np.full(len(te), y[tr].mean())
                all_preds[i].append(fallback)
                all_preds_clipped[i].append(fallback)
                fold_rmses.append(999.0)
                fold_rmses_clipped.append(999.0)
                print(f"  ERR {cfg['name']} fold{fi}: {e}")

        r_mean = np.mean(fold_rmses)
        r_clip = np.mean(fold_rmses_clipped)
        r_bei = fold_rmses[beisugi_fold_idx] if beisugi_fold_idx is not None else 0
        r_bei_clip = fold_rmses_clipped[beisugi_fold_idx] if beisugi_fold_idx is not None else 0
        non_bei = [r for j, r in enumerate(fold_rmses) if j != beisugi_fold_idx]
        clip_info = ""
        if abs(r_bei - r_bei_clip) > 0.01:
            clip_info = f" bei_clip:{r_bei_clip:.1f}"
        print(
            f"  [{i+1:>2}/{n}] {cfg['name']}: {r_mean:.2f} "
            f"(bei={r_bei:.1f}{clip_info}, 除bei={np.mean(non_bei):.2f}) "
            f"[{time.time()-t1:.1f}s]",
            flush=True,
        )

    # ============================================================
    # 個別モデルランキング
    # ============================================================
    indiv = sorted([(evaluate(all_preds, folds, y, [i])[0], i) for i in range(n)])
    print("\n--- 個別ランキング ---")
    for r, i in indiv:
        print(f"  {r:.2f} | {cfgs[i]['name']}")

    # ============================================================
    # 通常アンサンブル最適化
    # ============================================================
    print("\n--- アンサンブル最適化（通常） ---", flush=True)
    top = min(16, n)
    top_idx = [i for _, i in indiv[:top]]
    results_normal = []
    for k in range(8, 15):
        if k > top:
            break
        best_r, best_combo = np.inf, None
        for combo in combinations(top_idx, k):
            r, _ = evaluate(all_preds, folds, y, list(combo))
            if r < best_r:
                best_r = r
                best_combo = list(combo)
        if best_combo:
            w, rw = opt_weights(all_preds, folds, y, best_combo)
            _, fr = evaluate(all_preds, folds, y, best_combo, w)
            results_normal.append((f"通常Best{k}", rw, best_combo, w, fr))
            print(f"  通常Best{k}: {rw:.4f}", flush=True)
            for ci, wi in zip(best_combo, w):
                if wi > 0.02:
                    print(f"    {wi:.3f}: {cfgs[ci]['name']}")

    # ============================================================
    # ベイスギ対策付きアンサンブル最適化
    # ============================================================
    print("\n--- アンサンブル最適化（ベイスギ対策: クリッピング） ---", flush=True)
    results_clip = []
    for k in range(8, 15):
        if k > top:
            break
        best_r, best_combo = np.inf, None
        for combo in combinations(top_idx, k):
            r, _ = evaluate_with_clipping(
                all_preds, all_preds_clipped, folds, y, g, list(combo)
            )
            if r < best_r:
                best_r = r
                best_combo = list(combo)
        if best_combo:
            w, rw = opt_weights_with_clipping(
                all_preds, all_preds_clipped, folds, y, g, best_combo
            )
            _, fr = evaluate_with_clipping(
                all_preds, all_preds_clipped, folds, y, g, best_combo, w
            )
            results_clip.append((f"clip_Best{k}", rw, best_combo, w, fr))
            print(f"  clip_Best{k}: {rw:.4f}", flush=True)
            for ci, wi in zip(best_combo, w):
                if wi > 0.02:
                    print(f"    {wi:.3f}: {cfgs[ci]['name']}")

    # ============================================================
    # 結果統合・比較
    # ============================================================
    all_results = results_normal + results_clip
    all_results.sort(key=lambda x: x[1])
    best = all_results[0] if all_results else None

    print(f"\n{'='*70}")
    if best:
        print(f"BEST: {best[0]} = {best[1]:.4f}")
        print(f"前ベスト: {PREV_BEST:.4f}, 改善: {PREV_BEST - best[1]:+.4f}")
    print(f"{'='*70}")

    # fold別RMSE
    if best:
        print("\n--- fold別RMSE ---")
        beisugi_rmse = None
        non_beisugi_rmses = []
        for sp, r in zip(species_list, best[4]):
            marker = " ※外挿" if sp == "ベイスギ" else ""
            print(f"  {sp}: {r:.2f}{marker}")
            if sp == "ベイスギ":
                beisugi_rmse = r
            else:
                non_beisugi_rmses.append(r)
        print(f"\n  全体RMSE: {np.mean(best[4]):.4f}")
        print(f"  除ベイスギRMSE: {np.mean(non_beisugi_rmses):.4f}")
        if beisugi_rmse is not None:
            print(f"  ベイスギRMSE: {beisugi_rmse:.4f}")

    # ============================================================
    # 結果保存
    # ============================================================
    rows = []
    for nm, r, combo, w, fr in all_results:
        bei_idx = species_list.index("ベイスギ") if "ベイスギ" in species_list else None
        rows.append({
            "method": nm,
            "rmse": r,
            "beisugi_rmse": fr[bei_idx] if bei_idx is not None else None,
            "rmse_ex_beisugi": np.mean([
                ri for j, ri in enumerate(fr) if j != bei_idx
            ]),
            "models": str([cfgs[ci]["name"] for ci in combo]),
            "weights": str([f"{wi:.4f}" for wi in w]),
        })
    df_out = pd.DataFrame(rows)
    out_path = OUT_DIR / "issue89_cycle26_results.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n結果保存: {out_path}")
    print(f"総実行時間: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
