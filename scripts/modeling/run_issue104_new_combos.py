"""Issue #104: 未試行の前処理×回帰の網羅探索

新前処理: Wavelet, EMSC, Detrending (OSCは計算量大で除外)
新回帰: SVR(RBF), RandomForest, BayesianRidge
前処理キャッシュで高速化。参照モデル（高速サブセット）と新モデルを評価。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np, pandas as pd, warnings, time
warnings.filterwarnings("ignore")
from itertools import combinations
from scipy.optimize import minimize
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor, BayesianRidge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from src.eda.data_loader import load_train, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue20_savgol import apply_savgol
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue19_msc import compute_msc_reference
from src.preprocessing.issue62_additional_preprocessing import apply_piecewise_msc

WAVELET_OK = False
try:
    from src.preprocessing.issue39_wavelet_transform import wavelet_denoise
    WAVELET_OK = True
except Exception as e:
    print(f"[SKIP] Wavelet: {e}")

EMSC_OK = False
try:
    from src.preprocessing.issue25_emsc import apply_emsc
    EMSC_OK = True
except Exception as e:
    print(f"[SKIP] EMSC: {e}")

DETREND_OK = False
try:
    from src.preprocessing.issue24_detrending import apply_detrending
    DETREND_OK = True
except Exception as e:
    print(f"[SKIP] Detrend: {e}")

print(f"Import: Wavelet={WAVELET_OK}, EMSC={EMSC_OK}, Detrend={DETREND_OK}")

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

# ============================================================
# 前処理キャッシュ
# ============================================================
_pp_cache = {}

def do_preproc(X_tr, X_te, g, y_tr, name, fi):
    key = (name, fi)
    if key in _pp_cache:
        return _pp_cache[key]
    if name == "SNV":
        r = apply_snv(X_tr), apply_snv(X_te)
    elif name == "EPO(1)":
        P = compute_epo_projection(X_tr, g, n_components=1)
        r = apply_epo(X_tr, P), apply_epo(X_te, P)
    elif name == "PMSC":
        ref = compute_msc_reference(X_tr)
        r = apply_piecewise_msc(X_tr, ref, 3), apply_piecewise_msc(X_te, ref, 3)
    elif name == "SG2d+EPO(1)":
        xs = apply_savgol(X_tr, deriv=2, window_length=7)
        xst = apply_savgol(X_te, deriv=2, window_length=7)
        P = compute_epo_projection(xs, g, n_components=1)
        r = apply_epo(xs, P), apply_epo(xst, P)
    elif name == "SNV+SG2d":
        r = apply_savgol(apply_snv(X_tr), deriv=2, window_length=7), apply_savgol(apply_snv(X_te), deriv=2, window_length=7)
    elif name == "Wavelet":
        r = wavelet_denoise(X_tr), wavelet_denoise(X_te)
    elif name == "SNV+Wavelet":
        r = wavelet_denoise(apply_snv(X_tr)), wavelet_denoise(apply_snv(X_te))
    elif name == "EMSC":
        ref = compute_msc_reference(X_tr)
        r = apply_emsc(X_tr, ref), apply_emsc(X_te, ref)
    elif name == "SNV+EMSC":
        s1, s2 = apply_snv(X_tr), apply_snv(X_te)
        ref = compute_msc_reference(s1)
        r = apply_emsc(s1, ref), apply_emsc(s2, ref)
    elif name == "Detrend":
        r = apply_detrending(X_tr), apply_detrending(X_te)
    elif name == "SNV+Detrend":
        r = apply_detrending(apply_snv(X_tr)), apply_detrending(apply_snv(X_te))
    elif name == "Wavelet+EPO(1)":
        Xw1, Xw2 = wavelet_denoise(X_tr), wavelet_denoise(X_te)
        P = compute_epo_projection(Xw1, g, n_components=1)
        r = apply_epo(Xw1, P), apply_epo(Xw2, P)
    elif name == "EMSC+EPO(1)":
        ref = compute_msc_reference(X_tr)
        Xe1, Xe2 = apply_emsc(X_tr, ref), apply_emsc(X_te, ref)
        P = compute_epo_projection(Xe1, g, n_components=1)
        r = apply_epo(Xe1, P), apply_epo(Xe2, P)
    elif name == "Detrend+EPO(1)":
        Xd1, Xd2 = apply_detrending(X_tr), apply_detrending(X_te)
        P = compute_epo_projection(Xd1, g, n_components=1)
        r = apply_epo(Xd1, P), apply_epo(Xd2, P)
    else:
        r = X_tr.copy(), X_te.copy()
    _pp_cache[key] = r
    return r

def get_pls_scores(X_tr, X_te, y, nc, tf):
    nc = max(1, min(nc, X_tr.shape[1] - 1))
    yf = np.sqrt(y) if tf == "sqrt" else y.copy()
    pls = PLSRegression(n_components=nc)
    pls.fit(X_tr, yf)
    return pls.transform(X_tr), pls.transform(X_te), pls, yf

def inv_tf(pred, tf):
    return np.clip(pred, 0, None) ** 2 if tf == "sqrt" else pred

def do_pls(Xtr, Xte, y, nc, tf):
    _, _, pls, _ = get_pls_scores(Xtr, Xte, y, nc, tf)
    return inv_tf(pls.predict(Xte).ravel(), tf)

def do_huber(Xtr, Xte, y, nc, tf, eps=1.35):
    Ttr, Tte, _, yf = get_pls_scores(Xtr, Xte, y, nc, tf)
    sc = StandardScaler(); Ttr_s, Tte_s = sc.fit_transform(Ttr), sc.transform(Tte)
    h = HuberRegressor(epsilon=eps, max_iter=200, alpha=0.01)
    h.fit(Ttr_s, yf)
    return inv_tf(h.predict(Tte_s), tf)

def do_gbr(Xtr, Xte, y, nc, tf, ne=200, md=3, lr=0.05):
    Ttr, Tte, _, yf = get_pls_scores(Xtr, Xte, y, nc, tf)
    gbr = GradientBoostingRegressor(n_estimators=ne, max_depth=md, learning_rate=lr,
        subsample=0.8, min_samples_leaf=5, random_state=42,
        validation_fraction=0.15, n_iter_no_change=50, tol=0.01)
    gbr.fit(Ttr, yf)
    return inv_tf(gbr.predict(Tte), tf)

def do_svr(Xtr, Xte, y, nc, tf, C=10.0):
    Ttr, Tte, _, yf = get_pls_scores(Xtr, Xte, y, nc, tf)
    sc = StandardScaler(); Ttr_s, Tte_s = sc.fit_transform(Ttr), sc.transform(Tte)
    svr = SVR(kernel="rbf", C=C, gamma="scale", epsilon=0.1)
    svr.fit(Ttr_s, yf)
    return inv_tf(svr.predict(Tte_s), tf)

def do_rf(Xtr, Xte, y, nc, tf, ne=200):
    Ttr, Tte, _, yf = get_pls_scores(Xtr, Xte, y, nc, tf)
    rf = RandomForestRegressor(n_estimators=ne, max_depth=None, min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(Ttr, yf)
    return inv_tf(rf.predict(Tte), tf)

def do_br(Xtr, Xte, y, nc, tf):
    Ttr, Tte, _, yf = get_pls_scores(Xtr, Xte, y, nc, tf)
    sc = StandardScaler(); Ttr_s, Tte_s = sc.fit_transform(Ttr), sc.transform(Tte)
    br = BayesianRidge(max_iter=300)
    br.fit(Ttr_s, yf)
    return inv_tf(br.predict(Tte_s), tf)

def run_model(X_tr, X_te, y, g, cfg, fi):
    Xtr, Xte = do_preproc(X_tr, X_te, g, y, cfg["pp"], fi)
    t = cfg["type"]; nc, tf = cfg.get("nc", 4), cfg["tf"]
    if t == "pls": return do_pls(Xtr, Xte, y, nc, tf)
    elif t == "huber": return do_huber(Xtr, Xte, y, nc, tf, cfg.get("eps", 1.35))
    elif t == "gbr": return do_gbr(Xtr, Xte, y, nc, tf, cfg.get("ne", 200), cfg.get("md", 3), cfg.get("lr", 0.05))
    elif t == "svr": return do_svr(Xtr, Xte, y, nc, tf, cfg.get("C", 10.0))
    elif t == "rf": return do_rf(Xtr, Xte, y, nc, tf, cfg.get("ne", 200))
    elif t == "br": return do_br(Xtr, Xte, y, nc, tf)
    raise ValueError(f"Unknown: {t}")

def build_configs():
    cfgs = []
    # --- 参照モデル（高速のみ: AsLS/siPLS/iPLS除外） ---
    refs = [
        ("R1:EPO+PLS4s", "EPO(1)", 4, "sqrt", "pls", {}),
        ("R2:SNV+PLS4s", "SNV", 4, "sqrt", "pls", {}),
        ("R3:SNV+Hub4s", "SNV", 4, "sqrt", "huber", {}),
        ("R4:PMSC+Hub4s", "PMSC", 4, "sqrt", "huber", {}),
        ("R5:PMSC+GBR4s", "PMSC", 4, "sqrt", "gbr", {}),
        ("R6:SG2dEPO+GBR4r", "SG2d+EPO(1)", 4, "raw", "gbr", {}),
        ("R7:EPO+GBR150r", "EPO(1)", 4, "raw", "gbr", {"ne": 150, "lr": 0.08}),
        ("R8:SNV+SG2d+GBR4r", "SNV+SG2d", 4, "raw", "gbr", {}),
        ("R9:EPO+Hub4s", "EPO(1)", 4, "sqrt", "huber", {}),
        ("R10:PMSC+PLS4s", "PMSC", 4, "sqrt", "pls", {}),
    ]
    n_ref = len(refs)
    for name, pp, nc, tf, tp, extra in refs:
        c = {"name": name, "pp": pp, "nc": nc, "tf": tf, "type": tp}
        c.update(extra)
        cfgs.append(c)

    # --- 新前処理 ---
    new_pps = []
    if WAVELET_OK:
        new_pps += [("Wav", "Wavelet"), ("SNV+Wav", "SNV+Wavelet"), ("Wav+EPO", "Wavelet+EPO(1)")]
    if EMSC_OK:
        new_pps += [("EMSC", "EMSC"), ("SNV+EMSC", "SNV+EMSC"), ("EMSC+EPO", "EMSC+EPO(1)")]
    if DETREND_OK:
        new_pps += [("Det", "Detrend"), ("SNV+Det", "SNV+Detrend"), ("Det+EPO", "Detrend+EPO(1)")]

    # 各新前処理 × [PLS, Huber, SVR] × nc=4 (nc=5は厳選のみ)
    for pp_label, pp_name in new_pps:
        cfgs.append({"name": f"N:{pp_label}+PLS4s", "pp": pp_name, "nc": 4, "tf": "sqrt", "type": "pls"})
        cfgs.append({"name": f"N:{pp_label}+Hub4s", "pp": pp_name, "nc": 4, "tf": "sqrt", "type": "huber"})
        cfgs.append({"name": f"N:{pp_label}+SVR4s", "pp": pp_name, "nc": 4, "tf": "sqrt", "type": "svr"})
        cfgs.append({"name": f"N:{pp_label}+GBR4r", "pp": pp_name, "nc": 4, "tf": "raw", "type": "gbr"})
        cfgs.append({"name": f"N:{pp_label}+RF4s", "pp": pp_name, "nc": 4, "tf": "sqrt", "type": "rf"})
        cfgs.append({"name": f"N:{pp_label}+BR4s", "pp": pp_name, "nc": 4, "tf": "sqrt", "type": "br"})

    # SVR/RF with existing preprocessing
    for pp_label, pp_name in [("SNV", "SNV"), ("EPO", "EPO(1)"), ("PMSC", "PMSC")]:
        cfgs.append({"name": f"N:{pp_label}+SVR4s", "pp": pp_name, "nc": 4, "tf": "sqrt", "type": "svr"})
        cfgs.append({"name": f"N:{pp_label}+RF4s", "pp": pp_name, "nc": 4, "tf": "sqrt", "type": "rf"})
        cfgs.append({"name": f"N:{pp_label}+BR4s", "pp": pp_name, "nc": 4, "tf": "sqrt", "type": "br"})

    return cfgs, n_ref

# ============================================================
# アンサンブル
# ============================================================
def eval_ens(all_p, folds, y, idx, w=None):
    frs = []
    for fi, (_, te) in enumerate(folds):
        ps = [all_p[m][fi] for m in idx]
        if w is not None:
            wn = np.array(w); wn /= wn.sum()
            e = sum(wi * p for wi, p in zip(wn, ps))
        else:
            e = np.mean(ps, axis=0)
        frs.append(rmse(y[te], np.clip(e, 0, 300)))
    return np.mean(frs), frs

def opt_w(all_p, folds, y, idx, nr=12):
    nn = len(idx)
    def obj(w):
        wn = np.abs(w)/np.sum(np.abs(w))
        return eval_ens(all_p, folds, y, idx, wn)[0]
    br, bw = np.inf, np.ones(nn)/nn
    for s in range(nr):
        w0 = np.random.RandomState(s).dirichlet(np.ones(nn))
        res = minimize(obj, w0, method="Nelder-Mead", options={"maxiter": 1500})
        if res.fun < br:
            br = res.fun
            bw = np.abs(res.x)/np.sum(np.abs(res.x))
    return bw, br

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
    print("Issue #104: 未試行の前処理×回帰の網羅探索")
    print("=" * 70)

    cfgs, n_ref = build_configs()
    n = len(cfgs)
    print(f"参照: {n_ref}, 新: {n - n_ref}, 合計: {n}")

    # 前処理プリキャッシュ
    pp_names = sorted(set(c["pp"] for c in cfgs))
    print(f"\n前処理プリキャッシュ ({len(pp_names)}種):", flush=True)
    for pp in pp_names:
        t1 = time.time()
        ok = True
        for fi, (tr, te) in enumerate(folds):
            try:
                do_preproc(X[tr], X[te], g[tr], y[tr], pp, fi)
            except Exception as e:
                print(f"  ERR {pp} fold{fi}: {e}")
                ok = False
                break
        if ok:
            print(f"  {pp}: {time.time()-t1:.1f}s", flush=True)

    # モデル評価
    all_p = [[] for _ in range(n)]
    rmses_all = []
    valid = []

    print(f"\n--- モデル評価 ---\n", flush=True)
    for i, cfg in enumerate(cfgs):
        t1 = time.time()
        frs = []; err = False
        for fi, (tr, te) in enumerate(folds):
            try:
                pred = run_model(X[tr], X[te], y[tr], g[tr], cfg, fi)
                all_p[i].append(pred)
                frs.append(rmse(y[te], pred))
            except Exception as e:
                fb = np.full(len(te), y[tr].mean())
                all_p[i].append(fb)
                frs.append(999.0)
                err = True
                if fi == 0: print(f"  ERR {cfg['name']}: {e}")
        rm = np.mean(frs)
        rmses_all.append(rm)
        if not err and rm < 100:
            valid.append(i)
        tag = "REF" if i < n_ref else "NEW"
        print(f"  [{i+1:>2}/{n}] [{tag}] {cfg['name']}: {rm:.2f} [{time.time()-t1:.1f}s]", flush=True)
        if time.time() - t0 > 420:
            print(f"\n  [TIMEOUT] 7分経過、残り{n-i-1}モデルスキップ")
            for j in range(i+1, n):
                for fi, (tr, te) in enumerate(folds):
                    all_p[j].append(np.full(len(te), y[tr].mean()))
                rmses_all.append(999.0)
            break

    # ランキング
    sa = sorted([(rmses_all[i], i) for i in valid])
    print(f"\n--- 参照モデル上位5 ---")
    for r, i in [(r, i) for r, i in sa if i < n_ref][:5]:
        print(f"  {r:.2f} | {cfgs[i]['name']}")

    print(f"\n--- 新モデル上位15 ---")
    for r, i in [(r, i) for r, i in sa if i >= n_ref][:15]:
        print(f"  {r:.2f} | {cfgs[i]['name']}")

    # 参照アンサンブル
    rv = [i for i in valid if i < n_ref]
    if len(rv) >= 3:
        _, r_ref = opt_w(all_p, folds, y, rv, 8)
        print(f"\n参照アンサンブル RMSE: {r_ref:.4f}")
    else:
        r_ref = 999.0

    # 全体アンサンブル
    print(f"\n--- アンサンブル最適化 ---\n", flush=True)
    top_n = min(16, len(sa))
    top_idx = [i for _, i in sa[:top_n]]
    best_ov = None
    for k in range(8, min(14, top_n + 1)):
        if time.time() - t0 > 460: print("  [TIMEOUT]"); break
        br, bc = np.inf, None
        for combo in combinations(top_idx, k):
            r, _ = eval_ens(all_p, folds, y, list(combo))
            if r < br: br = r; bc = list(combo)
        if bc:
            w, rw = opt_w(all_p, folds, y, bc, 10)
            _, fr = eval_ens(all_p, folds, y, bc, w)
            nn = [cfgs[c]["name"] for c in bc if c >= n_ref]
            mk = f" [NEW:{len(nn)}]" if nn else ""
            print(f"  Best{k}: {rw:.4f}{mk}", flush=True)
            for ci, wi in zip(bc, w):
                if wi > 0.02:
                    t2 = " *NEW*" if ci >= n_ref else ""
                    print(f"    {wi:.3f}: {cfgs[ci]['name']}{t2}")
            if best_ov is None or rw < best_ov[1]:
                best_ov = (f"Best{k}", rw, bc, w, fr, nn)

    # サマリ
    print(f"\n{'='*70}")
    print(f"参照アンサンブル RMSE: {r_ref:.4f}")
    if best_ov:
        print(f"ベストアンサンブル RMSE: {best_ov[1]:.4f} ({best_ov[0]})")
        print(f"改善(参照比): {r_ref - best_ov[1]:+.4f}")
        if best_ov[5]:
            print(f"新モデル貢献: {', '.join(best_ov[5])}")
        print(f"\nfold別RMSE:")
        for sp, r in zip(species_list, best_ov[4]):
            mk = " (外挿)" if sp == "ベイスギ" else ""
            print(f"  {sp}: {r:.2f}{mk}")
    print(f"{'='*70}")

    # CSV
    rows = [{"model": cfgs[i]["name"], "rmse": rmses_all[i],
             "is_new": "NEW" if i >= n_ref else "REF"} for _, i in sa]
    pd.DataFrame(rows).to_csv(OUT_DIR / "issue104_new_combos_results.csv", index=False)
    print(f"\n結果保存: {OUT_DIR / 'issue104_new_combos_results.csv'}")
    print(f"総実行時間: {time.time() - t0:.0f}s")

if __name__ == "__main__":
    main()
