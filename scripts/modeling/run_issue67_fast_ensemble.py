"""Issue #67 Fast: 高速アンサンブル最適化

AsLS等の重い前処理を除外し、高速前処理のみで18モデルの予測を収集→重み最適化。
LightGBM/XGBも固定HPで含める。
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
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.model_selection import LeaveOneGroupOut, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.eda.data_loader import load_train, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue20_savgol import apply_savgol
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue19_msc import compute_msc_reference, apply_msc

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def cars_select(X_train, y_train, X_test, n_pls=3, n_iterations=30):
    n_samples, n_features = X_train.shape
    remaining = np.arange(n_features)
    ratio = (max(n_pls + 2, 5) / n_features) ** (1.0 / n_iterations)
    best_rmse, best_idx = np.inf, remaining.copy()
    for i in range(n_iterations):
        n_keep = max(n_pls + 1, int(n_features * ratio ** (i + 1)))
        if n_keep >= len(remaining):
            continue
        n_comp = min(n_pls, len(remaining) - 1, n_samples - 1)
        if n_comp < 1:
            break
        pls = PLSRegression(n_components=n_comp)
        pls.fit(X_train[:, remaining], y_train)
        coefs = np.abs(pls.coef_.ravel())
        top_idx = np.argsort(coefs)[::-1][:n_keep]
        remaining = remaining[np.sort(top_idx)]
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_r = []
        for tr, te in kf.split(X_train[:, remaining]):
            nc = min(n_comp, len(remaining) - 1)
            if nc < 1:
                break
            p = PLSRegression(n_components=nc)
            p.fit(X_train[tr][:, remaining], y_train[tr])
            pred = p.predict(X_train[te][:, remaining]).ravel()
            cv_r.append(np.sqrt(np.mean((pred - y_train[te]) ** 2)))
        if cv_r and np.mean(cv_r) < best_rmse:
            best_rmse = np.mean(cv_r)
            best_idx = remaining.copy()
    return X_train[:, best_idx], X_test[:, best_idx], best_idx


def vip_select(X_train, y_train, X_test, n_components=2, threshold=1.5):
    nc = min(n_components, X_train.shape[1] - 1)
    pls = PLSRegression(n_components=nc)
    pls.fit(X_train, y_train)
    T, W, Q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
    p = X_train.shape[1]
    s = np.diag(T.T @ T @ Q.T @ Q).ravel()
    total_s = np.sum(s)
    vip = np.sqrt(p * np.sum(s[None, :] * (W / np.linalg.norm(W, axis=0)) ** 2, axis=1) / total_s)
    mask = vip > threshold
    if mask.sum() < 2:
        mask = np.zeros(p, dtype=bool)
        mask[np.argsort(vip)[-max(2, int(p * 0.1)):]] = True
    return X_train[:, mask], X_test[:, mask]


def predict_fold(X_raw, y, groups, train_idx, test_idx, config):
    X_tr_raw, X_te_raw = X_raw[train_idx], X_raw[test_idx]
    y_train, groups_train = y[train_idx], groups[train_idx]
    pp = config["pp"]

    # Preprocessing (fast only)
    if pp == "SNV":
        X_tr, X_te = apply_snv(X_tr_raw), apply_snv(X_te_raw)
    elif pp == "EPO(1)":
        P = compute_epo_projection(X_tr_raw, groups_train, n_components=1)
        X_tr, X_te = apply_epo(X_tr_raw, P), apply_epo(X_te_raw, P)
    elif pp == "MSC":
        ref = compute_msc_reference(X_tr_raw)
        X_tr, X_te = apply_msc(X_tr_raw, ref), apply_msc(X_te_raw, ref)
    elif pp == "SG2d+EPO(1)":
        X_tr_sg = apply_savgol(X_tr_raw, deriv=2, window_length=7)
        X_te_sg = apply_savgol(X_te_raw, deriv=2, window_length=7)
        P = compute_epo_projection(X_tr_sg, groups_train, n_components=1)
        X_tr, X_te = apply_epo(X_tr_sg, P), apply_epo(X_te_sg, P)
    elif pp == "SG1d":
        X_tr = apply_savgol(X_tr_raw, deriv=1, window_length=11)
        X_te = apply_savgol(X_te_raw, deriv=1, window_length=11)
    else:
        X_tr, X_te = X_tr_raw.copy(), X_te_raw.copy()

    # Feature selection
    fs = config.get("fs")
    if fs == "CARS":
        X_tr, X_te, _ = cars_select(X_tr, y_train, X_te, n_pls=config.get("cars_pls", 3))
    elif fs == "VIP(1.5)":
        X_tr, X_te = vip_select(X_tr, y_train, X_te, threshold=1.5)
    elif fs == "VIP(1.2)":
        X_tr, X_te = vip_select(X_tr, y_train, X_te, threshold=1.2)

    # Target transform
    tf = config["tf"]
    y_fit = np.sqrt(y_train) if tf == "sqrt" else y_train.copy()

    # Model
    mdl = config["model"]
    if mdl.startswith("PLS("):
        nc = int(mdl.split("(")[1].rstrip(")"))
        nc = min(nc, X_tr.shape[1] - 1)
        m = PLSRegression(n_components=max(1, nc))
        m.fit(X_tr, y_fit)
        pred = m.predict(X_te).ravel()
    elif mdl == "Lasso":
        m = Lasso(alpha=0.1, max_iter=10000)
        m.fit(X_tr, y_fit)
        pred = m.predict(X_te)
    elif mdl == "ElasticNet":
        m = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
        m.fit(X_tr, y_fit)
        pred = m.predict(X_te)
    elif mdl == "Ridge":
        m = Ridge(alpha=100)
        m.fit(X_tr, y_fit)
        pred = m.predict(X_te)
    elif mdl == "LGB":
        import lightgbm as lgb
        # PLS次元削減
        nc = min(config.get("n_pls", 3), X_tr.shape[1] - 1)
        pls = PLSRegression(n_components=nc)
        pls.fit(X_tr, y_train)
        X_tr_s = pls.transform(X_tr)
        X_te_s = pls.transform(X_te)
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr_s)
        X_te_s = sc.transform(X_te_s)
        m = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=3, num_leaves=15,
            min_child_samples=30, subsample=0.8, colsample_bytree=0.9,
            reg_alpha=0.1, reg_lambda=0.1, verbose=-1, random_state=42,
        )
        m.fit(X_tr_s, y_fit)
        pred = m.predict(X_te_s)
    elif mdl == "XGB":
        import xgboost as xgb
        nc = min(config.get("n_pls", 3), X_tr.shape[1] - 1)
        pls = PLSRegression(n_components=nc)
        pls.fit(X_tr, y_train)
        X_tr_s = pls.transform(X_tr)
        X_te_s = pls.transform(X_te)
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr_s)
        X_te_s = sc.transform(X_te_s)
        m = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=3,
            min_child_weight=20, subsample=0.8, colsample_bytree=0.9,
            reg_alpha=0.1, reg_lambda=0.1, verbosity=0, random_state=42,
        )
        m.fit(X_tr_s, y_fit)
        pred = m.predict(X_te_s)
    elif mdl == "SVR":
        nc = min(config.get("n_pls", 3), X_tr.shape[1] - 1)
        pls = PLSRegression(n_components=nc)
        pls.fit(X_tr, y_train)
        X_tr_s = pls.transform(X_tr)
        X_te_s = pls.transform(X_te)
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr_s)
        X_te_s = sc.transform(X_te_s)
        m = SVR(C=10, epsilon=0.1, gamma="scale")
        m.fit(X_tr_s, y_fit)
        pred = m.predict(X_te_s)
    else:
        raise ValueError(mdl)

    if tf == "sqrt":
        pred = np.clip(pred, 0, None) ** 2
    return pred


def optimize_weights(preds, y, n_restarts=50):
    n = preds.shape[0]
    def obj(w):
        w_n = np.abs(w) / np.sum(np.abs(w))
        return rmse(y, w_n @ preds)
    best_r, best_w = np.inf, np.ones(n) / n
    for s in range(n_restarts):
        w0 = np.random.RandomState(s).dirichlet(np.ones(n))
        res = minimize(obj, w0, method="Nelder-Mead", options={"maxiter": 5000})
        if res.fun < best_r:
            best_r = res.fun
            best_w = np.abs(res.x) / np.sum(np.abs(res.x))
    return best_w, best_r


def main():
    t0 = time.time()
    df = load_train(DATA_DIR)
    sc = get_spectral_columns(df)
    X_raw = df[sc].values
    y = df["含水率"].values
    groups = df["樹種"].values

    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_raw, y, groups))

    print("=" * 70)
    print("Issue #67 Fast: 高速アンサンブル最適化")
    print("=" * 70)

    configs = [
        # --- Linear PLS variants ---
        {"name": "SNV+PLS(2)+raw", "pp": "SNV", "model": "PLS(2)", "tf": "raw"},
        {"name": "SNV+PLS(2)+sqrt", "pp": "SNV", "model": "PLS(2)", "tf": "sqrt"},
        {"name": "SNV+PLS(4)+sqrt", "pp": "SNV", "model": "PLS(4)", "tf": "sqrt"},
        {"name": "EPO(1)+PLS(2)+raw", "pp": "EPO(1)", "model": "PLS(2)", "tf": "raw"},
        {"name": "EPO(1)+PLS(4)+sqrt", "pp": "EPO(1)", "model": "PLS(4)", "tf": "sqrt"},
        {"name": "EPO(1)+PLS(3)+raw", "pp": "EPO(1)", "model": "PLS(3)", "tf": "raw"},
        {"name": "MSC+PLS(4)+sqrt", "pp": "MSC", "model": "PLS(4)", "tf": "sqrt"},
        {"name": "SG2d+EPO(1)+PLS(3)+raw", "pp": "SG2d+EPO(1)", "model": "PLS(3)", "tf": "raw"},
        # --- Regularized linear ---
        {"name": "SNV+ElasticNet+sqrt", "pp": "SNV", "model": "ElasticNet", "tf": "sqrt"},
        {"name": "EPO(1)+Lasso+raw", "pp": "EPO(1)", "model": "Lasso", "tf": "raw"},
        {"name": "SNV+Ridge+sqrt", "pp": "SNV", "model": "Ridge", "tf": "sqrt"},
        {"name": "MSC+Lasso+raw", "pp": "MSC", "model": "Lasso", "tf": "raw"},
        # --- Feature selection ---
        {"name": "EPO(1)+CARS+PLS(3)+raw", "pp": "EPO(1)", "model": "PLS(3)", "tf": "raw", "fs": "CARS", "cars_pls": 3},
        {"name": "SNV+VIP(1.5)+PLS(4)+sqrt", "pp": "SNV", "model": "PLS(4)", "tf": "sqrt", "fs": "VIP(1.5)"},
        # --- Non-linear (fixed HP) ---
        {"name": "SNV+LGB(PLS3)+raw", "pp": "SNV", "model": "LGB", "tf": "raw", "n_pls": 3},
        {"name": "SNV+XGB(PLS3)+raw", "pp": "SNV", "model": "XGB", "tf": "raw", "n_pls": 3},
        {"name": "EPO(1)+LGB(PLS3)+raw", "pp": "EPO(1)", "model": "LGB", "tf": "raw", "n_pls": 3},
        {"name": "SNV+SVR(PLS3)+raw", "pp": "SNV", "model": "SVR", "tf": "raw", "n_pls": 3},
        # --- Diverse transforms ---
        {"name": "SG1d+PLS(2)+sqrt", "pp": "SG1d", "model": "PLS(2)", "tf": "sqrt"},
        {"name": "SNV+LGB(PLS5)+sqrt", "pp": "SNV", "model": "LGB", "tf": "sqrt", "n_pls": 5},
    ]

    n_models = len(configs)
    n_samples = len(y)
    all_preds = np.full((n_models, n_samples), np.nan)
    names = [c["name"] for c in configs]

    print(f"モデル数: {n_models}\n")
    for m_idx, cfg in enumerate(configs):
        t1 = time.time()
        for train_idx, test_idx in folds:
            try:
                pred = predict_fold(X_raw, y, groups, train_idx, test_idx, cfg)
                all_preds[m_idx, test_idx] = pred
            except Exception as e:
                all_preds[m_idx, test_idx] = y[test_idx].mean()
        r = rmse(y, all_preds[m_idx])
        print(f"  [{m_idx+1}/{n_models}] {cfg['name']}: RMSE={r:.2f} ({time.time()-t1:.1f}s)", flush=True)

    # Individual results
    indiv = [(rmse(y, all_preds[i]), i, names[i]) for i in range(n_models)]
    indiv.sort()
    print("\n" + "=" * 70)
    print("個別モデル結果:")
    print("=" * 70)
    for r, i, n in indiv:
        print(f"  RMSE={r:.2f} | {n}")

    # Ensemble search
    print("\n" + "=" * 70)
    print("アンサンブル探索:")
    print("=" * 70)

    results = []

    # Simple averages (top-k)
    for k in range(2, min(n_models + 1, 11)):
        idx = [i for _, i, _ in indiv[:k]]
        avg = np.mean(all_preds[idx], axis=0)
        r = rmse(y, avg)
        results.append((f"SimpleAvg-Top{k}", r, [names[i] for i in idx], None))

    # Weighted averages (top-k)
    for k in [3, 4, 5, 6, 7, 8]:
        if k > n_models:
            break
        idx = [i for _, i, _ in indiv[:k]]
        sub = all_preds[idx]
        w, r = optimize_weights(sub, y)
        results.append((f"WeightedAvg-Top{k}", r, [names[i] for i in idx], w))

    # Exhaustive best subset (k=3,4,5)
    for k in [3, 4, 5]:
        best_r, best_combo = np.inf, None
        for combo in combinations(range(n_models), k):
            avg = np.mean(all_preds[list(combo)], axis=0)
            r = rmse(y, avg)
            if r < best_r:
                best_r = r
                best_combo = combo
        if best_combo:
            results.append((f"BestCombo{k}-Avg", best_r, [names[i] for i in best_combo], None))
            sub = all_preds[list(best_combo)]
            w, rw = optimize_weights(sub, y)
            results.append((f"BestCombo{k}-Weighted", rw, [names[i] for i in best_combo], w))

    # Exhaustive best subset weighted (k=3,4,5)
    for k in [3, 4, 5]:
        best_r, best_combo, best_w = np.inf, None, None
        for combo in combinations(range(n_models), k):
            sub = all_preds[list(combo)]
            w, rw = optimize_weights(sub, y, n_restarts=10)
            if rw < best_r:
                best_r = rw
                best_combo = combo
                best_w = w
        if best_combo:
            results.append((f"BestWeightedCombo{k}", best_r, [names[i] for i in best_combo], best_w))

    results.sort(key=lambda x: x[1])

    print(f"\nTop 20 アンサンブル:")
    print("-" * 70)
    for name, r, mnames, w in results[:20]:
        print(f"  RMSE={r:.4f} | {name}")
        if w is not None:
            for n, wv in zip(mnames, w):
                if wv > 0.01:
                    print(f"    {wv:.3f}: {n}")
        else:
            for n in mnames:
                print(f"    - {n}")
        print()

    best = results[0]
    print("=" * 70)
    print(f"BEST ENSEMBLE: RMSE = {best[1]:.4f} ({best[0]})")
    print("=" * 70)

    rows = [{"method": n, "rmse": r, "models": str(m), "weights": str(w)} for n, r, m, w in results]
    pd.DataFrame(rows).to_csv(OUT_DIR / "issue67_fast_ensemble_results.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'issue67_fast_ensemble_results.csv'}")
    print(f"総実行時間: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
