"""Issue #67: アンサンブルをfold-RMSE平均で評価（ベースライン17.03と同一指標）

最初のアンサンブル最適化結果から得た有望な組み合わせを、
LOSO-CV fold毎のRMSE平均で評価する。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import warnings
warnings.filterwarnings("ignore")
import time
from scipy.optimize import minimize

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.model_selection import LeaveOneGroupOut, KFold

from src.eda.data_loader import load_train, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue20_savgol import apply_savgol
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue19_msc import compute_msc_reference, apply_msc
from src.preprocessing.issue62_additional_preprocessing import (
    apply_asls, apply_piecewise_msc,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"


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


def predict_fold(X_tr_raw, X_te_raw, y_train, groups_train, config):
    pp = config["pp"]
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
    elif pp == "MSC":
        ref = compute_msc_reference(X_tr_raw)
        X_tr, X_te = apply_msc(X_tr_raw, ref), apply_msc(X_te_raw, ref)
    elif pp == "SG2d+EPO(1)":
        X_tr_sg = apply_savgol(X_tr_raw, deriv=2, window_length=7)
        X_te_sg = apply_savgol(X_te_raw, deriv=2, window_length=7)
        P = compute_epo_projection(X_tr_sg, groups_train, n_components=1)
        X_tr, X_te = apply_epo(X_tr_sg, P), apply_epo(X_te_sg, P)
    else:
        X_tr, X_te = X_tr_raw.copy(), X_te_raw.copy()

    fs = config.get("fs")
    if fs == "CARS":
        X_tr, X_te, _ = cars_select(X_tr, y_train, X_te, n_pls=config.get("cars_pls", 3))
    elif fs == "VIP(1.5)":
        X_tr, X_te = vip_select(X_tr, y_train, X_te, threshold=1.5)

    tf = config["tf"]
    y_fit = np.sqrt(y_train) if tf == "sqrt" else y_train.copy()

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
    else:
        raise ValueError(mdl)

    if tf == "sqrt":
        pred = np.clip(pred, 0, None) ** 2
    return pred


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
    print("Issue #67: fold-RMSE平均によるアンサンブル評価")
    print("  ベースライン: 17.03 (v2 5-model SimpleAvg)")
    print("=" * 70)

    # 評価対象モデル（多様性重視で選定）
    models = [
        {"name": "SNV+PLS(2)+raw", "pp": "SNV", "model": "PLS(2)", "tf": "raw"},
        {"name": "SNV+PLS(2)+sqrt", "pp": "SNV", "model": "PLS(2)", "tf": "sqrt"},
        {"name": "SNV+PLS(4)+sqrt", "pp": "SNV", "model": "PLS(4)", "tf": "sqrt"},
        {"name": "EPO(1)+PLS(2)+raw", "pp": "EPO(1)", "model": "PLS(2)", "tf": "raw"},
        {"name": "EPO(1)+PLS(4)+sqrt", "pp": "EPO(1)", "model": "PLS(4)", "tf": "sqrt"},
        {"name": "EPO(1)+PLS(3)+raw", "pp": "EPO(1)", "model": "PLS(3)", "tf": "raw"},
        {"name": "MSC+PLS(4)+sqrt", "pp": "MSC", "model": "PLS(4)", "tf": "sqrt"},
        {"name": "SG2d+EPO(1)+PLS(3)+raw", "pp": "SG2d+EPO(1)", "model": "PLS(3)", "tf": "raw"},
        {"name": "SNV+ElasticNet+sqrt", "pp": "SNV", "model": "ElasticNet", "tf": "sqrt"},
        {"name": "EPO(1)+Lasso+raw", "pp": "EPO(1)", "model": "Lasso", "tf": "raw"},
        {"name": "PiecewiseMSC+Lasso+raw", "pp": "PiecewiseMSC(seg=3)", "model": "Lasso", "tf": "raw"},
        {"name": "EPO(1)+CARS+PLS(3)+raw", "pp": "EPO(1)", "model": "PLS(3)", "tf": "raw", "fs": "CARS", "cars_pls": 3},
        {"name": "SNV+VIP(1.5)+PLS(4)+sqrt", "pp": "SNV", "model": "PLS(4)", "tf": "sqrt", "fs": "VIP(1.5)"},
        {"name": "SNV+AsLS(1e6)+PLS(2)+sqrt", "pp": "SNV+AsLS(1e6)", "model": "PLS(2)", "tf": "sqrt"},
    ]

    n_models = len(models)
    n_folds = len(folds)

    # fold毎の予測を収集
    fold_preds = np.zeros((n_models, n_folds), dtype=object)  # fold_idx -> pred array
    all_preds = [None] * n_models  # model_idx -> full predictions array

    for m_idx, cfg in enumerate(models):
        t1 = time.time()
        preds_list = []
        fold_rmses = []
        for f_idx, (train_idx, test_idx) in enumerate(folds):
            try:
                pred = predict_fold(
                    X_raw[train_idx], X_raw[test_idx],
                    y[train_idx], groups[train_idx], cfg
                )
                preds_list.append((test_idx, pred))
                fold_rmses.append(rmse(y[test_idx], pred))
            except Exception as e:
                preds_list.append((test_idx, np.full(len(test_idx), y[train_idx].mean())))
                fold_rmses.append(999.0)

        all_preds[m_idx] = preds_list
        mean_r = np.mean(fold_rmses)
        print(f"  [{m_idx+1}/{n_models}] {cfg['name']}: fold-RMSE={mean_r:.2f} ± {np.std(fold_rmses):.2f} ({time.time()-t1:.1f}s)", flush=True)

    # アンサンブル評価（fold-RMSE平均）
    print("\n" + "=" * 70)
    print("アンサンブル評価（fold-RMSE平均）")
    print("=" * 70)

    def eval_ensemble_fold_rmse(model_indices, weights=None):
        fold_rmses = []
        for f_idx, (train_idx, test_idx) in enumerate(folds):
            preds = []
            for m_idx in model_indices:
                _, pred = all_preds[m_idx][f_idx]
                preds.append(pred)
            if weights is not None:
                w = np.array(weights)
                w = w / w.sum()
                ensemble_pred = sum(w_i * p for w_i, p in zip(w, preds))
            else:
                ensemble_pred = np.mean(preds, axis=0)
            ensemble_pred = np.clip(ensemble_pred, 0, 200)
            fold_rmses.append(rmse(y[test_idx], ensemble_pred))
        return np.mean(fold_rmses), np.std(fold_rmses), fold_rmses

    def optimize_weights_fold(model_indices, n_restarts=30):
        n = len(model_indices)
        def obj(w):
            w_n = np.abs(w) / np.sum(np.abs(w))
            r, _, _ = eval_ensemble_fold_rmse(model_indices, w_n)
            return r
        best_r, best_w = np.inf, np.ones(n) / n
        for s in range(n_restarts):
            w0 = np.random.RandomState(s).dirichlet(np.ones(n))
            res = minimize(obj, w0, method="Nelder-Mead", options={"maxiter": 3000})
            if res.fun < best_r:
                best_r = res.fun
                best_w = np.abs(res.x) / np.sum(np.abs(res.x))
        return best_w, best_r

    # 個別モデルのfold-RMSE
    indiv = []
    for m_idx, cfg in enumerate(models):
        r, s, _ = eval_ensemble_fold_rmse([m_idx])
        indiv.append((r, m_idx, cfg["name"]))
    indiv.sort()

    print("\n個別モデル (fold-RMSE平均):")
    for r, i, n in indiv:
        print(f"  {r:.2f} | {n}")

    # アンサンブル探索
    results = []

    # Top-K SimpleAvg
    for k in range(2, min(n_models + 1, 10)):
        idx = [i for _, i, _ in indiv[:k]]
        r, s, fr = eval_ensemble_fold_rmse(idx)
        names = [models[i]["name"] for i in idx]
        results.append((f"SimpleAvg-Top{k}", r, s, names, None))

    # Top-K WeightedAvg
    for k in [3, 4, 5, 6, 7, 8]:
        if k > n_models:
            break
        idx = [i for _, i, _ in indiv[:k]]
        w, r_w = optimize_weights_fold(idx)
        names = [models[i]["name"] for i in idx]
        _, s, _ = eval_ensemble_fold_rmse(idx, w)
        results.append((f"WeightedAvg-Top{k}", r_w, s, names, w))

    # ベスト3-5組み合わせ（全探索）
    from itertools import combinations
    for k in [3, 4, 5]:
        best_r, best_combo = np.inf, None
        for combo in combinations(range(n_models), k):
            r, _, _ = eval_ensemble_fold_rmse(list(combo))
            if r < best_r:
                best_r = r
                best_combo = combo
        if best_combo:
            names = [models[i]["name"] for i in best_combo]
            _, s, _ = eval_ensemble_fold_rmse(list(best_combo))
            results.append((f"BestCombo{k}-Avg", best_r, s, names, None))
            w, r_w = optimize_weights_fold(list(best_combo))
            _, s2, _ = eval_ensemble_fold_rmse(list(best_combo), w)
            results.append((f"BestCombo{k}-Weighted", r_w, s2, names, w))

    results.sort(key=lambda x: x[1])

    print(f"\nTop 15 アンサンブル (fold-RMSE平均):")
    print("-" * 70)
    for name, r, s, mnames, w in results[:15]:
        print(f"  RMSE={r:.4f} ± {s:.2f} | {name}")
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
    print(f"BEST ENSEMBLE (fold-RMSE): {best[1]:.4f} ± {best[2]:.2f}")
    print(f"Method: {best[0]}")
    print(f"Baseline: 17.03")
    print(f"改善幅: {17.03 - best[1]:.4f}")
    print("=" * 70)

    # ベスト構成のfold詳細
    if best[0].startswith("BestCombo") and "Weighted" in best[0]:
        k = int(best[0].replace("BestCombo", "").replace("-Weighted", ""))
        # Find the combo
        best_r2, best_combo2 = np.inf, None
        for combo in combinations(range(n_models), k):
            r2, _, _ = eval_ensemble_fold_rmse(list(combo))
            if r2 < best_r2:
                best_r2 = r2
                best_combo2 = combo
        if best_combo2:
            _, _, fold_rmses = eval_ensemble_fold_rmse(list(best_combo2), best[4])
            print("\nFold詳細:")
            for sp, fr in zip(species_names, fold_rmses):
                print(f"  {sp}: {fr:.2f}")

    print(f"\n総実行時間: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
