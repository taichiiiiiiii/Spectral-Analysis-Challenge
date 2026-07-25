"""Issue #67: アンサンブル最適化 - fold-level予測値ベースの重み最適化

現行ベスト(17.03)を超えるため、多様なモデルのLOSO-CV予測を収集し、
最適な重み付き平均/スタッキングを検索する。
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

from src.eda.data_loader import load_train, get_spectral_columns, get_wavenumbers
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue20_savgol import apply_savgol
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue19_msc import compute_msc_reference, apply_msc
from src.preprocessing.issue62_additional_preprocessing import (
    apply_asls, apply_piecewise_msc, apply_whittaker,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def cars_select(X_train, y_train, X_test, n_pls=3, n_iterations=30):
    """CARS波長選択"""
    n_samples, n_features = X_train.shape
    remaining = np.arange(n_features)
    ratio = (max(n_pls + 2, 5) / n_features) ** (1.0 / n_iterations)
    best_rmse = np.inf
    best_idx = remaining.copy()
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
        cv_rmses = []
        for tr, te in kf.split(X_train[:, remaining]):
            n_c = min(n_comp, len(remaining) - 1)
            if n_c < 1:
                break
            p = PLSRegression(n_components=n_c)
            p.fit(X_train[tr][:, remaining], y_train[tr])
            pred = p.predict(X_train[te][:, remaining]).ravel()
            cv_rmses.append(np.sqrt(np.mean((pred - y_train[te]) ** 2)))
        if cv_rmses:
            r = np.mean(cv_rmses)
            if r < best_rmse:
                best_rmse = r
                best_idx = remaining.copy()
    return X_train[:, best_idx], X_test[:, best_idx], best_idx


def vip_select(X_train, y_train, X_test, n_components=2, threshold=1.5):
    """VIP変数選択"""
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_train, y_train)
    T = pls.x_scores_
    W = pls.x_weights_
    Q = pls.y_loadings_
    p = X_train.shape[1]
    vip = np.zeros(p)
    s = np.diag(T.T @ T @ Q.T @ Q).ravel()
    total_s = np.sum(s)
    for j in range(p):
        weight = np.sum(s * (W[j, :] / np.linalg.norm(W, axis=0)) ** 2)
        vip[j] = np.sqrt(p * weight / total_s)
    mask = vip > threshold
    if mask.sum() < 2:
        mask = np.argsort(vip)[-max(2, int(p * 0.1)):]
        return X_train[:, mask], X_test[:, mask]
    return X_train[:, mask], X_test[:, mask]


def collect_fold_predictions(X_raw, y, groups, folds, model_configs):
    """各モデル構成でLOSO-CV fold毎の予測値を収集"""
    n_samples = len(y)
    n_models = len(model_configs)
    all_predictions = np.full((n_models, n_samples), np.nan)

    for m_idx, config in enumerate(model_configs):
        name = config["name"]
        t0 = time.time()

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            try:
                pred = _predict_single_fold(
                    X_raw, y, groups, train_idx, test_idx, config
                )
                all_predictions[m_idx, test_idx] = pred
            except Exception as e:
                all_predictions[m_idx, test_idx] = y[test_idx].mean()

        fold_rmse = rmse(y, all_predictions[m_idx])
        elapsed = time.time() - t0
        print(f"  [{m_idx+1}/{n_models}] {name}: RMSE={fold_rmse:.2f} ({elapsed:.1f}s)", flush=True)

    return all_predictions


def _predict_single_fold(X_raw, y, groups, train_idx, test_idx, config):
    """1つのfoldで予測"""
    X_train_raw = X_raw[train_idx]
    X_test_raw = X_raw[test_idx]
    y_train = y[train_idx]
    groups_train = groups[train_idx]
    pp = config["preprocessing"]
    model_type = config["model"]
    transform = config["transform"]

    # 前処理
    if pp == "SNV":
        X_tr, X_te = apply_snv(X_train_raw), apply_snv(X_test_raw)
    elif pp == "EPO(1)":
        P = compute_epo_projection(X_train_raw, groups_train, n_components=1)
        X_tr, X_te = apply_epo(X_train_raw, P), apply_epo(X_test_raw, P)
    elif pp == "SNV+AsLS(1e6)":
        X_tr = apply_asls(apply_snv(X_train_raw), lam=1e6)
        X_te = apply_asls(apply_snv(X_test_raw), lam=1e6)
    elif pp == "PiecewiseMSC(seg=3)":
        ref = compute_msc_reference(X_train_raw)
        X_tr = apply_piecewise_msc(X_train_raw, ref, 3)
        X_te = apply_piecewise_msc(X_test_raw, ref, 3)
    elif pp == "MSC":
        ref = compute_msc_reference(X_train_raw)
        X_tr, X_te = apply_msc(X_train_raw, ref), apply_msc(X_test_raw, ref)
    elif pp == "SG2d+EPO(1)":
        X_tr_sg = apply_savgol(X_train_raw, deriv=2, window_length=7)
        X_te_sg = apply_savgol(X_test_raw, deriv=2, window_length=7)
        P = compute_epo_projection(X_tr_sg, groups_train, n_components=1)
        X_tr, X_te = apply_epo(X_tr_sg, P), apply_epo(X_te_sg, P)
    elif pp == "Whittaker(1e3)":
        X_tr = apply_whittaker(X_train_raw, lam=1e3)
        X_te = apply_whittaker(X_test_raw, lam=1e3)
    else:
        X_tr, X_te = X_train_raw.copy(), X_test_raw.copy()

    # 特徴量選択(optional)
    feat_sel = config.get("feature_selection")
    if feat_sel == "CARS":
        n_pls_cars = config.get("cars_pls", 3)
        X_tr, X_te, _ = cars_select(X_tr, y_train, X_te, n_pls=n_pls_cars, n_iterations=30)
    elif feat_sel == "VIP(1.5)":
        X_tr, X_te = vip_select(X_tr, y_train, X_te, n_components=2, threshold=1.5)
    elif feat_sel == "VIP(1.2)":
        X_tr, X_te = vip_select(X_tr, y_train, X_te, n_components=2, threshold=1.2)

    # 目的変数変換
    if transform == "sqrt":
        y_fit = np.sqrt(y_train)
    elif transform == "log1p":
        y_fit = np.log1p(y_train)
    else:
        y_fit = y_train.copy()

    # モデル学習・予測
    if model_type.startswith("PLS("):
        nc = int(model_type.split("(")[1].rstrip(")"))
        nc = min(nc, X_tr.shape[1] - 1)
        model = PLSRegression(n_components=max(1, nc))
        model.fit(X_tr, y_fit)
        pred = model.predict(X_te).ravel()
    elif model_type == "Lasso(0.1)":
        model = Lasso(alpha=0.1, max_iter=10000)
        model.fit(X_tr, y_fit)
        pred = model.predict(X_te)
    elif model_type == "ElasticNet(0.1)":
        model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
        model.fit(X_tr, y_fit)
        pred = model.predict(X_te)
    elif model_type == "Ridge(100)":
        model = Ridge(alpha=100)
        model.fit(X_tr, y_fit)
        pred = model.predict(X_te)
    else:
        raise ValueError(f"Unknown model: {model_type}")

    # 逆変換
    if transform == "sqrt":
        pred = np.clip(pred, 0, None) ** 2
    elif transform == "log1p":
        pred = np.expm1(np.clip(pred, 0, 20))

    return pred


def optimize_weights(predictions, y, method="minimize"):
    """予測の重み付き平均を最適化"""
    n_models = predictions.shape[0]

    def objective(w):
        w_norm = np.abs(w) / np.sum(np.abs(w))
        blended = np.sum(w_norm[:, None] * predictions, axis=0)
        return rmse(y, blended)

    # ランダム初期値で複数回最適化
    best_result = None
    best_rmse = np.inf

    for seed in range(50):
        rng = np.random.RandomState(seed)
        w0 = rng.dirichlet(np.ones(n_models))
        res = minimize(objective, w0, method="Nelder-Mead",
                      options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8})
        if res.fun < best_rmse:
            best_rmse = res.fun
            best_result = res

    w_opt = np.abs(best_result.x) / np.sum(np.abs(best_result.x))
    return w_opt, best_rmse


def search_best_subsets(predictions, y, model_names, max_models=8):
    """ベストなモデルサブセットを探索"""
    n_models = predictions.shape[0]
    results = []

    # まず全モデルの単体性能
    individual = [(rmse(y, predictions[i]), i) for i in range(n_models)]
    individual.sort()

    # Top-K単純平均
    for k in range(2, min(n_models + 1, max_models + 1)):
        top_k_idx = [idx for _, idx in individual[:k]]
        avg_pred = np.mean(predictions[top_k_idx], axis=0)
        r = rmse(y, avg_pred)
        names = [model_names[i] for i in top_k_idx]
        results.append(("SimpleAvg-Top" + str(k), r, top_k_idx, None, names))

    # Top-K重み最適化
    for k in [3, 4, 5, 6, 7, 8]:
        if k > n_models:
            break
        top_k_idx = [idx for _, idx in individual[:k]]
        sub_preds = predictions[top_k_idx]
        w_opt, r_opt = optimize_weights(sub_preds, y)
        names = [model_names[i] for i in top_k_idx]
        results.append((f"WeightedAvg-Top{k}", r_opt, top_k_idx, w_opt, names))

    # 多様性ベース: 相関が低いモデルの組み合わせ
    corr_matrix = np.corrcoef(predictions)
    for k in [3, 4, 5, 6]:
        if k > n_models:
            break
        # 貪欲法: 最も良いモデルから始めて、相関が低いものを追加
        selected = [individual[0][1]]
        remaining = list(range(n_models))
        remaining.remove(selected[0])
        while len(selected) < k and remaining:
            best_score = -1
            best_idx = remaining[0]
            for idx in remaining:
                avg_corr = np.mean([corr_matrix[idx, s] for s in selected])
                diversity_score = (1 - avg_corr) * (1 / (rmse(y, predictions[idx]) + 1))
                if diversity_score > best_score:
                    best_score = diversity_score
                    best_idx = idx
            selected.append(best_idx)
            remaining.remove(best_idx)

        # 単純平均
        avg_pred = np.mean(predictions[selected], axis=0)
        r = rmse(y, avg_pred)
        names = [model_names[i] for i in selected]
        results.append((f"DiverseAvg-{k}", r, selected, None, names))

        # 重み最適化
        sub_preds = predictions[selected]
        w_opt, r_opt = optimize_weights(sub_preds, y)
        results.append((f"DiverseWeighted-{k}", r_opt, selected, w_opt, names))

    # 全組み合わせ探索 (小規模の場合)
    if n_models <= 12:
        for k in [3, 4, 5]:
            best_combo_rmse = np.inf
            best_combo = None
            for combo in combinations(range(n_models), k):
                avg_pred = np.mean(predictions[list(combo)], axis=0)
                r = rmse(y, avg_pred)
                if r < best_combo_rmse:
                    best_combo_rmse = r
                    best_combo = combo
            if best_combo is not None:
                names = [model_names[i] for i in best_combo]
                results.append((f"BestCombo-{k}-Avg", best_combo_rmse, list(best_combo), None, names))
                # 重み最適化
                sub_preds = predictions[list(best_combo)]
                w_opt, r_opt = optimize_weights(sub_preds, y)
                results.append((f"BestCombo-{k}-Weighted", r_opt, list(best_combo), w_opt, names))

    results.sort(key=lambda x: x[1])
    return results


def main():
    t_start = time.time()
    df_train = load_train(DATA_DIR)
    spectral_cols = get_spectral_columns(df_train)
    X_raw = df_train[spectral_cols].values
    y = df_train["含水率"].values
    groups = df_train["樹種"].values

    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_raw, y, groups))

    print("=" * 70)
    print("Issue #67: アンサンブル最適化 - 多様なモデルの重み最適化")
    print("=" * 70)

    # 多様なモデル構成（Issue #62のTop結果 + 新手法）
    model_configs = [
        # Top performers from grid
        {"name": "SNV+AsLS(1e6)+PLS(2)+sqrt", "preprocessing": "SNV+AsLS(1e6)", "model": "PLS(2)", "transform": "sqrt"},
        {"name": "SNV+ElasticNet(0.1)+sqrt", "preprocessing": "SNV", "model": "ElasticNet(0.1)", "transform": "sqrt"},
        {"name": "PiecewiseMSC+Lasso(0.1)+raw", "preprocessing": "PiecewiseMSC(seg=3)", "model": "Lasso(0.1)", "transform": "raw"},
        {"name": "SNV+PLS(2)+sqrt", "preprocessing": "SNV", "model": "PLS(2)", "transform": "sqrt"},
        {"name": "SNV+PLS(2)+raw", "preprocessing": "SNV", "model": "PLS(2)", "transform": "raw"},
        {"name": "EPO(1)+Lasso(0.1)+raw", "preprocessing": "EPO(1)", "model": "Lasso(0.1)", "transform": "raw"},
        {"name": "MSC+PLS(4)+sqrt", "preprocessing": "MSC", "model": "PLS(4)", "transform": "sqrt"},

        # Submission v2 models
        {"name": "SG2d+EPO(1)+PLS(3)+raw", "preprocessing": "SG2d+EPO(1)", "model": "PLS(3)", "transform": "raw"},
        {"name": "EPO(1)+PLS(4)+sqrt", "preprocessing": "EPO(1)", "model": "PLS(4)", "transform": "sqrt"},

        # Feature selection variants
        {"name": "EPO(1)+CARS+PLS(3)+raw", "preprocessing": "EPO(1)", "model": "PLS(3)", "transform": "raw", "feature_selection": "CARS", "cars_pls": 3},
        {"name": "SNV+VIP(1.5)+PLS(4)+sqrt", "preprocessing": "SNV", "model": "PLS(4)", "transform": "sqrt", "feature_selection": "VIP(1.5)"},
        {"name": "SNV+VIP(1.2)+PLS(3)+sqrt", "preprocessing": "SNV", "model": "PLS(3)", "transform": "sqrt", "feature_selection": "VIP(1.2)"},

        # Additional diversity
        {"name": "Whittaker+Lasso(0.1)+raw", "preprocessing": "Whittaker(1e3)", "model": "Lasso(0.1)", "transform": "raw"},
        {"name": "PiecewiseMSC+PLS(4)+sqrt", "preprocessing": "PiecewiseMSC(seg=3)", "model": "PLS(4)", "transform": "sqrt"},
        {"name": "SNV+PLS(4)+sqrt", "preprocessing": "SNV", "model": "PLS(4)", "transform": "sqrt"},
        {"name": "EPO(1)+PLS(2)+raw", "preprocessing": "EPO(1)", "model": "PLS(2)", "transform": "raw"},
        {"name": "MSC+Lasso(0.1)+raw", "preprocessing": "MSC", "model": "Lasso(0.1)", "transform": "raw"},
        {"name": "SNV+Ridge(100)+sqrt", "preprocessing": "SNV", "model": "Ridge(100)", "transform": "sqrt"},
    ]

    print(f"\nモデル数: {len(model_configs)}")
    print("=" * 70)
    print("Step 1: LOSO-CV fold予測値の収集")
    print("=" * 70)

    predictions = collect_fold_predictions(X_raw, y, groups, folds, model_configs)
    model_names = [c["name"] for c in model_configs]

    print("\n" + "=" * 70)
    print("Step 2: アンサンブル最適化")
    print("=" * 70)

    results = search_best_subsets(predictions, y, model_names, max_models=8)

    print(f"\nTop 20 アンサンブル結果:")
    print("-" * 70)
    for name, r, indices, weights, names in results[:20]:
        print(f"  RMSE={r:.4f} | {name}")
        if weights is not None:
            for n, w in zip(names, weights):
                if w > 0.01:
                    print(f"    {w:.3f}: {n}")
        else:
            for n in names:
                print(f"    - {n}")
        print()

    # 最良結果
    best = results[0]
    print("=" * 70)
    print(f"BEST ENSEMBLE: RMSE = {best[1]:.4f}")
    print(f"Method: {best[0]}")
    print("Models:")
    if best[3] is not None:
        for n, w in zip(best[4], best[3]):
            print(f"  {w:.4f}: {n}")
    else:
        for n in best[4]:
            print(f"  - {n}")
    print("=" * 70)

    # CSV出力
    rows = []
    for name, r, indices, weights, names in results:
        rows.append({"ensemble_method": name, "rmse": r, "models": str(names),
                     "weights": str(weights) if weights is not None else "equal"})
    pd.DataFrame(rows).to_csv(OUT_DIR / "issue67_ensemble_results.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'issue67_ensemble_results.csv'}")
    print(f"総実行時間: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
