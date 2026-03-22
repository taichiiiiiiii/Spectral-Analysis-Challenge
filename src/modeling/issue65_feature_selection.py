"""Issue #65: 高度な特徴量選択手法 (VIP, CARS, iPLS, siPLS) の評価

VIP (Variable Importance in Projection)、CARS (Competitive Adaptive Reweighted Sampling)、
Interval PLS (iPLS)、Synergy Interval PLS (siPLS) を実装し、
複数の前処理・モデル・目的変数変換と組み合わせてLOSO-CVで評価する。
"""
import warnings
from itertools import combinations

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.model_selection import LeaveOneGroupOut, KFold

warnings.filterwarnings("ignore")


# ============================================================
# VIP (Variable Importance in Projection)
# ============================================================

def compute_vip_scores(X: np.ndarray, y: np.ndarray, n_components: int = 3) -> np.ndarray:
    """PLSモデルからVIPスコアを計算する。

    VIP_j = sqrt(p * sum_a(SS_a * w_ja^2) / sum_a(SS_a))

    Parameters
    ----------
    X : (n_samples, n_features)
    y : (n_samples,)
    n_components : PLS成分数

    Returns
    -------
    vip : (n_features,) VIPスコア
    """
    n_samples, n_features = X.shape
    n_comp = min(n_components, n_features - 1, n_samples - 1)
    if n_comp < 1:
        return np.ones(n_features)

    pls = PLSRegression(n_components=n_comp)
    pls.fit(X, y.reshape(-1, 1))

    # T: scores (n_samples, n_comp), W: weights (n_features, n_comp)
    T = pls.x_scores_
    W = pls.x_weights_

    # SS_a = regression coefficient for each component
    Q = pls.y_loadings_  # (1, n_comp)
    SS = np.diag(T.T @ T @ (Q.T @ Q))  # (n_comp,)

    vip = np.zeros(n_features)
    ss_total = np.sum(SS)
    if ss_total < 1e-12:
        return np.ones(n_features)

    for j in range(n_features):
        s = 0.0
        for a in range(n_comp):
            s += SS[a] * (W[j, a] ** 2)
        vip[j] = np.sqrt(n_features * s / ss_total)

    return vip


def vip_select(X_train: np.ndarray, y_train: np.ndarray,
               X_test: np.ndarray, threshold: float = 1.0,
               n_components: int = 3) -> tuple:
    """VIPスコアに基づく変数選択。

    Returns
    -------
    (X_train_sel, X_test_sel, selected_indices, vip_scores)
    """
    vip = compute_vip_scores(X_train, y_train, n_components)
    mask = vip >= threshold
    if mask.sum() < 2:
        # 閾値が高すぎる場合、上位5変数を選択
        top_idx = np.argsort(vip)[::-1][:5]
        mask = np.zeros(len(vip), dtype=bool)
        mask[top_idx] = True

    selected = np.where(mask)[0]
    return X_train[:, selected], X_test[:, selected], selected, vip


# ============================================================
# CARS (Competitive Adaptive Reweighted Sampling)
# ============================================================

def cars_select(X_train: np.ndarray, y_train: np.ndarray,
                X_test: np.ndarray, n_pls: int = 3,
                n_iterations: int = 30) -> tuple:
    """CARS波長選択。

    Returns
    -------
    (X_train_sel, X_test_sel, selected_indices)
    """
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
            rmse = np.mean(cv_rmses)
            if rmse < best_rmse:
                best_rmse = rmse
                best_idx = remaining.copy()

    return X_train[:, best_idx], X_test[:, best_idx], best_idx


# ============================================================
# Interval PLS (iPLS)
# ============================================================

def ipls_evaluate_intervals(X_train: np.ndarray, y_train: np.ndarray,
                            n_intervals: int = 20,
                            n_components: int = 3) -> list:
    """各区間のPLS CVスコアを評価する。

    Returns
    -------
    list of (interval_idx, start, end, cv_rmse)
    """
    n_features = X_train.shape[1]
    interval_size = n_features // n_intervals
    results = []

    for i in range(n_intervals):
        start = i * interval_size
        end = start + interval_size if i < n_intervals - 1 else n_features
        if end - start < 2:
            continue

        X_int = X_train[:, start:end]
        n_comp = min(n_components, end - start - 1, X_train.shape[0] - 1)
        if n_comp < 1:
            continue

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_rmses = []
        for tr, te in kf.split(X_int):
            pls = PLSRegression(n_components=n_comp)
            pls.fit(X_int[tr], y_train[tr])
            pred = pls.predict(X_int[te]).ravel()
            cv_rmses.append(np.sqrt(np.mean((pred - y_train[te]) ** 2)))

        results.append((i, start, end, np.mean(cv_rmses)))

    return results


def ipls_select(X_train: np.ndarray, y_train: np.ndarray,
                X_test: np.ndarray, n_intervals: int = 20,
                n_components: int = 3, n_best: int = 1) -> tuple:
    """iPLSによるベスト区間の選択。

    Returns
    -------
    (X_train_sel, X_test_sel, selected_indices, interval_results)
    """
    interval_results = ipls_evaluate_intervals(
        X_train, y_train, n_intervals, n_components
    )
    if not interval_results:
        return X_train, X_test, np.arange(X_train.shape[1]), []

    # ベスト区間を選択
    sorted_results = sorted(interval_results, key=lambda x: x[3])
    selected_indices = []
    for idx, start, end, _ in sorted_results[:n_best]:
        selected_indices.extend(range(start, end))
    selected_indices = np.array(sorted(set(selected_indices)))

    return (X_train[:, selected_indices], X_test[:, selected_indices],
            selected_indices, interval_results)


# ============================================================
# Synergy Interval PLS (siPLS)
# ============================================================

def sipls_select(X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray, n_intervals: int = 20,
                 n_components: int = 3,
                 n_combine: int = 3) -> tuple:
    """siPLS: iPLS区間評価後、上位区間の組み合わせを評価。

    上位区間からn_combine個を組み合わせ、CV-RMSEが最小の組み合わせを選択。

    Returns
    -------
    (X_train_sel, X_test_sel, selected_indices, best_combo_info)
    """
    interval_results = ipls_evaluate_intervals(
        X_train, y_train, n_intervals, n_components
    )
    if not interval_results:
        return X_train, X_test, np.arange(X_train.shape[1]), {}

    # 上位区間を候補として選択（上位n_combine*2個）
    sorted_results = sorted(interval_results, key=lambda x: x[3])
    n_candidates = min(n_combine * 2, len(sorted_results))
    candidates = sorted_results[:n_candidates]

    best_rmse = np.inf
    best_indices = None
    best_combo = None

    # 組み合わせを評価
    for combo in combinations(range(n_candidates), n_combine):
        indices = []
        for c in combo:
            _, start, end, _ = candidates[c]
            indices.extend(range(start, end))
        indices = np.array(sorted(set(indices)))

        if len(indices) < 2:
            continue

        X_combo = X_train[:, indices]
        n_comp = min(n_components, len(indices) - 1, X_train.shape[0] - 1)
        if n_comp < 1:
            continue

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_rmses = []
        for tr, te in kf.split(X_combo):
            pls = PLSRegression(n_components=n_comp)
            pls.fit(X_combo[tr], y_train[tr])
            pred = pls.predict(X_combo[te]).ravel()
            cv_rmses.append(np.sqrt(np.mean((pred - y_train[te]) ** 2)))

        rmse = np.mean(cv_rmses)
        if rmse < best_rmse:
            best_rmse = rmse
            best_indices = indices
            best_combo = combo

    if best_indices is None:
        best_indices = np.arange(X_train.shape[1])

    return (X_train[:, best_indices], X_test[:, best_indices],
            best_indices, {"combo": best_combo, "rmse": best_rmse})


# ============================================================
# モデルフィッティング
# ============================================================

def fit_predict_model(X_train, y_train, X_test, model_name: str):
    """モデルをフィットして予測を返す。"""
    n_features = X_train.shape[1]
    n_samples = X_train.shape[0]

    if model_name.startswith("PLS"):
        n_comp = int(model_name.replace("PLS(", "").replace(")", ""))
        n_comp = min(n_comp, n_features - 1, n_samples - 1)
        if n_comp < 1:
            return None
        model = PLSRegression(n_components=n_comp)
        model.fit(X_train, y_train)
        return model.predict(X_test).ravel()

    elif model_name.startswith("Lasso"):
        alpha = float(model_name.replace("Lasso(", "").replace(")", ""))
        model = Lasso(alpha=alpha, max_iter=5000)
        model.fit(X_train, y_train)
        return model.predict(X_test)

    elif model_name.startswith("ElasticNet"):
        alpha = float(model_name.replace("ElasticNet(", "").replace(")", ""))
        model = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=5000)
        model.fit(X_train, y_train)
        return model.predict(X_test)

    return None


# ============================================================
# LOSO-CV評価
# ============================================================

def loso_cv_evaluate(X_raw: np.ndarray, y: np.ndarray, groups: np.ndarray,
                     preprocess_fn, feature_select_fn,
                     model_name: str, transform: str = "raw") -> dict:
    """LOSO-CVで特徴量選択+モデルの組み合わせを評価する。

    Parameters
    ----------
    X_raw : 生スペクトルデータ
    y : 目的変数（含水率）
    groups : 樹種ラベル
    preprocess_fn : callable(X_train, X_test, groups_train) -> (X_train_pp, X_test_pp)
    feature_select_fn : callable(X_train, y_train, X_test) -> (X_train_sel, X_test_sel, indices)
    model_name : モデル名
    transform : "raw" or "sqrt"

    Returns
    -------
    dict with rmse, rmse_std, fold_rmses, n_features_selected
    """
    logo = LeaveOneGroupOut()
    fold_rmses = []
    n_features_list = []

    for train_idx, test_idx in logo.split(X_raw, y, groups):
        X_train, X_test = X_raw[train_idx], X_raw[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]

        # 前処理
        X_train_pp, X_test_pp = preprocess_fn(X_train, X_test, groups_train)

        # 目的変数変換
        if transform == "sqrt":
            y_fit = np.sqrt(y_train)
        else:
            y_fit = y_train

        # 特徴量選択
        result = feature_select_fn(X_train_pp, y_fit, X_test_pp)
        X_train_sel, X_test_sel = result[0], result[1]
        n_features_list.append(X_train_sel.shape[1])

        # モデルフィット・予測
        pred = fit_predict_model(X_train_sel, y_fit, X_test_sel, model_name)
        if pred is None:
            fold_rmses.append(999.0)
            continue

        # 逆変換
        if transform == "sqrt":
            pred = np.clip(pred, 0, None) ** 2

        rmse = np.sqrt(np.mean((pred - y_test) ** 2))
        fold_rmses.append(rmse)

    return {
        "rmse": np.mean(fold_rmses),
        "rmse_std": np.std(fold_rmses),
        "fold_rmses": fold_rmses,
        "n_features_avg": np.mean(n_features_list) if n_features_list else 0
    }


# ============================================================
# 前処理関数ファクトリ
# ============================================================

def make_snv_preprocessor():
    """SNV前処理"""
    from src.preprocessing.issue18_snv import apply_snv

    def fn(X_train, X_test, groups_train):
        return apply_snv(X_train), apply_snv(X_test)
    return fn, "SNV"


def make_epo_preprocessor(n_comp=1):
    """EPO前処理"""
    from src.preprocessing.issue18_snv import apply_snv
    from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo

    def fn(X_train, X_test, groups_train):
        P = compute_epo_projection(X_train, groups_train, n_components=n_comp)
        return apply_epo(X_train, P), apply_epo(X_test, P)
    return fn, f"EPO({n_comp})"


def make_snv_asls_preprocessor(lam=1e6):
    """SNV + AsLS前処理"""
    from src.preprocessing.issue18_snv import apply_snv
    from src.preprocessing.issue62_additional_preprocessing import apply_asls

    def fn(X_train, X_test, groups_train):
        X_tr = apply_snv(X_train)
        X_te = apply_snv(X_test)
        X_tr = apply_asls(X_tr, lam=lam)
        X_te = apply_asls(X_te, lam=lam)
        return X_tr, X_te
    return fn, f"SNV+AsLS({lam:.0e})"


# ============================================================
# 特徴量選択関数ファクトリ
# ============================================================

def make_no_selection():
    """特徴量選択なし（ベースライン）"""
    def fn(X_train, y_train, X_test):
        return X_train, X_test, np.arange(X_train.shape[1])
    return fn, "None"


def make_vip_selection(threshold=1.0, n_components=3):
    """VIPベース選択"""
    def fn(X_train, y_train, X_test):
        return vip_select(X_train, y_train, X_test,
                          threshold=threshold, n_components=n_components)[:3]
    return fn, f"VIP(>{threshold})"


def make_cars_selection(n_pls=3, n_iterations=30):
    """CARS選択"""
    def fn(X_train, y_train, X_test):
        return cars_select(X_train, y_train, X_test,
                           n_pls=n_pls, n_iterations=n_iterations)
    return fn, f"CARS(n={n_iterations},pls={n_pls})"


def make_ipls_selection(n_intervals=20, n_components=3, n_best=1):
    """iPLS選択"""
    def fn(X_train, y_train, X_test):
        result = ipls_select(X_train, y_train, X_test,
                             n_intervals=n_intervals,
                             n_components=n_components,
                             n_best=n_best)
        return result[0], result[1], result[2]
    return fn, f"iPLS(int={n_intervals},best={n_best})"


def make_sipls_selection(n_intervals=20, n_components=3, n_combine=3):
    """siPLS選択"""
    def fn(X_train, y_train, X_test):
        result = sipls_select(X_train, y_train, X_test,
                              n_intervals=n_intervals,
                              n_components=n_components,
                              n_combine=n_combine)
        return result[0], result[1], result[2]
    return fn, f"siPLS(int={n_intervals},comb={n_combine})"


# ============================================================
# グリッド評価
# ============================================================

def run_feature_selection_experiment(X_raw, y, groups):
    """全組み合わせを評価してDataFrameで結果を返す。"""
    import pandas as pd

    # 前処理の定義
    preprocessors = [
        make_snv_preprocessor(),
        make_epo_preprocessor(1),
        make_snv_asls_preprocessor(1e6),
    ]

    # 特徴量選択の定義
    feature_selectors = [
        make_no_selection(),
        # VIP
        make_vip_selection(0.8, 3),
        make_vip_selection(1.0, 3),
        make_vip_selection(1.2, 3),
        make_vip_selection(1.5, 3),
        # CARS
        make_cars_selection(2, 20),
        make_cars_selection(3, 30),
        make_cars_selection(4, 50),
        # iPLS
        make_ipls_selection(10, 3, 1),
        make_ipls_selection(20, 3, 1),
        make_ipls_selection(30, 3, 1),
        make_ipls_selection(50, 3, 1),
        # siPLS
        make_sipls_selection(10, 3, 2),
        make_sipls_selection(20, 3, 3),
        make_sipls_selection(30, 3, 3),
        make_sipls_selection(20, 3, 4),
    ]

    # モデルの定義
    models = ["PLS(2)", "PLS(3)", "PLS(4)", "Lasso(0.1)", "ElasticNet(0.1)"]

    # 目的変数変換
    transforms = ["raw", "sqrt"]

    results = []
    total = len(preprocessors) * len(feature_selectors) * len(models) * len(transforms)
    count = 0

    for pp_fn, pp_name in preprocessors:
        for fs_fn, fs_name in feature_selectors:
            for model_name in models:
                for transform in transforms:
                    count += 1
                    combo_name = f"{pp_name} | {fs_name} | {model_name} | {transform}"
                    print(f"[{count}/{total}] {combo_name}", end=" ... ", flush=True)

                    try:
                        result = loso_cv_evaluate(
                            X_raw, y, groups,
                            pp_fn, fs_fn,
                            model_name, transform
                        )
                        print(f"RMSE={result['rmse']:.2f} (±{result['rmse_std']:.2f}), "
                              f"n_feat={result['n_features_avg']:.0f}")
                        results.append({
                            "preprocessing": pp_name,
                            "feature_selection": fs_name,
                            "model": model_name,
                            "transform": transform,
                            "rmse": result["rmse"],
                            "rmse_std": result["rmse_std"],
                            "n_features_avg": result["n_features_avg"],
                            "fold_rmses": str(result["fold_rmses"]),
                        })
                    except Exception as e:
                        print(f"ERROR: {e}")
                        results.append({
                            "preprocessing": pp_name,
                            "feature_selection": fs_name,
                            "model": model_name,
                            "transform": transform,
                            "rmse": 999.0,
                            "rmse_std": 0.0,
                            "n_features_avg": 0,
                            "fold_rmses": str(e),
                        })

    return pd.DataFrame(results)
