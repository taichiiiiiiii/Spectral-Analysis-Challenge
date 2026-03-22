"""Issue #65: 高度な特徴量選択手法 (VIP, CARS, iPLS, siPLS) の評価

VIP (Variable Importance in Projection)、CARS (Competitive Adaptive Reweighted Sampling)、
Interval PLS (iPLS)、Synergy Interval PLS (siPLS) を実装し、
複数の前処理・モデル・目的変数変換と組み合わせてLOSO-CVで評価する。

高速化: 前処理結果をLOSOフォールド単位でキャッシュし、
特徴量選択・モデルの組み合わせごとに再利用する。
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
    """PLSモデルからVIPスコアを計算する。"""
    n_samples, n_features = X.shape
    n_comp = min(n_components, n_features - 1, n_samples - 1)
    if n_comp < 1:
        return np.ones(n_features)

    pls = PLSRegression(n_components=n_comp)
    pls.fit(X, y.reshape(-1, 1))

    T = pls.x_scores_
    W = pls.x_weights_
    Q = pls.y_loadings_
    SS = np.diag(T.T @ T @ (Q.T @ Q))

    ss_total = np.sum(SS)
    if ss_total < 1e-12:
        return np.ones(n_features)

    vip = np.sqrt(n_features * np.sum(SS[np.newaxis, :] * W ** 2, axis=1) / ss_total)
    return vip


def vip_select(X_train, y_train, X_test, threshold=1.0, n_components=3):
    """VIPスコアに基づく変数選択。"""
    vip = compute_vip_scores(X_train, y_train, n_components)
    mask = vip >= threshold
    if mask.sum() < 2:
        top_idx = np.argsort(vip)[::-1][:5]
        mask = np.zeros(len(vip), dtype=bool)
        mask[top_idx] = True
    selected = np.where(mask)[0]
    return X_train[:, selected], X_test[:, selected], selected


# ============================================================
# CARS (Competitive Adaptive Reweighted Sampling)
# ============================================================

def cars_select(X_train, y_train, X_test, n_pls=3, n_iterations=30):
    """CARS波長選択。"""
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

        kf = KFold(n_splits=3, shuffle=True, random_state=42)
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

def ipls_evaluate_intervals(X_train, y_train, n_intervals=20, n_components=3):
    """各区間のPLS CVスコアを評価する。"""
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

        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        cv_rmses = []
        for tr, te in kf.split(X_int):
            pls = PLSRegression(n_components=n_comp)
            pls.fit(X_int[tr], y_train[tr])
            pred = pls.predict(X_int[te]).ravel()
            cv_rmses.append(np.sqrt(np.mean((pred - y_train[te]) ** 2)))

        results.append((i, start, end, np.mean(cv_rmses)))

    return results


def ipls_select(X_train, y_train, X_test, n_intervals=20, n_components=3, n_best=1):
    """iPLSによるベスト区間の選択。"""
    interval_results = ipls_evaluate_intervals(X_train, y_train, n_intervals, n_components)
    if not interval_results:
        return X_train, X_test, np.arange(X_train.shape[1])

    sorted_results = sorted(interval_results, key=lambda x: x[3])
    selected_indices = []
    for idx, start, end, _ in sorted_results[:n_best]:
        selected_indices.extend(range(start, end))
    selected_indices = np.array(sorted(set(selected_indices)))

    return X_train[:, selected_indices], X_test[:, selected_indices], selected_indices


# ============================================================
# Synergy Interval PLS (siPLS)
# ============================================================

def sipls_select(X_train, y_train, X_test, n_intervals=20, n_components=3, n_combine=3):
    """siPLS: 上位区間の組み合わせを評価。"""
    interval_results = ipls_evaluate_intervals(X_train, y_train, n_intervals, n_components)
    if not interval_results:
        return X_train, X_test, np.arange(X_train.shape[1])

    sorted_results = sorted(interval_results, key=lambda x: x[3])
    n_candidates = min(n_combine + 2, len(sorted_results))
    candidates = sorted_results[:n_candidates]

    best_rmse = np.inf
    best_indices = None

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

        kf = KFold(n_splits=3, shuffle=True, random_state=42)
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

    if best_indices is None:
        best_indices = np.arange(X_train.shape[1])

    return X_train[:, best_indices], X_test[:, best_indices], best_indices


# ============================================================
# モデルフィッティング
# ============================================================

def fit_predict_model(X_train, y_train, X_test, model_name):
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
# 前処理関数
# ============================================================

def preprocess_snv(X_train, X_test, groups_train):
    from src.preprocessing.issue18_snv import apply_snv
    return apply_snv(X_train), apply_snv(X_test)


def preprocess_epo(X_train, X_test, groups_train, n_comp=1):
    from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
    P = compute_epo_projection(X_train, groups_train, n_components=n_comp)
    return apply_epo(X_train, P), apply_epo(X_test, P)


def preprocess_snv_asls(X_train, X_test, groups_train, lam=1e6):
    from src.preprocessing.issue18_snv import apply_snv
    from src.preprocessing.issue62_additional_preprocessing import apply_asls
    X_tr = apply_asls(apply_snv(X_train), lam=lam)
    X_te = apply_asls(apply_snv(X_test), lam=lam)
    return X_tr, X_te


# ============================================================
# 高速化版: キャッシュ付きグリッド評価
# ============================================================

def run_feature_selection_experiment(X_raw, y, groups):
    """前処理をLOSOフォールド単位でキャッシュし、全組み合わせを効率的に評価。"""
    import pandas as pd
    import time

    logo = LeaveOneGroupOut()
    unique_species = np.unique(groups)
    n_folds = len(unique_species)

    # LOSOフォールドを事前生成
    folds = list(logo.split(X_raw, y, groups))

    # 前処理定義
    preprocess_configs = [
        ("SNV", preprocess_snv, {}),
        ("EPO(1)", preprocess_epo, {"n_comp": 1}),
        ("SNV+AsLS(1e6)", preprocess_snv_asls, {"lam": 1e6}),
    ]

    # 特徴量選択定義: (名前, 関数, kwargs)
    # 関数シグネチャ: fn(X_train, y_train, X_test, **kwargs) -> (X_tr_sel, X_te_sel, indices)
    fs_configs = [
        ("None", None, {}),
        # VIP
        ("VIP(>0.8)", vip_select, {"threshold": 0.8, "n_components": 3}),
        ("VIP(>1.0)", vip_select, {"threshold": 1.0, "n_components": 3}),
        ("VIP(>1.2)", vip_select, {"threshold": 1.2, "n_components": 3}),
        ("VIP(>1.5)", vip_select, {"threshold": 1.5, "n_components": 3}),
        # CARS
        ("CARS(n=20,pls=2)", cars_select, {"n_pls": 2, "n_iterations": 20}),
        ("CARS(n=30,pls=3)", cars_select, {"n_pls": 3, "n_iterations": 30}),
        ("CARS(n=50,pls=4)", cars_select, {"n_pls": 4, "n_iterations": 50}),
        # iPLS
        ("iPLS(int=10)", ipls_select, {"n_intervals": 10, "n_components": 3, "n_best": 1}),
        ("iPLS(int=20)", ipls_select, {"n_intervals": 20, "n_components": 3, "n_best": 1}),
        ("iPLS(int=30)", ipls_select, {"n_intervals": 30, "n_components": 3, "n_best": 1}),
        ("iPLS(int=50)", ipls_select, {"n_intervals": 50, "n_components": 3, "n_best": 1}),
        # siPLS
        ("siPLS(int=10,c=2)", sipls_select, {"n_intervals": 10, "n_components": 3, "n_combine": 2}),
        ("siPLS(int=20,c=3)", sipls_select, {"n_intervals": 20, "n_components": 3, "n_combine": 3}),
        ("siPLS(int=30,c=3)", sipls_select, {"n_intervals": 30, "n_components": 3, "n_combine": 3}),
        ("siPLS(int=20,c=4)", sipls_select, {"n_intervals": 20, "n_components": 3, "n_combine": 4}),
    ]

    models = ["PLS(2)", "PLS(3)", "PLS(4)", "Lasso(0.1)", "ElasticNet(0.1)"]
    transforms = ["raw", "sqrt"]

    results = []
    total_pp = len(preprocess_configs)
    total_fs = len(fs_configs)
    total_combos = total_pp * total_fs * len(models) * len(transforms)
    combo_count = 0

    for pp_idx, (pp_name, pp_fn, pp_kwargs) in enumerate(preprocess_configs):
        print(f"\n{'='*70}")
        print(f"前処理 [{pp_idx+1}/{total_pp}]: {pp_name}")
        print(f"{'='*70}")

        # LOSOフォールドごとに前処理結果をキャッシュ
        t0 = time.time()
        pp_cache = []
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            X_train, X_test = X_raw[train_idx], X_raw[test_idx]
            groups_train = groups[train_idx]
            X_tr_pp, X_te_pp = pp_fn(X_train, X_test, groups_train, **pp_kwargs)
            pp_cache.append({
                "X_train": X_tr_pp,
                "X_test": X_te_pp,
                "y_train": y[train_idx],
                "y_test": y[test_idx],
                "groups_train": groups_train,
                "species": unique_species[fold_idx] if fold_idx < len(unique_species) else f"fold{fold_idx}",
            })
        print(f"  前処理キャッシュ完了: {time.time()-t0:.1f}秒")

        # 各特徴量選択を適用
        for fs_idx, (fs_name, fs_fn, fs_kwargs) in enumerate(fs_configs):
            t1 = time.time()

            # LOSOフォールドごとに特徴量選択結果をキャッシュ
            fs_cache = []
            for fold_data in pp_cache:
                y_train_raw = fold_data["y_train"]
                y_train_sqrt = np.sqrt(y_train_raw)

                if fs_fn is None:
                    # 選択なし
                    fs_cache.append({
                        "raw": (fold_data["X_train"], fold_data["X_test"]),
                        "sqrt": (fold_data["X_train"], fold_data["X_test"]),
                        "n_feat_raw": fold_data["X_train"].shape[1],
                        "n_feat_sqrt": fold_data["X_train"].shape[1],
                    })
                else:
                    entry = {}
                    for tfm, y_fit in [("raw", y_train_raw), ("sqrt", y_train_sqrt)]:
                        X_tr_sel, X_te_sel, _ = fs_fn(
                            fold_data["X_train"], y_fit, fold_data["X_test"], **fs_kwargs
                        )
                        entry[tfm] = (X_tr_sel, X_te_sel)
                        entry[f"n_feat_{tfm}"] = X_tr_sel.shape[1]
                    fs_cache.append(entry)

            fs_time = time.time() - t1

            # 各モデル・変換で評価
            for model_name in models:
                for transform in transforms:
                    combo_count += 1
                    fold_rmses = []
                    n_feats = []

                    for fold_idx, fold_data in enumerate(pp_cache):
                        X_tr_sel, X_te_sel = fs_cache[fold_idx][transform]
                        n_feats.append(X_tr_sel.shape[1])

                        y_train = fold_data["y_train"]
                        y_test = fold_data["y_test"]

                        if transform == "sqrt":
                            y_fit = np.sqrt(y_train)
                        else:
                            y_fit = y_train

                        pred = fit_predict_model(X_tr_sel, y_fit, X_te_sel, model_name)
                        if pred is None:
                            fold_rmses.append(999.0)
                            continue

                        if transform == "sqrt":
                            pred = np.clip(pred, 0, None) ** 2

                        rmse = np.sqrt(np.mean((pred - y_test) ** 2))
                        fold_rmses.append(rmse)

                    mean_rmse = np.mean(fold_rmses)
                    std_rmse = np.std(fold_rmses)
                    avg_feat = np.mean(n_feats)

                    print(f"  [{combo_count}/{total_combos}] {fs_name} | {model_name} | {transform}"
                          f" → RMSE={mean_rmse:.2f} ±{std_rmse:.2f} (n_feat={avg_feat:.0f})"
                          f"{'  [fs:' + f'{fs_time:.1f}s]' if model_name == models[0] and transform == transforms[0] else ''}")

                    results.append({
                        "preprocessing": pp_name,
                        "feature_selection": fs_name,
                        "model": model_name,
                        "transform": transform,
                        "rmse": mean_rmse,
                        "rmse_std": std_rmse,
                        "n_features_avg": avg_feat,
                    })

    return pd.DataFrame(results)
