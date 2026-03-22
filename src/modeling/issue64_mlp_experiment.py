"""MLP (Multi-Layer Perceptron) 実験モジュール

対応Issue: #64
scikit-learn MLPRegressorを使用し、PLS/PCAスコア特徴量でLOSO-CV評価を行う。
前処理×PLS成分数×MLP構成×目的変数変換のグリッドサーチ。
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut

from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue19_msc import compute_msc_reference
from src.preprocessing.issue62_additional_preprocessing import (
    apply_asls,
    apply_piecewise_msc,
)


def rmse(y_true, y_pred):
    """RMSE計算"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def apply_target_transform(y, method):
    """目的変数変換を適用"""
    if method == "raw":
        return y.copy()
    elif method == "log1p":
        return np.log1p(y)
    elif method == "sqrt":
        return np.sqrt(y)
    else:
        raise ValueError(f"Unknown target transform: {method}")


def inverse_target_transform(y, method):
    """目的変数の逆変換"""
    if method == "raw":
        return y.copy()
    elif method == "log1p":
        return np.expm1(y)
    elif method == "sqrt":
        return y ** 2
    else:
        raise ValueError(f"Unknown target transform: {method}")


def apply_preprocessing(X_train_raw, X_test_raw, groups_train, method):
    """前処理を適用する（data leakage防止: trainのみでfit）

    Parameters
    ----------
    X_train_raw, X_test_raw : ndarray
    groups_train : ndarray, train側の樹種ラベル
    method : str, 前処理名

    Returns
    -------
    X_train, X_test : 前処理済みデータ
    """
    if method == "SNV":
        return apply_snv(X_train_raw), apply_snv(X_test_raw)

    elif method == "EPO(1)":
        P = compute_epo_projection(X_train_raw, groups_train, n_components=1)
        return apply_epo(X_train_raw, P), apply_epo(X_test_raw, P)

    elif method == "SNV+AsLS(1e6)":
        X_tr_snv = apply_snv(X_train_raw)
        X_te_snv = apply_snv(X_test_raw)
        return apply_asls(X_tr_snv, lam=1e6), apply_asls(X_te_snv, lam=1e6)

    elif method == "PiecewiseMSC(seg=3)":
        ref = compute_msc_reference(X_train_raw)
        return (
            apply_piecewise_msc(X_train_raw, ref, n_segments=3),
            apply_piecewise_msc(X_test_raw, ref, n_segments=3),
        )

    else:
        raise ValueError(f"Unknown preprocessing: {method}")


def extract_pls_scores(X_train, X_test, y_train, n_components):
    """PLS回帰でスコア（潜在変数）を抽出する。

    trainでfit → train/testをtransform。
    """
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_train, y_train)
    T_train = pls.transform(X_train)
    T_test = pls.transform(X_test)
    return T_train, T_test, pls


def run_single_config(
    X_raw, y, groups,
    preprocess_method,
    n_pls_components,
    hidden_layers,
    activation,
    alpha,
    learning_rate_init,
    target_transform,
    random_state=42,
):
    """単一の設定でLOSO-CVを実行し、RMSEを返す。

    Returns
    -------
    dict: 結果辞書（RMSE, fold別RMSE等）
    """
    logo = LeaveOneGroupOut()
    all_errors_sq = []
    fold_results = []

    for train_idx, test_idx in logo.split(X_raw, y, groups):
        X_train_raw = X_raw[train_idx]
        X_test_raw = X_raw[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        groups_train = groups[train_idx]
        species = groups[test_idx][0]

        # 1. 前処理
        X_train_pp, X_test_pp = apply_preprocessing(
            X_train_raw, X_test_raw, groups_train, preprocess_method
        )

        # 2. 目的変数変換
        y_train_t = apply_target_transform(y_train, target_transform)

        # 3. PLSスコア抽出
        T_train, T_test, _ = extract_pls_scores(
            X_train_pp, X_test_pp, y_train_t, n_pls_components
        )

        # 4. StandardScaler（trainでfit）
        scaler = StandardScaler()
        T_train_s = scaler.fit_transform(T_train)
        T_test_s = scaler.transform(T_test)

        # 5. MLPRegressor
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlp = MLPRegressor(
                hidden_layer_sizes=hidden_layers,
                activation=activation,
                solver="adam",
                alpha=alpha,
                learning_rate="adaptive",
                learning_rate_init=learning_rate_init,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=random_state,
                n_iter_no_change=10,
                batch_size=min(200, max(len(y_train_t) // 5, 32)),
            )
            mlp.fit(T_train_s, y_train_t)

        pred_t = mlp.predict(T_test_s)

        # 6. 逆変換（数値爆発をクリップ）
        pred_t = np.clip(pred_t, -50, 50)  # expm1安全範囲
        pred = inverse_target_transform(pred_t, target_transform)
        pred = np.clip(pred, 0, 500)  # 含水率の物理的範囲

        errors_sq = (pred - y_test) ** 2
        all_errors_sq.extend(errors_sq.tolist())
        fold_rmse = float(np.sqrt(np.mean(errors_sq)))
        fold_results.append({"species": species, "rmse": fold_rmse, "n_samples": len(y_test)})

    overall_rmse = float(np.sqrt(np.mean(all_errors_sq)))

    return {
        "preprocess": preprocess_method,
        "n_pls": n_pls_components,
        "hidden_layers": str(hidden_layers),
        "activation": activation,
        "alpha": alpha,
        "lr_init": learning_rate_init,
        "target_transform": target_transform,
        "rmse": overall_rmse,
        "fold_results": fold_results,
    }


def build_experiment_configs():
    """実験グリッドを構築する（~50件の選定済み組み合わせ）。

    全組合せは非現実的なので、重要な軸を系統的にカバーする組み合わせを選定。
    SNV+AsLS(1e6)は計算コストが非常に高いため、最小限に絞る。
    """
    configs = []

    # --- グループ1: 前処理×PLS成分数のスイープ（固定MLP設定） ---
    for prep in ["SNV", "EPO(1)", "PiecewiseMSC(seg=3)"]:
        for n_pls in [2, 4, 6, 8]:
            configs.append({
                "preprocess": prep,
                "n_pls": n_pls,
                "hidden_layers": (50, 25),
                "activation": "relu",
                "alpha": 0.01,
                "lr_init": 0.01,
                "target_transform": "log1p",
            })

    # SNV+AsLS(1e6)は計算コストが非常に高いため除外
    # （AsLSは1サンプルあたり反復最適化が必要で、13-fold CVでは非現実的）

    # --- グループ2: MLP構造のスイープ（固定前処理: SNV, PLS=6） ---
    for hidden in [(50,), (100,), (50, 25), (100, 50), (100, 50, 25)]:
        for act in ["relu", "tanh"]:
            configs.append({
                "preprocess": "SNV",
                "n_pls": 6,
                "hidden_layers": hidden,
                "activation": act,
                "alpha": 0.01,
                "lr_init": 0.01,
                "target_transform": "log1p",
            })

    # --- グループ3: 正則化強度のスイープ ---
    for alpha in [0.001, 0.01, 0.1, 1.0]:
        for lr in [0.001, 0.01]:
            configs.append({
                "preprocess": "SNV",
                "n_pls": 6,
                "hidden_layers": (50, 25),
                "activation": "relu",
                "alpha": alpha,
                "lr_init": lr,
                "target_transform": "log1p",
            })

    # --- グループ4: 目的変数変換の比較 ---
    for prep in ["SNV", "EPO(1)", "PiecewiseMSC(seg=3)"]:
        for tt in ["raw", "sqrt", "log1p"]:
            configs.append({
                "preprocess": prep,
                "n_pls": 6,
                "hidden_layers": (50, 25),
                "activation": "relu",
                "alpha": 0.01,
                "lr_init": 0.01,
                "target_transform": tt,
            })

    # --- グループ5: EPO(1)での構造・正則化スイープ ---
    for hidden in [(50,), (50, 25), (100, 50)]:
        for alpha in [0.01, 0.1]:
            configs.append({
                "preprocess": "EPO(1)",
                "n_pls": 6,
                "hidden_layers": hidden,
                "activation": "relu",
                "alpha": alpha,
                "lr_init": 0.01,
                "target_transform": "log1p",
            })

    # --- グループ6: tanh + 高正則化（過学習抑制） ---
    for prep in ["SNV", "EPO(1)"]:
        for alpha in [0.1, 1.0]:
            configs.append({
                "preprocess": prep,
                "n_pls": 6,
                "hidden_layers": (50, 25),
                "activation": "tanh",
                "alpha": alpha,
                "lr_init": 0.01,
                "target_transform": "log1p",
            })

    # 重複を除去
    seen = set()
    unique_configs = []
    for c in configs:
        key = (c["preprocess"], c["n_pls"], str(c["hidden_layers"]),
               c["activation"], c["alpha"], c["lr_init"], c["target_transform"])
        if key not in seen:
            seen.add(key)
            unique_configs.append(c)

    return unique_configs


def run_experiment(X_raw, y, groups, configs=None, verbose=True):
    """全設定でLOSO-CVを実行し、結果をDataFrameで返す。

    Parameters
    ----------
    X_raw : ndarray (n_samples, n_features)
    y : ndarray (n_samples,)
    groups : ndarray (n_samples,)
    configs : list[dict], 実験設定。Noneの場合build_experiment_configs()を使用
    verbose : bool

    Returns
    -------
    pd.DataFrame: 全結果（RMSE順にソート）
    """
    if configs is None:
        configs = build_experiment_configs()

    if verbose:
        print(f"=== MLP実験: {len(configs)}パターンを評価 ===\n")

    results = []
    for i, cfg in enumerate(configs):
        if verbose:
            print(f"[{i+1}/{len(configs)}] {cfg['preprocess']} | PLS={cfg['n_pls']} | "
                  f"MLP{cfg['hidden_layers']} | {cfg['activation']} | "
                  f"alpha={cfg['alpha']} | lr={cfg['lr_init']} | target={cfg['target_transform']}",
                  end=" ... ", flush=True)

        try:
            result = run_single_config(
                X_raw, y, groups,
                preprocess_method=cfg["preprocess"],
                n_pls_components=cfg["n_pls"],
                hidden_layers=cfg["hidden_layers"],
                activation=cfg["activation"],
                alpha=cfg["alpha"],
                learning_rate_init=cfg["lr_init"],
                target_transform=cfg["target_transform"],
            )
            results.append(result)
            if verbose:
                print(f"RMSE = {result['rmse']:.4f}")
        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")
            results.append({
                "preprocess": cfg["preprocess"],
                "n_pls": cfg["n_pls"],
                "hidden_layers": str(cfg["hidden_layers"]),
                "activation": cfg["activation"],
                "alpha": cfg["alpha"],
                "lr_init": cfg["lr_init"],
                "target_transform": cfg["target_transform"],
                "rmse": np.nan,
                "fold_results": [],
            })

    # DataFrameに変換（fold_resultsは別途保持）
    fold_data = {i: r.pop("fold_results", []) for i, r in enumerate(results)}
    df_results = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)

    return df_results, fold_data
