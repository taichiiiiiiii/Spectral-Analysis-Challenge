"""Issue #70: サンプル重み付け戦略のLOSO-CV評価

樹種ごとのサンプル数の不均衡に対して、
複数の重み付け戦略の効果を検証する。

重み付け戦略:
- no_weight: 重みなし（ベースライン）
- inv_freq: 逆頻度重み  w = N / (n_species * count)
- sqrt_inv: 平方根逆頻度  w = sqrt(max_count / count)
- log_inv: 対数逆頻度  w = log(max_count / count + 1)

モデル:
- Lasso(0.1), Ridge(100), ElasticNet(0.1): sample_weight引数を使用
- PLS(2), PLS(4): sqrt(w)でX,yを重み付け
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import time
import warnings

warnings.filterwarnings("ignore")

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.model_selection import LeaveOneGroupOut

from src.eda.data_loader import load_train, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue19_msc import compute_msc_reference
from src.preprocessing.issue62_additional_preprocessing import apply_piecewise_msc

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_sample_weights(species_train: np.ndarray, strategy: str) -> np.ndarray:
    """訓練foldの樹種分布から各サンプルの重みを計算する。

    Parameters
    ----------
    species_train : 訓練foldの樹種ラベル配列
    strategy : 重み付け戦略名

    Returns
    -------
    weights : 各サンプルの重み
    """
    if strategy == "no_weight":
        return np.ones(len(species_train))

    unique, counts = np.unique(species_train, return_counts=True)
    count_map = dict(zip(unique, counts))
    n_total = len(species_train)
    n_species = len(unique)
    max_count = counts.max()

    weights = np.zeros(len(species_train))
    for i, sp in enumerate(species_train):
        c = count_map[sp]
        if strategy == "inv_freq":
            weights[i] = n_total / (n_species * c)
        elif strategy == "sqrt_inv":
            weights[i] = np.sqrt(max_count / c)
        elif strategy == "log_inv":
            weights[i] = np.log(max_count / c + 1)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    return weights


def preprocess(X_train, X_val, groups_train):
    """SNV -> EPO(1) -> PiecewiseMSC(seg=3)"""
    # SNV
    X_tr = apply_snv(X_train)
    X_va = apply_snv(X_val)

    # EPO(1) - trainで投影行列を計算
    P = compute_epo_projection(X_tr, groups_train, n_components=1)
    X_tr = apply_epo(X_tr, P)
    X_va = apply_epo(X_va, P)

    # PiecewiseMSC(seg=3) - trainでリファレンスを計算
    ref = compute_msc_reference(X_tr)
    X_tr = apply_piecewise_msc(X_tr, ref, n_segments=3)
    X_va = apply_piecewise_msc(X_va, ref, n_segments=3)

    return X_tr, X_va


def main():
    print("=" * 70)
    print("Issue #70: サンプル重み付け戦略のLOSO-CV評価")
    print("=" * 70)

    # データ読み込み
    df = load_train(DATA_DIR)
    spec_cols = get_spectral_columns(df)
    X = df[spec_cols].values.astype(float)
    y = df["含水率"].values.astype(float)
    groups = df["樹種"].values

    print(f"\nデータ: {X.shape[0]}サンプル, {X.shape[1]}波長")
    print(f"樹種数: {len(np.unique(groups))}")

    # 樹種ごとのサンプル数
    print("\n樹種サンプル数:")
    unique_species, counts = np.unique(groups, return_counts=True)
    for sp, cnt in sorted(zip(unique_species, counts), key=lambda x: -x[1]):
        print(f"  {sp}: {cnt}")

    # 実験設定
    weight_strategies = ["no_weight", "inv_freq", "sqrt_inv", "log_inv"]
    models_config = [
        ("PLS", {"n_components": 2}),
        ("PLS", {"n_components": 4}),
        ("Lasso", {"alpha": 0.1}),
        ("Ridge", {"alpha": 100}),
        ("ElasticNet", {"alpha": 0.1}),
    ]
    target_transforms = ["raw", "sqrt"]

    total = len(weight_strategies) * len(models_config) * len(target_transforms)
    print(f"\n実験数: {total}")
    print("前処理をfoldごとに1回のみ実行（高速化）")
    print("-" * 70)

    start_all = time.time()

    # LOSO-CVのfold分割を事前に取得
    logo = LeaveOneGroupOut()
    folds = list(logo.split(X, y, groups))

    # 各foldで前処理を事前計算（重み付けに依存しない）
    print("\n前処理を13 foldで事前計算中...")
    t_pre = time.time()
    preprocessed_folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_train, X_val = X[train_idx], X[val_idx]
        groups_train = groups[train_idx]
        X_tr, X_va = preprocess(X_train, X_val, groups_train)
        preprocessed_folds.append({
            "train_idx": train_idx,
            "val_idx": val_idx,
            "X_tr": X_tr,
            "X_va": X_va,
            "groups_train": groups_train,
        })
        species_left_out = np.unique(groups[val_idx])[0]
        print(f"  fold {fold_idx+1:2d}: {species_left_out} (train={len(train_idx)}, val={len(val_idx)})")
    print(f"前処理完了: {time.time() - t_pre:.1f}s\n")

    # 全組合せを実行
    results = []
    count = 0

    for target_tf in target_transforms:
        for model_name, model_params in models_config:
            for ws in weight_strategies:
                count += 1
                param_str = ", ".join(f"{k}={v}" for k, v in model_params.items())
                label = f"{model_name}({param_str})"

                t0 = time.time()
                fold_rmses = []

                for fold_data in preprocessed_folds:
                    train_idx = fold_data["train_idx"]
                    val_idx = fold_data["val_idx"]
                    X_tr = fold_data["X_tr"]
                    X_va = fold_data["X_va"]
                    groups_train = fold_data["groups_train"]

                    y_train = y[train_idx]
                    y_val = y[val_idx]

                    # 目的変数変換
                    if target_tf == "sqrt":
                        y_tr = np.sqrt(y_train)
                    else:
                        y_tr = y_train.copy()

                    # 重み計算（訓練foldのみから）
                    weights = compute_sample_weights(groups_train, ws)

                    # モデル学習・予測
                    if model_name.startswith("PLS"):
                        n_comp = model_params["n_components"]
                        model = PLSRegression(n_components=n_comp, scale=False)
                        sqrt_w = np.sqrt(weights)
                        X_tr_w = X_tr * sqrt_w[:, None]
                        y_tr_w = y_tr * sqrt_w
                        model.fit(X_tr_w, y_tr_w)
                        pred = model.predict(X_va).ravel()
                    else:
                        if model_name == "Lasso":
                            model = Lasso(alpha=model_params["alpha"], max_iter=2000, tol=1e-3)
                        elif model_name == "Ridge":
                            model = Ridge(alpha=model_params["alpha"], solver="cholesky")
                        elif model_name == "ElasticNet":
                            model = ElasticNet(alpha=model_params["alpha"], max_iter=2000, tol=1e-3)
                        model.fit(X_tr, y_tr, sample_weight=weights)
                        pred = model.predict(X_va).ravel()

                    # 逆変換
                    if target_tf == "sqrt":
                        pred = pred ** 2

                    pred = np.clip(pred, 0, None)
                    rmse = np.sqrt(np.mean((y_val - pred) ** 2))
                    fold_rmses.append(rmse)

                mean_rmse = np.mean(fold_rmses)
                elapsed = time.time() - t0

                results.append({
                    "model": label,
                    "weight_strategy": ws,
                    "target_transform": target_tf,
                    "fold_rmse_mean": round(mean_rmse, 4),
                    "time_sec": round(elapsed, 1),
                })

                print(f"  [{count:2d}/{total}] {label:30s} | weight={ws:12s} | "
                      f"target={target_tf:4s} | RMSE={mean_rmse:7.4f} | {elapsed:.1f}s",
                      flush=True)

    elapsed_all = time.time() - start_all
    print(f"\n総実行時間: {elapsed_all:.1f}s")

    # 結果をDataFrameに変換
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("fold_rmse_mean").reset_index(drop=True)

    # Top20表示
    print("\n" + "=" * 70)
    print("Top 20 結果 (fold-RMSE mean 昇順)")
    print("=" * 70)
    print(df_results.head(20).to_string(index=False))

    # 重み付き vs 重みなし比較
    print("\n" + "=" * 70)
    print("重み付け効果の比較 (各モデル・target_transformごと)")
    print("=" * 70)

    for target_tf in target_transforms:
        print(f"\n--- target_transform = {target_tf} ---")
        for model_name, model_params in models_config:
            param_str = ", ".join(f"{k}={v}" for k, v in model_params.items())
            label = f"{model_name}({param_str})"

            subset = df_results[
                (df_results["model"] == label) &
                (df_results["target_transform"] == target_tf)
            ].sort_values("fold_rmse_mean")

            baseline_row = subset[subset["weight_strategy"] == "no_weight"]
            if baseline_row.empty:
                continue
            baseline_rmse = baseline_row["fold_rmse_mean"].values[0]

            print(f"\n  {label} (baseline RMSE = {baseline_rmse:.4f}):")
            for _, row in subset.iterrows():
                diff = row["fold_rmse_mean"] - baseline_rmse
                sign = "+" if diff >= 0 else ""
                marker = " <-- baseline" if row["weight_strategy"] == "no_weight" else ""
                print(f"    {row['weight_strategy']:12s}: RMSE={row['fold_rmse_mean']:.4f} "
                      f"({sign}{diff:.4f}){marker}")

    # CSV保存
    csv_path = OUT_DIR / "issue70_sample_weights_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n結果をCSVに保存: {csv_path}")


if __name__ == "__main__":
    main()
