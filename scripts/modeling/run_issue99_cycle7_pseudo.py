"""Issue #99: Cycle7 - 反復的Pseudo Labeling

testデータ(550サンプル)はラベルなしだが、スペクトル情報は使える。
現在のベストモデルでtest予測 → 高確信サンプルの疑似ラベルをtrainに追加 → 再学習を反復。
testドメインの分布情報がモデルに組み込まれ、LBが改善する可能性。

LOSO-CVではfold内テストサンプルのスペクトルを「ラベルなしターゲット」として扱う。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import warnings
import time
from itertools import combinations
from scipy.optimize import minimize

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut

from src.eda.data_loader import load_train, load_test, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue20_savgol import apply_savgol
from src.preprocessing.issue62_additional_preprocessing import apply_asls, apply_piecewise_msc
from src.preprocessing.issue19_msc import compute_msc_reference
from src.modeling.issue65_feature_selection import sipls_select, ipls_select

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ============================================================
# 前処理
# ============================================================

def preprocess(X_tr, X_te, groups_tr, pp_name):
    """前処理を適用する。"""
    if pp_name == "SNV":
        return apply_snv(X_tr), apply_snv(X_te)
    elif pp_name == "EPO(1)":
        P = compute_epo_projection(X_tr, groups_tr, n_components=1)
        return apply_epo(X_tr, P), apply_epo(X_te, P)
    elif pp_name == "SNV+AsLS(1e6)":
        return apply_asls(apply_snv(X_tr), lam=1e6), apply_asls(apply_snv(X_te), lam=1e6)
    elif pp_name == "SG2d+EPO(1)":
        xs = apply_savgol(X_tr, deriv=2, window_length=7)
        xst = apply_savgol(X_te, deriv=2, window_length=7)
        P = compute_epo_projection(xs, groups_tr, n_components=1)
        return apply_epo(xs, P), apply_epo(xst, P)
    return X_tr.copy(), X_te.copy()


# ============================================================
# 特徴量選択
# ============================================================

def feature_select(X_tr, X_te, y, fs_name):
    """特徴量選択を適用する。"""
    if not fs_name:
        return X_tr, X_te
    if fs_name.startswith("siPLS"):
        p = fs_name.replace("siPLS(", "").rstrip(")").split(",")
        X_tr, X_te, _ = sipls_select(
            X_tr, y, X_te,
            n_intervals=int(p[0]), n_components=3, n_combine=int(p[1])
        )
    elif fs_name.startswith("iPLS"):
        n = int(fs_name.replace("iPLS(", "").rstrip(")"))
        X_tr, X_te, _ = ipls_select(
            X_tr, y, X_te,
            n_intervals=n, n_components=3, n_best=1
        )
    return X_tr, X_te


# ============================================================
# 個別モデル予測
# ============================================================

def pred_pls(Xtr, Xte, y, nc, tf):
    """PLSモデルで予測"""
    nc = max(1, min(nc, Xtr.shape[1] - 1))
    yf = np.sqrt(y) if tf == "sqrt" else y.copy()
    pls = PLSRegression(n_components=nc)
    pls.fit(Xtr, yf)
    p = pls.predict(Xte).ravel()
    return np.clip(p, 0, None) ** 2 if tf == "sqrt" else p


def pred_huber(Xtr, Xte, y, nc, tf, eps=1.35):
    """PLS次元削減 + Huber回帰で予測"""
    nc = max(1, min(nc, Xtr.shape[1] - 1))
    yf = np.sqrt(y) if tf == "sqrt" else y.copy()
    pls = PLSRegression(n_components=nc)
    pls.fit(Xtr, yf)
    Ttr = pls.transform(Xtr)
    Tte = pls.transform(Xte)
    sc = StandardScaler()
    Ttr_s = sc.fit_transform(Ttr)
    Tte_s = sc.transform(Tte)
    h = HuberRegressor(epsilon=eps, max_iter=200, alpha=0.01)
    h.fit(Ttr_s, yf)
    p = h.predict(Tte_s)
    return np.clip(p, 0, None) ** 2 if tf == "sqrt" else p


# ============================================================
# train_and_predict: 前処理→特徴量選択→モデル予測の統合
# ============================================================

def train_and_predict(X_tr, y_tr, groups_tr, X_te, cfg):
    """1つのモデル構成で学習・予測を行う。

    Parameters
    ----------
    X_tr : (n_train, n_features) 生スペクトル
    y_tr : (n_train,) 含水率
    groups_tr : (n_train,) 樹種ラベル
    X_te : (n_test, n_features) テストスペクトル
    cfg : dict モデル構成

    Returns
    -------
    pred : (n_test,) 予測値
    """
    Xtr, Xte = preprocess(X_tr, X_te, groups_tr, cfg.get("pp", "raw"))
    Xtr, Xte = feature_select(Xtr, Xte, y_tr, cfg.get("fs"))

    t = cfg.get("type", "pls")
    if t == "pls":
        return pred_pls(Xtr, Xte, y_tr, cfg.get("nc", 4), cfg.get("tf", "sqrt"))
    elif t == "huber":
        return pred_huber(
            Xtr, Xte, y_tr, cfg.get("nc", 4), cfg.get("tf", "sqrt"),
            eps=cfg.get("eps", 1.35)
        )
    else:
        raise ValueError(f"Unknown model type: {t}")


# ============================================================
# Pseudo Labeling アンサンブル用モデル構成
# ============================================================

ENSEMBLE_CONFIGS = [
    # 分散計算用: 高速モデルのみ（特徴量選択なし）
    {"name": "E1:EPO+PLS4+sqrt", "pp": "EPO(1)", "nc": 4, "tf": "sqrt", "type": "pls"},
    {"name": "E2:SNV+PLS4+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt", "type": "pls"},
    {"name": "E3:SG2d+EPO+PLS3+raw", "pp": "SG2d+EPO(1)", "nc": 3, "tf": "raw",
     "type": "pls"},
    {"name": "E4:EPO+PLS3+sqrt", "pp": "EPO(1)", "nc": 3, "tf": "sqrt", "type": "pls"},
]


# ============================================================
# 反復的Pseudo Labeling
# ============================================================

def pseudo_label_iteration(X_train, y_train, groups_train, X_test,
                           n_iterations=3, confidence_ratio=0.3,
                           model_configs=None):
    """反復的Pseudo Labeling

    1. 現在のモデルでtest予測
    2. アンサンブル分散が小さい（高確信度）サンプルを選択
    3. 選択サンプルを疑似ラベル付きでtrainに追加
    4. 再学習 → 1に戻る

    Parameters
    ----------
    X_train : (n_train, n_features)
    y_train : (n_train,)
    groups_train : (n_train,)
    X_test : (n_test, n_features)
    n_iterations : int 反復回数
    confidence_ratio : float 高確信サンプルの割合 (0-1)
    model_configs : list[dict] アンサンブル用モデル構成

    Returns
    -------
    X_tr_aug : augmented特徴量
    y_tr_aug : augmented目的変数
    g_tr_aug : augmentedグループラベル
    """
    if model_configs is None:
        model_configs = ENSEMBLE_CONFIGS

    n_select = int(len(X_test) * confidence_ratio)
    if n_select == 0:
        return X_train.copy(), y_train.copy(), groups_train.copy()

    X_tr_aug = X_train.copy()
    y_tr_aug = y_train.copy()
    g_tr_aug = groups_train.copy()

    for iteration in range(n_iterations):
        # 複数モデルで予測（アンサンブル分散を計算）
        predictions = []
        for model_cfg in model_configs:
            try:
                pred = train_and_predict(
                    X_tr_aug, y_tr_aug, g_tr_aug, X_test, model_cfg
                )
                predictions.append(pred)
            except Exception as e:
                # エラー時はスキップ
                pass

        if len(predictions) < 2:
            # 2モデル未満では分散計算不可 → 中断
            break

        preds = np.array(predictions)  # (n_models, n_test)
        mean_pred = preds.mean(axis=0)
        std_pred = preds.std(axis=0)

        # 低分散（高確信度）サンプルを選択
        confident_idx = np.argsort(std_pred)[:n_select]

        # 疑似ラベル付きで追加
        X_pseudo = X_test[confident_idx]
        y_pseudo = mean_pred[confident_idx]
        g_pseudo = np.array(["pseudo"] * len(confident_idx))

        # 毎回元のtrainからリセット
        X_tr_aug = np.vstack([X_train, X_pseudo])
        y_tr_aug = np.concatenate([y_train, y_pseudo])
        g_tr_aug = np.concatenate([groups_train, g_pseudo])

        print(f"    Iter {iteration+1}: {n_select} pseudo samples added, "
              f"pred range [{y_pseudo.min():.1f}, {y_pseudo.max():.1f}], "
              f"mean std={std_pred.mean():.2f}")

    return X_tr_aug, y_tr_aug, g_tr_aug


# ============================================================
# LOSO-CV 評価
# ============================================================

def loso_cv_pseudo(X_raw, y, groups, n_iterations, confidence_ratio,
                   final_cfg, ensemble_cfgs=None):
    """LOSO-CVでPseudo Labeling付きモデルを評価。

    各foldで、テストサンプルを「ラベルなしターゲット」として扱い、
    Pseudo Labelingで学習データを拡張してから最終モデルで予測する。
    """
    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_raw, y, groups))
    sp_names = [np.unique(groups[te])[0] for _, te in folds]

    fold_rmses = []
    fold_preds = []

    for f_idx, (tr_idx, te_idx) in enumerate(folds):
        X_tr, X_te = X_raw[tr_idx], X_raw[te_idx]
        y_tr = y[tr_idx]
        g_tr = groups[tr_idx]

        print(f"  Fold {f_idx} ({sp_names[f_idx]}, n_te={len(te_idx)}):")

        # Pseudo Labeling
        X_tr_aug, y_tr_aug, g_tr_aug = pseudo_label_iteration(
            X_tr, y_tr, g_tr, X_te,
            n_iterations=n_iterations,
            confidence_ratio=confidence_ratio,
            model_configs=ensemble_cfgs,
        )

        # 最終モデルで予測
        pred = train_and_predict(X_tr_aug, y_tr_aug, g_tr_aug, X_te, final_cfg)
        pred = np.clip(pred, 0, 300)

        r = rmse(y[te_idx], pred)
        fold_rmses.append(r)
        fold_preds.append(pred)
        tag = " (参考)" if sp_names[f_idx] == "ベイスギ" else ""
        print(f"    → RMSE = {r:.2f}{tag}")

    mean_r = np.mean(fold_rmses)
    nb = [r for sp, r in zip(sp_names, fold_rmses) if sp != "ベイスギ"]
    mean_nb = np.mean(nb) if nb else mean_r

    return mean_r, mean_nb, fold_rmses, fold_preds, sp_names


# ============================================================
# メイン
# ============================================================

def main():
    t0 = time.time()
    df_train = load_train(DATA_DIR)
    df_test = load_test(DATA_DIR)
    sc = get_spectral_columns(df_train)

    X_raw = df_train[sc].values
    y = df_train["含水率"].values
    groups = df_train["樹種"].values
    X_test_raw = df_test[sc].values

    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_raw, y, groups))
    sp_names = [np.unique(groups[te])[0] for _, te in folds]

    print("=" * 70)
    print("Issue #99: Cycle7 - 反復的Pseudo Labeling")
    print(f"  ベースライン: RMSE = 17.03")
    print("=" * 70)

    # ============================================================
    # ベースモデル（アンサンブル分散計算用）
    # ============================================================
    ensemble_cfgs = ENSEMBLE_CONFIGS

    # ============================================================
    # 最終予測用モデル構成（高速: 特徴量選択なし）
    # ============================================================
    final_cfgs = [
        {"name": "EPO+PLS4+sqrt", "pp": "EPO(1)", "nc": 4, "tf": "sqrt", "type": "pls"},
        {"name": "SNV+PLS4+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt", "type": "pls"},
        {"name": "SG2d+EPO+PLS3+raw", "pp": "SG2d+EPO(1)", "nc": 3, "tf": "raw",
         "type": "pls"},
        {"name": "EPO+PLS3+sqrt", "pp": "EPO(1)", "nc": 3, "tf": "sqrt", "type": "pls"},
        {"name": "SNV+Huber+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt",
         "type": "huber", "eps": 1.35},
    ]

    # 特徴量選択あり最終モデル（ベスト設定のみ後で追加評価）
    final_cfgs_slow = [
        {"name": "SNV+iPLS50+PLS4+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt",
         "fs": "iPLS(50)", "type": "pls"},
        {"name": "SNV+AsLS+siPLS+PLS4+sqrt", "pp": "SNV+AsLS(1e6)", "nc": 4, "tf": "sqrt",
         "fs": "siPLS(30,3)", "type": "pls"},
    ]

    # ============================================================
    # パラメータグリッド（高速探索）
    # ============================================================
    iter_values = [1, 2]
    ratio_values = [0.2, 0.5]

    results = []
    total_runs = len(final_cfgs) * len(iter_values) * len(ratio_values)
    run_count = 0

    print(f"\n総実験数: {total_runs}")
    print(f"  最終モデル: {len(final_cfgs)}")
    print(f"  n_iterations: {iter_values}")
    print(f"  confidence_ratio: {ratio_values}")

    # ============================================================
    # まずベースライン（Pseudo Labelingなし）
    # ============================================================
    print(f"\n{'='*70}")
    print("ベースライン（Pseudo Labelingなし）")
    print(f"{'='*70}")

    baseline_results = []
    for cfg in final_cfgs:
        fold_rmses = []
        for f_idx, (tr_idx, te_idx) in enumerate(folds):
            try:
                pred = train_and_predict(
                    X_raw[tr_idx], y[tr_idx], groups[tr_idx],
                    X_raw[te_idx], cfg
                )
                pred = np.clip(pred, 0, 300)
                fold_rmses.append(rmse(y[te_idx], pred))
            except Exception as e:
                fold_rmses.append(999.0)
                print(f"  ERR {cfg['name']} fold {f_idx}: {e}")

        mean_r = np.mean(fold_rmses)
        nb = [r for sp, r in zip(sp_names, fold_rmses) if sp != "ベイスギ"]
        mean_nb = np.mean(nb) if nb else mean_r
        baseline_results.append((cfg["name"], mean_r, mean_nb))
        print(f"  {cfg['name']}: {mean_r:.2f} (除ベイスギ: {mean_nb:.2f})")

    # ============================================================
    # Pseudo Labeling グリッドサーチ
    # ============================================================
    print(f"\n{'='*70}")
    print("Pseudo Labeling グリッドサーチ")
    print(f"{'='*70}")

    for cfg in final_cfgs:
        for n_iter in iter_values:
            for ratio in ratio_values:
                run_count += 1
                t1 = time.time()
                label = f"{cfg['name']}|iter={n_iter}|ratio={ratio}"
                print(f"\n[{run_count}/{total_runs}] {label}")

                mean_r, mean_nb, fold_rmses, _, _ = loso_cv_pseudo(
                    X_raw, y, groups,
                    n_iterations=n_iter,
                    confidence_ratio=ratio,
                    final_cfg=cfg,
                    ensemble_cfgs=ensemble_cfgs,
                )

                elapsed = time.time() - t1
                print(f"  → RMSE={mean_r:.2f} (除ベイスギ={mean_nb:.2f}) [{elapsed:.0f}s]")

                results.append({
                    "final_model": cfg["name"],
                    "n_iterations": n_iter,
                    "confidence_ratio": ratio,
                    "mean_rmse": mean_r,
                    "mean_rmse_excl_beisugi": mean_nb,
                    **{f"fold_{sp}": r for sp, r in zip(sp_names, fold_rmses)},
                })

    # ============================================================
    # 結果まとめ
    # ============================================================
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("mean_rmse")

    print(f"\n{'='*70}")
    print("結果ランキング Top 20")
    print(f"{'='*70}")
    for i, row in df_results.head(20).iterrows():
        print(f"  {row['mean_rmse']:.2f} (除ベイスギ:{row['mean_rmse_excl_beisugi']:.2f}) "
              f"| {row['final_model']} | iter={row['n_iterations']} ratio={row['confidence_ratio']}")

    # ベスト vs ベースライン
    best = df_results.iloc[0]
    best_baseline_rmse = min(b[1] for b in baseline_results)
    print(f"\nベストPseudo (高速モデル): {best['mean_rmse']:.4f}")
    print(f"ベストベースライン: {best_baseline_rmse:.4f}")
    print(f"改善: {best_baseline_rmse - best['mean_rmse']:+.4f}")
    print(f"全体ベスト: 17.03")

    # ============================================================
    # ベスト iter/ratio で遅いモデル（iPLS/siPLS含む）も評価
    # ============================================================
    best_n_iter_fast = int(best["n_iterations"])
    best_ratio_fast = float(best["confidence_ratio"])

    print(f"\n{'='*70}")
    print(f"特徴量選択ありモデル追加評価 (iter={best_n_iter_fast}, ratio={best_ratio_fast})")
    print(f"{'='*70}")

    for cfg in final_cfgs_slow:
        # ベースライン
        fold_rmses_bl = []
        for f_idx, (tr_idx, te_idx) in enumerate(folds):
            try:
                pred = train_and_predict(
                    X_raw[tr_idx], y[tr_idx], groups[tr_idx], X_raw[te_idx], cfg
                )
                fold_rmses_bl.append(rmse(y[te_idx], np.clip(pred, 0, 300)))
            except Exception as e:
                fold_rmses_bl.append(999.0)
        bl_r = np.mean(fold_rmses_bl)

        # Pseudo Labeling
        t1 = time.time()
        mean_r, mean_nb, fold_rmses_pl, _, _ = loso_cv_pseudo(
            X_raw, y, groups,
            n_iterations=best_n_iter_fast,
            confidence_ratio=best_ratio_fast,
            final_cfg=cfg,
            ensemble_cfgs=ensemble_cfgs,
        )
        elapsed = time.time() - t1
        print(f"  {cfg['name']}: baseline={bl_r:.2f} → pseudo={mean_r:.2f} "
              f"(除ベイスギ={mean_nb:.2f}) [{elapsed:.0f}s]")

        results.append({
            "final_model": cfg["name"],
            "n_iterations": best_n_iter_fast,
            "confidence_ratio": best_ratio_fast,
            "mean_rmse": mean_r,
            "mean_rmse_excl_beisugi": mean_nb,
            **{f"fold_{sp}": r for sp, r in zip(sp_names, fold_rmses_pl)},
        })

    # 再ソート
    df_results = pd.DataFrame(results).sort_values("mean_rmse")
    best = df_results.iloc[0]

    # ============================================================
    # ベスト設定でテスト予測（本番）
    # ============================================================
    print(f"\n{'='*70}")
    print("テスト予測（ベスト設定）")
    print(f"{'='*70}")

    best_final_cfg_name = best["final_model"]
    best_n_iter = int(best["n_iterations"])
    best_ratio = float(best["confidence_ratio"])

    # 最終モデル構成を取得
    best_final_cfg = None
    for cfg in final_cfgs:
        if cfg["name"] == best_final_cfg_name:
            best_final_cfg = cfg
            break

    if best_final_cfg is not None:
        print(f"  最終モデル: {best_final_cfg_name}")
        print(f"  n_iterations: {best_n_iter}")
        print(f"  confidence_ratio: {best_ratio}")

        # テスト予測用Pseudo Labeling
        print(f"\n  Pseudo Labeling (本番):")
        X_tr_aug, y_tr_aug, g_tr_aug = pseudo_label_iteration(
            X_raw, y, groups, X_test_raw,
            n_iterations=best_n_iter,
            confidence_ratio=best_ratio,
            model_configs=ensemble_cfgs,
        )
        print(f"  Augmented train size: {len(y_tr_aug)} (orig: {len(y)})")

        # 最終予測
        final_pred = train_and_predict(
            X_tr_aug, y_tr_aug, g_tr_aug, X_test_raw, best_final_cfg
        )
        final_pred = np.clip(final_pred, 0, 300)

        # 提出ファイル生成
        sub = pd.read_csv(DATA_DIR / "sample_submit.csv", header=None)
        sub.iloc[:, 1] = final_pred
        sub_path = OUT_DIR.parent / "submission_v6_pseudo.csv"
        sub.to_csv(sub_path, index=False, header=False)
        print(f"\n  提出ファイル: {sub_path}")
        print(f"  予測値: min={final_pred.min():.1f}, max={final_pred.max():.1f}, "
              f"mean={final_pred.mean():.1f}")

    # ============================================================
    # 結果CSV保存
    # ============================================================
    result_path = OUT_DIR / "issue99_cycle7_pseudo_results.csv"
    df_results.to_csv(result_path, index=False)
    print(f"\n結果CSV: {result_path}")
    print(f"総実行時間: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
