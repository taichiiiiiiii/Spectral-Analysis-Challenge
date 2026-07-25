"""Issue #98: サイクル6 - Adversarial Validation + サンプル重み付け学習

train(13樹種)とtest(6樹種)のドメインシフトを定量化し、
testに似たtrainサンプルに高い重みを付けて学習することでLBスコア改善を狙う。

Step 1: Adversarial Validation (GBCでtrain/test判別 → AUC計測)
Step 2: importance weightingで各trainサンプルに重みを付与
Step 3: 重み付き回帰モデル (PLS+WeightedRidge, WeightedGBR)
Step 4: LOSO-CV + テスト予測 + 提出ファイル生成
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import warnings
import time

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

from src.eda.data_loader import load_train, load_test, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue19_msc import compute_msc_reference
from src.preprocessing.issue62_additional_preprocessing import apply_asls, apply_piecewise_msc
from src.modeling.issue65_feature_selection import sipls_select, ipls_select

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ============================================================
# Step 1: Adversarial Validation
# ============================================================

def compute_adversarial_weights(X_train, X_test, n_pca=50, clip_min=0.1, clip_max=10.0):
    """Adversarial Validationでtrain/testの区別度を測定し、
    各trainサンプルの「testらしさ」に基づく重みを計算する。

    Parameters
    ----------
    X_train : (n_train, n_features) trainスペクトル
    X_test : (n_test, n_features) testスペクトル
    n_pca : int PCA次元数
    clip_min : float 重みの下限
    clip_max : float 重みの上限

    Returns
    -------
    weights : (n_train,) 正規化された重み (平均1)
    auc : float Adversarial Validation AUC
    train_probs : (n_train,) 各trainサンプルの「testらしさ」確率
    """
    X_all = np.vstack([X_train, X_test])
    y_domain = np.array([0] * len(X_train) + [1] * len(X_test))

    # PCA次元削減（1555次元は多すぎる）
    n_comp = min(n_pca, X_all.shape[1], X_all.shape[0] - 1)
    pca = PCA(n_components=n_comp)
    X_all_pca = pca.fit_transform(X_all)

    # GBCで5-fold CVの予測確率を取得
    clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, random_state=42,
        learning_rate=0.1, subsample=0.8
    )
    probs = cross_val_predict(
        clf, X_all_pca, y_domain, cv=5, method='predict_proba'
    )[:, 1]

    train_probs = probs[:len(X_train)]

    # AUCを計算（ドメインシフトの定量化）
    auc = roc_auc_score(y_domain, probs)

    # importance weighting: w = p(test) / p(train) = prob / (1 - prob)
    weights = train_probs / (1 - train_probs + 1e-8)
    weights = np.clip(weights, clip_min, clip_max)
    weights = weights / weights.mean()  # 平均1に正規化

    return weights, auc, train_probs


# ============================================================
# Step 2: 前処理
# ============================================================

def preprocess(X_tr_raw, X_te_raw, groups_train, pp):
    """前処理を適用する。"""
    if pp == "SNV":
        return apply_snv(X_tr_raw), apply_snv(X_te_raw)
    elif pp == "EPO(1)":
        P = compute_epo_projection(X_tr_raw, groups_train, n_components=1)
        return apply_epo(X_tr_raw, P), apply_epo(X_te_raw, P)
    elif pp == "SNV+AsLS":
        return apply_asls(apply_snv(X_tr_raw), lam=1e6), apply_asls(apply_snv(X_te_raw), lam=1e6)
    elif pp == "PMSC":
        ref = compute_msc_reference(X_tr_raw)
        return apply_piecewise_msc(X_tr_raw, ref, 3), apply_piecewise_msc(X_te_raw, ref, 3)
    return X_tr_raw.copy(), X_te_raw.copy()


# ============================================================
# Step 3: 重み付きモデル
# ============================================================

def predict_pls_weighted_ridge(X_tr, X_te, y_train, weights_tr, cfg):
    """PLS次元削減 + 重み付きRidge回帰

    PLSは重み付き学習が直接できないため、PLS変換後にWeighted Ridgeで学習する。

    Parameters
    ----------
    X_tr : (n_tr, n_features)
    X_te : (n_te, n_features)
    y_train : (n_tr,)
    weights_tr : (n_tr,) サンプル重み
    cfg : dict 設定 (nc, tf, alpha)
    """
    nc = cfg.get("nc", 4)
    tf = cfg.get("tf", "sqrt")
    alpha = cfg.get("alpha", 1.0)

    nc = min(nc, X_tr.shape[1] - 1, X_tr.shape[0] - 1)
    nc = max(1, nc)

    y_fit = np.sqrt(y_train) if tf == "sqrt" else y_train.copy()

    pls = PLSRegression(n_components=nc)
    pls.fit(X_tr, y_fit)
    T_tr = pls.transform(X_tr)
    T_te = pls.transform(X_te)

    ridge = Ridge(alpha=alpha)
    ridge.fit(T_tr, y_fit, sample_weight=weights_tr)
    pred = ridge.predict(T_te)

    if tf == "sqrt":
        pred = np.clip(pred, 0, None) ** 2
    return pred


def predict_weighted_gbr(X_tr, X_te, y_train, weights_tr, cfg):
    """PLS次元削減 + 重み付きGBR

    Parameters
    ----------
    X_tr : (n_tr, n_features)
    X_te : (n_te, n_features)
    y_train : (n_tr,)
    weights_tr : (n_tr,) サンプル重み
    cfg : dict 設定
    """
    nc = cfg.get("nc", 4)
    tf = cfg.get("tf", "raw")

    nc = min(nc, X_tr.shape[1] - 1, X_tr.shape[0] - 1)
    nc = max(1, nc)

    y_fit = np.sqrt(y_train) if tf == "sqrt" else y_train.copy()

    pls = PLSRegression(n_components=nc)
    pls.fit(X_tr, y_fit)
    T_tr = pls.transform(X_tr)
    T_te = pls.transform(X_te)

    gbr = GradientBoostingRegressor(
        n_estimators=cfg.get("n_est", 200),
        max_depth=cfg.get("max_depth", 3),
        learning_rate=cfg.get("lr", 0.05),
        subsample=cfg.get("subsample", 0.8),
        min_samples_leaf=cfg.get("min_leaf", 5),
        random_state=42,
        validation_fraction=0.15,
        n_iter_no_change=50,
        tol=0.01,
    )
    gbr.fit(T_tr, y_fit, sample_weight=weights_tr)
    pred = gbr.predict(T_te)

    if tf == "sqrt":
        pred = np.clip(pred, 0, None) ** 2
    return pred


def predict_pls_no_weight(X_tr, X_te, y_train, weights_tr, cfg):
    """重みなしPLS（ベースライン比較用）"""
    nc = cfg.get("nc", 4)
    tf = cfg.get("tf", "sqrt")

    nc = min(nc, X_tr.shape[1] - 1, X_tr.shape[0] - 1)
    nc = max(1, nc)

    y_fit = np.sqrt(y_train) if tf == "sqrt" else y_train.copy()

    pls = PLSRegression(n_components=nc)
    pls.fit(X_tr, y_fit)
    pred = pls.predict(X_te).ravel()

    if tf == "sqrt":
        pred = np.clip(pred, 0, None) ** 2
    return pred


# ============================================================
# Step 4: メイン
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
    print("Issue #98: サイクル6 - Adversarial Validation + サンプル重み付け")
    print(f"  ベースライン: RMSE = 17.03")
    print("=" * 70)

    # ============================================================
    # Step 1: Adversarial Validation（全train vs 全test）
    # ============================================================
    print("\n--- Step 1: Adversarial Validation ---\n")

    weights_full, auc_full, probs_full = compute_adversarial_weights(
        X_raw, X_test_raw, n_pca=50
    )

    print(f"  AUC (全train vs 全test): {auc_full:.4f}")
    print(f"  → AUC=0.5: 区別不可, AUC=1.0: 完全分離")
    print(f"  重み分布: min={weights_full.min():.3f}, max={weights_full.max():.3f}, "
          f"mean={weights_full.mean():.3f}, std={weights_full.std():.3f}")

    # 樹種ごとの平均重み
    print("\n  樹種ごとの平均重み (testらしさ):")
    for sp in np.unique(groups):
        mask = groups == sp
        avg_w = weights_full[mask].mean()
        avg_p = probs_full[mask].mean()
        print(f"    {sp}: 重み={avg_w:.3f}, testらしさ={avg_p:.3f} (n={mask.sum()})")

    # ============================================================
    # Step 2: LOSO-CV評価（事前計算した重みを再利用して高速化）
    # ============================================================
    print("\n--- Step 2: LOSO-CV評価 ---\n")

    # 各foldのadversarial weightsを事前計算（高速化）
    print("  各foldのadversarial weightsを事前計算中...")
    fold_weights = {}
    for f_idx, (train_idx, test_idx) in enumerate(folds):
        w_fold, auc_fold, _ = compute_adversarial_weights(
            X_raw[train_idx], X_test_raw, n_pca=50
        )
        fold_weights[f_idx] = w_fold
        print(f"    fold {f_idx} ({sp_names[f_idx]}): AUC={auc_fold:.4f}", flush=True)
    print("  事前計算完了\n")

    # 前処理結果も事前キャッシュ
    print("  前処理結果を事前キャッシュ中...")
    pp_list = ["EPO(1)", "SNV", "SNV+AsLS", "PMSC"]
    pp_cache = {}
    for pp in pp_list:
        t_pp = time.time()
        pp_cache[pp] = []
        for f_idx, (train_idx, test_idx) in enumerate(folds):
            X_tr_pp, X_te_pp = preprocess(
                X_raw[train_idx], X_raw[test_idx], groups[train_idx], pp
            )
            pp_cache[pp].append((X_tr_pp, X_te_pp))
        print(f"    {pp}: {time.time()-t_pp:.1f}s", flush=True)
    print("  前処理キャッシュ完了\n")

    # モデル構成
    configs = []

    # 前処理 × モデル × 変換の組み合わせ
    for pp in pp_list:
        for tf in ["sqrt", "raw"]:
            # PLS (重みなし baseline)
            configs.append({
                "name": f"PLS:{pp}+PLS4+{tf}(no_weight)",
                "pp": pp, "nc": 4, "tf": tf, "alpha": 1.0,
                "func": predict_pls_no_weight,
                "use_weight": False,
            })

            # PLS + Weighted Ridge
            for alpha in [0.1, 1.0, 10.0]:
                configs.append({
                    "name": f"WRidge:{pp}+PLS4+Ridge(a={alpha})+{tf}",
                    "pp": pp, "nc": 4, "tf": tf, "alpha": alpha,
                    "func": predict_pls_weighted_ridge,
                    "use_weight": True,
                })

            # Weighted GBR (rawの方が相性良いが両方試す)
            configs.append({
                "name": f"WGBR:{pp}+PLS4+GBR+{tf}",
                "pp": pp, "nc": 4, "tf": tf,
                "n_est": 200, "max_depth": 3, "lr": 0.05,
                "func": predict_weighted_gbr,
                "use_weight": True,
            })

            # GBR (重みなし比較)
            configs.append({
                "name": f"GBR:{pp}+PLS4+GBR+{tf}(no_weight)",
                "pp": pp, "nc": 4, "tf": tf,
                "n_est": 200, "max_depth": 3, "lr": 0.05,
                "func": predict_weighted_gbr,
                "use_weight": False,
            })

    n_models = len(configs)
    all_preds = [[] for _ in range(n_models)]
    all_rmses = []

    print(f"  総モデル数: {n_models}")
    print(f"  (重みあり vs なし を比較)\n")

    for m_idx, cfg in enumerate(configs):
        t1 = time.time()
        fold_rmses = []

        for f_idx, (train_idx, test_idx) in enumerate(folds):
            try:
                y_train = y[train_idx]

                # 事前キャッシュから前処理結果を取得
                X_tr_pp, X_te_pp = pp_cache[cfg["pp"]][f_idx]

                # 事前計算した重みを取得
                if cfg["use_weight"]:
                    w_fold = fold_weights[f_idx]
                else:
                    w_fold = np.ones(len(train_idx))

                pred = cfg["func"](X_tr_pp, X_te_pp, y_train, w_fold, cfg)
                all_preds[m_idx].append(pred)
                fold_rmses.append(rmse(y[test_idx], pred))
            except Exception as e:
                fallback = np.full(len(test_idx), y[train_idx].mean())
                all_preds[m_idx].append(fallback)
                fold_rmses.append(999.0)
                print(f"  ERROR {cfg['name']} fold {f_idx} ({sp_names[f_idx]}): {e}")

        mean_r = np.mean(fold_rmses)
        all_rmses.append(mean_r)
        elapsed = time.time() - t1

        # 進捗表示
        nb = [r for sp, r in zip(sp_names, fold_rmses) if sp != "ベイスギ"]
        avg_nb = np.mean(nb) if nb else mean_r
        print(f"  [{m_idx+1:>3}/{n_models}] {cfg['name']}: "
              f"{mean_r:.2f} (除ベイスギ:{avg_nb:.2f}) ({elapsed:.1f}s)", flush=True)

    # ============================================================
    # 結果ランキング
    # ============================================================
    ranking = sorted(range(n_models), key=lambda i: all_rmses[i])

    print(f"\n--- 個別モデル Top 20 ---\n")
    for rank, i in enumerate(ranking[:20]):
        cfg = configs[i]
        fold_rmses = [rmse(y[te], all_preds[i][f]) for f, (_, te) in enumerate(folds)]
        nb = [r for sp, r in zip(sp_names, fold_rmses) if sp != "ベイスギ"]
        w_tag = "重みあり" if cfg["use_weight"] else "重みなし"
        print(f"  {rank+1:>2}. {all_rmses[i]:.2f} (除ベイスギ:{np.mean(nb):.2f}) "
              f"[{w_tag}] | {cfg['name']}")

    # 重みあり vs 重みなし比較
    print(f"\n--- 重みあり vs 重みなし 比較 ---\n")
    weighted_results = [(i, all_rmses[i]) for i in range(n_models) if configs[i]["use_weight"]]
    unweighted_results = [(i, all_rmses[i]) for i in range(n_models) if not configs[i]["use_weight"]]
    weighted_results.sort(key=lambda x: x[1])
    unweighted_results.sort(key=lambda x: x[1])

    print(f"  重みあり ベスト5:")
    for i, r in weighted_results[:5]:
        print(f"    {r:.2f} | {configs[i]['name']}")
    print(f"  重みなし ベスト5:")
    for i, r in unweighted_results[:5]:
        print(f"    {r:.2f} | {configs[i]['name']}")

    # 同一設定の重みあり/なし比較
    print(f"\n--- 同一設定の重み効果 ---\n")
    for pp in ["EPO(1)", "SNV", "SNV+AsLS", "PMSC"]:
        for tf in ["sqrt", "raw"]:
            # PLS baseline
            pls_key = f"PLS:{pp}+PLS4+{tf}(no_weight)"
            # Weighted Ridge (alpha=1.0)
            wridge_key = f"WRidge:{pp}+PLS4+Ridge(a=1.0)+{tf}"
            # GBR no weight
            gbr_nw_key = f"GBR:{pp}+PLS4+GBR+{tf}(no_weight)"
            # GBR weighted
            gbr_w_key = f"WGBR:{pp}+PLS4+GBR+{tf}"

            pls_rmse = next((all_rmses[i] for i in range(n_models) if configs[i]["name"] == pls_key), None)
            wridge_rmse = next((all_rmses[i] for i in range(n_models) if configs[i]["name"] == wridge_key), None)
            gbr_nw_rmse = next((all_rmses[i] for i in range(n_models) if configs[i]["name"] == gbr_nw_key), None)
            gbr_w_rmse = next((all_rmses[i] for i in range(n_models) if configs[i]["name"] == gbr_w_key), None)

            if all(v is not None for v in [pls_rmse, wridge_rmse, gbr_nw_rmse, gbr_w_rmse]):
                print(f"  {pp}+{tf}:")
                print(f"    PLS(no_weight)={pls_rmse:.2f}, WRidge={wridge_rmse:.2f} (差:{wridge_rmse-pls_rmse:+.2f})")
                print(f"    GBR(no_weight)={gbr_nw_rmse:.2f}, WGBR={gbr_w_rmse:.2f} (差:{gbr_w_rmse-gbr_nw_rmse:+.2f})")

    # ============================================================
    # ベスト結果 fold詳細
    # ============================================================
    best_idx = ranking[0]
    best_cfg = configs[best_idx]
    best_rmse = all_rmses[best_idx]

    print(f"\n{'='*70}")
    print(f"ベストモデル: {best_rmse:.4f} ({best_cfg['name']})")
    print(f"ベースライン: 17.03")
    print(f"{'='*70}")

    print(f"\nfold詳細:")
    for f_idx, (_, test_idx) in enumerate(folds):
        r = rmse(y[test_idx], all_preds[best_idx][f_idx])
        tag = " ※参考" if sp_names[f_idx] == "ベイスギ" else ""
        print(f"  {sp_names[f_idx]}: {r:.2f}{tag}")

    non_bs = [rmse(y[te], all_preds[best_idx][f])
              for f, (_, te) in enumerate(folds)
              if sp_names[f] != "ベイスギ"]
    print(f"  ベイスギ除外平均: {np.mean(non_bs):.4f}")

    # ============================================================
    # テスト予測 + 提出ファイル生成
    # ============================================================
    print(f"\n--- テスト予測 ---\n")

    # 全trainで adversarial weight を計算
    weights_all, auc_all, _ = compute_adversarial_weights(X_raw, X_test_raw, n_pca=50)
    print(f"  全データ AUC: {auc_all:.4f}")

    # ベスト設定でテスト予測
    X_tr_pp, X_te_pp = preprocess(X_raw, X_test_raw, groups, best_cfg["pp"])
    w_for_test = weights_all if best_cfg["use_weight"] else np.ones(len(X_raw))

    test_pred = best_cfg["func"](X_tr_pp, X_te_pp, y, w_for_test, best_cfg)
    test_pred = np.clip(test_pred, 0, 300)

    # 提出ファイル生成
    sub = pd.read_csv(DATA_DIR / "sample_submit.csv", header=None)
    sub[1] = sub[1].astype(float)
    sub.iloc[:, 1] = test_pred
    sub_path = OUT_DIR.parent / "submission_v6_advval.csv"
    sub.to_csv(sub_path, index=False, header=False)
    print(f"  提出ファイル: {sub_path}")
    print(f"  予測統計: mean={test_pred.mean():.1f}, std={test_pred.std():.1f}, "
          f"min={test_pred.min():.1f}, max={test_pred.max():.1f}")

    # ============================================================
    # 結果保存
    # ============================================================
    rows = []
    for i in ranking:
        cfg = configs[i]
        fold_rmses = [rmse(y[te], all_preds[i][f]) for f, (_, te) in enumerate(folds)]
        nb = [r for sp, r in zip(sp_names, fold_rmses) if sp != "ベイスギ"]
        rows.append({
            "rank": len(rows) + 1,
            "name": cfg["name"],
            "weighted": cfg["use_weight"],
            "mean_rmse": all_rmses[i],
            "mean_rmse_excl_beisugi": np.mean(nb) if nb else all_rmses[i],
            **{f"fold_{sp}": r for sp, r in zip(sp_names, fold_rmses)},
        })

    result_path = OUT_DIR / "issue98_cycle6_advval_results.csv"
    pd.DataFrame(rows).to_csv(result_path, index=False)
    print(f"\n結果CSV: {result_path}")
    print(f"総実行時間: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
