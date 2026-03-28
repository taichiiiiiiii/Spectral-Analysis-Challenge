"""Issue #104: LightGBM + PLS特徴量 → アンサンブル統合 (最小版)

高速実行: SNVのみ、少数設定、ベースラインは既存提出ファイル活用
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np, pandas as pd, warnings, time
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut
from src.eda.data_loader import load_train, load_test, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue19_msc import compute_msc_reference
from src.preprocessing.issue62_additional_preprocessing import apply_piecewise_msc

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEST11_SUB = Path(__file__).resolve().parents[2] / "outputs" / "submission_best11_rmse13.70.csv"


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def preproc(X_tr, X_te, name):
    if name == "SNV":
        return apply_snv(X_tr), apply_snv(X_te)
    elif name == "PMSC":
        ref = compute_msc_reference(X_tr)
        return apply_piecewise_msc(X_tr, ref, 3), apply_piecewise_msc(X_te, ref, 3)
    return X_tr.copy(), X_te.copy()


def get_pls_scores(X_tr, X_te, y, nc, tf):
    nc = max(1, min(nc, X_tr.shape[1] - 1))
    yf = np.sqrt(y) if tf == "sqrt" else y.copy()
    pls = PLSRegression(n_components=nc)
    pls.fit(X_tr, yf)
    return pls.transform(X_tr), pls.transform(X_te), pls, yf


def inv_tf(pred, tf):
    if tf == "sqrt":
        return np.clip(pred, 0, None) ** 2
    return pred


# LightGBM設定 (最小グリッド: 6設定のみ)
LGBM_CONFIGS = [
    {"name": "LGB:SNV+PLS4+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt"},
    {"name": "LGB:SNV+PLS5+sqrt", "pp": "SNV", "nc": 5, "tf": "sqrt"},
    {"name": "LGB:SNV+PLS6+sqrt", "pp": "SNV", "nc": 6, "tf": "sqrt"},
    {"name": "LGB:SNV+PLS4+raw", "pp": "SNV", "nc": 4, "tf": "raw"},
    {"name": "LGB:PMSC+PLS4+sqrt", "pp": "PMSC", "nc": 4, "tf": "sqrt"},
    {"name": "LGB:PMSC+PLS5+sqrt", "pp": "PMSC", "nc": 5, "tf": "sqrt"},
]


def run_lgbm_fold(Ttr, yf, seed=42):
    """LightGBM 1fold実行"""
    n_val = max(1, int(len(yf) * 0.15))
    idx = np.random.RandomState(seed).permutation(len(yf))
    vi, ti = idx[:n_val], idx[n_val:]
    dtrain = lgb.Dataset(Ttr[ti], label=yf[ti])
    dval = lgb.Dataset(Ttr[vi], label=yf[vi], reference=dtrain)
    params = {"objective": "regression", "metric": "rmse",
              "learning_rate": 0.05, "max_depth": 4, "num_leaves": 15,
              "subsample": 0.8, "colsample_bytree": 0.8,
              "min_child_samples": 5, "random_state": seed, "verbose": -1,
              "n_jobs": 1}
    model = lgb.train(params, dtrain, num_boost_round=500, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(-1)])
    return model


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
    bei_idx = species_list.index("ベイスギ") if "ベイスギ" in species_list else None

    print("=" * 70)
    print("Issue #104: LightGBM + PLS特徴量 (最小版)")
    print("=" * 70)
    sys.stdout.flush()

    # Phase 1: LightGBM LOSO-CV
    print(f"\n--- Phase 1: LightGBM LOSO-CV ({len(LGBM_CONFIGS)}設定) ---\n")
    sys.stdout.flush()

    results = []
    lgbm_oofs = {}

    for ci, cfg in enumerate(LGBM_CONFIGS):
        t1 = time.time()
        oof = np.full(len(y), np.nan)
        fold_rmses = []
        for fi, (tr, te) in enumerate(folds):
            Xtr, Xte = preproc(X[tr], X[te], cfg["pp"])
            Ttr, Tte, _, yf = get_pls_scores(Xtr, Xte, y[tr], cfg["nc"], cfg["tf"])
            model = run_lgbm_fold(Ttr, yf)
            pred = inv_tf(model.predict(Tte), cfg["tf"])
            oof[te] = pred
            fold_rmses.append(rmse(y[te], pred))

        overall = rmse(y, oof)
        bei_r = fold_rmses[bei_idx] if bei_idx is not None else 0
        non_bei = [r for j, r in enumerate(fold_rmses) if j != bei_idx]
        results.append({"name": cfg["name"], "rmse": overall, "bei": bei_r,
                         "non_bei": np.mean(non_bei), "cfg": cfg, "folds": fold_rmses})
        lgbm_oofs[cfg["name"]] = oof
        print(f"  [{ci+1}/{len(LGBM_CONFIGS)}] {cfg['name']}: {overall:.2f} "
              f"(bei={bei_r:.1f}, 除bei={np.mean(non_bei):.2f}) [{time.time()-t1:.1f}s]")
        sys.stdout.flush()

    results.sort(key=lambda x: x["rmse"])
    print(f"\n--- LightGBM ランキング ---")
    for i, r in enumerate(results):
        print(f"  {i+1}. {r['name']}: {r['rmse']:.2f}")
    sys.stdout.flush()

    top3 = results[:3]

    # Phase 2: アンサンブル (既存Best11提出ファイル活用)
    print(f"\n--- Phase 2: アンサンブル評価 ---\n")
    sys.stdout.flush()

    # LightGBM同士
    for n in [2, 3]:
        oof_mean = np.mean([lgbm_oofs[r["name"]] for r in results[:n]], axis=0)
        r = rmse(y, oof_mean)
        print(f"  Top{n} LGB均等平均: {r:.4f}")

    # fold別 (best LGB)
    best_lgb = results[0]
    print(f"\n  ベストLGB fold別:")
    for fi, (_, te) in enumerate(folds):
        r = rmse(y[te], lgbm_oofs[best_lgb["name"]][te])
        print(f"    {species_list[fi]}: {r:.2f}")

    # Phase 3: テスト予測 & 提出
    print(f"\n--- Phase 3: テスト予測 & 提出 ---\n")
    sys.stdout.flush()
    df_test = load_test(DATA_DIR)
    X_test = df_test[sc_cols].values
    test_ids = df_test["sample number"].values

    # Top3 LGBMのテスト予測
    lgbm_test_preds = []
    for r in top3:
        cfg = r["cfg"]
        Xtr, Xte = preproc(X, X_test, cfg["pp"])
        nc = max(1, min(cfg["nc"], Xtr.shape[1] - 1))
        tf = cfg["tf"]
        yf = np.sqrt(y) if tf == "sqrt" else y.copy()
        pls = PLSRegression(n_components=nc)
        pls.fit(Xtr, yf)
        Ttr, Tte = pls.transform(Xtr), pls.transform(Xte)
        model = run_lgbm_fold(Ttr, yf)
        pred = inv_tf(model.predict(Tte), tf)
        lgbm_test_preds.append(pred)
        print(f"  LGB test: {r['name']} mean={pred.mean():.2f}, std={pred.std():.2f}")
        sys.stdout.flush()

    # Best11提出ファイルのテスト予測
    if BEST11_SUB.exists():
        best11_df = pd.read_csv(BEST11_SUB, header=None, names=["id", "pred"])
        best11_test = best11_df["pred"].values
        print(f"  Best11 test loaded: mean={best11_test.mean():.2f}")

        # ブレンド
        lgbm_test_mean = np.mean(lgbm_test_preds[:3], axis=0)
        print(f"\n--- Best11 + LightGBM ブレンド ---")
        for w in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
            blend = (1 - w) * best11_test + w * lgbm_test_mean
            blend = np.clip(blend, 0, 300)
            sub = pd.DataFrame({"id": test_ids, "pred": blend})
            sub_name = f"submission_issue104_b11_lgb_w{w:.2f}.csv"
            sub_path = Path(__file__).resolve().parents[2] / "outputs" / sub_name
            sub.to_csv(sub_path, index=False, header=False)
            print(f"  w_lgb={w:.2f}: mean={blend.mean():.2f}, std={blend.std():.2f} → {sub_name}")
    else:
        print(f"  Best11提出ファイルなし: {BEST11_SUB}")
        lgbm_test_mean = np.mean(lgbm_test_preds, axis=0)

    # LGBMのみの提出
    lgbm_only = np.clip(np.mean(lgbm_test_preds, axis=0), 0, 300)
    sub = pd.DataFrame({"id": test_ids, "pred": lgbm_only})
    sub_path = Path(__file__).resolve().parents[2] / "outputs" / f"submission_issue104_lgbm_only.csv"
    sub.to_csv(sub_path, index=False, header=False)
    print(f"  LGBMのみ: mean={lgbm_only.mean():.2f} → submission_issue104_lgbm_only.csv")

    # 結果CSV
    pd.DataFrame([{"name": r["name"], "rmse": r["rmse"], "bei": r["bei"],
                    "non_bei": r["non_bei"]} for r in results]).to_csv(
        OUT_DIR / "issue104_lgbm_results.csv", index=False)

    print(f"\n{'='*70}")
    print(f"最終結果:")
    print(f"  LightGBM Top3:")
    for i, r in enumerate(top3):
        print(f"    {i+1}. {r['name']}: RMSE={r['rmse']:.2f}")
    print(f"  提出ファイル: outputs/ 配下に生成")
    print(f"  実行時間: {time.time()-t0:.0f}s")
    print(f"{'='*70}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
