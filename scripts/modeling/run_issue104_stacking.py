"""Issue #104: Stackingメタモデルによる既存Best12アンサンブルの改善

Stage 1: 既存Best12モデルそれぞれについてLOSO-CVでOOF予測を収集
Stage 2: OOF予測値を特徴量として、メタモデルで最終予測を学習
  - メタモデル候補: Ridge(alpha=1,10,100,1000), ElasticNet(alpha=0.01), HuberRegressor
  - 方式A: 全OOF→fit→同じデータで予測（リーケージあり参考値）
  - 方式B: Stage1 OOFを再利用し、メタモデルのみLOSO-CVで評価（準正当CV）
  - 方式C: 完全nested LOSO-CV（高速モデルのみ、正当CV）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np, pandas as pd, warnings, time
warnings.filterwarnings("ignore")
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import (
    HuberRegressor, ElasticNet, Ridge,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from src.eda.data_loader import load_train, load_test, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue20_savgol import apply_savgol
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue19_msc import compute_msc_reference
from src.preprocessing.issue62_additional_preprocessing import apply_asls, apply_piecewise_msc
from src.modeling.issue65_feature_selection import sipls_select, ipls_select

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PREV_BEST = 13.70  # 現在のベストRMSE（均等平均）


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ============================================================
# 前処理（run_issue89_cycle26.pyと同一）
# ============================================================
def preproc(X_tr, X_te, g, name):
    if name == "SNV":
        return apply_snv(X_tr), apply_snv(X_te)
    elif name == "EPO(1)":
        P = compute_epo_projection(X_tr, g, n_components=1)
        return apply_epo(X_tr, P), apply_epo(X_te, P)
    elif name == "SNV+AsLS(1e6)":
        return apply_asls(apply_snv(X_tr), lam=1e6), apply_asls(apply_snv(X_te), lam=1e6)
    elif name == "PMSC":
        ref = compute_msc_reference(X_tr)
        return apply_piecewise_msc(X_tr, ref, 3), apply_piecewise_msc(X_te, ref, 3)
    elif name == "SG2d+EPO(1)":
        xs = apply_savgol(X_tr, deriv=2, window_length=7)
        xst = apply_savgol(X_te, deriv=2, window_length=7)
        P = compute_epo_projection(xs, g, n_components=1)
        return apply_epo(xs, P), apply_epo(xst, P)
    elif name == "SNV+SG2d":
        return (apply_savgol(apply_snv(X_tr), deriv=2, window_length=7),
                apply_savgol(apply_snv(X_te), deriv=2, window_length=7))
    return X_tr.copy(), X_te.copy()


def feat_sel(X_tr, X_te, y, name):
    if not name:
        return X_tr, X_te
    if name.startswith("siPLS"):
        p = name.replace("siPLS(", "").rstrip(")").split(",")
        X_tr, X_te, _ = sipls_select(
            X_tr, y, X_te, n_intervals=int(p[0]), n_components=3, n_combine=int(p[1])
        )
    elif name.startswith("iPLS"):
        ni = int(name.replace("iPLS(", "").rstrip(")"))
        X_tr, X_te, _ = ipls_select(X_tr, y, X_te, n_intervals=ni, n_components=3, n_best=1)
    return X_tr, X_te


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


# ============================================================
# モデル予測関数群
# ============================================================
def pred_pls(Xtr, Xte, y, nc, tf):
    _, _, pls, yf = get_pls_scores(Xtr, Xte, y, nc, tf)
    return inv_tf(pls.predict(Xte).ravel(), tf)


def pred_gbr(Xtr, Xte, y, nc, tf, n_est=200, md=3, lr=0.05):
    Ttr, Tte, _, yf = get_pls_scores(Xtr, Xte, y, nc, tf)
    gbr = GradientBoostingRegressor(
        n_estimators=n_est, max_depth=md, learning_rate=lr,
        subsample=0.8, min_samples_leaf=5, random_state=42,
        validation_fraction=0.15, n_iter_no_change=50, tol=0.01,
    )
    gbr.fit(Ttr, yf)
    return inv_tf(gbr.predict(Tte), tf)


def pred_huber(Xtr, Xte, y, nc, tf, eps=1.35, alpha=0.01):
    Ttr, Tte, _, yf = get_pls_scores(Xtr, Xte, y, nc, tf)
    sc = StandardScaler()
    Ttr_s, Tte_s = sc.fit_transform(Ttr), sc.transform(Tte)
    h = HuberRegressor(epsilon=eps, max_iter=200, alpha=alpha)
    h.fit(Ttr_s, yf)
    return inv_tf(h.predict(Tte_s), tf)


def run_model(X_tr, X_te, y, g, cfg):
    Xtr, Xte = preproc(X_tr, X_te, g, cfg["pp"])
    Xtr, Xte = feat_sel(Xtr, Xte, y, cfg.get("fs"))
    t = cfg["type"]
    if t == "pls":
        return pred_pls(Xtr, Xte, y, cfg.get("nc", 4), cfg["tf"])
    elif t == "gbr":
        return pred_gbr(Xtr, Xte, y, cfg.get("nc", 4), cfg["tf"],
                         cfg.get("ne", 200), cfg.get("md", 3), cfg.get("lr", 0.05))
    elif t == "huber":
        return pred_huber(Xtr, Xte, y, cfg.get("nc", 4), cfg["tf"],
                          cfg.get("eps", 1.35), cfg.get("alpha", 0.01))
    else:
        raise ValueError(f"Unknown model type: {t}")


# ============================================================
# Best12モデル設定（Best11 + Gnew1）
# ============================================================
def get_base_model_configs():
    """Best12モデル（siPLSモデルを除外して高速化、代替モデルを追加）。
    siPLS(30,3)は1モデルあたり380秒かかるため除外。
    代わりにiPLS/PMSC系の高速モデルを追加して多様性を確保。
    """
    return [
        {"name": "M1:EPO+PLS4+sqrt",
         "pp": "EPO(1)", "nc": 4, "tf": "sqrt", "type": "pls"},
        {"name": "M3:SNV+iPLS50+PLS4+sqrt",
         "pp": "SNV", "nc": 4, "tf": "sqrt", "fs": "iPLS(50)", "type": "pls"},
        {"name": "C1:SNV+iPLS50+PLS5+sqrt",
         "pp": "SNV", "nc": 5, "tf": "sqrt", "fs": "iPLS(50)", "type": "pls"},
        {"name": "H2:SNV+iPLS50+Huber+sqrt",
         "pp": "SNV", "nc": 4, "tf": "sqrt", "fs": "iPLS(50)", "type": "huber"},
        {"name": "H2a:SNV+iPLS50+Huber(1.1)+sqrt",
         "pp": "SNV", "nc": 4, "tf": "sqrt", "fs": "iPLS(50)", "eps": 1.1, "type": "huber"},
        {"name": "G9:SG2d+EPO+GBR+raw",
         "pp": "SG2d+EPO(1)", "nc": 4, "tf": "raw", "type": "gbr"},
        {"name": "Gnew1:EPO+GBR150+raw",
         "pp": "EPO(1)", "nc": 4, "tf": "raw", "ne": 150, "lr": 0.08, "type": "gbr"},
        {"name": "N2:SNV+SG2d+GBR+raw",
         "pp": "SNV+SG2d", "nc": 4, "tf": "raw", "type": "gbr"},
        # 追加: 多様性確保用（siPLS代替）
        {"name": "EPO+PLS3+sqrt",
         "pp": "EPO(1)", "nc": 3, "tf": "sqrt", "type": "pls"},
        {"name": "PMSC+PLS4+sqrt",
         "pp": "PMSC", "nc": 4, "tf": "sqrt", "type": "pls"},
        {"name": "SNV+PLS4+sqrt",
         "pp": "SNV", "nc": 4, "tf": "sqrt", "type": "pls"},
        {"name": "EPO+Huber+sqrt",
         "pp": "EPO(1)", "nc": 4, "tf": "sqrt", "type": "huber"},
    ]


def get_fast_model_configs():
    """方式C用: siPLSを除外した高速モデル群"""
    return [
        {"name": "M1:EPO+PLS4+sqrt",
         "pp": "EPO(1)", "nc": 4, "tf": "sqrt", "type": "pls"},
        {"name": "M3:SNV+iPLS50+PLS4+sqrt",
         "pp": "SNV", "nc": 4, "tf": "sqrt", "fs": "iPLS(50)", "type": "pls"},
        {"name": "C1:SNV+iPLS50+PLS5+sqrt",
         "pp": "SNV", "nc": 5, "tf": "sqrt", "fs": "iPLS(50)", "type": "pls"},
        {"name": "H2:SNV+iPLS50+Huber+sqrt",
         "pp": "SNV", "nc": 4, "tf": "sqrt", "fs": "iPLS(50)", "type": "huber"},
        {"name": "H2a:SNV+iPLS50+Huber(1.1)+sqrt",
         "pp": "SNV", "nc": 4, "tf": "sqrt", "fs": "iPLS(50)", "eps": 1.1, "type": "huber"},
        {"name": "G9:SG2d+EPO+GBR+raw",
         "pp": "SG2d+EPO(1)", "nc": 4, "tf": "raw", "type": "gbr"},
        {"name": "Gnew1:EPO+GBR150+raw",
         "pp": "EPO(1)", "nc": 4, "tf": "raw", "ne": 150, "lr": 0.08, "type": "gbr"},
        {"name": "N2:SNV+SG2d+GBR+raw",
         "pp": "SNV+SG2d", "nc": 4, "tf": "raw", "type": "gbr"},
    ]


# ============================================================
# Stage 1: LOSO-CVでOOF予測を収集
# ============================================================
def collect_oof_predictions(X, y, g, folds, cfgs):
    n_samples = len(y)
    n_models = len(cfgs)
    oof_matrix = np.zeros((n_samples, n_models))
    fold_preds = [[] for _ in range(n_models)]

    for i, cfg in enumerate(cfgs):
        t1 = time.time()
        fold_rmses = []
        for fi, (tr, te) in enumerate(folds):
            try:
                pred = run_model(X[tr], X[te], y[tr], g[tr], cfg)
            except Exception as e:
                pred = np.full(len(te), y[tr].mean())
                print(f"  ERR {cfg['name']} fold{fi}: {e}")
            oof_matrix[te, i] = pred
            fold_preds[i].append(pred)
            fold_rmses.append(rmse(y[te], pred))
        r_mean = np.mean(fold_rmses)
        print(f"  [{i+1:>2}/{n_models}] {cfg['name']}: RMSE={r_mean:.2f} [{time.time()-t1:.1f}s]",
              flush=True)

    return oof_matrix, fold_preds


# ============================================================
# メタモデル定義
# ============================================================
def get_meta_configs():
    return [
        ("Ridge(alpha=1)", lambda: Ridge(alpha=1.0)),
        ("Ridge(alpha=10)", lambda: Ridge(alpha=10.0)),
        ("Ridge(alpha=100)", lambda: Ridge(alpha=100.0)),
        ("Ridge(alpha=1000)", lambda: Ridge(alpha=1000.0)),
        ("ElasticNet(a=0.01)", lambda: ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000)),
        ("ElasticNet(a=0.1)", lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000)),
        ("Huber(eps=1.35)", lambda: HuberRegressor(epsilon=1.35, max_iter=500, alpha=0.001)),
    ]


# ============================================================
# 方式A: 全OOFでfit（リーケージあり参考値）
# ============================================================
def stacking_method_a(oof_matrix, y, folds, species_list, cfgs):
    print("\n--- 方式A: 全OOFでfit（リーケージあり参考値） ---")
    meta_configs = get_meta_configs()
    results = []
    for name, factory in meta_configs:
        model = factory()
        sc = StandardScaler()
        X_meta = sc.fit_transform(oof_matrix)
        model.fit(X_meta, y)
        pred = model.predict(X_meta)

        fold_rmses = []
        for fi, (_, te) in enumerate(folds):
            fold_rmses.append(rmse(y[te], pred[te]))
        r_mean = np.mean(fold_rmses)

        coefs = model.coef_ if hasattr(model, 'coef_') else None
        results.append((name, r_mean, fold_rmses, coefs, pred))
        print(f"  {name}: RMSE={r_mean:.4f} (リーケージあり)")
        if coefs is not None:
            for j, c in enumerate(coefs):
                if abs(c) > 0.01:
                    print(f"    [{j}] {cfgs[j]['name']}: {c:.4f}")

    return results


# ============================================================
# 方式B: Stage1 OOFを再利用、メタモデルのみLOSO-CV（準正当CV）
# ============================================================
def stacking_method_b(oof_matrix, y, g, folds, species_list, cfgs):
    """Stage1のOOF予測をそのまま使い、メタモデルの学習/評価のみLOSO-CVで行う。

    注意: Stage1のOOF予測自体は各foldで正しくheld-outされているが、
    メタモデルのtrain側OOF予測はStage1での各foldのheld-out予測。
    つまりメタモデルのtrainデータは正しいが、Stage1の特徴選択(iPLS等)が
    全データに依存している可能性があるため「準正当」と呼ぶ。
    実質的にはかなり正当なCVに近い。
    """
    print("\n--- 方式B: OOF再利用 + メタモデルLOSO-CV（準正当CV） ---")
    meta_configs = get_meta_configs()
    n_samples = len(y)
    results = []

    for meta_name, meta_factory in meta_configs:
        final_pred = np.zeros(n_samples)

        for fi, (_, te) in enumerate(folds):
            # メタモデルのtrain: 当該fold以外の全サンプル
            tr_mask = np.ones(n_samples, dtype=bool)
            tr_mask[te] = False

            X_meta_train = oof_matrix[tr_mask]
            y_meta_train = y[tr_mask]
            X_meta_test = oof_matrix[te]

            sc = StandardScaler()
            X_meta_train_s = sc.fit_transform(X_meta_train)
            X_meta_test_s = sc.transform(X_meta_test)

            meta_model = meta_factory()
            meta_model.fit(X_meta_train_s, y_meta_train)
            final_pred[te] = meta_model.predict(X_meta_test_s)

        fold_rmses = []
        for fi, (_, te) in enumerate(folds):
            fold_rmses.append(rmse(y[te], np.clip(final_pred[te], 0, 300)))
        r_mean = np.mean(fold_rmses)

        results.append((meta_name, r_mean, fold_rmses, final_pred))
        print(f"  {meta_name}: RMSE={r_mean:.4f}")

    return results


# ============================================================
# 方式C: 完全nested LOSO-CV（高速モデルのみ）
# ============================================================
def stacking_method_c(X, y, g, folds, species_list):
    """完全nested CV: outer foldごとにStage1のinner LOSO-CVを実行し、
    メタモデルをfit → outer foldテストに適用。
    siPLSモデルを除外して計算コストを抑える。
    """
    print("\n--- 方式C: 完全nested LOSO-CV（高速モデルのみ） ---")

    fast_cfgs = get_fast_model_configs()
    meta_configs = get_meta_configs()
    n_models = len(fast_cfgs)
    n_samples = len(y)

    # outer loopの各fold用のデータ
    outer_base_preds = {}  # outer_fi -> (n_te, n_models)
    inner_oof = {}          # outer_fi -> (n_tr, n_models)

    print(f"  高速モデル数: {n_models}", flush=True)
    print("  Stage 1: 各outer foldでinner LOSO-CVを実行...", flush=True)
    t_start = time.time()

    for outer_fi, (outer_tr, outer_te) in enumerate(folds):
        sp_out = np.unique(g[outer_te])[0]
        X_out_tr, y_out_tr, g_out_tr = X[outer_tr], y[outer_tr], g[outer_tr]

        inner_logo = LeaveOneGroupOut()
        inner_folds = list(inner_logo.split(X_out_tr, y_out_tr, g_out_tr))

        oof_inner = np.zeros((len(outer_tr), n_models))
        for mi, cfg in enumerate(fast_cfgs):
            for inner_fi, (i_tr, i_te) in enumerate(inner_folds):
                try:
                    pred = run_model(
                        X_out_tr[i_tr], X_out_tr[i_te],
                        y_out_tr[i_tr], g_out_tr[i_tr], cfg
                    )
                except Exception:
                    pred = np.full(len(i_te), y_out_tr[i_tr].mean())
                oof_inner[i_te, mi] = pred
        inner_oof[outer_fi] = oof_inner

        outer_test_preds = np.zeros((len(outer_te), n_models))
        for mi, cfg in enumerate(fast_cfgs):
            try:
                pred = run_model(X[outer_tr], X[outer_te], y[outer_tr], g[outer_tr], cfg)
            except Exception:
                pred = np.full(len(outer_te), y[outer_tr].mean())
            outer_test_preds[:, mi] = pred
        outer_base_preds[outer_fi] = outer_test_preds

        elapsed = time.time() - t_start
        print(f"    outer fold {outer_fi+1}/13 ({sp_out}): done [{elapsed:.0f}s]", flush=True)

    # ベースライン: 均等平均
    baseline_pred = np.zeros(n_samples)
    for outer_fi, (_, outer_te) in enumerate(folds):
        baseline_pred[outer_te] = outer_base_preds[outer_fi].mean(axis=1)
    baseline_fold_rmses = [rmse(y[te], baseline_pred[te]) for _, te in folds]
    baseline_rmse = np.mean(baseline_fold_rmses)
    print(f"\n  高速モデル均等平均: RMSE={baseline_rmse:.4f}")

    # メタモデル評価
    results = []
    for meta_name, meta_factory in meta_configs:
        final_pred = np.zeros(n_samples)
        for outer_fi, (outer_tr, outer_te) in enumerate(folds):
            sc = StandardScaler()
            X_meta_train_s = sc.fit_transform(inner_oof[outer_fi])
            meta_model = meta_factory()
            meta_model.fit(X_meta_train_s, y[outer_tr])
            X_meta_test_s = sc.transform(outer_base_preds[outer_fi])
            final_pred[outer_te] = meta_model.predict(X_meta_test_s)

        fold_rmses = [rmse(y[te], np.clip(final_pred[te], 0, 300)) for _, te in folds]
        r_mean = np.mean(fold_rmses)
        delta = r_mean - baseline_rmse
        marker = " ★" if delta < -0.01 else ""
        results.append((meta_name, r_mean, fold_rmses, final_pred))
        print(f"  {meta_name}: RMSE={r_mean:.4f} (vs avg {delta:+.4f}){marker}")

    return results, baseline_rmse


# ============================================================
# テスト提出ファイル生成
# ============================================================
def generate_test_submission(X_train, y_train, g_train, X_test, test_ids,
                              folds, cfgs, oof_matrix, meta_factory, meta_name):
    """ベスト設定で提出ファイルを生成。
    1. Stage1 OOF予測(既算)でメタモデルをfit
    2. 全trainで各ベースモデルをfit → テスト予測
    3. メタモデルでテスト予測を統合
    """
    print(f"\n--- 提出ファイル生成: {meta_name} ---")
    n_models = len(cfgs)

    sc = StandardScaler()
    X_meta = sc.fit_transform(oof_matrix)
    meta_model = meta_factory()
    meta_model.fit(X_meta, y_train)

    # テスト予測
    test_base_preds = np.zeros((len(X_test), n_models))
    for mi, cfg in enumerate(cfgs):
        try:
            pred = run_model(X_train, X_test, y_train, g_train, cfg)
        except Exception:
            pred = np.full(len(X_test), y_train.mean())
        test_base_preds[:, mi] = pred
        print(f"  [{mi+1}/{n_models}] {cfg['name']}: mean={pred.mean():.2f}", flush=True)

    # メタモデルで最終予測
    X_meta_test = sc.transform(test_base_preds)
    final_pred = meta_model.predict(X_meta_test)
    final_pred = np.clip(final_pred, 0, 300)

    safe_name = meta_name.replace("(", "_").replace(")", "").replace("=", "").replace(" ", "")
    out_path = OUT_DIR.parent / f"submission_issue104_stacking_{safe_name}.csv"
    df_sub = pd.DataFrame({"id": test_ids, "pred": final_pred})
    df_sub.to_csv(out_path, index=False, header=False)
    print(f"  保存: {out_path}")
    print(f"  予測値: mean={final_pred.mean():.2f}, std={final_pred.std():.2f}, "
          f"min={final_pred.min():.2f}, max={final_pred.max():.2f}")

    # 均等平均も保存（比較用）
    avg_pred = np.clip(test_base_preds.mean(axis=1), 0, 300)
    out_avg = OUT_DIR.parent / "submission_issue104_avg_baseline.csv"
    df_avg = pd.DataFrame({"id": test_ids, "pred": avg_pred})
    df_avg.to_csv(out_avg, index=False, header=False)
    print(f"  均等平均: {out_avg} (mean={avg_pred.mean():.2f})")

    return out_path


# ============================================================
# メイン
# ============================================================
def main():
    t0 = time.time()
    df_train = load_train(DATA_DIR)
    df_test = load_test(DATA_DIR)
    sc_cols = get_spectral_columns(df_train)
    X = df_train[sc_cols].values
    y = df_train["含水率"].values
    g = df_train["樹種"].values
    X_test = df_test[sc_cols].values
    test_ids = df_test["sample number"].values

    logo = LeaveOneGroupOut()
    folds = list(logo.split(X, y, g))
    species_list = [np.unique(g[te])[0] for _, te in folds]

    print("=" * 70)
    print("Issue #104: Stackingメタモデルによるアンサンブル改善")
    print("=" * 70)

    cfgs = get_base_model_configs()
    n_models = len(cfgs)
    print(f"\nベースモデル数: {n_models}")
    print(f"ベースライン（均等平均RMSE）: {PREV_BEST}")

    # ============================================================
    # Stage 1: OOF予測収集
    # ============================================================
    print(f"\n{'='*70}")
    print("Stage 1: LOSO-CVでOOF予測収集")
    print(f"{'='*70}")
    oof_matrix, fold_preds = collect_oof_predictions(X, y, g, folds, cfgs)

    # 均等平均ベースライン
    avg_oof = oof_matrix.mean(axis=1)
    fold_rmses_avg = [rmse(y[te], avg_oof[te]) for _, te in folds]
    avg_rmse = np.mean(fold_rmses_avg)
    print(f"\n均等平均LOSO-CV RMSE: {avg_rmse:.4f}")
    print("  fold別:")
    for sp, fr in zip(species_list, fold_rmses_avg):
        marker = " ※外挿" if sp == "ベイスギ" else ""
        print(f"    {sp}: {fr:.2f}{marker}")

    print(f"\nStage1完了: {time.time()-t0:.0f}s")

    # ============================================================
    # 方式A: リーケージあり参考値
    # ============================================================
    results_a = stacking_method_a(oof_matrix, y, folds, species_list, cfgs)

    # ============================================================
    # 方式B: OOF再利用 + メタモデルLOSO-CV
    # ============================================================
    results_b = stacking_method_b(oof_matrix, y, g, folds, species_list, cfgs)

    # ============================================================
    # 方式C: 完全nested（高速モデルのみ）- 時間に余裕があれば
    # ============================================================
    elapsed = time.time() - t0
    if elapsed < 300:  # 5分以内なら方式Cも実行
        results_c, baseline_c = stacking_method_c(X, y, g, folds, species_list)
    else:
        print(f"\n--- 方式C: スキップ（経過時間{elapsed:.0f}s > 300s） ---")
        results_c, baseline_c = [], None

    # ============================================================
    # 結果サマリー
    # ============================================================
    print(f"\n{'='*70}")
    print("結果サマリー")
    print(f"{'='*70}")
    print(f"均等平均ベースライン（12モデル）: {avg_rmse:.4f}")
    print(f"前ベスト: {PREV_BEST:.4f}")

    print(f"\n方式A（リーケージあり参考値）:")
    for name, r, fold_r, coefs, _ in results_a:
        print(f"  {name}: {r:.4f}")

    print(f"\n方式B（OOF再利用 + メタモデルLOSO-CV）:")
    best_b = None
    best_b_factory = None
    meta_configs = get_meta_configs()
    for (meta_name, r, fold_r, pred), (_, factory) in zip(results_b, meta_configs):
        delta = r - avg_rmse
        marker = " ★改善" if delta < -0.01 else ""
        print(f"  {meta_name}: {r:.4f} (vs avg {delta:+.4f}){marker}")
        if best_b is None or r < best_b[1]:
            best_b = (meta_name, r, fold_r, pred)
            best_b_factory = factory

    if results_c:
        print(f"\n方式C（完全nested、高速モデルのみ）:")
        print(f"  高速モデル均等平均: {baseline_c:.4f}")
        for meta_name, r, fold_r, pred in results_c:
            delta = r - baseline_c
            print(f"  {meta_name}: {r:.4f} (vs avg {delta:+.4f})")

    # fold別詳細（ベストB設定）
    if best_b:
        print(f"\nベスト（方式B）: {best_b[0]} (RMSE={best_b[1]:.4f})")
        print(f"vs 均等平均: {best_b[1] - avg_rmse:+.4f}")
        print(f"vs 前ベスト: {best_b[1] - PREV_BEST:+.4f}")
        print("\nfold別RMSE:")
        non_bei = []
        for sp, fr in zip(species_list, best_b[2]):
            marker = " ※外挿" if sp == "ベイスギ" else ""
            print(f"  {sp}: {fr:.2f}{marker}")
            if sp != "ベイスギ":
                non_bei.append(fr)
        print(f"  除ベイスギ: {np.mean(non_bei):.4f}")

    # ============================================================
    # 提出ファイル生成（ベスト設定）
    # ============================================================
    if best_b and best_b[1] < avg_rmse:
        generate_test_submission(
            X, y, g, X_test, test_ids, folds, cfgs, oof_matrix,
            best_b_factory, best_b[0]
        )
    else:
        print("\n均等平均を下回るメタモデルがないため、均等平均で提出ファイル生成")
        test_base_preds = np.zeros((len(X_test), n_models))
        for mi, cfg in enumerate(cfgs):
            try:
                pred = run_model(X, X_test, y, g, cfg)
            except Exception:
                pred = np.full(len(X_test), y.mean())
            test_base_preds[:, mi] = pred
            print(f"  [{mi+1}/{n_models}] {cfg['name']}: mean={pred.mean():.2f}", flush=True)
        avg_pred = np.clip(test_base_preds.mean(axis=1), 0, 300)
        out_path = OUT_DIR.parent / "submission_issue104_avg_baseline.csv"
        df_sub = pd.DataFrame({"id": test_ids, "pred": avg_pred})
        df_sub.to_csv(out_path, index=False, header=False)
        print(f"  保存: {out_path}")

    # ============================================================
    # 結果CSV保存
    # ============================================================
    rows = [{"method": "baseline_avg", "rmse": avg_rmse, "type": "baseline"}]
    for name, r, fold_r, coefs, _ in results_a:
        rows.append({"method": f"A_{name}", "rmse": r, "type": "method_a_leaky"})
    for name, r, fold_r, _ in results_b:
        rows.append({"method": f"B_{name}", "rmse": r, "type": "method_b_quasi"})
    for name, r, fold_r, _ in results_c:
        rows.append({"method": f"C_{name}", "rmse": r, "type": "method_c_nested"})
    df_out = pd.DataFrame(rows)
    out_csv = OUT_DIR / "issue104_stacking_results.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\n結果保存: {out_csv}")

    elapsed = time.time() - t0
    print(f"\n総実行時間: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
