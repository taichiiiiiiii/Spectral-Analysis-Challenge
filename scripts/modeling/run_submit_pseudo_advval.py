"""Pseudo Labeling + Adversarial Validation重み付けの提出ファイル生成

高速化のため、ベースモデル4つのみ使用し、グリッドサーチなし。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import warnings, time
warnings.filterwarnings("ignore")

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import roc_auc_score
from src.eda.data_loader import load_train, load_test, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue20_savgol import apply_savgol

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs"

def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))

def predict_4models(X_train, y_train, groups, X_test):
    """4モデル均等平均（submission_v5と同じ構成）"""
    preds = []
    # M1: EPO(1)+PLS(4)+sqrt
    P = compute_epo_projection(X_train, groups, 1)
    Xtr, Xte = apply_epo(X_train, P), apply_epo(X_test, P)
    pls = PLSRegression(n_components=4); pls.fit(Xtr, np.sqrt(y_train))
    preds.append(np.clip(pls.predict(Xte).ravel(), 0, None)**2)

    # M2: SG2d+EPO(1)+PLS(3)+raw
    xs = apply_savgol(X_train, deriv=2, window_length=7)
    xst = apply_savgol(X_test, deriv=2, window_length=7)
    P = compute_epo_projection(xs, groups, 1)
    pls = PLSRegression(n_components=3); pls.fit(apply_epo(xs, P), y_train)
    preds.append(pls.predict(apply_epo(xst, P)).ravel())

    # M3, M4はSNV系のため省略（高速化のため2モデルで）
    return np.mean(preds, axis=0)

def main():
    t0 = time.time()
    df_train = load_train(DATA_DIR)
    df_test = load_test(DATA_DIR)
    spec_cols = get_spectral_columns(df_train)
    X = df_train[spec_cols].values
    y = df_train["含水率"].values
    g = df_train["樹種"].values
    X_test = df_test[spec_cols].values
    test_ids = df_test["sample number"].values

    # ============ Part 1: Adversarial Validation ============
    print("=" * 60)
    print("Part 1: Adversarial Validation")
    print("=" * 60)

    X_all = np.vstack([X, X_test])
    y_domain = np.array([0]*len(X) + [1]*len(X_test))

    # PCA次元削減
    pca = PCA(n_components=50, random_state=42)
    X_all_pca = pca.fit_transform(X_all)

    # GBCで判別
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    probs = cross_val_predict(clf, X_all_pca, y_domain, cv=5, method='predict_proba')[:, 1]
    train_probs = probs[:len(X)]

    auc = roc_auc_score(y_domain, probs)
    print(f"AUC: {auc:.4f}")
    print(f"Train probs: min={train_probs.min():.3f}, max={train_probs.max():.3f}, "
          f"mean={train_probs.mean():.3f}")

    # 重み計算: importance weighting
    weights = train_probs / (1 - train_probs + 1e-8)
    weights = np.clip(weights, 0.1, 10)
    weights = weights / weights.mean()
    print(f"Weights: min={weights.min():.3f}, max={weights.max():.3f}")

    # 重み付き学習でテスト予測
    print("\n--- 重み付きモデル ---")
    # EPO + PLS scores + Weighted Ridge
    P = compute_epo_projection(X, g, 1)
    X_epo = apply_epo(X, P)
    X_test_epo = apply_epo(X_test, P)

    for nc in [4, 6]:
        pls = PLSRegression(n_components=nc)
        pls.fit(X_epo, np.sqrt(y))
        T_tr = pls.transform(X_epo)
        T_te = pls.transform(X_test_epo)

        # 重み付きRidge
        ridge = Ridge(alpha=1.0)
        ridge.fit(T_tr, np.sqrt(y), sample_weight=weights)
        pred = np.clip(ridge.predict(T_te), 0, None)**2
        pred = np.clip(pred, 0, 300)

        # 重みなしRidge（比較用）
        ridge_nw = Ridge(alpha=1.0)
        ridge_nw.fit(T_tr, np.sqrt(y))
        pred_nw = np.clip(ridge_nw.predict(T_te), 0, None)**2

        print(f"  nc={nc} weighted: range=[{pred.min():.1f}, {pred.max():.1f}], mean={pred.mean():.1f}")
        print(f"  nc={nc} unweighted: range=[{pred_nw.min():.1f}, {pred_nw.max():.1f}], mean={pred_nw.mean():.1f}")

    # LOSO-CV評価（重み付きRidge）
    logo = LeaveOneGroupOut()
    folds = list(logo.split(X, y, g))
    sp = [np.unique(g[te])[0] for _, te in folds]

    cv_pred_w = np.zeros_like(y)
    cv_pred_nw = np.zeros_like(y)
    for fi, (tr, te) in enumerate(folds):
        P = compute_epo_projection(X[tr], g[tr], 1)
        Xtr, Xte2 = apply_epo(X[tr], P), apply_epo(X[te], P)
        pls = PLSRegression(n_components=4); pls.fit(Xtr, np.sqrt(y[tr]))
        T_tr, T_te = pls.transform(Xtr), pls.transform(Xte2)

        # 重み付き
        ridge = Ridge(alpha=1.0)
        ridge.fit(T_tr, np.sqrt(y[tr]), sample_weight=weights[tr])
        cv_pred_w[te] = np.clip(ridge.predict(T_te), 0, None)**2

        # 重みなし
        ridge2 = Ridge(alpha=1.0)
        ridge2.fit(T_tr, np.sqrt(y[tr]))
        cv_pred_nw[te] = np.clip(ridge2.predict(T_te), 0, None)**2

    print(f"\nCV RMSE weighted: {rmse(y, cv_pred_w):.2f}")
    print(f"CV RMSE unweighted: {rmse(y, cv_pred_nw):.2f}")

    # ============ Part 2: Pseudo Labeling (簡易版) ============
    print("\n" + "=" * 60)
    print("Part 2: Pseudo Labeling (1 iteration, ratio=0.3)")
    print("=" * 60)

    # Step 1: ベースモデル4つでtest予測
    base_preds = []
    for nc, tt in [(4, "sqrt"), (3, "raw"), (6, "sqrt")]:
        P = compute_epo_projection(X, g, 1)
        Xtr, Xte = apply_epo(X, P), apply_epo(X_test, P)
        yf = np.sqrt(y) if tt == "sqrt" else y.copy()
        pls = PLSRegression(n_components=nc); pls.fit(Xtr, yf)
        p = pls.predict(Xte).ravel()
        if tt == "sqrt": p = np.clip(p, 0, None)**2
        base_preds.append(p)

    # SNV版も追加
    Xtr_snv, Xte_snv = apply_snv(X), apply_snv(X_test)
    pls = PLSRegression(n_components=4); pls.fit(Xtr_snv, np.sqrt(y))
    p = np.clip(pls.predict(Xte_snv).ravel(), 0, None)**2
    base_preds.append(p)

    base_preds = np.array(base_preds)
    mean_pred = base_preds.mean(axis=0)
    std_pred = base_preds.std(axis=0)

    # Step 2: 高確信度サンプル選択（分散が小さい上位30%）
    n_select = int(len(X_test) * 0.3)
    confident_idx = np.argsort(std_pred)[:n_select]

    X_pseudo = X_test[confident_idx]
    y_pseudo = mean_pred[confident_idx]
    print(f"Selected {n_select} pseudo samples, "
          f"pred range=[{y_pseudo.min():.1f}, {y_pseudo.max():.1f}], "
          f"mean std of selected={std_pred[confident_idx].mean():.2f}")

    # Step 3: 拡張trainで再学習
    X_aug = np.vstack([X, X_pseudo])
    y_aug = np.concatenate([y, y_pseudo])
    g_aug = np.concatenate([g, np.array(['pseudo'] * len(confident_idx))])

    print(f"Augmented train: {X_aug.shape[0]} samples (original {len(X)} + pseudo {len(X_pseudo)})")

    # EPO(1) + PLS(4) + sqrt で再学習
    P = compute_epo_projection(X_aug, g_aug, 1)
    Xtr_aug = apply_epo(X_aug, P)
    Xte_aug = apply_epo(X_test, P)
    pls = PLSRegression(n_components=4)
    pls.fit(Xtr_aug, np.sqrt(y_aug))
    pred_pseudo = np.clip(pls.predict(Xte_aug).ravel(), 0, None)**2
    pred_pseudo = np.clip(pred_pseudo, 0, 300)

    print(f"Pseudo pred: range=[{pred_pseudo.min():.1f}, {pred_pseudo.max():.1f}], mean={pred_pseudo.mean():.1f}")

    # LOSO-CVでの評価（pseudo labelingあり vs なし）
    cv_base = np.zeros_like(y)
    cv_pseudo = np.zeros_like(y)
    for fi, (tr, te) in enumerate(folds):
        # base
        P = compute_epo_projection(X[tr], g[tr], 1)
        Xtr, Xte2 = apply_epo(X[tr], P), apply_epo(X[te], P)
        pls = PLSRegression(n_components=4); pls.fit(Xtr, np.sqrt(y[tr]))
        cv_base[te] = np.clip(pls.predict(Xte2).ravel(), 0, None)**2

        # pseudo: test fold をtargetとしてpseudo label
        preds_fold = []
        for nc2 in [3, 4, 6]:
            pls2 = PLSRegression(n_components=nc2); pls2.fit(Xtr, np.sqrt(y[tr]))
            preds_fold.append(np.clip(pls2.predict(Xte2).ravel(), 0, None)**2)
        fold_mean = np.mean(preds_fold, axis=0)
        fold_std = np.std(preds_fold, axis=0)
        n_sel = max(1, int(len(te) * 0.3))
        conf_idx = np.argsort(fold_std)[:n_sel]

        X_te_conf = X[te][conf_idx]
        y_te_conf = fold_mean[conf_idx]
        X_aug2 = np.vstack([X[tr], X_te_conf])
        y_aug2 = np.concatenate([y[tr], y_te_conf])
        g_aug2 = np.concatenate([g[tr], np.array(['pseudo']*len(conf_idx))])

        P2 = compute_epo_projection(X_aug2, g_aug2, 1)
        Xtr2 = apply_epo(X_aug2, P2)
        Xte3 = apply_epo(X[te], P2)
        pls3 = PLSRegression(n_components=4); pls3.fit(Xtr2, np.sqrt(y_aug2))
        cv_pseudo[te] = np.clip(pls3.predict(Xte3).ravel(), 0, None)**2

    print(f"\nCV RMSE base: {rmse(y, cv_base):.2f}")
    print(f"CV RMSE pseudo: {rmse(y, cv_pseudo):.2f}")
    for fi, (_, te) in enumerate(folds):
        b = rmse(y[te], cv_base[te])
        p = rmse(y[te], cv_pseudo[te])
        diff = p - b
        print(f"  {sp[fi]}: base={b:.2f}, pseudo={p:.2f}, diff={diff:+.2f}")

    # ============ 提出ファイル生成 ============
    print("\n=== 提出ファイル生成 ===")

    # v5ベースライン（4モデル均等、重みなし）
    pred_base = np.clip(mean_pred, 0, 300)

    # Adversarial重み付き
    P = compute_epo_projection(X, g, 1)
    Xtr, Xte = apply_epo(X, P), apply_epo(X_test, P)
    pls = PLSRegression(n_components=4); pls.fit(Xtr, np.sqrt(y))
    T_tr, T_te = pls.transform(Xtr), pls.transform(Xte)
    ridge = Ridge(alpha=1.0)
    ridge.fit(T_tr, np.sqrt(y), sample_weight=weights)
    pred_advval = np.clip(np.clip(ridge.predict(T_te), 0, None)**2, 0, 300)

    for tag, pred in [
        ("base4", pred_base),
        ("advval", pred_advval),
        ("pseudo", pred_pseudo),
    ]:
        sub = pd.DataFrame({0: test_ids.astype(int), 1: pred})
        path = OUT_DIR / f"submission_v8_{tag}.csv"
        sub.to_csv(path, index=False, header=False)
        print(f"  {path}: range=[{pred.min():.1f}, {pred.max():.1f}], mean={pred.mean():.1f}")

    # 全手法のブレンド
    blend = np.clip((pred_base + pred_advval + pred_pseudo) / 3, 0, 300)
    sub = pd.DataFrame({0: test_ids.astype(int), 1: blend})
    path = OUT_DIR / "submission_v8_blend.csv"
    sub.to_csv(path, index=False, header=False)
    print(f"  {path}: range=[{blend.min():.1f}, {blend.max():.1f}], mean={blend.mean():.1f}")

    print(f"\nTotal: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
