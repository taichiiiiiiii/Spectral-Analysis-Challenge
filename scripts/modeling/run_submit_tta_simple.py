"""TTA + シンプルモデルの提出ファイル生成

戦略:
1. シンプルなPLSモデル（PLS成分数2-3）は汎化しやすい可能性
2. TTAで予測を安定化
3. クリッピング範囲の最適化
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import warnings, time
warnings.filterwarnings("ignore")

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut
from src.eda.data_loader import load_train, load_test, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue20_savgol import apply_savgol
from src.preprocessing.issue62_additional_preprocessing import apply_asls

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs"

def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))

def tta_predict(X_train, y_train, groups, X_test, model_func, n_aug=20, seed=42):
    """TTA: テストスペクトルにノイズを加えて予測平均化"""
    rng = np.random.RandomState(seed)
    all_preds = []

    # 元のスペクトルで予測
    pred_orig = model_func(X_train, y_train, groups, X_test)
    all_preds.append(pred_orig)

    for i in range(n_aug):
        X_aug = X_test.copy()
        # ベースラインオフセット
        X_aug += rng.normal(0, 0.001, size=(X_test.shape[0], 1))
        # 乗算ノイズ
        X_aug *= rng.normal(1.0, 0.002, size=(X_test.shape[0], 1))
        # ガウシアンノイズ
        X_aug += rng.normal(0, 0.0005, size=X_test.shape)

        pred = model_func(X_train, y_train, groups, X_aug)
        all_preds.append(pred)

    return np.mean(all_preds, axis=0)

def model_epo_pls(X_train, y_train, groups, X_test, nc=4, tf="sqrt"):
    P = compute_epo_projection(X_train, groups, 1)
    Xtr, Xte = apply_epo(X_train, P), apply_epo(X_test, P)
    yf = np.sqrt(y_train) if tf == "sqrt" else y_train
    pls = PLSRegression(n_components=nc); pls.fit(Xtr, yf)
    p = pls.predict(Xte).ravel()
    return np.clip(p, 0, None)**2 if tf == "sqrt" else p

def model_snv_pls(X_train, y_train, groups, X_test, nc=4, tf="sqrt"):
    Xtr, Xte = apply_snv(X_train), apply_snv(X_test)
    yf = np.sqrt(y_train) if tf == "sqrt" else y_train
    pls = PLSRegression(n_components=nc); pls.fit(Xtr, yf)
    p = pls.predict(Xte).ravel()
    return np.clip(p, 0, None)**2 if tf == "sqrt" else p

def model_sg2d_epo_pls(X_train, y_train, groups, X_test, nc=3, tf="raw"):
    xs = apply_savgol(X_train, deriv=2, window_length=7)
    xst = apply_savgol(X_test, deriv=2, window_length=7)
    P = compute_epo_projection(xs, groups, 1)
    Xtr, Xte = apply_epo(xs, P), apply_epo(xst, P)
    yf = np.sqrt(y_train) if tf == "sqrt" else y_train
    pls = PLSRegression(n_components=nc); pls.fit(Xtr, yf)
    p = pls.predict(Xte).ravel()
    return np.clip(p, 0, None)**2 if tf == "sqrt" else p

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

    logo = LeaveOneGroupOut()
    folds = list(logo.split(X, y, g))
    sp = [np.unique(g[te])[0] for _, te in folds]

    # ============ Part 1: シンプルモデル比較 ============
    print("=" * 60)
    print("Part 1: PLS成分数の影響（少ない方が汎化する？）")
    print("=" * 60)

    test_preds = {}

    for nc in [2, 3, 4, 5, 6]:
        cv_pred = np.zeros_like(y)
        for fi, (tr, te) in enumerate(folds):
            cv_pred[te] = model_epo_pls(X[tr], y[tr], g[tr], X[te], nc=nc)
        cv = rmse(y, cv_pred)
        # テスト予測
        test_preds[f"EPO_PLS{nc}_sqrt"] = model_epo_pls(X, y, g, X_test, nc=nc)
        print(f"  EPO+PLS({nc})+sqrt: CV={cv:.2f}, test_mean={test_preds[f'EPO_PLS{nc}_sqrt'].mean():.1f}")

    for nc in [2, 3, 4]:
        cv_pred = np.zeros_like(y)
        for fi, (tr, te) in enumerate(folds):
            cv_pred[te] = model_snv_pls(X[tr], y[tr], g[tr], X[te], nc=nc)
        cv = rmse(y, cv_pred)
        test_preds[f"SNV_PLS{nc}_sqrt"] = model_snv_pls(X, y, g, X_test, nc=nc)
        print(f"  SNV+PLS({nc})+sqrt: CV={cv:.2f}, test_mean={test_preds[f'SNV_PLS{nc}_sqrt'].mean():.1f}")

    for nc in [2, 3]:
        cv_pred = np.zeros_like(y)
        for fi, (tr, te) in enumerate(folds):
            cv_pred[te] = model_sg2d_epo_pls(X[tr], y[tr], g[tr], X[te], nc=nc)
        cv = rmse(y, cv_pred)
        test_preds[f"SG2d_EPO_PLS{nc}_raw"] = model_sg2d_epo_pls(X, y, g, X_test, nc=nc)
        print(f"  SG2d+EPO+PLS({nc})+raw: CV={cv:.2f}, test_mean={test_preds[f'SG2d_EPO_PLS{nc}_raw'].mean():.1f}")

    # ============ Part 2: TTA ============
    print("\n" + "=" * 60)
    print("Part 2: TTA (n_aug=20)")
    print("=" * 60)

    # TTA with EPO+PLS4+sqrt
    pred_no_tta = model_epo_pls(X, y, g, X_test, nc=4)
    pred_tta = tta_predict(X, y, g, X_test,
                           lambda Xtr, ytr, gr, Xte: model_epo_pls(Xtr, ytr, gr, Xte, nc=4),
                           n_aug=20)
    print(f"  EPO+PLS4+sqrt no TTA: mean={pred_no_tta.mean():.1f}, std={pred_no_tta.std():.1f}")
    print(f"  EPO+PLS4+sqrt with TTA: mean={pred_tta.mean():.1f}, std={pred_tta.std():.1f}")
    print(f"  Pred diff (TTA-noTTA): mean={np.abs(pred_tta-pred_no_tta).mean():.2f}, max={np.abs(pred_tta-pred_no_tta).max():.2f}")

    # TTA with 4モデルアンサンブル
    tta_preds = []
    models = [
        ("EPO+PLS4+sqrt", lambda Xtr,ytr,gr,Xte: model_epo_pls(Xtr,ytr,gr,Xte,nc=4)),
        ("SG2d+EPO+PLS3+raw", lambda Xtr,ytr,gr,Xte: model_sg2d_epo_pls(Xtr,ytr,gr,Xte,nc=3)),
        ("SNV+PLS4+sqrt", lambda Xtr,ytr,gr,Xte: model_snv_pls(Xtr,ytr,gr,Xte,nc=4)),
        ("EPO+PLS3+sqrt", lambda Xtr,ytr,gr,Xte: model_epo_pls(Xtr,ytr,gr,Xte,nc=3)),
    ]

    for name, mfunc in models:
        p = tta_predict(X, y, g, X_test, mfunc, n_aug=20)
        tta_preds.append(p)
        print(f"  TTA {name}: mean={p.mean():.1f}")

    tta_ensemble = np.clip(np.mean(tta_preds, axis=0), 0, 300)

    # ============ Part 3: クリッピング最適化 ============
    print("\n" + "=" * 60)
    print("Part 3: テスト予測統計・クリッピング")
    print("=" * 60)

    for clip_max in [100, 150, 200, 250, 300]:
        pred_clipped = np.clip(tta_ensemble, 0, clip_max)
        print(f"  clip_max={clip_max}: mean={pred_clipped.mean():.1f}, max={pred_clipped.max():.1f}")

    # ============ 提出ファイル生成 ============
    print("\n=== 提出ファイル ===")

    # シンプルモデル平均
    simple_avg = np.clip(np.mean([
        test_preds["EPO_PLS3_sqrt"],
        test_preds["EPO_PLS4_sqrt"],
        test_preds["SNV_PLS3_sqrt"],
        test_preds["SG2d_EPO_PLS3_raw"],
    ], axis=0), 0, 300)

    for tag, pred in [
        ("tta_ensemble", tta_ensemble),
        ("simple_avg", simple_avg),
        ("epo_pls3", np.clip(test_preds["EPO_PLS3_sqrt"], 0, 300)),
    ]:
        sub = pd.DataFrame({0: test_ids.astype(int), 1: pred})
        path = OUT_DIR / f"submission_v8_{tag}.csv"
        sub.to_csv(path, index=False, header=False)
        print(f"  {path}: range=[{pred.min():.1f}, {pred.max():.1f}], mean={pred.mean():.1f}")

    print(f"\nTotal: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
