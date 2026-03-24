"""di-PLS + 既存PLS均等平均アンサンブルの提出ファイル生成"""
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
from src.preprocessing.issue43_dipls import fit_predict_dipls
from src.modeling.issue65_feature_selection import sipls_select, ipls_select

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))

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

    all_test_preds = []  # テスト予測を集める
    all_cv_preds = {}    # CV予測

    # ============ 既存PLS 4モデル（submission_v5と同じ） ============

    # M1: EPO(1)+PLS(4)+sqrt
    print("M1: EPO(1)+PLS(4)+sqrt")
    cv_pred = np.zeros_like(y)
    for fi, (tr, te) in enumerate(folds):
        P = compute_epo_projection(X[tr], g[tr], 1)
        Xtr, Xte = apply_epo(X[tr], P), apply_epo(X[te], P)
        pls = PLSRegression(n_components=4); pls.fit(Xtr, np.sqrt(y[tr]))
        cv_pred[te] = np.clip(pls.predict(Xte).ravel(), 0, None)**2
    print(f"  CV RMSE: {rmse(y, cv_pred):.2f}")
    all_cv_preds["M1"] = cv_pred.copy()
    # テスト予測
    P = compute_epo_projection(X, g, 1)
    Xtr, Xte = apply_epo(X, P), apply_epo(X_test, P)
    pls = PLSRegression(n_components=4); pls.fit(Xtr, np.sqrt(y))
    all_test_preds.append(np.clip(pls.predict(Xte).ravel(), 0, None)**2)

    # M2: SG2d+EPO(1)+PLS(3)+raw
    print("M2: SG2d+EPO(1)+PLS(3)+raw")
    cv_pred = np.zeros_like(y)
    for fi, (tr, te) in enumerate(folds):
        xs, xst = apply_savgol(X[tr], deriv=2, window_length=7), apply_savgol(X[te], deriv=2, window_length=7)
        P = compute_epo_projection(xs, g[tr], 1)
        Xtr, Xte2 = apply_epo(xs, P), apply_epo(xst, P)
        pls = PLSRegression(n_components=3); pls.fit(Xtr, y[tr])
        cv_pred[te] = pls.predict(Xte2).ravel()
    print(f"  CV RMSE: {rmse(y, cv_pred):.2f}")
    all_cv_preds["M2"] = cv_pred.copy()
    xs = apply_savgol(X, deriv=2, window_length=7)
    xst = apply_savgol(X_test, deriv=2, window_length=7)
    P = compute_epo_projection(xs, g, 1)
    Xtr, Xte = apply_epo(xs, P), apply_epo(xst, P)
    pls = PLSRegression(n_components=3); pls.fit(Xtr, y)
    all_test_preds.append(pls.predict(Xte).ravel())

    # M3: SNV+AsLS+siPLS(30,3)+PLS(4)+sqrt
    print("M3: SNV+AsLS+siPLS(30,3)+PLS(4)+sqrt")
    cv_pred = np.zeros_like(y)
    for fi, (tr, te) in enumerate(folds):
        Xtr_s, Xte_s = apply_asls(apply_snv(X[tr]), lam=1e6), apply_asls(apply_snv(X[te]), lam=1e6)
        Xtr_sel, Xte_sel, _ = sipls_select(Xtr_s, y[tr], Xte_s, n_intervals=30, n_components=3, n_combine=3)
        nc = min(4, Xtr_sel.shape[1]-1)
        pls = PLSRegression(n_components=max(1,nc)); pls.fit(Xtr_sel, np.sqrt(y[tr]))
        cv_pred[te] = np.clip(pls.predict(Xte_sel).ravel(), 0, None)**2
    print(f"  CV RMSE: {rmse(y, cv_pred):.2f}")
    all_cv_preds["M3"] = cv_pred.copy()
    Xtr_s = apply_asls(apply_snv(X), lam=1e6)
    Xte_s = apply_asls(apply_snv(X_test), lam=1e6)
    Xtr_sel, Xte_sel, _ = sipls_select(Xtr_s, y, Xte_s, n_intervals=30, n_components=3, n_combine=3)
    nc = min(4, Xtr_sel.shape[1]-1)
    pls = PLSRegression(n_components=max(1,nc)); pls.fit(Xtr_sel, np.sqrt(y))
    all_test_preds.append(np.clip(pls.predict(Xte_sel).ravel(), 0, None)**2)

    # M4: SNV+iPLS(50)+PLS(4)+sqrt
    print("M4: SNV+iPLS(50)+PLS(4)+sqrt")
    cv_pred = np.zeros_like(y)
    for fi, (tr, te) in enumerate(folds):
        Xtr_snv, Xte_snv = apply_snv(X[tr]), apply_snv(X[te])
        Xtr_sel, Xte_sel, _ = ipls_select(Xtr_snv, y[tr], Xte_snv, n_intervals=50, n_components=3, n_best=1)
        nc = min(4, Xtr_sel.shape[1]-1)
        pls = PLSRegression(n_components=max(1,nc)); pls.fit(Xtr_sel, np.sqrt(y[tr]))
        cv_pred[te] = np.clip(pls.predict(Xte_sel).ravel(), 0, None)**2
    print(f"  CV RMSE: {rmse(y, cv_pred):.2f}")
    all_cv_preds["M4"] = cv_pred.copy()
    Xtr_snv, Xte_snv = apply_snv(X), apply_snv(X_test)
    Xtr_sel, Xte_sel, _ = ipls_select(Xtr_snv, y, Xte_snv, n_intervals=50, n_components=3, n_best=1)
    nc = min(4, Xtr_sel.shape[1]-1)
    pls = PLSRegression(n_components=max(1,nc)); pls.fit(Xtr_sel, np.sqrt(y))
    all_test_preds.append(np.clip(pls.predict(Xte_sel).ravel(), 0, None)**2)

    # ============ di-PLS モデル群 ============

    # di-PLSはsource=train, target=testでドメイン適応
    # LOSO-CVではfold内test樹種をtargetとして扱う

    dipls_configs = [
        ("diPLS_SNV_nc3_l1_sqrt", "snv", 3, 1.0, "sqrt"),
        ("diPLS_SNV_nc4_l1_sqrt", "snv", 4, 1.0, "sqrt"),
        ("diPLS_SNV_nc4_l10_sqrt", "snv", 4, 10.0, "sqrt"),
        ("diPLS_EPO_nc4_l1_sqrt", "epo", 4, 1.0, "sqrt"),
        ("diPLS_SNV_nc3_l5_raw", "snv", 3, 5.0, "raw"),
        ("diPLS_raw_nc4_l1_sqrt", "raw", 4, 1.0, "sqrt"),
    ]

    for name, pp, nc, lam, tt in dipls_configs:
        print(f"{name}")
        cv_pred = np.zeros_like(y)
        for fi, (tr, te) in enumerate(folds):
            if pp == "snv":
                Xtr, Xte2 = apply_snv(X[tr]), apply_snv(X[te])
            elif pp == "epo":
                P = compute_epo_projection(X[tr], g[tr], 1)
                Xtr, Xte2 = apply_epo(X[tr], P), apply_epo(X[te], P)
            else:
                Xtr, Xte2 = X[tr].copy(), X[te].copy()

            ytr = np.sqrt(y[tr]) if tt == "sqrt" else y[tr].copy()
            pred = fit_predict_dipls(Xtr, ytr, Xte2, n_components=nc, dipls_lambda=lam)
            if tt == "sqrt":
                pred = np.clip(pred, 0, None)**2
            cv_pred[te] = pred

        cv_rmse = rmse(y, cv_pred)
        print(f"  CV RMSE: {cv_rmse:.2f}")
        all_cv_preds[name] = cv_pred.copy()

        # テスト予測
        if pp == "snv":
            Xtr, Xte2 = apply_snv(X), apply_snv(X_test)
        elif pp == "epo":
            P = compute_epo_projection(X, g, 1)
            Xtr, Xte2 = apply_epo(X, P), apply_epo(X_test, P)
        else:
            Xtr, Xte2 = X.copy(), X_test.copy()

        ytr = np.sqrt(y) if tt == "sqrt" else y.copy()
        pred = fit_predict_dipls(Xtr, ytr, Xte2, n_components=nc, dipls_lambda=lam)
        if tt == "sqrt":
            pred = np.clip(pred, 0, None)**2
        all_test_preds.append(pred)

    # ============ アンサンブル ============

    n_models = len(all_test_preds)
    print(f"\n=== アンサンブル ({n_models}モデル) ===")

    # 均等平均
    avg_pred = np.clip(np.mean(all_test_preds, axis=0), 0, 300)

    # PLS 4モデルのみ均等平均
    pls_avg = np.clip(np.mean(all_test_preds[:4], axis=0), 0, 300)

    # di-PLSのみ均等平均
    dipls_avg = np.clip(np.mean(all_test_preds[4:], axis=0), 0, 300)

    # CV評価
    for name, preds_dict in [("全モデル均等", all_cv_preds),]:
        vals = list(preds_dict.values())
        avg = np.mean(vals, axis=0)
        print(f"  {name} CV RMSE: {rmse(y, avg):.2f}")
        # fold別
        for fi, (_, te) in enumerate(folds):
            fr = rmse(y[te], avg[te])
            print(f"    {sp[fi]}: {fr:.2f}")

    # 提出ファイル
    for tag, pred in [("all_equal", avg_pred), ("pls4_equal", pls_avg), ("dipls_equal", dipls_avg)]:
        sub = pd.DataFrame({0: test_ids.astype(int), 1: pred})
        path = OUT_DIR / f"submission_v8_{tag}.csv"
        sub.to_csv(path, index=False, header=False)
        print(f"\nSaved: {path}")
        print(f"  range: [{pred.min():.1f}, {pred.max():.1f}], mean: {pred.mean():.1f}")

    print(f"\nTotal time: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
