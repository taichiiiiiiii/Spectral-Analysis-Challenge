"""di-PLS + 既存PLS均等平均アンサンブルの提出ファイル生成
中間結果をnpzで保存し、段階的に実行可能にする。
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
from src.preprocessing.issue43_dipls import fit_predict_dipls
from src.modeling.issue65_feature_selection import sipls_select, ipls_select

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE = OUT_DIR / "dipls_ensemble_cache.npz"

def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))

def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
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
    n_test = X_test.shape[0]

    # Load cache if exists
    cached = {}
    if CACHE.exists():
        data = np.load(CACHE, allow_pickle=True)
        for k in data.files:
            cached[k] = data[k]
        print(f"Loaded cache with {len(cached)//2} models", flush=True)

    def save_cache():
        np.savez(CACHE, **cached)

    if stage in ("pls", "all"):
        # M1: EPO(1)+PLS(4)+sqrt
        if "M1_cv" not in cached:
            print("M1: EPO(1)+PLS(4)+sqrt", flush=True)
            cv_pred = np.zeros_like(y)
            for fi, (tr, te) in enumerate(folds):
                P = compute_epo_projection(X[tr], g[tr], 1)
                Xtr, Xte = apply_epo(X[tr], P), apply_epo(X[te], P)
                pls = PLSRegression(n_components=4); pls.fit(Xtr, np.sqrt(y[tr]))
                cv_pred[te] = np.clip(pls.predict(Xte).ravel(), 0, None)**2
            print(f"  CV RMSE: {rmse(y, cv_pred):.2f}", flush=True)
            P = compute_epo_projection(X, g, 1)
            Xtr, Xte = apply_epo(X, P), apply_epo(X_test, P)
            pls = PLSRegression(n_components=4); pls.fit(Xtr, np.sqrt(y))
            test_pred = np.clip(pls.predict(Xte).ravel(), 0, None)**2
            cached["M1_cv"] = cv_pred; cached["M1_test"] = test_pred
            save_cache()

        # M2: SG2d+EPO(1)+PLS(3)+raw
        if "M2_cv" not in cached:
            print("M2: SG2d+EPO(1)+PLS(3)+raw", flush=True)
            cv_pred = np.zeros_like(y)
            for fi, (tr, te) in enumerate(folds):
                xs = apply_savgol(X[tr], deriv=2, window_length=7)
                xst = apply_savgol(X[te], deriv=2, window_length=7)
                P = compute_epo_projection(xs, g[tr], 1)
                Xtr, Xte2 = apply_epo(xs, P), apply_epo(xst, P)
                pls = PLSRegression(n_components=3); pls.fit(Xtr, y[tr])
                cv_pred[te] = pls.predict(Xte2).ravel()
            print(f"  CV RMSE: {rmse(y, cv_pred):.2f}", flush=True)
            xs = apply_savgol(X, deriv=2, window_length=7)
            xst = apply_savgol(X_test, deriv=2, window_length=7)
            P = compute_epo_projection(xs, g, 1)
            Xtr, Xte = apply_epo(xs, P), apply_epo(xst, P)
            pls = PLSRegression(n_components=3); pls.fit(Xtr, y)
            test_pred = pls.predict(Xte).ravel()
            cached["M2_cv"] = cv_pred; cached["M2_test"] = test_pred
            save_cache()

        # M3: SNV+AsLS+siPLS(30,3)+PLS(4)+sqrt
        if "M3_cv" not in cached:
            print("M3: SNV+AsLS+siPLS(30,3)+PLS(4)+sqrt", flush=True)
            cv_pred = np.zeros_like(y)
            for fi, (tr, te) in enumerate(folds):
                print(f"  fold {fi} ({sp[fi]})...", flush=True)
                Xtr_s = apply_asls(apply_snv(X[tr]), lam=1e6)
                Xte_s = apply_asls(apply_snv(X[te]), lam=1e6)
                Xtr_sel, Xte_sel, _ = sipls_select(Xtr_s, y[tr], Xte_s, n_intervals=30, n_components=3, n_combine=3)
                nc = min(4, Xtr_sel.shape[1]-1)
                pls = PLSRegression(n_components=max(1,nc)); pls.fit(Xtr_sel, np.sqrt(y[tr]))
                cv_pred[te] = np.clip(pls.predict(Xte_sel).ravel(), 0, None)**2
            print(f"  CV RMSE: {rmse(y, cv_pred):.2f}", flush=True)
            Xtr_s = apply_asls(apply_snv(X), lam=1e6)
            Xte_s = apply_asls(apply_snv(X_test), lam=1e6)
            Xtr_sel, Xte_sel, _ = sipls_select(Xtr_s, y, Xte_s, n_intervals=30, n_components=3, n_combine=3)
            nc = min(4, Xtr_sel.shape[1]-1)
            pls = PLSRegression(n_components=max(1,nc)); pls.fit(Xtr_sel, np.sqrt(y))
            test_pred = np.clip(pls.predict(Xte_sel).ravel(), 0, None)**2
            cached["M3_cv"] = cv_pred; cached["M3_test"] = test_pred
            save_cache()

        # M4: SNV+iPLS(50)+PLS(4)+sqrt
        if "M4_cv" not in cached:
            print("M4: SNV+iPLS(50)+PLS(4)+sqrt", flush=True)
            cv_pred = np.zeros_like(y)
            for fi, (tr, te) in enumerate(folds):
                Xtr_snv, Xte_snv = apply_snv(X[tr]), apply_snv(X[te])
                Xtr_sel, Xte_sel, _ = ipls_select(Xtr_snv, y[tr], Xte_snv, n_intervals=50, n_components=3, n_best=1)
                nc = min(4, Xtr_sel.shape[1]-1)
                pls = PLSRegression(n_components=max(1,nc)); pls.fit(Xtr_sel, np.sqrt(y[tr]))
                cv_pred[te] = np.clip(pls.predict(Xte_sel).ravel(), 0, None)**2
            print(f"  CV RMSE: {rmse(y, cv_pred):.2f}", flush=True)
            Xtr_snv, Xte_snv = apply_snv(X), apply_snv(X_test)
            Xtr_sel, Xte_sel, _ = ipls_select(Xtr_snv, y, Xte_snv, n_intervals=50, n_components=3, n_best=1)
            nc = min(4, Xtr_sel.shape[1]-1)
            pls = PLSRegression(n_components=max(1,nc)); pls.fit(Xtr_sel, np.sqrt(y))
            test_pred = np.clip(pls.predict(Xte_sel).ravel(), 0, None)**2
            cached["M4_cv"] = cv_pred; cached["M4_test"] = test_pred
            save_cache()

        print(f"PLS models done in {time.time()-t0:.0f}s", flush=True)

    if stage in ("dipls", "all"):
        dipls_configs = [
            ("diPLS_SNV_nc3_l1_sqrt", "snv", 3, 1.0, "sqrt"),
            ("diPLS_SNV_nc4_l1_sqrt", "snv", 4, 1.0, "sqrt"),
            ("diPLS_SNV_nc4_l10_sqrt", "snv", 4, 10.0, "sqrt"),
            ("diPLS_EPO_nc4_l1_sqrt", "epo", 4, 1.0, "sqrt"),
            ("diPLS_SNV_nc3_l5_raw", "snv", 3, 5.0, "raw"),
            ("diPLS_raw_nc4_l1_sqrt", "raw", 4, 1.0, "sqrt"),
        ]

        for name, pp, nc, lam, tt in dipls_configs:
            if f"{name}_cv" in cached:
                print(f"{name}: cached, CV RMSE={rmse(y, cached[f'{name}_cv']):.2f}", flush=True)
                continue
            t1 = time.time()
            print(f"{name}", flush=True)
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

            print(f"  CV RMSE: {rmse(y, cv_pred):.2f} ({time.time()-t1:.0f}s)", flush=True)

            if pp == "snv":
                Xtr, Xte2 = apply_snv(X), apply_snv(X_test)
            elif pp == "epo":
                P = compute_epo_projection(X, g, 1)
                Xtr, Xte2 = apply_epo(X, P), apply_epo(X_test, P)
            else:
                Xtr, Xte2 = X.copy(), X_test.copy()

            ytr = np.sqrt(y) if tt == "sqrt" else y.copy()
            test_pred = fit_predict_dipls(Xtr, ytr, Xte2, n_components=nc, dipls_lambda=lam)
            if tt == "sqrt":
                test_pred = np.clip(test_pred, 0, None)**2

            cached[f"{name}_cv"] = cv_pred
            cached[f"{name}_test"] = test_pred
            save_cache()

        print(f"di-PLS models done in {time.time()-t0:.0f}s", flush=True)

    if stage in ("ensemble", "all"):
        # Collect all models
        model_order = ["M1", "M2", "M3", "M4",
                       "diPLS_SNV_nc3_l1_sqrt", "diPLS_SNV_nc4_l1_sqrt",
                       "diPLS_SNV_nc4_l10_sqrt", "diPLS_EPO_nc4_l1_sqrt",
                       "diPLS_SNV_nc3_l5_raw", "diPLS_raw_nc4_l1_sqrt"]

        available = [m for m in model_order if f"{m}_cv" in cached]
        print(f"\n=== アンサンブル ({len(available)}モデル) ===", flush=True)

        for m in available:
            print(f"  {m}: CV RMSE = {rmse(y, cached[f'{m}_cv']):.2f}", flush=True)

        # 個別fold RMSE表示
        for m in available:
            cv = cached[f"{m}_cv"]
            fold_rmses = [f"{sp[fi]}:{rmse(y[te], cv[te]):.1f}" for fi, (_, te) in enumerate(folds)]
            print(f"    {m} folds: {', '.join(fold_rmses)}", flush=True)

        all_cv = [cached[f"{m}_cv"] for m in available]
        all_test = [cached[f"{m}_test"] for m in available]

        # 全モデル均等
        avg_cv = np.mean(all_cv, axis=0)
        avg_test = np.clip(np.mean(all_test, axis=0), 0, 300)
        print(f"\n全モデル均等 CV RMSE: {rmse(y, avg_cv):.2f}", flush=True)
        for fi, (_, te) in enumerate(folds):
            print(f"  {sp[fi]}: {rmse(y[te], avg_cv[te]):.2f}", flush=True)

        # PLS4のみ
        pls_models = [m for m in available if m.startswith("M")]
        if pls_models:
            pls_cv = np.mean([cached[f"{m}_cv"] for m in pls_models], axis=0)
            pls_test = np.clip(np.mean([cached[f"{m}_test"] for m in pls_models], axis=0), 0, 300)
            print(f"\nPLS4モデル CV RMSE: {rmse(y, pls_cv):.2f}", flush=True)

        # di-PLSのみ
        dipls_models = [m for m in available if m.startswith("diPLS")]
        if dipls_models:
            dipls_cv = np.mean([cached[f"{m}_cv"] for m in dipls_models], axis=0)
            dipls_test = np.clip(np.mean([cached[f"{m}_test"] for m in dipls_models], axis=0), 0, 300)
            print(f"\ndi-PLSモデル CV RMSE: {rmse(y, dipls_cv):.2f}", flush=True)

        # 提出ファイル
        submissions = [("all_equal", avg_test)]
        if pls_models:
            submissions.append(("pls4_equal", pls_test))
        if dipls_models:
            submissions.append(("dipls_equal", dipls_test))

        for tag, pred in submissions:
            sub = pd.DataFrame({0: test_ids.astype(int), 1: pred})
            path = OUT_DIR / f"submission_v8_{tag}.csv"
            sub.to_csv(path, index=False, header=False)
            print(f"\nSaved: {path}", flush=True)
            print(f"  range: [{pred.min():.1f}, {pred.max():.1f}], mean: {pred.mean():.1f}", flush=True)

    print(f"\nTotal time: {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
