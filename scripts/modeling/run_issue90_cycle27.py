"""Issue #90: サイクル27 - 水分吸収帯特徴量エンジニアリング

仮説: 水分吸収帯(5200/6900 cm⁻¹)に特化した特徴量は樹種に依存しにくく、
ドメインシフトに対してロバスト。PLSスコア+水分帯特徴量のハイブリッドで改善。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np, pandas as pd, warnings, time
warnings.filterwarnings("ignore")
from itertools import combinations
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from src.eda.data_loader import load_train, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue20_savgol import apply_savgol
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue19_msc import compute_msc_reference
from src.preprocessing.issue62_additional_preprocessing import apply_asls, apply_piecewise_msc
from src.modeling.issue65_feature_selection import sipls_select, ipls_select

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def _wn_idx(wavenumbers, target):
    return int(np.argmin(np.abs(wavenumbers - target)))

def _wn_range(wavenumbers, low, high):
    i_low, i_high = _wn_idx(wavenumbers, low), _wn_idx(wavenumbers, high)
    if i_low > i_high:
        i_low, i_high = i_high, i_low
    return slice(i_low, i_high + 1)

# === 水分帯特徴量 ===
def compute_water_band_features(X, wavenumbers):
    """水分吸収帯ベースの特徴量を計算"""
    eps = 1e-10
    n_samples = X.shape[0]
    features = {}

    # 主要波数
    idx_5200 = _wn_idx(wavenumbers, 5200)
    idx_6900 = _wn_idx(wavenumbers, 6900)
    idx_8300 = _wn_idx(wavenumbers, 8300)  # CH伸縮帯（参照）
    idx_6000 = _wn_idx(wavenumbers, 6000)  # 構造帯
    idx_4600 = _wn_idx(wavenumbers, 4600)  # 参照帯
    idx_7500 = _wn_idx(wavenumbers, 7500)  # 参照帯

    # 1. ピーク強度
    features["peak_5200"] = X[:, idx_5200]
    features["peak_6900"] = X[:, idx_6900]

    # 2. 帯域平均（±50 cm⁻¹）
    rng_5200 = _wn_range(wavenumbers, 5150, 5250)
    rng_6900 = _wn_range(wavenumbers, 6850, 6950)
    features["mean_5200"] = np.mean(X[:, rng_5200], axis=1)
    features["mean_6900"] = np.mean(X[:, rng_6900], axis=1)

    # 3. バンド面積（台形積分）
    features["area_5200"] = np.trapezoid(X[:, rng_5200], axis=1)
    features["area_6900"] = np.trapezoid(X[:, rng_6900], axis=1)

    # 4. バンド比（樹種不変）
    features["ratio_5200_8300"] = X[:, idx_5200] / (X[:, idx_8300] + eps)
    features["ratio_6900_8300"] = X[:, idx_6900] / (X[:, idx_8300] + eps)
    features["ratio_5200_6000"] = X[:, idx_5200] / (X[:, idx_6000] + eps)
    features["ratio_6900_6000"] = X[:, idx_6900] / (X[:, idx_6000] + eps)
    features["ratio_5200_6900"] = X[:, idx_5200] / (X[:, idx_6900] + eps)

    # 5. NDMI
    b5200 = X[:, idx_5200]
    b6900 = X[:, idx_6900]
    features["ndmi"] = (b6900 - b5200) / (b6900 + b5200 + eps)

    # 6. 1次微分ピーク
    X_d1 = savgol_filter(X, window_length=11, polyorder=2, deriv=1, axis=1)
    features["d1_5200"] = X_d1[:, idx_5200]
    features["d1_6900"] = X_d1[:, idx_6900]

    # 7. 2次微分ピーク
    X_d2 = savgol_filter(X, window_length=11, polyorder=2, deriv=2, axis=1)
    features["d2_5200"] = X_d2[:, idx_5200]
    features["d2_6900"] = X_d2[:, idx_6900]

    # 8. 面積比
    ref_range = _wn_range(wavenumbers, 7400, 7600)
    ref_area = np.trapezoid(X[:, ref_range], axis=1) + eps
    features["area_ratio_5200"] = features["area_5200"] / ref_area
    features["area_ratio_6900"] = features["area_6900"] / ref_area

    return pd.DataFrame(features)

# === 前処理（issue89と同じ） ===
def preproc(X_tr, X_te, g, name):
    if name == "SNV": return apply_snv(X_tr), apply_snv(X_te)
    elif name == "EPO(1)":
        P = compute_epo_projection(X_tr, g, n_components=1)
        return apply_epo(X_tr, P), apply_epo(X_te, P)
    elif name == "PMSC":
        ref = compute_msc_reference(X_tr)
        return apply_piecewise_msc(X_tr, ref, 3), apply_piecewise_msc(X_te, ref, 3)
    elif name == "SNV+AsLS(1e6)":
        return apply_asls(apply_snv(X_tr), lam=1e6), apply_asls(apply_snv(X_te), lam=1e6)
    return X_tr.copy(), X_te.copy()

def feat_sel(X_tr, X_te, y, name):
    if not name: return X_tr, X_te
    if name.startswith("siPLS"):
        p = name.replace("siPLS(","").rstrip(")").split(",")
        X_tr, X_te, _ = sipls_select(X_tr, y, X_te, n_intervals=int(p[0]), n_components=3, n_combine=int(p[1]))
    elif name.startswith("iPLS"):
        n = int(name.replace("iPLS(","").rstrip(")"))
        X_tr, X_te, _ = ipls_select(X_tr, y, X_te, n_intervals=n, n_components=3, n_best=1)
    return X_tr, X_te

def main():
    t0 = time.time()
    df = load_train(DATA_DIR)
    sc_cols = get_spectral_columns(df)
    X = df[sc_cols].values
    y = df["含水率"].values
    g = df["樹種"].values
    wavenumbers = np.array([float(c) for c in sc_cols])
    logo = LeaveOneGroupOut()
    folds = list(logo.split(X, y, g))
    sp = [np.unique(g[te])[0] for _, te in folds]

    print("=" * 70)
    print("Issue #90: サイクル27 - 水分吸収帯特徴量エンジニアリング")
    print("=" * 70)

    # --- 水分帯特徴量のみのモデル ---
    models = [
        ("WB_Ridge:raw", "raw", "ridge"),
        ("WB_Ridge:SNV", "SNV", "ridge"),
        ("WB_Huber:raw", "raw", "huber"),
        ("WB_Huber:SNV", "SNV", "huber"),
        ("WB_GBR:raw", "raw", "gbr"),
        ("WB_GBR:SNV", "SNV", "gbr"),
    ]

    print("\n--- 水分帯特徴量のみ ---\n", flush=True)
    wb_results = {}
    for name, pp_name, model_type in models:
        fold_rmses = []
        for fi, (tr, te) in enumerate(folds):
            X_tr_pp, X_te_pp = preproc(X[tr], X[te], g[tr], pp_name)
            wb_tr = compute_water_band_features(X_tr_pp, wavenumbers).values
            wb_te = compute_water_band_features(X_te_pp, wavenumbers).values
            sc = StandardScaler()
            wb_tr_s = sc.fit_transform(wb_tr)
            wb_te_s = sc.transform(wb_te)
            yf = np.sqrt(y[tr])
            try:
                if model_type == "ridge":
                    m = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
                    m.fit(wb_tr_s, yf)
                    pred = np.clip(m.predict(wb_te_s), 0, None) ** 2
                elif model_type == "huber":
                    m = HuberRegressor(epsilon=1.35, max_iter=200)
                    m.fit(wb_tr_s, yf)
                    pred = np.clip(m.predict(wb_te_s), 0, None) ** 2
                elif model_type == "gbr":
                    m = GradientBoostingRegressor(
                        n_estimators=100, max_depth=3, learning_rate=0.05,
                        subsample=0.8, min_samples_leaf=5, random_state=42,
                        validation_fraction=0.15, n_iter_no_change=50, tol=0.01)
                    m.fit(wb_tr_s, y[tr])  # GBR: raw target
                    pred = m.predict(wb_te_s)
                fold_rmses.append(rmse(y[te], pred))
            except Exception as e:
                fold_rmses.append(999.0)
                print(f"  ERR {name} f{fi}: {e}")
        bei_idx = [j for j, s in enumerate(sp) if s == "ベイスギ"]
        bei_r = fold_rmses[bei_idx[0]] if bei_idx else 0
        non_bei = [r for j, r in enumerate(fold_rmses) if j not in bei_idx]
        print(f"  {name}: {np.mean(fold_rmses):.2f} (bei={bei_r:.1f}, 除bei={np.mean(non_bei):.2f})")
        wb_results[name] = fold_rmses

    # --- PLSスコア + 水分帯特徴量のハイブリッド ---
    hybrid_configs = [
        {"name": "Hybrid_GBR:EPO+WB+PLS4+raw", "pp": "EPO(1)", "nc": 4, "tf": "raw",
         "ne": 200, "md": 3, "lr": 0.05},
        {"name": "Hybrid_GBR:SNV+WB+PLS4+raw", "pp": "SNV", "nc": 4, "tf": "raw",
         "ne": 200, "md": 3, "lr": 0.05},
        {"name": "Hybrid_Huber:SNV+WB+PLS4+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt"},
        {"name": "Hybrid_Huber:EPO+WB+PLS4+sqrt", "pp": "EPO(1)", "nc": 4, "tf": "sqrt"},
        {"name": "Hybrid_Ridge:SNV+WB+PLS4+sqrt", "pp": "SNV", "nc": 4, "tf": "sqrt"},
    ]

    print("\n--- PLSスコア + 水分帯特徴量ハイブリッド ---\n", flush=True)
    hybrid_results = {}
    for cfg in hybrid_configs:
        fold_rmses = []
        all_preds_list = []
        for fi, (tr, te) in enumerate(folds):
            X_tr_pp, X_te_pp = preproc(X[tr], X[te], g[tr], cfg["pp"])

            # PLSスコア
            nc = max(1, min(cfg["nc"], X_tr_pp.shape[1] - 1))
            yf = np.sqrt(y[tr]) if cfg["tf"] == "sqrt" else y[tr].copy()
            pls = PLSRegression(n_components=nc)
            pls.fit(X_tr_pp, yf)
            T_tr = pls.transform(X_tr_pp)
            T_te = pls.transform(X_te_pp)

            # 水分帯特徴量
            wb_tr = compute_water_band_features(X_tr_pp, wavenumbers).values
            wb_te = compute_water_band_features(X_te_pp, wavenumbers).values

            # 結合
            F_tr = np.hstack([T_tr, wb_tr])
            F_te = np.hstack([T_te, wb_te])

            sc_feat = StandardScaler()
            F_tr_s = sc_feat.fit_transform(F_tr)
            F_te_s = sc_feat.transform(F_te)

            try:
                if "GBR" in cfg["name"]:
                    m = GradientBoostingRegressor(
                        n_estimators=cfg.get("ne", 200), max_depth=cfg.get("md", 3),
                        learning_rate=cfg.get("lr", 0.05), subsample=0.8,
                        min_samples_leaf=5, random_state=42,
                        validation_fraction=0.15, n_iter_no_change=50, tol=0.01)
                    m.fit(F_tr_s, yf if cfg["tf"] == "raw" else yf)
                    pred = m.predict(F_te_s)
                elif "Huber" in cfg["name"]:
                    m = HuberRegressor(epsilon=1.35, max_iter=200, alpha=0.01)
                    m.fit(F_tr_s, yf)
                    pred = m.predict(F_te_s)
                elif "Ridge" in cfg["name"]:
                    m = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
                    m.fit(F_tr_s, yf)
                    pred = m.predict(F_te_s)

                if cfg["tf"] == "sqrt":
                    pred = np.clip(pred, 0, None) ** 2

                fold_rmses.append(rmse(y[te], pred))
                all_preds_list.append(pred)
            except Exception as e:
                fold_rmses.append(999.0)
                all_preds_list.append(np.full(len(te), y[tr].mean()))
                print(f"  ERR {cfg['name']} f{fi}: {e}")

        bei_idx = [j for j, s in enumerate(sp) if s == "ベイスギ"]
        bei_r = fold_rmses[bei_idx[0]] if bei_idx else 0
        non_bei = [r for j, r in enumerate(fold_rmses) if j not in bei_idx]
        print(f"  {cfg['name']}: {np.mean(fold_rmses):.2f} (bei={bei_r:.1f}, 除bei={np.mean(non_bei):.2f})")
        hybrid_results[cfg["name"]] = {"rmses": fold_rmses, "preds": all_preds_list}

    # 結果保存
    all_results = []
    for name, rmses in wb_results.items():
        all_results.append({"model": name, "mean_rmse": np.mean(rmses),
                           "bei_rmse": rmses[[j for j, s in enumerate(sp) if s == "ベイスギ"][0]]})
    for name, data in hybrid_results.items():
        rmses = data["rmses"]
        all_results.append({"model": name, "mean_rmse": np.mean(rmses),
                           "bei_rmse": rmses[[j for j, s in enumerate(sp) if s == "ベイスギ"][0]]})

    pd.DataFrame(all_results).to_csv(OUT_DIR / "issue90_cycle27_results.csv", index=False)
    print(f"\n保存完了. 時間: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
