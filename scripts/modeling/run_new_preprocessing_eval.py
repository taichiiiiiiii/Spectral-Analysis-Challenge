"""新規前処理・特徴量手法のLOSO-CV評価

TCA, Spectral Autocorrelation, Wavelet, OPLS の4手法を評価する。
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut

from src.eda.data_loader import load_train, load_test, get_spectral_columns, get_wavenumbers
from src.preprocessing.issue38_tca import tca_transform
from src.preprocessing.issue39_wavelet_transform import wavelet_denoise, extract_wavelet_features
from src.preprocessing.issue40_opls import OPLSFilter
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.feature_engineering.issue41_spectral_autocorrelation import compute_spectral_autocorrelation


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def loso_cv(X, y, groups, model_fn):
    """LOSO-CV で RMSE を計算する。"""
    logo = LeaveOneGroupOut()
    fold_rmses = []
    for train_idx, test_idx in logo.split(X, y, groups):
        model = model_fn()
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx]).ravel()
        fold_rmses.append(rmse(y[test_idx], pred))
    return np.mean(fold_rmses), np.std(fold_rmses)


def main():
    data_dir = Path("Input_data")
    train_df = load_train(data_dir)
    test_df = load_test(data_dir)
    spectral_cols = get_spectral_columns(train_df)

    X_train = train_df[spectral_cols].values
    X_test = test_df[spectral_cols].values
    y = train_df["含水率"].values
    groups = train_df["樹種"].values
    wavenumbers = get_wavenumbers(spectral_cols)

    results = []

    # Baseline: Raw + PLS
    print("=== Baseline: Raw + PLS ===")
    for nc in [2, 4, 6]:
        mean_rmse, std_rmse = loso_cv(X_train, y, groups, lambda: PLSRegression(n_components=nc))
        print(f"  PLS(nc={nc}): RMSE={mean_rmse:.2f} ± {std_rmse:.2f}")
        results.append({"method": f"Raw+PLS({nc})", "rmse": mean_rmse, "rmse_std": std_rmse})

    # 1. TCA + PLS
    print("\n=== TCA + PLS ===")
    logo = LeaveOneGroupOut()
    for n_tca in [5, 10, 20]:
        fold_rmses = []
        for train_idx, test_idx in logo.split(X_train, y, groups):
            X_tr, X_te = X_train[train_idx], X_train[test_idx]
            Z_tr, Z_te = tca_transform(X_tr, X_te, n_components=n_tca, kernel="rbf")
            nc_pls = min(n_tca, 4)
            pls = PLSRegression(n_components=nc_pls)
            pls.fit(Z_tr, y[train_idx])
            pred = pls.predict(Z_te).ravel()
            fold_rmses.append(rmse(y[test_idx], pred))
        mean_r = np.mean(fold_rmses)
        std_r = np.std(fold_rmses)
        print(f"  TCA(n={n_tca})+PLS({nc_pls}): RMSE={mean_r:.2f} ± {std_r:.2f}")
        results.append({"method": f"TCA({n_tca})+PLS({nc_pls})", "rmse": mean_r, "rmse_std": std_r})

    # 2. Wavelet Denoise + PLS
    print("\n=== Wavelet Denoise + PLS ===")
    for wv, lv in [("db4", 3), ("db4", 5), ("sym6", 3)]:
        X_wv = wavelet_denoise(X_train, wavelet=wv, level=lv)
        for nc in [2, 4]:
            mean_rmse, std_rmse = loso_cv(X_wv, y, groups, lambda: PLSRegression(n_components=nc))
            print(f"  Wavelet({wv},lv={lv})+PLS({nc}): RMSE={mean_rmse:.2f} ± {std_rmse:.2f}")
            results.append({"method": f"Wavelet({wv},lv={lv})+PLS({nc})", "rmse": mean_rmse, "rmse_std": std_rmse})

    # 3. Wavelet Denoise + EPO + PLS
    print("\n=== Wavelet Denoise + EPO + PLS ===")
    X_wv_db4 = wavelet_denoise(X_train, wavelet="db4", level=3)
    P_epo = compute_epo_projection(X_wv_db4, groups, n_components=1)
    X_wv_epo = apply_epo(X_wv_db4, P_epo)
    for nc in [2, 4]:
        mean_rmse, std_rmse = loso_cv(X_wv_epo, y, groups, lambda: PLSRegression(n_components=nc))
        print(f"  Wavelet+EPO(1)+PLS({nc}): RMSE={mean_rmse:.2f} ± {std_rmse:.2f}")
        results.append({"method": f"Wavelet+EPO(1)+PLS({nc})", "rmse": mean_rmse, "rmse_std": std_rmse})

    # 4. OPLS + PLS
    print("\n=== OPLS + PLS ===")
    for n_opls in [1, 2, 3, 5]:
        logo2 = LeaveOneGroupOut()
        fold_rmses = []
        for train_idx, test_idx in logo2.split(X_train, y, groups):
            opls = OPLSFilter(n_components=n_opls)
            X_tr_f = opls.fit_transform(X_train[train_idx], y[train_idx])
            X_te_f = opls.transform(X_train[test_idx])
            pls = PLSRegression(n_components=4)
            pls.fit(X_tr_f, y[train_idx])
            pred = pls.predict(X_te_f).ravel()
            fold_rmses.append(rmse(y[test_idx], pred))
        mean_r = np.mean(fold_rmses)
        std_r = np.std(fold_rmses)
        print(f"  OPLS({n_opls})+PLS(4): RMSE={mean_r:.2f} ± {std_r:.2f}")
        results.append({"method": f"OPLS({n_opls})+PLS(4)", "rmse": mean_r, "rmse_std": std_r})

    # 5. Spectral Autocorrelation + PLS
    print("\n=== Spectral Autocorrelation features + PLS ===")
    acf_feats = compute_spectral_autocorrelation(X_train, lags=[1, 2, 5, 10, 20, 50, 100])
    # ACF alone
    mean_rmse, std_rmse = loso_cv(acf_feats.values, y, groups, lambda: PLSRegression(n_components=2))
    print(f"  ACF only + PLS(2): RMSE={mean_rmse:.2f} ± {std_rmse:.2f}")
    results.append({"method": "ACF+PLS(2)", "rmse": mean_rmse, "rmse_std": std_rmse})

    # ACF + EPO features
    P_epo_raw = compute_epo_projection(X_train, groups, n_components=1)
    X_epo = apply_epo(X_train, P_epo_raw)
    X_epo_acf = np.hstack([X_epo, acf_feats.values])
    for nc in [4, 6]:
        mean_rmse, std_rmse = loso_cv(X_epo_acf, y, groups, lambda: PLSRegression(n_components=nc))
        print(f"  EPO+ACF+PLS({nc}): RMSE={mean_rmse:.2f} ± {std_rmse:.2f}")
        results.append({"method": f"EPO+ACF+PLS({nc})", "rmse": mean_rmse, "rmse_std": std_rmse})

    # 6. Wavelet Features + PLS
    print("\n=== Wavelet Features + PLS ===")
    wv_feats = extract_wavelet_features(X_train, wavelet="db4", level=5)
    X_epo_wvf = np.hstack([X_epo, wv_feats.values])
    for nc in [4, 6]:
        mean_rmse, std_rmse = loso_cv(X_epo_wvf, y, groups, lambda: PLSRegression(n_components=nc))
        print(f"  EPO+WaveletFeats+PLS({nc}): RMSE={mean_rmse:.2f} ± {std_rmse:.2f}")
        results.append({"method": f"EPO+WaveletFeats+PLS({nc})", "rmse": mean_rmse, "rmse_std": std_rmse})

    # 7. OPLS + EPO + PLS (combination)
    print("\n=== OPLS + EPO + PLS ===")
    logo3 = LeaveOneGroupOut()
    fold_rmses = []
    for train_idx, test_idx in logo3.split(X_train, y, groups):
        opls = OPLSFilter(n_components=2)
        X_tr_f = opls.fit_transform(X_train[train_idx], y[train_idx])
        X_te_f = opls.transform(X_train[test_idx])
        P = compute_epo_projection(X_tr_f, groups[train_idx], n_components=1)
        X_tr_fe = apply_epo(X_tr_f, P)
        X_te_fe = apply_epo(X_te_f, P)
        pls = PLSRegression(n_components=4)
        pls.fit(X_tr_fe, y[train_idx])
        pred = pls.predict(X_te_fe).ravel()
        fold_rmses.append(rmse(y[test_idx], pred))
    mean_r = np.mean(fold_rmses)
    std_r = np.std(fold_rmses)
    print(f"  OPLS(2)+EPO(1)+PLS(4): RMSE={mean_r:.2f} ± {std_r:.2f}")
    results.append({"method": "OPLS(2)+EPO(1)+PLS(4)", "rmse": mean_r, "rmse_std": std_r})

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (sorted by RMSE)")
    print("=" * 60)
    results_df = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    print(results_df.to_string(index=False))

    results_df.to_csv("outputs/new_preprocessing_eval.csv", index=False)
    print("\nSaved to outputs/new_preprocessing_eval.csv")


if __name__ == "__main__":
    main()
