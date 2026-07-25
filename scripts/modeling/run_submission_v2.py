"""改良版提出パイプライン: 新5モデルSimpleAvg (RMSE=17.03)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut, KFold

from src.eda.data_loader import load_train, load_test, get_spectral_columns, get_wavenumbers
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue20_savgol import apply_savgol
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def cars_select(X_train, y_train, X_test, n_pls=3, n_iterations=30):
    """CARS波長選択"""
    n_samples, n_features = X_train.shape
    remaining = np.arange(n_features)
    ratio = (max(n_pls + 2, 5) / n_features) ** (1.0 / n_iterations)
    best_rmse = np.inf
    best_idx = remaining.copy()

    for i in range(n_iterations):
        n_keep = max(n_pls + 1, int(n_features * ratio ** (i + 1)))
        if n_keep >= len(remaining):
            continue
        n_comp = min(n_pls, len(remaining) - 1, n_samples - 1)
        if n_comp < 1:
            break
        pls = PLSRegression(n_components=n_comp)
        pls.fit(X_train[:, remaining], y_train)
        coefs = np.abs(pls.coef_.ravel())
        top_idx = np.argsort(coefs)[::-1][:n_keep]
        remaining = remaining[np.sort(top_idx)]

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_rmses = []
        for tr, te in kf.split(X_train[:, remaining]):
            n_c = min(n_comp, len(remaining) - 1)
            if n_c < 1:
                break
            p = PLSRegression(n_components=n_c)
            p.fit(X_train[tr][:, remaining], y_train[tr])
            pred = p.predict(X_train[te][:, remaining]).ravel()
            cv_rmses.append(np.sqrt(np.mean((pred - y_train[te]) ** 2)))
        if cv_rmses:
            rmse = np.mean(cv_rmses)
            if rmse < best_rmse:
                best_rmse = rmse
                best_idx = remaining.copy()

    return X_train[:, best_idx], X_test[:, best_idx], best_idx


def predict_ensemble_v2(X_train, y_train, groups_train, X_test):
    """新5モデルSimpleAvgで予測"""
    np.random.seed(42)
    predictions = []
    y_sqrt = np.sqrt(y_train)

    # M1: SNV+PLS(2)
    X_tr_snv = apply_snv(X_train)
    X_te_snv = apply_snv(X_test)
    pls = PLSRegression(n_components=2)
    pls.fit(X_tr_snv, y_train)
    predictions.append(pls.predict(X_te_snv).ravel())
    print("  M1: SNV+PLS(2) done")

    # M2: SNV+PLS(2)+sqrt(y)
    pls = PLSRegression(n_components=2)
    pls.fit(X_tr_snv, y_sqrt)
    pred_sqrt = pls.predict(X_te_snv).ravel()
    predictions.append(np.clip(pred_sqrt, 0, None) ** 2)
    print("  M2: SNV+PLS(2)+sqrt(y) done")

    # M3: EPO(1)+CARS+PLS(3)
    P = compute_epo_projection(X_train, groups_train, n_components=1)
    X_tr_epo = apply_epo(X_train, P)
    X_te_epo = apply_epo(X_test, P)
    X_tr_sel, X_te_sel, _ = cars_select(X_tr_epo, y_train, X_te_epo, n_pls=3, n_iterations=30)
    n_c = min(3, X_tr_sel.shape[1] - 1)
    pls = PLSRegression(n_components=max(1, n_c))
    pls.fit(X_tr_sel, y_train)
    predictions.append(pls.predict(X_te_sel).ravel())
    print("  M3: EPO(1)+CARS+PLS(3) done")

    # M4: SG2d(w=7)+EPO(1)+PLS(3)
    X_tr_sg = apply_savgol(X_train, deriv=2, window_length=7)
    X_te_sg = apply_savgol(X_test, deriv=2, window_length=7)
    P_sg = compute_epo_projection(X_tr_sg, groups_train, n_components=1)
    X_tr_sg_epo = apply_epo(X_tr_sg, P_sg)
    X_te_sg_epo = apply_epo(X_te_sg, P_sg)
    pls = PLSRegression(n_components=3)
    pls.fit(X_tr_sg_epo, y_train)
    predictions.append(pls.predict(X_te_sg_epo).ravel())
    print("  M4: SG2d(w=7)+EPO(1)+PLS(3) done")

    # M5: EPO(1)+PLS(4)+sqrt(y)
    pls = PLSRegression(n_components=4)
    pls.fit(X_tr_epo, y_sqrt)
    pred_sqrt = pls.predict(X_te_epo).ravel()
    predictions.append(np.clip(pred_sqrt, 0, None) ** 2)
    print("  M5: EPO(1)+PLS(4)+sqrt(y) done")

    # SimpleAvg + clip
    avg_pred = np.mean(predictions, axis=0)
    avg_pred = np.clip(avg_pred, 0, 200)
    return avg_pred


def main():
    df_train = load_train(DATA_DIR)
    df_test = load_test(DATA_DIR)
    spectral_cols = get_spectral_columns(df_train)

    X_train = df_train[spectral_cols].values
    y_train = df_train["含水率"].values
    groups_train = df_train["樹種"].values
    X_test = df_test[spectral_cols].values

    print("=== 提出ファイル生成 (v2: 新5モデルSimpleAvg) ===\n")

    preds = predict_ensemble_v2(X_train, y_train, groups_train, X_test)

    print(f"\n予測統計:")
    print(f"  サンプル数: {len(preds)}")
    print(f"  平均: {preds.mean():.2f}")
    print(f"  範囲: [{preds.min():.2f}, {preds.max():.2f}]")
    print(f"  中央値: {np.median(preds):.2f}")

    # 提出ファイル生成
    submission = pd.DataFrame({
        "sample_number": df_test["sample number"].values,
        "prediction": preds,
    })
    output_path = OUT_DIR / "submission_v2.csv"
    submission.to_csv(output_path, index=False, header=False)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
