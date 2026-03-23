"""提出パイプライン v5: 4モデルSimpleAvg (fold-RMSE=15.22)

BestCombo4-Avg:
  1. EPO(1)+PLS(4)+sqrt
  2. SG2d+EPO(1)+PLS(3)+raw
  3. SNV+AsLS(1e6)+siPLS(30,3)+PLS(4)+sqrt
  4. SNV+iPLS(50)+PLS(4)+sqrt
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut

from src.eda.data_loader import load_train, load_test, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue20_savgol import apply_savgol
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue62_additional_preprocessing import apply_asls
from src.modeling.issue65_feature_selection import sipls_select, ipls_select

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def predict_4models(X_train, y_train, groups_train, X_test):
    np.random.seed(42)
    predictions = []
    y_sqrt = np.sqrt(y_train)

    # M1: EPO(1)+PLS(4)+sqrt
    P = compute_epo_projection(X_train, groups_train, n_components=1)
    X_tr_epo = apply_epo(X_train, P)
    X_te_epo = apply_epo(X_test, P)
    pls = PLSRegression(n_components=4)
    pls.fit(X_tr_epo, y_sqrt)
    pred = np.clip(pls.predict(X_te_epo).ravel(), 0, None) ** 2
    predictions.append(pred)
    print("  M1: EPO(1)+PLS(4)+sqrt done")

    # M2: SG2d+EPO(1)+PLS(3)+raw
    X_tr_sg = apply_savgol(X_train, deriv=2, window_length=7)
    X_te_sg = apply_savgol(X_test, deriv=2, window_length=7)
    P_sg = compute_epo_projection(X_tr_sg, groups_train, n_components=1)
    X_tr_sg_epo = apply_epo(X_tr_sg, P_sg)
    X_te_sg_epo = apply_epo(X_te_sg, P_sg)
    pls = PLSRegression(n_components=3)
    pls.fit(X_tr_sg_epo, y_train)
    predictions.append(pls.predict(X_te_sg_epo).ravel())
    print("  M2: SG2d+EPO(1)+PLS(3)+raw done")

    # M3: SNV+AsLS(1e6)+siPLS(30,3)+PLS(4)+sqrt
    X_tr_snv = apply_snv(X_train)
    X_te_snv = apply_snv(X_test)
    X_tr_asls = apply_asls(X_tr_snv, lam=1e6)
    X_te_asls = apply_asls(X_te_snv, lam=1e6)
    X_tr_sel, X_te_sel, _ = sipls_select(X_tr_asls, y_train, X_te_asls,
                                          n_intervals=30, n_components=3, n_combine=3)
    nc = min(4, X_tr_sel.shape[1] - 1)
    pls = PLSRegression(n_components=max(1, nc))
    pls.fit(X_tr_sel, y_sqrt)
    pred = np.clip(pls.predict(X_te_sel).ravel(), 0, None) ** 2
    predictions.append(pred)
    print("  M3: SNV+AsLS+siPLS(30,3)+PLS(4)+sqrt done")

    # M4: SNV+iPLS(50)+PLS(4)+sqrt
    X_tr_sel2, X_te_sel2, _ = ipls_select(X_tr_snv, y_train, X_te_snv,
                                            n_intervals=50, n_components=3, n_best=1)
    nc = min(4, X_tr_sel2.shape[1] - 1)
    pls = PLSRegression(n_components=max(1, nc))
    pls.fit(X_tr_sel2, y_sqrt)
    pred = np.clip(pls.predict(X_te_sel2).ravel(), 0, None) ** 2
    predictions.append(pred)
    print("  M4: SNV+iPLS(50)+PLS(4)+sqrt done")

    avg_pred = np.mean(predictions, axis=0)
    avg_pred = np.clip(avg_pred, 0, 300)
    return avg_pred, predictions


def main():
    df_train = load_train(DATA_DIR)
    df_test = load_test(DATA_DIR)
    spectral_cols = get_spectral_columns(df_train)

    X_train = df_train[spectral_cols].values
    y_train = df_train["含水率"].values
    groups_train = df_train["樹種"].values
    X_test = df_test[spectral_cols].values

    print("=" * 60)
    print("提出パイプライン v5: 4モデルSimpleAvg")
    print("  M1: EPO(1)+PLS(4)+sqrt")
    print("  M2: SG2d+EPO(1)+PLS(3)+raw")
    print("  M3: SNV+AsLS(1e6)+siPLS(30,3)+PLS(4)+sqrt")
    print("  M4: SNV+iPLS(50)+PLS(4)+sqrt")
    print("=" * 60)

    # LOSO-CV検証
    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_train, y_train, groups_train))
    species = [np.unique(groups_train[te])[0] for _, te in folds]

    print("\n=== LOSO-CV 検証 ===\n")
    fold_rmses = []
    for f_idx, (tr, te) in enumerate(folds):
        avg, _ = predict_4models(X_train[tr], y_train[tr], groups_train[tr], X_train[te])
        fr = rmse(y_train[te], avg)
        fold_rmses.append(fr)
        print(f"  Fold {f_idx+1} ({species[f_idx]}): RMSE={fr:.2f}\n")

    print("=" * 60)
    print(f"fold-RMSE平均: {np.mean(fold_rmses):.4f} ± {np.std(fold_rmses):.2f}")
    print(f"v3: 15.99 | v2: 17.03")
    print(f"改善幅(vs v3): {15.99 - np.mean(fold_rmses):.4f}")
    print("=" * 60)
    for sp, fr in zip(species, fold_rmses):
        print(f"  {sp}: {fr:.2f}")

    # テスト予測
    print("\n=== テスト予測 ===\n")
    avg, indiv = predict_4models(X_train, y_train, groups_train, X_test)
    print(f"\n予測統計: mean={avg.mean():.2f}, range=[{avg.min():.2f}, {avg.max():.2f}]")

    submission = pd.DataFrame({
        "sample_number": df_test["sample number"].values,
        "prediction": avg,
    })
    output_path = OUT_DIR / "submission_v5.csv"
    submission.to_csv(output_path, index=False, header=False)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
