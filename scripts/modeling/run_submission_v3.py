"""提出パイプライン v3: 5モデルSimpleAvg (fold-RMSE=15.99)

BestCombo5-Avg:
  1. EPO(1)+PLS(4)+sqrt
  2. SG2d+EPO(1)+PLS(3)+raw
  3. PiecewiseMSC+Lasso+raw
  4. SNV+VIP(1.5)+PLS(4)+sqrt
  5. SNV+AsLS(1e6)+PLS(2)+sqrt

LOSO-CV検証 + テスト予測 + 提出ファイル生成。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import LeaveOneGroupOut, KFold

from src.eda.data_loader import load_train, load_test, get_spectral_columns, get_wavenumbers
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue20_savgol import apply_savgol
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue19_msc import compute_msc_reference
from src.preprocessing.issue62_additional_preprocessing import apply_asls, apply_piecewise_msc

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def vip_select(X_train, y_train, X_test, n_components=2, threshold=1.5):
    nc = min(n_components, X_train.shape[1] - 1)
    pls = PLSRegression(n_components=nc)
    pls.fit(X_train, y_train)
    T, W, Q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
    p = X_train.shape[1]
    s = np.diag(T.T @ T @ Q.T @ Q).ravel()
    total_s = np.sum(s)
    vip = np.sqrt(p * np.sum(s[None, :] * (W / np.linalg.norm(W, axis=0)) ** 2, axis=1) / total_s)
    mask = vip > threshold
    if mask.sum() < 2:
        mask = np.zeros(p, dtype=bool)
        mask[np.argsort(vip)[-max(2, int(p * 0.1)):]] = True
    return X_train[:, mask], X_test[:, mask]


def predict_5models(X_train, y_train, groups_train, X_test):
    """5モデルSimpleAvgで予測"""
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

    # M3: PiecewiseMSC(seg=3)+Lasso(0.1)+raw
    ref = compute_msc_reference(X_train)
    X_tr_pmsc = apply_piecewise_msc(X_train, ref, n_segments=3)
    X_te_pmsc = apply_piecewise_msc(X_test, ref, n_segments=3)
    lasso = Lasso(alpha=0.1, max_iter=10000)
    lasso.fit(X_tr_pmsc, y_train)
    predictions.append(lasso.predict(X_te_pmsc))
    print("  M3: PiecewiseMSC+Lasso(0.1)+raw done")

    # M4: SNV+VIP(1.5)+PLS(4)+sqrt
    X_tr_snv = apply_snv(X_train)
    X_te_snv = apply_snv(X_test)
    X_tr_vip, X_te_vip = vip_select(X_tr_snv, y_train, X_te_snv, n_components=2, threshold=1.5)
    nc = min(4, X_tr_vip.shape[1] - 1)
    pls = PLSRegression(n_components=max(1, nc))
    pls.fit(X_tr_vip, y_sqrt)
    pred = np.clip(pls.predict(X_te_vip).ravel(), 0, None) ** 2
    predictions.append(pred)
    print("  M4: SNV+VIP(1.5)+PLS(4)+sqrt done")

    # M5: SNV+AsLS(1e6)+PLS(2)+sqrt
    X_tr_asls = apply_asls(X_tr_snv, lam=1e6)
    X_te_asls = apply_asls(X_te_snv, lam=1e6)
    pls = PLSRegression(n_components=2)
    pls.fit(X_tr_asls, y_sqrt)
    pred = np.clip(pls.predict(X_te_asls).ravel(), 0, None) ** 2
    predictions.append(pred)
    print("  M5: SNV+AsLS(1e6)+PLS(2)+sqrt done")

    # SimpleAvg + clip
    avg_pred = np.mean(predictions, axis=0)
    avg_pred = np.clip(avg_pred, 0, 200)
    return avg_pred, predictions


def validate_loso(X_raw, y, groups):
    """LOSO-CVで検証"""
    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_raw, y, groups))
    species_names = [np.unique(groups[te])[0] for _, te in folds]

    all_preds = np.zeros(len(y))
    fold_rmses = []

    print("=== LOSO-CV 検証 ===\n")
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        avg_pred, _ = predict_5models(
            X_raw[train_idx], y[train_idx], groups[train_idx], X_raw[test_idx]
        )
        all_preds[test_idx] = avg_pred
        fr = rmse(y[test_idx], avg_pred)
        fold_rmses.append(fr)
        print(f"  Fold {fold_idx+1} ({species_names[fold_idx]}): RMSE={fr:.2f}\n")

    mean_rmse = np.mean(fold_rmses)
    std_rmse = np.std(fold_rmses)
    overall_rmse = rmse(y, all_preds)

    print("=" * 60)
    print(f"fold-RMSE平均: {mean_rmse:.4f} ± {std_rmse:.2f}")
    print(f"全体RMSE: {overall_rmse:.4f}")
    print(f"ベースライン(v2): 17.03")
    print(f"改善幅: {17.03 - mean_rmse:.4f}")
    print("=" * 60)

    print("\nFold詳細:")
    for sp, fr in zip(species_names, fold_rmses):
        print(f"  {sp}: {fr:.2f}")

    return mean_rmse, fold_rmses


def generate_submission(X_train, y_train, groups_train, X_test, df_test):
    """テスト予測・提出ファイル生成"""
    print("\n=== テスト予測・提出ファイル生成 ===\n")
    avg_pred, individual_preds = predict_5models(X_train, y_train, groups_train, X_test)

    print(f"\n予測統計:")
    print(f"  サンプル数: {len(avg_pred)}")
    print(f"  平均: {avg_pred.mean():.2f}")
    print(f"  範囲: [{avg_pred.min():.2f}, {avg_pred.max():.2f}]")
    print(f"  中央値: {np.median(avg_pred):.2f}")

    print(f"\n個別モデル予測の相関:")
    model_names = ["EPO+PLS4+sqrt", "SG2d+EPO+PLS3", "PMSC+Lasso", "SNV+VIP+PLS4+sqrt", "SNV+AsLS+PLS2+sqrt"]
    corr = np.corrcoef(individual_preds)
    for i in range(5):
        for j in range(i+1, 5):
            print(f"  {model_names[i]} vs {model_names[j]}: {corr[i,j]:.3f}")

    submission = pd.DataFrame({
        "sample_number": df_test["sample number"].values,
        "prediction": avg_pred,
    })
    output_path = OUT_DIR / "submission_v3.csv"
    submission.to_csv(output_path, index=False, header=False)
    print(f"\nSaved: {output_path}")
    return avg_pred


def main():
    df_train = load_train(DATA_DIR)
    df_test = load_test(DATA_DIR)
    spectral_cols = get_spectral_columns(df_train)

    X_train = df_train[spectral_cols].values
    y_train = df_train["含水率"].values
    groups_train = df_train["樹種"].values
    X_test = df_test[spectral_cols].values

    print("=" * 60)
    print("提出パイプライン v3: 5モデルSimpleAvg")
    print("  M1: EPO(1)+PLS(4)+sqrt")
    print("  M2: SG2d+EPO(1)+PLS(3)+raw")
    print("  M3: PiecewiseMSC(seg=3)+Lasso(0.1)+raw")
    print("  M4: SNV+VIP(1.5)+PLS(4)+sqrt")
    print("  M5: SNV+AsLS(1e6)+PLS(2)+sqrt")
    print("=" * 60)

    # LOSO-CV検証
    mean_rmse, fold_rmses = validate_loso(X_train, y_train, groups_train)

    # テスト予測・提出
    preds = generate_submission(X_train, y_train, groups_train, X_test, df_test)


if __name__ == "__main__":
    main()
