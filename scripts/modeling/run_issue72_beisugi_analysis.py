"""Issue #72: ベイスギfold対策分析

ベイスギの含水率外挿問題を分析し、改善策を検証する。
1. ベイスギfoldの含水率分布分析
2. 予測値クリッピング範囲の最適化
3. sqrt/power変換がベイスギ外挿を緩和するか検証
4. ベイスギ除外時の12種fold-RMSEも計算
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import warnings
warnings.filterwarnings("ignore")
import time

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import LeaveOneGroupOut

from src.eda.data_loader import load_train, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue20_savgol import apply_savgol
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue19_msc import compute_msc_reference
from src.preprocessing.issue62_additional_preprocessing import apply_asls, apply_piecewise_msc

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"


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


def predict_v3_ensemble(X_tr_raw, X_te_raw, y_train, groups_train):
    """v3の5モデルSimpleAvg"""
    predictions = []
    y_sqrt = np.sqrt(y_train)

    # M1: EPO(1)+PLS(4)+sqrt
    P = compute_epo_projection(X_tr_raw, groups_train, n_components=1)
    X_tr_epo = apply_epo(X_tr_raw, P)
    X_te_epo = apply_epo(X_te_raw, P)
    pls = PLSRegression(n_components=4)
    pls.fit(X_tr_epo, y_sqrt)
    pred = np.clip(pls.predict(X_te_epo).ravel(), 0, None) ** 2
    predictions.append(pred)

    # M2: SG2d+EPO(1)+PLS(3)+raw
    X_tr_sg = apply_savgol(X_tr_raw, deriv=2, window_length=7)
    X_te_sg = apply_savgol(X_te_raw, deriv=2, window_length=7)
    P_sg = compute_epo_projection(X_tr_sg, groups_train, n_components=1)
    X_tr_sg_epo = apply_epo(X_tr_sg, P_sg)
    X_te_sg_epo = apply_epo(X_te_sg, P_sg)
    pls = PLSRegression(n_components=3)
    pls.fit(X_tr_sg_epo, y_train)
    predictions.append(pls.predict(X_te_sg_epo).ravel())

    # M3: PiecewiseMSC+Lasso(0.1)+raw
    ref = compute_msc_reference(X_tr_raw)
    X_tr_pmsc = apply_piecewise_msc(X_tr_raw, ref, 3)
    X_te_pmsc = apply_piecewise_msc(X_te_raw, ref, 3)
    lasso = Lasso(alpha=0.1, max_iter=10000)
    lasso.fit(X_tr_pmsc, y_train)
    predictions.append(lasso.predict(X_te_pmsc))

    # M4: SNV+VIP(1.5)+PLS(4)+sqrt
    X_tr_snv = apply_snv(X_tr_raw)
    X_te_snv = apply_snv(X_te_raw)
    X_tr_vip, X_te_vip = vip_select(X_tr_snv, y_train, X_te_snv, threshold=1.5)
    nc = min(4, X_tr_vip.shape[1] - 1)
    pls = PLSRegression(n_components=max(1, nc))
    pls.fit(X_tr_vip, y_sqrt)
    pred = np.clip(pls.predict(X_te_vip).ravel(), 0, None) ** 2
    predictions.append(pred)

    # M5: SNV+AsLS(1e6)+PLS(2)+sqrt
    X_tr_asls = apply_asls(X_tr_snv, lam=1e6)
    X_te_asls = apply_asls(X_te_snv, lam=1e6)
    pls = PLSRegression(n_components=2)
    pls.fit(X_tr_asls, y_sqrt)
    pred = np.clip(pls.predict(X_te_asls).ravel(), 0, None) ** 2
    predictions.append(pred)

    return np.mean(predictions, axis=0), predictions


def main():
    t0 = time.time()
    df = load_train(DATA_DIR)
    sc = get_spectral_columns(df)
    X_raw = df[sc].values
    y = df["含水率"].values
    groups = df["樹種"].values

    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_raw, y, groups))
    species = [np.unique(groups[te])[0] for _, te in folds]

    print("=" * 70)
    print("Issue #72: ベイスギfold対策分析")
    print("=" * 70)

    # 1. 含水率分布分析
    print("\n--- 1. 樹種別含水率分布 ---")
    for sp in np.unique(groups):
        mask = groups == sp
        y_sp = y[mask]
        print(f"  {sp}: n={len(y_sp)}, mean={y_sp.mean():.1f}, "
              f"min={y_sp.min():.1f}, max={y_sp.max():.1f}, "
              f"std={y_sp.std():.1f}")

    beisugi_mask = groups == "ベイスギ"
    y_beisugi = y[beisugi_mask]
    y_others = y[~beisugi_mask]
    print(f"\n  ベイスギ外挿サンプル (>{y_others.max():.1f}%): "
          f"{np.sum(y_beisugi > y_others.max())}/{len(y_beisugi)}")

    # 2. v3アンサンブルのfold予測を収集
    print("\n--- 2. v3アンサンブルのfold分析 ---")
    all_ensemble_preds = np.zeros(len(y))
    fold_rmses = []
    fold_details = []

    for f_idx, (train_idx, test_idx) in enumerate(folds):
        sp = species[f_idx]
        ens_pred, _ = predict_v3_ensemble(
            X_raw[train_idx], X_raw[test_idx],
            y[train_idx], groups[train_idx]
        )
        all_ensemble_preds[test_idx] = ens_pred
        fr = rmse(y[test_idx], ens_pred)
        fold_rmses.append(fr)

        # 詳細分析
        y_true = y[test_idx]
        residuals = ens_pred - y_true
        detail = {
            "species": sp,
            "n": len(test_idx),
            "rmse": fr,
            "y_mean": y_true.mean(),
            "y_max": y_true.max(),
            "pred_mean": ens_pred.mean(),
            "pred_max": ens_pred.max(),
            "bias": residuals.mean(),
            "n_extrapolation": np.sum(y_true > y_others.max()) if sp == "ベイスギ" else 0,
        }
        fold_details.append(detail)
        print(f"  {sp}: RMSE={fr:.2f}, bias={residuals.mean():.2f}, "
              f"y_range=[{y_true.min():.1f}, {y_true.max():.1f}], "
              f"pred_range=[{ens_pred.min():.1f}, {ens_pred.max():.1f}]")

    mean_rmse_13 = np.mean(fold_rmses)
    beisugi_idx = species.index("ベイスギ")
    mean_rmse_12 = np.mean([fr for i, fr in enumerate(fold_rmses) if i != beisugi_idx])

    print(f"\n  13種fold-RMSE平均: {mean_rmse_13:.4f}")
    print(f"  12種fold-RMSE平均 (ベイスギ除外): {mean_rmse_12:.4f}")
    print(f"  ベイスギfold RMSE: {fold_rmses[beisugi_idx]:.2f}")

    # 3. クリッピング範囲の最適化
    print("\n--- 3. 予測クリッピングの効果 ---")
    clip_values = [150, 175, 200, 220, 250, 300, 350, 400]
    for clip_max in clip_values:
        clipped = np.clip(all_ensemble_preds, 0, clip_max)
        fr_list = []
        for f_idx, (_, test_idx) in enumerate(folds):
            fr_list.append(rmse(y[test_idx], clipped[test_idx]))
        mean_r = np.mean(fr_list)
        beisugi_r = fr_list[beisugi_idx]
        print(f"  clip=[0, {clip_max:3d}]: 13種={mean_r:.4f}, ベイスギ={beisugi_r:.2f}")

    # 4. ベイスギの外挿サンプルのみの分析
    print("\n--- 4. ベイスギの外挿 vs 内挿サンプル ---")
    beisugi_test_idx = folds[beisugi_idx][1]
    y_bei = y[beisugi_test_idx]
    pred_bei = all_ensemble_preds[beisugi_test_idx]
    extrap_mask = y_bei > y_others.max()
    interp_mask = ~extrap_mask

    if interp_mask.sum() > 0:
        print(f"  内挿サンプル (≤{y_others.max():.1f}%): n={interp_mask.sum()}, "
              f"RMSE={rmse(y_bei[interp_mask], pred_bei[interp_mask]):.2f}")
    if extrap_mask.sum() > 0:
        print(f"  外挿サンプル (>{y_others.max():.1f}%): n={extrap_mask.sum()}, "
              f"RMSE={rmse(y_bei[extrap_mask], pred_bei[extrap_mask]):.2f}")
        print(f"    実測値: {y_bei[extrap_mask].min():.1f} ~ {y_bei[extrap_mask].max():.1f}")
        print(f"    予測値: {pred_bei[extrap_mask].min():.1f} ~ {pred_bei[extrap_mask].max():.1f}")

    # 5. 個別モデルのベイスギ予測分析
    print("\n--- 5. v3各モデルのベイスギfold予測 ---")
    train_idx, test_idx = folds[beisugi_idx]
    _, individual_preds = predict_v3_ensemble(
        X_raw[train_idx], X_raw[test_idx],
        y[train_idx], groups[train_idx]
    )
    model_names = ["EPO+PLS4+sqrt", "SG2d+EPO+PLS3", "PMSC+Lasso", "SNV+VIP+PLS4+sqrt", "SNV+AsLS+PLS2+sqrt"]
    for name, pred in zip(model_names, individual_preds):
        r = rmse(y[test_idx], pred)
        bias = np.mean(pred - y[test_idx])
        print(f"  {name}: RMSE={r:.2f}, bias={bias:.2f}, "
              f"pred_range=[{pred.min():.1f}, {pred.max():.1f}]")

    print(f"\n総実行時間: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
