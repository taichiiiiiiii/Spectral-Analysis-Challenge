"""Issue #78: チェリーfold深層分析

チェリーfoldのRMSE=23.91が最も悪い原因を多角的に分析する。

分析内容:
1. 含水率分布の比較
2. スペクトル特徴の比較
3. 前処理後のスペクトル距離
4. EPO適用前後の予測改善度
5. 水分吸収帯の吸収特性
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eda.data_loader import load_train, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo

from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler


def main():
    data_dir = Path(__file__).resolve().parents[2] / "Input_data"
    df = load_train(data_dir)
    spectral_cols = get_spectral_columns(df)
    wavenumbers = np.array([float(c) for c in spectral_cols])

    X_all = df[spectral_cols].values
    y_all = df["含水率"].values
    species_all = df["樹種"].values

    cherry_mask = species_all == "チェリー"
    X_cherry = X_all[cherry_mask]
    y_cherry = y_all[cherry_mask]

    other_mask = ~cherry_mask
    X_other = X_all[other_mask]
    y_other = y_all[other_mask]

    species_list = sorted(df["樹種"].unique())

    print("=" * 80)
    print("Issue #78: チェリーfold深層分析")
    print("=" * 80)

    # =========================================================================
    # 1. 含水率分布の比較
    # =========================================================================
    print("\n" + "=" * 80)
    print("1. 含水率分布の比較")
    print("=" * 80)

    print(f"\n{'樹種':<16} {'N':>4} {'平均':>8} {'標準偏差':>8} {'最小':>8} {'最大':>8} {'中央値':>8} {'IQR':>8}")
    print("-" * 80)

    for sp in species_list:
        mask = species_all == sp
        y_sp = y_all[mask]
        q1, q3 = np.percentile(y_sp, [25, 75])
        marker = " <<<" if sp == "チェリー" else ""
        print(f"{sp:<16} {len(y_sp):>4} {y_sp.mean():>8.2f} {y_sp.std():>8.2f} "
              f"{y_sp.min():>8.2f} {y_sp.max():>8.2f} {np.median(y_sp):>8.2f} {q3 - q1:>8.2f}{marker}")

    print(f"\n{'全train':<16} {len(y_all):>4} {y_all.mean():>8.2f} {y_all.std():>8.2f} "
          f"{y_all.min():>8.2f} {y_all.max():>8.2f} {np.median(y_all):>8.2f}")

    # チェリーの含水率がtrain他樹種のレンジにどの程度含まれるか
    print("\n--- チェリー含水率の外挿度分析 ---")
    y_other_min, y_other_max = y_other.min(), y_other.max()
    cherry_outside = np.sum((y_cherry < y_other_min) | (y_cherry > y_other_max))
    print(f"Train他樹種の含水率レンジ: [{y_other_min:.2f}, {y_other_max:.2f}]")
    print(f"チェリーの含水率レンジ:    [{y_cherry.min():.2f}, {y_cherry.max():.2f}]")
    print(f"外挿サンプル数: {cherry_outside}/{len(y_cherry)} "
          f"({cherry_outside/len(y_cherry)*100:.1f}%)")

    # チェリーの含水率と他樹種の含水率の重複度（パーセンタイル的に）
    for sp in species_list:
        if sp == "チェリー":
            continue
        y_sp = y_all[species_all == sp]
        overlap_min = max(y_cherry.min(), y_sp.min())
        overlap_max = min(y_cherry.max(), y_sp.max())
        if overlap_min < overlap_max:
            cherry_in_overlap = np.sum((y_cherry >= overlap_min) & (y_cherry <= overlap_max))
            sp_in_overlap = np.sum((y_sp >= overlap_min) & (y_sp <= overlap_max))
            print(f"  vs {sp:<14}: 重複レンジ [{overlap_min:.1f}, {overlap_max:.1f}], "
                  f"チェリー{cherry_in_overlap}/{len(y_cherry)}サンプル, "
                  f"{sp}{sp_in_overlap}/{len(y_sp)}サンプル")

    # =========================================================================
    # 2. スペクトル特徴の比較
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. スペクトル特徴（平均、標準偏差）の比較")
    print("=" * 80)

    # 全体平均スペクトルとの距離（ユークリッド・コサイン）
    mean_all = X_all.mean(axis=0)
    print(f"\n{'樹種':<16} {'サンプル数':>6} {'平均スペクトルとのEuclid距離':>28} {'コサイン距離':>14} {'スペクトルstd平均':>16}")
    print("-" * 90)

    species_distances = {}
    for sp in species_list:
        mask = species_all == sp
        X_sp = X_all[mask]
        mean_sp = X_sp.mean(axis=0)
        eucl = np.linalg.norm(mean_sp - mean_all)
        cos_dist = 1 - np.dot(mean_sp, mean_all) / (np.linalg.norm(mean_sp) * np.linalg.norm(mean_all))
        std_mean = X_sp.std(axis=0).mean()
        species_distances[sp] = {"eucl": eucl, "cos": cos_dist}
        marker = " <<<" if sp == "チェリー" else ""
        print(f"{sp:<16} {mask.sum():>6} {eucl:>28.4f} {cos_dist:>14.6f} {std_mean:>16.4f}{marker}")

    # スペクトル間のペアワイズ距離（樹種平均間）
    print("\n--- 樹種間平均スペクトルのユークリッド距離行列（チェリー行のみ抽出）---")
    means_by_species = {}
    for sp in species_list:
        means_by_species[sp] = X_all[species_all == sp].mean(axis=0)

    cherry_mean = means_by_species["チェリー"]
    dists_to_cherry = []
    for sp in species_list:
        if sp == "チェリー":
            continue
        d = np.linalg.norm(cherry_mean - means_by_species[sp])
        dists_to_cherry.append((sp, d))
    dists_to_cherry.sort(key=lambda x: x[1])
    print("チェリーとの距離（近い順）:")
    for sp, d in dists_to_cherry:
        print(f"  {sp:<16}: {d:.4f}")

    # =========================================================================
    # 3. 各前処理後のチェリースペクトルのtrain平均との距離
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. 各前処理後のチェリースペクトルのtrain平均との距離")
    print("=" * 80)

    # Raw
    mean_other_raw = X_other.mean(axis=0)
    dist_raw = np.mean([np.linalg.norm(x - mean_other_raw) for x in X_cherry])
    dist_other_raw = np.mean([np.linalg.norm(x - mean_other_raw) for x in X_other])

    # SNV
    X_all_snv = apply_snv(X_all)
    X_cherry_snv = X_all_snv[cherry_mask]
    X_other_snv = X_all_snv[other_mask]
    mean_other_snv = X_other_snv.mean(axis=0)
    dist_snv = np.mean([np.linalg.norm(x - mean_other_snv) for x in X_cherry_snv])
    dist_other_snv = np.mean([np.linalg.norm(x - mean_other_snv) for x in X_other_snv])

    # 1st derivative (Savitzky-Golay approximation: simple diff)
    X_all_d1 = np.diff(X_all, axis=1)
    X_cherry_d1 = X_all_d1[cherry_mask]
    X_other_d1 = X_all_d1[other_mask]
    mean_other_d1 = X_other_d1.mean(axis=0)
    dist_d1 = np.mean([np.linalg.norm(x - mean_other_d1) for x in X_cherry_d1])
    dist_other_d1 = np.mean([np.linalg.norm(x - mean_other_d1) for x in X_other_d1])

    # SNV + 1st derivative
    X_all_snv_d1 = np.diff(X_all_snv, axis=1)
    X_cherry_snv_d1 = X_all_snv_d1[cherry_mask]
    X_other_snv_d1 = X_all_snv_d1[other_mask]
    mean_other_snv_d1 = X_other_snv_d1.mean(axis=0)
    dist_snv_d1 = np.mean([np.linalg.norm(x - mean_other_snv_d1) for x in X_cherry_snv_d1])
    dist_other_snv_d1 = np.mean([np.linalg.norm(x - mean_other_snv_d1) for x in X_other_snv_d1])

    print(f"\n{'前処理':<20} {'チェリー→他平均':>18} {'他樹種→他平均':>18} {'比率(Cherry/Other)':>20}")
    print("-" * 80)
    for name, dc, do in [
        ("Raw", dist_raw, dist_other_raw),
        ("SNV", dist_snv, dist_other_snv),
        ("1st Derivative", dist_d1, dist_other_d1),
        ("SNV + 1st Deriv", dist_snv_d1, dist_other_snv_d1),
    ]:
        print(f"{name:<20} {dc:>18.4f} {do:>18.4f} {dc/do:>20.4f}")

    # 樹種別の距離比率ランキング
    print("\n--- 前処理別・樹種別 距離比率ランキング（Raw） ---")
    print(f"{'樹種':<16} {'→他平均距離':>14} {'比率':>8}")
    print("-" * 40)
    sp_dists_raw = []
    for sp in species_list:
        mask_sp = species_all == sp
        mask_rest = ~mask_sp
        mean_rest = X_all[mask_rest].mean(axis=0)
        d = np.mean([np.linalg.norm(x - mean_rest) for x in X_all[mask_sp]])
        d_rest = np.mean([np.linalg.norm(x - mean_rest) for x in X_all[mask_rest]])
        sp_dists_raw.append((sp, d, d / d_rest))
    sp_dists_raw.sort(key=lambda x: -x[2])
    for sp, d, ratio in sp_dists_raw:
        marker = " <<<" if sp == "チェリー" else ""
        print(f"{sp:<16} {d:>14.4f} {ratio:>8.4f}{marker}")

    # =========================================================================
    # 4. EPO適用前後のチェリーの予測改善度
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. EPO適用前後のチェリー予測改善度（LOSO-CVのチェリーfold）")
    print("=" * 80)

    n_pls = 4
    species_unique = np.unique(species_all)

    for n_epo in [0, 3, 5, 7]:
        fold_results = []
        for sp in species_unique:
            test_mask = species_all == sp
            train_mask = ~test_mask
            X_train = X_all[train_mask]
            X_test = X_all[test_mask]
            y_train = y_all[train_mask]
            y_test = y_all[test_mask]
            groups_train = species_all[train_mask]

            if n_epo > 0:
                P = compute_epo_projection(X_train, groups_train, n_components=n_epo)
                X_train = apply_epo(X_train, P)
                X_test = apply_epo(X_test, P)

            pls = PLSRegression(n_components=n_pls)
            pls.fit(X_train, y_train)
            pred = pls.predict(X_test).ravel()
            rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
            bias = float(np.mean(pred - y_test))
            fold_results.append((sp, rmse, bias, len(y_test)))

        method_name = "Raw PLS" if n_epo == 0 else f"EPO(n={n_epo}) + PLS"
        print(f"\n--- {method_name} (n_pls={n_pls}) ---")
        print(f"{'樹種':<16} {'RMSE':>8} {'Bias':>8} {'N':>4}")
        print("-" * 40)
        total_rmse_list = []
        for sp, rmse, bias, n in fold_results:
            marker = " <<<" if sp == "チェリー" else ""
            print(f"{sp:<16} {rmse:>8.2f} {bias:>8.2f} {n:>4}{marker}")
            total_rmse_list.append((rmse, n))
        # 全体RMSE
        total_se = sum(rmse**2 * n for rmse, n in total_rmse_list)
        total_n = sum(n for _, n in total_rmse_list)
        overall_rmse = np.sqrt(total_se / total_n)
        print(f"{'Overall':<16} {overall_rmse:>8.2f}")

    # =========================================================================
    # 5. 水分吸収帯の吸収特性
    # =========================================================================
    print("\n" + "=" * 80)
    print("5. 水分吸収帯（5200 cm⁻¹, 6900 cm⁻¹）付近の吸収特性")
    print("=" * 80)

    # 波数カラムのインデックスを見つける
    target_bands = {
        "5200 cm⁻¹": 5200,
        "6900 cm⁻¹": 6900,
    }

    for band_name, target_wn in target_bands.items():
        # ±100 cm-1の範囲のカラムを取得
        band_mask = np.abs(wavenumbers - target_wn) <= 100
        band_indices = np.where(band_mask)[0]
        band_wn = wavenumbers[band_indices]

        print(f"\n--- {band_name} 帯 (±100 cm⁻¹, {len(band_indices)}波長) ---")
        print(f"波数範囲: {band_wn.min():.1f} - {band_wn.max():.1f} cm⁻¹")

        print(f"\n{'樹種':<16} {'平均吸光度':>12} {'標準偏差':>10} {'CV(%)':>8} {'含水率相関':>10}")
        print("-" * 60)

        for sp in species_list:
            mask = species_all == sp
            X_sp_band = X_all[mask][:, band_indices]
            y_sp = y_all[mask]
            mean_abs = X_sp_band.mean()
            std_abs = X_sp_band.std()
            cv = std_abs / mean_abs * 100 if mean_abs != 0 else 0

            # 帯域平均の吸光度と含水率の相関
            band_avg = X_sp_band.mean(axis=1)
            if len(y_sp) > 2:
                corr = np.corrcoef(band_avg, y_sp)[0, 1]
            else:
                corr = np.nan
            marker = " <<<" if sp == "チェリー" else ""
            print(f"{sp:<16} {mean_abs:>12.6f} {std_abs:>10.6f} {cv:>8.2f} {corr:>10.4f}{marker}")

    # 水分吸収帯の感度分析（含水率に対する吸光度変化率）
    print("\n--- 水分吸収帯の含水率感度（回帰係数）---")
    for band_name, target_wn in target_bands.items():
        band_mask = np.abs(wavenumbers - target_wn) <= 100
        band_indices = np.where(band_mask)[0]

        print(f"\n{band_name}:")
        print(f"{'樹種':<16} {'回帰係数(吸光度/含水率%)':>24} {'R²':>8}")
        print("-" * 52)

        for sp in species_list:
            mask = species_all == sp
            band_avg = X_all[mask][:, band_indices].mean(axis=1)
            y_sp = y_all[mask]

            if len(y_sp) > 2:
                # 単回帰
                coeffs = np.polyfit(y_sp, band_avg, 1)
                pred_band = np.polyval(coeffs, y_sp)
                ss_res = np.sum((band_avg - pred_band) ** 2)
                ss_tot = np.sum((band_avg - band_avg.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                marker = " <<<" if sp == "チェリー" else ""
                print(f"{sp:<16} {coeffs[0]:>24.8f} {r2:>8.4f}{marker}")

    # =========================================================================
    # 追加分析: チェリーの予測誤差の詳細分解
    # =========================================================================
    print("\n" + "=" * 80)
    print("追加: チェリーfoldの予測誤差詳細分解（Raw PLS n=4）")
    print("=" * 80)

    test_mask = species_all == "チェリー"
    train_mask = ~test_mask
    X_train = X_all[train_mask]
    X_test = X_all[test_mask]
    y_train = y_all[train_mask]
    y_test = y_all[test_mask]

    pls = PLSRegression(n_components=n_pls)
    pls.fit(X_train, y_train)
    pred = pls.predict(X_test).ravel()

    residuals = pred - y_test
    abs_residuals = np.abs(residuals)

    print(f"\nサンプル数: {len(y_test)}")
    print(f"RMSE: {np.sqrt(np.mean(residuals**2)):.2f}")
    print(f"MAE: {np.mean(abs_residuals):.2f}")
    print(f"Bias (平均誤差): {np.mean(residuals):.2f}")
    print(f"Bias成分 (Bias²): {np.mean(residuals)**2:.2f}")
    print(f"Variance成分: {np.var(residuals):.2f}")
    print(f"RMSE² = Bias² + Variance: {np.mean(residuals)**2 + np.var(residuals):.2f}")

    # 含水率レンジ別の誤差
    print("\n--- 含水率レンジ別の予測誤差 ---")
    ranges = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 120), (120, 200)]
    for lo, hi in ranges:
        in_range = (y_test >= lo) & (y_test < hi)
        if in_range.sum() > 0:
            rmse_range = np.sqrt(np.mean(residuals[in_range] ** 2))
            bias_range = np.mean(residuals[in_range])
            print(f"  含水率 [{lo:>3}, {hi:>3}): N={in_range.sum():>3}, "
                  f"RMSE={rmse_range:.2f}, Bias={bias_range:.2f}, "
                  f"実測平均={y_test[in_range].mean():.2f}, 予測平均={pred[in_range].mean():.2f}")

    # 最大誤差サンプルの特定
    print("\n--- 誤差Top10サンプル ---")
    sorted_idx = np.argsort(-abs_residuals)
    print(f"{'順位':>4} {'実測値':>8} {'予測値':>8} {'誤差':>8} {'abs誤差':>8}")
    for i, idx in enumerate(sorted_idx[:10]):
        print(f"{i+1:>4} {y_test[idx]:>8.2f} {pred[idx]:>8.2f} "
              f"{residuals[idx]:>8.2f} {abs_residuals[idx]:>8.2f}")

    # PLS latent space での位置（train vs チェリー）
    print("\n--- PLSスコア空間でのチェリーの位置 ---")
    T_train = pls.transform(X_train)
    T_test = pls.transform(X_test)

    for comp in range(min(n_pls, 4)):
        train_scores = T_train[:, comp]
        test_scores = T_test[:, comp]
        print(f"  成分{comp+1}: Train [{train_scores.min():.2f}, {train_scores.max():.2f}] "
              f"(mean={train_scores.mean():.2f}, std={train_scores.std():.2f}), "
              f"Cherry [{test_scores.min():.2f}, {test_scores.max():.2f}] "
              f"(mean={test_scores.mean():.2f}, std={test_scores.std():.2f})")

    # チェリーのPLSスコアがtrain分布から外れている度合い
    print("\n--- チェリーPLSスコアのtrain分布からの逸脱度（Mahalanobis） ---")
    scaler = StandardScaler()
    T_train_scaled = scaler.fit_transform(T_train)
    T_test_scaled = scaler.transform(T_test)
    mahal_cherry = np.sqrt(np.sum(T_test_scaled ** 2, axis=1))
    print(f"  チェリーのMahalanobis距離: mean={mahal_cherry.mean():.2f}, "
          f"max={mahal_cherry.max():.2f}, "
          f"std={mahal_cherry.std():.2f}")

    # 他の全foldのMahalanobis距離との比較
    print("\n--- 各fold（樹種）のPLSスコアMahalanobis距離 ---")
    print(f"{'樹種':<16} {'Mean Mahal':>12} {'Max Mahal':>12} {'RMSE':>8}")
    print("-" * 52)
    for sp in species_unique:
        t_mask = species_all == sp
        tr_mask = ~t_mask
        X_tr = X_all[tr_mask]
        X_te = X_all[t_mask]
        y_tr = y_all[tr_mask]
        y_te = y_all[t_mask]

        pls_tmp = PLSRegression(n_components=n_pls)
        pls_tmp.fit(X_tr, y_tr)
        T_tr = pls_tmp.transform(X_tr)
        T_te = pls_tmp.transform(X_te)
        pred_tmp = pls_tmp.predict(X_te).ravel()
        rmse_tmp = np.sqrt(np.mean((pred_tmp - y_te) ** 2))

        sc = StandardScaler()
        sc.fit(T_tr)
        T_te_sc = sc.transform(T_te)
        mah = np.sqrt(np.sum(T_te_sc ** 2, axis=1))
        marker = " <<<" if sp == "チェリー" else ""
        print(f"{sp:<16} {mah.mean():>12.2f} {mah.max():>12.2f} {rmse_tmp:>8.2f}{marker}")

    print("\n" + "=" * 80)
    print("分析完了")
    print("=" * 80)


if __name__ == "__main__":
    main()
