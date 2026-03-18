"""全EDA一括実行スクリプト

各Issueの分析を実行し、グラフをoutputs/edaに保存、結果をコンソールに出力する。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eda.data_loader import load_train, load_test, get_spectral_columns, get_wavenumbers
from src.eda.issue2_spectrum_analysis import (
    compute_mean_spectrum_by_species,
    compute_mean_spectrum_by_moisture_range,
    compute_spectral_correlation_with_moisture,
    find_water_absorption_bands,
)
from src.eda.issue3_moisture_analysis import (
    compute_moisture_stats,
    compute_moisture_stats_by_species,
    detect_outliers,
    check_log_transform_benefit,
    check_sample_number_duplicates,
)
from src.eda.issue4_distribution_comparison import (
    check_species_overlap,
    compute_spectral_statistics,
    compute_pca_projection,
    compute_domain_shift_score,
)
from src.eda.issue8_spectral_outlier import detect_spectral_outliers
from src.eda.issue9_wood_type_analysis import (
    assign_wood_type,
    compute_mean_spectrum_by_wood_type,
    compute_moisture_stats_by_wood_type,
    compute_wood_type_ratio,
)
from src.eda.issue10_nonlinearity_analysis import (
    compare_linear_vs_log_correlation,
    find_most_nonlinear_wavenumbers,
)
from src.eda.issue11_baseline_analysis import (
    compute_baseline_drift,
    compute_effective_rank,
    compute_adjacent_correlation,
)
from src.eda.issue12_coverage_analysis import (
    assess_extrapolation_risk,
    compute_moisture_range_coverage,
)
from src.eda.issue13_loso_cv_validity import (
    compute_fold_coverage,
    identify_problematic_folds,
    compute_fold_sample_stats,
)
from src.eda.issue14_absorption_bands import (
    compute_correlation_map,
    compare_theory_vs_data,
)
from src.eda.issue15_pls_components import (
    run_loso_cv_pls,
    find_optimal_components,
    compare_log_vs_raw_target,
)
from src.eda.issue16_cv_map import compute_cv_map, find_high_cv_bands
from src.eda.issue17_drying_speed import (
    compute_drying_stats,
    assess_sample_imbalance,
)

# 日本語フォント設定
jp_fonts = [f.name for f in fm.fontManager.ttflist if "Hiragino" in f.name or "Gothic" in f.name or "Noto Sans CJK" in f.name]
if jp_fonts:
    plt.rcParams["font.family"] = jp_fonts[0]
else:
    plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = PROJECT_ROOT / "Input_data"


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def run():
    # データ読み込み
    print("データ読み込み中...")
    train = load_train(DATA_DIR)
    test = load_test(DATA_DIR)
    spectral_cols = get_spectral_columns(train)
    wavenumbers = get_wavenumbers(spectral_cols)
    print(f"  train: {train.shape}, test: {test.shape}")
    print(f"  スペクトル列数: {len(spectral_cols)}, 波数範囲: {wavenumbers.min():.1f} - {wavenumbers.max():.1f} cm⁻¹")

    # =========================================================
    # Issue #2: スペクトルの可視化
    # =========================================================
    separator("Issue #2: スペクトルの可視化（樹種別・含水率別）")

    mean_by_species = compute_mean_spectrum_by_species(train, spectral_cols)
    mean_by_moisture = compute_mean_spectrum_by_moisture_range(train, spectral_cols)
    corr = compute_spectral_correlation_with_moisture(train, spectral_cols)
    bands = find_water_absorption_bands(wavenumbers, corr)

    print(f"  水分吸収帯（正の相関トップ10波数）: {bands['top_positive'][:5]}")
    print(f"  水分吸収帯（負の相関トップ10波数）: {bands['top_negative'][:5]}")

    # 樹種別平均スペクトル
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax = axes[0, 0]
    for species in mean_by_species.index:
        ax.plot(wavenumbers, mean_by_species.loc[species, spectral_cols].values, label=species, alpha=0.7)
    ax.set_xlabel("波数 (cm⁻¹)")
    ax.set_ylabel("吸光度")
    ax.set_title("樹種別平均スペクトル")
    ax.legend(fontsize=6, ncol=2)
    ax.invert_xaxis()

    # 含水率別平均スペクトル
    ax = axes[0, 1]
    for label in mean_by_moisture.index:
        ax.plot(wavenumbers, mean_by_moisture.loc[label, spectral_cols].values, label=label, linewidth=2)
    ax.set_xlabel("波数 (cm⁻¹)")
    ax.set_ylabel("吸光度")
    ax.set_title("含水率レベル別平均スペクトル")
    ax.legend()
    ax.invert_xaxis()

    # 含水率との相関
    ax = axes[1, 0]
    corr_vals = np.array([corr[c] for c in spectral_cols])
    ax.plot(wavenumbers, corr_vals)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    for wn in [5200, 6900]:
        ax.axvline(wn, color="red", linestyle="--", alpha=0.5, label=f"{wn} cm⁻¹")
    ax.set_xlabel("波数 (cm⁻¹)")
    ax.set_ylabel("相関係数")
    ax.set_title("各波数と含水率の相関")
    ax.legend()
    ax.invert_xaxis()

    # 全サンプルスペクトル（透過度で色分け）
    ax = axes[1, 1]
    moisture = train["含水率"].values
    norm = plt.Normalize(moisture.min(), moisture.max())
    cmap = plt.cm.viridis
    for i in range(0, len(train), max(1, len(train) // 100)):
        color = cmap(norm(moisture[i]))
        ax.plot(wavenumbers, train.iloc[i][spectral_cols].values.astype(float), color=color, alpha=0.3, linewidth=0.3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, label="含水率 (%)")
    ax.set_xlabel("波数 (cm⁻¹)")
    ax.set_ylabel("吸光度")
    ax.set_title("全スペクトル（含水率で色分け）")
    ax.invert_xaxis()

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "issue02_spectrum_visualization.png", dpi=150)
    plt.close(fig)
    print("  → issue02_spectrum_visualization.png 保存完了")

    # =========================================================
    # Issue #3: 含水率の分布確認
    # =========================================================
    separator("Issue #3: 含水率の分布確認")

    stats = compute_moisture_stats(train)
    print(f"  基本統計:\n{stats.to_string()}")

    stats_by_sp = compute_moisture_stats_by_species(train)
    print(f"\n  樹種別統計:\n{stats_by_sp.to_string()}")

    outliers = detect_outliers(train)
    print(f"\n  Zスコア外れ値数: {outliers.sum()}")

    log_benefit = check_log_transform_benefit(train)
    print(f"  log変換: 元の歪度={log_benefit['original_skewness']:.3f}, log後={log_benefit['log_skewness']:.3f}, 推奨={log_benefit['recommend_log']}")

    dup_check = check_sample_number_duplicates(train)
    print(f"  sample number重複: {dup_check}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.hist(train["含水率"], bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("含水率 (%)")
    ax.set_ylabel("頻度")
    ax.set_title("含水率の分布")

    ax = axes[1]
    ax.hist(np.log1p(train["含水率"]), bins=50, edgecolor="black", alpha=0.7, color="orange")
    ax.set_xlabel("log1p(含水率)")
    ax.set_ylabel("頻度")
    ax.set_title("log1p変換後の分布")

    ax = axes[2]
    species_order = stats_by_sp.sort_values("mean").index
    bp_data = [train[train["樹種"] == sp]["含水率"].values for sp in species_order]
    ax.boxplot(bp_data, labels=species_order, vert=True)
    ax.set_ylabel("含水率 (%)")
    ax.set_title("樹種別含水率箱ひげ図")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "issue03_moisture_distribution.png", dpi=150)
    plt.close(fig)
    print("  → issue03_moisture_distribution.png 保存完了")

    # =========================================================
    # Issue #4: train/testスペクトル分布比較
    # =========================================================
    separator("Issue #4: train/testスペクトル分布比較")

    overlap = check_species_overlap(train, test)
    print(f"  樹種重複: {overlap}")

    spec_stats = compute_spectral_statistics(train, test, spectral_cols)
    print(f"  スペクトル統計:\n{spec_stats.to_string()}")

    domain_shift = compute_domain_shift_score(train, test, spectral_cols)
    print(f"  ドメインシフト: {domain_shift}")

    pca_result = compute_pca_projection(train, test, spectral_cols, n_components=10)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.scatter(pca_result["train_scores"][:, 0], pca_result["train_scores"][:, 1],
               alpha=0.5, s=10, label="train")
    ax.scatter(pca_result["test_scores"][:, 0], pca_result["test_scores"][:, 1],
               alpha=0.5, s=10, label="test")
    ax.set_xlabel(f"PC1 ({pca_result['explained_variance_ratio'][0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca_result['explained_variance_ratio'][1]:.1%})")
    ax.set_title("PCA: train vs test")
    ax.legend()

    ax = axes[1]
    ax.scatter(pca_result["train_scores"][:, 0], pca_result["train_scores"][:, 2],
               alpha=0.5, s=10, label="train")
    ax.scatter(pca_result["test_scores"][:, 0], pca_result["test_scores"][:, 2],
               alpha=0.5, s=10, label="test")
    ax.set_xlabel(f"PC1 ({pca_result['explained_variance_ratio'][0]:.1%})")
    ax.set_ylabel(f"PC3 ({pca_result['explained_variance_ratio'][2]:.1%})")
    ax.set_title("PCA: PC1 vs PC3")
    ax.legend()

    ax = axes[2]
    cumvar = np.cumsum(pca_result["explained_variance_ratio"])
    ax.bar(range(1, 11), pca_result["explained_variance_ratio"], alpha=0.7, label="個別")
    ax.plot(range(1, 11), cumvar, "ro-", label="累積")
    ax.set_xlabel("主成分")
    ax.set_ylabel("寄与率")
    ax.set_title("PCA寄与率")
    ax.legend()

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "issue04_train_test_comparison.png", dpi=150)
    plt.close(fig)
    print("  → issue04_train_test_comparison.png 保存完了")

    # =========================================================
    # Issue #8: スペクトル異常値検出
    # =========================================================
    separator("Issue #8: スペクトル異常値検出")

    outlier_result = detect_spectral_outliers(train, spectral_cols)
    print(f"  検出された異常値数: {outlier_result['outlier_count']} / {len(train)}")
    print(f"  T²閾値: {outlier_result['t2_threshold']:.2f}, Q閾値: {outlier_result['q_threshold']:.2f}")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(outlier_result["t2_values"], outlier_result["q_values"], alpha=0.5, s=10)
    ax.axvline(outlier_result["t2_threshold"], color="red", linestyle="--", label=f"T² 閾値")
    ax.axhline(outlier_result["q_threshold"], color="blue", linestyle="--", label=f"Q 閾値")
    ax.set_xlabel("Hotelling's T²")
    ax.set_ylabel("Q残差")
    ax.set_title("異常値検出（T² vs Q残差）")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "issue08_outlier_detection.png", dpi=150)
    plt.close(fig)
    print("  → issue08_outlier_detection.png 保存完了")

    # =========================================================
    # Issue #9: 針葉樹 vs 広葉樹
    # =========================================================
    separator("Issue #9: 針葉樹 vs 広葉樹のスペクトル特性")

    train_wt = assign_wood_type(train)
    mean_by_wt = compute_mean_spectrum_by_wood_type(train_wt, spectral_cols)
    moisture_by_wt = compute_moisture_stats_by_wood_type(train_wt)
    wt_ratio = compute_wood_type_ratio(train, test)

    print(f"  含水率統計（木材タイプ別）:\n{moisture_by_wt.to_string()}")
    print(f"  wood type比率: {wt_ratio}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for wt in mean_by_wt.index:
        ax.plot(wavenumbers, mean_by_wt.loc[wt, spectral_cols].values, label=wt, linewidth=2)
    ax.set_xlabel("波数 (cm⁻¹)")
    ax.set_ylabel("吸光度")
    ax.set_title("針葉樹 vs 広葉樹 平均スペクトル")
    ax.legend()
    ax.invert_xaxis()

    ax = axes[1]
    diff = mean_by_wt.loc["softwood", spectral_cols].values.astype(float) - mean_by_wt.loc["hardwood", spectral_cols].values.astype(float)
    ax.plot(wavenumbers, diff)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("波数 (cm⁻¹)")
    ax.set_ylabel("差分（softwood - hardwood）")
    ax.set_title("スペクトル差分")
    ax.invert_xaxis()

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "issue09_wood_type_analysis.png", dpi=150)
    plt.close(fig)
    print("  → issue09_wood_type_analysis.png 保存完了")

    # =========================================================
    # Issue #10: 非線形性確認
    # =========================================================
    separator("Issue #10: 含水率とスペクトルの非線形性確認")

    log_vs_linear = compare_linear_vs_log_correlation(train, spectral_cols)
    log_improves_pct = log_vs_linear["log_improves"].mean()
    print(f"  log変換で相関が改善する波数の割合: {log_improves_pct:.1%}")

    nonlinear_wn = find_most_nonlinear_wavenumbers(train, spectral_cols)
    print(f"  最も非線形性が高い波数（トップ5）:\n{nonlinear_wn.head().to_string()}")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(wavenumbers, log_vs_linear["linear_corr"].values, label="線形相関", alpha=0.7)
    ax.plot(wavenumbers, log_vs_linear["log_corr"].values, label="log相関", alpha=0.7)
    ax.set_xlabel("波数 (cm⁻¹)")
    ax.set_ylabel("|相関係数|")
    ax.set_title("線形 vs log変換 相関係数")
    ax.legend()
    ax.invert_xaxis()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "issue10_nonlinearity.png", dpi=150)
    plt.close(fig)
    print("  → issue10_nonlinearity.png 保存完了")

    # =========================================================
    # Issue #11: ベースラインドリフト・多重共線性
    # =========================================================
    separator("Issue #11: ベースラインドリフトと多重共線性")

    drift = compute_baseline_drift(train, spectral_cols)
    print(f"  ベースラインドリフト: mean={drift['mean_drift']:.4f}, max={drift['max_drift']:.4f}")

    eff_rank = compute_effective_rank(train, spectral_cols)
    print(f"  有効ランク: 95%={eff_rank['rank_95']}, 99%={eff_rank['rank_99']} / {eff_rank['total_features']}次元")
    print(f"  圧縮率(95%): {eff_rank['compression_ratio_95']:.4f}")

    adj_corr = compute_adjacent_correlation(train, spectral_cols)
    print(f"  隣接波数相関: mean={adj_corr['mean_adjacent_corr']:.4f}, >0.99比率={adj_corr['high_corr_ratio']:.1%}")

    fig, ax = plt.subplots(figsize=(10, 5))
    cumvar = eff_rank["cumulative_variance"]
    ax.plot(range(1, len(cumvar) + 1), cumvar)
    ax.axhline(0.95, color="red", linestyle="--", label=f"95% (rank={eff_rank['rank_95']})")
    ax.axhline(0.99, color="orange", linestyle="--", label=f"99% (rank={eff_rank['rank_99']})")
    ax.set_xlabel("主成分数")
    ax.set_ylabel("累積寄与率")
    ax.set_title("累積分散寄与率")
    ax.legend()
    ax.set_xlim(0, 50)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "issue11_effective_rank.png", dpi=150)
    plt.close(fig)
    print("  → issue11_effective_rank.png 保存完了")

    # =========================================================
    # Issue #12: 含水率カバレッジ
    # =========================================================
    separator("Issue #12: 含水率カバレッジ（外挿リスク）")

    extrap_risk = assess_extrapolation_risk(train, test, spectral_cols)
    print(f"  外挿リスク: {extrap_risk}")

    moisture_cov = compute_moisture_range_coverage(train)
    print(f"  含水率カバレッジ: {moisture_cov}")

    # =========================================================
    # Issue #13: LOSO-CV妥当性
    # =========================================================
    separator("Issue #13: LOSO-CVの妥当性確認")

    fold_cov = compute_fold_coverage(train)
    print(f"  fold別カバレッジ:\n{fold_cov.to_string()}")

    fold_stats = compute_fold_sample_stats(train)
    print(f"\n  fold別サンプル数:\n{fold_stats.to_string()}")

    problems = identify_problematic_folds(train)
    if len(problems) > 0:
        print(f"\n  問題のあるfold:\n{problems.to_string()}")
    else:
        print("\n  問題のあるfoldはありません")

    fig, ax = plt.subplots(figsize=(12, 6))
    for _, row in fold_cov.iterrows():
        color = "green" if row["is_covered"] else "red"
        ax.barh(row["holdout_species"], row["holdout_max"] - row["holdout_min"],
                left=row["holdout_min"], color=color, alpha=0.6, edgecolor="black")
    ax.set_xlabel("含水率 (%)")
    ax.set_title("LOSO-CV fold別含水率レンジ（緑=カバー済, 赤=未カバー）")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "issue13_loso_cv_coverage.png", dpi=150)
    plt.close(fig)
    print("  → issue13_loso_cv_coverage.png 保存完了")

    # =========================================================
    # Issue #14: 近赤外吸収帯照合
    # =========================================================
    separator("Issue #14: 近赤外吸収帯の分光学的照合")

    corr_map = compute_correlation_map(train, spectral_cols)
    theory_vs_data = compare_theory_vs_data(corr_map, wavenumbers)
    print(f"  理論値 vs データ:\n{theory_vs_data.to_string()}")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(corr_map.index, corr_map.values)
    for _, row in theory_vs_data.iterrows():
        ax.axvline(row["theoretical_wn"], color="red", linestyle="--", alpha=0.5)
        ax.annotate(row["band_name"], xy=(row["theoretical_wn"], row["correlation_at_theory"]),
                    fontsize=7, rotation=45, ha="left")
    ax.set_xlabel("波数 (cm⁻¹)")
    ax.set_ylabel("相関係数")
    ax.set_title("含水率との相関マップ + 理論的吸収帯")
    ax.invert_xaxis()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "issue14_absorption_bands.png", dpi=150)
    plt.close(fig)
    print("  → issue14_absorption_bands.png 保存完了")

    # =========================================================
    # Issue #15: PLSコンポーネント数評価
    # =========================================================
    separator("Issue #15: PLSコンポーネント数の事前評価")

    print("  PLS LOSO-CV実行中（raw & log）... これには数分かかります")
    pls_comparison = compare_log_vs_raw_target(train, spectral_cols, max_components=20)
    print(f"  PLS結果:\n{pls_comparison.to_string()}")

    raw_result = run_loso_cv_pls(train, spectral_cols, max_components=20, use_log=False)
    optimal = find_optimal_components(raw_result)
    print(f"  最適成分数: {optimal}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(pls_comparison["n_components"], pls_comparison["rmse_raw"], "b-o", label="Raw目的変数", markersize=4)
    ax.plot(pls_comparison["n_components"], pls_comparison["rmse_log"], "r-o", label="Log目的変数", markersize=4)
    ax.set_xlabel("PLS成分数")
    ax.set_ylabel("LOSO-CV RMSE")
    ax.set_title("PLS成分数 vs RMSE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "issue15_pls_components.png", dpi=150)
    plt.close(fig)
    print("  → issue15_pls_components.png 保存完了")

    # =========================================================
    # Issue #16: 変動係数マップ
    # =========================================================
    separator("Issue #16: 各波数の変動係数（CV）マップ")

    cv_map = compute_cv_map(train, spectral_cols)
    high_cv = find_high_cv_bands(train, spectral_cols)
    print(f"  高CV波数（トップ10）:\n{high_cv.head(10).to_string()}")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(cv_map.index, cv_map.values)
    ax.set_xlabel("波数 (cm⁻¹)")
    ax.set_ylabel("変動係数 (CV)")
    ax.set_title("各波数の変動係数マップ")
    ax.invert_xaxis()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "issue16_cv_map.png", dpi=150)
    plt.close(fig)
    print("  → issue16_cv_map.png 保存完了")

    # =========================================================
    # Issue #17: 乾燥速度・サンプル数偏り
    # =========================================================
    separator("Issue #17: 樹種ごとの乾燥速度とサンプル数偏り")

    drying = compute_drying_stats(train)
    print(f"  乾燥統計:\n{drying.to_string()}")

    imbalance = assess_sample_imbalance(train, spectral_cols)
    print(f"  サンプル不均衡: {imbalance}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.barh(drying["species"], drying["sample_count"], color="steelblue")
    ax.set_xlabel("サンプル数")
    ax.set_title("樹種別サンプル数")

    ax = axes[1]
    ax.barh(drying["species"], drying["moisture_range"], color="coral")
    ax.set_xlabel("含水率レンジ (%)")
    ax.set_title("樹種別含水率レンジ")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "issue17_drying_imbalance.png", dpi=150)
    plt.close(fig)
    print("  → issue17_drying_imbalance.png 保存完了")

    # =========================================================
    # サマリー
    # =========================================================
    separator("EDA完了サマリー")
    print(f"  全グラフ保存先: {OUTPUT_DIR}")
    print(f"  生成ファイル数: {len(list(OUTPUT_DIR.glob('*.png')))}")


if __name__ == "__main__":
    run()
