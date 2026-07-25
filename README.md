# 近赤外研究会 スペクトル分析チャレンジ

## 概要

木材の近赤外スペクトルデータから**含水率（%）を予測する回帰タスク**。

近赤外分光法における従来のケモメトリクス的アプローチ（PLS/PCRなど）と、機械学習・深層学習アプローチを比較・検証することを目的としたコンペティション。

- **主催**: 近赤外研究会
- **プラットフォーム**: SIGNATE
- **評価指標**: RMSE（小さいほど良い）
- **現在のベストLBスコア**: RMSE = 17.03

---

## データ

> **本リポジトリにデータは同梱されていません。** データはSIGNATEコンペティションページで規約に同意の上、各自で取得し `Input_data/` に配置してください（コンペティションデータの再配布はSIGNATE規約により行いません）。

| ファイル | 内容 | サンプル数 |
|---|---|---|
| `train.csv` | 学習用データ | 1,322 |
| `test.csv` | 評価用データ | 550 |
| `sample_submit.csv` | 提出サンプル | 550 |

### カラム説明

| カラム名 | 説明 |
|---|---|
| `sample number` | サンプル通し番号 |
| `species number` | 樹種番号 |
| `樹種` | 樹種名 |
| `含水率` | 含水率（%）← **予測対象**（trainのみ） |
| `9993.76781` 〜 `3999.82139` | 波数（cm⁻¹）のスペクトル強度（1,555列） |

### 樹種の構成

- **train（13種）**: イチョウ、ウエンジ、ウォールナット、クリ、スプルース、チェリー、トチ、ナラ、ヒノキ、ベイスギ、ベイマツ、ホワイトオーク、米ヒバ
- **test（6種）**: クスノキ、ケヤキ、スギ、タモ、チーク、ヤマザクラ

> ⚠️ **trainとtestで樹種が完全に異なる** → 未知樹種への汎化が必須

---

## 核心的な課題認識

trainとtestで樹種が完全に異なるため、樹種固有の特徴に過学習するとtestで性能が出ない。
**水分由来のスペクトル特徴（水分吸収帯：約5,200 cm⁻¹ および 6,900 cm⁻¹付近）を捉えることが鍵。**

---

## 提出履歴とスコア

| バージョン | 構成 | fold-RMSE | LBスコア | 備考 |
|---|---|---|---|---|
| v2 | 5モデル平均（EPO+PLS系） | 17.03 | **17.03** | ベスト提出 |
| v9_top12 | 12種前処理×PLS均等平均 | 18.29 | 17.14 | CV-LB差が小さい |
| v8_blend | v6+v9ブレンド | — | 18.11 | |
| v10_mega | メガブレンド | — | 18.62 | |
| best11 | 15モデル重み最適化 | 14.10 | 21.56 | CV過学習 |
| v12_opt12 | 18モデル最適化12選 | — | 21.56 | CV過学習 |

### CV-LB乖離の分析（Issue #106）

- **重み最適化アンサンブルはCVを過学習**する傾向が強い（best11: CV=14.10→LB=21.56）
- **均等平均のシンプルなアンサンブル**がLBとの対応が良い（v9: CV=18.29→LB=17.14）
- **含水率類似パターンのCV**がLBスコアに最も近い指標となる（v9 Pattern C: CV=16.70 vs LB=17.14、差=0.44）

---

## 実装済み手法

### 前処理（22モジュール）

| カテゴリ | 手法 |
|---|---|
| 散乱補正 | SNV, MSC, EMSC, Constituent EMSC |
| 微分処理 | Savitzky-Golay（1次・2次微分） |
| ベースライン補正 | Detrending, AsLS |
| 直交補正 | EPO, OSC, OPLS |
| ドメイン適応 | TCA, Weighted TCA, JDA, BDA, KMM, di-PLS, Subspace Alignment, MMD Selection |
| その他 | Wavelet Transform, JSMKPLS |

### モデリング（14モジュール）

| カテゴリ | 手法 |
|---|---|
| ケモメトリクス | PLS回帰（成分数4）, PCR, LWPLS |
| 非線形モデル | SVR, RandomForest, GradientBoosting, LightGBM, MLP |
| 特徴選択 | siPLS, iPLS（区間PLS選択） |
| 目的変数変換 | sqrt変換, log1p変換 |
| アンサンブル | 均等平均, 重み付き平均, スタッキング |
| ドメイン適応 | TCA+PLS, Test-Augmented EPO, Pseudo Labeling |

### 特徴量エンジニアリング（2モジュール）

- 樹種不変特徴量（Species-Invariant Features）
- スペクトル自己相関特徴（Spectral Autocorrelation）

---

## EDAから得られた設計指針まとめ

| 設計項目 | 指針 |
|---|---|
| 検証戦略 | LOSO-CV（Leave-One-Species-Out）。ベイスギfoldは含水率外挿のため参考値として扱う |
| 目的変数（PLS/PCR） | **raw（変換なし）**。log変換すると12成分以降でRMSEが発散 |
| 目的変数（ML/DLモデル） | **log1p変換** → 予測後にexpm1で逆変換。またはsqrt変換 |
| 特徴量 | 1,555次元直接入力は過学習リスク大 → PLSスコア（4成分）またはPCAスコアを推奨 |
| スペクトル前処理 | SNV or MSC → Savitzky-Golay微分（順序重要） |
| 異常値 | Hotelling's T² > 99%点 かつ Q残差 > 99%点 の複数サンプルを除外検討 |
| サンプル重み付け | 不均衡比率3.6倍（ホワイトオーク51件 vs トチ183件） → ML訓練時に重み付けを検討 |
| 高CVバンド（8,630〜8,660 cm⁻¹） | **アウトライアーの影響**（除外後はCV≈3.9）。信号としては使わない |
| 重要な吸収帯 | 水分帯：5,200 cm⁻¹（OH結合倍音）・6,900 cm⁻¹（OH基本音）付近 |
| 外挿リスク | testの含水率範囲はtrainでカバーされているが、特定fold（ベイスギ）は要注意 |
| アンサンブル戦略 | 均等平均がLBとの対応が良い。重み最適化はCV過学習のリスクが高い |

---

## ディレクトリ構成

```
.
├── Input_data/                  # コンペ入力データ（SIGNATEから取得・非同梱・Git管理外）
│
├── src/                         # モジュール本体
│   ├── eda/                     #   EDA分析（Issue #2〜#17、14ファイル）
│   │   ├── data_loader.py       #     データ読み込み共通ユーティリティ
│   │   ├── issue2_spectrum_analysis.py
│   │   ├── issue3_moisture_analysis.py
│   │   ├── issue4_distribution_comparison.py
│   │   ├── issue8_spectral_outlier.py
│   │   ├── issue9_wood_type_analysis.py
│   │   ├── issue10_nonlinearity_analysis.py
│   │   ├── issue11_baseline_analysis.py
│   │   ├── issue12_coverage_analysis.py
│   │   ├── issue13_loso_cv_validity.py
│   │   ├── issue14_absorption_bands.py
│   │   ├── issue15_pls_components.py
│   │   ├── issue16_cv_map.py
│   │   └── issue17_drying_speed.py
│   │
│   ├── preprocessing/           #   スペクトル前処理（Issue #18〜#50、22ファイル）
│   │   ├── issue18_snv.py       #     SNV
│   │   ├── issue19_msc.py       #     MSC
│   │   ├── issue20_savgol.py    #     Savitzky-Golay微分
│   │   ├── issue21_preprocessing_comparison.py
│   │   ├── issue22_epo.py       #     EPO
│   │   ├── issue23_osc.py       #     OSC
│   │   ├── issue24_detrending.py
│   │   ├── issue25_emsc.py      #     EMSC
│   │   ├── issue38_tca.py       #     TCA
│   │   ├── issue39_wavelet_transform.py   # Wavelet
│   │   ├── issue40_opls.py      #     OPLS
│   │   ├── issue43_dipls.py     #     di-PLS
│   │   ├── issue44_subspace_alignment.py  # SA
│   │   ├── issue45_constituent_emsc.py    # C-EMSC
│   │   ├── issue46_mmd_selection.py       # MMD
│   │   ├── issue47_jsmkpls.py   #     JSMKPLS
│   │   ├── issue48_jda.py       #     JDA
│   │   ├── issue49_bda.py       #     BDA
│   │   ├── issue50_kmm.py       #     KMM
│   │   ├── issue50_weighted_tca.py        # Weighted TCA
│   │   └── issue62_additional_preprocessing.py  # AsLS, Piecewise MSC等
│   │
│   ├── modeling/                #   モデリング（Issue #26〜#66、14ファイル）
│   │   ├── issue26_pls_epo_optimization.py
│   │   ├── issue27_nonlinear_models.py
│   │   ├── issue28_domain_adaptation.py
│   │   ├── issue29_epo_preprocessing_combo.py
│   │   ├── issue30_stacking.py
│   │   ├── issue31_submission_pipeline.py
│   │   ├── issue32_weighted_ensemble.py
│   │   ├── issue33_lwpls.py     #     LWPLS
│   │   ├── issue34_test_augmented_epo.py
│   │   ├── issue62_grid_evaluation.py
│   │   ├── issue63_nonlinear_optuna.py
│   │   ├── issue64_mlp_experiment.py
│   │   ├── issue65_feature_selection.py   # siPLS, iPLS
│   │   └── issue66_target_transforms.py   # sqrt, log変換
│   │
│   └── feature_engineering/     #   特徴量エンジニアリング（Issue #37, #41）
│       ├── issue37_species_invariant.py
│       └── issue41_spectral_autocorrelation.py
│
├── scripts/                     # 実行スクリプト（81ファイル）
│   ├── eda/
│   │   └── run_all_eda.py
│   ├── preprocessing/
│   │   └── run_issue22_25.py
│   ├── modeling/                #   提出・実験スクリプト（主要なもの）
│   │   ├── run_submission_v2.py           # v2提出 (LB=17.03)
│   │   ├── run_submission_v3.py           # v3提出
│   │   ├── run_submission_v5.py           # v5提出 (fold-RMSE=15.22)
│   │   ├── run_submit_v9_diverse.py       # v9多様性PLS (LB=17.14)
│   │   ├── run_issue105_final_ensemble.py # 18モデル統合アンサンブル
│   │   ├── run_issue106_cv_patterns.py    # CVパターン別RMSE分析
│   │   └── ...                            # その他70+スクリプト
│   └── feature_engineering/
│       └── run_issue37.py
│
├── outputs/                     # 可視化・分析結果（実行時に生成・Git管理外）
│
├── tests/                       # ユニットテスト（64ファイル）
├── main.py                      # メインエントリポイント
├── pyproject.toml               # プロジェクト設定
└── uv.lock                      # 依存関係ロック
```
