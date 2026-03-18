# 近赤外研究会 スペクトル分析チャレンジ

## 概要

木材の近赤外スペクトルデータから**含水率（%）を予測する回帰タスク**。

近赤外分光法における従来のケモメトリクス的アプローチ（PLS/PCRなど）と、機械学習・深層学習アプローチを比較・検証することを目的としたコンペティション。

- **主催**: 近赤外研究会
- **プラットフォーム**: SIGNATE
- **評価指標**: RMSE（小さいほど良い）

---

## データ

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

## 競技計画

### フェーズ0：環境・検証戦略の確立

同一の物理サンプルを乾燥させながら繰り返し測定しているため、`sample number` の重複状況を確認し、適切な検証戦略を決定する。

| 確認結果 | 採用する検証戦略 |
|---|---|
| `sample number` が測定回ごとのID | **LOSO-CV**（Leave-One-Species-Out）|
| `sample number` が物理サンプルごとのID | **GroupKFold**（物理サンプル単位） + LOSO-CV |

- [ ] `sample number` の重複確認
- [ ] 評価パイプラインの整備（RMSE計算・ログ管理）

---

### フェーズ1：EDA（データ理解）

| Issue | 内容 |
|---|---|
| [#2](../../issues/2) | スペクトルの可視化（樹種別・含水率別） |
| [#3](../../issues/3) | 含水率の分布確認・対数変換の必要性判断 |
| [#4](../../issues/4) | train/testのスペクトル分布比較（ドメインシフト確認） |

**確認ポイント**
- 含水率の分布（0.84〜298.6%）→ 右裾の確認 → **log1p変換の必要性を判断**
- 水分吸収帯（約5,200 cm⁻¹・6,900 cm⁻¹）の挙動
- スペクトルの異常値・欠損の有無
- train/testのPCA・t-SNEによる分布比較

---

### フェーズ2：前処理パイプライン構築

| Issue | 内容 |
|---|---|
| [#5](../../issues/5) | SNV（Standard Normal Variate）の実装・効果検証 |
| [#6](../../issues/6) | MSC（Multiple Scatter Correction）の実装・効果検証 |
| [#7](../../issues/7) | Savitzky-Golay微分（1次・2次）の実装・効果検証 |

**前処理の適用順序**（順序が重要）
```
Raw → [SNV or MSC] → [1次 or 2次微分] → [波数範囲選択]
```

**比較する前処理の組み合わせ**

| 前処理 | 内容 |
|---|---|
| Raw | 前処理なし |
| SNV | 散乱補正 |
| MSC | 散乱補正（リファレンス基準） |
| SNV + 1st deriv | 散乱補正 → 1次微分 |
| SNV + 2nd deriv | 散乱補正 → 2次微分 |
| MSC + 1st deriv | 散乱補正 → 1次微分 |
| MSC + 2nd deriv | 散乱補正 → 2次微分 |

> ⚠️ **MSCのリファレンス（train平均スペクトル）はtrainのみで計算し、testに適用する**

---

### フェーズ3：ケモメトリクス系モデル（ベースライン）

- **PLS回帰**：成分数をLOSO-CVでチューニング（最適成分数: **4成分**、ベースラインRMSE: 22.3）
- **PCR**：主成分数をLOSO-CVでチューニング
- PLSのスコアベクトルは後続のMLモデルの特徴量としても活用

> ⚠️ **PLSでは目的変数のlog変換を使わない**（log変換すると12成分以降でRMSEが発散することをEDAで確認）

---

### フェーズ4：機械学習系モデル

> ⚠️ **1,555次元を直接入力すると過学習しやすいため、PLS/PCAスコアを特徴量として使用**

| モデル | 特徴量 | チューニング |
|---|---|---|
| Ridge / Lasso / ElasticNet | PLSスコア or PCAスコア | Optuna |
| SVR | PLSスコア or PCAスコア | Optuna |
| LightGBM / XGBoost | PLSスコア or PCAスコア | Optuna |

---

### フェーズ5：深層学習系モデル

- **1D-CNN**（優先）：スペクトルの局所パターンを学習
- **Transformer**（オプション）：データ量（1,322サンプル）が少ないためリスクあり
- データ拡張の検討（ノイズ付加・波数シフト・ミックスアップ）

---

### フェーズ6：モデル比較・アンサンブル・最終提出

- 全手法のLOSO-CV RMSE一覧表を作成し比較
- 上位モデルのアンサンブル（加重平均・スタッキング）
- 最終提出

---

## EDAから得られた設計指針まとめ

| 設計項目 | 指針 |
|---|---|
| 検証戦略 | LOSO-CV（Leave-One-Species-Out）。ベイスギfoldは含水率外挿のため参考値として扱う |
| 目的変数（PLS/PCR） | **raw（変換なし）**。log変換すると12成分以降でRMSEが発散 |
| 目的変数（ML/DLモデル） | **log1p変換** → 予測後にexpm1で逆変換 |
| 特徴量 | 1,555次元直接入力は過学習リスク大 → PLSスコア（4成分）またはPCAスコアを推奨 |
| スペクトル前処理 | SNV or MSC → Savitzky-Golay微分（順序重要） |
| 異常値 | Hotelling's T² > 99%点 かつ Q残差 > 99%点 の複数サンプルを除外検討 |
| サンプル重み付け | 不均衡比率3.6倍（ホワイトオーク51件 vs トチ183件） → ML訓練時に重み付けを検討 |
| 高CVバンド（8,630〜8,660 cm⁻¹） | **アウトライアーの影響**（除外後はCV≈3.9）。信号としては使わない |
| 重要な吸収帯 | 水分帯：5,200 cm⁻¹（OH結合倍音）・6,900 cm⁻¹（OH基本音）付近 |
| 外挿リスク | testの含水率範囲はtrainでカバーされているが、特定fold（ベイスギ）は要注意 |

---

## ディレクトリ構成（developブランチ）

```
.
├── Input_data/                  # コンペティション入力データ
│   ├── train.csv                #   学習データ（1,322サンプル）
│   ├── test.csv                 #   評価データ（550サンプル）
│   └── sample_submit.csv        #   提出サンプル
│
├── src/                         # モジュール本体
│   ├── eda/                     #   EDA分析（Issue #2〜#17）
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
│   ├── preprocessing/           #   スペクトル前処理（Issue #18〜#25, #38〜#50）
│   │   ├── issue18_snv.py       #     SNV
│   │   ├── issue19_msc.py       #     MSC
│   │   ├── issue20_savgol.py    #     Savitzky-Golay微分
│   │   ├── issue21_preprocessing_comparison.py
│   │   ├── issue22_epo.py       #     EPO
│   │   ├── issue23_osc.py       #     OSC
│   │   ├── issue24_detrending.py
│   │   ├── issue25_emsc.py      #     EMSC
│   │   ├── issue38_tca.py       #     TCA
│   │   ├── issue39_wavelet_transform.py
│   │   ├── issue40_opls.py      #     OPLS
│   │   ├── issue43_dipls.py     #     di-PLS
│   │   ├── issue44_subspace_alignment.py
│   │   ├── issue45_constituent_emsc.py
│   │   ├── issue46_mmd_selection.py
│   │   ├── issue47_jsmkpls.py   #     JSMKPLS
│   │   ├── issue48_jda.py       #     JDA
│   │   ├── issue49_bda.py       #     BDA
│   │   ├── issue50_kmm.py       #     KMM
│   │   └── issue50_weighted_tca.py
│   │
│   ├── modeling/                #   モデリング（Issue #26〜#34）
│   │   ├── issue26_pls_epo_optimization.py
│   │   ├── issue27_nonlinear_models.py
│   │   ├── issue28_domain_adaptation.py
│   │   ├── issue29_epo_preprocessing_combo.py
│   │   ├── issue30_stacking.py
│   │   ├── issue31_submission_pipeline.py
│   │   ├── issue32_weighted_ensemble.py
│   │   ├── issue33_lwpls.py
│   │   └── issue34_test_augmented_epo.py
│   │
│   └── feature_engineering/     #   特徴量エンジニアリング（Issue #37, #41）
│       ├── issue37_species_invariant.py
│       └── issue41_spectral_autocorrelation.py
│
├── scripts/                     # 実行スクリプト（run_*）
│   ├── eda/
│   │   └── run_all_eda.py       #     全EDA一括実行
│   ├── preprocessing/
│   │   └── run_issue22_25.py    #     前処理 #22-25 一括実行
│   ├── modeling/
│   │   ├── run_issue26.py       #     PLS+EPOグリッドサーチ
│   │   ├── run_issue27_29.py    #     非線形モデル + EPO組合せ
│   │   ├── run_issue28.py       #     ドメイン適応評価
│   │   ├── run_issue30.py       #     スタッキング評価
│   │   ├── run_issue31.py       #     提出ファイル生成
│   │   ├── run_issue32_improved_ensemble.py
│   │   ├── run_issue33_34.py    #     LWPLS + Test-Aug EPO
│   │   ├── run_deep_dive.py     #     前処理深掘り
│   │   ├── run_deep_dive_part2.py
│   │   ├── run_deep_dive_part3.py
│   │   ├── run_da_deep_dive.py  #     ドメイン適応深掘り (JDA/BDA/WTCA)
│   │   ├── run_improved_ensemble_v2.py
│   │   ├── run_new_preprocessing_eval.py   # 新規前処理評価
│   │   ├── run_new_preprocessing_eval2.py
│   │   ├── run_new_preprocessing_eval3.py
│   │   ├── run_new_preprocessing_eval3b.py
│   │   ├── run_new_preprocessing_eval3c.py
│   │   └── run_submission_v2.py #     改良版提出 (RMSE=17.03)
│   └── feature_engineering/
│       └── run_issue37.py       #     樹種不変特徴量評価
│
├── outputs/                     # 可視化・分析結果
│   ├── eda/                     #     EDA可視化 (15 PNG)
│   ├── preprocessing/           #     前処理可視化 (8 PNG)
│   ├── modeling/                #     モデリング可視化 (8 PNG)
│   └── feature_engineering/     #     特徴量可視化 (1 PNG)
│
├── tests/                       # ユニットテスト（全issue番号付き、38ファイル）
├── main.py                      # メインエントリポイント
├── pyproject.toml               # プロジェクト設定
└── uv.lock                      # 依存関係ロック
```
