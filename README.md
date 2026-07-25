# 近赤外研究会 スペクトル分析チャレンジ

木材の近赤外（NIR）スペクトルから**含水率（%）を予測する回帰タスク**の解法リポジトリ。
近赤外分光法における従来のケモメトリクス的アプローチ（PLS/PCR など）と、機械学習・深層学習アプローチを比較・検証することを目的としています。

- **主催**: 近赤外研究会
- **プラットフォーム**: SIGNATE
- **評価指標**: RMSE（小さいほど良い）
- **ベスト LB スコア**: RMSE = 17.03（提出 v2）

---

## セットアップ

依存関係は [uv](https://github.com/astral-sh/uv) で管理しています。

```bash
# 依存関係の同期（.venv を作成）
uv sync

# データを配置（下記「データ」参照）
mkdir -p Input_data
# train.csv / test.csv / sample_submit.csv を Input_data/ に置く

# テスト実行
uv run pytest

# 動作確認
uv run python main.py
```

> Python 3.11 以上が必要です。主要依存: numpy, pandas, scikit-learn, scipy, lightgbm, pywavelets, matplotlib, seaborn, diplslib。

---

## データ

> **本リポジトリにデータは同梱されていません。** SIGNATE コンペティションページで規約に同意の上、各自で取得し `Input_data/` に配置してください（コンペティションデータの再配布は SIGNATE 規約により行いません。`Input_data/` は `.gitignore` で除外済みです）。

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
| `含水率` | 含水率（%）← **予測対象**（train のみ） |
| `9993.76781` 〜 `3999.82139` | 波数（cm⁻¹）のスペクトル強度（1,555 列） |

### 樹種の構成

- **train（13 種）**: イチョウ、ウエンジ、ウォールナット、クリ、スプルース、チェリー、トチ、ナラ、ヒノキ、ベイスギ、ベイマツ、ホワイトオーク、米ヒバ
- **test（6 種）**: クスノキ、ケヤキ、スギ、タモ、チーク、ヤマザクラ

> ⚠️ **train と test で樹種が完全に異なる** → 未知樹種への汎化が必須

---

## 核心的な課題認識

train と test で樹種が完全に異なるため、樹種固有の特徴に過学習すると test で性能が出ません。
**水分由来のスペクトル特徴（水分吸収帯：約 5,200 cm⁻¹ および 6,900 cm⁻¹ 付近）を捉えることが鍵。**

---

## 提出履歴とスコア

| バージョン | 構成 | fold-RMSE | LB スコア | 備考 |
|---|---|---|---|---|
| v2 | 5 モデル平均（EPO+PLS 系） | 17.03 | **17.03** | ベスト提出 |
| v9_top12 | 12 種前処理 × PLS 均等平均 | 18.29 | 17.14 | CV-LB 差が小さい |
| v8_blend | v6+v9 ブレンド | — | 18.11 | |
| v10_mega | メガブレンド | — | 18.62 | |
| best11 | 15 モデル重み最適化 | 14.10 | 21.56 | CV 過学習 |
| v12_opt12 | 18 モデル最適化 12 選 | — | 21.56 | CV 過学習 |

### CV-LB 乖離の分析

- **重み最適化アンサンブルは CV を過学習**する傾向が強い（best11: CV=14.10 → LB=21.56）
- **均等平均のシンプルなアンサンブル**が LB との対応が良い（v9: CV=18.29 → LB=17.14）
- **含水率類似パターンの CV** が LB スコアに最も近い指標となる（v9 Pattern C: CV=16.70 vs LB=17.14、差=0.44）

---

## 実装済み手法（`src/` 再利用モジュール）

### 前処理 — `src/preprocessing/`（12 モジュール）

| モジュール | 手法 |
|---|---|
| `issue18_snv` | SNV（Standard Normal Variate） |
| `issue19_msc` | MSC（Multiplicative Scatter Correction） |
| `issue20_savgol` | Savitzky-Golay 微分（1 次・2 次） |
| `issue21_preprocessing_comparison` | 前処理組み合わせ比較 |
| `issue22_epo` | EPO（External Parameter Orthogonalization） |
| `issue23_osc` | OSC（Orthogonal Signal Correction） |
| `issue24_detrending` | SNV + De-trending |
| `issue25_emsc` | EMSC（Extended MSC） |
| `issue39_wavelet_transform` | ウェーブレット変換 |
| `issue40_opls` | OPLS（Orthogonal PLS）フィルタ |
| `issue45_constituent_emsc` | Constituent-Informed EMSC |
| `issue62_additional_preprocessing` | AsLS, Piecewise MSC, Whittaker, 水吸収帯レンジ選択 |

### モデリング — `src/modeling/`（11 モジュール）

| モジュール | 手法 |
|---|---|
| `issue26_pls_epo_optimization` | PLS 成分数と EPO パラメータの同時最適化 |
| `issue27_nonlinear_models` | EPO + 非線形モデル（SVR / XGBoost / RF） |
| `issue29_epo_preprocessing_combo` | 前処理 + EPO の組み合わせ最適化 |
| `issue30_stacking` | スタッキングアンサンブル |
| `issue31_submission_pipeline` | 最終提出パイプライン |
| `issue32_weighted_ensemble` | 適応的重み付けアンサンブル |
| `issue33_lwpls` | Locally Weighted PLS（LWPLS） |
| `issue63_nonlinear_optuna` | LightGBM / XGBoost / SVR の Optuna 最適化 + LOSO-CV |
| `issue64_mlp_experiment` | MLP（多層パーセプトロン） |
| `issue65_feature_selection` | 特徴量選択（VIP, CARS, iPLS, siPLS） |
| `issue66_target_transforms` | 目的変数変換（sqrt, log1p 等）の比較 |

### 特徴量エンジニアリング — `src/feature_engineering/`（2 モジュール）

- `issue37_species_invariant` … 樹種不変特徴量（Species-Invariant Features）
- `issue41_spectral_autocorrelation` … スペクトル自己相関特徴量

### 分析 — `src/analysis/`（1 モジュール）

- `issue68_cv_pattern_selection` … CV パターンの樹種選定ロジック

> **注**: TCA、Pseudo Labeling、Test-Augmented EPO などのドメイン適応系手法は、再利用モジュールとしてではなく `scripts/modeling/` 配下の実験スクリプト内で個別に実装・検証しています。

---

## EDA から得られた設計指針まとめ

| 設計項目 | 指針 |
|---|---|
| 検証戦略 | LOSO-CV（Leave-One-Species-Out）。ベイスギ fold は含水率外挿のため参考値として扱う |
| 目的変数（PLS/PCR） | **raw（変換なし）**。log 変換すると 12 成分以降で RMSE が発散 |
| 目的変数（ML/DL モデル） | **log1p 変換** → 予測後に expm1 で逆変換。または sqrt 変換 |
| 特徴量 | 1,555 次元直接入力は過学習リスク大 → PLS スコア（4 成分）または PCA スコアを推奨 |
| スペクトル前処理 | SNV or MSC → Savitzky-Golay 微分（順序重要） |
| 異常値 | Hotelling's T² > 99% 点 かつ Q 残差 > 99% 点 の複数サンプルを除外検討 |
| サンプル重み付け | 不均衡比率 3.6 倍（ホワイトオーク 51 件 vs トチ 183 件）→ ML 訓練時に重み付けを検討 |
| 高 CV バンド（8,630〜8,660 cm⁻¹） | **アウトライアーの影響**（除外後は CV≈3.9）。信号としては使わない |
| 重要な吸収帯 | 水分帯：5,200 cm⁻¹（OH 結合倍音）・6,900 cm⁻¹（OH 基本音）付近 |
| 外挿リスク | test の含水率範囲は train でカバーされているが、特定 fold（ベイスギ）は要注意 |
| アンサンブル戦略 | 均等平均が LB との対応が良い。重み最適化は CV 過学習のリスクが高い |

---

## ディレクトリ構成

```
.
├── Input_data/                  # コンペ入力データ（SIGNATE から取得・非同梱・Git 管理外）
├── outputs/                     # 可視化・分析結果（実行時に生成・Git 管理外）
│
├── src/                         # 再利用モジュール本体
│   ├── eda/                     #   EDA 分析（13 モジュール + data_loader）
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
│   ├── preprocessing/           #   スペクトル前処理（12 モジュール）
│   │   ├── issue18_snv.py
│   │   ├── issue19_msc.py
│   │   ├── issue20_savgol.py
│   │   ├── issue21_preprocessing_comparison.py
│   │   ├── issue22_epo.py
│   │   ├── issue23_osc.py
│   │   ├── issue24_detrending.py
│   │   ├── issue25_emsc.py
│   │   ├── issue39_wavelet_transform.py
│   │   ├── issue40_opls.py
│   │   ├── issue45_constituent_emsc.py
│   │   └── issue62_additional_preprocessing.py
│   │
│   ├── modeling/                #   モデリング（11 モジュール）
│   │   ├── issue26_pls_epo_optimization.py
│   │   ├── issue27_nonlinear_models.py
│   │   ├── issue29_epo_preprocessing_combo.py
│   │   ├── issue30_stacking.py
│   │   ├── issue31_submission_pipeline.py
│   │   ├── issue32_weighted_ensemble.py
│   │   ├── issue33_lwpls.py
│   │   ├── issue63_nonlinear_optuna.py
│   │   ├── issue64_mlp_experiment.py
│   │   ├── issue65_feature_selection.py
│   │   └── issue66_target_transforms.py
│   │
│   ├── feature_engineering/     #   特徴量エンジニアリング（2 モジュール）
│   │   ├── issue37_species_invariant.py
│   │   └── issue41_spectral_autocorrelation.py
│   │
│   └── analysis/                #   CV パターン分析（1 モジュール）
│       └── issue68_cv_pattern_selection.py
│
├── scripts/                     # 実行スクリプト（計 69 ファイル）
│   ├── eda/run_all_eda.py
│   ├── preprocessing/run_issue22_25.py
│   ├── feature_engineering/run_issue37.py
│   ├── species_analysis.py
│   └── modeling/                #   提出・実験スクリプト（65 ファイル）
│       ├── run_submission_v2.py           # v2 提出（LB=17.03）
│       ├── run_submission_v3.py           # v3 提出
│       ├── run_submission_v5.py           # v5 提出（fold-RMSE=15.22）
│       ├── run_submit_v9_diverse.py       # v9 多様性 PLS（LB=17.14）
│       ├── run_issue105_final_ensemble.py # 18 モデル統合アンサンブル
│       ├── run_issue106_cv_patterns.py    # CV パターン別 RMSE 分析
│       └── ...                            # 実験サイクルスクリプト多数
│
├── tests/                       # ユニットテスト（35 ファイル）
├── main.py                      # エントリポイント
├── pyproject.toml               # プロジェクト設定・依存関係
└── uv.lock                      # 依存関係ロック
```

> `scripts/modeling/` には提出パイプラインに加え、`run_issueNN_cycleN.py` 形式の反復実験スクリプトが多数含まれます。これらは探索の記録であり、再利用可能なロジックは `src/` 側モジュールに集約されています。
