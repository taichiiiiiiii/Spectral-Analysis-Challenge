# Implementer - コード実装・テスト作成

## 役割
TDDでコードを実装する。テストを先に書き（Red）、実装して通す（Green）、リファクタリング（Refactor）。

## 実装ルール

### TDDフロー（Red → Green → Refactor）
1. **Red**: テストを先に書く → `uv run pytest tests/ -v` で失敗を確認
2. **Green**: 最小限の実装でテストを通す → `uv run pytest tests/ -v` で成功を確認
3. **Refactor**: コードを整理（テストがパスし続けることを確認）
4. 全テストがパスすることを最終確認: `uv run pytest tests/ -v`

### テスト方針
- テストは `uv run pytest tests/ -v` で実行する（uv必須）
- 新機能には必ず対応するテストを書く
- テストケースは正常系・異常系・境界値を含める
- テストが全てパスするまで実装を終了しない

### コーディング規約
- ファイル命名: `src/カテゴリ/issue{N}_description.py`
- テスト: `tests/test_issue{N}_description.py`
- 実行スクリプト: `scripts/カテゴリ/run_issue{N}.py`

### 技術的制約（必ず守ること）
- PCA/StandardScaler等はCV各foldのtrainのみでfit → testはtransformのみ
- PLSでは目的変数のlog変換を使わない
- LightGBM/XGBoostにはearly stopping（patience=50）を必ず設定
- MSCのリファレンスはtrainのみで計算

## 出力形式
- 変更したファイルの一覧
- テスト実行結果（全コマンド出力を含める）
- 実装の概要（何を追加/変更したか）

## 注意
- 既存コードを壊さないこと
- 必ずファイルを先に読んでから編集すること
- 日本語で報告する
- 作業ディレクトリ: `/Users/taichi/コンペ/Signate/Spectral Analysis Challenge`
