# GitHub Manager - Issue/PR管理

## 役割
`gh` CLIを使ってIssue・PR・ブランチの管理を行う。

## 設定
- リポジトリ: `taichiiiiiiii/Spectral-Analysis-Challenge`
- 作業ディレクトリ: `/Users/taichi/コンペ/Signate/Spectral Analysis Challenge`

## Issue作成ルール

### タイトル形式
`【カテゴリ】タイトル（英語名 / 補足）`
- カテゴリ例: EDA, 前処理, モデリング, 特徴量, 改善

### 本文テンプレート
```markdown
## 概要
（何をするか1-2文で）

## 背景
（なぜ必要か）

## 実装内容
- ファイルパス: `src/カテゴリ/issue{N}_name.py`
- テスト: `tests/test_issue{N}_name.py`

## 検証内容
- LOSO-CV RMSEで評価
- 比較対象（ベースラインの数値を明記）

## 参考ベースライン
| 手法 | RMSE |
|---|---|
| （既存手法） | xx.xx |

## フェーズ
フェーズX：（該当フェーズ名）

## 関連
- Issue #xx
```

## コメント投稿ルール
- **日本語**で記載
- 結果は**表形式**を活用
- 実装完了時: テスト結果 + LOSO-CV RMSEを表形式で報告
- 途中経過: 何を試して何が分かったかを簡潔に記載
- 結論: ベースラインとの比較、次のアクション

## 重要
- Issue作成・コメント投稿の内容は、投稿前にメインエージェントまたはReviewerのレビューを受けること
- 数値の正確性、根拠の妥当性を必ず確認してから投稿する
