"""樹種間のスペクトル類似度分析スクリプト"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# データ読み込み
train = pd.read_csv("Input_data/train.csv", encoding="shift_jis")
test = pd.read_csv("Input_data/test.csv", encoding="shift_jis")

# カラム特定
meta_cols_train = ["sample number", "species number", "樹種", "含水率"]
meta_cols_test = ["sample number", "species number", "樹種"]
spec_cols = [c for c in train.columns if c not in meta_cols_train]

print("=" * 70)
print("1. 樹種一覧とサンプル数")
print("=" * 70)

print("\n【トレーニングデータ】")
train_counts = train["樹種"].value_counts().sort_index()
for sp, cnt in train_counts.items():
    print(f"  {sp}: {cnt} サンプル")
print(f"  合計: {len(train)} サンプル, {len(train_counts)} 樹種")

print("\n【テストデータ】")
test_counts = test["樹種"].value_counts().sort_index()
for sp, cnt in test_counts.items():
    print(f"  {sp}: {cnt} サンプル")
print(f"  合計: {len(test)} サンプル, {len(test_counts)} 樹種")

# 樹種別平均スペクトル計算
train_mean = train.groupby("樹種")[spec_cols].mean()
test_mean = test.groupby("樹種")[spec_cols].mean()

# コサイン類似度
cos_sim = cosine_similarity(test_mean.values, train_mean.values)
cos_sim_df = pd.DataFrame(cos_sim, index=test_mean.index, columns=train_mean.index)

# ユークリッド距離
euc_dist = euclidean_distances(test_mean.values, train_mean.values)
euc_dist_df = pd.DataFrame(euc_dist, index=test_mean.index, columns=train_mean.index)

print("\n" + "=" * 70)
print("2. 含水率の統計（トレーニングデータ）")
print("=" * 70)
mc_stats = train.groupby("樹種")["含水率"].agg(["mean", "std", "min", "max", "count"])
mc_stats.columns = ["平均", "標準偏差", "最小", "最大", "サンプル数"]
print(f"\n{'樹種':<12} {'平均':>8} {'標準偏差':>8} {'最小':>8} {'最大':>8} {'N':>4}")
print("-" * 52)
for sp, row in mc_stats.iterrows():
    print(f"{sp:<12} {row['平均']:>8.2f} {row['標準偏差']:>8.2f} {row['最小']:>8.2f} {row['最大']:>8.2f} {int(row['サンプル数']):>4}")

print(f"\n全体: 平均={train['含水率'].mean():.2f}, 標準偏差={train['含水率'].std():.2f}")

print("\n" + "=" * 70)
print("3. テスト樹種ごとの類似度ランキング（上位3位）")
print("=" * 70)

for test_sp in test_mean.index:
    print(f"\n■ テスト樹種: {test_sp}")

    # コサイン類似度（降順）
    cos_rank = cos_sim_df.loc[test_sp].sort_values(ascending=False)
    # ユークリッド距離（昇順）
    euc_rank = euc_dist_df.loc[test_sp].sort_values(ascending=True)

    print(f"  {'順位':<4} {'コサイン類似度':>30}  {'ユークリッド距離':>30}")
    print(f"  {'':<4} {'樹種':>14} {'類似度':>14}  {'樹種':>14} {'距離':>14}")
    print("  " + "-" * 68)
    for i in range(3):
        cos_sp = cos_rank.index[i]
        cos_val = cos_rank.iloc[i]
        euc_sp = euc_rank.index[i]
        euc_val = euc_rank.iloc[i]
        print(f"  {i+1:<4} {cos_sp:>14} {cos_val:>14.6f}  {euc_sp:>14} {euc_val:>14.4f}")

print("\n" + "=" * 70)
print("4. コサイン類似度の全体マトリックス（テスト×トレイン）")
print("=" * 70)
print()
# 見やすく表示
header = f"{'':>12}" + "".join(f"{c:>10}" for c in cos_sim_df.columns)
print(header)
for idx, row in cos_sim_df.iterrows():
    vals = "".join(f"{v:>10.4f}" for v in row)
    print(f"{idx:>12}{vals}")
