"""Issue #105: 既存提出ファイルのブレンドによる最終提出生成

既存のサブ実験で生成された提出ファイルを読み込み、
様々なブレンド戦略で最終提出ファイルを生成する。

存在する提出ファイルのみ使用。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "Input_data"


def load_submission(path):
    """提出ファイルを読み込み (ヘッダーなし、2列: id, pred)"""
    df = pd.read_csv(path, header=None, names=["id", "pred"])
    return df


def save_submission(ids, preds, path):
    """提出ファイルを保存 (ヘッダーなし、2列)"""
    df = pd.DataFrame({"id": ids, "pred": preds})
    df.to_csv(path, index=False, header=False)
    print(f"  保存: {path.name} (mean={preds.mean():.2f}, std={preds.std():.2f})")


def main():
    print("=" * 70)
    print("Issue #105: 提出ファイルブレンド")
    print("=" * 70)

    # ============================================================
    # Phase 1: 提出ファイルの読み込み
    # ============================================================
    candidates = {
        "best11": OUT_DIR / "submission_best11_rmse13.70.csv",
        "lgbm_combo": OUT_DIR / "submission_v12_lgbm_best11_plus_top3.csv",
        "lwpls": OUT_DIR / "submission_v12_lwpls_best.csv",
        "new_combos": OUT_DIR / "submission_v12_new_combos_best11_plus_top5.csv",
        "stacking": OUT_DIR / "submission_v12_stacking_ridge10.csv",
    }

    submissions = {}
    for name, path in candidates.items():
        if path.exists():
            df = load_submission(path)
            submissions[name] = df
            print(f"  [OK] {name}: {path.name} ({len(df)} rows, "
                  f"mean={df['pred'].mean():.2f}, std={df['pred'].std():.2f})")
        else:
            print(f"  [SKIP] {name}: {path.name} (ファイルなし)")

    if len(submissions) < 2:
        print("\n提出ファイルが2つ未満のため、ブレンド不可。")
        print("代替: 既存の全提出ファイルからブレンドを試みます。")
        # 既存の主要な提出ファイルを代替として使用
        alt_candidates = {
            "best11": OUT_DIR / "submission_best11_rmse13.70.csv",
            "v11_combo": OUT_DIR / "submission_v11_blend_best_combo.csv",
            "v11_multi": OUT_DIR / "submission_v11_blend_multi.csv",
            "v10_mega": OUT_DIR / "submission_v10_mega_blend.csv",
            "v10_v6_diverse": OUT_DIR / "submission_v10_v6_diverse.csv",
            "v9_top12": OUT_DIR / "submission_v9_top12.csv",
            "v8_blend": OUT_DIR / "submission_v8_blend.csv",
        }
        submissions = {}
        for name, path in alt_candidates.items():
            if path.exists():
                df = load_submission(path)
                submissions[name] = df
                print(f"  [ALT] {name}: {path.name} ({len(df)} rows, "
                      f"mean={df['pred'].mean():.2f}, std={df['pred'].std():.2f})")

    if len(submissions) < 2:
        print("ブレンド不可。終了。")
        return

    # IDs確認
    ref_ids = list(submissions.values())[0]["id"].values
    for name, df in submissions.items():
        assert np.array_equal(df["id"].values, ref_ids), f"{name}のIDが一致しません"

    names = list(submissions.keys())
    preds = {name: df["pred"].values for name, df in submissions.items()}

    print(f"\n利用可能な提出: {len(submissions)}個")
    print(f"  {', '.join(names)}")

    # ============================================================
    # Phase 2: ブレンド戦略
    # ============================================================
    results = []

    # --- 2a. 全提出の均等平均 ---
    all_preds = np.array([preds[n] for n in names])
    equal_avg = np.mean(all_preds, axis=0)
    equal_avg = np.clip(equal_avg, 0, 300)
    results.append(("equal_avg", equal_avg, "全提出均等平均"))

    # --- 2b. LightGBM重視の重み付き平均 ---
    if "lgbm_combo" in preds:
        # lgbm_combo最重視
        weight_map = {
            "lgbm_combo": 0.40,
            "best11": 0.30,
            "stacking": 0.15,
            "new_combos": 0.15,
        }
        available_weights = {k: v for k, v in weight_map.items() if k in preds}
        if len(available_weights) > 1:
            total_w = sum(available_weights.values())
            weighted = sum(preds[k] * (w / total_w)
                           for k, w in available_weights.items())
            weighted = np.clip(weighted, 0, 300)
            results.append(("lgbm_heavy", weighted, "LightGBM重視"))

    # --- 2c. 2-way ブレンド (best11 と各提出) ---
    for alpha in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        for other_name in names:
            if other_name == "best11":
                continue
            if "best11" not in preds:
                continue
            blend = (1 - alpha) * preds["best11"] + alpha * preds[other_name]
            blend = np.clip(blend, 0, 300)
            label = f"best11+{other_name}_a{alpha:.2f}"
            results.append((label, blend, f"best11 + {other_name} (alpha={alpha})"))

    # --- 2d. 3-way ブレンド ---
    if len(names) >= 3:
        for alpha in [0.10, 0.15, 0.20]:
            others = [n for n in names if n != "best11"]
            if "best11" in preds and len(others) >= 2:
                other_mean = np.mean([preds[n] for n in others], axis=0)
                blend = (1 - alpha) * preds["best11"] + alpha * other_mean
                blend = np.clip(blend, 0, 300)
                label = f"3way_a{alpha:.2f}"
                results.append((label, blend, f"3-way: best11 + others均等 (alpha={alpha})"))

    # --- 2e. 全提出のメディアン ---
    median_pred = np.median(all_preds, axis=0)
    median_pred = np.clip(median_pred, 0, 300)
    results.append(("median", median_pred, "全提出メディアン"))

    # --- 2f. トリム平均 (上下1個除外) ---
    if len(names) >= 4:
        sorted_preds = np.sort(all_preds, axis=0)
        trim_avg = np.mean(sorted_preds[1:-1], axis=0)
        trim_avg = np.clip(trim_avg, 0, 300)
        results.append(("trim_avg", trim_avg, "トリム平均"))

    # ============================================================
    # Phase 3: 結果表示と保存
    # ============================================================
    print(f"\n--- ブレンド結果一覧 ({len(results)}パターン) ---\n")
    for label, pred, desc in results:
        print(f"  {label}: mean={pred.mean():.2f}, std={pred.std():.2f} | {desc}")

    # 提出ファイル生成
    print(f"\n--- 提出ファイル生成 ---\n")

    # 1. 均等平均
    save_submission(ref_ids, results[0][1],
                    OUT_DIR / "submission_v12_final_blend_equal.csv")

    # 2. LightGBM重視 (あれば)
    for label, pred, desc in results:
        if label == "lgbm_heavy":
            save_submission(ref_ids, pred,
                            OUT_DIR / "submission_v12_final_blend_lgbm_heavy.csv")
            break

    # 3. 最も有望な設定 (best11ベースのa=0.10ブレンド or 均等平均)
    # best11 + 最良other のalpha=0.10を最有望とする
    best_candidate = results[0]  # デフォルト: 均等平均
    for label, pred, desc in results:
        if "lgbm_heavy" in label:
            best_candidate = (label, pred, desc)
            break
        if "lgbm_combo_a0.10" in label:
            best_candidate = (label, pred, desc)
            break

    save_submission(ref_ids, best_candidate[1],
                    OUT_DIR / "submission_v12_final_best.csv")
    print(f"  最有望: {best_candidate[0]} ({best_candidate[2]})")

    # 全結果CSV
    rows = []
    for label, pred, desc in results:
        rows.append({
            "label": label,
            "mean": pred.mean(),
            "std": pred.std(),
            "min": pred.min(),
            "max": pred.max(),
            "description": desc,
        })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_DIR / "modeling" / "issue105_blend_results.csv", index=False)
    print(f"\n結果CSV: outputs/modeling/issue105_blend_results.csv")

    print(f"\n{'='*70}")
    print(f"完了: {len(results)}パターンのブレンドを生成")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
