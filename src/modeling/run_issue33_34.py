"""Issue #33-34: LWPLS + Test-Augmented EPO 評価"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVR
from sklearn.decomposition import PCA

from src.eda.data_loader import load_train, load_test, get_spectral_columns, get_wavenumbers
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.modeling.issue33_lwpls import lwpls_predict
from src.modeling.issue34_test_augmented_epo import compute_augmented_epo_projection

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # データ読み込み
    df_train = load_train(DATA_DIR)
    df_test = load_test(DATA_DIR)
    spectral_cols = get_spectral_columns(df_train)
    wn = get_wavenumbers(spectral_cols)

    X_raw = df_train[spectral_cols].values
    y = df_train["含水率"].values
    groups = df_train["樹種"].values
    X_test = df_test[spectral_cols].values

    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_raw, y, groups))
    results = []

    # === Issue #34: Test-Augmented EPO ===
    print("=== Issue #34: Test-Augmented EPO ===\n")

    for n_epo in [1, 2, 3]:
        fold_rmses = []
        for train_idx, test_idx in folds:
            # テスト樹種を「未知テスト」と見なし、Test-Augmented EPOを計算
            P = compute_augmented_epo_projection(
                X_raw[train_idx], groups[train_idx],
                X_raw[test_idx],  # LOSO-CVではhold-outをテストデータとして使用
                n_components=n_epo,
            )
            X_tr = X_raw[train_idx] @ P
            X_te = X_raw[test_idx] @ P
            pls = PLSRegression(n_components=4)
            pls.fit(X_tr, y[train_idx])
            pred = pls.predict(X_te).ravel()
            fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))

        rmse, std = np.mean(fold_rmses), np.std(fold_rmses)
        results.append({"method": f"TestAugEPO({n_epo})+PLS(4)", "rmse": rmse, "rmse_std": std})
        print(f"TestAugEPO({n_epo})+PLS(4): RMSE={rmse:.2f} ± {std:.2f}")

    # 通常EPOベースライン
    fold_rmses = []
    for train_idx, test_idx in folds:
        P = compute_epo_projection(X_raw[train_idx], groups[train_idx], n_components=1)
        X_tr = X_raw[train_idx] @ P
        X_te = X_raw[test_idx] @ P
        pls = PLSRegression(n_components=4)
        pls.fit(X_tr, y[train_idx])
        pred = pls.predict(X_te).ravel()
        fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
    rmse_epo, std_epo = np.mean(fold_rmses), np.std(fold_rmses)
    results.append({"method": "EPO(1)+PLS(4) [baseline]", "rmse": rmse_epo, "rmse_std": std_epo})
    print(f"\nEPO(1)+PLS(4) [baseline]: RMSE={rmse_epo:.2f} ± {std_epo:.2f}")

    # === Issue #33: LWPLS ===
    print("\n=== Issue #33: LWPLS ===\n")

    # PCA次元削減してからLWPLS（計算量削減）
    for n_pca in [10, 20, 30]:
        for k in [30, 50, 80]:
            for n_comp in [2, 3, 4]:
                fold_rmses = []
                for train_idx, test_idx in folds:
                    # EPO適用
                    P = compute_epo_projection(X_raw[train_idx], groups[train_idx], n_components=1)
                    X_tr = X_raw[train_idx] @ P
                    X_te = X_raw[test_idx] @ P

                    # PCA次元削減
                    pca = PCA(n_components=n_pca)
                    X_tr_pca = pca.fit_transform(X_tr)
                    X_te_pca = pca.transform(X_te)

                    pred = lwpls_predict(X_tr_pca, y[train_idx], X_te_pca,
                                        n_components=n_comp, k=k)
                    fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))

                rmse, std = np.mean(fold_rmses), np.std(fold_rmses)
                name = f"EPO+PCA({n_pca})+LWPLS(k={k},nc={n_comp})"
                results.append({"method": name, "rmse": rmse, "rmse_std": std})
                print(f"  {name}: RMSE={rmse:.2f} ± {std:.2f}")

    # 結果まとめ
    result_df = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    print("\n=== 全結果（上位10） ===")
    print(result_df.head(10).to_string(index=False))

    # プロット（上位15）
    top15 = result_df.head(15)
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ["green" if "baseline" in m else "steelblue" for m in top15["method"]]
    ax.barh(range(len(top15)), top15["rmse"], color=colors)
    ax.errorbar(top15["rmse"], range(len(top15)), xerr=top15["rmse_std"],
                fmt="none", color="black", capsize=3)
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(top15["method"], fontsize=8)
    ax.set_xlabel("RMSE (LOSO-CV)")
    ax.set_title("Issue #33-34: LWPLS + Test-Augmented EPO")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "issue33_34_lwpls_testaug_epo.png", dpi=150)
    plt.close()
    print(f"\nSaved: {OUT_DIR / 'issue33_34_lwpls_testaug_epo.png'}")


if __name__ == "__main__":
    main()
