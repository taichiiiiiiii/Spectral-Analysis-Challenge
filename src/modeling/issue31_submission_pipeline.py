"""最終提出パイプライン

対応Issue: #31
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/31

SimpleAvgアンサンブル（EPO+PLS, Raw+PLS, EPO+SVR, SNV+PLS）で予測。
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo


def predict_ensemble(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    spectral_cols: list[str],
    n_pls_components: int = 4,
    n_epo: int = 1,
) -> np.ndarray:
    """SimpleAvgアンサンブルで予測を生成する。

    Parameters
    ----------
    train_df : pd.DataFrame, 学習データ
    test_df : pd.DataFrame, テストデータ
    spectral_cols : list[str], スペクトル列名
    n_pls_components : int, PLS成分数
    n_epo : int, EPO成分数

    Returns
    -------
    np.ndarray of shape (n_test,), クリッピング済み予測値
    """
    X_train = train_df[spectral_cols].values
    X_test = test_df[spectral_cols].values
    y_train = train_df["含水率"].values
    groups_train = train_df["樹種"].values

    predictions = []

    # Model 1: EPO + PLS
    P = compute_epo_projection(X_train, groups_train, n_components=n_epo)
    X_tr_epo = apply_epo(X_train, P)
    X_te_epo = apply_epo(X_test, P)
    pls1 = PLSRegression(n_components=n_pls_components)
    pls1.fit(X_tr_epo, y_train)
    predictions.append(pls1.predict(X_te_epo).ravel())

    # Model 2: Raw + PLS
    pls2 = PLSRegression(n_components=n_pls_components)
    pls2.fit(X_train, y_train)
    predictions.append(pls2.predict(X_test).ravel())

    # Model 3: EPO + PCA + SVR
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_epo)
    X_te_s = scaler.transform(X_te_epo)
    pca = PCA(n_components=10)
    X_tr_pca = pca.fit_transform(X_tr_s)
    X_te_pca = pca.transform(X_te_s)
    svr = SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1)
    svr.fit(X_tr_pca, y_train)
    predictions.append(svr.predict(X_te_pca))

    # Model 4: SNV + PLS
    X_tr_snv = apply_snv(X_train)
    X_te_snv = apply_snv(X_test)
    pls3 = PLSRegression(n_components=n_pls_components)
    pls3.fit(X_tr_snv, y_train)
    predictions.append(pls3.predict(X_te_snv).ravel())

    # Simple average
    avg_pred = np.mean(predictions, axis=0)

    # Clip to physical range (moisture content: 0-200%)
    avg_pred = np.clip(avg_pred, 0, 200)

    return avg_pred


def generate_submission(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    spectral_cols: list[str],
    output_path: Path,
    n_pls_components: int = 4,
    n_epo: int = 1,
) -> None:
    """提出ファイルを生成する。

    Parameters
    ----------
    train_df : pd.DataFrame
    test_df : pd.DataFrame
    spectral_cols : list[str]
    output_path : Path
    n_pls_components : int
    n_epo : int
    """
    preds = predict_ensemble(train_df, test_df, spectral_cols, n_pls_components, n_epo)

    submission = pd.DataFrame({
        "sample_number": test_df["sample number"].values,
        "prediction": preds,
    })
    submission.to_csv(output_path, index=False, header=False)
