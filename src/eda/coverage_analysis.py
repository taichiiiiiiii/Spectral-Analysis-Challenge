"""含水率カバレッジ分析モジュール（Issue #12）"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def compute_pc_coverage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    spectral_cols: list[str],
    n_components: int = 5,
) -> dict:
    X_train = train_df[spectral_cols].values
    X_test = test_df[spectral_cols].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=n_components)
    train_scores = pca.fit_transform(X_train_scaled)
    test_scores = pca.transform(X_test_scaled)

    # testがtrainのPC空間の範囲外にあるサンプルの割合
    train_min = train_scores.min(axis=0)
    train_max = train_scores.max(axis=0)
    outside = np.any(
        (test_scores < train_min) | (test_scores > train_max), axis=1
    )
    outside_ratio = float(outside.mean())

    return {
        "train_scores": train_scores,
        "test_scores": test_scores,
        "test_outside_ratio": outside_ratio,
        "n_components": n_components,
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }


def assess_extrapolation_risk(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    spectral_cols: list[str],
) -> dict:
    coverage = compute_pc_coverage(train_df, test_df, spectral_cols)
    outside_ratio = coverage["test_outside_ratio"]

    if outside_ratio < 0.1:
        risk_level = "low"
        recommendation = "trainのカバレッジ内に収まっており、外挿リスクは低い。通常のモデル構築で問題ない。"
    elif outside_ratio < 0.3:
        risk_level = "medium"
        recommendation = "一部のtestサンプルがtrain範囲外。データ拡張や正則化の強化を検討。"
    else:
        risk_level = "high"
        recommendation = "多くのtestサンプルがtrain範囲外。ドメイン適応手法の検討が必要。"

    return {
        "risk_level": risk_level,
        "outside_ratio": outside_ratio,
        "recommendation": recommendation,
    }


def compute_moisture_range_coverage(
    df: pd.DataFrame,
    low_threshold: float = 10.0,
    high_threshold: float = 150.0,
) -> dict:
    moisture = df["含水率"]
    low_count = int((moisture < low_threshold).sum())
    high_count = int((moisture > high_threshold).sum())

    return {
        "low_moisture_count": low_count,
        "high_moisture_count": high_count,
        "low_moisture_ratio": low_count / len(df),
        "high_moisture_ratio": high_count / len(df),
    }
