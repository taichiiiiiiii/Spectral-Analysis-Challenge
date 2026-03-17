"""Weighted Ensemble（適応的重み付けアンサンブル）

対応Issue: #32
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/32
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import LeaveOneGroupOut


def optimize_weights(oof_matrix: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OOF予測値に対してRMSEを最小化する重みを求める。

    制約: 重みの合計=1, 各重み>=0

    Parameters
    ----------
    oof_matrix : np.ndarray of shape (n_samples, n_models)
    y : np.ndarray of shape (n_samples,)

    Returns
    -------
    np.ndarray of shape (n_models,)
    """
    n_models = oof_matrix.shape[1]
    init_weights = np.ones(n_models) / n_models

    def objective(w):
        pred = oof_matrix @ w
        return np.sqrt(np.mean((pred - y) ** 2))

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0, 1)] * n_models

    result = minimize(
        objective, init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result.x


def evaluate_weighted_ensemble(
    oof_df: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
) -> pd.DataFrame:
    """Weighted Ensembleの各種手法をLOSO-CVで評価する。

    Parameters
    ----------
    oof_df : pd.DataFrame, OOF予測値
    y : np.ndarray, 真値
    groups : np.ndarray, グループラベル

    Returns
    -------
    pd.DataFrame with columns: method, rmse, rmse_std
    """
    logo = LeaveOneGroupOut()
    folds = list(logo.split(oof_df.values, y, groups))
    oof_matrix = oof_df.values
    rows = []

    # SimpleAvg baseline
    fold_rmses = []
    for train_idx, test_idx in folds:
        pred = oof_matrix[test_idx].mean(axis=1)
        fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
    rows.append({"method": "SimpleAvg", "rmse": np.mean(fold_rmses), "rmse_std": np.std(fold_rmses)})

    # Optimized weights (nested CV)
    fold_rmses = []
    for train_idx, test_idx in folds:
        weights = optimize_weights(oof_matrix[train_idx], y[train_idx])
        pred = oof_matrix[test_idx] @ weights
        fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
    rows.append({"method": "WeightedAvg(opt)", "rmse": np.mean(fold_rmses), "rmse_std": np.std(fold_rmses)})

    # Inverse-variance weighting
    fold_rmses = []
    for train_idx, test_idx in folds:
        variances = np.array([
            np.mean((oof_matrix[train_idx, i] - y[train_idx]) ** 2)
            for i in range(oof_matrix.shape[1])
        ])
        inv_var = 1.0 / (variances + 1e-10)
        weights = inv_var / inv_var.sum()
        pred = oof_matrix[test_idx] @ weights
        fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
    rows.append({"method": "WeightedAvg(inv-var)", "rmse": np.mean(fold_rmses), "rmse_std": np.std(fold_rmses)})

    # ElasticNet meta-learner
    fold_rmses = []
    for train_idx, test_idx in folds:
        meta = ElasticNet(alpha=0.1, l1_ratio=0.5, positive=True, max_iter=5000)
        meta.fit(oof_matrix[train_idx], y[train_idx])
        pred = meta.predict(oof_matrix[test_idx])
        fold_rmses.append(float(np.sqrt(np.mean((pred - y[test_idx]) ** 2))))
    rows.append({"method": "Stacking(ElasticNet)", "rmse": np.mean(fold_rmses), "rmse_std": np.std(fold_rmses)})

    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
