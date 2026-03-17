"""Test-Augmented EPO

対応Issue: #34
https://github.com/taichiiiiiiii/Spectral-Analysis-Challenge/issues/34

通常のEPOは訓練データの樹種間変動のみを除去するが、
テストデータの樹種情報が未知のため、テストデータ固有の変動方向が残る。

Test-Augmented EPO:
テストデータの平均をグループ平均に追加してEPO投影を計算することで、
テストドメインの変動方向も除去する。
"""
import numpy as np


def compute_augmented_epo_projection(
    X_train: np.ndarray,
    groups: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 1,
) -> np.ndarray:
    """Test-Augmented EPO投影行列を計算する。

    訓練データの樹種平均にテストデータの全体平均を追加して
    グループ間変動を計算し、EPO投影を行う。

    Parameters
    ----------
    X_train : np.ndarray of shape (n_train, n_features)
    groups : np.ndarray of shape (n_train,), 樹種ラベル
    X_test : np.ndarray of shape (n_test, n_features)
    n_components : int, 除去する成分数

    Returns
    -------
    P : np.ndarray of shape (n_features, n_features), 対称投影行列
    """
    # 訓練データの樹種平均
    unique_groups = np.unique(groups)
    group_means = np.array([X_train[groups == g].mean(axis=0) for g in unique_groups])

    # テストデータの全体平均を追加
    test_mean = X_test.mean(axis=0, keepdims=True)
    augmented_means = np.vstack([group_means, test_mean])

    # 中心化してSVD
    augmented_centered = augmented_means - augmented_means.mean(axis=0)
    U, S, Vt = np.linalg.svd(augmented_centered, full_matrices=False)

    # 除去する成分数を制限
    n_comp = min(n_components, augmented_means.shape[0] - 1, Vt.shape[0])
    D = Vt[:n_comp].T  # (n_features, n_comp)

    # 投影行列: P = I - D @ D^T
    P = np.eye(X_train.shape[1]) - D @ D.T

    return P


def apply_epo(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    """EPO投影を適用する。"""
    return X @ P
