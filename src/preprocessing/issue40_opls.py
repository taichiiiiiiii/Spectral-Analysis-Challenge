"""OPLS (Orthogonal Partial Least Squares) フィルタ

yに直交する系統的変動（ここでは樹種差やベースライン変動）をXから除去する前処理。
OSCの改良版で、PLS内部の直交分解に基づく。

参考: Trygg & Wold, "Orthogonal projections to latent structures (O-PLS)",
      Journal of Chemometrics 2002.
"""
import numpy as np


class OPLSFilter:
    """OPLS前処理フィルタ。

    yに直交する変動成分を除去し、yに関連する変動のみを保持する。

    Parameters
    ----------
    n_components : 除去する直交成分の数
    """

    def __init__(self, n_components: int = 1):
        self.n_components = n_components
        self.W_orth_ = None
        self.P_orth_ = None
        self.mean_X_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OPLSFilter":
        """学習データでOPLSフィルタを学習する。"""
        n, p = X.shape
        self.mean_X_ = X.mean(axis=0)
        X_c = X - self.mean_X_
        y_c = y - y.mean()

        W_orth_list = []
        P_orth_list = []

        for _ in range(self.n_components):
            # PLS weight vector: w = X^T y / ||X^T y||
            w = X_c.T @ y_c
            w = w / (np.linalg.norm(w) + 1e-12)

            # PLS score: t = Xw
            t = X_c @ w

            # PLS loading: p = X^T t / (t^T t)
            p = X_c.T @ t / (t @ t + 1e-12)

            # Orthogonal weight: w_orth = p - (w^T p) w
            w_orth = p - (w @ p) * w
            w_orth = w_orth / (np.linalg.norm(w_orth) + 1e-12)

            # Orthogonal score: t_orth = X w_orth
            t_orth = X_c @ w_orth

            # Orthogonal loading: p_orth = X^T t_orth / (t_orth^T t_orth)
            p_orth = X_c.T @ t_orth / (t_orth @ t_orth + 1e-12)

            # Remove orthogonal component from X
            X_c = X_c - np.outer(t_orth, p_orth)

            W_orth_list.append(w_orth)
            P_orth_list.append(p_orth)

        self.W_orth_ = np.column_stack(W_orth_list)
        self.P_orth_ = np.column_stack(P_orth_list)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """直交成分を除去する。"""
        X_c = X - self.mean_X_
        for j in range(self.n_components):
            w_orth = self.W_orth_[:, j]
            p_orth = self.P_orth_[:, j]
            t_orth = X_c @ w_orth
            X_c = X_c - np.outer(t_orth, p_orth)
        return X_c + self.mean_X_

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """fit + transform。"""
        self.fit(X, y)
        return self.transform(X)
