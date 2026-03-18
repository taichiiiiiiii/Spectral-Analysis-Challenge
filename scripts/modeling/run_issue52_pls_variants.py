"""Issue #52: PLS派生手法の検証（Sparse PLS / Kernel PLS / O2PLS）"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.eda.data_loader import load_train, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue38_tca import tca_transform

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# === Sparse PLS (NIPALS + L1 thresholding) ===
class SparsePLS:
    """Sparse PLS via soft-thresholding on weights."""
    def __init__(self, n_components=4, alpha=0.1, max_iter=500, tol=1e-6):
        self.n_components = n_components
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def _soft_threshold(self, w, alpha):
        return np.sign(w) * np.maximum(np.abs(w) - alpha, 0)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, p = X.shape

        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = X.std(axis=0)
        self.x_std_[self.x_std_ == 0] = 1.0
        self.y_mean_ = y.mean()

        Xc = (X - self.x_mean_) / self.x_std_
        yc = y - self.y_mean_

        self.weights_ = np.zeros((p, self.n_components))
        self.loadings_ = np.zeros((p, self.n_components))
        self.scores_ = np.zeros((n, self.n_components))
        self.y_loadings_ = np.zeros(self.n_components)
        self.coef_ = np.zeros(p)
        self.selected_features_ = []

        E = Xc.copy()
        f = yc.copy()

        for k in range(self.n_components):
            w = E.T @ f
            w = self._soft_threshold(w, self.alpha * np.max(np.abs(w)))
            w_norm = np.linalg.norm(w)
            if w_norm < 1e-10:
                break
            w = w / w_norm

            t = E @ w
            p_loading = E.T @ t / (t @ t)
            q = f @ t / (t @ t)

            E = E - np.outer(t, p_loading)
            f = f - t * q

            self.weights_[:, k] = w
            self.loadings_[:, k] = p_loading
            self.scores_[:, k] = t
            self.y_loadings_[k] = q
            self.selected_features_.append(np.where(np.abs(w) > 1e-10)[0])

        # Compute regression coefficients
        W = self.weights_[:, :k+1] if w_norm >= 1e-10 else self.weights_[:, :max(1,k)]
        P = self.loadings_[:, :W.shape[1]]
        Q = self.y_loadings_[:W.shape[1]]
        try:
            R = W @ np.linalg.inv(P.T @ W)
            self.coef_ = (R @ Q) / self.x_std_
        except np.linalg.LinAlgError:
            self.coef_ = np.zeros(p)

        self.intercept_ = self.y_mean_ - self.x_mean_ / self.x_std_ @ (self.coef_ * self.x_std_)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_ + (self.y_mean_ - self.x_mean_ @ self.coef_)

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        Xc = (X - self.x_mean_) / self.x_std_
        return Xc @ self.weights_


# === Kernel PLS ===
class KernelPLS:
    """Kernel PLS regression using RBF kernel."""
    def __init__(self, n_components=4, gamma=None):
        self.n_components = n_components
        self.gamma = gamma

    def _rbf_kernel(self, X1, X2):
        gamma = self.gamma or 1.0 / X1.shape[1]
        sq_dist = np.sum(X1**2, axis=1, keepdims=True) + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
        return np.exp(-gamma * sq_dist)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.X_train_ = X.copy()
        self.y_mean_ = y.mean()
        yc = y - self.y_mean_

        K = self._rbf_kernel(X, X)
        n = K.shape[0]
        ones = np.ones((n, n)) / n
        K_centered = K - ones @ K - K @ ones + ones @ K @ ones
        self.K_train_mean_row_ = K.mean(axis=0)
        self.K_train_mean_ = K.mean()

        T = np.zeros((n, self.n_components))
        U = np.zeros((n, self.n_components))
        self.alphas_ = np.zeros((n, self.n_components))

        Kc = K_centered.copy()
        f = yc.copy()

        for k in range(self.n_components):
            alpha = Kc @ f
            alpha_norm = np.linalg.norm(alpha)
            if alpha_norm < 1e-10:
                self.n_components = k
                break
            alpha = alpha / alpha_norm

            t = Kc @ alpha
            q = f @ t / (t @ t)

            Kc = Kc - np.outer(t, t) @ Kc / (t @ t)
            f = f - t * q

            T[:, k] = t
            self.alphas_[:, k] = alpha

        self.T_ = T[:, :self.n_components]
        # Solve for coefficients: y = T @ c + y_mean
        if self.n_components > 0:
            self.coef_ = np.linalg.lstsq(self.T_, yc, rcond=None)[0]
        else:
            self.coef_ = np.array([])
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        K = self._rbf_kernel(X, self.X_train_)
        K_centered = K - K.mean(axis=1, keepdims=True) - self.K_train_mean_row_ + self.K_train_mean_

        T_test = K_centered @ self.alphas_[:, :self.n_components]
        return T_test @ self.coef_ + self.y_mean_


# === O2PLS ===
class O2PLS:
    """Simplified O2PLS: removes orthogonal components from X before PLS."""
    def __init__(self, n_components=4, n_ortho_x=1, n_ortho_y=0):
        self.n_components = n_components
        self.n_ortho_x = n_ortho_x
        self.n_ortho_y = n_ortho_y

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        self.x_mean_ = X.mean(axis=0)
        self.y_mean_ = y.mean()
        Xc = X - self.x_mean_
        yc = y - self.y_mean_

        # Step 1: Standard PLS to get joint components
        pls_init = PLSRegression(n_components=min(self.n_components + self.n_ortho_x, X.shape[1] - 1), max_iter=500)
        pls_init.fit(Xc, yc)
        T_joint = pls_init.x_scores_

        # Step 2: Find X-orthogonal components
        # Residual after removing joint variation
        E = Xc - T_joint @ np.linalg.lstsq(T_joint, Xc, rcond=None)[0]

        if self.n_ortho_x > 0 and E.shape[1] > self.n_ortho_x:
            U, S, Vt = np.linalg.svd(E, full_matrices=False)
            self.ortho_loadings_x_ = Vt[:self.n_ortho_x].T
        else:
            self.ortho_loadings_x_ = np.zeros((X.shape[1], 0))

        # Step 3: Remove orthogonal components and refit PLS
        Xc_filtered = Xc - Xc @ self.ortho_loadings_x_ @ self.ortho_loadings_x_.T

        self.pls_ = PLSRegression(n_components=min(self.n_components, Xc_filtered.shape[1] - 1), max_iter=500)
        self.pls_.fit(Xc_filtered, yc)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        Xc = X - self.x_mean_
        Xc_filtered = Xc - Xc @ self.ortho_loadings_x_ @ self.ortho_loadings_x_.T
        return self.pls_.predict(Xc_filtered).ravel() + self.y_mean_

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        Xc = X - self.x_mean_
        Xc_filtered = Xc - Xc @ self.ortho_loadings_x_ @ self.ortho_loadings_x_.T
        return self.pls_.transform(Xc_filtered)


def loso_cv(X, y, groups, model_fn, preprocess_fn=None, target_transform=None):
    logo = LeaveOneGroupOut()
    fold_rmses = []
    for train_idx, test_idx in logo.split(X, y, groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        if preprocess_fn:
            X_tr, X_te = preprocess_fn(X_tr, X_te, y_tr, groups[train_idx])

        if target_transform == "sqrt":
            y_tr_t = np.sqrt(y_tr)
        else:
            y_tr_t = y_tr

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = model_fn()
        model.fit(X_tr_s, y_tr_t)
        pred = model.predict(X_te_s).ravel()

        if target_transform == "sqrt":
            pred = np.clip(pred, 0, None) ** 2

        fold_rmses.append(np.sqrt(mean_squared_error(y_te, pred)))

    return np.mean(fold_rmses), np.std(fold_rmses)


def preprocess_raw(X_tr, X_te, y_tr, groups):
    return X_tr, X_te

def preprocess_snv(X_tr, X_te, y_tr, groups):
    return apply_snv(X_tr), apply_snv(X_te)

def preprocess_epo(X_tr, X_te, y_tr, groups):
    P = compute_epo_projection(X_tr, groups, n_components=1)
    return apply_epo(X_tr, P), apply_epo(X_te, P)

def preprocess_tca(X_tr, X_te, y_tr, groups):
    return tca_transform(X_tr, X_te, n_components=10, gamma=0.001)


def main():
    df = load_train(DATA_DIR)
    spectral_cols = get_spectral_columns(df)
    X = df[spectral_cols].values
    y = df["含水率"].values
    groups = df["樹種"].values

    results = []

    # === 1. Sparse PLS ===
    print("=== 1. Sparse PLS ===")
    preprocessings = {
        "Raw": preprocess_raw,
        "SNV": preprocess_snv,
        "EPO": preprocess_epo,
        "TCA(10)": preprocess_tca,
    }

    for pp_name, pp_fn in preprocessings.items():
        for alpha in [0.01, 0.05, 0.1, 0.3, 0.5]:
            for n_comp in [2, 4, 6]:
                for target in [None, "sqrt"]:
                    label = f"{pp_name}+SparsePLS(n={n_comp},α={alpha})" + ("+sqrt(y)" if target else "")
                    try:
                        rmse, std = loso_cv(X, y, groups,
                                           lambda a=alpha, n=n_comp: SparsePLS(n_components=n, alpha=a),
                                           pp_fn, target)
                        results.append({"category": "SparsePLS", "method": label, "rmse": round(rmse, 2), "rmse_std": round(std, 2)})
                        print(f"  {label}: RMSE={rmse:.2f}")
                    except Exception as e:
                        print(f"  {label}: ERROR - {e}")

    # === 2. Kernel PLS ===
    print("\n=== 2. Kernel PLS ===")
    for pp_name, pp_fn in preprocessings.items():
        for gamma in [0.0001, 0.001, 0.01, None]:
            for n_comp in [2, 4, 6]:
                for target in [None, "sqrt"]:
                    gamma_str = f"γ={gamma}" if gamma else "γ=auto"
                    label = f"{pp_name}+KernelPLS(n={n_comp},{gamma_str})" + ("+sqrt(y)" if target else "")
                    try:
                        rmse, std = loso_cv(X, y, groups,
                                           lambda g=gamma, n=n_comp: KernelPLS(n_components=n, gamma=g),
                                           pp_fn, target)
                        results.append({"category": "KernelPLS", "method": label, "rmse": round(rmse, 2), "rmse_std": round(std, 2)})
                        print(f"  {label}: RMSE={rmse:.2f}")
                    except Exception as e:
                        print(f"  {label}: ERROR - {e}")

    # === 3. O2PLS ===
    print("\n=== 3. O2PLS ===")
    for pp_name, pp_fn in preprocessings.items():
        for n_ortho in [1, 2, 3]:
            for n_comp in [2, 4, 6]:
                for target in [None, "sqrt"]:
                    label = f"{pp_name}+O2PLS(n={n_comp},ortho={n_ortho})" + ("+sqrt(y)" if target else "")
                    try:
                        rmse, std = loso_cv(X, y, groups,
                                           lambda nc=n_comp, no=n_ortho: O2PLS(n_components=nc, n_ortho_x=no),
                                           pp_fn, target)
                        results.append({"category": "O2PLS", "method": label, "rmse": round(rmse, 2), "rmse_std": round(std, 2)})
                        print(f"  {label}: RMSE={rmse:.2f}")
                    except Exception as e:
                        print(f"  {label}: ERROR - {e}")

    # === Baselines ===
    print("\n=== Baselines ===")
    baselines = [
        ("Raw+PLS(4)", preprocess_raw, lambda: PLSRegression(n_components=4, max_iter=500), None),
        ("EPO+PLS(4)", preprocess_epo, lambda: PLSRegression(n_components=4, max_iter=500), None),
        ("TCA(10)+PLS(4)+sqrt(y)", preprocess_tca, lambda: PLSRegression(n_components=4, max_iter=500), "sqrt"),
    ]
    for label, pp_fn, model_fn, target in baselines:
        rmse, std = loso_cv(X, y, groups, model_fn, pp_fn, target)
        results.append({"category": "baseline", "method": label, "rmse": round(rmse, 2), "rmse_std": round(std, 2)})
        print(f"  {label}: RMSE={rmse:.2f}")

    # === Results ===
    results_df = pd.DataFrame(results).sort_values("rmse")
    print("\n=== 全結果 Top 25 ===")
    print(results_df.head(25).to_string(index=False))
    results_df.to_csv(OUT_DIR / "issue52_pls_variants.csv", index=False)

    # === Figure ===
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    for i, (cat, title) in enumerate([("SparsePLS", "Sparse PLS"), ("KernelPLS", "Kernel PLS"), ("O2PLS", "O2PLS")]):
        ax = axes[i]
        cat_df = results_df[results_df["category"] == cat].head(15)
        if len(cat_df) == 0:
            ax.set_title(f"{title} (no results)")
            continue
        colors = ['#e74c3c' if r < 21 else '#f39c12' if r < 22 else '#3498db' for r in cat_df['rmse']]
        ax.barh(range(len(cat_df)), cat_df["rmse"], color=colors)
        ax.set_yticks(range(len(cat_df)))
        ax.set_yticklabels(cat_df["method"], fontsize=6)
        ax.set_xlabel("RMSE")
        ax.set_title(title)
        ax.axvline(22.30, color='blue', linestyle='--', alpha=0.5, label='Raw+PLS(4)=22.30')
        ax.axvline(19.69, color='red', linestyle='--', alpha=0.7, label='TCA+PLS+sqrt=19.69')
        ax.axvline(18.95, color='green', linestyle='--', alpha=0.7, label='TCA+EN+sqrt=18.95')
        ax.legend(fontsize=6)
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "issue52_pls_variants.png", dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {OUT_DIR / 'issue52_pls_variants.png'}")


if __name__ == "__main__":
    main()
