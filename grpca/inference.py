"""Graph learning from data and structural-prior comparison utilities."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

try:
    from sklearn.covariance import GraphicalLassoCV
except ImportError:
    GraphicalLassoCV = None  # type: ignore[assignment,misc]

try:
    from sklearn.linear_model import LassoCV
except ImportError:
    LassoCV = None  # type: ignore[assignment,misc]


def infer_precision_graph(
    X: np.ndarray,
    *,
    random_state: int = 42,
    standardize: bool = True,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Infer a sparse conditional-dependence graph from data.

    Tries, in order:
    1. ``GraphicalLassoCV`` (sklearn) – promotes RuntimeWarnings to errors
       and falls back on failure.
    2. *Neighbourhood Lasso* via per-column ``LassoCV``.
    3. Pseudo-inverse of the correlation matrix.

    Parameters
    ----------
    X : ndarray (n, p) – observations (raw or pre-standardised).
    random_state : int – seed for LassoCV.
    standardize : bool – z-score X before estimation.

    Returns
    -------
    Theta : ndarray (p, p) – estimated precision.
    W : ndarray (p, p) – |Theta| with zero diagonal.
    method : str – estimator name that was used.
    """
    Xs = np.asarray(X, dtype=float)
    if standardize:
        mu = Xs.mean(axis=0)
        sd = Xs.std(axis=0, ddof=1)
        sd = np.where(~np.isfinite(sd) | (sd < 1e-10), 1.0, sd)
        Xs = (Xs - mu) / sd

    p = Xs.shape[1]
    Theta = None
    method: str | None = None

    # --- 1. GraphicalLassoCV ---
    if GraphicalLassoCV is not None:
        try:
            gl = GraphicalLassoCV()
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                gl.fit(Xs)
            Theta = np.asarray(gl.precision_, dtype=float)
            method = "GraphicalLassoCV"
        except Exception:
            Theta = None

    # --- 2. Neighbourhood Lasso ---
    if Theta is None and LassoCV is not None:
        Theta = np.zeros((p, p), dtype=float)
        for j in range(p):
            y = Xs[:, j]
            Xj = np.delete(Xs, j, axis=1)
            lcv = LassoCV(cv=5, random_state=random_state).fit(Xj, y)
            beta = lcv.coef_
            idx_other = [i for i in range(p) if i != j]
            Theta[j, idx_other] = beta
        Theta = 0.5 * (Theta + Theta.T)
        np.fill_diagonal(Theta, np.abs(np.diag(np.cov(Xs, rowvar=False))))
        method = "NeighborhoodLasso"

    # --- 3. Pseudo-inverse fallback ---
    if Theta is None:
        C = np.corrcoef(Xs, rowvar=False)
        Theta = np.linalg.pinv(C)
        method = "PseudoInverseCorrelation"

    W = np.abs(Theta)
    np.fill_diagonal(W, 0.0)
    return Theta, W, method  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Graph-comparison utilities
# --------------------------------------------------------------------------- #


def edge_list_topM(
    W: np.ndarray,
    M: int,
) -> list[tuple[int, int, float]]:
    """Return the *M* strongest undirected edges in adjacency matrix *W*."""
    p = W.shape[0]
    pairs: list[tuple[int, int, float]] = []
    for i in range(p):
        for j in range(i + 1, p):
            pairs.append((i, j, float(W[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:M]


def nonzero_edge_count(W: np.ndarray, tol: float = 1e-12) -> int:
    """Number of non-zero undirected edges in *W*."""
    p = W.shape[0]
    cnt = 0
    for i in range(p):
        for j in range(i + 1, p):
            if abs(W[i, j]) > tol:
                cnt += 1
    return cnt


def jaccard_topM(W_hat: np.ndarray, W_prior: np.ndarray, M: int) -> float:
    """Jaccard index of the top-*M* edge sets of two adjacency matrices."""
    A = {(i, j) for i, j, _ in edge_list_topM(W_hat, M)}
    B = {(i, j) for i, j, _ in edge_list_topM(W_prior, M)}
    if len(A | B) == 0:
        return np.nan
    return len(A & B) / len(A | B)


def spearman_union_support(
    W_hat: np.ndarray,
    W_prior: np.ndarray,
    M: int,
) -> float:
    """Spearman rank correlation on the union of top-*M* supports."""
    A = {(i, j) for i, j, _ in edge_list_topM(W_hat, M)}
    B = {(i, j) for i, j, _ in edge_list_topM(W_prior, M)}
    U = sorted(A | B)
    if len(U) < 3:
        return np.nan
    v1 = pd.Series([W_hat[i, j] for i, j in U]).rank()
    v2 = pd.Series([W_prior[i, j] for i, j in U]).rank()
    return float(v1.corr(v2))
