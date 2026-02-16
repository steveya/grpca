"""Graph construction utilities for Graph-Regularized PCA.

Functions
---------
spectral_normalize : Normalize a graph Laplacian by its largest eigenvalue.
tenor_chain_graph  : Build adjacency for a sequential tenor chain.
hierarchical_graph : Build adjacency with community reinforcement and dampened edges.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg


def spectral_normalize(L: np.ndarray) -> np.ndarray:
    """Normalize a graph Laplacian so its largest eigenvalue equals 1.

    Parameters
    ----------
    L : ndarray, shape (p, p)
        Combinatorial Laplacian (D − W).

    Returns
    -------
    ndarray, shape (p, p)
        Spectrally-normalized Laplacian.
    """
    evals = linalg.eigh(L, eigvals_only=True)
    lmax = float(np.max(evals)) if len(evals) else 0.0
    if lmax <= 0:
        return L.copy()
    return L / lmax


def _laplacian(W: np.ndarray) -> np.ndarray:
    D = np.diag(W.sum(axis=1))
    return D - W


def tenor_chain_graph(
    maturities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a tenor-chain graph over ordered maturities.

    Edge weights between adjacent tenors are inversely proportional to the
    squared maturity gap: ``w_{i,i+1} = 1 / (m_{i+1} - m_i)^2``.

    Parameters
    ----------
    maturities : array-like, shape (p,)
        Sorted maturities in years (e.g., [1, 2, 3, 5, 7, 10, …]).

    Returns
    -------
    W : ndarray, shape (p, p) – adjacency matrix.
    L : ndarray, shape (p, p) – combinatorial Laplacian.
    L_tilde : ndarray, shape (p, p) – spectrally normalized Laplacian.
    """
    m = np.asarray(maturities, dtype=float)
    p = len(m)
    W = np.zeros((p, p), dtype=float)
    for i in range(p - 1):
        gap = float(abs(m[i + 1] - m[i]))
        w = 1.0 / (gap * gap)
        W[i, i + 1] = w
        W[i + 1, i] = w
    L = _laplacian(W)
    return W, L, spectral_normalize(L)


def hierarchical_graph(
    maturities: np.ndarray,
    communities: dict[str, list[float]] | None = None,
    community_boost: float = 0.35,
    dampened_edges: list[tuple[float, float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a hierarchical graph with community reinforcement.

    Starts from a tenor chain, then:
    1. Adds intra-community edges with ``community_boost``.
    2. Dampens specified cross-boundary edges multiplicatively.

    Parameters
    ----------
    maturities : array-like, shape (p,)
        Sorted maturities in years.
    communities : dict mapping name → list of maturities.
        Default: ``{"deliverable": [7, 8, 9, 10]}``.
    community_boost : float
        Additive weight boost for intra-community edges.
    dampened_edges : list of (m1, m2, factor)
        Each tuple dampens the (m1, m2) edge by ``factor``.
        Default: ``[(10.0, 15.0, 0.10)]``.

    Returns
    -------
    W, L, L_tilde : ndarray, each shape (p, p)
    """
    m = np.asarray(maturities, dtype=float)
    p = len(m)
    W = np.zeros((p, p), dtype=float)

    # Base: tenor chain
    for i in range(p - 1):
        gap = float(abs(m[i + 1] - m[i]))
        w = 1.0 / (gap * gap)
        W[i, i + 1] = w
        W[i + 1, i] = w

    idx_map = {float(v): i for i, v in enumerate(m)}

    # Community reinforcement
    if communities is None:
        communities = {"deliverable": [7.0, 8.0, 9.0, 10.0]}
    for _name, members in communities.items():
        present = [v for v in members if v in idx_map]
        for ii in range(len(present)):
            for jj in range(ii + 1, len(present)):
                a, b = idx_map[present[ii]], idx_map[present[jj]]
                W[a, b] += community_boost
                W[b, a] += community_boost

    # Dampened boundary edges
    if dampened_edges is None:
        dampened_edges = [(10.0, 15.0, 0.10)]
    for m1, m2, factor in dampened_edges:
        if m1 in idx_map and m2 in idx_map:
            i1, i2 = idx_map[m1], idx_map[m2]
            W[i1, i2] *= factor
            W[i2, i1] *= factor

    L = _laplacian(W)
    return W, L, spectral_normalize(L)
