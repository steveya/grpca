"""Reconstruction loss, subspace distance and related metrics."""

from __future__ import annotations

import numpy as np
from scipy import linalg


def reconstruction_loss(
    Z: np.ndarray,
    V: np.ndarray,
    per_obs: bool = False,
) -> np.ndarray | float:
    r"""Frobenius reconstruction loss.

    .. math:: \ell = \frac{\|Z - Z V V^\top\|_F^2}{n \cdot p}

    Parameters
    ----------
    Z : ndarray, shape (n, p) – standardized data.
    V : ndarray, shape (p, k) – loading matrix.
    per_obs : bool – if True, return per-observation losses shape (n,).

    Returns
    -------
    float or ndarray
    """
    Z_hat = (Z @ V) @ V.T
    n, p = Z.shape
    if per_obs:
        return np.sum((Z - Z_hat) ** 2, axis=1) / p
    return float(np.linalg.norm(Z - Z_hat, ord="fro") ** 2 / (n * p))


def projector(V: np.ndarray) -> np.ndarray:
    r"""Orthogonal projector :math:`V V^\top`."""
    return V @ V.T


def projector_distance(P: np.ndarray, Q: np.ndarray, k: int) -> float:
    r"""Normalized Frobenius distance between two projectors.

    .. math:: d = \frac{\|P - Q\|_F}{\sqrt{2k}}
    """
    return float(np.linalg.norm(P - Q, ord="fro") / np.sqrt(2.0 * k))


def subspace_distance(V1: np.ndarray, V2: np.ndarray, k: int) -> float:
    r"""Davis–Kahan subspace distance: :math:`\sin(\theta_{\max})`.

    Parameters
    ----------
    V1, V2 : ndarray, shape (p, r)
        Loading matrices; the first *k* columns define the subspaces.
    k : int – number of leading columns to compare.

    Returns
    -------
    float in [0, 1]
    """
    A = np.asarray(V1, dtype=float)[:, :k]
    B = np.asarray(V2, dtype=float)[:, :k]
    angles = linalg.subspace_angles(A, B)
    return float(np.sin(np.max(angles)))


def component_distances(V1: np.ndarray, V2: np.ndarray, k: int) -> np.ndarray:
    r"""Per-component absolute cosine distance :math:`1 - |\cos\theta_i|`.

    Invariant to sign flips. Returns array of shape ``(k,)``.
    """
    A = np.asarray(V1, dtype=float)[:, :k]
    B = np.asarray(V2, dtype=float)[:, :k]
    nA = np.linalg.norm(A, axis=0)
    nB = np.linalg.norm(B, axis=0)
    cos_abs = np.abs(np.sum(A * B, axis=0) / (nA * nB))
    cos_abs = np.clip(cos_abs, 0.0, 1.0)
    return 1.0 - cos_abs


def eigengap(
    eigenvalues: np.ndarray,
    k: int,
    eps: float = 1e-10,
) -> tuple[float, float, bool]:
    r"""Eigengap :math:`\lambda_k - \lambda_{k+1}`.

    Parameters
    ----------
    eigenvalues : 1-D array (need not be sorted).
    k : int – position at which gap is measured (1-based rank).
    eps : float – minimum returned gap.

    Returns
    -------
    gap : float – ``max(gap_raw, eps)``.
    gap_raw : float – raw difference (may be ≤ 0).
    small_gap : bool – True if the raw gap is ≤ eps or non-finite.
    """
    evals = np.sort(np.asarray(eigenvalues, dtype=float))[::-1]
    if len(evals) <= k:
        return float(eps), float(eps), True
    gap_raw = float(evals[k - 1] - evals[k])
    small = (not np.isfinite(gap_raw)) or (gap_raw <= eps)
    gap = float(max(gap_raw, eps)) if np.isfinite(gap_raw) else float(eps)
    return gap, gap_raw, small
