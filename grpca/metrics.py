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


def masked_reconstruction_loss(
    Z: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray,
    per_obs: bool = False,
) -> np.ndarray | float:
    """Masked reconstruction loss for matrices with missing entries.

    Parameters
    ----------
    Z : ndarray, shape (n, p)
        Standardized data matrix (may contain NaN where missing).
    U : ndarray, shape (n, k)
        Example-factor matrix.
    V : ndarray, shape (p, k)
        Feature-factor matrix.
    mask : ndarray, shape (n, p)
        Boolean observed-entry mask.
    per_obs : bool, default False
        If True, return row-wise MSE on observed entries.
    """
    z_arr = np.asarray(Z, dtype=float)
    u_arr = np.asarray(U, dtype=float)
    v_arr = np.asarray(V, dtype=float)
    mask_arr = np.asarray(mask, dtype=bool)

    if z_arr.ndim != 2:
        raise ValueError("Z must be 2-dimensional")
    if u_arr.ndim != 2 or v_arr.ndim != 2:
        raise ValueError("U and V must be 2-dimensional")
    if mask_arr.shape != z_arr.shape:
        raise ValueError("mask must have the same shape as Z")
    if u_arr.shape[0] != z_arr.shape[0] or v_arr.shape[0] != z_arr.shape[1]:
        raise ValueError("U and V shapes are incompatible with Z")
    if u_arr.shape[1] != v_arr.shape[1]:
        raise ValueError("U and V must have the same latent dimension")

    z_safe = np.where(mask_arr, z_arr, 0.0)
    pred = u_arr @ v_arr.T
    resid = np.where(mask_arr, pred - z_safe, 0.0)

    if per_obs:
        obs_per_row = np.sum(mask_arr, axis=1)
        sse_per_row = np.sum(resid * resid, axis=1)
        out = np.full(z_arr.shape[0], np.nan, dtype=float)
        valid = obs_per_row > 0
        out[valid] = sse_per_row[valid] / obs_per_row[valid]
        return out

    n_obs = int(np.sum(mask_arr))
    if n_obs <= 0:
        return np.nan
    return float(np.sum(resid * resid) / n_obs)


def spectral_data_scale(
    Z: np.ndarray,
    mask: np.ndarray,
    *,
    exact_threshold: int = 300,
    power_iter_steps: int = 30,
    eps: float = 1e-12,
) -> float:
    """Return train-only spectral scale ``s_X = ||Z_fill||_2^2``.

    Parameters
    ----------
    Z : ndarray, shape (n, p)
        Standardized matrix, potentially containing NaN at missing entries.
    mask : ndarray, shape (n, p)
        Observed-entry mask.
    exact_threshold : int, default 300
        Use exact SVD when ``min(n, p) <= exact_threshold``.
    power_iter_steps : int, default 30
        Number of power-iteration updates for large matrices.
    eps : float, default 1e-12
        Lower bound for numerical safety.
    """
    z_arr = np.asarray(Z, dtype=float)
    m_arr = np.asarray(mask, dtype=bool)
    if z_arr.ndim != 2:
        raise ValueError("Z must be 2-dimensional")
    if m_arr.shape != z_arr.shape:
        raise ValueError("mask must have the same shape as Z")

    z_fill = np.where(m_arr, z_arr, 0.0)
    n, p = z_fill.shape
    dim_min = min(n, p)

    if dim_min == 0:
        return float(eps)

    if dim_min <= int(exact_threshold):
        svals = linalg.svdvals(z_fill)
        sigma_max = float(svals[0]) if len(svals) else 0.0
        return float(max(sigma_max * sigma_max, eps))

    gram = z_fill.T @ z_fill
    eigvec = np.ones(p, dtype=float)
    eigvec_norm = float(np.linalg.norm(eigvec))
    if eigvec_norm <= 0.0:
        eigvec = np.zeros(p, dtype=float)
        eigvec[0] = 1.0
    else:
        eigvec = eigvec / eigvec_norm

    for _ in range(max(int(power_iter_steps), 1)):
        eigvec_next = gram @ eigvec
        norm_next = float(np.linalg.norm(eigvec_next))
        if norm_next <= 0.0 or not np.isfinite(norm_next):
            return float(eps)
        eigvec = eigvec_next / norm_next

    rayleigh = float(eigvec.T @ gram @ eigvec)
    if not np.isfinite(rayleigh):
        return float(eps)
    return float(max(rayleigh, eps))


def feature_roughness(V: np.ndarray, Lf: np.ndarray) -> float:
    """Return feature-graph roughness ``tr(V^T Lf V)``."""
    v_arr = np.asarray(V, dtype=float)
    lf_arr = np.asarray(Lf, dtype=float)
    if v_arr.ndim != 2 or lf_arr.ndim != 2 or lf_arr.shape[0] != lf_arr.shape[1]:
        raise ValueError("Invalid shapes for V or Lf")
    if lf_arr.shape[0] != v_arr.shape[0]:
        raise ValueError("Lf shape is incompatible with V")
    return float(np.trace(v_arr.T @ lf_arr @ v_arr))


def example_roughness(U: np.ndarray, Ls: np.ndarray) -> float:
    """Return example-graph roughness ``tr(U^T Ls U)``."""
    u_arr = np.asarray(U, dtype=float)
    ls_arr = np.asarray(Ls, dtype=float)
    if u_arr.ndim != 2 or ls_arr.ndim != 2 or ls_arr.shape[0] != ls_arr.shape[1]:
        raise ValueError("Invalid shapes for U or Ls")
    if ls_arr.shape[0] != u_arr.shape[0]:
        raise ValueError("Ls shape is incompatible with U")
    return float(np.trace(u_arr.T @ ls_arr @ u_arr))


def procrustes_distance(V_new: np.ndarray, V_old: np.ndarray) -> float:
    """Orthogonal-Procrustes distance ``min_R ||V_new - V_old R||_F``."""
    a = np.asarray(V_new, dtype=float)
    b = np.asarray(V_old, dtype=float)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("V_new and V_old must be 2-dimensional")
    if a.shape != b.shape:
        raise ValueError("V_new and V_old must have the same shape")

    cross = b.T @ a
    u, _, vt = np.linalg.svd(cross, full_matrices=False)
    r_opt = u @ vt
    return float(np.linalg.norm(a - b @ r_opt, ord="fro"))


def normalized_reconstruction_difference(
    X_true: np.ndarray,
    X_hat_a: np.ndarray,
    X_hat_b: np.ndarray,
    mask: np.ndarray,
    eps: float = 1e-12,
) -> float:
    r"""Normalized RMS difference between two reconstructions on observed entries.

    Computes

    .. math::
        \frac{\sqrt{\frac{1}{|\Omega|}\sum_{(i,j)\in\Omega}
        (\hat X^{(a)}_{ij} - \hat X^{(b)}_{ij})^2}}
        {\sqrt{\frac{1}{|\Omega|}\sum_{(i,j)\in\Omega} X_{ij}^2} + \varepsilon}

    Parameters
    ----------
    X_true : ndarray, shape (n, p)
        Reference matrix used only for normalization scale.
    X_hat_a, X_hat_b : ndarray, shape (n, p)
        Two reconstructed matrices to compare.
    mask : ndarray, shape (n, p)
        Observed-entry mask.
    eps : float, default 1e-12
        Numerical stabilizer for denominator.
    """
    x_true = np.asarray(X_true, dtype=float)
    x_a = np.asarray(X_hat_a, dtype=float)
    x_b = np.asarray(X_hat_b, dtype=float)
    m = np.asarray(mask, dtype=bool)

    if x_true.shape != x_a.shape or x_true.shape != x_b.shape:
        raise ValueError("X_true, X_hat_a and X_hat_b must have the same shape")
    if m.shape != x_true.shape:
        raise ValueError("mask must have the same shape as the input matrices")

    n_obs = int(np.sum(m))
    if n_obs <= 0:
        return np.nan

    diff = np.where(m, x_a - x_b, 0.0)
    numer = float(np.sqrt(np.sum(diff * diff) / n_obs))
    scale = np.where(m, x_true, 0.0)
    denom = float(np.sqrt(np.sum(scale * scale) / n_obs)) + float(eps)
    return numer / denom


def orthonormal_basis(V: np.ndarray) -> np.ndarray:
    """Return an economy-QR orthonormal basis spanning ``col(V)``.

    Parameters
    ----------
    V : ndarray, shape (p, k)
        Loading matrix (need not be orthonormal).

    Returns
    -------
    ndarray, shape (p, r)
        Matrix with orthonormal columns spanning the same subspace as ``V``.
    """
    v_arr = np.asarray(V, dtype=float)
    if v_arr.ndim != 2:
        raise ValueError("V must be 2-dimensional")
    if v_arr.shape[1] == 0:
        raise ValueError("V must have at least one column")
    q, _ = np.linalg.qr(v_arr, mode="reduced")
    return q


def subspace_projector(V: np.ndarray) -> np.ndarray:
    r"""Orthogonal projector onto ``col(V)`` using QR basis.

    .. math:: P = Q Q^\top,\; Q = \operatorname{orthonormal\_basis}(V)
    """
    q = orthonormal_basis(V)
    return q @ q.T


def projector(V: np.ndarray) -> np.ndarray:
    r"""Backward-compatible projector helper.

    Notes
    -----
    For non-orthonormal loadings (GLRM), ``V V^\top`` is not a projector onto
    ``col(V)``. This helper now returns :func:`subspace_projector`.
    """
    return subspace_projector(V)


def projector_distance(
    A: np.ndarray,
    B: np.ndarray,
    k: int,
    *,
    inputs: str = "projectors",
) -> float:
    r"""Normalized Frobenius distance between two projectors.

    .. math:: d = \frac{\|P - Q\|_F}{\sqrt{2k}}

    Parameters
    ----------
    A, B : ndarray
        Either projectors (when ``inputs='projectors'``) or loading matrices
        (when ``inputs='loadings'``).
    k : int
        Number of columns/subspace rank used for normalization.
    inputs : {'projectors', 'loadings'}, default 'projectors'
        Interpretation mode for ``A`` and ``B``.
    """
    if inputs == "projectors":
        p_mat = np.asarray(A, dtype=float)
        q_mat = np.asarray(B, dtype=float)
    elif inputs == "loadings":
        p_mat = subspace_projector(np.asarray(A, dtype=float)[:, :k])
        q_mat = subspace_projector(np.asarray(B, dtype=float)[:, :k])
    else:
        raise ValueError("inputs must be one of {'projectors', 'loadings'}")

    return float(np.linalg.norm(p_mat - q_mat, ord="fro") / np.sqrt(2.0 * k))


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
    a = orthonormal_basis(np.asarray(V1, dtype=float)[:, :k])
    b = orthonormal_basis(np.asarray(V2, dtype=float)[:, :k])
    angles = linalg.subspace_angles(a, b)
    return float(np.sin(np.max(angles)))


def component_distances(V1: np.ndarray, V2: np.ndarray, k: int) -> np.ndarray:
    r"""PCA-only per-component absolute cosine distance :math:`1 - |\cos\theta_i|`.

    Notes
    -----
    This metric assumes component columns are already comparable (for example,
    orthonormal PCA eigenvectors with fixed ordering). For GLRM loadings, use
    :func:`component_distances_procrustes`.

    Invariant to sign flips. Returns array of shape ``(k,)``.
    """
    A = np.asarray(V1, dtype=float)[:, :k]
    B = np.asarray(V2, dtype=float)[:, :k]
    nA = np.linalg.norm(A, axis=0)
    nB = np.linalg.norm(B, axis=0)
    cos_abs = np.abs(np.sum(A * B, axis=0) / (nA * nB))
    cos_abs = np.clip(cos_abs, 0.0, 1.0)
    return 1.0 - cos_abs


def component_distances_procrustes(
    V_new: np.ndarray,
    V_old: np.ndarray,
    k: int,
) -> np.ndarray:
    r"""Per-component distances after optimal orthogonal Procrustes alignment.

    Returns ``1 - |cos|`` for each aligned component column.
    """
    a = np.asarray(V_new, dtype=float)[:, :k]
    b = np.asarray(V_old, dtype=float)[:, :k]
    if a.shape != b.shape:
        raise ValueError("V_new and V_old must have the same first-k shape")

    cross = b.T @ a
    u, _, vt = np.linalg.svd(cross, full_matrices=False)
    r_opt = u @ vt
    a_aligned = a @ r_opt

    n_a = np.linalg.norm(a_aligned, axis=0)
    n_b = np.linalg.norm(b, axis=0)
    denom = n_a * n_b
    cos_abs = np.zeros(k, dtype=float)
    valid = denom > 0.0
    cos_abs[valid] = np.abs(
        np.sum(a_aligned[:, valid] * b[:, valid], axis=0) / denom[valid]
    )
    cos_abs = np.clip(cos_abs, 0.0, 1.0)
    return 1.0 - cos_abs


def eigengap(
    eigenvalues: np.ndarray,
    k: int,
    eps: float = 1e-10,
) -> tuple[float, float, bool]:
    r"""Legacy eigengap helper :math:`\lambda_k - \lambda_{k+1}`.

    Notes
    -----
    Retained for backward compatibility with older spectral-PCA workflows.
    The unified GLRM estimator does not rely on eigengap-based tuning.

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
