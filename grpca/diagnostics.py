"""Statistical diagnostics: HAC standard errors, DM test, loading matching."""

from __future__ import annotations

from itertools import permutations
from math import erf, sqrt

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None  # type: ignore[assignment]


def newey_west_se(d: np.ndarray, lag: int) -> float:
    """Newey–West HAC standard error of the sample mean.

    Parameters
    ----------
    d : 1-D array of loss differences.
    lag : number of HAC lags.

    Returns
    -------
    float – standard error (NaN if degenerate).
    """
    x = np.asarray(d, dtype=float)
    x = x[np.isfinite(x)]
    T = len(x)
    if T < 3:
        return np.nan
    u = x - np.mean(x)
    gamma0 = float(np.dot(u, u) / T)
    s = gamma0
    lag = int(max(0, min(lag, T - 1)))
    for h in range(1, lag + 1):
        gamma_h = float(np.dot(u[h:], u[:-h]) / T)
        w_h = 1.0 - h / (lag + 1.0)
        s += 2.0 * w_h * gamma_h
    var_mean = s / T
    if var_mean <= 0:
        return np.nan
    return float(np.sqrt(var_mean))


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def dm_test(
    losses_a: np.ndarray,
    losses_b: np.ndarray,
    lag: int,
    alternative: str = "less",
) -> dict:
    """Diebold–Mariano test for equal predictive ability.

    Parameters
    ----------
    losses_a, losses_b : 1-D arrays of out-of-sample losses.
    lag : HAC truncation lag.
    alternative : ``'less'`` (A < B), ``'greater'``, or ``'two-sided'``.

    Returns
    -------
    dict with ``mean_d``, ``se``, ``dm_stat``, ``p_value``, ``win_rate``.
    """
    d = np.asarray(losses_a, dtype=float) - np.asarray(losses_b, dtype=float)
    dbar = float(np.mean(d))
    se = newey_west_se(d, lag)
    dm = dbar / se if np.isfinite(se) and se > 0 else np.nan

    if not np.isfinite(dm):
        p = np.nan
    elif alternative == "less":
        p = _normal_cdf(dm)
    elif alternative == "greater":
        p = 1.0 - _normal_cdf(dm)
    else:
        p = 2.0 * (1.0 - _normal_cdf(abs(dm)))

    return {
        "mean_d": dbar,
        "se": se,
        "dm_stat": dm,
        "p_value": p,
        "win_rate": float(np.mean(d < 0)),
    }


def best_match_scores(C: np.ndarray) -> tuple[float, float]:
    """Optimal column-matching on an absolute-cosine matrix.

    Uses the Hungarian algorithm when available; brute-force otherwise.

    Parameters
    ----------
    C : ndarray (k, k) – absolute inner-product or cosine matrix.

    Returns
    -------
    mean_score, min_score : floats in [0, 1].
    """
    C = np.asarray(C, dtype=float)
    k = C.shape[0]
    if linear_sum_assignment is not None:
        r, c = linear_sum_assignment(-C)
        matched = C[r, c]
        return float(np.mean(matched)), float(np.min(matched))
    best: tuple[float, np.ndarray] | None = None
    for perm in permutations(range(k)):
        vals = np.array([C[i, perm[i]] for i in range(k)], dtype=float)
        score = float(np.sum(vals))
        if best is None or score > best[0]:
            best = (score, vals)
    vals = best[1]  # type: ignore[index]
    return float(np.mean(vals)), float(np.min(vals))
