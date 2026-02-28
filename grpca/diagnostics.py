"""Statistical diagnostics and GLRM diagnostic data-prep helpers."""

from __future__ import annotations

from itertools import permutations
from math import erf, sqrt
from typing import Any, Iterable

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None  # type: ignore[assignment]

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


def summarize_cv_surface(cv_results: dict[str, Any]) -> Any:
    """Summarize a CV surface into a tidy table.

    Returns a pandas DataFrame when pandas is available, otherwise a list of
    dictionaries copied from ``cv_results['results_table']``.
    """
    rows = cv_results.get("results_table", [])
    if not isinstance(rows, list):
        raise ValueError("cv_results['results_table'] must be a list")
    if pd is not None:
        return pd.DataFrame(rows)
    return [dict(r) for r in rows]


def extract_selected_tau_series(
    cv_results_over_time: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return plot-friendly selected-``tau`` values over time/windows."""
    out: list[dict[str, Any]] = []
    for i, res in enumerate(cv_results_over_time):
        out.append(
            {
                "index": i,
                "window": res.get("window", i),
                "tau": res.get("selected_tau", np.nan),
            }
        )
    return out


def extract_selected_rho_series(
    cv_results_over_time: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return plot-friendly selected-``rho`` values over time/windows."""
    out: list[dict[str, Any]] = []
    for i, res in enumerate(cv_results_over_time):
        out.append(
            {
                "index": i,
                "window": res.get("window", i),
                "rho": res.get("selected_rho", np.nan),
            }
        )
    return out


def extract_selected_lambda_series(
    cv_results_over_time: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return plot-friendly selected ``lambda_s`` / ``lambda_f`` over time."""
    out: list[dict[str, Any]] = []
    for i, res in enumerate(cv_results_over_time):
        out.append(
            {
                "index": i,
                "window": res.get("window", i),
                "lambda_s": res.get("selected_lambda_s", np.nan),
                "lambda_f": res.get("selected_lambda_f", np.nan),
            }
        )
    return out


def extract_validation_loss_surface(
    cv_results: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return plot-friendly validation-loss surface records over ``(tau, rho)``."""
    rows = cv_results.get("results_table", [])
    if not isinstance(rows, list):
        raise ValueError("cv_results['results_table'] must be a list")
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "tau": row.get("tau", np.nan),
                "rho": row.get("rho", np.nan),
                "lambda_s": row.get("lambda_s", np.nan),
                "lambda_f": row.get("lambda_f", np.nan),
                "val_loss": row.get("val_loss", np.nan),
            }
        )
    return out
