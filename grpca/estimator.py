r"""Sklearn-compatible Graph-Regularized PCA as a unified masked GLRM.

The estimator solves

.. math::

    \min_{U,V}\; \tfrac{1}{2s_X}\|P_\Omega(Z-UV^\top)\|_F^2
    + \tfrac12\lambda_s\operatorname{tr}(U^\top \tilde L_s U)
    + \tfrac12\lambda_f\operatorname{tr}(V^\top \tilde L_f V)
    + \tfrac12\mu_u\|U\|_F^2
    + \tfrac12\mu_v\|V\|_F^2,

where ``Z`` is the standardized training matrix and missing entries are
encoded by ``np.nan``. The train-only data scale is
``s_X = ||Z_fill||_2^2`` with ``Z_fill = where(observed, Z, 0)``.

Examples
--------
Feature graph only

>>> from grpca import GraphRegularizedPCA
>>> from grpca.graphs import tenor_chain_graph
>>> _, _, Lf = tenor_chain_graph(maturities)
>>> model = GraphRegularizedPCA(n_components=4, graph=Lf, tau=1e-2, rho=0.0)
>>> model.fit(X_train)

Both graphs with automatic day-chain example graph from dates

>>> model = GraphRegularizedPCA(
...     n_components=4,
...     graph=Lf,
...     use_day_chain=True,
...     cv="nested_ts",
... )
>>> model.fit(X_train, dates=train_dates)
"""

from __future__ import annotations

import inspect
import os
import warnings
from typing import Any, Callable, cast

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from .graphs import day_chain_graph, spectral_normalize_laplacian
from .metrics import (
    example_roughness,
    feature_roughness,
    masked_reconstruction_loss,
    normalized_reconstruction_difference,
    procrustes_distance,
    spectral_data_scale,
)
from .splitters import NestedTimeSeriesSplit


_DEFAULT_TAU_GRID = np.array(
    [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1], dtype=float
)
_DEFAULT_RHO_GRID = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)


class GraphRegularizedPCA(BaseEstimator, TransformerMixin):
    r"""Masked two-sided graph-regularized low-rank model (GLRM).

    Objective
    ---------
    .. math::
        \min_{U,V}\; \tfrac{1}{2s_X}\|P_\Omega(Z-UV^\top)\|_F^2
        + \tfrac12\lambda_s\operatorname{tr}(U^\top \tilde L_s U)
        + \tfrac12\lambda_f\operatorname{tr}(V^\top \tilde L_f V)
        + \tfrac12\mu_u\|U\|_F^2
        + \tfrac12\mu_v\|V\|_F^2

    The feature graph (``graph``) and example graph (``example_graph``) are
    optional; missing entries must be represented as ``np.nan``.

    Both Laplacians are spectrally normalized and the data term is normalized
    by a train-only spectral scale ``s_X = ||Z_fill||_2^2`` so regularization
    parameters are more portable across windows/datasets.

    If both feature and example graphs are absent (and no automatic day-chain
    graph is used), the model reduces to ridge-regularized low-rank
    factorization. If additionally ``tau=0`` (equivalently
    ``lambda_s=lambda_f=0``), only ridge penalties ``mu_u`` and ``mu_v`` are
    active.

    With no missingness and tiny ridge penalties, this typically approximates
    PCA-like factors but does not use the spectral eigen-decomposition solver.

    Notes
    -----
    Legacy spectral-only parameters ``alpha``, ``gamma``, and ``gamma_grid``
    are accepted for backward compatibility and emit deprecation warnings;
    they are ignored by the GLRM solver.
    """

    def __init__(
        self,
        n_components: int = 4,
        graph: np.ndarray | None = None,
        example_graph: np.ndarray | None = None,
        tau: float | None = None,
        rho: float | None = None,
        lambda_f: float | None = None,
        lambda_s: float | None = None,
        cv: str | Callable[..., tuple[np.ndarray, np.ndarray]] | Any | None = None,
        tau_grid: np.ndarray | None = None,
        rho_grid: np.ndarray | None = None,
        delta: float = 0.01,
        standardize: bool = True,
        mu_u: float = 1e-4,
        mu_v: float = 1e-4,
        max_iter: int = 500,
        tol: float = 1e-6,
        init: str = "pca",
        use_day_chain: bool = False,
        random_state: int | None = None,
        verbose: bool = False,
        alpha: float | None = None,
        gamma: float | None = None,
        gamma_grid: np.ndarray | None = None,
    ) -> None:
        self.n_components = n_components
        self.graph = graph
        self.example_graph = example_graph
        self.tau = tau
        self.rho = rho
        self.lambda_f = lambda_f
        self.lambda_s = lambda_s
        self.cv = cv
        self.tau_grid = tau_grid
        self.rho_grid = rho_grid
        self.delta = delta
        self.standardize = standardize
        self.mu_u = mu_u
        self.mu_v = mu_v
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.use_day_chain = use_day_chain
        self.random_state = random_state
        self.verbose = verbose
        self.alpha = alpha
        self.gamma = gamma
        self.gamma_grid = gamma_grid

    # ---- properties ----------------------------------------------------------

    @property
    def loadings_(self) -> np.ndarray:
        """Loading matrix in (p, k) finance convention (transpose of ``components_``)."""
        check_is_fitted(self)
        return self.components_.T

    # ---- static helpers -----------------------------------------------------

    @staticmethod
    def suggest_tau_grid(
        graph: np.ndarray | None = None,  # noqa: ARG004
        X: np.ndarray | None = None,  # noqa: ARG004
    ) -> np.ndarray:
        """Return a conservative default tau grid."""
        return _DEFAULT_TAU_GRID.copy()

    @staticmethod
    def suggest_rho_grid(
        graph: np.ndarray | None = None,  # noqa: ARG004
        X: np.ndarray | None = None,  # noqa: ARG004
    ) -> np.ndarray:
        """Return a conservative default rho grid."""
        return _DEFAULT_RHO_GRID.copy()

    @staticmethod
    def suggest_gamma_grid(
        graph: np.ndarray | None = None,  # noqa: ARG004
        X: np.ndarray | None = None,  # noqa: ARG004
    ) -> np.ndarray:
        """Deprecated alias for :meth:`suggest_tau_grid`."""
        warnings.warn(
            "suggest_gamma_grid() is deprecated; use suggest_tau_grid().",
            DeprecationWarning,
            stacklevel=2,
        )
        return _DEFAULT_TAU_GRID.copy()

    # ---- fit -----------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y=None,  # noqa: ARG002
        *,
        dates: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        P_prev: np.ndarray | None = None,
    ) -> "GraphRegularizedPCA":
        """Fit the GLRM model on ``X`` with NaN-missing entries."""
        X = self._check_array_allow_nan(X)
        n, p = X.shape
        if self.n_components < 1:
            raise ValueError("n_components must be >= 1")
        k = int(min(self.n_components, n, p))
        if k < 1:
            raise ValueError("Cannot fit with empty dimensions")

        if mask is None:
            obs_mask = np.isfinite(X)
        else:
            obs_mask = np.asarray(mask, dtype=bool)
            if obs_mask.shape != X.shape:
                raise ValueError("mask must have the same shape as X")
            if np.any(obs_mask & ~np.isfinite(X)):
                raise ValueError("mask marks non-finite entries as observed")

        self._warn_deprecated_spectral_args()
        self.mu_u_ = float(max(self.mu_u, 0.0))
        self.mu_v_ = float(max(self.mu_v, 0.0))

        outer_mean, outer_scale = self._compute_standardization_stats(X, obs_mask)
        Lf = self._validate_graph(self.graph, p, name="graph")

        if self.cv is not None:
            if (
                self.lambda_s is not None
                or self.lambda_f is not None
                or self.tau is not None
                or self.rho is not None
            ):
                warnings.warn(
                    "When cv is set, lambda_s/lambda_f/tau/rho are ignored in favor of CV selection.",
                    UserWarning,
                    stacklevel=2,
                )
            cv_out = self._fit_cv_glrm(
                X=X,
                mask=obs_mask,
                dates=dates,
                P_prev=P_prev,
                Lf=Lf,
            )
            self.tau_ = float(cv_out["selected_tau"])
            self.rho_ = float(cv_out["selected_rho"])
            self.lambda_s_ = float(cv_out["selected_lambda_s"])
            self.lambda_f_ = float(cv_out["selected_lambda_f"])
            self.cv_results_ = cv_out
        else:
            if P_prev is not None:
                warnings.warn(
                    "P_prev is only used during cv-based tie-breaking; ignored when cv=None.",
                    UserWarning,
                    stacklevel=2,
                )
            self.lambda_s_, self.lambda_f_ = self._resolve_lambdas()
            self.tau_ = float(self.lambda_s_ + self.lambda_f_)
            self.rho_ = (
                float(self.lambda_s_ / self.tau_)
                if self.tau_ > 0.0
                else (float(self.rho) if self.rho is not None else 0.0)
            )
            self.cv_results_ = None

        self.mask_train_ = obs_mask.copy()
        self.has_missing_ = bool(np.any(~obs_mask))
        self.n_observed_ = int(np.sum(obs_mask))
        self.observed_fraction_ = float(self.n_observed_ / obs_mask.size)
        self.mean_ = outer_mean
        self.scale_ = outer_scale
        Z = self._standardize(X, mask=obs_mask)
        self.data_scale_ = float(spectral_data_scale(Z, obs_mask))

        if self.cv_results_ is not None:
            self.cv_results_["data_scale_outer"] = float(self.data_scale_)

        full_idx = np.arange(n, dtype=int)
        Ls = self._resolve_example_graph(
            n_total=n,
            train_idx=full_idx,
            dates=dates,
        )

        U0, V0 = self._initialize_factors(Z, obs_mask, n=n, p=p, k=k)
        solve_out = self._solve_glrm(
            Z=Z,
            mask=obs_mask,
            U0=U0,
            V0=V0,
            Ls=Ls,
            Lf=Lf,
            lambda_s=self.lambda_s_,
            lambda_f=self.lambda_f_,
            data_scale=self.data_scale_,
        )

        self.U_ = solve_out["U"]
        self.V_ = solve_out["V"]
        self.components_ = self.V_.T
        self.n_components_ = k
        self.explained_variance_ = None
        self.all_eigenvalues_ = None
        self.objective_trace_ = solve_out["objective_trace"]
        self.n_iter_ = solve_out["n_iter"]
        self.converged_ = solve_out["converged"]
        self.Z_train_ = Z.copy()
        self.L_s_tilde_ = Ls
        self.L_f_tilde_ = Lf
        self.has_feature_graph_ = bool(self.graph is not None)
        self.has_example_graph_ = bool(
            self.example_graph is not None or (dates is not None and self.use_day_chain)
        )
        self.n_features_in_ = p
        if self._debug_enabled():
            self._debug_assert(self.mean_.shape == (p,), "mean_ has incorrect shape")
            mean_check = np.zeros(p, dtype=float)
            for j in range(p):
                vals_j = X[self.mask_train_[:, j], j]
                mean_check[j] = float(np.nanmean(vals_j)) if vals_j.size > 0 else 0.0
            self._debug_assert(
                np.allclose(self.mean_, mean_check, atol=1e-12, rtol=0.0),
                "mean_ is not computed from observed entries only",
            )
            if self.scale_ is not None:
                self._debug_assert(
                    self.scale_.shape == (p,), "scale_ has incorrect shape"
                )
                self._debug_assert(
                    np.all(np.isfinite(self.scale_)), "scale_ must be finite"
                )
                self._debug_assert(
                    np.all(self.scale_ > 0.0), "scale_ must be strictly positive"
                )
                scale_check = np.ones(p, dtype=float)
                for j in range(p):
                    vals_j = X[self.mask_train_[:, j], j]
                    if vals_j.size >= 2:
                        sd_j = float(np.nanstd(vals_j, ddof=1))
                        scale_check[j] = (
                            sd_j if np.isfinite(sd_j) and sd_j >= 1e-10 else 1.0
                        )
                self._debug_assert(
                    np.allclose(self.scale_, scale_check, atol=1e-12, rtol=0.0),
                    "scale_ is not computed from observed entries only",
                )
            self._debug_assert(
                np.all(np.isfinite(self.U_)), "U_ contains non-finite values"
            )
            self._debug_assert(
                np.all(np.isfinite(self.V_)), "V_ contains non-finite values"
            )
            self._debug_assert(
                np.all(np.isfinite(self.Z_train_[self.mask_train_])),
                "Observed standardized train entries must be finite",
            )
            self._debug_assert(
                np.all(np.isnan(self.Z_train_[~self.mask_train_])),
                "Missing standardized train entries must remain NaN",
            )
            train_loss = masked_reconstruction_loss(
                self.Z_train_, self.U_, self.V_, self.mask_train_, per_obs=False
            )
            self._debug_assert(
                np.isfinite(float(train_loss)), "Train loss is not finite"
            )
        return self

    # ---- private utilities ---------------------------------------------------

    def _fit_cv_glrm(
        self,
        *,
        X: np.ndarray,
        mask: np.ndarray,
        dates: np.ndarray | None,
        P_prev: np.ndarray | None,
        Lf: np.ndarray,
    ) -> dict[str, Any]:
        tau_grid = (
            np.asarray(self.tau_grid, dtype=float)
            if self.tau_grid is not None
            else _DEFAULT_TAU_GRID.copy()
        )
        rho_grid = (
            np.asarray(self.rho_grid, dtype=float)
            if self.rho_grid is not None
            else _DEFAULT_RHO_GRID.copy()
        )
        if tau_grid.ndim != 1 or len(tau_grid) == 0:
            raise ValueError("tau_grid must be a non-empty 1D array")
        if rho_grid.ndim != 1 or len(rho_grid) == 0:
            raise ValueError("rho_grid must be a non-empty 1D array")
        if np.any(~np.isfinite(tau_grid)) or np.any(tau_grid < 0.0):
            raise ValueError("tau_grid must contain finite non-negative values")
        if np.any(~np.isfinite(rho_grid)) or np.any(
            (rho_grid < 0.0) | (rho_grid > 1.0)
        ):
            raise ValueError("rho_grid must contain finite values in [0, 1]")

        inner_idx, val_idx = self._resolve_cv_split(X, dates)
        if inner_idx.size == 0 or val_idx.size == 0:
            raise ValueError("CV split produced empty inner or validation set")
        if self._debug_enabled():
            self._debug_assert(
                np.intersect1d(inner_idx, val_idx).size == 0,
                "CV split leakage: inner and validation indices overlap",
            )

        X_inner = X[inner_idx]
        X_val = X[val_idx]
        mask_inner = mask[inner_idx]
        mask_val = mask[val_idx]

        inner_mean, inner_scale = self._compute_standardization_stats(
            X_inner, mask_inner
        )
        Z_inner = self._standardize_with_stats(
            X_inner, mask_inner, inner_mean, inner_scale
        )
        Z_val = self._standardize_with_stats(X_val, mask_val, inner_mean, inner_scale)
        if self._debug_enabled():
            self._debug_assert(
                np.all(np.isfinite(Z_inner) == mask_inner),
                "Inner standardized matrix does not preserve missing mask",
            )
            self._debug_assert(
                np.all(np.isfinite(Z_val) == mask_val),
                "Validation standardized matrix does not preserve missing mask",
            )
        data_scale_inner = float(spectral_data_scale(Z_inner, mask_inner))

        Ls_inner = self._resolve_example_graph(
            n_total=X.shape[0],
            train_idx=inner_idx,
            dates=dates,
        )

        p = X.shape[1]
        k_inner = int(min(self.n_components, X_inner.shape[0], p))
        if k_inner < 1:
            raise ValueError("Invalid inner split: latent dimension becomes zero")

        V_prev = self._prepare_previous_loadings(P_prev, p, k_inner)
        results_table: list[dict[str, Any]] = []

        for tau in tau_grid:
            for rho in rho_grid:
                lambda_s = float(tau * rho)
                lambda_f = float(tau * (1.0 - rho))

                U0, V0 = self._initialize_factors(
                    Z_inner,
                    mask_inner,
                    n=X_inner.shape[0],
                    p=p,
                    k=k_inner,
                )
                solve_out = self._solve_glrm(
                    Z=Z_inner,
                    mask=mask_inner,
                    U0=U0,
                    V0=V0,
                    Ls=Ls_inner,
                    Lf=Lf,
                    lambda_s=lambda_s,
                    lambda_f=lambda_f,
                    data_scale=data_scale_inner,
                )

                V_cand = cast(np.ndarray, solve_out["V"])
                U_cand = cast(np.ndarray, solve_out["U"])
                U_val = self._solve_scores(
                    Z_val,
                    mask_val,
                    V_cand,
                    data_scale=data_scale_inner,
                )
                val_sse = self._masked_sse(Z_val, mask_val, U_val, V_cand)
                val_loss = float(0.5 * val_sse / data_scale_inner)
                if self._debug_enabled():
                    self._debug_assert(
                        U_val.shape == (Z_val.shape[0], V_cand.shape[1]),
                        "Validation scores shape mismatch",
                    )
                    self._debug_assert(
                        np.all(np.isfinite(U_val)),
                        "Validation scores contain non-finite values",
                    )
                    self._debug_assert(
                        np.isfinite(val_loss),
                        "Validation loss is not finite",
                    )

                proc_prev = np.nan
                if V_prev is not None:
                    try:
                        proc_prev = float(procrustes_distance(V_cand, V_prev))
                    except ValueError:
                        proc_prev = np.nan

                results_table.append(
                    {
                        "tau": float(tau),
                        "rho": float(rho),
                        "lambda_s": lambda_s,
                        "lambda_f": lambda_f,
                        "val_loss": val_loss,
                        "converged": bool(solve_out["converged"]),
                        "n_iter": int(solve_out["n_iter"]),
                        "loading_roughness": float(feature_roughness(V_cand, Lf)),
                        "score_roughness": float(example_roughness(U_cand, Ls_inner)),
                        "procrustes_to_prev": proc_prev,
                    }
                )

        val_losses = np.asarray([row["val_loss"] for row in results_table], dtype=float)
        finite_mask = np.isfinite(val_losses)
        tiny_tol = 1e-12
        stability_tiebreak_used = False

        if np.any(finite_mask):
            min_val_loss = float(np.min(val_losses[finite_mask]))
            candidate_mask = finite_mask & (
                val_losses <= (1.0 + self.delta) * min_val_loss + tiny_tol
            )
            if not np.any(candidate_mask):
                best_finite = int(
                    np.where(finite_mask)[0][np.argmin(val_losses[finite_mask])]
                )
                candidate_mask[best_finite] = True
        else:
            min_val_loss = np.nan
            candidate_mask = np.zeros(len(results_table), dtype=bool)
            candidate_mask[0] = True

        candidate_idx = np.where(candidate_mask)[0]
        tau_candidates = np.asarray(
            [results_table[i]["tau"] for i in candidate_idx], dtype=float
        )
        tau_min = float(np.min(tau_candidates))
        idx_tau = candidate_idx[
            np.isclose(tau_candidates, tau_min, rtol=0.0, atol=tiny_tol)
        ]

        idx_after_tau = idx_tau
        if V_prev is not None and len(idx_tau) > 1:
            proc_vals = np.asarray(
                [results_table[i]["procrustes_to_prev"] for i in idx_tau],
                dtype=float,
            )
            finite_proc = np.isfinite(proc_vals)
            if np.any(finite_proc):
                stability_tiebreak_used = True
                proc_min = float(np.min(proc_vals[finite_proc]))
                keep = finite_proc & np.isclose(
                    proc_vals, proc_min, rtol=0.0, atol=tiny_tol
                )
                idx_after_tau = idx_tau[keep]

        rho_candidates = np.asarray(
            [results_table[i]["rho"] for i in idx_after_tau],
            dtype=float,
        )
        rho_min = float(np.min(rho_candidates))
        idx_rho = idx_after_tau[
            np.isclose(rho_candidates, rho_min, rtol=0.0, atol=tiny_tol)
        ]
        selected_idx = int(np.min(idx_rho))
        selected = results_table[selected_idx]

        return {
            "tau_grid": tau_grid.copy(),
            "rho_grid": rho_grid.copy(),
            "results_table": results_table,
            "selected_tau": float(selected["tau"]),
            "selected_rho": float(selected["rho"]),
            "selected_lambda_s": float(selected["lambda_s"]),
            "selected_lambda_f": float(selected["lambda_f"]),
            "min_val_loss": float(min_val_loss),
            "candidate_mask": candidate_mask.copy(),
            "inner_mean": inner_mean.copy(),
            "inner_scale": (
                inner_scale.copy() if isinstance(inner_scale, np.ndarray) else None
            ),
            "stability_tiebreak_used": bool(stability_tiebreak_used),
            "selected_idx": selected_idx,
            "inner_idx": inner_idx.copy(),
            "val_idx": val_idx.copy(),
            "data_scale_inner": float(data_scale_inner),
            "data_scale_outer": np.nan,
        }

    def _resolve_cv_split(
        self,
        X: np.ndarray,
        dates: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(self.cv, str):
            if self.cv == "nested_ts" and dates is not None:
                splitter = NestedTimeSeriesSplit(inner_duration="270D")
                inner_idx, val_idx = splitter.split(X, dates=dates)
            elif self.cv in {"nested", "nested_ts"}:
                splitter = NestedTimeSeriesSplit(inner_frac=0.75)
                inner_idx, val_idx = splitter.split(X)
            else:
                raise ValueError(
                    "Unsupported cv string. Expected one of {'nested_ts', 'nested'} or a splitter/callable."
                )
        elif hasattr(self.cv, "split"):
            splitter = cast(Any, self.cv)
            try:
                split_out = splitter.split(X, dates=dates)
            except TypeError:
                split_out = splitter.split(X)
            inner_idx, val_idx = cast(tuple[np.ndarray, np.ndarray], split_out)
        elif callable(self.cv):
            cv_callable = cast(Callable[..., Any], self.cv)
            try:
                split_out = cv_callable(X, dates=dates)
            except TypeError:
                split_out = cv_callable(X)
            inner_idx, val_idx = cast(tuple[np.ndarray, np.ndarray], split_out)
        else:
            raise ValueError(
                "cv must be None, 'nested_ts', 'nested', a splitter with split(), or a callable."
            )

        return np.asarray(inner_idx, dtype=int), np.asarray(val_idx, dtype=int)

    def _resolve_example_graph(
        self,
        *,
        n_total: int,
        train_idx: np.ndarray,
        dates: np.ndarray | None,
    ) -> np.ndarray:
        n_sub = int(len(train_idx))
        if self.example_graph is not None:
            g_full = np.asarray(self.example_graph, dtype=float)
            if g_full.ndim != 2 or g_full.shape != (n_total, n_total):
                raise ValueError(
                    f"example_graph must have shape ({n_total}, {n_total})"
                )
            if not np.all(np.isfinite(g_full)):
                raise ValueError("example_graph must contain finite values")
            if not np.allclose(g_full, g_full.T, rtol=0.0, atol=1e-12):
                raise ValueError("example_graph must be symmetric")
            g_sub = g_full[np.ix_(train_idx, train_idx)]
            return spectral_normalize_laplacian(g_sub)

        if dates is not None and self.use_day_chain:
            dates_arr = np.asarray(dates)
            if len(dates_arr) != n_total:
                raise ValueError("dates length must match n_samples")
            _, _, ls_sub = day_chain_graph(dates_arr[train_idx], normalize=True)
            return ls_sub

        return np.zeros((n_sub, n_sub), dtype=float)

    def _prepare_previous_loadings(
        self,
        P_prev: np.ndarray | None,
        p: int,
        k: int,
    ) -> np.ndarray | None:
        if P_prev is None:
            return None
        prev = np.asarray(P_prev, dtype=float)
        if prev.ndim != 2 or prev.shape[0] != p:
            return None
        if prev.shape[1] < k:
            return None
        return prev[:, :k].copy()

    @staticmethod
    def _check_array_allow_nan(X: np.ndarray) -> np.ndarray:
        kwargs: dict[str, Any] = {"dtype": np.float64}
        params = inspect.signature(check_array).parameters
        if "ensure_all_finite" in params:
            kwargs["ensure_all_finite"] = "allow-nan"
        else:
            kwargs["force_all_finite"] = "allow-nan"
        return check_array(X, **kwargs)

    @staticmethod
    def _debug_enabled() -> bool:
        flag = os.getenv("GRPCA_DEBUG", "").strip().lower()
        return flag in {"1", "true", "yes", "on"}

    @staticmethod
    def _debug_assert(condition: bool, message: str) -> None:
        if not condition:
            raise AssertionError(message)

    def _warn_deprecated_spectral_args(self) -> None:
        if (
            self.alpha is not None
            or self.gamma is not None
            or self.gamma_grid is not None
        ):
            warnings.warn(
                "alpha/gamma/gamma_grid are deprecated and ignored in GLRM mode.",
                DeprecationWarning,
                stacklevel=2,
            )

    @staticmethod
    def _validate_graph(graph: np.ndarray | None, size: int, name: str) -> np.ndarray:
        if graph is None:
            return np.zeros((size, size), dtype=float)
        g = np.asarray(graph, dtype=float)
        if g.ndim != 2 or g.shape != (size, size):
            raise ValueError(f"{name} must have shape ({size}, {size})")
        if not np.all(np.isfinite(g)):
            raise ValueError(f"{name} must contain finite values")
        if not np.allclose(g, g.T, rtol=0.0, atol=1e-12):
            raise ValueError(f"{name} must be symmetric")
        return spectral_normalize_laplacian(g)

    def _compute_standardization_stats(
        self,
        X: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        n, p = X.shape
        mean = np.zeros(p, dtype=float)
        scale = np.ones(p, dtype=float)
        for j in range(p):
            observed = mask[:, j]
            vals = X[observed, j]
            if vals.size > 0:
                mean[j] = float(np.nanmean(vals))
            else:
                mean[j] = 0.0
            if self.standardize:
                if vals.size < 2:
                    scale[j] = 1.0
                else:
                    sd = float(np.nanstd(vals, ddof=1))
                    scale[j] = sd if np.isfinite(sd) and sd >= 1e-10 else 1.0
        if not self.standardize:
            return mean, None
        _ = n
        return mean, scale

    def _standardize(self, X: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        x_arr = np.asarray(X, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        mask_arr = np.isfinite(x_arr) if mask is None else np.asarray(mask, dtype=bool)
        if mask_arr.shape != x_arr.shape:
            raise ValueError("mask must have the same shape as X")
        return self._standardize_with_stats(x_arr, mask_arr, self.mean_, self.scale_)

    @staticmethod
    def _standardize_with_stats(
        X: np.ndarray,
        mask: np.ndarray,
        mean: np.ndarray,
        scale: np.ndarray | None,
    ) -> np.ndarray:
        Z = np.full_like(X, np.nan, dtype=float)
        centered = X - mean
        if scale is not None:
            centered = centered / scale
        Z[mask] = centered[mask]
        return Z

    def _resolve_lambdas(self) -> tuple[float, float]:
        if self.lambda_s is not None or self.lambda_f is not None:
            ls = float(self.lambda_s) if self.lambda_s is not None else 0.0
            lf = float(self.lambda_f) if self.lambda_f is not None else 0.0
        elif self.tau is not None or self.rho is not None:
            if self.tau is None or self.rho is None:
                raise ValueError("Both tau and rho must be provided together.")
            if not 0.0 <= float(self.rho) <= 1.0:
                raise ValueError("rho must be in [0, 1]")
            ls = float(self.tau) * float(self.rho)
            lf = float(self.tau) * (1.0 - float(self.rho))
        else:
            ls = 0.0
            lf = 0.0

        if ls < 0.0 or lf < 0.0:
            raise ValueError("lambda_s and lambda_f must be non-negative")
        return ls, lf

    def _initialize_factors(
        self,
        Z: np.ndarray,
        mask: np.ndarray,
        *,
        n: int,
        p: int,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.random_state)
        if self.init == "random":
            U0 = 0.01 * rng.standard_normal((n, k))
            V0 = 0.01 * rng.standard_normal((p, k))
            return U0, V0
        if self.init != "pca":
            raise ValueError("init must be one of {'pca', 'random'}")

        Z_imp = np.where(mask, Z, 0.0)
        _, svals, vt = np.linalg.svd(Z_imp, full_matrices=False)
        V0 = vt[:k, :].T
        if svals.size >= k:
            V0 = V0 * np.sqrt(np.maximum(svals[:k], 1e-12))[np.newaxis, :]
        U0 = Z_imp @ V0
        return U0, V0

    def _objective(
        self,
        Z: np.ndarray,
        mask: np.ndarray,
        U: np.ndarray,
        V: np.ndarray,
        Ls: np.ndarray,
        Lf: np.ndarray,
        lambda_s: float,
        lambda_f: float,
        data_scale: float,
    ) -> float:
        z_safe = np.where(mask, Z, 0.0)
        pred = U @ V.T
        resid = np.where(mask, pred - z_safe, 0.0)
        data = 0.5 * float(np.sum(resid * resid)) / float(data_scale)
        rough_s = 0.5 * float(lambda_s) * example_roughness(U, Ls)
        rough_f = 0.5 * float(lambda_f) * feature_roughness(V, Lf)
        ridge_u = 0.5 * self.mu_u_ * float(np.sum(U * U))
        ridge_v = 0.5 * self.mu_v_ * float(np.sum(V * V))
        return data + rough_s + rough_f + ridge_u + ridge_v

    def _grad_U(
        self,
        Z: np.ndarray,
        mask: np.ndarray,
        U: np.ndarray,
        V: np.ndarray,
        Ls: np.ndarray,
        lambda_s: float,
        data_scale: float,
    ) -> np.ndarray:
        z_safe = np.where(mask, Z, 0.0)
        resid = np.where(mask, U @ V.T - z_safe, 0.0)
        return (
            (resid @ V) / float(data_scale)
            + float(lambda_s) * (Ls @ U)
            + self.mu_u_ * U
        )

    def _grad_V(
        self,
        Z: np.ndarray,
        mask: np.ndarray,
        U: np.ndarray,
        V: np.ndarray,
        Lf: np.ndarray,
        lambda_f: float,
        data_scale: float,
    ) -> np.ndarray:
        z_safe = np.where(mask, Z, 0.0)
        resid = np.where(mask, U @ V.T - z_safe, 0.0)
        return (
            (resid.T @ U) / float(data_scale)
            + float(lambda_f) * (Lf @ V)
            + self.mu_v_ * V
        )

    def _solve_glrm(
        self,
        *,
        Z: np.ndarray,
        mask: np.ndarray,
        U0: np.ndarray,
        V0: np.ndarray,
        Ls: np.ndarray,
        Lf: np.ndarray,
        lambda_s: float,
        lambda_f: float,
        data_scale: float,
    ) -> dict[str, Any]:
        U = np.asarray(U0, dtype=float).copy()
        V = np.asarray(V0, dtype=float).copy()

        min_step = 1e-12
        max_backtracking = 50
        objective_trace: list[float] = []

        obj_prev = self._objective(
            Z,
            mask,
            U,
            V,
            Ls,
            Lf,
            lambda_s,
            lambda_f,
            data_scale,
        )
        objective_trace.append(obj_prev)
        converged = False

        for it in range(1, self.max_iter + 1):
            obj_curr = obj_prev

            grad_u = self._grad_U(Z, mask, U, V, Ls, lambda_s, data_scale)
            step_u = 1.0
            accepted_u = False
            for _ in range(max_backtracking):
                U_try = U - step_u * grad_u
                obj_try = self._objective(
                    Z,
                    mask,
                    U_try,
                    V,
                    Ls,
                    Lf,
                    lambda_s,
                    lambda_f,
                    data_scale,
                )
                if obj_try <= obj_curr + 1e-12:
                    U = U_try
                    obj_curr = obj_try
                    accepted_u = True
                    break
                step_u *= 0.5
                if step_u < min_step:
                    break
            if not accepted_u:
                warnings.warn(
                    "Backtracking failed on U update; stopping early.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break

            grad_v = self._grad_V(Z, mask, U, V, Lf, lambda_f, data_scale)
            step_v = 1.0
            accepted_v = False
            for _ in range(max_backtracking):
                V_try = V - step_v * grad_v
                obj_try = self._objective(
                    Z,
                    mask,
                    U,
                    V_try,
                    Ls,
                    Lf,
                    lambda_s,
                    lambda_f,
                    data_scale,
                )
                if obj_try <= obj_curr + 1e-12:
                    V = V_try
                    obj_curr = obj_try
                    accepted_v = True
                    break
                step_v *= 0.5
                if step_v < min_step:
                    break
            if not accepted_v:
                warnings.warn(
                    "Backtracking failed on V update; stopping early.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break

            objective_trace.append(obj_curr)
            rel_dec = (obj_prev - obj_curr) / max(abs(obj_prev), 1.0)
            if self.verbose:
                print(
                    f"[GraphRegularizedPCA] iter={it} obj={obj_curr:.8e} "
                    f"rel_dec={rel_dec:.3e}"
                )
            if rel_dec < self.tol:
                converged = True
                obj_prev = obj_curr
                break
            obj_prev = obj_curr

        return {
            "U": U,
            "V": V,
            "objective_trace": np.asarray(objective_trace, dtype=float),
            "n_iter": int(len(objective_trace) - 1),
            "converged": bool(converged),
        }

    def _solve_scores(
        self,
        Z: np.ndarray,
        mask: np.ndarray,
        V: np.ndarray,
        data_scale: float | None = None,
    ) -> np.ndarray:
        z_arr = np.asarray(Z, dtype=float)
        mask_arr = np.asarray(mask, dtype=bool)
        v_arr = np.asarray(V, dtype=float)
        if z_arr.ndim != 2 or mask_arr.shape != z_arr.shape:
            raise ValueError("Z and mask must be 2D with identical shape")
        if v_arr.ndim != 2 or v_arr.shape[0] != z_arr.shape[1]:
            raise ValueError("V must have shape (n_features, n_components)")

        s_x = float(self.data_scale_) if data_scale is None else float(data_scale)
        s_x = max(s_x, 1e-12)
        n = z_arr.shape[0]
        k = v_arr.shape[1]
        U_new = np.zeros((n, k), dtype=float)
        eye = np.eye(k, dtype=float)

        dense_rows = np.all(mask_arr, axis=1)
        if np.any(dense_rows):
            U_new[dense_rows, :] = z_arr[dense_rows, :] @ v_arr

        for i in range(n):
            if dense_rows[i]:
                continue
            obs = mask_arr[i]
            if not np.any(obs):
                continue
            A = v_arr[obs, :]
            b = z_arr[i, obs]
            gram = A.T @ A + (s_x * self.mu_u_) * eye
            rhs = A.T @ b
            try:
                U_new[i] = np.linalg.solve(gram, rhs)
            except np.linalg.LinAlgError:
                U_new[i] = np.linalg.lstsq(gram, rhs, rcond=None)[0]

        if self._debug_enabled():
            self._debug_assert(U_new.shape == (n, k), "Solved scores shape mismatch")
            self._debug_assert(
                np.all(np.isfinite(U_new)),
                "Solved scores contain non-finite values",
            )
        return U_new

    def _transform_standardized(self, Z: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self._solve_scores(Z, mask, self.V_, data_scale=self.data_scale_)

    # ---- transform / reconstruct --------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Estimate scores for each row of ``X`` using observed entries only."""
        check_is_fitted(self)
        X = self._check_array_allow_nan(X)
        Z = self._standardize(X)
        mask = np.isfinite(Z)
        scores = self._transform_standardized(Z, mask)
        if self._debug_enabled():
            self._debug_assert(
                scores.shape == (X.shape[0], int(self.n_components_)),
                "transform() output shape mismatch",
            )
            self._debug_assert(
                np.all(np.isfinite(scores)),
                "transform() returned non-finite scores",
            )
        return scores

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """Reconstruct full matrix from score matrix."""
        check_is_fitted(self)
        scores = np.asarray(X_transformed, dtype=float)
        Z_hat = scores @ self.V_.T
        if self.scale_ is not None:
            return Z_hat * self.scale_ + self.mean_
        return Z_hat + self.mean_

    # ---- evaluation ----------------------------------------------------------

    def reconstruction_error(
        self,
        X: np.ndarray,
        k: int | None = None,
        per_obs: bool = False,
        normalized: bool = False,
    ) -> float | np.ndarray:
        """Compute masked reconstruction error on observed entries only."""
        check_is_fitted(self)
        X = self._check_array_allow_nan(X)
        Z = self._standardize(X)
        mask = np.isfinite(Z)
        scores = self._transform_standardized(Z, mask)
        if k is None:
            k_use = int(self.n_components_)
        else:
            k_use = int(k)
            if k_use < 1 or k_use > int(self.n_components_):
                raise ValueError(f"k must be in [1, {self.n_components_}], got {k_use}")

        scores_k = scores[:, :k_use]
        loadings_k = self.V_[:, :k_use]
        if not normalized:
            out = masked_reconstruction_loss(
                Z,
                scores_k,
                loadings_k,
                mask,
                per_obs=per_obs,
            )
            if self._debug_enabled():
                obs_per_row = np.sum(mask, axis=1)
                if per_obs:
                    out_arr = np.asarray(out, dtype=float)
                    self._debug_assert(
                        out_arr.shape == (Z.shape[0],),
                        "per_obs loss shape mismatch",
                    )
                    self._debug_assert(
                        np.all(np.isfinite(out_arr[obs_per_row > 0])),
                        "per_obs loss must be finite on rows with observations",
                    )
                    self._debug_assert(
                        np.all(np.isnan(out_arr[obs_per_row == 0])),
                        "per_obs loss must be NaN on rows with zero observations",
                    )
                else:
                    n_obs = int(np.sum(mask))
                    if n_obs > 0:
                        sse = self._masked_sse(Z, mask, scores_k, loadings_k)
                        manual = float(sse / n_obs)
                        self._debug_assert(
                            np.isclose(float(out), manual, atol=1e-10, rtol=0.0),
                            "Masked loss value/count mismatch",
                        )
            return out

        if per_obs:
            raise ValueError("per_obs=True is not supported when normalized=True")
        sse = self._masked_sse(Z, mask, scores_k, loadings_k)
        out = float(0.5 * sse / max(float(self.data_scale_), 1e-12))
        if self._debug_enabled():
            self._debug_assert(
                np.isfinite(out),
                "normalized reconstruction error is not finite",
            )
        return out

    def score(self, X: np.ndarray, y=None) -> float:  # noqa: ARG002
        """Negative reconstruction error (sklearn convention: higher is better)."""
        return -float(self.reconstruction_error(X))

    @staticmethod
    def _masked_sse(
        Z: np.ndarray,
        mask: np.ndarray,
        U: np.ndarray,
        V: np.ndarray,
    ) -> float:
        z_safe = np.where(mask, Z, 0.0)
        resid = np.where(mask, (U @ V.T) - z_safe, 0.0)
        return float(np.sum(resid * resid))

    def stability_report(
        self,
        other: "GraphRegularizedPCA",
        X: np.ndarray | None = None,
    ) -> dict[str, float | None]:
        """Compare two fitted GLRM models with non-orthogonal diagnostics."""
        check_is_fitted(self)
        check_is_fitted(other)

        k_use = int(min(self.n_components_, other.n_components_))
        V1 = self.loadings_[:, :k_use]
        V2 = other.loadings_[:, :k_use]

        self_lf = np.asarray(
            getattr(self, "L_f_tilde_", np.zeros((V1.shape[0], V1.shape[0]))),
            dtype=float,
        )
        other_lf = np.asarray(
            getattr(other, "L_f_tilde_", np.zeros((V2.shape[0], V2.shape[0]))),
            dtype=float,
        )

        self_u = np.asarray(self.U_[:, :k_use], dtype=float)
        other_u = np.asarray(other.U_[:, :k_use], dtype=float)
        self_ls = np.asarray(
            getattr(self, "L_s_tilde_", np.zeros((self_u.shape[0], self_u.shape[0]))),
            dtype=float,
        )
        other_ls = np.asarray(
            getattr(
                other,
                "L_s_tilde_",
                np.zeros((other_u.shape[0], other_u.shape[0])),
            ),
            dtype=float,
        )

        out: dict[str, float | None] = {
            "loading_procrustes_distance": procrustes_distance(V1, V2),
            "self_loading_roughness": float(feature_roughness(V1, self_lf)),
            "other_loading_roughness": float(feature_roughness(V2, other_lf)),
            "self_score_roughness": float(example_roughness(self_u, self_ls)),
            "other_score_roughness": float(example_roughness(other_u, other_ls)),
            "self_reconstruction_error": None,
            "other_reconstruction_error": None,
            "normalized_reconstruction_difference": None,
        }

        if X is not None:
            X_arr = self._check_array_allow_nan(X)
            mask = np.isfinite(X_arr)
            out["self_reconstruction_error"] = float(
                cast(float, self.reconstruction_error(X_arr, per_obs=False))
            )
            out["other_reconstruction_error"] = float(
                cast(float, other.reconstruction_error(X_arr, per_obs=False))
            )
            xhat_self = self.inverse_transform(self.transform(X_arr))
            xhat_other = other.inverse_transform(other.transform(X_arr))
            out["normalized_reconstruction_difference"] = float(
                normalized_reconstruction_difference(X_arr, xhat_self, xhat_other, mask)
            )
        return out
