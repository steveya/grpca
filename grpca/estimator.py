"""Sklearn-compatible Graph-Regularized PCA estimator.

Example
-------
>>> from grpca import GraphRegularizedPCA
>>> from grpca.graphs import tenor_chain_graph
>>> W, L, Ltilde = tenor_chain_graph(maturities)
>>> model = GraphRegularizedPCA(n_components=4, graph=Ltilde,
...                             cv='nested_ts', standardize=True)
>>> model.fit(X_train, dates=train_dates)
>>> loss = model.reconstruction_error(X_test, k=3)
"""

from __future__ import annotations

import os
from typing import Any, Callable, cast

import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from .metrics import eigengap, projector, projector_distance, reconstruction_loss
from .splitters import NestedTimeSeriesSplit


# Default gamma grid (dimensionless α / gap_k)
_DEFAULT_GAMMA_GRID = np.array(
    [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.2, 0.3],
    dtype=float,
)


class GraphRegularizedPCA(BaseEstimator, TransformerMixin):
    r"""Graph-Regularized PCA with optional built-in cross-validation.

    Solves the modified eigenvalue problem

    .. math:: \hat V_k = \arg\max_{V^\top V = I_k} \operatorname{tr}
              V^\top (\Sigma - \alpha \tilde L)\, V

    where :math:`\Sigma` is the sample covariance, :math:`\tilde L` is a
    spectrally-normalised graph Laplacian and :math:`\alpha \ge 0`.

    When ``cv`` is set, :math:`\gamma = \alpha / \mathrm{gap}_k` is tuned on
    a nested validation block and the smallest-:math:`\gamma` conservative
    rule is applied with optional projector-stability tie-breaking.

    Parameters
    ----------
    n_components : int, default 4
        Number of components to keep.
    graph : array-like (p, p) or None
        Spectrally-normalised graph Laplacian.  ``None`` → standard PCA.
    alpha : float or None
        Absolute regularisation strength.  Ignored when ``cv`` is set.
    gamma : float or None
        Dimensionless strength (``alpha = gamma × eigengap_k``).
    cv : {``'nested_ts'``, ``'nested'``}, splitter-like, callable, or None
        Cross-validation strategy. If splitter-like, must implement
        ``split(X, dates=...) -> (inner_idx, val_idx)``. If callable, must be
        ``cv(X, dates=None) -> (inner_idx, val_idx)``.
    gamma_grid : array-like or None
        Grid of gamma values for CV search.
    delta : float, default 0.01
        Candidate-set tolerance (fraction above minimum validation loss).
    standardize : bool, default True
        If True, center and scale features (z-score) before computing the
        covariance.  Set to False when using :class:`sklearn.pipeline.Pipeline`
        with an external :class:`~sklearn.preprocessing.StandardScaler`.

    Attributes
    ----------
    components\_ : ndarray (n_components, p)
        Principal axes (rows).  Use :attr:`loadings_` for the (p, k) form.
    explained_variance\_ : ndarray (n_components,)
        Eigenvalues of the modified covariance.
    mean\_ : ndarray (p,)
        Feature means.
    scale\_ : ndarray (p,) or None
        Feature standard deviations (None when ``standardize=False``).
    alpha\_ : float
        Final regularisation strength used.
    gamma\_ : float
        Final dimensionless strength.
    eigengap\_ : float
        Eigengap at *k* of the (outer) sample covariance.
    cv_results\_ : dict or None
    Detailed CV trace (``gamma_grid``, ``alpha_grid`` for inner alpha
    values, ``val_losses``, ``candidate_mask``, etc.).
    """

    def __init__(
        self,
        n_components: int = 4,
        graph: np.ndarray | None = None,
        alpha: float | None = None,
        gamma: float | None = None,
        cv: str | Callable[..., tuple[np.ndarray, np.ndarray]] | Any | None = None,
        gamma_grid: np.ndarray | None = None,
        delta: float = 0.01,
        standardize: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.graph = graph
        self.alpha = alpha
        self.gamma = gamma
        self.cv = cv
        self.gamma_grid = gamma_grid
        self.delta = delta
        self.standardize = standardize
        self.random_state = random_state

    # ---- properties ----------------------------------------------------------

    @property
    def loadings_(self) -> np.ndarray:
        """Loading matrix in (p, k) finance convention (transpose of ``components_``)."""
        check_is_fitted(self)
        return self.components_.T

    # ---- static helpers ------------------------------------------------------

    @staticmethod
    def suggest_gamma_grid(
        graph: np.ndarray | None = None,  # noqa: ARG004
        X: np.ndarray | None = None,  # noqa: ARG004
    ) -> np.ndarray:
        """Return a conservative default gamma grid ``[0, 1e-4, …, 0.3]``."""
        return _DEFAULT_GAMMA_GRID.copy()

    @staticmethod
    def suggest_alpha_range(
        graph: np.ndarray,  # noqa: ARG004
        X: np.ndarray,
        n_components: int = 4,
        n_points: int = 50,
    ) -> np.ndarray:
        """Suggest an absolute-alpha grid calibrated to the data scale.

        Uses the eigengap at *n_components* of the sample covariance to set
        the upper bound at ``0.3 × gap_k``.
        """
        Z = np.asarray(X, dtype=float)
        Sigma = np.cov(Z, rowvar=False, ddof=1)
        evals = np.sort(linalg.eigvalsh(Sigma))[::-1]
        if len(evals) > n_components:
            gap_k = float(evals[n_components - 1] - evals[n_components])
        else:
            gap_k = float(evals[-1])
        gap_k = max(gap_k, 1e-10)
        alpha_max = 0.3 * gap_k
        alpha_min = 1e-4 * gap_k
        return np.concatenate(([0.0], np.geomspace(alpha_min, alpha_max, n_points - 1)))

    # ---- fit -----------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y=None,  # noqa: ARG002
        *,
        dates: np.ndarray | None = None,
        P_prev: np.ndarray | None = None,
    ) -> "GraphRegularizedPCA":
        """Fit the estimator.

        Parameters
        ----------
        X : array-like (n_samples, n_features)
        y : ignored
        dates : date array for time-series CV.
        P_prev : (p, k) projector from prior window for stability tie-breaking.
        """
        X = check_array(X, dtype=np.float64)
        n, p = X.shape

        # ---- standardize -----------------------------------------------------
        self.mean_ = np.mean(X, axis=0)
        if self.standardize:
            sd = np.std(X, axis=0, ddof=1)
            self.scale_ = np.where(sd < 1e-10, 1.0, sd)
        else:
            self.scale_ = None

        Z = self._standardize(X)
        Sigma = np.cov(Z, rowvar=False, ddof=1)

        # optional debug check for outer-train scaling only
        debug_enabled = bool(getattr(self, "debug", False)) or (
            os.getenv("GRPCA_DEBUG_SCALER", "").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        if debug_enabled and n > 1:
            z_mean = np.mean(Z, axis=0)
            if not np.allclose(z_mean, 0.0, atol=1e-7, rtol=0.0):
                raise AssertionError("Outer standardized train mean is not near zero.")
            if self.standardize:
                z_std = np.std(Z, axis=0, ddof=1)
                non_constant = np.std(X, axis=0, ddof=1) >= 1e-10
                if np.any(non_constant):
                    if not np.allclose(z_std[non_constant], 1.0, atol=1e-5, rtol=0.0):
                        raise AssertionError(
                            "Outer standardized train std is not near one."
                        )

        # ---- graph -----------------------------------------------------------
        if self.graph is not None:
            Ltilde = np.asarray(self.graph, dtype=float)
            if Ltilde.ndim != 2 or Ltilde.shape != (p, p):
                raise ValueError(f"graph must be a square ({p}, {p}) matrix")
            if not np.all(np.isfinite(Ltilde)):
                raise ValueError("graph must contain only finite values")
            if not np.allclose(Ltilde, Ltilde.T, rtol=0.0, atol=1e-12):
                raise ValueError("graph must be symmetric")
        else:
            Ltilde = np.zeros((p, p))

        self.n_features_in_ = p
        k = min(self.n_components, p)

        # ---- determine alpha -------------------------------------------------
        if self.cv is not None and self.graph is not None:
            cv_out = self._fit_cv(X, Ltilde, k, dates, P_prev)
            gamma_star = float(cv_out["gamma_star"])
            k_eff = int(cv_out["k_tune_used"])
            evals_outer = np.sort(linalg.eigvalsh(Sigma))[::-1]
            gap_outer, gap_raw_outer, small_outer = eigengap(evals_outer, k_eff)
            if (not np.isfinite(gap_outer)) or (gap_outer <= 1e-10):
                gap_outer = 1e-10
                small_outer = True
            self.alpha_ = float(gamma_star * gap_outer)
            self.gamma_ = gamma_star
            self.eigengap_ = float(gap_outer)
            self.small_gap_ = bool(small_outer)

            cv_out["gap_outer"] = float(gap_outer)
            cv_out["gap_raw_outer"] = float(gap_raw_outer)
            cv_out["small_gap_outer"] = bool(small_outer)
            cv_out["alpha_star_outer"] = float(self.alpha_)
            self.cv_results_ = cv_out
        elif self.gamma is not None and self.graph is not None:
            evals = np.sort(linalg.eigvalsh(Sigma))[::-1]
            gap, _, small = eigengap(evals, k)
            self.eigengap_ = gap
            self.alpha_ = float(self.gamma * gap)
            self.gamma_ = float(self.gamma)
            self.small_gap_ = small
            self.cv_results_ = None
        elif self.alpha is not None:
            evals = np.sort(linalg.eigvalsh(Sigma))[::-1]
            gap, _, small = eigengap(evals, k)
            self.eigengap_ = gap
            self.alpha_ = float(self.alpha)
            self.gamma_ = float(self.alpha_ / gap) if gap > 0 else 0.0
            self.small_gap_ = small
            self.cv_results_ = None
        else:
            self.alpha_ = 0.0
            self.gamma_ = 0.0
            self.eigengap_ = np.nan
            self.small_gap_ = False
            self.cv_results_ = None

        # ---- eigen-decomposition ---------------------------------------------
        A = Sigma - self.alpha_ * Ltilde
        eigenvalues, eigenvectors = linalg.eigh(A)
        order = np.argsort(eigenvalues)[::-1]

        self.components_ = eigenvectors[:, order[:k]].T  # (k, p)
        self.explained_variance_ = eigenvalues[order[:k]]
        self.all_eigenvalues_ = eigenvalues[order]
        self.n_components_ = k
        return self

    # ---- CV ------------------------------------------------------------------

    def _fit_cv(self, X, Ltilde, k, dates, P_prev):
        gamma_grid = (
            np.asarray(self.gamma_grid, dtype=float)
            if self.gamma_grid is not None
            else _DEFAULT_GAMMA_GRID.copy()
        )
        X = np.asarray(X, dtype=float)

        # --- nested split ---
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
                inner_idx, val_idx = splitter.split(X, dates=dates)
            except TypeError:
                inner_idx, val_idx = splitter.split(X)
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

        inner_idx = np.asarray(inner_idx, dtype=int)
        val_idx = np.asarray(val_idx, dtype=int)

        if len(inner_idx) == 0 or len(val_idx) == 0:
            return {
                "gamma_grid": gamma_grid.copy(),
                "alpha_grid": np.zeros_like(gamma_grid, dtype=float),
                "val_losses": np.full_like(gamma_grid, np.nan, dtype=float),
                "candidate_mask": np.zeros_like(gamma_grid, dtype=bool),
                "min_val_loss": np.nan,
                "gap_inner": 1e-10,
                "gap_raw_inner": 1e-10,
                "small_gap_inner": True,
                "k_tune_used": 1,
                "selected_idx": 0,
                "flatness": np.nan,
                "gamma_star": float(gamma_grid[0]) if len(gamma_grid) > 0 else 0.0,
                "alpha_star_inner": 0.0,
                "inner_mean": None,
                "inner_scale": None,
                "stability_tiebreak_used": False,
            }

        X_inner = X[inner_idx]
        X_val = X[val_idx]

        mu_in = np.mean(X_inner, axis=0)
        sd_in = None
        if self.standardize:
            sd_in = np.std(X_inner, axis=0, ddof=1)
            sd_in = np.where(sd_in < 1e-10, 1.0, sd_in)
            Z_inner = (X_inner - mu_in) / sd_in
            Z_val = (X_val - mu_in) / sd_in
        else:
            Z_inner = X_inner - mu_in
            Z_val = X_val - mu_in

        Sigma_inner = np.cov(Z_inner, rowvar=False, ddof=1)

        p = X.shape[1]
        rank_eff = int(np.linalg.matrix_rank(Sigma_inner))
        k_eff = int(min(k, p, max(rank_eff, 1)))

        evals_inner = np.sort(linalg.eigvalsh(Sigma_inner))[::-1]
        gap_inner, gap_raw_inner, small_inner = eigengap(evals_inner, k_eff)
        if (not np.isfinite(gap_inner)) or (gap_inner <= 1e-10):
            gap_inner = 1e-10
            small_inner = True
        alpha_vals = gamma_grid * float(gap_inner)

        val_losses: list[float] = []
        stab_dists: list[float] = []

        for a in alpha_vals:
            A_inner = Sigma_inner - float(a) * Ltilde
            ev, evec = linalg.eigh(A_inner)
            order = np.argsort(ev)[::-1]
            V_a = evec[:, order[:k_eff]]
            val_losses.append(float(reconstruction_loss(Z_val, V_a)))

            if P_prev is None:
                stab_dists.append(np.nan)
            else:
                P_a = projector(V_a)
                p_prev_arr = np.asarray(P_prev, dtype=float)
                if (
                    p_prev_arr.ndim != 2
                    or p_prev_arr.shape[0] != P_a.shape[0]
                    or p_prev_arr.shape[1] != P_a.shape[1]
                ):
                    stab_dists.append(np.nan)
                else:
                    try:
                        stab_dists.append(projector_distance(P_a, p_prev_arr, k_eff))
                    except Exception:
                        stab_dists.append(np.nan)

        val_losses_arr = np.asarray(val_losses, dtype=float)
        stab_arr = np.asarray(stab_dists, dtype=float)
        finite_mask = np.isfinite(val_losses_arr)
        if not np.any(finite_mask):
            min_idx = 0
            min_val = np.nan
            cand_mask = np.zeros_like(val_losses_arr, dtype=bool)
            cand_mask[min_idx] = True
        else:
            finite_vals = val_losses_arr[finite_mask]
            min_val = float(np.min(finite_vals))
            tie_tol = 1e-12
            cand_mask = finite_mask & (
                val_losses_arr <= (1.0 + self.delta) * min_val + tie_tol
            )
            cand_idx_tmp = np.where(cand_mask)[0]
            if len(cand_idx_tmp) == 0:
                min_idx = int(np.where(finite_mask)[0][np.argmin(finite_vals)])
                cand_mask[min_idx] = True

        cand_idx = np.where(cand_mask)[0]

        if len(cand_idx) == 0:
            cand_idx = np.array([0], dtype=int)

        # Selection: smallest gamma, stability tie-break
        gamma_cand = gamma_grid[cand_idx]
        stab_cand = stab_arr[cand_idx]
        stability_tiebreak_used = False

        if P_prev is None or np.all(~np.isfinite(stab_cand)):
            gamma_star = float(np.min(gamma_cand))
        else:
            stability_tiebreak_used = True
            tie_tol = 1e-12
            stab_min = float(np.nanmin(stab_cand))
            stab_best = np.isfinite(stab_cand) & np.isclose(
                stab_cand, stab_min, rtol=0.0, atol=tie_tol
            )
            gamma_star = (
                float(np.min(gamma_cand[stab_best]))
                if np.any(stab_best)
                else float(np.min(gamma_cand))
            )

        exact = np.where(gamma_grid == gamma_star)[0]
        if len(exact) > 0:
            sel_idx = int(exact[0])
        else:
            sel_idx = int(np.argmin(np.abs(gamma_grid - gamma_star)))
        gamma_star = float(gamma_grid[sel_idx])

        second_best = (
            float(np.partition(val_losses_arr, 1)[1])
            if len(val_losses_arr) >= 2
            else min_val
        )
        if np.isfinite(min_val) and np.isfinite(second_best):
            flatness = float((second_best - min_val) / max(abs(min_val), 1e-12))
        else:
            flatness = np.nan

        alpha_star_inner = float(gamma_star * gap_inner)

        return {
            "gamma_grid": gamma_grid.copy(),
            "alpha_grid": alpha_vals.copy(),
            "val_losses": val_losses_arr.copy(),
            "candidate_mask": cand_mask.copy(),
            "min_val_loss": min_val,
            "gap_inner": float(gap_inner),
            "gap_raw_inner": float(gap_raw_inner),
            "small_gap_inner": bool(small_inner),
            "k_tune_used": k_eff,
            "selected_idx": int(sel_idx),
            "flatness": flatness,
            "gamma_star": gamma_star,
            "alpha_star_inner": alpha_star_inner,
            "inner_mean": mu_in.copy(),
            "inner_scale": sd_in.copy() if isinstance(sd_in, np.ndarray) else None,
            "stability_tiebreak_used": stability_tiebreak_used,
        }

    # ---- private standardization --------------------------------------------

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        Z = X - self.mean_
        if self.scale_ is not None:
            Z = Z / self.scale_
        return Z

    # ---- transform / reconstruct --------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project *X* into the component space, shape ``(n, k)``."""
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64)
        Z = self._standardize(X)
        return Z @ self.components_.T  # (n, k)

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """Reconstruct from scores, shape ``(n, p)``."""
        check_is_fitted(self)
        Z_hat = X_transformed @ self.components_  # (n, p)
        if self.scale_ is not None:
            return Z_hat * self.scale_ + self.mean_
        return Z_hat + self.mean_

    # ---- evaluation ----------------------------------------------------------

    def reconstruction_error(
        self,
        X: np.ndarray,
        k: int | None = None,
        per_obs: bool = False,
    ) -> float | np.ndarray:
        """Compute reconstruction error on *X*.

        Parameters
        ----------
        X : array-like (n, p)
        k : int or None – use first *k* components (default: all).
        per_obs : bool – return per-observation vector.
        """
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64)
        Z = self._standardize(X)
        k_use = k if k is not None else self.n_components_
        V = self.loadings_[:, :k_use]  # (p, k_use)
        return reconstruction_loss(Z, V, per_obs=per_obs)

    def score(self, X: np.ndarray, y=None) -> float:  # noqa: ARG002
        """Negative reconstruction error (sklearn convention: higher is better)."""
        return -float(self.reconstruction_error(X))

    def stability_report(
        self,
        other: "GraphRegularizedPCA",
        k: int | None = None,
    ) -> dict:
        """Compare this fitted model with *other*.

        Returns
        -------
        dict with subspace_distance, projector_distance, component_distances.
        """
        check_is_fitted(self)
        check_is_fitted(other)
        k_use = k if k is not None else min(self.n_components_, other.n_components_)
        V1 = self.loadings_
        V2 = other.loadings_
        from .metrics import subspace_distance as _sd, component_distances as _cd

        return {
            "subspace_distance": _sd(V1, V2, k_use),
            "projector_distance": projector_distance(
                projector(V1[:, :k_use]),
                projector(V2[:, :k_use]),
                k_use,
            ),
            "component_distances": _cd(V1, V2, k_use),
        }
