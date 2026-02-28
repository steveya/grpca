"""Unified GLRM-based Graph-Regularized PCA package.

Exports the sklearn-style `GraphRegularizedPCA` estimator, splitters, and
diagnostic helpers for CV surface inspection.
"""

from .diagnostics import (
    extract_selected_lambda_series,
    extract_selected_rho_series,
    extract_selected_tau_series,
    extract_validation_loss_surface,
    summarize_cv_surface,
)
from .estimator import GraphRegularizedPCA
from .splitters import RollingWindowSplitter, NestedTimeSeriesSplit

__all__ = [
    "GraphRegularizedPCA",
    "RollingWindowSplitter",
    "NestedTimeSeriesSplit",
    "summarize_cv_surface",
    "extract_selected_tau_series",
    "extract_selected_rho_series",
    "extract_selected_lambda_series",
    "extract_validation_loss_surface",
]
