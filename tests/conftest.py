"""Shared fixtures for grpca test suite."""

from __future__ import annotations

import numpy as np
import pytest

from grpca.graphs import day_chain_graph, spectral_normalize_laplacian

# ---------------------------------------------------------------------------
# RNG
# ---------------------------------------------------------------------------

@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic random number generator."""
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Small matrices & Laplacians
# ---------------------------------------------------------------------------

def _chain_laplacian(p: int) -> np.ndarray:
    """Build a spectrally-normalized chain Laplacian of size *p*."""
    w = np.zeros((p, p), dtype=float)
    for i in range(p - 1):
        w[i, i + 1] = 1.0
        w[i + 1, i] = 1.0
    d = np.diag(w.sum(axis=1))
    L = d - w
    return spectral_normalize_laplacian(L)


@pytest.fixture
def chain_laplacian_8() -> np.ndarray:
    """8×8 spectrally-normalized chain Laplacian."""
    return _chain_laplacian(8)


@pytest.fixture
def chain_laplacian_6() -> np.ndarray:
    """6×6 spectrally-normalized chain Laplacian."""
    return _chain_laplacian(6)


@pytest.fixture
def example_chain_laplacian_80() -> np.ndarray:
    """80×80 day-chain Laplacian (observation side)."""
    _, _, ltilde = day_chain_graph(np.arange(80), normalize=True)
    return ltilde


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

@pytest.fixture
def dense_data_80x8(rng: np.random.Generator) -> np.ndarray:
    """Dense (no missing) data matrix (80, 8)."""
    return rng.normal(size=(80, 8))


@pytest.fixture
def sparse_data_80x8(rng: np.random.Generator) -> np.ndarray:
    """Data matrix (80, 8) with ~15 % missing entries."""
    x = rng.normal(size=(80, 8))
    x[rng.random(x.shape) < 0.15] = np.nan
    return x


@pytest.fixture
def dense_data_120x6(rng: np.random.Generator) -> np.ndarray:
    """Dense data matrix (120, 6)."""
    return rng.normal(size=(120, 6))


@pytest.fixture
def dates_80():
    """80-element business-day date index."""
    import pandas as pd
    return pd.bdate_range("2023-01-02", periods=80)


# ---------------------------------------------------------------------------
# Helpers available to all tests
# ---------------------------------------------------------------------------

def colwise_nanstd_or_one(x: np.ndarray) -> np.ndarray:
    """Column-wise std (ddof=1) on observed entries, floored at 1."""
    p = x.shape[1]
    out = np.ones(p, dtype=float)
    for j in range(p):
        vals = x[np.isfinite(x[:, j]), j]
        if vals.size < 2:
            continue
        sd = float(np.nanstd(vals, ddof=1))
        out[j] = sd if np.isfinite(sd) and sd >= 1e-10 else 1.0
    return out
