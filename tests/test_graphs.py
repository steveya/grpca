"""Tests for grpca.graphs — graph construction and spectral normalization."""

from __future__ import annotations

import numpy as np
import pytest

from grpca.graphs import (
    day_chain_graph,
    hierarchical_graph,
    spectral_normalize_laplacian,
    tenor_chain_graph,
)

# ---------------------------------------------------------------------------
# spectral_normalize_laplacian
# ---------------------------------------------------------------------------


class TestSpectralNormalize:
    def test_unit_top_eigenvalue(self):
        """After normalization, max eigenvalue should be 1."""
        w = np.array(
            [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=float
        )
        L = np.diag(w.sum(1)) - w
        Ltilde = spectral_normalize_laplacian(L)
        evals = np.linalg.eigvalsh(Ltilde)
        assert float(np.max(evals)) == pytest.approx(1.0, abs=1e-10)

    def test_zero_laplacian_returns_copy(self):
        """All-zero matrix should be returned unchanged."""
        L = np.zeros((5, 5))
        Ltilde = spectral_normalize_laplacian(L)
        assert np.allclose(Ltilde, 0.0)

    def test_symmetry_preserved(self):
        L = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=float)
        Ltilde = spectral_normalize_laplacian(L)
        assert np.allclose(Ltilde, Ltilde.T, atol=1e-14)

    def test_psd_preserved(self):
        """Normalized Laplacian should remain positive semi-definite."""
        L = np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]], dtype=float)
        Ltilde = spectral_normalize_laplacian(L)
        evals = np.linalg.eigvalsh(Ltilde)
        assert np.all(evals >= -1e-12)

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            spectral_normalize_laplacian(np.ones((3, 4)))


# ---------------------------------------------------------------------------
# day_chain_graph
# ---------------------------------------------------------------------------


class TestDayChainGraph:
    def test_shape_and_symmetry(self):
        dates = np.arange(np.datetime64("2021-01-01"), np.datetime64("2021-01-06"))
        W, L, Ltilde = day_chain_graph(dates)
        n = len(dates)
        assert W.shape == (n, n)
        assert L.shape == (n, n)
        assert Ltilde.shape == (n, n)
        assert np.allclose(W, W.T)
        assert np.allclose(L, L.T)
        assert np.allclose(Ltilde, Ltilde.T)

    def test_adjacency_is_chain(self):
        """Only consecutive pairs should be connected."""
        dates = np.arange(4)
        W, _, _ = day_chain_graph(dates)
        assert W[0, 1] == 1.0
        assert W[1, 2] == 1.0
        assert W[2, 3] == 1.0
        assert W[0, 2] == 0.0
        assert W[0, 3] == 0.0

    def test_laplacian_row_sums_zero(self):
        dates = np.arange(6)
        _, L, _ = day_chain_graph(dates)
        assert np.allclose(L.sum(axis=1), 0.0, atol=1e-12)

    def test_normalized_top_eigenvalue(self):
        dates = np.arange(10)
        _, _, Ltilde = day_chain_graph(dates, normalize=True)
        evals = np.linalg.eigvalsh(Ltilde)
        assert float(np.max(evals)) == pytest.approx(1.0, abs=1e-10)

    def test_no_normalize(self):
        dates = np.arange(5)
        _, L, Ltilde_no = day_chain_graph(dates, normalize=False)
        assert np.allclose(L, Ltilde_no)

    def test_single_point(self):
        W, L, Ltilde = day_chain_graph(np.array([0]))
        assert W.shape == (1, 1)
        assert W[0, 0] == 0.0
        assert L[0, 0] == 0.0


# ---------------------------------------------------------------------------
# tenor_chain_graph
# ---------------------------------------------------------------------------


class TestTenorChainGraph:
    def test_shape_and_symmetry(self):
        maturities = np.array([1, 2, 3, 5, 7, 10])
        W, L, Ltilde = tenor_chain_graph(maturities)
        p = len(maturities)
        assert W.shape == (p, p)
        assert np.allclose(W, W.T)
        assert np.allclose(L, L.T)

    def test_inverse_squared_gap_weights(self):
        """Edge weights should be 1/(gap^2)."""
        maturities = np.array([1.0, 2.0, 5.0])
        W, _, _ = tenor_chain_graph(maturities)
        assert W[0, 1] == pytest.approx(1.0 / (1.0**2))
        assert W[1, 2] == pytest.approx(1.0 / (3.0**2))
        assert W[0, 2] == 0.0

    def test_laplacian_row_sums_zero(self):
        maturities = np.array([1, 2, 3, 5, 10])
        _, L, _ = tenor_chain_graph(maturities)
        assert np.allclose(L.sum(axis=1), 0.0, atol=1e-12)

    def test_normalized_top_eigenvalue(self):
        maturities = np.array([1, 2, 3, 5, 7, 10, 20, 30])
        _, _, Ltilde = tenor_chain_graph(maturities)
        evals = np.linalg.eigvalsh(Ltilde)
        assert float(np.max(evals)) == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# hierarchical_graph
# ---------------------------------------------------------------------------


class TestHierarchicalGraph:
    def test_default_community_boost(self):
        maturities = np.array([1, 2, 3, 5, 7, 8, 9, 10, 15, 20, 30], dtype=float)
        W, L, Ltilde = hierarchical_graph(maturities)
        p = len(maturities)
        assert W.shape == (p, p)
        assert np.allclose(W, W.T)
        # Community members (7, 8, 9, 10) at indices 4-7 should have boosted edges
        assert W[4, 5] > 0  # 7->8 (chain + boost)
        assert W[4, 7] > 0  # 7->10 (community boost only, not in chain directly)

    def test_custom_communities(self):
        maturities = np.array([1.0, 2.0, 3.0, 5.0])
        W, _, _ = hierarchical_graph(
            maturities,
            communities={"short": [1.0, 2.0, 3.0]},
            community_boost=1.0,
            dampened_edges=[],
        )
        # 1-2 should have chain weight + community boost
        chain_12 = 1.0 / (1.0**2)
        assert W[0, 1] == pytest.approx(chain_12 + 1.0)
        # 1-3 is not a chain neighbor but IS in the community
        assert W[0, 2] == pytest.approx(1.0)

    def test_dampened_edges(self):
        maturities = np.array([1.0, 2.0, 3.0])
        W_base, _, _ = tenor_chain_graph(maturities)
        W_damp, _, _ = hierarchical_graph(
            maturities,
            communities={},
            dampened_edges=[(2.0, 3.0, 0.5)],
        )
        assert W_damp[1, 2] == pytest.approx(W_base[1, 2] * 0.5)

    def test_symmetry_and_psd(self):
        maturities = np.array([1, 2, 5, 7, 8, 9, 10, 20], dtype=float)
        _, _, Ltilde = hierarchical_graph(maturities)
        assert np.allclose(Ltilde, Ltilde.T, atol=1e-14)
        evals = np.linalg.eigvalsh(Ltilde)
        assert np.all(evals >= -1e-12)
