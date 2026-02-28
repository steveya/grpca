"""Tests for grpca.inference — graph learning and comparison utilities."""

from __future__ import annotations

import numpy as np
import pytest

from grpca.inference import (
    edge_list_topM,
    infer_precision_graph,
    jaccard_topM,
    nonzero_edge_count,
    spearman_union_support,
)

# ---------------------------------------------------------------------------
# infer_precision_graph
# ---------------------------------------------------------------------------


class TestInferPrecisionGraph:
    def test_returns_three_outputs(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(100, 5))
        Theta, W, method = infer_precision_graph(X)
        assert Theta.shape == (5, 5)
        assert W.shape == (5, 5)
        assert isinstance(method, str)

    def test_w_has_zero_diagonal(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(80, 4))
        _, W, _ = infer_precision_graph(X)
        assert np.allclose(np.diag(W), 0.0)

    def test_w_nonnegative(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(60, 4))
        _, W, _ = infer_precision_graph(X)
        assert np.all(W >= 0)

    def test_correlated_structure(self):
        """Strongly correlated pairs should have higher precision entries."""
        rng = np.random.default_rng(4)
        n = 200
        z1 = rng.normal(size=n)
        z2 = rng.normal(size=n)
        X = np.column_stack([z1, z1 + 0.1 * rng.normal(size=n), z2, z2 + 0.1 * rng.normal(size=n)])
        _, W, _ = infer_precision_graph(X)
        # (0,1) and (2,3) should have the strongest edges
        strong = [W[0, 1], W[2, 3]]
        weak = [W[0, 2], W[0, 3], W[1, 2], W[1, 3]]
        assert min(strong) > max(weak) * 0.5  # robust check


# ---------------------------------------------------------------------------
# edge_list_topM
# ---------------------------------------------------------------------------


class TestEdgeListTopM:
    def test_correct_number(self):
        W = np.array([[0, 3, 1], [3, 0, 2], [1, 2, 0]], dtype=float)
        top = edge_list_topM(W, M=2)
        assert len(top) == 2

    def test_sorted_descending(self):
        W = np.array([[0, 3, 1, 5], [3, 0, 2, 0], [1, 2, 0, 4], [5, 0, 4, 0]], dtype=float)
        top = edge_list_topM(W, M=6)
        weights = [t[2] for t in top]
        assert weights == sorted(weights, reverse=True)

    def test_m_larger_than_edges(self):
        W = np.array([[0, 1], [1, 0]], dtype=float)
        top = edge_list_topM(W, M=10)
        assert len(top) == 1


# ---------------------------------------------------------------------------
# nonzero_edge_count
# ---------------------------------------------------------------------------


class TestNonzeroEdgeCount:
    def test_chain_graph(self):
        p = 5
        W = np.zeros((p, p))
        for i in range(p - 1):
            W[i, i + 1] = W[i + 1, i] = 1.0
        assert nonzero_edge_count(W) == p - 1

    def test_complete_graph(self):
        p = 4
        W = np.ones((p, p)) - np.eye(p)
        assert nonzero_edge_count(W) == p * (p - 1) // 2

    def test_empty_graph(self):
        assert nonzero_edge_count(np.zeros((5, 5))) == 0


# ---------------------------------------------------------------------------
# jaccard_topM
# ---------------------------------------------------------------------------


class TestJaccardTopM:
    def test_identical_graphs_one(self):
        W = np.array([[0, 3, 1], [3, 0, 2], [1, 2, 0]], dtype=float)
        j = jaccard_topM(W, W, M=2)
        assert j == pytest.approx(1.0)

    def test_disjoint_graphs_zero(self):
        W1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
        W2 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
        j = jaccard_topM(W1, W2, M=1)
        assert j == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# spearman_union_support
# ---------------------------------------------------------------------------


class TestSpearmanUnionSupport:
    def test_identical_graphs_one(self):
        W = np.array([[0, 3, 1], [3, 0, 2], [1, 2, 0]], dtype=float)
        r = spearman_union_support(W, W, M=3)
        assert r == pytest.approx(1.0, abs=1e-8)

    def test_small_union_nan(self):
        W1 = np.array([[0, 1], [1, 0]], dtype=float)
        W2 = np.array([[0, 1], [1, 0]], dtype=float)
        r = spearman_union_support(W1, W2, M=1)
        # union has 1 edge — too small for Spearman
        assert np.isnan(r)
