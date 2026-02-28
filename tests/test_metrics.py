"""Tests for grpca.metrics — reconstruction loss, subspace distance, roughness."""

from __future__ import annotations

import numpy as np
import pytest

from grpca.metrics import (
    component_distances,
    component_distances_procrustes,
    eigengap,
    example_roughness,
    feature_roughness,
    masked_reconstruction_loss,
    normalized_reconstruction_difference,
    orthonormal_basis,
    procrustes_distance,
    reconstruction_loss,
    spectral_data_scale,
    subspace_distance,
    subspace_projector,
)

# ---------------------------------------------------------------------------
# reconstruction_loss
# ---------------------------------------------------------------------------


class TestReconstructionLoss:
    def test_zero_loss_perfect_fit(self):
        """If V spans the data, loss should be 0."""
        rng = np.random.default_rng(1)
        V = np.linalg.qr(rng.normal(size=(6, 3)))[0]
        Z = rng.normal(size=(20, 3)) @ V.T  # exactly in span(V)
        loss = reconstruction_loss(Z, V)
        assert loss == pytest.approx(0.0, abs=1e-10)

    def test_per_obs_shape(self):
        rng = np.random.default_rng(2)
        Z = rng.normal(size=(15, 8))
        V = np.linalg.qr(rng.normal(size=(8, 3)))[0]
        per = reconstruction_loss(Z, V, per_obs=True)
        assert per.shape == (15,)
        assert np.all(per >= 0)

    def test_per_obs_mean_matches_aggregate(self):
        rng = np.random.default_rng(3)
        Z = rng.normal(size=(30, 5))
        V = np.linalg.qr(rng.normal(size=(5, 2)))[0]
        agg = reconstruction_loss(Z, V)
        per = reconstruction_loss(Z, V, per_obs=True)
        assert float(np.mean(per)) == pytest.approx(agg, rel=1e-8)


# ---------------------------------------------------------------------------
# masked_reconstruction_loss
# ---------------------------------------------------------------------------


class TestMaskedReconstructionLoss:
    def test_no_missing_matches_dense(self):
        rng = np.random.default_rng(4)
        Z = rng.normal(size=(20, 6))
        U = rng.normal(size=(20, 2))
        V = rng.normal(size=(6, 2))
        mask = np.ones_like(Z, dtype=bool)

        masked = masked_reconstruction_loss(Z, U, V, mask)
        manual = float(np.mean((Z - U @ V.T) ** 2))
        assert masked == pytest.approx(manual, rel=1e-8)

    def test_ignores_masked_entries(self):
        Z = np.array([[1.0, 2.0], [3.0, 4.0]])
        U = np.array([[0.0], [0.0]])
        V = np.array([[1.0], [1.0]])
        mask = np.array([[True, False], [False, True]])
        # observed: (0,0)=1.0 and (1,1)=4.0, pred all 0
        expected = float((1.0**2 + 4.0**2) / 2)
        assert masked_reconstruction_loss(Z, U, V, mask) == pytest.approx(expected)

    def test_per_obs_nans_for_empty_rows(self):
        Z = np.array([[1.0, 2.0], [np.nan, np.nan]])
        mask = np.isfinite(Z)
        U = np.zeros((2, 1))
        V = np.ones((2, 1))
        per = masked_reconstruction_loss(Z, U, V, mask, per_obs=True)
        assert per.shape == (2,)
        assert np.isfinite(per[0])
        assert np.isnan(per[1])

    def test_empty_mask_returns_nan(self):
        Z = np.zeros((3, 3))
        mask = np.zeros_like(Z, dtype=bool)
        assert np.isnan(masked_reconstruction_loss(Z, np.zeros((3, 1)), np.zeros((3, 1)), mask))


# ---------------------------------------------------------------------------
# spectral_data_scale
# ---------------------------------------------------------------------------


class TestSpectralDataScale:
    def test_positive(self):
        rng = np.random.default_rng(5)
        Z = rng.normal(size=(30, 8))
        mask = np.ones_like(Z, dtype=bool)
        s = spectral_data_scale(Z, mask)
        assert s > 0

    def test_identity_data(self):
        """For identity matrix, sigma_1^2 = 1."""
        Z = np.eye(4)
        mask = np.ones_like(Z, dtype=bool)
        s = spectral_data_scale(Z, mask)
        assert s == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# roughness metrics
# ---------------------------------------------------------------------------


class TestRoughness:
    def test_feature_roughness_zero_for_constant_loading(self):
        """Constant loadings have zero difference → zero roughness."""
        V = np.ones((5, 2)) / np.sqrt(5)
        # chain Laplacian
        w = np.zeros((5, 5))
        for i in range(4):
            w[i, i + 1] = w[i + 1, i] = 1.0
        L = np.diag(w.sum(1)) - w
        assert feature_roughness(V, L) == pytest.approx(0.0, abs=1e-12)

    def test_example_roughness_nonnegative(self):
        rng = np.random.default_rng(6)
        U = rng.normal(size=(10, 3))
        w = np.zeros((10, 10))
        for i in range(9):
            w[i, i + 1] = w[i + 1, i] = 1.0
        L = np.diag(w.sum(1)) - w
        assert example_roughness(U, L) >= 0.0

    def test_incompatible_shape_raises(self):
        with pytest.raises(ValueError):
            feature_roughness(np.ones((3, 2)), np.eye(4))


# ---------------------------------------------------------------------------
# procrustes_distance
# ---------------------------------------------------------------------------


class TestProcrustesDistance:
    def test_identical_matrices(self):
        V = np.eye(5, 3)
        assert procrustes_distance(V, V) == pytest.approx(0.0, abs=1e-12)

    def test_orthogonal_rotation_invariant(self):
        """Procrustes distance should be zero for orthogonally rotated copies."""
        rng = np.random.default_rng(7)
        V = np.linalg.qr(rng.normal(size=(8, 3)))[0]
        R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        V_rot = V @ R
        assert procrustes_distance(V, V_rot) == pytest.approx(0.0, abs=1e-8)

    def test_different_matrices_positive(self):
        rng = np.random.default_rng(8)
        A = rng.normal(size=(6, 2))
        B = rng.normal(size=(6, 2)) + 5.0
        assert procrustes_distance(A, B) > 0.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            procrustes_distance(np.ones((3, 2)), np.ones((4, 2)))


# ---------------------------------------------------------------------------
# subspace_distance
# ---------------------------------------------------------------------------


class TestSubspaceDistance:
    def test_same_subspace_zero(self):
        V = np.eye(5, 3)
        assert subspace_distance(V, V, k=3) == pytest.approx(0.0, abs=1e-10)

    def test_orthogonal_subspaces_one(self):
        """Orthogonal subspaces should have sin(theta_max) = 1."""
        V1 = np.eye(6)[:, :3]
        V2 = np.eye(6)[:, 3:6]  # columns 3, 4, 5
        d = subspace_distance(V1, V2, k=3)
        assert d == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# component_distances
# ---------------------------------------------------------------------------


class TestComponentDistances:
    def test_identical_columns_zero(self):
        V = np.eye(5, 3)
        d = component_distances(V, V, k=3)
        assert np.allclose(d, 0.0, atol=1e-12)

    def test_sign_invariant(self):
        V = np.eye(5, 3)
        V_neg = -V
        d = component_distances(V, V_neg, k=3)
        assert np.allclose(d, 0.0, atol=1e-12)


class TestComponentDistancesProcrustes:
    def test_same_matrix_zero(self):
        V = np.eye(6, 3)
        d = component_distances_procrustes(V, V, k=3)
        assert np.allclose(d, 0.0, atol=1e-12)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            component_distances_procrustes(np.ones((3, 2)), np.ones((4, 2)), k=2)


# ---------------------------------------------------------------------------
# orthonormal_basis / subspace_projector
# ---------------------------------------------------------------------------


class TestOrthonormalBasis:
    def test_orthonormality(self):
        rng = np.random.default_rng(9)
        V = rng.normal(size=(8, 3))
        Q = orthonormal_basis(V)
        assert Q.shape == (8, 3)
        assert np.allclose(Q.T @ Q, np.eye(3), atol=1e-12)

    def test_projector_idempotent(self):
        rng = np.random.default_rng(10)
        V = rng.normal(size=(6, 2))
        P = subspace_projector(V)
        assert np.allclose(P @ P, P, atol=1e-12)


# ---------------------------------------------------------------------------
# eigengap
# ---------------------------------------------------------------------------


class TestEigengap:
    def test_clear_gap(self):
        evals = np.array([10.0, 5.0, 1.0, 0.5])
        gap, gap_raw, small = eigengap(evals, k=2)
        assert gap_raw == pytest.approx(4.0)
        assert gap == pytest.approx(4.0)
        assert small is False

    def test_no_gap_returns_eps(self):
        evals = np.array([1.0, 1.0, 1.0])
        gap, gap_raw, small = eigengap(evals, k=2)
        assert gap_raw == pytest.approx(0.0)
        assert small is True

    def test_k_exceeds_length(self):
        evals = np.array([5.0, 3.0])
        gap, _, small = eigengap(evals, k=3)
        assert small is True


# ---------------------------------------------------------------------------
# normalized_reconstruction_difference
# ---------------------------------------------------------------------------


class TestNormalizedReconstructionDifference:
    def test_identical_reconstructions_zero(self):
        X = np.ones((5, 4))
        mask = np.ones_like(X, dtype=bool)
        d = normalized_reconstruction_difference(X, X, X, mask)
        assert d == pytest.approx(0.0, abs=1e-12)

    def test_empty_mask_nan(self):
        X = np.ones((3, 3))
        mask = np.zeros_like(X, dtype=bool)
        assert np.isnan(normalized_reconstruction_difference(X, X, X + 1, mask))
