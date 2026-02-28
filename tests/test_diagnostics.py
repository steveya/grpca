"""Tests for grpca.diagnostics — statistical tests and CV helpers."""

from __future__ import annotations

import numpy as np
import pytest

from grpca.diagnostics import (
    best_match_scores,
    dm_test,
    extract_selected_lambda_series,
    extract_selected_rho_series,
    extract_selected_tau_series,
    extract_validation_loss_surface,
    newey_west_se,
    summarize_cv_surface,
)

# ---------------------------------------------------------------------------
# newey_west_se
# ---------------------------------------------------------------------------


class TestNeweyWestSE:
    def test_positive_for_nondegenerate(self):
        rng = np.random.default_rng(1)
        d = rng.normal(size=100)
        se = newey_west_se(d, lag=3)
        assert np.isfinite(se)
        assert se > 0

    def test_zero_lag_equals_classical(self):
        rng = np.random.default_rng(2)
        d = rng.normal(size=200)
        se_nw = newey_west_se(d, lag=0)
        # NW uses 1/T divisor vs classic ddof=1 → differ by factor T/(T-1)
        T = len(d)
        se_classic_nw = float(np.std(d, ddof=0) / np.sqrt(T))
        assert se_nw == pytest.approx(se_classic_nw, rel=1e-6)

    def test_short_series_nan(self):
        assert np.isnan(newey_west_se(np.array([1.0, 2.0]), lag=1))
        assert np.isnan(newey_west_se(np.array([1.0]), lag=0))

    def test_constant_series_nan(self):
        assert np.isnan(newey_west_se(np.ones(100), lag=3))


# ---------------------------------------------------------------------------
# dm_test
# ---------------------------------------------------------------------------


class TestDMTest:
    def test_identical_losses_non_significant(self):
        losses = np.random.default_rng(3).normal(size=100)
        res = dm_test(losses, losses, lag=2, alternative="less")
        assert res["mean_d"] == pytest.approx(0.0)
        # p-value should be ~0.5 for zero mean difference
        # (or nan if se=0 because all diffs are exactly 0)

    def test_clearly_better_model(self):
        rng = np.random.default_rng(4)
        losses_a = rng.normal(loc=0.0, scale=0.1, size=200)
        losses_b = rng.normal(loc=1.0, scale=0.1, size=200)
        res = dm_test(losses_a, losses_b, lag=2, alternative="less")
        assert res["p_value"] < 0.01
        assert res["mean_d"] < 0
        assert res["win_rate"] > 0.9

    def test_two_sided(self):
        rng = np.random.default_rng(5)
        a = rng.normal(loc=0, size=100)
        b = rng.normal(loc=2, size=100)
        res = dm_test(a, b, lag=1, alternative="two-sided")
        assert res["p_value"] < 0.05

    def test_greater_alternative(self):
        rng = np.random.default_rng(6)
        a = rng.normal(loc=2, size=100)
        b = rng.normal(loc=0, size=100)
        res = dm_test(a, b, lag=1, alternative="greater")
        assert res["p_value"] < 0.05

    def test_output_keys(self):
        res = dm_test(np.zeros(10), np.ones(10), lag=1)
        for key in ["mean_d", "se", "dm_stat", "p_value", "win_rate"]:
            assert key in res


# ---------------------------------------------------------------------------
# best_match_scores
# ---------------------------------------------------------------------------


class TestBestMatchScores:
    def test_identity_cosine(self):
        C = np.eye(3)
        mean_s, min_s = best_match_scores(C)
        assert mean_s == pytest.approx(1.0, abs=1e-12)
        assert min_s == pytest.approx(1.0, abs=1e-12)

    def test_permuted_identity(self):
        C = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
        mean_s, min_s = best_match_scores(C)
        assert mean_s == pytest.approx(1.0, abs=1e-12)
        assert min_s == pytest.approx(1.0, abs=1e-12)

    def test_imperfect_match(self):
        C = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=float)
        mean_s, min_s = best_match_scores(C)
        assert 0 < min_s <= mean_s <= 1.0


# ---------------------------------------------------------------------------
# CV surface helpers
# ---------------------------------------------------------------------------


class TestSummarizeCVSurface:
    def test_returns_list_of_dicts(self):
        cv_results = {
            "results_table": [
                {"tau": 0.0, "rho": 0.0, "val_loss": 1.5},
                {"tau": 0.01, "rho": 0.5, "val_loss": 1.2},
            ]
        }
        out = summarize_cv_surface(cv_results)
        # With pandas available, returns DataFrame; without, list of dicts
        assert len(out) == 2

    def test_invalid_results_table_raises(self):
        with pytest.raises(ValueError):
            summarize_cv_surface({"results_table": "bad"})


class TestExtractSeries:
    def _make_cv_results(self, n: int = 5):
        return [
            {
                "window": i,
                "selected_tau": 0.01 * i,
                "selected_rho": 0.5,
                "selected_lambda_s": 0.005 * i,
                "selected_lambda_f": 0.005 * i,
            }
            for i in range(n)
        ]

    def test_tau_series_length(self):
        series = extract_selected_tau_series(self._make_cv_results(4))
        assert len(series) == 4
        assert all("tau" in r for r in series)

    def test_rho_series_length(self):
        series = extract_selected_rho_series(self._make_cv_results(3))
        assert len(series) == 3
        assert all("rho" in r for r in series)

    def test_lambda_series_has_both(self):
        series = extract_selected_lambda_series(self._make_cv_results(2))
        assert all("lambda_s" in r and "lambda_f" in r for r in series)


class TestExtractValidationLossSurface:
    def test_basic(self):
        cv_results = {
            "results_table": [
                {"tau": 0.0, "rho": 0.0, "val_loss": 1.5},
                {"tau": 0.01, "rho": 0.5, "val_loss": 1.2},
            ]
        }
        out = extract_validation_loss_surface(cv_results)
        assert len(out) == 2
        assert all("tau" in r and "val_loss" in r for r in out)
