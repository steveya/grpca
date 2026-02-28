import numpy as np
import pytest

from grpca.estimator import GraphRegularizedPCA
from grpca.graphs import day_chain_graph, spectral_normalize_laplacian
from grpca.metrics import (
    example_roughness,
    feature_roughness,
    masked_reconstruction_loss,
)
from grpca.splitters import NestedTimeSeriesSplit


def _colwise_nanstd_or_one(x: np.ndarray) -> np.ndarray:
    p = x.shape[1]
    out = np.ones(p, dtype=float)
    for j in range(p):
        vals = x[np.isfinite(x[:, j]), j]
        if vals.size < 2:
            out[j] = 1.0
            continue
        sd = float(np.nanstd(vals, ddof=1))
        out[j] = sd if np.isfinite(sd) and sd >= 1e-10 else 1.0
    return out


def _make_feature_chain_laplacian(p: int) -> np.ndarray:
    w = np.zeros((p, p), dtype=float)
    for i in range(p - 1):
        w[i, i + 1] = 1.0
        w[i + 1, i] = 1.0
    d = np.diag(w.sum(axis=1))
    l = d - w
    return spectral_normalize_laplacian(l)


def _make_example_chain_laplacian(n: int) -> np.ndarray:
    _, _, ltilde = day_chain_graph(np.arange(n), normalize=True)
    return ltilde


def _rng() -> np.random.Generator:
    return np.random.default_rng(7)


def test_fit_standardization_uses_observed_train_entries_only():
    rng = _rng()
    n_train, n_test, p = 70, 30, 6

    x_train = rng.normal(size=(n_train, p))
    x_test = rng.normal(size=(n_test, p)) + 8.0

    train_missing = rng.random((n_train, p)) < 0.15
    test_missing = rng.random((n_test, p)) < 0.20
    x_train[train_missing] = np.nan
    x_test[test_missing] = np.nan

    model = GraphRegularizedPCA(n_components=3, max_iter=200, random_state=0)
    model.fit(x_train)

    expected_mean = np.nanmean(x_train, axis=0)
    expected_scale = _colwise_nanstd_or_one(x_train)

    assert np.allclose(model.mean_, expected_mean, atol=1e-12, rtol=0.0)
    assert np.allclose(model.scale_, expected_scale, atol=1e-12, rtol=0.0)

    z_test = (x_test - expected_mean) / expected_scale
    mask_test = np.isfinite(x_test)
    scores = model.transform(x_test)
    pred_std = scores @ model.V_.T
    resid = np.where(mask_test, pred_std - z_test, 0.0)
    manual = float(np.sum(resid * resid) / np.sum(mask_test))

    model_loss = float(model.reconstruction_error(x_test))
    assert model_loss == pytest.approx(manual, rel=0.0, abs=1e-10)


def test_cv_inner_scaler_uses_inner_train_only():
    rng = _rng()
    n, p = 420, 5
    dates = np.arange(np.datetime64("2020-01-01"), np.datetime64("2021-02-24"))[:n]

    x = rng.normal(size=(n, p))
    split = NestedTimeSeriesSplit(inner_duration="270D")
    inner_idx, val_idx = split.split(x, dates=dates)

    x[val_idx] += 6.0
    miss = rng.random((n, p)) < 0.10
    x[miss] = np.nan

    model = GraphRegularizedPCA(
        n_components=2,
        cv="nested_ts",
        tau_grid=np.array([0.0, 1e-3]),
        rho_grid=np.array([0.0, 0.5, 1.0]),
        random_state=0,
        max_iter=120,
    )
    model.fit(x, dates=dates)

    cv = model.cv_results_
    assert cv is not None

    x_inner = x[inner_idx]
    expected_mean = np.nanmean(x_inner, axis=0)
    expected_scale = _colwise_nanstd_or_one(x_inner)

    assert np.allclose(cv["inner_mean"], expected_mean, atol=1e-10, rtol=0.0)
    assert np.allclose(cv["inner_scale"], expected_scale, atol=1e-10, rtol=0.0)

    full_mean = np.nanmean(x, axis=0)
    assert float(np.linalg.norm(cv["inner_mean"] - full_mean)) > 1e-2


def test_masked_reconstruction_ignores_nans():
    z = np.array(
        [
            [1.0, np.nan, 2.0],
            [0.0, 1.0, np.nan],
        ],
        dtype=float,
    )
    mask = np.isfinite(z)
    u = np.array([[1.0], [2.0]], dtype=float)
    v = np.array([[0.5], [1.5], [0.25]], dtype=float)

    pred = u @ v.T
    resid = np.where(mask, pred - z, 0.0)
    manual = float(np.sum(resid * resid) / np.sum(mask))

    metric_loss = float(masked_reconstruction_loss(z, u, v, mask))
    assert metric_loss == pytest.approx(manual, rel=0.0, abs=1e-12)

    x = np.array(
        [
            [1.0, np.nan, 2.0],
            [0.0, 1.0, np.nan],
            [0.5, 0.8, 0.2],
        ],
        dtype=float,
    )
    model = GraphRegularizedPCA(n_components=1, random_state=0, max_iter=200)
    model.fit(x)
    z_fit = model._standardize(x)
    scores = model.transform(x)
    manual_model = float(
        masked_reconstruction_loss(z_fit, scores, model.V_, np.isfinite(z_fit))
    )
    assert float(model.reconstruction_error(x)) == pytest.approx(
        manual_model, rel=0.0, abs=1e-10
    )


def test_transform_with_missing_solves_scores_from_observed_entries():
    rng = _rng()
    x = rng.normal(size=(60, 8))
    x[rng.random(x.shape) < 0.2] = np.nan

    model = GraphRegularizedPCA(n_components=3, random_state=0, max_iter=160)
    model.fit(x)

    x_new = rng.normal(size=(12, 8))
    x_new[rng.random(x_new.shape) < 0.3] = np.nan

    scores = model.transform(x_new)
    assert scores.shape == (12, 3)
    assert np.all(np.isfinite(scores))


def test_transform_dense_rows_matches_z_times_v():
    rng = _rng()
    x = rng.normal(size=(80, 7))

    model = GraphRegularizedPCA(
        n_components=3,
        random_state=0,
        max_iter=180,
        mu_u=1e-6,
    )
    model.fit(x)

    x_new = rng.normal(size=(15, 7))
    z_new = model._standardize(x_new)
    scores = model.transform(x_new)
    expected = z_new @ model.V_

    assert scores.shape == expected.shape
    assert np.allclose(scores, expected, atol=1e-10, rtol=0.0)


def test_reconstruction_error_per_obs_respects_observed_mask():
    rng = _rng()
    x = rng.normal(size=(90, 6))
    x[rng.random(x.shape) < 0.2] = np.nan

    model = GraphRegularizedPCA(n_components=2, random_state=0, max_iter=150)
    model.fit(x)

    x_new = rng.normal(size=(10, 6))
    x_new[rng.random(x_new.shape) < 0.25] = np.nan
    x_new[0, :] = np.nan

    per_obs = model.reconstruction_error(x_new, per_obs=True)
    mask = np.isfinite(x_new)
    obs_per_row = np.sum(mask, axis=1)

    assert per_obs.shape == (x_new.shape[0],)
    assert np.isnan(per_obs[obs_per_row == 0]).all()
    assert np.isfinite(per_obs[obs_per_row > 0]).all()


def test_feature_graph_only_runs_and_loading_roughness_finite():
    rng = _rng()
    x = rng.normal(size=(90, 7))
    x[rng.random(x.shape) < 0.1] = np.nan
    lf = _make_feature_chain_laplacian(x.shape[1])

    model = GraphRegularizedPCA(
        n_components=2,
        graph=lf,
        lambda_f=1e-2,
        lambda_s=0.0,
        random_state=0,
        max_iter=150,
    )
    model.fit(x)
    rough = feature_roughness(model.V_, model.L_f_tilde_)
    assert np.isfinite(rough)


def test_example_graph_only_runs_and_score_roughness_finite():
    rng = _rng()
    x = rng.normal(size=(80, 6))
    x[rng.random(x.shape) < 0.1] = np.nan
    ls = _make_example_chain_laplacian(x.shape[0])

    model = GraphRegularizedPCA(
        n_components=2,
        example_graph=ls,
        lambda_s=1e-2,
        lambda_f=0.0,
        random_state=0,
        max_iter=150,
    )
    model.fit(x)
    rough = example_roughness(model.U_, model.L_s_tilde_)
    assert np.isfinite(rough)


def test_both_graphs_runs_and_is_finite():
    rng = _rng()
    x = rng.normal(size=(85, 6))
    x[rng.random(x.shape) < 0.12] = np.nan

    model = GraphRegularizedPCA(
        n_components=2,
        graph=_make_feature_chain_laplacian(x.shape[1]),
        example_graph=_make_example_chain_laplacian(x.shape[0]),
        lambda_s=5e-3,
        lambda_f=5e-3,
        random_state=0,
        max_iter=180,
    )
    model.fit(x)

    assert np.isfinite(model.reconstruction_error(x))
    assert np.all(np.isfinite(model.U_))
    assert np.all(np.isfinite(model.V_))


def test_no_graphs_runs_as_low_rank_factorization():
    rng = _rng()
    x = rng.normal(size=(70, 5))
    model = GraphRegularizedPCA(
        n_components=2,
        graph=None,
        example_graph=None,
        tau=0.0,
        rho=0.0,
        random_state=0,
        max_iter=160,
    )
    model.fit(x)
    assert model.has_feature_graph_ is False
    assert model.has_example_graph_ is False
    assert np.isfinite(model.reconstruction_error(x))


def test_objective_trace_nonincreasing_up_to_small_tolerance():
    rng = _rng()
    x = rng.normal(size=(75, 7))
    x[rng.random(x.shape) < 0.1] = np.nan

    model = GraphRegularizedPCA(n_components=2, random_state=0, max_iter=180)
    model.fit(x)

    diffs = np.diff(model.objective_trace_)
    assert np.all(diffs <= 1e-8)


def test_cv_selects_tau_rho_from_grid():
    rng = _rng()
    x = rng.normal(size=(120, 6))
    x[rng.random(x.shape) < 0.1] = np.nan

    tau_grid = np.array([0.0, 1e-3, 1e-2], dtype=float)
    rho_grid = np.array([0.0, 0.5, 1.0], dtype=float)

    model = GraphRegularizedPCA(
        n_components=2,
        cv="nested",
        tau_grid=tau_grid,
        rho_grid=rho_grid,
        random_state=0,
        max_iter=120,
    )
    model.fit(x)

    assert np.any(np.isclose(model.tau_, tau_grid, atol=1e-15, rtol=0.0))
    assert np.any(np.isclose(model.rho_, rho_grid, atol=1e-15, rtol=0.0))


def test_cv_candidate_rule_prefers_smaller_tau_on_flat_surface():
    rng = _rng()
    x = rng.normal(size=(130, 6))

    tau_grid = np.array([0.0, 1e-3, 1e-2], dtype=float)
    rho_grid = np.array([0.0, 0.5, 1.0], dtype=float)

    model = GraphRegularizedPCA(
        n_components=2,
        graph=np.zeros((x.shape[1], x.shape[1]), dtype=float),
        cv="nested",
        tau_grid=tau_grid,
        rho_grid=rho_grid,
        delta=1.0,
        random_state=0,
        max_iter=80,
    )
    model.fit(x)

    assert model.tau_ == pytest.approx(float(np.min(tau_grid)), rel=0.0, abs=1e-15)


def test_no_graph_no_missing_approximates_low_rank_svd():
    rng = _rng()
    x = rng.normal(size=(70, 9))
    k = 3

    model = GraphRegularizedPCA(
        n_components=k,
        graph=None,
        example_graph=None,
        lambda_s=0.0,
        lambda_f=0.0,
        mu_u=1e-8,
        mu_v=1e-8,
        max_iter=600,
        tol=1e-9,
        random_state=0,
    )
    model.fit(x)

    z = model._standardize(x)
    scores = model.transform(x)
    manual = float(np.mean((z - scores @ model.V_.T) ** 2))

    model_err = float(model.reconstruction_error(x))
    assert model_err == pytest.approx(manual, rel=0.0, abs=1e-10)


def test_day_chain_graph_shapes_and_symmetry():
    dates = np.array(
        [
            np.datetime64("2021-01-01"),
            np.datetime64("2021-01-02"),
            np.datetime64("2021-01-05"),
            np.datetime64("2021-01-06"),
        ]
    )
    w, l, ltilde = day_chain_graph(dates)

    assert w.shape == (4, 4)
    assert l.shape == (4, 4)
    assert ltilde.shape == (4, 4)
    assert np.allclose(w, w.T, atol=1e-12, rtol=0.0)
    assert np.allclose(l, l.T, atol=1e-12, rtol=0.0)
    assert np.allclose(ltilde, ltilde.T, atol=1e-12, rtol=0.0)


def test_spectral_normalize_laplacian_has_unit_top_eigenvalue_for_nonzero_graph():
    w = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    d = np.diag(w.sum(axis=1))
    l = d - w
    ltilde = spectral_normalize_laplacian(l)
    top = float(np.max(np.linalg.eigvalsh(ltilde)))
    assert top == pytest.approx(1.0, rel=0.0, abs=1e-10)
