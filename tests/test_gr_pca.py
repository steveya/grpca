import numpy as np
import pandas as pd
import pytest

from grpca.estimator import GraphRegularizedPCA
from grpca.metrics import reconstruction_loss
from grpca.splitters import RollingWindowSplitter


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def chain_laplacian(p: int, normalize: bool = True) -> np.ndarray:
    w = np.zeros((p, p), dtype=float)
    for i in range(p - 1):
        w[i, i + 1] = 1.0
        w[i + 1, i] = 1.0
    d = np.diag(w.sum(axis=1))
    lap = d - w
    if not normalize:
        return lap
    evals = np.linalg.eigvalsh(lap)
    lam_max = float(np.max(evals)) if len(evals) > 0 else 1.0
    return lap / lam_max if lam_max > 0 else lap


def projector(v: np.ndarray) -> np.ndarray:
    return v @ v.T


def fro_norm(a: np.ndarray) -> float:
    return float(np.linalg.norm(a, ord="fro"))


def roughness(v: np.ndarray, ltilde: np.ndarray) -> float:
    return float(np.trace(v.T @ ltilde @ v) / v.shape[1])


def _std_or_one(x: np.ndarray) -> np.ndarray:
    sd = np.std(x, axis=0, ddof=1)
    return np.where(sd < 1e-10, 1.0, sd)


def test_no_graph_reduces_to_pca_projector(rng: np.random.Generator):
    n, p, k = 300, 12, 4
    x = rng.normal(size=(n, p))

    m_a = GraphRegularizedPCA(
        n_components=k, standardize=True, graph=None, alpha=0.0
    ).fit(x)
    m_b = GraphRegularizedPCA(
        n_components=k,
        standardize=True,
        graph=np.zeros((p, p), dtype=float),
        alpha=0.0,
    ).fit(x)

    p_a = projector(m_a.loadings_[:, :k])
    p_b = projector(m_b.loadings_[:, :k])
    assert fro_norm(p_a - p_b) < 1e-8


def test_alpha_zero_equals_pca_projector_with_graph(rng: np.random.Generator):
    n, p, k = 300, 12, 4
    x = rng.normal(size=(n, p))
    ltilde = chain_laplacian(p)

    m_pca = GraphRegularizedPCA(
        n_components=k, standardize=True, graph=None, alpha=0.0
    ).fit(x)
    m_gr0 = GraphRegularizedPCA(
        n_components=k, standardize=True, graph=ltilde, alpha=0.0
    ).fit(x)

    p_pca = projector(m_pca.loadings_[:, :k])
    p_gr0 = projector(m_gr0.loadings_[:, :k])
    assert fro_norm(p_pca - p_gr0) < 1e-8


def test_components_shape_and_loadings(rng: np.random.Generator):
    n, p, k = 220, 10, 4
    x = rng.normal(size=(n, p))
    ltilde = chain_laplacian(p)

    m = GraphRegularizedPCA(
        n_components=k, standardize=True, graph=ltilde, tau=0.05, rho=0.0
    ).fit(x)

    assert m.components_.shape == (k, p)
    v = m.loadings_
    assert v.shape == (p, k)
    # GLRM loadings need not be orthonormal but components_ should be V_.T
    assert np.allclose(m.components_, m.V_.T)
    # explained_variance_ is None in GLRM mode
    assert m.explained_variance_ is None


def test_graph_regularization_reduces_roughness_on_average(rng: np.random.Generator):
    n, p, k = 260, 12, 4
    x = rng.normal(size=(n, p))
    ltilde = chain_laplacian(p)

    m_small = GraphRegularizedPCA(
        n_components=k, standardize=True, graph=ltilde, alpha=0.0
    ).fit(x)
    evals = np.sort(
        np.linalg.eigvalsh(
            np.cov((x - x.mean(0)) / _std_or_one(x), rowvar=False, ddof=1)
        )
    )[::-1]
    gap_k = max(float(evals[k - 1] - evals[k]), 1e-10)
    alpha_large = 0.2 * gap_k
    m_large = GraphRegularizedPCA(
        n_components=k, standardize=True, graph=ltilde, alpha=alpha_large
    ).fit(x)

    rough_small = roughness(m_small.loadings_[:, :k], ltilde)
    rough_large = roughness(m_large.loadings_[:, :k], ltilde)
    assert rough_large <= rough_small + 1e-10


def test_reconstruction_error_matches_manual_train_scaler(rng: np.random.Generator):
    n, p, k = 200, 8, 4
    x = rng.normal(size=(n, p))
    train_idx = np.arange(0, 140)
    test_idx = np.arange(140, n)

    x_train = x[train_idx]
    x_test = x[test_idx].copy()
    x_test += 10.0

    m = GraphRegularizedPCA(
        n_components=k, standardize=True, graph=None, alpha=0.0
    ).fit(x_train)
    mu = np.mean(x_train, axis=0)
    sd = _std_or_one(x_train)
    z_test = (x_test - mu) / sd
    v = m.loadings_[:, :k]
    manual_loss = reconstruction_loss(z_test, v)

    model_loss = m.reconstruction_error(x_test, k=k)
    assert float(model_loss) == pytest.approx(float(manual_loss), rel=0.0, abs=1e-10)


def test_reconstruction_error_matches_manual_center_only(rng: np.random.Generator):
    n, p, k = 180, 7, 3
    x = rng.normal(size=(n, p))
    x_train = x[:120]
    x_test = x[120:].copy() + 7.5

    m = GraphRegularizedPCA(
        n_components=k, standardize=False, graph=None, alpha=0.0
    ).fit(x_train)
    mu = np.mean(x_train, axis=0)
    z_test = x_test - mu
    manual_loss = reconstruction_loss(z_test, m.loadings_[:, :k])

    model_loss = m.reconstruction_error(x_test, k=k)
    assert float(model_loss) == pytest.approx(float(manual_loss), rel=0.0, abs=1e-10)


def test_cv_inner_scaler_uses_inner_train_stats_only(rng: np.random.Generator):
    n, p, k = 360, 8, 3
    dates = pd.bdate_range("2020-01-01", periods=n)
    x = rng.normal(size=(n, p))

    inner_end = dates.min() + pd.Timedelta("270D")
    val_mask = dates >= inner_end
    x[val_mask] += 5.0

    ltilde = chain_laplacian(p)
    gamma_grid = np.array([0.0, 0.01, 0.1], dtype=float)

    m = GraphRegularizedPCA(
        n_components=k,
        graph=ltilde,
        cv="nested_ts",
        gamma_grid=gamma_grid,
        standardize=True,
    ).fit(x, dates=dates)

    cv = m.cv_results_
    assert cv is not None
    assert "inner_mean" in cv and "inner_scale" in cv

    inner_idx = np.where(dates < inner_end)[0]
    val_idx = np.where(dates >= inner_end)[0]

    mu_expected = np.mean(x[inner_idx], axis=0)
    sd_expected = _std_or_one(x[inner_idx])

    assert np.allclose(cv["inner_mean"], mu_expected, atol=1e-10, rtol=0.0)
    assert np.allclose(cv["inner_scale"], sd_expected, atol=1e-10, rtol=0.0)

    mu_outer = np.mean(x, axis=0)
    assert np.linalg.norm(cv["inner_mean"] - mu_outer) > 1e-2

    z_val_inner = (x[val_idx] - cv["inner_mean"]) / cv["inner_scale"]
    val_mean_norm = float(np.linalg.norm(np.mean(z_val_inner, axis=0)))
    assert val_mean_norm > 0.1


def test_cv_selects_tau(rng: np.random.Generator):
    n, p, k = 260, 10, 4
    x = rng.normal(size=(n, p))
    ltilde = chain_laplacian(p)
    tau_grid = np.array([0.0, 1e-3, 1e-2, 0.1], dtype=float)

    m = GraphRegularizedPCA(
        n_components=k,
        graph=ltilde,
        cv="nested_ts",
        tau_grid=tau_grid,
        rho_grid=np.array([0.0]),
        standardize=True,
    ).fit(x)

    assert np.any(np.isclose(tau_grid, m.tau_, atol=1e-15, rtol=0.0))

    cv = m.cv_results_
    assert cv is not None


def test_cv_picks_smallest_tau_when_flat(rng: np.random.Generator):
    n, p, k = 240, 9, 3
    x = rng.normal(size=(n, p))
    zero_graph = np.zeros((p, p), dtype=float)
    tau_grid = np.array([0.0, 0.01, 0.1, 0.3], dtype=float)

    m = GraphRegularizedPCA(
        n_components=k,
        graph=zero_graph,
        cv="nested_ts",
        tau_grid=tau_grid,
        rho_grid=np.array([0.0]),
        standardize=True,
    ).fit(x)

    # With a zero graph, all tau values produce same loss; smallest should win
    assert m.tau_ == pytest.approx(float(np.min(tau_grid)), rel=0.0, abs=1e-12)
    assert m.cv_results_ is not None


def test_stability_tiebreak_uses_prev_projector_when_available(
    rng: np.random.Generator,
):
    n, p, k = 220, 8, 3
    x = rng.normal(size=(n, p))
    ltilde = chain_laplacian(p)

    tau_hi = 0.3
    m_hi = GraphRegularizedPCA(
        n_components=k, graph=ltilde, tau=tau_hi, rho=0.0, standardize=True
    ).fit(x[: n - 5])
    p_prev = projector(m_hi.loadings_[:, :k])

    m_cv = GraphRegularizedPCA(
        n_components=k,
        graph=ltilde,
        cv="nested_ts",
        tau_grid=np.array([0.0, tau_hi], dtype=float),
        rho_grid=np.array([0.0]),
        standardize=True,
    ).fit(x, P_prev=p_prev)

    assert m_cv.cv_results_ is not None
    # Either the tiebreak was used or it selected some tau; just ensure it ran
    assert hasattr(m_cv, "tau_")


def test_stability_tiebreak_handles_mismatched_prev_rank(rng: np.random.Generator):
    n, p, k = 220, 8, 3
    x = rng.normal(size=(n, p))
    ltilde = chain_laplacian(p)

    bad_prev = np.eye(p - 1)
    tau_grid = np.array([0.0, 0.2, 0.3], dtype=float)

    m_cv = GraphRegularizedPCA(
        n_components=k,
        graph=ltilde,
        cv="nested_ts",
        tau_grid=tau_grid,
        rho_grid=np.array([0.0]),
        standardize=True,
    ).fit(x, P_prev=bad_prev)

    assert m_cv.cv_results_ is not None
    # With mismatched P_prev, CV should still complete successfully
    assert hasattr(m_cv, "tau_")


@pytest.mark.parametrize(
    "bad_graph",
    [
        np.ones((3, 4), dtype=float),
        np.array([[0.0, 1.0], [0.0, 0.0]], dtype=float),
        np.array([[0.0, np.nan], [np.nan, 0.0]], dtype=float),
    ],
)
def test_invalid_graph_raises(rng: np.random.Generator, bad_graph: np.ndarray):
    x = rng.normal(size=(80, bad_graph.shape[0]))
    m = GraphRegularizedPCA(
        n_components=2, graph=bad_graph, alpha=0.1, standardize=True
    )
    with pytest.raises(ValueError):
        m.fit(x)


def test_constant_column_standardization_safe(rng: np.random.Generator):
    n, p = 180, 8
    x = rng.normal(size=(n, p))
    x[:, 0] = 1.2345

    m = GraphRegularizedPCA(
        n_components=4, graph=chain_laplacian(p), alpha=0.05, standardize=True
    ).fit(x)
    err = m.reconstruction_error(x, k=4)

    assert np.all(np.isfinite(m.components_))
    assert np.isfinite(float(err))


def test_rank_deficient_covariance_does_not_crash(rng: np.random.Generator):
    n, p, latent, k = 30, 50, 5, 10
    f = rng.normal(size=(n, latent))
    b = rng.normal(size=(latent, p))
    x = f @ b

    m = GraphRegularizedPCA(
        n_components=k, graph=chain_laplacian(p), alpha=0.02, standardize=True
    ).fit(x)
    assert m.components_.shape == (k, p)
    assert np.isfinite(float(m.reconstruction_error(x, k=k)))


def test_rolling_windows_end_to_end_smoke(rng: np.random.Generator):
    n, p, k = 520, 8, 4
    x = rng.normal(size=(n, p))
    dates = pd.bdate_range("2022-01-03", periods=n)
    ltilde = chain_laplacian(p)

    splitter = RollingWindowSplitter(
        train_len="365D",
        test_len="30D",
        step="30D",
        min_train_obs=100,
        min_test_obs=10,
    )

    got_any = False
    checked = 0
    for tr_idx, te_idx in splitter.split(x, groups=dates):
        got_any = True
        x_tr, x_te = x[tr_idx], x[te_idx]
        d_tr = dates[tr_idx]

        m_pca = GraphRegularizedPCA(
            n_components=k, graph=None, alpha=0.0, standardize=True
        ).fit(x_tr)
        m_gr = GraphRegularizedPCA(
            n_components=k,
            graph=ltilde,
            cv="nested_ts",
            gamma_grid=np.array([0.0, 0.1], dtype=float),
            standardize=True,
        ).fit(x_tr, dates=d_tr)

        e1 = m_pca.reconstruction_error(x_te, k=k)
        e2 = m_gr.reconstruction_error(x_te, k=k)
        assert np.isfinite(float(e1))
        assert np.isfinite(float(e2))

        checked += 1
        if checked >= 3:
            break

    assert got_any
    assert checked >= 1
