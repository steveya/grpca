# grpca

**Graph-Regularized PCA** — an sklearn-compatible estimator for low-rank matrix factorization with two-sided Laplacian penalties and native missing-data support.

---

## Why grpca?

Standard PCA treats features as exchangeable and observations as i.i.d.
In many applications — yield curves, spatial data, sensor networks — you *know*
that neighbouring features or consecutive observations should behave similarly.

**grpca** lets you encode that knowledge as graph Laplacians and inject it
directly into the low-rank factorization objective, producing smoother,
more interpretable factors while retaining the nonparametric flexibility of PCA.

### Key Features

- **Two-sided graph regularization** — penalize roughness on features *and* observations simultaneously via graph Laplacian quadratic forms.
- **Missing data** — pass `np.nan` entries; the masked GLRM objective handles them natively.
- **Sklearn-compatible** — `BaseEstimator` / `TransformerMixin` interface with `fit()`, `transform()`, `inverse_transform()`, `components_`.
- **Nested time-series CV** — automatic hyperparameter selection with leakage-safe inner scaling and conservative tie-breaking.
- **Spectral normalization** — Laplacians are normalized so that the same `tau` grid works portably across datasets and window sizes.
- **Rolling-window infrastructure** — `RollingWindowSplitter` and `NestedTimeSeriesSplit` for production backtesting.
- **Rich diagnostics** — Diebold–Mariano tests, Procrustes distances, roughness metrics, and CV surface summaries.

## Quick Start

```python
import numpy as np
from grpca import GraphRegularizedPCA
from grpca.graphs import tenor_chain_graph

# Synthetic yield-curve-like data (100 days × 6 tenors)
rng = np.random.default_rng(0)
X = rng.normal(size=(100, 6))
maturities = np.array([1, 2, 3, 5, 7, 10], dtype=float)

# Build a tenor-chain graph
W, L, Ltilde = tenor_chain_graph(maturities)

# Fit with feature-graph regularization
model = GraphRegularizedPCA(
    n_components=3,
    graph=Ltilde,
    tau=0.01,
    rho=0.0,
)
model.fit(X)

print("Loadings shape:", model.V_.shape)    # (6, 3)
print("Selected tau:", model.tau_)           # 0.01
```

## Installation

```bash
pip install -e .          # editable install
pip install -e .[test]    # with test dependencies
pip install -e .[docs]    # with documentation tools
pip install -e .[dev]     # everything
```

## Documentation

Full documentation is available at [grpca.readthedocs.io](https://grpca.readthedocs.io).

## License

MIT
