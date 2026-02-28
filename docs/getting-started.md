# Getting Started

This guide walks through installing **grpca** and fitting your first
graph-regularized model.

## Installation

```bash
pip install -e .
```

For development (tests + docs + linting):

```bash
pip install -e .[dev]
```

## Minimal Example

```python
import numpy as np
from grpca import GraphRegularizedPCA
from grpca.graphs import tenor_chain_graph

# 1. Prepare data — daily yield changes on 6 bond tenors
rng = np.random.default_rng(42)
X = rng.normal(size=(200, 6))
maturities = np.array([1, 2, 3, 5, 7, 10], dtype=float)

# 2. Build a feature graph (tenor chain)
W, L, Ltilde = tenor_chain_graph(maturities)
#   W       — adjacency matrix (p × p)
#   L       — combinatorial Laplacian
#   Ltilde  — spectrally-normalized Laplacian (pass this to the model)

# 3. Fit
model = GraphRegularizedPCA(
    n_components=3,
    graph=Ltilde,           # feature-side graph
    tau=0.01,               # regularization strength
    rho=0.0,                # all penalty on features (rho=0)
    standardize=True,
)
model.fit(X)

# 4. Inspect
print(model.V_.shape)       # (6, 3) — loading matrix
print(model.U_.shape)       # (200, 3) — score matrix
print(model.tau_)            # 0.01
print(model.converged_)      # True
```

## With Cross-Validation

Let the model choose `tau` automatically:

```python
import pandas as pd

dates = pd.bdate_range("2023-01-02", periods=200)

model = GraphRegularizedPCA(
    n_components=3,
    graph=Ltilde,
    cv="nested_ts",                                  # nested time-series CV
    tau_grid=np.array([0.0, 1e-3, 1e-2, 0.1]),      # candidates
    rho_grid=np.array([0.0]),                        # feature-only
    standardize=True,
)
model.fit(X, dates=dates)

print("Selected tau:", model.tau_)
print("Validation loss:", model.cv_results_["min_val_loss"])
```

## With Missing Data

Simply pass `np.nan` for missing entries:

```python
X_missing = X.copy()
X_missing[rng.random(X.shape) < 0.1] = np.nan  # 10% missing

model = GraphRegularizedPCA(n_components=3, random_state=0)
model.fit(X_missing)

# Transform new data (also with missing entries)
X_new = rng.normal(size=(20, 6))
X_new[0, 2] = np.nan
scores = model.transform(X_new)  # (20, 3)
```

## Out-of-Sample Evaluation

```python
X_train, X_test = X[:150], X[150:]
model.fit(X_train)

loss = model.reconstruction_error(X_test)
print(f"OOS MSE: {loss:.4f}")
```

## Next Steps

- [Theory](theory.md) — the mathematical formulation
- [User Guide](user-guide/basic-usage.md) — detailed usage patterns
- [API Reference](api/estimator.md) — full parameter and attribute docs
