# Basic Usage

## Fitting a Model

The main entry point is `GraphRegularizedPCA`. It follows the sklearn
`fit` / `transform` / `inverse_transform` pattern.

### No Graph (Baseline PCA-like)

```python
from grpca import GraphRegularizedPCA

model = GraphRegularizedPCA(n_components=3, standardize=True)
model.fit(X)
```

With no graphs and `tau=0`, this reduces to a ridge-regularized low-rank
factorization that closely approximates PCA.

### Feature Graph Only

```python
from grpca.graphs import tenor_chain_graph

maturities = [1, 2, 3, 5, 7, 10, 20, 30]
W, L, Ltilde = tenor_chain_graph(maturities)

model = GraphRegularizedPCA(
    n_components=3,
    graph=Ltilde,        # feature-side spectrally-normalized Laplacian
    tau=0.01,
    rho=0.0,             # all penalty on features
)
model.fit(X)
```

### Observation Graph Only

```python
from grpca.graphs import day_chain_graph

W, L, Ltilde = day_chain_graph(dates)

model = GraphRegularizedPCA(
    n_components=3,
    example_graph=Ltilde,   # observation-side Laplacian
    tau=0.01,
    rho=1.0,                # all penalty on observations
)
model.fit(X)
```

### Both Graphs

```python
model = GraphRegularizedPCA(
    n_components=3,
    graph=Ltilde_tenor,
    example_graph=Ltilde_day,
    tau=0.01,
    rho=0.5,             # split evenly
)
model.fit(X)
```

## Accessing Results

After fitting:

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `model.V_` | `(p, k)` | Feature loadings |
| `model.U_` | `(n, k)` | Observation scores |
| `model.components_` | `(k, p)` | sklearn-convention loadings |
| `model.loadings_` | `(p, k)` | Alias for `V_` |
| `model.tau_` | scalar | Selected/applied $\tau$ |
| `model.rho_` | scalar | Selected/applied $\rho$ |
| `model.lambda_s_` | scalar | Effective observation penalty |
| `model.lambda_f_` | scalar | Effective feature penalty |
| `model.mean_` | `(p,)` | Per-feature mean |
| `model.scale_` | `(p,)` | Per-feature std |
| `model.converged_` | bool | Convergence flag |
| `model.n_iter_` | int | Iterations used |
| `model.objective_trace_` | `(n_iter,)` | Objective value per iteration |

## Transform & Reconstruct

```python
# Project new data into the latent space
scores = model.transform(X_new)       # (n_new, k)

# Reconstruct from scores
X_hat = model.inverse_transform(scores)  # (n_new, p)

# Out-of-sample reconstruction error
loss = model.reconstruction_error(X_test)
```

## Automatic Day-Chain Graph

Instead of building the observation graph manually, use `use_day_chain=True`:

```python
model = GraphRegularizedPCA(
    n_components=3,
    graph=Ltilde_tenor,
    use_day_chain=True,     # auto-build observation graph from dates
    cv="nested_ts",
)
model.fit(X, dates=dates)
```
