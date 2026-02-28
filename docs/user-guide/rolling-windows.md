# Rolling Windows

For time-series applications (e.g., yield curve analysis), **grpca** provides
a `RollingWindowSplitter` for production-style backtesting.

## RollingWindowSplitter

```python
from grpca.splitters import RollingWindowSplitter
import pandas as pd
import numpy as np

dates = pd.bdate_range("2020-01-01", periods=1000)
X = np.random.default_rng(0).normal(size=(1000, 10))

splitter = RollingWindowSplitter(
    train_len="365D",      # 1-year training window
    test_len="92D",        # 3-month test window
    step="30D",            # roll forward 1 month
    min_train_obs=180,     # require at least 180 training obs
    min_test_obs=40,       # require at least 40 test obs
)
```

### Basic Iteration

```python
for train_idx, test_idx in splitter.split(X, groups=dates):
    X_train, X_test = X[train_idx], X[test_idx]
    # fit and evaluate...
```

### With Window Info

```python
for end_dt, train_idx, test_idx in splitter.split_with_info(X, groups=dates):
    print(f"Window ending {end_dt}: {len(train_idx)} train, {len(test_idx)} test")
```

## Full Backtest Pattern

```python
from grpca import GraphRegularizedPCA
from grpca.graphs import tenor_chain_graph
from grpca.metrics import procrustes_distance

maturities = np.array([1, 2, 3, 5, 7, 10, 20, 30], dtype=float)
_, _, Ltilde = tenor_chain_graph(maturities)

results = []
prev_V = None

for end_dt, tr_idx, te_idx in splitter.split_with_info(X, groups=dates):
    X_tr, X_te = X[tr_idx], X[te_idx]

    model = GraphRegularizedPCA(
        n_components=3,
        graph=Ltilde,
        cv="nested_ts",
        tau_grid=np.array([0.0, 1e-3, 1e-2, 0.1]),
        rho_grid=np.array([0.0]),
    )
    model.fit(X_tr, dates=dates[tr_idx])

    loss = model.reconstruction_error(X_te)

    drift = (
        procrustes_distance(model.V_, prev_V) if prev_V is not None else float("nan")
    )

    results.append({
        "window_end": end_dt,
        "tau": model.tau_,
        "oos_loss": loss,
        "loading_drift": drift,
    })
    prev_V = model.V_.copy()
```

## Tabular Mode

If no `groups` (dates) are provided, the splitter uses integer-based
windowing:

```python
splitter = RollingWindowSplitter(
    train_len=200, test_len=50, step=25,
    min_train_obs=100, min_test_obs=20,
)
for tr, te in splitter.split(X):
    ...
```

## NestedTimeSeriesSplit

This is used internally by `GraphRegularizedPCA` when `cv="nested_ts"`.
It creates a single inner-train / validation split:

```python
from grpca.splitters import NestedTimeSeriesSplit

split = NestedTimeSeriesSplit(inner_duration="270D")
inner_idx, val_idx = split.split(X_train, dates=train_dates)
```
