# Missing Data

**grpca** handles missing data natively through a masked objective function.
Missing entries must be encoded as `np.nan`.

## How It Works

The GLRM objective only sums over observed entries via the mask $P_\Omega$:

$$\tfrac{1}{2s_X}\|P_\Omega(Z - UV^\top)\|_F^2$$

This means:

- Missing entries **do not contribute** to the reconstruction loss.
- Gradients are computed only on observed entries.
- Standardization statistics (mean, std) use only observed values per column.

## Basic Usage

```python
import numpy as np
from grpca import GraphRegularizedPCA

# Data with missing entries
X = np.random.default_rng(0).normal(size=(100, 8))
X[np.random.default_rng(1).random(X.shape) < 0.15] = np.nan

model = GraphRegularizedPCA(n_components=3, random_state=0)
model.fit(X)

print(f"Observed fraction: {model.observed_fraction_:.1%}")
print(f"Has missing: {model.has_missing_}")
```

## Transform with Missing Data

When transforming new data with missing entries, scores are solved by
projecting only the observed entries:

```python
X_new = np.random.default_rng(2).normal(size=(20, 8))
X_new[0, :3] = np.nan  # first row has 3 missing features

scores = model.transform(X_new)  # (20, 3), all finite
```

## Per-Observation Reconstruction Error

Rows with varying missingness patterns get per-row MSE:

```python
per_obs = model.reconstruction_error(X_new, per_obs=True)
# per_obs.shape = (20,)
# Rows with no observed entries → NaN
```

## Imputation

Reconstruct missing entries by inverse-transforming:

```python
scores = model.transform(X)
X_hat = model.inverse_transform(scores)  # (100, 8), fully dense
```

`X_hat` fills in the missing entries with the model's best low-rank estimate.
