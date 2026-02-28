# Cross-Validation

**grpca** includes built-in nested time-series cross-validation for automatic
hyperparameter selection.

## Enabling CV

Set `cv` to one of:

| Value | Behaviour |
|-------|-----------|
| `None` | No CV; use fixed `tau`/`rho` or `lambda_s`/`lambda_f` |
| `"nested_ts"` | Built-in `NestedTimeSeriesSplit` with `inner_duration='270D'` |
| `"nested"` | Alias for `"nested_ts"` |
| splitter object | Any object with `.split(X, dates=...) → (inner, val)` |
| callable | `cv(X, dates=None) → (inner_idx, val_idx)` |

## Specifying Grids

```python
model = GraphRegularizedPCA(
    n_components=3,
    graph=Ltilde,
    cv="nested_ts",
    tau_grid=np.array([0.0, 1e-4, 1e-3, 1e-2, 0.1]),
    rho_grid=np.array([0.0, 0.5, 1.0]),
    delta=0.01,     # tolerance for conservative selection
)
model.fit(X, dates=dates)
```

!!! note "Default grids"
    If not provided, defaults are:

    ```python
    tau_grid = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1]
    rho_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    ```

    You can also use the helper methods:

    ```python
    tau_grid = GraphRegularizedPCA.suggest_tau_grid()
    rho_grid = GraphRegularizedPCA.suggest_rho_grid()
    ```

## Selection Rule

1. Evaluate masked reconstruction loss for every $(\tau, \rho)$ on the validation fold.
2. Find $\ell^* = \min$ validation loss.
3. **Candidate set**: keep $(\tau, \rho)$ with loss $\le (1 + \delta) \cdot \ell^*$.
4. **Tie-break** (in order):
    - Smallest $\tau$ (parsimony)
    - Smallest Procrustes distance to `P_prev` (stability, if provided)
    - Smallest $\rho$

## Leakage-Safe Scaling

For each inner split:

- Inner-train statistics are computed from **observed entries only**.
- The validation fold is standardized with inner-train statistics.
- The final `mean_` / `scale_` always correspond to the full outer-train fit.

## Inspecting CV Results

After fit, `model.cv_results_` contains:

```python
cv = model.cv_results_

cv["selected_tau"]         # chosen tau
cv["selected_rho"]         # chosen rho
cv["min_val_loss"]         # best validation loss
cv["stability_tiebreak_used"]  # whether P_prev was used as tiebreak
cv["results_table"]        # list of dicts, one per candidate
cv["inner_mean"]           # inner-train mean
cv["inner_scale"]          # inner-train scale
```

Use `summarize_cv_surface()` for a tidy table:

```python
from grpca import summarize_cv_surface
df = summarize_cv_surface(model.cv_results_)
```

## Stability Tie-Breaking with `P_prev`

When running rolling-window analyses, pass the previous window's loadings
projector to prefer hyperparameters that produce similar factors:

```python
model.fit(X_train, dates=train_dates, P_prev=prev_V_projector)
```

This only affects tie-breaking among candidates within $\delta$ of the
best validation loss.
