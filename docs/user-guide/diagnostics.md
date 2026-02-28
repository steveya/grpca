# Diagnostics

**grpca** includes statistical tests and helper functions for comparing
models and inspecting CV results.

## Diebold–Mariano Test

Compare out-of-sample reconstruction losses between two models:

```python
from grpca.diagnostics import dm_test

# losses_a, losses_b: per-window OOS losses from rolling backtest
result = dm_test(losses_a, losses_b, lag=2, alternative="less")

print(f"DM stat: {result['dm_stat']:.3f}")
print(f"p-value: {result['p_value']:.4f}")
print(f"Win rate: {result['win_rate']:.1%}")
```

The test uses **Newey–West HAC** standard errors to account for serial
correlation in loss differences.

## Roughness Metrics

Measure how smooth the fitted factors are relative to the graph:

```python
from grpca.metrics import feature_roughness, example_roughness

f_rough = feature_roughness(model.V_, Ltilde_tenor)
s_rough = example_roughness(model.U_, Ltilde_day)
```

Lower roughness = smoother factors along the graph edges.

## Procrustes Distance

Measure loading stability across windows:

```python
from grpca.metrics import procrustes_distance

drift = procrustes_distance(V_new, V_old)
```

This finds the optimal orthogonal rotation $R$ minimizing
$\|V_{\text{new}} - V_{\text{old}} R\|_F$.

## Subspace Distance

Davis–Kahan sine of the largest principal angle:

```python
from grpca.metrics import subspace_distance

d = subspace_distance(V_new, V_old, k=3)
# d ∈ [0, 1]; 0 = identical subspaces
```

## CV Surface Inspection

```python
from grpca import summarize_cv_surface, extract_validation_loss_surface

# Tidy table of all (tau, rho) candidates
df = summarize_cv_surface(model.cv_results_)

# Plot-friendly loss surface
surface = extract_validation_loss_surface(model.cv_results_)
```

## Tracking Selected Hyperparameters Over Time

After a rolling-window backtest, summarize how the selected hyperparameters
evolved:

```python
from grpca import (
    extract_selected_tau_series,
    extract_selected_rho_series,
    extract_selected_lambda_series,
)

tau_series = extract_selected_tau_series(all_cv_results)
rho_series = extract_selected_rho_series(all_cv_results)
lambda_series = extract_selected_lambda_series(all_cv_results)
```
