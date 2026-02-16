# grpca

Graph-Regularized PCA package with a sklearn-compatible estimator and time-series CV utilities.

## Install

```bash
pip install -e .
```

## Run tests

```bash
pytest -q tests/test_gr_pca.py
```

## Package contents

- `grpca.estimator.GraphRegularizedPCA`
- `grpca.splitters.RollingWindowSplitter`
- `grpca.splitters.NestedTimeSeriesSplit`
- `grpca.metrics`, `grpca.graphs`, `grpca.diagnostics`, `grpca.inference`
