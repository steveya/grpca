# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] — 2025-01-01

### Added

- `GraphRegularizedPCA` estimator with masked GLRM objective and alternating
  gradient descent solver.
- Spectral Laplacian normalization and spectral data scaling for portable
  hyperparameter grids.
- (τ, ρ) reparameterization separating regularization intensity from
  observation/feature allocation.
- Nested time-series cross-validation with leakage-safe inner scaling and
  conservative selection rule (parsimony + stability tie-breaking).
- `RollingWindowSplitter` and `NestedTimeSeriesSplit` for time-series
  backtesting workflows.
- Graph construction helpers: `tenor_chain_graph`, `day_chain_graph`,
  `hierarchical_graph`.
- Metrics: `reconstruction_loss`, `masked_reconstruction_loss`,
  `procrustes_distance`, `subspace_distance`, `feature_roughness`,
  `example_roughness`, `eigengap`.
- Diagnostics: `dm_test` (Diebold–Mariano), `newey_west_se`,
  `best_match_scores`, CV surface helpers.
- Inference: `infer_precision_graph`, `jaccard_topM`,
  `spearman_union_support`.
- Native missing-data support via `np.nan` entries.
