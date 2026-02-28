# Theory

This page describes the mathematical formulation behind **grpca**.

## Objective

The estimator solves a **masked two-sided graph-regularized low-rank model** (GLRM):

$$
\min_{U \in \mathbb{R}^{n \times k},\, V \in \mathbb{R}^{p \times k}}\;
\underbrace{\tfrac{1}{2s_X}\|P_\Omega(Z - UV^\top)\|_F^2}_{\text{reconstruction}}
+ \underbrace{\tfrac{1}{2}\lambda_s \operatorname{tr}(U^\top \widetilde{L}_s U)}_{\text{observation smoothness}}
+ \underbrace{\tfrac{1}{2}\lambda_f \operatorname{tr}(V^\top \widetilde{L}_f V)}_{\text{feature smoothness}}
+ \tfrac{1}{2}\mu_u \|U\|_F^2
+ \tfrac{1}{2}\mu_v \|V\|_F^2
$$

where:

| Symbol | Meaning |
|--------|---------|
| $Z$ | Standardized data matrix $(n \times p)$ |
| $U$ | Observation embeddings / scores $(n \times k)$ |
| $V$ | Feature loadings $(p \times k)$ |
| $P_\Omega$ | Projection onto observed entries |
| $\widetilde{L}_s$ | Spectrally-normalized observation-side Laplacian |
| $\widetilde{L}_f$ | Spectrally-normalized feature-side Laplacian |
| $s_X$ | Spectral data scale $= \sigma_1^2(Z_{\mathrm{fill}})$ |
| $\mu_u, \mu_v$ | Ridge regularization for numerical conditioning |

## Spectral Normalization

Graph Laplacians are normalized so their largest eigenvalue equals 1:

$$\widetilde{L} = \frac{L}{\lambda_{\max}(L)}$$

This makes the regularization strength **portable** — the same $\tau$ grid
works across different graph topologies, sample sizes, and datasets without
manual rescaling.

## Data Scaling

The reconstruction term is divided by the **spectral data scale**:

$$s_X = \|Z_{\mathrm{fill}}\|_2^2 = \sigma_1^2(Z_{\mathrm{fill}})$$

where $Z_{\mathrm{fill}}$ replaces missing entries with zero. This ensures the
reconstruction penalty is order-one regardless of sample size or data variance.

## The $(\tau, \rho)$ Parameterization

Rather than tuning $\lambda_s$ and $\lambda_f$ independently, **grpca** uses a
reparameterization that separates *intensity* from *allocation*:

$$\lambda_s = \tau \cdot \rho, \qquad \lambda_f = \tau \cdot (1 - \rho)$$

| Parameter | Range | Interpretation |
|-----------|-------|----------------|
| $\tau$ | $[0, \infty)$ | Total regularization strength |
| $\rho$ | $[0, 1]$ | Fraction allocated to observation graph |

Special cases:

- $\rho = 0$: feature-graph only (e.g., tenor smoothing)
- $\rho = 1$: observation-graph only (e.g., temporal smoothing)
- $\tau = 0$: no graph regularization (ridge-only low-rank factorization)

## Laplacian Quadratic Forms

The graph penalties have a natural **pairwise-difference** interpretation:

**Feature side:**

$$\operatorname{tr}(V^\top \widetilde{L}_f V) = \tfrac{1}{2}\sum_{i \sim j} \tilde{w}_{ij}^{(f)} \|v_i - v_j\|_2^2$$

This penalizes oscillation of loading vectors across neighbouring features
(e.g., adjacent bond tenors).

**Observation side:**

$$\operatorname{tr}(U^\top \widetilde{L}_s U) = \tfrac{1}{2}\sum_{t \sim t'} \tilde{w}_{tt'}^{(s)} \|u_t - u_{t'}\|_2^2$$

This penalizes abrupt jumps in score vectors between consecutive observations
(e.g., adjacent trading days).

## Solver

The objective is minimized by **alternating gradient descent with backtracking
line search**:

1. Fix $V$, update $U$ by gradient descent on

$$\nabla_U = \tfrac{1}{s_X} P_\Omega(UV^\top - Z) V + \lambda_s \widetilde{L}_s U + \mu_u U$$

2. Fix $U$, update $V$ by gradient descent on

$$\nabla_V = \tfrac{1}{s_X} P_\Omega(UV^\top - Z)^\top U + \lambda_f \widetilde{L}_f V + \mu_v V$$

Each step uses an Armijo backtracking line search for the step size. The
initialization is truncated SVD of $Z_{\mathrm{fill}}$.

## Cross-Validation Selection Rule

When `cv` is set, the estimator searches over `tau_grid × rho_grid`:

1. For each candidate $(\tau, \rho)$, fit on the inner-train split and
   evaluate masked reconstruction loss on the validation split.
2. Find the minimum validation loss $\ell^*$.
3. Keep all candidates within $(1 + \delta)$ of $\ell^*$.
4. Among those, select:
   - Smallest $\tau$ (parsimony)
   - If tie: smallest Procrustes distance to `P_prev` (stability)
   - If tie: smallest $\rho$

The inner-train scaler is computed from **inner-train observed entries only**,
ensuring no leakage. The final model is refit on the full outer training set
with the selected $(\tau, \rho)$.

## References

- Paradkar, S. & Udell, M. (2017). *Graph-Regularized Generalized Low-Rank Models.* CVPR Workshop.
