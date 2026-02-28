# Graph Construction

**grpca** provides helper functions for building graph Laplacians from
domain knowledge. All helpers return a triple `(W, L, Ltilde)`:

- `W` — adjacency matrix
- `L` — combinatorial Laplacian $(D - W)$
- `Ltilde` — spectrally-normalized Laplacian (pass this to the estimator)

## Tenor Chain Graph

For ordered features (e.g., bond tenors), edge weights are inversely
proportional to the squared gap:

$$w_{i,i+1} = \frac{1}{(m_{i+1} - m_i)^2}$$

```python
from grpca.graphs import tenor_chain_graph

maturities = np.array([1, 2, 3, 5, 7, 10, 20, 30], dtype=float)
W, L, Ltilde = tenor_chain_graph(maturities)
```

Nearby tenors (1Y–2Y) get strong edges; distant tenors (10Y–20Y) get weak
edges.

## Day Chain Graph

For sequential observations, connects each row to its neighbours with unit
weight:

```python
from grpca.graphs import day_chain_graph

W, L, Ltilde = day_chain_graph(dates, normalize=True)
```

## Hierarchical Graph

Builds on the tenor chain with additional intra-community edges and
dampened cross-boundary edges:

```python
from grpca.graphs import hierarchical_graph

W, L, Ltilde = hierarchical_graph(
    maturities,
    communities={"deliverable": [7.0, 8.0, 9.0, 10.0]},
    community_boost=0.35,
    dampened_edges=[(10.0, 15.0, 0.10)],
)
```

## Custom Graphs

You can build any graph Laplacian manually and normalize it:

```python
from grpca.graphs import spectral_normalize_laplacian

# Your custom adjacency matrix
W_custom = ...
D = np.diag(W_custom.sum(axis=1))
L = D - W_custom
Ltilde = spectral_normalize_laplacian(L)

model = GraphRegularizedPCA(n_components=3, graph=Ltilde)
```

The Laplacian must be:

- Square and symmetric
- Positive semi-definite
- Free of NaN/Inf values
