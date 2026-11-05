# Building models

```rust,ignore
{{#include ../../keanu/tests/book.rs:building}}
```

## Probabilistic nodes

Each distribution has a builder method taking its parameters, which can be other nodes or plain values, and a
`_with_shape` form for a node whose value has a given shape (broadcast with the parameters' shapes).

| Kind | Distributions |
|---|---|
| Continuous | `gaussian`, `half_gaussian`, `cauchy`, `half_cauchy`, `beta`, `chi_squared`, `dirichlet`, `exponential` (takes a rate), `gamma` (scale θ, shape k), `inverse_gamma`, `laplace`, `logistic`, `log_normal`, `multivariate_gaussian`, `pareto`, `student_t`, `triangular`, `uniform`, `kde` |
| Correlation matrices | `lkj_cholesky(k, eta)`: the k × k lower triangular Cholesky factor `L` of a correlation matrix `L Lᵀ` under the LKJ(η) prior (`lkj_cholesky_with_shape(&[.., k, k], eta)` for a batch) |
| Discrete | `bernoulli` (bool), `binomial`, `categorical`, `geometric`, `multinomial`, `poisson`, `uniform_int` |
| Your own | `custom` (see [Custom distributions](custom-distributions.md)) |

## Deterministic nodes

Handles (`DoubleVar`, `IntVar`, `BoolVar`) support:

- **Operators:** `+ - * / %` and unary `-` between handles, and between handles and plain values. Boolean handles
  support `& | ^ !`.
- **Methods** for everything the tensors support:
  - element-wise math (`exp`, `ln`, `sigmoid`, `ln_gamma`, ...)
  - comparisons (`gt`, `le`, ...) and masks
  - reductions (`sum`, `sum_axes`, `mean`, `max`, ...) and scans (`cumsum`)
  - stable `log_sum_exp`, `softmax` and `log_softmax` (shifted by the maximum, so large or `-∞` inputs stay exact)
  - shapes (`reshape`, `permute`, `slice`, `broadcast_to`, ...) and indexing (`gather`, `scatter_add`; see
    [Indexing](#indexing))
  - linear algebra (`matmul`, `inverse`, `cholesky`, `determinant`, `tensor_multiply`)
  - casts (`to_double`, `to_int`)
- **Control flow on the builder:**
  - `where_(predicate, then, otherwise)` (Java's `If`)
  - `multiplexer(selector, options)`
  - `cpt(conditions, cases, default)` for conditional probability tables
- **Arbitrary functions:** `lambda`, with `lambda_with_jacobian` for a differentiable one.

floor, ceil, round, sign, comparisons and casts aren't differentiable, so a model whose latents flow through them is
sampled with Metropolis–Hastings rather than NUTS (see `Sampler::for_model`).

## Indexing

Hierarchical models give each observation the parameter of its group: in maths `mu[group[i]]`. `gather(axis,
indices)` selects elements along an axis by an integer tensor or `IntVar`, as numpy's `take`, so
`group_mu.gather(0, group)` has one element per observation. Indices may repeat, and they may be a node whose value
changes, such as a categorical latent: a cascade recomputes the selection and gradients follow the current indices.
`scatter_add(axis, indices, size)` is the adjoint, adding each element into its group (gradients of a gather are
scatter-adds and vice versa), and the builder's `stack(axis, parts)` joins nodes of the same shape along a new
axis. Indices outside the axis fail evaluation with an error, and gradients never flow into the indices.

```rust,ignore
{{#include ../../keanu/tests/book.rs:indexing}}
```

## Labels

`.label("name")` names a node so it can be found again with `graph.find("name")`, including after saving and
loading. Labels must be unique within a graph.
