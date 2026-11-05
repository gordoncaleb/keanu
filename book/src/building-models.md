# Building models

```rust,ignore
{{#include ../../keanu/tests/book.rs:building}}
```

## Probabilistic nodes

Each distribution has a builder method taking its parameters, which can be other nodes or plain values, and a
`_with_shape` form for a node whose value has a given shape (broadcast with the parameters' shapes).

| Kind | Distributions |
|---|---|
| Continuous | `gaussian`, `half_gaussian`, `cauchy`, `half_cauchy`, `beta`, `chi_squared`, `dirichlet`, `exponential` (takes a rate), `gamma` (scale θ, shape k), `inverse_gamma`, `laplace`, `logistic`, `log_normal`, `multivariate_gaussian`, `pareto`, `student_t`, `triangular`, `uniform`, `weibull` (shape α, scale β), `von_mises` (an angle), `kde` |
| Correlation matrices | `lkj_cholesky(k, eta)`: the k × k lower triangular Cholesky factor `L` of a correlation matrix `L Lᵀ` under the LKJ(η) prior (`lkj_cholesky_with_shape(&[.., k, k], eta)` for a batch) |
| Discrete | `bernoulli` (bool), `binomial`, `categorical`, `geometric`, `multinomial`, `negative_binomial` (mean μ, dispersion α), `ordered_logistic`, `poisson`, `uniform_int` |
| Your own | `custom` (see [Custom distributions](custom-distributions.md)) |

### Counts, durations, angles and ordered categories

- `negative_binomial(mu, alpha)` is over-dispersed count data in PyMC's mean–dispersion form: mean μ and variance
  μ + μ²/α, approaching `poisson(mu)` as α grows.
- `weibull(alpha, beta)` has shape α and scale β, for durations and lifetimes.
- `von_mises(mu, kappa)` is a distribution of angles with mean direction μ and concentration κ (uniform on the circle
  at κ = 0). Its support is circular, so NUTS and the other gradient based algorithms move through a circular
  transform and pass freely between -π and π; samples are reported in [-π, π).
- `ordered_logistic(eta, cutpoints)` is Stan's ordered logistic regression: a category index in 0..K from a location
  η and K − 1 strictly increasing cutpoints (the last axis of `cutpoints`). Cutpoints that don't increase are
  outside the valid region: the log probability is −∞ there (so Metropolis–Hastings rejects a proposal of them) and
  sampling is an error. Latent cutpoints are best given an ordered prior.

```rust,ignore
{{#include ../../keanu/tests/book.rs:new_distributions}}
```

## Truncated and censored distributions

`truncated(base, parameters, lower, upper)` restricts a univariate distribution to `[lower, upper]` and renormalises
it by the probability the base gives that range; `censored(base, parameters, lower, upper)` models values that are
clipped to the bounds, so a value equal to `lower` has probability P(X ≤ lower) and a value equal to `upper` has
P(X ≥ upper). The base is a `DistributionKind` with a CDF (Gaussian, LogNormal, Exponential, Gamma, Beta, StudentT,
Cauchy, Laplace, Logistic, Uniform, Weibull, Poisson, Binomial or NegativeBinomial) and its parameters are node ids, in the base's order.
Either bound may be infinite; for a discrete base the bounds are whole numbers and the range includes them.

```rust,ignore
{{#include ../../keanu/tests/book.rs:truncated_censored}}
```

A truncated latent's support is the base's support intersected with the bounds, so NUTS, the gradient optimizers
and the other gradient based algorithms keep it inside them. Truncations are sampled by inverting the base's CDF
where it has a closed form (Gaussian, LogNormal, Exponential, Cauchy, Laplace, Logistic, Uniform, Weibull) and by rejection
otherwise, which fails with an error if the bounds hold too little of the base's probability. Censoring is meant for
observed data. Each of these distributions also has `log_cdf` and `log_ccdf` (with gradients) in
`keanu::distributions`, accurate far into the tails.

## Mixtures

`mixture(component, parameters, weights)` is a mixture of C distributions of one kind, with density
Σ_c w_c p_c(x). The weights are non-negative and sum to 1 along their last axis, the component axis, of length C (a
`dirichlet` prior keeps a latent weight vector on the simplex; weights off it give a log probability of −∞, and
an error when sampled). Each of the component's parameters, given as node
ids in the component's order, has the same trailing component axis, of length C, or of length 1 or no axes at all
to share one value between the components. The value has the parameters' batch shape (their shapes without the
component axis); `mixture_with_shape` gives a larger one, such as a vector of observations.

```rust,ignore
{{#include ../../keanu/tests/book.rs:mixture}}
```

The component can be any distribution with one value per element (the continuous and count distributions, but
not `dirichlet`, `multivariate_gaussian`, `lkj_cholesky`, `kde`, `categorical`, `multinomial`, `ordered_logistic` or
`bernoulli`), or a truncated or censored one; a mixture itself can't be truncated or mixed again. The log density is
computed as logsumexp_c(log w_c + log p_c(x)), so it stays accurate far in the tails, and its gradient with respect
to the value, the component parameters and the weights is exact, for NUTS and the optimizers. The support is the
component's, widened to hold every component's (the smallest lower bound and the largest upper bound). Values are
drawn by picking a component by its weight and drawing from it.

A mixture's likelihood doesn't change when components swap labels, so a posterior over unconstrained component
parameters has a copy of each mode per labelling. Order the components to remove the copies, as the
[Gaussian mixture example](examples.md#a-gaussian-mixture) orders the means.

## Factors

`factor(expr)` adds an arbitrary term to the joint log probability, like PyMC's `Potential` or Stan's
`target +=`: the factor node's value is the sum of `expr`, and the model's log probability is the sum of its
probabilistic nodes' log probabilities and its factors' values. Use it for soft constraints, penalties, and
likelihoods that no distribution provides. A factor that isn't finite (NaN or ±∞) makes the point impossible.

```rust,ignore
{{#include ../../keanu/tests/book.rs:factor}}
```

Factors count in the log likelihood (`Model::log_likelihood`), not in the log prior. Their gradients and Hessians
are those of the expression, so NUTS, the optimizers, variational inference and the Laplace approximation follow
them, and Metropolis–Hastings updates them incrementally with the nodes a proposal reaches. A factor is a term of
the density rather than a generative step, so forward sampling, predictive sampling and the particle filter ignore
it. Factors are saved and loaded with the model, but scan bodies can't hold them.

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
  - linear algebra (`matmul`, `inverse`, `cholesky`, `determinant`, `tensor_multiply`, `solve`,
    `solve_triangular`, `slogdet`, `eigh`). On the graph each node has one value, so `slogdet()` and `eigh()` return
    two nodes: `determinant_sign` with `log_abs_determinant`, and `eigvalsh` with `eigh_vectors`. Eigenvector
    gradients assume distinct eigenvalues.
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
