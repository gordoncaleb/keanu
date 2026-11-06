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
| Zero-sum vectors | `zero_sum_normal(n, sigma)`: n elements summing to zero, N(0, σ² I) restricted to that subspace (`zero_sum_normal_with_shape(&[.., n], sigma)` for a batch); see [Sum-to-zero group effects](#sum-to-zero-group-effects) |
| Spatial (areal) effects | `icar(adjacency, sigma)`, `car(adjacency, tau, alpha)` and `bym2(adjacency, sigma, rho)` over the areas of an `Adjacency` |
| Correlation matrices | `lkj_cholesky(k, eta)`: the k × k lower triangular Cholesky factor `L` of a correlation matrix `L Lᵀ` under the LKJ(η) prior (`lkj_cholesky_with_shape(&[.., k, k], eta)` for a batch) |
| Discrete | `bernoulli` (bool), `binomial`, `beta_binomial`, `categorical`, `geometric`, `multinomial`, `negative_binomial` (mean μ, dispersion α), `ordered_logistic`, `poisson`, `uniform_int` |
| Logit and log forms | `bernoulli_logit` (bool), `binomial_logit`, `categorical_logit`, `poisson_log`; see [Zero-inflated, hurdle and logit counts](#zero-inflated-hurdle-and-logit-counts) |
| Extra zeros | `zero_inflated` and `hurdle` around a count distribution (or, for `hurdle`, a positive continuous one) |
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

### Zero-inflated, hurdle and logit counts

- `zero_inflated(kind, parameters, psi)` adds structural zeros to a count distribution: with probability ψ the
  value is 0, else it is a draw from `kind` (a `poisson`, `negative_binomial`, `binomial`, `beta_binomial`, a logit
  or log form, or a truncated count) with `parameters`. So P(0) = ψ + (1 − ψ) p(0) and P(k) = (1 − ψ) p(k), as PyMC's
  `ZeroInflatedPoisson` and its relatives.
- `hurdle(kind, parameters, psi)` separates the zeros from the positive values: 0 with probability ψ, else a draw from
  `kind` restricted to positive values, P(k) = (1 − ψ) p(k) / (1 − p(0)). With a positive continuous kind (`gamma`,
  `log_normal`, `exponential`, `weibull`, ...) a value of exactly 0 has probability ψ and a positive value the
  density (1 − ψ) p(x), for data such as rainfall with dry days.
- `beta_binomial(n, alpha, beta)` is a binomial whose success probability is Beta(α, β): over-dispersed successes
  out of n trials, with mean nα/(α + β). Its log CDF and survival function are finite sums, so it can be truncated
  and censored too.
- `bernoulli_logit(eta)`, `binomial_logit(eta, n)` and `categorical_logit(shape, &logits)` take log odds (or
  unnormalised log probabilities) and `poisson_log(eta)` the log of the mean, as Stan's `bernoulli_logit`,
  `binomial_logit`, `categorical_logit` and `poisson_log`. Their log probabilities are computed as −softplus(∓η),
  ηₓ − logsumexp(η) and kη − e^η, so they stay finite and accurate for any linear predictor (a logit of ±700
  included), with no sigmoid or exponential taken first.

```rust,ignore
{{#include ../../keanu/tests/book.rs:zero_inflated}}
```

Each has exact gradients and Hessian-vector products, for NUTS, the optimizers and the Laplace approximation. The
zero-inflated and hurdle forms, the beta-binomial and the logit and log forms with one value per element can be
mixture components, `log_density` kinds and children of a marginalised latent, and their pointwise log likelihoods
have one point per element. As latents, the ones with finitely many values (the beta-binomial, the binomial and
Bernoulli logit forms, `categorical_logit`, and zero-inflated or hurdle forms of those) are updated by Gibbs steps
and can be marginalised; the others by the discrete random walk. A ψ outside [0, 1] makes every value impossible.

### Sum-to-zero group effects

A model with an intercept and one effect per group, y ~ N(μ + a[group], σ), is not identified as written: adding a
constant to μ and subtracting it from every a gives the same likelihood, so only the prior holds them apart. The
usual fixes are to drop one group's effect (a reference level) or to make the effects sum to zero, so that μ is the
grand mean and each a_j a deviation from it. `zero_sum_normal(n, sigma)` is the second, as PyMC's `ZeroSumNormal`:
the density of N(0, σ² I) on the subspace of vectors summing to zero. Each effect's marginal standard deviation is
σ √((n − 1)/n), and draws are draws of N(0, σ² I) minus their mean.

```rust,ignore
{{#include ../../keanu/tests/book.rs:zero_sum}}
```

Its support is `Support::SumToZero`. The gradient based algorithms map it to n − 1 unconstrained coordinates with
Stan's `sum_to_zero_vector` transform (2.36 and later): the coordinates are those of the value in an orthonormal
basis of the subspace, so the map is an isometry and NUTS sees an isotropic Gaussian when the data are weak. A
value given with `set_value` or observed must sum to zero (to 10⁻⁸ relative); one that doesn't is impossible (log
probability −∞). For hierarchical models with several grouping factors, give each factor its own zero-sum effects.
The [radon example](examples.md) uses an intercept per county instead (partial pooling around a common mean), where
the identification comes from the hierarchical prior.

## Truncated and censored distributions

`truncated(base, parameters, lower, upper)` restricts a univariate distribution to `[lower, upper]` and renormalises
it by the probability the base gives that range; `censored(base, parameters, lower, upper)` models values that are
clipped to the bounds, so a value equal to `lower` has probability P(X ≤ lower) and a value equal to `upper` has
P(X ≥ upper). The base is a `DistributionKind` with a CDF (Gaussian, LogNormal, Exponential, Gamma, Beta, StudentT,
Cauchy, Laplace, Logistic, Uniform, Weibull, Poisson, Binomial, BetaBinomial or NegativeBinomial) and its parameters are node ids, in the base's order.
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
not `dirichlet`, `multivariate_gaussian`, `lkj_cholesky`, `zero_sum_normal`, `kde`, `categorical`, `multinomial`,
`ordered_logistic`, `bernoulli`, `bernoulli_logit` or `categorical_logit`), or a truncated, censored, zero-inflated
or hurdle one; a mixture itself can't be truncated or mixed again. The log density is
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

`log_density(kind, value, parameters)` is the log density of each element of a value under a distribution, as a
deterministic node: its sum is the log probability a node of that distribution with that value would have, so a
factor of it gives a likelihood to a value that isn't itself a probabilistic node (such as a transformed
observation). It works for every distribution with one value per element (the continuous and count distributions,
`bernoulli` and `bernoulli_logit`, and truncated, censored, zero-inflated and hurdle ones; not `dirichlet`, `multivariate_gaussian`, `lkj_cholesky`,
`zero_sum_normal`, `kde`, `categorical`, `multinomial`, `ordered_logistic` or mixtures), its value has the distribution's type (f64, i64 for
counts, or bool for `bernoulli` and `bernoulli_logit`), and it differentiates, to second order, as the distribution does.

```rust,ignore
{{#include ../../keanu/tests/book.rs:log_density}}
```

Factors count in the log likelihood (`Model::log_likelihood`), not in the log prior. Their gradients and Hessians
are those of the expression, so NUTS, the optimizers, variational inference and the Laplace approximation follow
them, and Metropolis–Hastings updates them incrementally with the nodes a proposal reaches. A factor is a term of
the density rather than a generative step, so forward sampling, predictive sampling and the particle filter ignore
it. Factors are saved and loaded with the model, but scan bodies can't hold them.

## Marginalising discrete latents

NUTS needs every latent to be continuous. A discrete latent with finitely many values can instead be summed out of
the model, as Stan users do by hand: `marginalize(&graph, &state, &[z])` rewrites the graph so that, in place of `z`
and the observations that depend on it, a factor holds their marginal likelihood,

log p(y | θ) = Σₛ logsumexpₖ (log p(zₛ = k | θ) + Σ log p(yₛ | zₛ = k, θ)),

computed by copying the path from `z` to its observations once for each value k, with `z` replaced by k. It returns
the new graph and state and a `Marginalized` record: `node(id)` gives the new node of each node of the original
graph (labels are kept), and `latents()` describes each marginalised latent. Afterwards
`recover_marginals(&model, &state, &marginalized, &samples)` gives, for each draw, the latent's conditional
distribution p(zₛ = k | draw), whose average over the draws is its posterior (with less Monte Carlo error than
counting sampled values); `draw` samples the latent from it, for draws of the original model's joint posterior.

```rust,ignore
{{#include ../../keanu/tests/book.rs:marginalize}}
```

The latent's distribution is `bernoulli`, `categorical` (or their logit forms) or another discrete one whose elements
take the same fixed, finite set of values (at most 1000; a `uniform_int`, a `binomial`, a `beta_binomial` or a
truncated count), and every probabilistic node
depending on it is observed and has an element-wise density (as for `log_density`). A scalar latent can reach its
observations through any deterministic nodes. For a vector latent of shape S, each element's observations must
depend on that element alone: the observations' shape starts with S (their trailing axes, such as repeated visits
to a site, are summed), and the path from the latent uses only element-wise operations, `where_`, casts, reshapes and
broadcasts that keep the leading axes, reductions over trailing axes, and `gather` with the latent as the indices
(choosing each element's component parameters, as in a mixture). Anything else, such as a matrix product or a sum
across the latent's elements, is rejected with `GraphError::CannotMarginalize`, naming the node in the way. Several
latents are marginalised one after another when each has its own observations.

A Gaussian mixture written with an explicit `categorical` assignment per point and marginalised this way has the
same log probability, gradients and posterior as the same model written with `mixture`; the
[Gaussian mixture example](examples.md#a-gaussian-mixture) fits both.

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

floor, ceil, round, sign, comparisons and casts aren't differentiable, so a model whose continuous latents flow
through them is sampled with Metropolis–Hastings rather than NUTS (see `Sampler::for_model`). Only the paths from
continuous latents matter: comparing a discrete latent with a constant (a change point with the time) is fine, and
such models get a compound sampler, NUTS for the continuous latents with Gibbs steps for the discrete ones.

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
