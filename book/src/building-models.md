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
| Skewed, extreme-value, bounded and positive | `skew_normal`, `ex_gaussian`, `asymmetric_laplace`, `gumbel`, `generalized_extreme_value`, `generalized_pareto`, `kumaraswamy`, `logit_normal`, `inverse_gaussian`, `half_student_t`, `log_logistic`, `rice`, `generalized_gamma`; see [Skewed, extreme-value and other continuous families](#skewed-extreme-value-and-other-continuous-families) |
| Zero-sum vectors | `zero_sum_normal(n, sigma)`: n elements summing to zero, N(0, σ² I) restricted to that subspace (`zero_sum_normal_with_shape(&[.., n], sigma)` for a batch); see [Sum-to-zero group effects](#sum-to-zero-group-effects) |
| Response times | `wiener(a, t0, w, v, sv, boundary)`: the Wiener first-passage time (drift-diffusion model) of a response time at the boundary reached (0 or 1, observed data), with boundary separation a, non-decision time t0, starting point w, drift v and drift variability sv (0 for the four-parameter model); see `keanu_distributions::Wiener` and `sample_wiener` |
| Spatial (areal) effects | `icar(adjacency, sigma)`, `car(adjacency, tau, alpha)` and `bym2(adjacency, sigma, rho)` over the areas of an `Adjacency`; see [Spatial models](spatial-models.md) |
| Correlation matrices | `lkj_cholesky(k, eta)`: the k × k lower triangular Cholesky factor `L` of a correlation matrix `L Lᵀ` under the LKJ(η) prior (`lkj_cholesky_with_shape(&[.., k, k], eta)` for a batch) |
| Vectors, matrices and directions | `multivariate_student_t`, `multivariate_student_t_cholesky`, `wishart`, `wishart_cholesky`, `inverse_wishart`, `inverse_wishart_cholesky`, `matrix_normal`, `von_mises_fisher` (a unit vector); see [Multivariate, ranking and other discrete families](#multivariate-ranking-and-other-discrete-families) |
| Discrete | `bernoulli` (bool), `binomial`, `beta_binomial`, `categorical`, `geometric`, `multinomial`, `negative_binomial` (mean μ, dispersion α), `ordered_logistic`, `poisson`, `uniform_int`, and `ordered_probit`, `hyper_geometric`, `skellam`, `dirichlet_multinomial`, `plackett_luce` (a ranking) |
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

### Skewed, extreme-value and other continuous families

Each of these follows Stan's parameterisation where Stan has the family and PyMC's otherwise; the rustdoc of each
`keanu_distributions` type gives its density. All have gradients and second derivatives in the value and every
parameter, and all can be mixture components. Those with a CDF can be [truncated or
censored](#truncated-and-censored-distributions); "inverse" means a truncation is sampled by inverting the CDF,
"rejection" by rejection sampling.

| Builder method | Parameters | Support | Following | CDF |
|---|---|---|---|---|
| `skew_normal` | location ξ, scale ω, shape α | real | Stan `skew_normal` | rejection |
| `ex_gaussian` | normal mean μ and sd σ, exponential rate λ | real | Stan `exp_mod_normal` | rejection |
| `asymmetric_laplace` | location μ (the τ-quantile), scale σ, skewness τ in (0, 1) | real | Stan `skew_double_exponential` | inverse |
| `gumbel` | location μ, scale β | real | Stan `gumbel` | inverse |
| `generalized_extreme_value` | location μ, scale σ, shape ξ (ξ > 0 heavy-tailed) | 1 + ξ(x − μ)/σ > 0 | PyMC `GenExtreme` | inverse |
| `generalized_pareto` | threshold μ, scale σ, shape ξ | x > μ (and x < μ − σ/ξ for ξ < 0) | the usual peaks-over-threshold form | inverse |
| `kumaraswamy` | shapes a, b | (0, 1) | PyMC `Kumaraswamy` | inverse |
| `logit_normal` | logit mean μ and sd σ | (0, 1) | PyMC `LogitNormal` | inverse |
| `inverse_gaussian` | mean μ, shape λ | positive | PyMC `Wald` | rejection |
| `half_student_t` | degrees of freedom ν (real), scale σ | positive | PyMC `HalfStudentT` | rejection |
| `log_logistic` | scale α (the median), shape β | positive | Stan `loglogistic` | inverse |
| `rice` | noncentrality ν, scale σ | positive | PyMC `Rice` | none |
| `generalized_gamma` | scale θ, shape k, power p (`gamma(θ, k)` at p = 1) | positive | Stacy's family | rejection |

The generalized extreme value and generalized Pareto supports depend on the parameters (like a uniform's on its
bounds), so a latent of either moves inside the support the current parameters give. Their densities, CDFs and
derivatives are continuous as ξ passes through 0, where they are the Gumbel and the (shifted) exponential.

### Multivariate, ranking and other discrete families

These follow Stan's parameterisation where Stan has the family and PyMC's otherwise; the rustdoc of each
`keanu_distributions` type gives its density. All have gradients and second derivatives in the value and every
parameter. The multivariate ones are distributions over the value's last axis (vectors) or last two axes
(matrices), with any leading axes a batch that the parameters' leading axes broadcast to; `_with_shape` sets the
whole value shape.

| Builder method | Parameters | Value and support | Following |
|---|---|---|---|
| `multivariate_student_t` | degrees of freedom ν (real), location μ [K], scale matrix Σ [K, K] | real [K] | Stan `multi_student_t` |
| `multivariate_student_t_cholesky` | ν, μ, the Cholesky factor L of Σ | real [K] | Stan `multi_student_t_cholesky` |
| `wishart` | ν > K − 1, scale S [K, K] (mean νS) | covariance matrix [K, K] | Stan `wishart` |
| `wishart_cholesky` | ν, the Cholesky factor of S | the Cholesky factor of a Wishart matrix | Stan `wishart_cholesky` |
| `inverse_wishart` | ν > K − 1, scale S (mean S/(ν − K − 1)) | covariance matrix [K, K] | Stan `inv_wishart` |
| `inverse_wishart_cholesky` | ν, the Cholesky factor of S | the Cholesky factor of an inverse Wishart matrix | Stan `inv_wishart_cholesky` |
| `matrix_normal` | mean M [n, p], row covariance U [n, n], column covariance V [p, p] | real [n, p] | PyMC `MatrixNormal` |
| `von_mises_fisher` | mean direction μ (a unit vector [p], p ≥ 2), concentration κ ≥ 0 | unit vector [p] | SciPy `vonmises_fisher` |
| `ordered_probit` | location η, increasing cutpoints | category in 0..K | Stan `ordered_probit` |
| `hyper_geometric` | population N, successes K, draws n (integers) | successes among the draws | PyMC `HyperGeometric` |
| `skellam` | means μ₁, μ₂ | the difference of two Poisson counts | the usual form |
| `dirichlet_multinomial` | count n, concentrations α [K] | counts [K] summing to n | PyMC `DirichletMultinomial` |
| `plackett_luce` | positive worths w [K] | a ranking [K]: the item indices from first to last | the usual form |

A covariance matrix latent (a Wishart or inverse Wishart) has the support `Support::CovarianceMatrix`, which NUTS
and the other gradient based algorithms move through with Stan's transform: K(K + 1)/2 unconstrained coordinates
fill a lower triangular factor L (its diagonal as logarithms) and the value is L Lᵀ, with the Jacobian from the Stan
reference manual. A Cholesky-factor latent (`wishart_cholesky`, `inverse_wishart_cholesky`) has
`Support::CovarianceCholesky`, the same factor without the product. A von Mises–Fisher latent has
`Support::UnitVector`, mapped as Stan's `unit_vector`: an unconstrained vector y normalised, with a standard normal
density on y. That map is not a bijection (every point on a ray from the origin gives the same direction), so, as
for angles, the Laplace approximation refuses such latents; NUTS samples them well.

The ordered probit and the hypergeometric have finitely many values per element, so as latents they are updated by
Gibbs steps and can be marginalised; the Skellam and the hypergeometric have one value per element, so they can be
mixture components and `log_density` kinds. The elements of a Dirichlet-multinomial draw must sum to n and those of
a ranking must be a permutation, so neither can be changed one element at a time: as latents they need a sampler
that proposes whole values.

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

### Mixtures of different families and transformed distributions

`mixture_of(components, weights)` mixes components of different kinds, such as a gamma and a log-normal: each
component is a pair of a kind and its parameter nodes, which have no component axis, and the weights' last axis has
one entry per component. The node's parents are the weights, then each component's parameters in turn. The
components can be any of the kinds `mixture` accepts (truncated, censored, zero-inflated, hurdle and transformed ones
included) as long as their values have one type. The log density, its exact gradient and Hessian-vector products,
sampling, pointwise log likelihoods and `log_density` work as for `mixture`. The support is the hull of the
components' supports: the smallest lower bound and the largest upper bound.

`transformed(base, parameters, bijector)` pushes a continuous element-wise distribution through an element-wise
bijector: the value is f(y) for a draw y of the base, and its log density is log p(f⁻¹(x)) + log |(f⁻¹)′(x)|. The
bijectors (`ElementBijector`) are `LowerBound(a)` (x = a + exp(y)), `UpperBound(b)` (x = b − exp(y)),
`Interval(a, b)` (x = a + (b − a)·logistic(y)) and `Affine { loc, scale }` (x = loc + scale·y), with constant
bounds, location and scale. So a Gaussian through `LowerBound(0.0)` is a log-normal, and through
`Interval(0.0, 1.0)` a logit-normal. The support is the image of the base's support. A transformed distribution is
element-wise, so it can be a mixture component or a `log_density` kind, but it has no CDF, so it can't be truncated
or censored. Its base can't be censored or a hurdle either, since their point masses have no Jacobian.

```rust,ignore
{{#include ../../keanu/tests/book.rs:mixture_of}}
```

The [cure-model example](examples.md#a-mixture-cure-model) mixes a censored Weibull with a cured component.

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
`bernoulli` and `bernoulli_logit`, truncated, censored, zero-inflated, hurdle and transformed ones, and mixtures of
different families, whose parameters are the weights and then each component's; not `dirichlet`,
`multivariate_gaussian`, `lkj_cholesky`, `zero_sum_normal`, `kde`, `categorical`, `multinomial`, `ordered_logistic`
or same-family mixtures), its value has the distribution's type (f64, i64 for
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

## Missing observations

Data often have gaps. Rather than dropping every row with one, the missing values can be treated as unknowns and
imputed with everything else, as PyMC does with masked observations. `observe_masked(&graph, &state, &[...])`
observes each node given as a `MaskedObservation` (its values and a mask of missing elements, or
`MaskedObservation::with_nan_missing(node, values)` with NaN marking them) and rewrites a node with gaps into three:

- `<label>_observed`, the same distribution with each parameter gathered at the observed elements, observed;
- `<label>_missing`, the same distribution gathered at the missing elements, a latent;
- `<label>`, a deterministic node of the original shape that puts every element back in its place, which every node
  that used the original reads.

The joint density is the original model's with the known elements observed. The missing parts are ordinary latents,
so `Sampler::for_model` samples continuous ones with NUTS and discrete ones with Gibbs steps, and recording
`<label>` while sampling gives draws with the imputed values in their original positions. The returned `Imputed`
record gives each node's parts (`get(id)`, with `missing_indices` for where the missing elements were) and maps
every other node (`node(id)`). The pointwise log likelihood has points for the observed elements only.

```rust,ignore
{{#include ../../keanu/tests/book.rs:observe_masked}}
```

The outcome of a regression already has a distribution. A covariate with gaps needs one too, as `x` has above, so
that its missing elements have a prior to be imputed from. The distribution must be element-wise (one value per
element with its own density: the continuous and count families, truncated, censored, zero-inflated, hurdle and
transformed ones, the Bernoulli and the categorical) with parameters that broadcast to the node's shape; others,
such as a multivariate Gaussian with a missing component, are rejected with `GraphError::CannotImpute`. A node with
no element missing is simply observed, and one with every element missing becomes `<label>_missing` alone. The
[missing data example](examples.md#missing-data) fits a regression with gaps in the outcome and a covariate.

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
