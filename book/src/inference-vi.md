# Variational inference (ADVI and Pathfinder)

Variational inference replaces sampling with optimization: it picks the member of a simple family of distributions
that is closest to the posterior, and uses that as the answer. Keanu implements **automatic differentiation
variational inference** (ADVI, Kucukelbir et al. 2017) with Stan's defaults, and **Pathfinder** (Zhang et al.
2022, [below](#pathfinder)). ADVI can subsample large data sets ([minibatches](#minibatches)). Both are usually
much faster than NUTS, and both return an approximation rather than exact draws.

## How it works

ADVI works in the same unconstrained space as NUTS (see
[Constrained latents](inference-sampling.md#constrained-latents)): each continuous latent is mapped to the real
line through the transform of its support, so a Gaussian over that space is always a valid distribution over the
latents. It fits the Gaussian by maximising the **evidence lower bound** (ELBO): the expected log density of the
model (plus the log-Jacobian of the transforms) under the Gaussian, plus the Gaussian's entropy. Each iteration
estimates the ELBO's gradient from draws ζ = μ + Lε with ε ~ N(0, I) (the reparameterisation trick) and takes an
adaptive step.

```rust,ignore
{{#include ../../keanu/tests/book.rs:advi}}
```

`fit` starts the Gaussian's mean at the latents' current values and leaves the state at the fitted mean. The
`AdviResult` holds:

- **`mean()` and `covariance()`:** the Gaussian in unconstrained space. `parameterisation().range(latent)` gives a
  latent's coordinates; `parameterisation().to_constrained(z)` maps a point back to the latents' values.
- **`elbo_trace()`:** the ELBO of the initial approximation, then one estimate every `eval_elbo` iterations. It
  should rise and level off.
- **`converged()`, `iterations()` and `eta()`:** whether the relative ELBO change fell below `tol_rel_obj`, how many
  iterations ran, and the step size scale used.
- **`sample(state, nodes, count, rng)`:** independent draws mapped back to the latents' own values, returned as
  `NetworkSamples` with the model's log probability at each draw, like the samplers' output.

## Families

- **Mean field** (`Advi::mean_field()`, the default): independent Gaussians, N(μ, diag(exp(2ω))). Cheap, but it
  cannot represent correlations, and it underestimates the marginal variance of correlated latents.
- **Full rank** (`Advi::full_rank()`): N(μ, LLᵀ) with a lower triangular L, so it captures linear correlations. It
  has d(d + 1)/2 scale parameters for d coordinates.

```rust,ignore
{{#include ../../keanu/tests/book.rs:advi_full_rank}}
```

## Settings

The defaults are Stan's:

- **`grad_samples`** (1) draws per gradient estimate and **`elbo_samples`** (100) per ELBO estimate.
- **`eval_elbo`** (100): how often the ELBO is estimated and convergence checked. The relative ELBO changes are kept
  in a buffer of the last max(0.1 · `max_iterations` / `eval_elbo`, 2), and the fit stops once their mean or
  median is below **`tol_rel_obj`** (0.01), or after **`max_iterations`** (10 000).
- **Step sizes:** η · k^(−1/2) / (1 + √s_k), elementwise, where s_k is a running average of the squared gradient.
  With **`adapt`** (true), η is picked from 100, 10, 1, 0.1 and 0.01 by running **`adapt_iterations`** (50)
  iterations with each and comparing the ELBOs; without it, **`eta`** (1) is used.

The default tolerance is loose: it can stop while the approximation is still moving. Lower `tol_rel_obj` to 0.001
when the estimates matter, and raise `grad_samples` for less noisy results.

## Minibatches

On a large data set each ADVI iteration costs a pass over all the data. With a **minibatch**, each iteration uses a
random batch of B of the N rows instead, and weighs the batch's log likelihood by N / B, so that it estimates the
full data's without bias and the gradient costs B rows rather than N. Build the model for one batch: every node
holding data (observations and covariates alike) has leading dimension B, holding any B rows at first. Then
describe the data in a `MinibatchGroup`:

```rust,ignore
{{#include ../../keanu/tests/book.rs:advi_minibatch}}
```

- **`observed(node, data)`** adds an observed node and its full data, N rows along axis 0; its log probability is
  weighted by N / B.
- **`covariate(node, data)`** adds a constant (or observed) node of covariates and its full data. Each batch sets it
  to the same rows as the observations, so row i of x stays with row i of y, and recomputes what depends on it,
  such as `intercept + slope * x`.
- **`factor(node)`** adds a factor whose term sums over the batch's rows, weighted by N / B too. Factors are never
  weighted implicitly.

Data with different numbers of rows (two data sets of a hierarchical model) go in separate groups, each with its
own batch size, combined with `Minibatch::new([group1, group2])`. Within a group, each epoch shuffles the rows
and takes them B at a time; the N mod B left over at the end of an epoch are skipped, so every batch is a uniformly
random set of B distinct rows. The weighting is exact in expectation for terms that are sums over independent rows
(a Gaussian, Poisson or Bernoulli observation, or a batch of multivariate events), not for a single multivariate
distribution over all the rows, such as a Gaussian process.

The configuration is checked when the fit starts: every node must have leading dimension B in the graph and its
data N rows with the same trailing shape and type, B must be at most N, and any latent, observation or factor that
depends on the batched data must be in the group, since its term would otherwise be a batch's but unweighted.
Minibatching supports only latents shared by every row; a latent per row (a local latent) is rejected.

The ELBO estimates are made on fresh batches, so they are noisy; convergence is judged on the running mean of the
last few estimates (as many as the change buffer holds) rather than on the latest one, and the noisy estimates
seldom meet a tight `tol_rel_obj`, so set `max_iterations` to what you can afford. The step size adaptation also
runs on minibatches, but compares the step sizes' ELBOs on one fixed batch. After the fit, the batched nodes get
back the values they had before it. `Parameterisation::set_weight` is the underlying mechanism, available to other
algorithms: it multiplies any node's term of the log probability, and its derivatives, by a weight.

Each coordinate's step is about η / √k, whatever the size of the gradient, so after k iterations the fitted mean
still moves by about that much from step to step, with or without minibatches. On a 100 000-row regression whose
coefficients have posterior standard deviations of 0.003, minibatches of 1000 rows give the coefficients to within
about 0.03 (full-batch ADVI, at 100 times the cost per iteration, is no closer) and their standard deviations to
within about 40%.

## Pathfinder

Pathfinder (Zhang, Carpenter, Gelman and Vehtari, 2022) is quasi-Newton variational inference. It runs L-BFGS
towards the mode of the unconstrained log density, keeping every point of the optimization path, and at each point
forms a Gaussian approximation of the posterior from what L-BFGS has learnt about the curvature: its covariance is
the L-BFGS estimate of the inverse Hessian (a diagonal plus a low-rank part built from the last six steps and
gradient changes), and its mean is the Newton step from the point. The approximation with the highest ELBO
(estimated from five draws each) is the path's answer. One path costs about as many gradient evaluations as
finding the mode, far fewer than ADVI's stochastic optimization.

```rust,ignore
{{#include ../../keanu/tests/book.rs:pathfinder}}
```

- **`single_path`** runs one path from the latents' current values and returns `draws` (1000) draws from its best
  approximation.
- **`multi_path`** (the usual choice) runs `paths` (4) paths, each from a random starting point (every
  unconstrained coordinate uniform on (−2, 2), as Stan initialises), pools their draws and resamples 1000 of them
  with Pareto-smoothed importance weights (see [Model comparison](model-comparison.md)): each draw's log density
  minus the log density of the approximation it came from. Several paths guard against a path that stops in a poor
  region, and the weights correct part of the Gaussians' error. Paths that fail are skipped and counted.

The `PathfinderResult` holds the draws in unconstrained space (`draws()`, with `log_density()` and
`approximation_log_density()`), `sample(state, nodes)` to map them to the latents' values as `NetworkSamples`,
each path's best approximation (`paths()`: its `mean()`, `covariance()`, `elbo()`, `elbo_trace()` and the iteration
it came from), and **`pareto_k()`**, the k̂ of the importance weights. Above about 0.7 (`is_reliable()` is false)
the approximation is too far from the posterior for importance sampling to fix, and the draws are only a rough
guide. Settings: `lbfgs` (an `Lbfgs` with history 6 and at most 1000 iterations), `elbo_draws` (5), `draws`
(1000), `paths` (4) and `resample` (true).

How good the approximation is depends on the model. On a correlated Gaussian, multi-path Pathfinder's draws match
the posterior's means and covariances closely; a single path's are only roughly right, because the best L-BFGS
approximation is not exact even there. Where the mode of the unconstrained density is far from the posterior's bulk
the approximations sit near the mode: in non-centred eight schools the mode has τ ≈ 29 (log τ ≈ 3.4) while the
posterior mean of τ is 3.6, k̂ is 0.5 to 1.1, and the means are within about one posterior standard deviation of
NUTS's. In the centred radon model the density grows without bound as the county intercepts' spread shrinks to 0,
L-BFGS heads that way, and k̂ is about 2.

### Starting NUTS from Pathfinder

Pathfinder's draws make good starting points for MCMC chains, each from its own point near the typical set rather
than all from the state's values. `initial_states(state, count, rng)` gives `count` copies of the state with the
latents set to distinct draws, and `sample_chains_from` (or `run_chains_from`) runs one chain from each:

```rust,ignore
{{#include ../../keanu/tests/book.rs:pathfinder_nuts}}
```

On the radon model, NUTS chains started this way converge (R-hat below 1.01 on all 89 parameters) with a 150-step
warm-up instead of the default 1000; there, chains from the default starting values do as well with the same
short warm-up, because those values are already close to the posterior. The gain is largest when the starting
values are poor or the warm-up is spent finding the typical set.

## Limitations

- **Continuous latents only.** A model with discrete latents is rejected; observe them, or sum them out (for
  example with a mixture), first.
- **A Gaussian in unconstrained space.** It cannot represent skewed or multimodal posteriors there, and a mean-field
  fit is overconfident about correlated latents. For a hierarchical model such as eight schools, use the
  non-centred form, as for NUTS.
- **Check with NUTS.** Use ADVI or Pathfinder for quick estimates, large models or starting values, and confirm
  results that matter with a sampler.

An initial approximation whose ELBO is not finite (a draw with zero probability) is an error, as is a fit whose
draws stop being finite part way through.
