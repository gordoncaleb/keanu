# Sequences (plates)

A sequence repeats a group of nodes, for example one group per time step, per agent or per data point. Nodes in one
group can depend on nodes in the previous group through **proxies**: placeholders that are connected to the previous
item's node with the same label.

```rust,ignore
{{#include ../../keanu/tests/book.rs:sequence}}
```

- `with_initial_state(label, node)` gives the first item's proxies something to connect to.
- `count(n, factory)` builds n items; `from_iter(data, factory)` builds one per data element.
- `double_proxy_for(label)` (and the `int_` and `bool_` forms) creates a proxy for the previous item's node.
- `with_transition` connects proxies to differently labelled nodes.

Each item's nodes are labelled in their own namespace (`Sequence_Item_<index>.<sequence id>.<label>`, prefixed with
the sequence's name if it has one), so they can be found after building or loading.

## Hidden Markov models and state-space models

Two kinds of time series have distributions of their own, whose log density is the exact marginal likelihood of the
observations with the hidden states summed or integrated out. Gradient based algorithms (NUTS, the gradient
optimizer) then work on the continuous parameters directly, with no per-step latent nodes.

**Hidden Markov models.** `hidden_markov_model(steps, emission, parameters, initial, transition)` observes a hidden
chain of K states: the first state is drawn from the `initial` probabilities ([K]), each later one from the row of
the `transition` matrix ([K, K]) for the state before it, and each step's observation from the `emission`
distribution with that state's parameters. The emission is any element-wise distribution kind (as for a mixture),
continuous or discrete, and its parameters carry a trailing state axis.

```rust,ignore
{{#include ../../keanu/tests/book.rs:hmm}}
```

- `initial` and `transition` are probabilities, not log probabilities: non-negative and summing to 1 along their last
  axis (within 1e-5). Give each row of the transition matrix a Dirichlet prior and stack the rows: they stay on the
  simplex under the gradient based algorithms. Probabilities off the simplex give a log probability of −∞ (so
  Metropolis–Hastings rejects a proposal of them), and an error when sampled.
- The value has shape [steps]; `hidden_markov_model_with_shape` takes [T, ...] for several observations per step, all
  emitted by that step's state. An emission parameter may have a leading time axis for parameters that vary over
  time.
- The log density is computed by the log-space forward algorithm in O(T K²) and its gradient by the backward
  algorithm, both exact. Drawing a value simulates the chain and then each observation. For the chain alone,
  `keanu_distributions::HiddenMarkovChain` gives the log likelihood of any per-state emission log probabilities, the
  smoothed state marginals, the Viterbi path and simulated state paths (see below).

**Linear-Gaussian state-space models.** `linear_gaussian_state_space(steps, initial_mean, initial_covariance,
transition_matrix, transition_covariance, observation_matrix, observation_covariance)` is the model of the Kalman
filter: a hidden state x of dimension d with x₀ ~ N(initial mean, initial covariance) and
x_{t+1} = F x_t + N(0, Q), observed as y_t = H x_t + N(0, R) in dimension m. The value is the observations, [steps, m].

```rust,ignore
{{#include ../../keanu/tests/book.rs:kalman}}
```

- Shapes: the initial mean [d]; the initial covariance, F and Q [d, d]; H [m, d]; R [m, m]. Mismatches are build
  errors. The covariances must be symmetric positive semidefinite; an innovation covariance H P Hᵀ + R that is not
  positive definite makes the observations impossible.
- The log likelihood uses Cholesky solves of the innovation covariance and the Joseph-form covariance update, which
  stays symmetric positive semidefinite over long series and near-deterministic models. Its gradient with respect to
  the observations and every parameter is exact (the filter differentiated in reverse mode), and so are its second
  derivatives (that reverse pass differentiated in forward mode), so Kalman models work with the Laplace
  approximation and anything else that needs Hessians. Covariance gradients are symmetric, as for the multivariate
  Gaussian.
- `keanu_distributions::LinearGaussianStateSpace::filter` returns the filtered means and covariances at each step.

## Smoothing, decoding and forecasting

The hidden states of both models are summed or integrated out of the likelihood, so a sampler draws only the
parameters. The states, and the future, are recovered afterwards: each draw of the parameters gives them an exact
conditional distribution, and repeating the computation over the draws gives their posterior with the parameters'
uncertainty included. `keanu_algorithms::timeseries` does this for a node, evaluating its parameters at each draw
(deterministic parents included), so the samples must record every latent the parameters depend on.

```rust,ignore
{{#include ../../keanu/tests/book.rs:smoothing}}
```

**State-space models.** `StateSpacePosterior::new(&model, &state, node, &samples)` (or `from_chains` for
`ChainSamples`) holds each draw's model, and gives at each draw:

- `smooth`: the Rauch–Tung–Striebel smoother's means and covariances of the states given every observation. It runs
  the same Joseph-form filter as the log density and keeps its covariances positive semidefinite; a singular
  predicted covariance (a part of the state that is known exactly) uses its pseudo-inverse. `mean_states` averages
  the smoothed means over the draws.
- `sample_states`: a draw of the whole state path from its posterior, by forward filtering and backward sampling,
  so that states and parameters together are draws from their joint posterior.
- `forecast(h)`: the Gaussian predictive distribution of the next h states and observations, from the last filtered
  state, for a time-invariant model; `sample_forecast(h, rng)` simulates one future path per draw, jointly over its
  steps (for intervals of totals or maxima). `forecast_with` and `sample_forecast_with` take each future step's
  matrices (`FutureStep`) from a closure of the draw, for time-varying models or known future inputs.

The same methods exist at fixed parameter values on `keanu_distributions::LinearGaussianStateSpace`: `smooth`,
`sample_posterior_states`, `forecast`, `forecast_with`, `sample_forecast` and `sample_forecast_with`.

**Hidden Markov models.** `HiddenMarkovPosterior::new(&model, &state, node, &samples)` gives each draw's
`viterbi` path (the most probable state sequence, with its log joint probability), `posterior_marginals`
(p(z_t = j | x), [T, K], by the forward-backward algorithm), their average over the draws `mean_marginals`, and
`sample_states`, a posterior path per draw by forward filtering and backward sampling. At fixed parameters,
`keanu_distributions::HiddenMarkovChain` has `viterbi`, `posterior_marginals` and `sample_posterior_states`, taking
each step's emission log probabilities under each state, which `DistributionKind::hidden_markov_chain` computes for a
node's value and parameters. The [HMM example](examples.md) decodes its regimes this way.

## Structural time series

A structural time series models a series as a sum of interpretable parts, each a small state-space model, plus
observation noise, as TensorFlow Probability's `tfp.sts` does. `keanu_graph::sts::StructuralModel` assembles the
components into one linear-Gaussian state-space node (their states stacked into one state vector, with
block-diagonal transition and noise matrices), so the Kalman filter integrates every component's states out and a
sampler draws only the scales and coefficients. `keanu_algorithms::sts::StructuralPosterior` then decomposes the
series into its components and forecasts it over the posterior draws.

```rust,ignore
{{#include ../../keanu/tests/book.rs:sts}}
```

The components (`Component::...`), each named for the decomposition:

| Component | State | Parameters |
|---|---|---|
| `local_level()` | a random-walk level | `level_scale` |
| `local_linear_trend()` | a level whose slope is a random walk | `level_scale`, `slope_scale` |
| `seasonal(S)` | the last S − 1 seasonal effects; S consecutive effects sum to zero apart from drift | `drift_scale` |
| `trigonometric_seasonal(P, J)` | J harmonics of period P steps (P need not be whole), a rotating pair each | `drift_scale` |
| `autoregressive(p)` | the last p values of an AR(p) process | `coefficients` [p], `scale` |
| `regression(X)` | none: an offset X β of the observations, X of shape [T, k] | `weights` [k] |

- **Parameters** are graph variables. Supply your own with `with_scale`, `with_slope_scale`, `with_coefficients`
  and `StructuralModel::with_observation_noise_scale` (a latent with your prior, or a constant to fix it), or let
  the builder create latents with default priors scaled by the series' standard deviation s: LogNormal(ln(0.05 s), 2)
  for each component scale, LogNormal(ln(0.1 s), 2) for the observation noise, N(0, 1) for AR coefficients and
  N(0, 10 s / max |x_j|) for regression weights. The created latents are labelled `"<component>.<parameter>"`;
  `StructuralTimeSeries::parameters` lists every parameter's node, which is what to record when sampling.
- **The initial state** is fixed: the level of the first local level or local linear trend component starts near
  the first observation, N(y₀, s²), and every other state at N(0, s²), a second level included, so that the levels
  sum to y₀. With a regression the level still starts at y₀, the first residual at the default weights' prior
  mean of 0.
- **Seasons are one step long**, because the state-space node is time-invariant. TFP's `num_steps_per_season`
  is the trigonometric form with the period in steps (52.18 for weekly data with a yearly cycle, or 7 × 24 for
  hourly data with a weekly one).
- **A regression** makes the likelihood a [factor](building-models.md#factors) of the state-space log density of the series minus
  X β (`log_density` accepts the `linear_gaussian_state_space` kind, giving the scalar Kalman log likelihood of a
  whole series), since an observed node can't have its observations shifted by a latent. Its gradients and second
  derivatives are exact as for the node.

`StructuralPosterior::new(&model, &state, &sts, &samples)` (or `from_chains`) evaluates the parameters at each draw:

- `decompose(level)`: each component's contribution to every observed step (its observation row times its smoothed
  states, or X β), and their sum `total`, the smoothed mean of the series. Means are averaged over the draws and the
  intervals of probability `level` are central quantiles of the mixture over draws of each draw's exact Gaussian,
  so they include the parameters' and the states' uncertainty. The components' means sum to the total's.
- `forecast(h, &future_covariates, level)`: the next h steps of the series (with the observation noise) and of
  each component, with one future design matrix [h, k] per regression component; `sample_forecast` draws one
  joint future path per draw.

The [CO₂ forecasting example](examples.md) fits a trend and a 12-month seasonal component with NUTS and checks a
two-year forecast against held-out months.

## Random walks, autoregressions and GARCH

Three processes defined by a recursion have distributions of their own, so a whole series is one node with an exact
log density, gradient and Hessian-vector products (no node per step). Each follows PyMC's distribution of the same
name. The value is a series with **time on its last axis**, [..., T]; leading axes are a batch of independent series,
and each parameter is a scalar or broadcasts to the batch shape.

```rust,ignore
{{#include ../../keanu/tests/book.rs:processes}}
```

- `gaussian_random_walk(T, mu, sigma, init_mu, init_sigma)`: x₀ ~ N(init_mu, init_sigma) and
  x_t = x_{t−1} + N(mu, sigma), so `mu` is the drift. T counts every value, so PyMC's `steps` is T − 1.
- `autoregressive(T, constant, rho, sigma, init_mu, init_sigma)`: AR(p) with `rho` the coefficients of lags 1 to p
  ([p], or [..., p] for a batch): the first p values are N(init_mu, init_sigma) and each later one
  N(constant + Σᵢ ρᵢ x_{t−i}, sigma). The process is conditioned on those first values rather than started from its
  stationary distribution, and the coefficients are not required to be stationary. For a stationary AR(1), pass
  init_mu = c / (1 − ρ) and init_sigma = σ / √(1 − ρ²) as nodes of ρ and σ.
- `garch11(T, omega, alpha, beta, initial_vol)`: r_t ~ N(0, σ_t) with σ₀ = initial_vol and
  σ_t² = ω + α r_{t−1}² + β σ_{t−1}². The parameters must satisfy ω > 0, α ≥ 0, β ≥ 0, α + β < 1 (covariance
  stationarity) and initial_vol > 0; outside that region the series is impossible (log probability −∞, which NUTS
  treats as a divergence) and drawing it is an error. A prior that stays inside, such as the first two coordinates
  of a Dirichlet above, avoids the boundary.

The `_with_shape` forms take a batch shape. `keanu_distributions::{GaussianRandomWalk, Autoregressive, Garch11}`
give the log densities, gradients, Hessian-vector products and simulation directly.

## Scan: a loop in one node

A sequence adds one group of nodes per item, so building a 10,000-step model makes tens of thousands of nodes. When
each step is a *deterministic* function of the previous step's state, the per-step inputs and some constants, a
**scan** expresses the loop as a single node holding the loop body as a nested graph:

```rust,ignore
{{#include ../../keanu/tests/book.rs:scan}}
```

`b.scan(init, &xs, &constants, body)` calls `body` once, on a fresh builder, with placeholders for the carry, for
step t of each per-step input (sliced along its first axis) and for each constant, and the body returns the next
carry. The scan's value stacks the carry after each step, shape `[T, carry shape...]`. `b.scan_steps(T, init,
&constants, body)` is the same loop without per-step inputs (T may be 0, giving an empty value).

- The body is deterministic. Random variables come from outside: draw a vector latent for the per-step noise and pass
  it as a per-step input, as with `eps` above. A body containing a distribution is rejected when building.
- The body only sees its placeholders. Values from the enclosing graph enter as constants; using an outer handle in
  the body doesn't compile.
- There is one carry tensor. Carry several values by packing them into one tensor with `stack` (and `slice` to read
  them back) — for example `[level, trend]` for a local linear trend.
- Scans are differentiable to first and second order (NUTS, optimizers, ADVI, Laplace) through the initial carry, the
  per-step inputs and the constants. Reverse mode recomputes each step's body from the stacked output instead of
  storing it, so memory stays proportional to T. Scans nest, and save, load and export to DOT (the body is drawn as
  a cluster).

Use a **scan** for long deterministic recursions (autoregressive means, state-space filters written by hand, ODE
solvers stepped explicitly): graph building is constant in T and the model stays small. Use an **unrolled sequence**
when a step's random variables have distributions that depend on the previous step and can't be rewritten as
independent noise passed in (a Poisson count whose rate is the previous count, a discrete state chain), when you want
each step's nodes labelled in the graph, and for the [particle filter](particle-filter.md), which adds
observations and samples latents one step at a time and so needs the steps as separate nodes.
