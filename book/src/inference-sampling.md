# Posterior sampling (MCMC)

Posterior distributions are rarely available in closed form. Markov chain Monte Carlo (MCMC) algorithms draw
samples whose distribution approaches the posterior, and from those you can estimate means, variances, probabilities
of events and so on.

## A model to sample

```rust,ignore
{{#include ../../keanu/tests/book.rs:sampling_model}}
```

`probe_for_non_zero_probability` samples latents without values from their priors, and keeps resampling until the
state has non-zero probability, which every sampler needs to start from.

## Metropolis–Hastings

Each step proposes new values for some of the latents and accepts or rejects them so that, over many steps, the
samples follow the posterior.

```rust,ignore
{{#include ../../keanu/tests/book.rs:metropolis_hastings}}
```

Settings on `MetropolisHastings`:

- **`proposal`:** how new values are proposed. `PriorProposal` (the default) samples from each latent's prior given
  its parents; `GaussianProposal::new(sigma)` takes Gaussian random-walk steps, with optional per-node sigmas.
- **`selector`:** which latents each step changes. `VariableSelector::Single` (the default) cycles through them one
  at a time; `Full` changes all of them.
- **`rejection`:** how a rejected proposal is undone (`RollbackToCachedValues` or `RollbackAndCascade`; they give
  the same chain).

`SamplingOptions` sets how many initial steps to drop (burn-in) and how often to record a sample (down-sampling).

## NUTS

The No-U-Turn sampler is Hamiltonian Monte Carlo that tunes its own trajectory length. It uses gradients, so it
mixes far better than Metropolis–Hastings on continuous models, especially with correlated latents. It needs every
latent to be continuous and the graph to be differentiable; models with discrete latents use it for their continuous
latents inside a compound sampler (below).

```rust,ignore
{{#include ../../keanu/tests/book.rs:nuts}}
```

Settings on `Nuts`:

- **Adaptation:** `adapt_count` (the steps used to adapt, Stan's `num_warmup`; default 1000), `adapt_step_size`,
  `initial_step_size` and `target_acceptance_prob` (default 0.8, as Stan's `delta`).
- **Mass matrix:** `potential` (a `QuadraticPotential`: the metric, the initial variance and the adaptation
  schedule) and `adapt_potential`.
- **Trajectories:** `max_tree_height`, and `max_energy_change` (the threshold for marking a trajectory divergent).
- **Diagnostics:** `save_statistics`, used with `sample_with_statistics` for per-step step sizes, acceptance rates,
  log probabilities and tree sizes. Divergences and energies are recorded either way (see below).
- **Space:** `space` (see below).

A `Nuts` value holds settings only, so it can be reused and shared between threads.

NUTS adapts as Stan does, with Stan's defaults. The step size is tuned by dual averaging towards the target
acceptance probability, starting from a reasonable step size found by doubling or halving `initial_step_size`. The
diagonal mass matrix is estimated in windows over the adaptation steps: the first 75 steps tune only the step size
while the chain finds the typical set; then come slow windows of 25, 50, 100, ... steps (the last stretched to fill
the space left) in which the draws' variances are also estimated, and at the end of each window the mass matrix is
set from them (regularised towards a small value), a new step size is found and dual averaging restarts; the last
50 steps tune only the step size again. `QuadraticPotential`'s `initial_buffer`, `adaptation_window` and
`terminal_buffer` set the three lengths. With fewer than 20 adaptation steps only the step size is adapted, and when
the three stages don't fit they take 15%, 75% and 10% of the steps instead. After adaptation the step size and mass
matrix stay fixed, so drop the adaptation steps (`SamplingOptions::with_drop(adapt_count)`) from the draws.

### Dense mass matrix

The default mass matrix is diagonal (Stan's `diag_e`): it learns each latent element's scale, but not correlations
between elements. A posterior with strong linear correlations (regression coefficients of uncentred predictors,
basis-function weights, the parameters of a growth curve) forms a narrow ridge, and with a diagonal metric NUTS has
to take a small step size and long trajectories to follow it. `QuadraticPotential::with_metric(Metric::Dense)`
adapts the whole inverse mass matrix instead (Stan's `dense_e`): in each slow window it estimates the covariance Σ
of the draws, regularised as Stan does to (n / (n + 5)) · Σ + 10⁻³ · 5 / (n + 5) · I for n draws, and draws
momentum through its Cholesky factor, so the ridge looks round to the integrator.

```rust,ignore
{{#include ../../keanu/tests/book.rs:nuts_dense}}
```

On a 2-D Gaussian with correlation 0.99 the dense metric gets over 10 times the effective sample size per gradient
evaluation of the diagonal one. It costs O(n²) per leapfrog step for n latent elements (the diagonal metric O(n)),
on top of the gradient, and a window needs more draws than there are elements to estimate it; so it suits models
of up to a few hundred correlated elements, and the default suits large models with weakly correlated latents.
`NutsStatistics::dense_inverse_metric` holds the adapted matrix (and `inverse_metric` its diagonal), and
[CmdStan CSV export](#exporting-to-arviz-and-stan-tools) writes it as CmdStan writes a dense metric. A dense NUTS
block works inside a compound sampler like any other (`Compound::nuts_with`).

### Constrained latents

Many latents can only take some values: a scale is positive, a probability is between 0 and 1, a Dirichlet draw is
on the simplex. By default NUTS samples each latent in **unconstrained space**: it maps the latent to the real line
through a transform chosen from its distribution's support (log for a lower bound, logit for an interval, the
isometric log-ratio for a simplex, and so on), moves there, and maps every draw back. The log-Jacobian of the
transforms is added to the log probability, so the draws still follow the posterior. Every point of a trajectory is
inside the support, so trajectories never diverge by stepping out of it, and latents near a boundary (a small
scale) mix well.

Draws, the state the sampler leaves and the recorded log probabilities are all in terms of the latents' own values;
the transforms only change how NUTS moves.

```rust,ignore
{{#include ../../keanu/tests/book.rs:nuts_transforms}}
```

`with_space(Space::Constrained)` opts out, moving in the latents' own values as earlier versions of Keanu did (and
giving the same draws for the same seed). Opting out is needed when a latent's bounds depend on another latent, for
example `x ~ Uniform(0, hi)` with a latent `hi`: a fixed transform can't follow a moving bound, so the default fails
with an error suggesting a reparameterisation (`u ~ Uniform(0, 1)` and `x = hi · u`) or `Space::Constrained`. A
starting value outside its support, or on its boundary, is an impossible start in either space.

Metropolis–Hastings always works on the latents' own values: its proposals are defined on them.

### Divergences and energy

`sample_with_statistics` returns `NutsStatistics` alongside the samples. Two of its diagnostics are always
recorded, whatever `save_statistics` says:

- **Divergences.** A trajectory is divergent when its Hamiltonian rises more than `max_energy_change` above
  where it started (as in Stan; a fall in energy is not a divergence): the leapfrog integrator has gone unstable, usually in a region of high curvature such as the neck
  of a funnel. The chain can't explore such a region properly, so its draws may be biased even when R-hat and the
  effective sample size look fine. `divergent_steps` lists the divergent steps after adaptation,
  `divergence_count()` counts them and `divergent_flags()` gives one flag per step. Any divergence is worth
  investigating. The usual cure is to reparameterise: a hierarchical model such as eight schools, with
  `theta ~ N(mu, tau)`, diverges as `tau` shrinks, while its non-centred form, `eta ~ N(0, 1)` and
  `theta = mu + tau · eta`, samples cleanly. Raising `target_acceptance_prob` (smaller steps) sometimes helps too.
- **Energy.** `energy` holds the Hamiltonian at each step's sample, adaptation steps included. Its *energy Bayesian
  fraction of missing information*, `diagnostics::e_bfmi` (Betancourt, 2016), is
  Σ(Eₙ − Eₙ₋₁)² / Σ(Eₙ − Ē)²: how far the energy moves between draws compared with how far it varies overall. Values
  below `E_BFMI_THRESHOLD` (0.3) mean resampling the momentum can't move the chain through the energy levels quickly,
  typically because of heavy tails, and the tails are poorly explored. Pass only the energies after adaptation.

```rust,ignore
{{#include ../../keanu/tests/book.rs:nuts_diagnostics}}
```

Both are per step. With `SamplingOptions` dropping `d` steps and recording every `k`-th, draw `i` is step
`d + i·k`. With several chains (below), check every chain.

## Discrete latents: Gibbs and compound sampling

Discrete latents (a change point, a mixture assignment, an inclusion indicator) have no gradients, so NUTS can't
move them. A **compound sampler** (`Compound`) splits the latents into blocks and cycles through one step per block,
each updating its block with everything else held fixed, as PyMC's compound steps do:

- **NUTS** for a block of continuous latents. In this block mode NUTS treats the other latents as constants and,
  whenever another step has changed them, refreshes its position, log probability and gradient before its next
  trajectory. Its step size and mass matrix adapt over its first `adapt_count` steps as usual, which are the compound
  sampler's first `adapt_count` steps, so drop them from the draws.
- **`Gibbs`** for discrete latents whose elements have finitely many values: Bernoulli, binomial, categorical,
  ordered logistic and discrete uniform (and truncated counts with a finite upper bound). It updates one element at a
  time, evaluating the log probability at every value the element can take and drawing from those exact
  conditional probabilities. Only the latent's Markov blanket (its own log probability, its children's and the
  factors it reaches) is evaluated. An element with more than `max_support_size` values (1000) is an error.
- **`DiscreteMetropolis`** for integer latents with infinitely many values (Poisson, negative binomial, geometric):
  a random walk that moves each element by ±1 or ±2 and accepts with the Metropolis probability.
- **Metropolis–Hastings** for anything else.

```rust,ignore
{{#include ../../keanu/tests/book.rs:compound}}
```

`Compound::for_model` chooses the steps as above, with NUTS last; `with_nuts_settings` changes the settings of its
NUTS step. Every latent must belong to exactly one step. `sample_with_statistics` returns `CompoundStatistics`,
whose `nuts` holds each NUTS step's `NutsStatistics` (divergences, energies, adaptation; pass it to `CmdStanCsv`
like plain NUTS's) and `discrete_acceptance_rate` each random walk's acceptance rate. `ChainSamples::rhat` and the
other diagnostics accept integer and boolean nodes too, so discrete latents can be checked for convergence.

Gibbs and `DiscreteMetropolis` can also sample a model of discrete latents alone:

```rust,ignore
{{#include ../../keanu/tests/book.rs:gibbs}}
```

Each element update evaluates the whole of each node in the latent's Markov blanket, so a sweep over a latent of n
elements costs about n such evaluations per value tried. Where a discrete latent can be summed out of the model
instead, the result (sampled by NUTS alone) usually mixes better still.

## Choosing automatically

`Sampler::for_model` picks a sampler from the kinds of latent:

- every latent continuous: NUTS if the log probability is differentiable with respect to them, otherwise
  Metropolis–Hastings;
- every latent discrete, with finitely many values per element: `Gibbs`;
- otherwise: `Compound::for_model` (typically Gibbs or `DiscreteMetropolis` for the discrete latents and NUTS for
  the continuous ones).

Before the compound sampler existed, any model with a discrete latent got Metropolis–Hastings for every latent;
`MetropolisHastings` asked for explicitly still works as before.

```rust,ignore
{{#include ../../keanu/tests/book.rs:choose_sampler}}
```

## Several chains

Because the graph is shared and each chain has its own state, `algorithms::mcmc::chains::run_chains` runs
independent, seeded chains (`sample_chains` collects their draws into `ChainSamples`), with any of the samplers. With the `parallel` feature they run on rayon's thread pool, with the same results as
running them one after another.

## Sequential Monte Carlo

Tempered sequential Monte Carlo (`Smc`), which samples a population of particles from the prior to the posterior
and estimates the marginal likelihood as it goes, and its use for simulator models without a likelihood
(approximate Bayesian computation), have their own chapter: [Simulation-based inference](simulation-based-inference.md).

## Predictive sampling

`PredictiveSampler` draws observed nodes as the model would generate them. `prior` samples the latents from their
priors and then the observations (the prior predictive distribution, for checking that the priors produce
plausible data). `posterior` takes posterior samples and, for each, draws the observations given those latent
values (the posterior predictive distribution, for checking the fitted model against the data). Both return
`NetworkSamples` of the requested nodes and work on a copy of the state, so the observations keep their values.

```rust,ignore
{{#include ../../keanu/tests/book.rs:predictive}}
```

The targets must be observed probabilistic nodes. Other observations stay fixed, so for a posterior draw the
samples need to hold only the latents the targets depend on; for a prior draw, an observation that a target depends
on must not itself depend on a latent (make it a target too). `with_draws(n)` uses `n` evenly spaced posterior
samples instead of all of them.

## Pointwise log likelihood

Model comparison by leave-one-out cross-validation or WAIC, and tools such as ArviZ, need the log likelihood of
each observation at each posterior draw. `PointwiseLogLik::from_chains` (or `from_samples` for one chain, or for the
draws of the Laplace approximation or ADVI) computes it after sampling, by setting each draw's latents in a copy
of the state and evaluating the observed nodes, so it works for every sampler. The samples must record every latent
the observations depend on.

```rust,ignore
{{#include ../../keanu/tests/book.rs:pointwise_log_likelihood}}
```

There is one point per element of an observed node whose distribution has one value per element (named `y[3]`,
or `y[1, 2]` for a matrix), as ArviZ labels them, and one per node for the others (a multivariate Gaussian, a hidden
Markov model). Factors are not observations, so they have no points. With the `parallel` feature the draws are
evaluated on rayon's thread pool.

The pointwise log likelihood is what [model comparison](model-comparison.md) by PSIS-LOO and WAIC needs.

## Exporting to ArviZ and Stan tools

`CmdStanCsv` writes each chain in the CmdStan CSV format, which ArviZ's `from_cmdstan`, R's `posterior` (through
`cmdstanr` or `rstan::read_stan_csv`) and other Stan tools read: comment lines with the run's settings, a header,
the adaptation results (step size and the diagonal of the inverse mass matrix, or the whole matrix one row per line
for a dense metric) and a line per draw.

```rust,ignore
{{#include ../../keanu/tests/book.rs:cmdstan_export}}
```

Columns follow Stan's conventions: a scalar is its node's label, and the elements of a tensor are the label with
1-based indices joined by dots (`theta.1`, `L.2.1`), written in column-major order as Stan writes matrices. The
sampler columns are `lp__` for every sampler and, for NUTS with saved statistics, `accept_stat__`, `stepsize__`,
`treedepth__`, `n_leapfrog__`, `divergent__` and `energy__`. Pass the `SamplingOptions` the draws were recorded with
so that each draw is matched with its step's statistics. Keanu's `lp__` is the model's log density at the
constrained values, without the Jacobian Stan includes, and the inverse mass matrix is in NUTS's unconstrained
coordinates.

In Python, `arviz.from_cmdstan(["output_1.csv", ...], log_likelihood="log_lik")` reads the files into an
`InferenceData` with posterior, sample statistics and log likelihood groups.

## Simulated annealing

`SimulatedAnnealing` uses Metropolis–Hastings steps at a decreasing temperature to find the most probable state. It
works for discrete latents, where the gradient optimizers can't be used.
