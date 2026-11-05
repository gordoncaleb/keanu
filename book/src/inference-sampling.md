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
latent to be continuous and the graph to be differentiable.

```rust,ignore
{{#include ../../keanu/tests/book.rs:nuts}}
```

Settings on `Nuts`:

- **Adaptation:** `adapt_count` (the steps used to adapt, Stan's `num_warmup`; default 1000), `adapt_step_size`,
  `initial_step_size` and `target_acceptance_prob` (default 0.8, as Stan's `delta`).
- **Mass matrix:** `potential` (a `QuadraticPotential`: the initial variance and the adaptation schedule) and
  `adapt_potential`.
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

## Choosing automatically

```rust,ignore
{{#include ../../keanu/tests/book.rs:choose_sampler}}
```

## Several chains

Because the graph is shared and each chain has its own state, `algorithms::mcmc::chains::run_chains` runs
independent, seeded chains. With the `parallel` feature they run on rayon's thread pool, with the same results as
running them one after another.

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

## Simulated annealing

`SimulatedAnnealing` uses Metropolis–Hastings steps at a decreasing temperature to find the most probable state. It
works for discrete latents, where the gradient optimizers can't be used.
