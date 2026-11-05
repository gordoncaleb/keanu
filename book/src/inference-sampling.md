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

- **Adaptation:** `adapt_count` (the samples used to adapt), `adapt_step_size`, `initial_step_size` and
  `target_acceptance_prob`.
- **Mass matrix:** `potential` and `adapt_potential`.
- **Trajectories:** `max_tree_height`, and `max_energy_change` (the threshold for marking a trajectory divergent).
- **Diagnostics:** `save_statistics`, used with `sample_with_statistics` for per-step step sizes, tree sizes and
  divergences.

- **Space:** `space` (see below).

A `Nuts` value holds settings only, so it can be reused and shared between threads.

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

## Simulated annealing

`SimulatedAnnealing` uses Metropolis–Hastings steps at a decreasing temperature to find the most probable state. It
works for discrete latents, where the gradient optimizers can't be used.
