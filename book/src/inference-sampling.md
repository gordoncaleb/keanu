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

A `Nuts` value holds settings only, so it can be reused and shared between threads.

## Choosing automatically

```rust,ignore
{{#include ../../keanu/tests/book.rs:choose_sampler}}
```

## Several chains

Because the graph is shared and each chain has its own state, `algorithms::mcmc::chains::run_chains` runs
independent, seeded chains. With the `parallel` feature they run on rayon's thread pool, with the same results as
running them one after another.

## Simulated annealing

`SimulatedAnnealing` uses Metropolis–Hastings steps at a decreasing temperature to find the most probable state. It
works for discrete latents, where the gradient optimizers can't be used.
