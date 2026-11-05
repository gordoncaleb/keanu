# Variational inference (ADVI)

Variational inference replaces sampling with optimization: it picks the member of a simple family of distributions
that is closest to the posterior, and uses that as the answer. Keanu implements **automatic differentiation
variational inference** (ADVI, Kucukelbir et al. 2017) with Stan's defaults. It is usually much faster than NUTS,
and it returns an approximation rather than exact draws.

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

## Limitations

- **Continuous latents only.** A model with discrete latents is rejected; observe them, or sum them out (for
  example with a mixture), first.
- **A Gaussian in unconstrained space.** It cannot represent skewed or multimodal posteriors there, and a mean-field
  fit is overconfident about correlated latents. For a hierarchical model such as eight schools, use the
  non-centred form, as for NUTS.
- **Check with NUTS.** Use ADVI for quick estimates, large models or starting values, and confirm results that
  matter with a sampler.

An initial approximation whose ELBO is not finite (a draw with zero probability) is an error, as is a fit whose
draws stop being finite part way through.
