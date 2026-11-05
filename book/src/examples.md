# Examples

These live in `keanu/examples`. Run one with `cargo run --release --example <name>`. Each is also an
integration test (`keanu/tests/examples.rs`): the examples ported from Java check the same results the Java
examples' tests check, and the hierarchical ones check multi-chain R-hat, divergences and posterior means.
The Gaussian process, stochastic volatility and hidden Markov model examples check their results against the
functions or parameters their synthetic data were generated from.

| Example | Question |
|---|---|
| `coal_mining_disasters` | When did the rate of coal mining disasters change? |
| `text_messaging` | When did someone's text messaging rate increase? |
| `ab_testing` | Is treatment A more likely to drive purchases than treatment B? |
| `cheating_students` | What proportion of students cheated, from a privacy-preserving survey? |
| `challenger_disaster` | Does temperature predict O-ring failure? |
| `custom_distribution` | A hierarchical model with user-defined distributions, reparameterised to sample well |
| `eight_schools` | How much do coaching effects differ between schools? Centred and non-centred forms |
| `radon` | How much does indoor radon vary between Minnesota counties? A varying-intercept model |
| `gaussian_mixture` | Which of two groups does each measurement come from, and what are the groups like? |
| `gp_regression` | What smooth function produced noisy measurements? Gaussian process hyperparameters by NUTS |
| `stochastic_volatility` | How volatile are daily returns, and how persistent is the volatility? A scan over latent innovations |
| `hmm` | When does a series switch regime, and how often? A two-state hidden Markov model |

The first five come from *Bayesian Methods for Hackers* by Cameron Davidson-Pilon; `eight_schools` and `radon` are
classic hierarchical models from Gelman et al. `gaussian_mixture` is checked against the mixture its synthetic data
are drawn from, with four chains and R-hat below 1.01.

## Coal mining disasters

```rust,ignore
{{#include ../../keanu/examples/coal_mining_disasters.rs}}
```

## A custom hyperprior

```rust,ignore
{{#include ../../keanu/examples/custom_distribution.rs}}
```

## Eight schools

The SAT coaching study (Rubin 1981; Gelman et al., *Bayesian Data Analysis*, §5.5) is the standard example of a
funnel. In the centred form the per-school effects `theta ~ N(mu, tau)` are latents; when `tau` is small they are
squeezed together and NUTS's trajectories diverge, even with a target acceptance of 0.95. The non-centred form
samples standard normal offsets `eta` and sets `theta = mu + tau * eta`, which has the same posterior without the
funnel. The example runs four chains of each form with `run_chains`, counts divergences from
`NutsStatistics::divergence_count`, and prints the non-centred form's `ChainSummary` (R-hat and effective sample
sizes). Its test checks that the non-centred form has no divergences, every R-hat is below 1.01 and the posterior
mean of `mu` is near the published 4.4, and that the centred form reports divergences.

```rust,ignore
{{#include ../../keanu/examples/eight_schools.rs}}
```

## Radon

Gelman & Hill's varying-intercept model of log radon in 919 Minnesota houses (*Data Analysis Using Regression and
Multilevel/Hierarchical Models*, 2007, ch. 12). Each house's county intercept is selected with
`a.gather(0, county)`. With this much data the centred form samples without divergences, so the intercepts are
latents directly. Its test checks R-hat below 1.01 for the hyperparameters and all 85 intercepts, and every
posterior mean against an independent computation (intercepts integrated out analytically). The posterior means
are mu_a ≈ 1.49, b ≈ −0.66, sigma_a ≈ 0.32 and sigma_y ≈ 0.73, as in the PyMC radon notebook with the same priors.

The data (`keanu/examples/data/radon.csv`: county index, county name, floor and log radon) is taken from
`examples/data/radon.csv` in the [PyMC examples repository](https://github.com/pymc-devs/pymc-examples) (MIT
licence, Copyright (c) 2020 PyMC), the Minnesota subset of the EPA National Radon Survey data distributed with
Gelman & Hill's book.

```rust,ignore
{{#include ../../keanu/examples/radon.rs}}
```

## A Gaussian mixture

A two-component mixture with a Dirichlet prior on the weights and ordered means, built as the first mean plus a
positive gap (Keanu has no distribution over ordered vectors; this is how Stan's ordered transform builds them).
Four NUTS chains recover the components, and `ChainSamples::summary` reports R-hat and effective sample sizes.

```rust,ignore
{{#include ../../keanu/examples/gaussian_mixture.rs}}
```

## Gaussian process regression

Forty noisy observations of sin(x) on [0, 10] (noise sd 0.2, from a fixed seed) are modelled with a zero-mean
Gaussian process with an exponentiated quadratic kernel (see [Gaussian processes](gaussian-processes.md)). The
amplitude, length scale and noise get half-Gaussian priors, and four NUTS chains (`sample_chains`) sample them from
the GP marginal likelihood, with the function itself integrated out. The prediction on a grid of 21 points averages
over the posterior: a second graph holds the same GP with constant hyperparameters, `predict` gives the conditional
mean and variance of the function, and each draw's hyperparameters are set into it and evaluated in turn. The
posterior-predictive mean is the average conditional mean, and its variance the average conditional variance plus
the variance of the conditional means. Keeping the prediction out of the sampled model means NUTS does not recompute
it at every leapfrog step.

Its test checks that the hyperparameters' R-hat is below 1.01, the noise is recovered, and the predictive mean is
within two predictive standard deviations of sin(x) at every grid point.

```rust,ignore
{{#include ../../keanu/examples/gp_regression.rs}}
```

## Stochastic volatility

The stochastic volatility model of the Stan User's Guide: daily returns r_t ~ N(0, exp(h_t / 2)), with the log
variance h_t a stationary AR(1) process, h_t = mu + phi (h_{t−1} − mu) + sigma eps_t. The innovations eps are
non-centred, a standard normal vector latent, and h is built from them with a [scan](sequences.md#scan-a-loop-in-one-node):
one node holding the loop body, with the scaled innovations as its per-step input and the level and persistence as
constants, so the graph has the same size for any number of days. The priors are mu ~ N(0, 10), (phi + 1) / 2 ~
Beta(20, 1.5) and sigma ~ HalfCauchy(5).

The data (`keanu/examples/data/returns.csv`) are 200 days simulated from the model with mu = −1, phi = 0.9 and
sigma = 0.4 by the example's `simulate` function with a fixed seed; the file holds the true log variance alongside
each return, and a test checks that `simulate` still reproduces it. Four NUTS chains with a target acceptance of
0.95 recover phi and sigma within their posterior uncertainty, with every R-hat close to 1. As phi nears 1 the level
mu and the slow movements of h trade off along a curved ridge, and a few trajectories in a thousand diverge there
even at a target acceptance of 0.99 or with longer adaptation, because the curvature there changes faster than any
one step size can follow; the test checks that fewer than 1% do.

```rust,ignore
{{#include ../../keanu/examples/stochastic_volatility.rs}}
```

## A hidden Markov model

Four hundred observations of a series that switches between two regimes, simulated with Gaussian emissions of means
−1 and 2 and the transition matrix [[0.95, 0.05], [0.10, 0.90]]. The model is a
[hidden Markov model distribution](sequences.md#hidden-markov-models-and-state-space-models): the hidden states are
summed out by the forward algorithm, so NUTS samples only the transition matrix, the means and the noise scale. Each
row of the transition matrix has a Dirichlet prior, and the rows are stacked into the matrix.

Swapping the two states' labels leaves the likelihood unchanged, so the posterior has two copies of each mode and
chains can settle in different ones (label switching), which makes R-hat meaningless. The example removes the copy
by ordering the emission means: the first mean plus a positive gap, as in the Gaussian mixture, so state 0 is always
the regime with the lower mean. Its test checks every R-hat below 1.01, the transition probabilities, means and noise
scale within their posterior uncertainty of the truth, and the ordering in every draw.

```rust,ignore
{{#include ../../keanu/examples/hmm.rs}}
```
