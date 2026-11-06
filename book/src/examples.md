# Examples

These live in `keanu/examples`. Run one with `cargo run --release --example <name>`. Each is also an
integration test (`keanu/tests/examples.rs`): the examples ported from Java check the same results the Java
examples' tests check, and the hierarchical ones check multi-chain R-hat, divergences and posterior means.
The Gaussian process, stochastic volatility, hidden Markov model, CO₂ forecasting, pharmacokinetic and multi-state examples check
their results against the functions or parameters their synthetic data were generated from.

| Example | Question |
|---|---|
| `coal_mining_disasters` | When did the rate of coal mining disasters change? |
| `text_messaging` | When did someone's text messaging rate increase? |
| `ab_testing` | Is treatment A more likely to drive purchases than treatment B? |
| `cheating_students` | What proportion of students cheated, from a privacy-preserving survey? |
| `challenger_disaster` | Does temperature predict O-ring failure? |
| `custom_distribution` | A hierarchical model with user-defined distributions, reparameterised to sample well |
| `eight_schools` | How much do coaching effects differ between schools? Centred and non-centred forms, compared by PSIS-LOO |
| `radon` | How much does indoor radon vary between Minnesota counties? A varying-intercept model |
| `gaussian_mixture` | Which of two groups does each measurement come from, and what are the groups like? |
| `occupancy` | Which sites does a species occupy, when surveys sometimes miss it? Occupancy latents sampled, marginalised and written as a zero-inflated binomial |
| `gp_regression` | What smooth function produced noisy measurements? Gaussian process hyperparameters by NUTS, exactly and with an HSGP |
| `stochastic_volatility` | How volatile are daily returns, and how persistent is the volatility? A scan over latent innovations |
| `hmm` | When does a series switch regime, and how often? A two-state hidden Markov model |
| `co2_forecast` | What will CO₂ be in two years, and how much of it is trend and season? A structural time series |
| `sir` | How fast did influenza spread through a boarding school, and how long were the boys ill? An SIR ODE |
| `pharmacokinetics` | How quickly is an oral dose absorbed and cleared? A one-compartment ODE, checked against its closed form |
| `multi_state` | How fast do people fall ill and die, seen only at clinic visits? An illness–death model with the matrix exponential |

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
`run_marginalized` fits the same mixture written with an explicit group assignment per measurement, summed out by
`marginalize` (see [Marginalising discrete latents](building-models.md#marginalising-discrete-latents)), and recovers
each measurement's group probabilities with `recover_marginals`.

```rust,ignore
{{#include ../../keanu/examples/gaussian_mixture.rs}}
```

## Site occupancy

A species occupies each of 120 simulated sites with probability ψ, and each of 4 visits to an occupied site detects
it with probability p, so a site with no detections may be empty or occupied and missed every time (MacKenzie et
al., 2002). `run_latent` keeps each site's occupancy as a Bernoulli latent, which the default compound sampler
updates with Gibbs steps between NUTS steps on ψ and p (see
[Discrete latents: Gibbs and compound sampling](inference-sampling.md#discrete-latents-gibbs-and-compound-sampling)); `run_marginalized` sums the occupancies out with
`marginalize` and recovers each site's probability of being occupied with `recover_marginals`; and
`run_zero_inflated` writes each site's number of detections directly as a
[zero-inflated](building-models.md#zero-inflated-hurdle-and-logit-counts) binomial, 0 for an empty site with
probability 1 − ψ. Its test checks that each form covers the true ψ and p with its 90% interval and that the three
agree, on ψ, p and the posterior number of occupied sites.

```rust,ignore
{{#include ../../keanu/examples/occupancy.rs}}
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

Its second half fits 200 points with an HSGP of 25 basis functions instead: NUTS samples the weights with the
hyperparameters, and the function on the grid is recorded at each draw (`prior.predict` with the same weights), so
the prediction is the draws' mean and standard deviation.

Its tests check that the hyperparameters' R-hat is below 1.01 (1.02 for the HSGP, whose amplitude is weakly
identified), the noise is recovered, and the predictive mean is within two predictive standard deviations of sin(x)
at every grid point (2.5 for the HSGP).

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
scale within their posterior uncertainty of the truth, and the ordering in every draw. It then decodes the regimes
with `HiddenMarkovPosterior` (see [smoothing, decoding and forecasting](sequences.md#smoothing-decoding-and-forecasting)):
the posterior probability of each regime at each step, averaged over the draws, puts at least 95% of the steps in
their simulated regime.

```rust,ignore
{{#include ../../keanu/examples/hmm.rs}}
```

## Forecasting CO₂

The monthly Mauna Loa CO₂ record is a trend, a yearly cycle and noise, and TensorFlow Probability's
structural time-series example fits them as such. This example does the same on simulated data of the same shape
(about 315 ppm rising 0.1 ppm a month, a cycle of ±3 ppm peaking in May), drawn from the model itself so that the
forecast can be checked against the truth: 96 months to fit and 24 held out. The model is built with
[`sts`](sequences.md#structural-time-series): a local linear trend plus a 12-month trigonometric seasonal component
with two harmonics, one state-space node whose six states the Kalman filter integrates out, so NUTS samples only
the four scales, with the builder's default priors. `StructuralPosterior` then splits the fitted months into trend and
season and forecasts the next 24, both with intervals over the posterior draws.

Its test checks R-hat below 1.02 for every scale, that the components add up to the smoothed series, that the
seasonal cycle has the simulated amplitude, and that the 90% and 50% forecast intervals cover the held-out months at
about their nominal rates.

```rust,ignore
{{#include ../../keanu/examples/co2_forecast.rs}}
```

## An SIR epidemic

The SIR model (susceptible, infected, recovered) of the 1978 influenza outbreak at an English boarding school (BMJ,
1978): 763 boys, with the number confined to bed counted on each of 14 days. The compartments follow
dS/dt = −β S I / N, dI/dt = β S I / N − γ I and dR/dt = γ I from one infected boy, solved by an
[ODE node](differential-equations.md), and each day's count is negative binomial around I(t) with dispersion φ.
The priors are those of the Stan case study by Grinsztajn, Semenova, Margossian and Riou (2021): β ~ N(2, 1) and
γ ~ N(0.4, 0.5), both truncated to positive values, and 1/φ ~ Exponential(5). Four NUTS chains give β ≈ 1.74 per
day, γ ≈ 0.54 per day, R₀ = β / γ ≈ 3.2 and a mean recovery time 1 / γ ≈ 1.85 days, as in the case study.

Its tests fit data simulated from the model with β = 1.7 and γ = 0.55 and check that both lie in their 90%
posterior intervals, check the boarding school posterior means against the case study's, and check the gradient of
the log probability (through the ODE's sensitivities) against finite differences to 10⁻⁶ relative.

```rust,ignore
{{#include ../../keanu/examples/sir.rs}}
```

## Pharmacokinetics

The one-compartment model with first-order absorption of an oral dose: the dose moves from the gut into the body at
the absorption rate kₐ and is cleared at kₑ = CL / V, and the measured concentration A_central / V has a log-normal
error. One subject's concentrations at the sampling times of R's `Theoph` study are simulated with a 320 mg dose,
kₐ = 1.5 per hour, CL = 2.8 L per hour and V = 35 L. The model is fitted twice, with the ODE solved numerically and
with its closed-form solution; the test checks that the two log probabilities and gradients agree to the solver's
tolerance at several points, that the two posteriors' means agree within their Monte Carlo error, and that each
parameter is recovered within its posterior uncertainty.

```rust,ignore
{{#include ../../keanu/examples/pharmacokinetics.rs}}
```

## A multi-state model

The illness–death model: a continuous-time Markov chain on healthy, ill and dead with intensities q₀₁ (falling
ill), q₀₂ (dying while healthy) and q₁₂ (dying while ill). 300 simulated subjects are seen at visits 0.5, 1 or
2 years apart until they die, and only their state at each visit is recorded, so when each transition happened is
interval-censored. The probability of each observed pair of states is an entry of P(Δt) = exp(Q Δt), one batched
[matrix exponential](differential-equations.md#the-matrix-exponential) over the three gaps, and each visit's state
is a categorical draw from the row of the state before it (as R's `msm` package fits panel data). The example also
reports the probability that someone healthy now is dead within 5 years, exp(5 Q)₀₂. Its test checks that each true
intensity lies in its 90% posterior interval and that the five-year probability is recovered.

```rust,ignore
{{#include ../../keanu/examples/multi_state.rs}}
```
