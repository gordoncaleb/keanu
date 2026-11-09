# Examples

These live in `keanu/examples`. Run one with `cargo run --release --example <name>`. Each is also an
integration test (`keanu/tests/examples.rs`): the examples ported from Java check the same results the Java
examples' tests check, and the hierarchical ones check multi-chain R-hat, divergences and posterior means.
The Gaussian process, stochastic volatility, hidden Markov model, CO₂ forecasting, pharmacokinetic, multi-state,
extreme-value, cure-model, causal-effect, missing-data and stochastic SIR examples check their results against the functions or parameters their synthetic data were generated
from, and the SMC model comparison checks its evidence against the closed form.

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
| `disease_mapping` | Where is a disease more common than expected? A BYM2 spatial model of counts on a lattice |
| `drift_diffusion` | How fast do people take in evidence, and does an easier task speed it up? A hierarchical drift-diffusion model of response times |
| `extreme_values` | How high will the river rise once in a hundred years? A generalized extreme value model of annual maxima, with return levels |
| `causal_effect` | Does a treatment work, when sicker people are more likely to take it? Conditioning against intervening, and the average treatment effect |
| `missing_data` | What do the data say when a fifth of the outcomes and a tenth of a covariate are missing? A regression with the gaps imputed |
| `cure_model` | What fraction of patients does a treatment cure, and how soon do the others relapse? A mixture cure model of censored survival times |
| `skew_student_t_regression` | How does a linear trend fit when large residuals favor one direction? |
| `poisson_binomial_calibration` | How should we calibrate individual success probabilities when only top-coded batch totals are recorded? |
| `polya_gamma_logistic` | How can auxiliary variables turn Bernoulli logistic regression into Gibbs updates with a Gaussian slope conditional? |
| `discrete_weibull_reliability` | How long do devices last when some are still running at the end of the test? |
| `multinomial_logit_regression` | How does a precinct-level covariate shift votes among three candidates? |
| `precision_gaussian_denoising` | What signal is consistent with noisy measurements and a precision prior that links neighbors? |
| `correlated_regression` | How can we fit two strongly correlated predictors in orthogonal coordinates? |
| `structured_gp` | Can a low-rank GP forecast a held-out period? Fourier and inducing-point priors |
| `multivariate_walk_forecast` | How does correlated innovation noise affect a two-series forecast? |
| `gp_regression` | What smooth function produced noisy measurements? Gaussian process hyperparameters by NUTS, exactly and with an HSGP |
| `stochastic_volatility` | How volatile are daily returns, and how persistent is the volatility? A scan over latent innovations |
| `hmm` | When does a series switch regime, and how often? A two-state hidden Markov model |
| `co2_forecast` | What will CO₂ be in two years, and how much of it is trend and season? A structural time series |
| `sir` | How fast did influenza spread through a boarding school, and how long were the boys ill? An SIR ODE |
| `pharmacokinetics` | How quickly is an oral dose absorbed and cleared? A one-compartment ODE, checked against its closed form |
| `multi_state` | How fast do people fall ill and die, seen only at clinic visits? An illness–death model with the matrix exponential |
| `smc_model_comparison` | Is a trend a straight line or a curve? Model comparison by the evidence that SMC estimates, checked against its closed form |
| `abc_stochastic_sir` | How fast does an epidemic spread when chance matters? A stochastic SIR simulator fitted by SMC-ABC |

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

## Disease mapping

Counts of a disease in the 100 areas of a 10 × 10 lattice are Poisson, with each area's expected count as an offset,
a covariate and a [BYM2](spatial-models.md#bym2) area effect that mixes a spatially structured ICAR with independent
noise. The counts are simulated from a known scale σ = 0.6, spatial proportion ρ = 0.7 and coefficients, and the
test checks that the 90% intervals of all four cover the truth, that they converge (R-hat below 1.01 over 4 chains
of 2,000 draws) and that the posterior mean area effects correlate with the true ones above 0.9.

```rust,ignore
{{#include ../../keanu/examples/disease_mapping.rs}}
```

## Drift diffusion

Two-choice response times of four subjects, each doing 100 trials in a hard and an easy condition, are modelled with
the drift-diffusion model: the Wiener first-passage time density (`GraphBuilder::wiener`) of each response time at
the boundary the response reached, observed as an integer parameter. Each subject has a boundary separation (how
cautious they are), a non-decision time and a drift in each condition, drawn around population values, and they
share a starting point. The data are simulated exactly with `keanu_distributions::sample_wiener` from known values;
the test checks that the 90% intervals of both conditions' population drifts, the starting point and every subject's
boundary separation and non-decision time cover the truth, that every parameter converges (R-hat below 1.01 over 4
chains of 1,000 draws) with no divergences, and that the easy condition's drift is clearly the larger.

```rust,ignore
{{#include ../../keanu/examples/drift_diffusion.rs}}
```

## Extreme values

Sixty annual maxima, simulated from a generalized extreme value distribution with location 30, scale 6 and shape
ξ = 0.15 (a heavy upper tail), are fitted with `generalized_extreme_value`. The T-year return level, the level
exceeded in any one year with probability 1/T, is the GEV's 1 − 1/T quantile, μ + σ((−ln(1 − 1/T))^(−ξ) − 1)/ξ; it
is a deterministic node of the parameters, so the posterior draws give its posterior directly. The GEV's support
depends on its parameters, so NUTS rejects the parameters that put an observed maximum outside it (a few
divergences, the test allows up to 1% of the draws). The test checks that the 90% intervals of μ, σ, ξ and the 10-,
50- and 100-year return levels cover the true values, with R-hat below 1.01 over 4 chains of 1,000 draws, and that
the return levels increase with the period in every draw.

```rust,ignore
{{#include ../../keanu/examples/extreme_values.rs}}
```

## A mixture cure model

Relapse times of 400 patients, followed for between 2 and 8 years, come from a mixture cure model: each patient is
cured with a probability that depends on a biomarker through a logistic regression, and otherwise relapses after a
Weibull time whose scale depends on the biomarker too. A patient's likelihood is (1 − π) f(t) for a relapse at t and
π + (1 − π) S(c) for a patient still relapse-free at the end of follow-up c. The model writes this as a
[mixture of different families](building-models.md#mixtures-of-different-families-and-transformed-distributions):
a Weibull censored above at c for the uncured, and for the cured a uniform on [c, c + 1] censored above at c, whose
mass all lies at or beyond c, so its survival at c is 1 and its density before c is 0. The data are simulated from
known values; the test checks that the 90% intervals of the cure model's two coefficients, the Weibull scale's two
coefficients and the Weibull shape cover the truth, with R-hat below 1.01 over 4 chains of 1,000 draws and no
divergences.

```rust,ignore
{{#include ../../keanu/examples/cure_model.rs}}
```

## A causal effect

Four hundred people have a severity score Z; sicker people are more likely to be treated (X, logistic in Z), and
both the treatment and the severity raise the outcome Y, by τ = 2 and 3 per unit of Z. The model is this causal
graph, fitted by NUTS. Conditioning on the treatment, the posterior predictive of Y among the treated and among the
untreated, compares sicker people with healthier ones as well as treated with untreated, and overstates the effect
almost threefold. [Intervening](causal-inference.md) on it, with `intervene` setting X for everyone and
`PredictiveSampler::interventional` drawing Y from the original posterior, removes the confounding: the difference
of the two interventions' means is the average treatment effect. The rewrite drops the treatment model (its
coefficients and logits), which influenced Y only through X. The data are simulated from known values; the test
checks that the predictive means under do(X = 0) and do(X = 1) are within 0.3 of the true interventional means
β₀ + τx + β_z E[Z], that the treated's and untreated's predictive means are within 0.3 of the true confounded means
(computed by quadrature), that the average treatment effect's 90% interval covers τ, and that R-hat is below 1.01
over 4 chains of 1,000 draws.

```rust,ignore
{{#include ../../keanu/examples/causal_effect.rs}}
```

## Missing data

A Gaussian regression of an outcome on two covariates, on 200 simulated rows, with a fifth of the outcomes and a
tenth of the first covariate missing completely at random. The first covariate is given a normal distribution with
unknown mean and scale, so that its gaps have a prior, and both it and the outcome are observed with
[`observe_masked`](building-models.md#missing-observations) from values with NaN in the gaps. The default sampler is
then NUTS over the coefficients, the scales and both sets of missing values. The test checks that the 90% intervals
of the three coefficients and σ cover the true values with R-hat below 1.01 over 4 chains of 1,000 draws, that the
imputed outcomes' posterior means correlate with the held-out values above 0.9 (the imputed covariate's above 0.8),
and that the observed outcomes stay in place.

```rust,ignore
{{#include ../../keanu/examples/missing_data.rs}}
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

## Model comparison by SMC evidence

A linear and a quadratic regression with standard normal priors and known noise, fitted by tempered SMC to 30
points simulated from 0.5 + x + 0.6 x². SMC estimates each model's log evidence, whose difference is the log Bayes
factor; both are linear-Gaussian, so the test checks the estimates against the closed form y ~ N(0, σ² I + X Xᵀ)
to 0.15 and the Bayes factor to 0.2, and that PSIS-LOO on the particles also prefers the quadratic model. See
[Simulation-based inference](simulation-based-inference.md#comparing-models-by-their-evidence).

```rust,ignore
{{#include ../../keanu/examples/smc_model_comparison.rs}}
```

## A stochastic SIR epidemic by ABC

The SIR epidemic in 200 people as a continuous-time Markov chain, simulated exactly by Gillespie's algorithm, whose
likelihood is intractable. The daily susceptible and infected counts are a simulator node of β and γ, and SMC-ABC
compares four summary statistics of simulated and observed outbreaks (final size, peak prevalence, peak day and
person-days of infection) with a Gaussian kernel. Its test fits an outbreak simulated with β = 1 and γ = 0.3 and
checks that both lie in their 90% posterior intervals, which are informative (over data seeds 1 to 8 they always
covered the truth). See [Simulation-based inference](simulation-based-inference.md#example-a-stochastic-epidemic).

```rust,ignore
{{#include ../../keanu/examples/abc_stochastic_sir.rs}}
```

## Hierarchical item response

A two-parameter logistic model fits a student's ability and each item's difficulty and discrimination. The model
uses indexed student–item pairs and a standard normal ability prior to fix the scale. Item difficulty and log
discrimination use non-centred hierarchical priors. The example generates seeded responses, holds out one item
per student, and reports the Brier score (mean squared probability error) and the item difficulty RMSE.

Run it with `cargo run --release --example item_response`. The model follows the structure of the
[Stan hierarchical 2PL case study](https://mc-stan.org/users/documentation/case-studies.html).

```rust,ignore
{{#include ../../keanu/examples/item_response.rs}}
```

## Lotka–Volterra predator–prey dynamics

The rates of prey growth and consumption and predator growth and death are inferred from noisy observations of
both populations. The example generates a reference trajectory with a separate fixed-step RK4 implementation,
fits the first 25 times with NUTS and a differentiable ODE, and forecasts the last five times. It reports
posterior mean rates and the forecast RMSE against the underlying trajectory.

Run it with `cargo run --release --example lotka_volterra`. The same differential equations are used in the
[Stan case study](https://mc-stan.org/users/documentation/case-studies/lotka-volterra-predator-prey.html) and
[PyMC example](https://www.pymc.io/projects/examples/en/latest/ode_models/ODE_Lotka_Volterra_multiple_ways.html),
which use historical hare and lynx observations.

```rust,ignore
{{#include ../../keanu/examples/lotka_volterra.rs}}
```

## Regression with predictor measurement error

A noisy measurement of a predictor can attenuate an ordinary least squares slope. This example fits the
predictor as a latent standard normal variable, with a separate Gaussian likelihood for its noisy measurement
and one for the outcome. On seeded synthetic data the test checks that the measurement-error model recovers
the generating slope more closely than regression on the noisy measurements.

Run it with `cargo run --release --example measurement_error`. [Stan measurement error guide](https://mc-stan.org/docs/2_39/stan-users-guide/measurement-error.html)

```rust,ignore
{{#include ../../keanu/examples/measurement_error.rs}}
```

## Spline regression

A Bayesian linear spline represents a nonlinear mean as a weighted sum of piecewise-linear basis functions.
The example fits noisy observations on one grid, then binds the posterior coefficients into a separate graph
for a smaller grid. It draws outcomes and checks the mean prediction against the generating curve.

Run it with `cargo run --release --example spline_regression`. [PyMC spline example](https://www.pymc.io/projects/examples/en/latest/howto/spline.html)

```rust,ignore
{{#include ../../keanu/examples/spline_regression.rs}}
```

## Euler–Maruyama for stochastic dynamics

A noisy mean-reverting process is discretized with an Euler–Maruyama step. Standard-normal innovations enter a
deterministic scan as per-step inputs. NUTS estimates the reversion rate and diffusion from simulated observations.

Run it with `cargo run --release --example euler_maruyama`. [PyMC SDE example](https://www.pymc.io/projects/examples/en/latest/time_series/Euler-Maruyama_and_SDEs.html)

```rust,ignore
{{#include ../../keanu/examples/euler_maruyama.rs}}
```

## Bayesian vector autoregression

A bivariate VAR(1) estimates how each series responds to its own lag and the other series' lag. The example
replays posterior coefficients through a scan on a separate forecast graph and checks them against the process's
conditional mean. The reported error measures the mean forecast, with future innovations averaged out.

Run it with `cargo run --release --example vector_autoregression`. [PyMC VAR example](https://www.pymc.io/projects/examples/en/latest/time_series/bayesian_var_model.html)

```rust,ignore
{{#include ../../keanu/examples/vector_autoregression.rs}}
```

## Hierarchical A/B experiments across markets

Binomial A/B counts share a baseline conversion rate and a hierarchy for the market treatment effects on the
log-odds scale. The example estimates the population mean effect and uses the law of total variance to report
the larger predictive uncertainty about the effect in a new market.

Run it with `cargo run --release --example hierarchical_experiments`. [PyMC multi-market meta-analysis](https://www.pymc.io/projects/examples/en/latest/causal_inference/multiple_experiments_meta_analysis.html)

```rust,ignore
{{#include ../../keanu/examples/hierarchical_experiments.rs}}
```

## Robust regression with learned tail weight

A Student-t observation model gives large residuals more probability than a Gaussian. Use
`location_scale_student_t(nu, mu, sigma)` to infer the positive real degrees of freedom `nu` jointly with the
regression coefficients and scale. The example generates data independently from a Student-t with three degrees
of freedom and checks recovery of the slope and evidence for heavy tails.

Run it with `cargo run --release --example robust_regression`.

```rust,ignore
{{#include ../../keanu/examples/robust_regression.rs}}
```

## Right-censored survival with shared frailty

An unobserved group effect shifts log event times. A Gaussian transformed by exponentiation gives the event-time
distribution. Follow-up ends at a fixed deadline, so records at the deadline contribute a survival probability.
The seeded test checks recovery of the treatment effect on log event time.

Run it with `cargo run --release --example censored_frailty`.

```rust,ignore
{{#include ../../keanu/examples/censored_frailty.rs}}
```

## Structured regression and forecasting

- `correlated_regression` fits a synthetic two-predictor Gaussian regression with a prior in QR coordinates.
  It maps draws back to the original coefficients with a triangular solve.
- `structured_gp` compares Fourier and inducing-point forecasts on a held-out period of a synthetic signal.
- `multivariate_walk_forecast` uses a correlated innovation factor to forecast two synthetic series from their
  last known values and checks the forecast's cross-covariance.
