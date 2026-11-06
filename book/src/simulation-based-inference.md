# Simulation-based inference

This chapter covers tempered sequential Monte Carlo (SMC), which samples any model and estimates its marginal
likelihood, and approximate Bayesian computation (ABC), which builds on it to fit models whose data can be
simulated but whose likelihood can't be written down.

## Sequential Monte Carlo and the marginal likelihood

`Smc` (tempered sequential Monte Carlo, as PyMC's `sample_smc`) moves a population of particles from the prior to
the posterior through the targets prior × likelihood^β, raising β from 0 to 1 in stages. Each stage chooses the
next β by bisection so that the particles' normalised effective sample size falls to `ess_threshold` (0.5),
resamples them by their weights (systematic resampling), and mutates them with Markov chain steps at that β:

- the continuous latents, in unconstrained space, by independent Metropolis–Hastings with a multivariate normal
  proposal fitted to the weighted particles (`SmcKernel::IndependentMetropolis`, the default), or by a random walk
  with the particles' covariance and per-particle scales tuned by the acceptance rate
  (`SmcKernel::RandomWalkMetropolis`). A stage takes steps until the mean correlation between the particles'
  coordinates before and after is at most `correlation_threshold` (0.01), up to `max_steps` (25);
- the discrete latents by `Gibbs` or `DiscreteMetropolis` steps at the stage's β, after each continuous step.

Because the particles start from the prior, SMC also estimates the **log marginal likelihood** (the evidence),
log p(y), as the sum over stages of the log mean incremental weight, and differences of log evidence are log Bayes
factors. `sample` runs `chains` (4) independent runs of `particles` (2000) each, in parallel with the `parallel`
feature; `SmcResult` holds their final particles as `ChainSamples` (so R-hat works across runs), each run's
`log_marginal_likelihood`, its temperatures `betas`, its `acceptance_rates` and `steps` per stage, and
`combined_log_marginal_likelihood` pools the runs' estimates.

```rust,ignore
{{#include ../../keanu/tests/book.rs:smc}}
```

The independent proposal suits posteriors that are roughly Gaussian in unconstrained space, and jumps between
well separated modes, which MCMC chains rarely cross: a symmetric bimodal posterior gets its modes in proportion.
Where latents move together strongly (here the switch and the two rates), mutation mixes slowly and every stage
uses its `max_steps`. Particles where the model is impossible (−∞, or a failed ODE solve) get zero weight and moves
to them are rejected. A model whose support bounds move with a latent is mutated in constrained space, where moves
out of the support are rejected. The tempered likelihood is pluggable (`smc::TemperedLikelihood`, used with
`sample_with`); the default, `ModelLikelihood`, is the observed nodes' and factors' terms.

## Comparing models by their evidence

The `smc_model_comparison` example fits a linear and a quadratic regression, with standard normal priors on the
coefficients and known noise, to 30 points simulated from a gently curved trend:

```rust,ignore
{{#include ../../keanu/examples/smc_model_comparison.rs:smc_regression}}
```

Each model's log evidence comes straight out of the SMC run:

```rust,ignore
{{#include ../../keanu/examples/smc_model_comparison.rs:smc_evidence}}
```

Both models are linear-Gaussian, so their evidence is known in closed form (y ~ N(0, σ² I + X Xᵀ)); the SMC
estimates match it to a few hundredths, and the log Bayes factor of about 8 in favour of the quadratic model gives
it a posterior probability above 0.999 at even prior odds. The evidence measures how well the prior predicted the
data, so it depends on the priors in a way that [PSIS-LOO](model-comparison.md) does not; the example also
computes LOO from the SMC particles, which prefers the quadratic model too.

## Simulator nodes

Some models are easiest to write as a program that simulates data: an epidemic with random contacts, a queue, an
agent-based model. Their likelihood, the probability of the observed data given the parameters, sums over every
random path that could have produced the data, and is usually intractable. `GraphBuilder::simulator` makes such a
model a node, like PyMC's `pm.Simulator`: a probabilistic node whose value of a given shape is drawn by a function
of its parameters' values and a random number generator. The handle type (`DoubleVar`, `IntVar` or `BoolVar`) sets
its value type.

```rust,ignore
{{#include ../../keanu/tests/book.rs:simulator}}
```

A simulator node can be sampled forward, so prior and posterior predictive checks work, and observed. It has no
likelihood: its log probability is the error `GraphError::NoLikelihood`, and every algorithm that needs the
likelihood (NUTS, Metropolis–Hastings, Gibbs and compound samplers, the optimizers, Laplace, ADVI, Pathfinder, the
particle filter, SMC on the model's likelihood, the pointwise log likelihood) rejects a model with a simulator
node before it starts. The function must be thread safe, since parallel chains call it at once, and must return a
value of the node's type and shape (anything else is `GraphError::Simulator`). Like lambdas and custom
distributions, simulator nodes can't be [saved](save-and-load.md).

## Approximate Bayesian computation

ABC replaces the likelihood by a comparison between data simulated at the parameters and the observed data.
`abc::Abc` does this for SMC, following PyMC's `sample_smc` on a model with a `pm.Simulator`. At each particle it
simulates every observed simulator node, reduces the simulated and observed values to summary statistics s, and
scores their difference with a kernel of scale ε:

- `AbcKernel::Gaussian` (the default): −½ ‖(s(simulated) − s(observed)) / ε‖²;
- `AbcKernel::Laplace`: −‖(s(simulated) − s(observed)) / ε‖₁.

The statistics (`SummaryStatistics`) are the data itself (`Identity`, the default), the sorted data (`Sort`, for
exchangeable data), its `Mean`, `Median` or `Variance`, or a user function of the value (`SummaryStatistics::custom`).
ε is one value for every statistic, or one per statistic (`with_epsilons`) when they have different scales. The
model's other observed nodes and factors keep their exact likelihood, which is added. `Smc::sample_abc` runs it:

```rust,ignore
{{#include ../../keanu/tests/book.rs:abc}}
```

As in PyMC, ε is fixed and SMC's tempering raises β from 0 to 1 as for any likelihood, so the stage at β targets
the prior times the kernel to the power β (for the Gaussian kernel, the kernel with scale ε / √β). Each particle
keeps the kernel value of the simulation it was accepted with, and a move is judged on a fresh simulation at the
proposed point (a pseudo-marginal Metropolis–Hastings step), so the final particles sample the ABC posterior

p_ε(θ | y) ∝ p(θ) · E over x simulated at θ of K((s(x) − s(y)) / ε),

which approaches the posterior given s(y) as ε shrinks, and the exact posterior if the statistics are sufficient
(as the mean is for this Gaussian). A smaller ε is more accurate but makes acceptance rarer: it needs more stages,
and more particles to keep them diverse. A useful scale is about the spread of the statistics between simulations
at one parameter value. The log marginal likelihood SMC reports is that of the ABC posterior with the kernel
unnormalised, as PyMC's, so it compares models only under the same kernel, ε and statistics. A statistic or
simulation of NaN makes a particle impossible; a function that fails stops the run. ABC needs continuous latents:
SMC mutates discrete latents over their Markov blankets, where a simulator has no terms.

## Example: a stochastic epidemic

The [SIR example](examples.md#an-sir-epidemic) follows the mean of an epidemic with an ODE. In a population of 200,
chance matters: the `abc_stochastic_sir` example simulates the epidemic as a continuous-time Markov chain with
Gillespie's algorithm (each infected person infects each susceptible one at rate β / N and recovers at rate γ),
so the same rates give visibly different outbreaks, and the probability of the observed daily counts is
intractable. The counts are a simulator node of the two rates:

```rust,ignore
{{#include ../../keanu/examples/abc_stochastic_sir.rs:abc_simulator}}
```

and ABC compares four summary statistics, each with its own ε: the final size and peak prevalence (fractions of
the population), the day of the peak, and the person-days of infection per person. The final size is set by
R₀ = β / γ and the person-days by the recovery time 1 / γ, so together they identify both rates.

```rust,ignore
{{#include ../../keanu/examples/abc_stochastic_sir.rs:abc_likelihood}}
```

```rust,ignore
{{#include ../../keanu/examples/abc_stochastic_sir.rs:abc_sample}}
```

On an outbreak simulated with β = 1 and γ = 0.3, the 90% intervals cover both rates (for example β in [0.85, 1.46]
and γ in [0.27, 0.41]), in a few seconds with the `parallel` feature.
