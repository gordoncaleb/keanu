# Examples

These live in `keanu/examples`. Run one with `cargo run --release --example <name>`. Each is also an
integration test (`keanu/tests/examples.rs`): the examples ported from Java check the same results the Java
examples' tests check, and the hierarchical ones check multi-chain R-hat, divergences and posterior means.

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

The first five come from *Bayesian Methods for Hackers* by Cameron Davidson-Pilon.

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
