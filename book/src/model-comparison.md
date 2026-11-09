# Model comparison (LOO and WAIC)

Two models of the same data can both fit well and still predict new data differently. Keanu compares models by
their **expected log pointwise predictive density** (elpd): how much probability each model gives to new
observations like the ones it was fitted to, summed over the observations. Higher is better. It is estimated, as
in R's `loo` and ArviZ, by Pareto-smoothed importance sampling leave-one-out cross-validation (**PSIS-LOO**) or by
the widely applicable information criterion (**WAIC**), from the posterior draws alone: no refitting.

Both start from the **pointwise log likelihood**: the log likelihood of each observation at each draw.
`PointwiseLogLik::from_chains` computes it after sampling, for the draws of any algorithm, as long as the samples
record every latent the observations depend on. Each element of an observed node whose distribution has a value
per element is a point (named like `y[3]`); other nodes, such as a multivariate Gaussian, are one point each.

```rust,ignore
{{#include ../../keanu/tests/book.rs:loo_fit}}
```

`loo` returns the totals with their standard errors (`elpd`, `se`, the effective number of parameters `p` and
`looic = −2·elpd`), the Monte Carlo standard error of the elpd, and for each point its elpd and its **Pareto k̂**.

## Pareto k̂

Leaving out observation i is approximated by reweighting the draws by 1 / p(yᵢ | θ). When one observation is very
influential, a few draws get almost all the weight. PSIS fits a generalized Pareto distribution to the largest
weights and smooths them; the fitted shape k̂ tells you whether the result can be trusted:

- k̂ ≤ min(1 − 1/log₁₀(S), 0.7) for S draws (0.7 from 2200 draws): the estimate is reliable.
- above it: the point is too influential for importance sampling. `Loo::warnings` names such points and
  `bad_points()` lists them. More draws don't help much; consider a model that predicts the point better (for
  example with heavier-tailed errors) or K-fold cross-validation, which refits without each fold.

The PSIS algorithm itself is available as `keanu::algorithms::psis::psis`, for any importance sampling problem.

## Refitting on held-out data

`kfold` takes one fold ID per observation. Each callback receives the training and held-out indices in the
original observation order. Fit on the training indices, then build a graph that observes the held-out values.
Use `NetworkSamples::map_nodes` to bind each chain's parameter draws to the corresponding nodes of that graph.
Evaluate the held-out observations with `PointwiseLogLik::from_samples` or combine the remapped chains with
`ChainSamples::new` and use `PointwiseLogLik::from_chains`. Return one likelihood point per held-out index in
the order given to the callback.

`refit_loo` uses a separate fold for each observation, so it costs as many fits as there are observations.
Both functions take the log of the mean likelihood over posterior draws for each held-out observation and
return the scores in the original observation order. `elpd()` sums those scores.

## Comparing models

`compare` ranks models best first. `elpd_diff` is each model's elpd minus the best one's (so it is negative for the
others), and `se_diff` its standard error, computed from the pointwise differences, which accounts for the models
being evaluated on the same observations. A difference smaller than a couple of standard errors, or smaller than
about 4 in absolute value, says the models predict about equally well.

```rust,ignore
{{#include ../../keanu/tests/book.rs:loo_compare}}
```

## Model weights

Rather than pick one model, you can average their predictions. `compare` reports **stacking** weights (Yao et al.,
2018): the mixture weights that maximise the leave-one-out log score of the combined prediction. Stacking favours
models that predict different observations well, so it can give weight to a model with a lower elpd. The
alternative **pseudo-BMA+** weights are the softmax of the elpds, averaged over a Bayesian bootstrap of the points
so that they reflect the uncertainty of the estimates.

```rust,ignore
{{#include ../../keanu/tests/book.rs:loo_weights}}
```

## WAIC

`waic` estimates the elpd as the log pointwise predictive density less the variance of each point's log likelihood
over the draws. It is cheaper but less robust than PSIS-LOO, and it warns about points whose variance (p_waic) is
above 0.4. Prefer PSIS-LOO; WAIC values work with `compare` too.

The eight schools example (`cargo run --release --example eight_schools`) reports PSIS-LOO for both forms of its
hierarchical model, with k̂ per school; its elpd of about −30.8 agrees with ArviZ's for the same model.
