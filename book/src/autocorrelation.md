# Autocorrelation and effective sample size

Successive MCMC samples are correlated. The **autocorrelation** at each lag shows how quickly a chain forgets where
it was. The **effective sample size** (ESS) is the number of independent samples the chain is worth, which sets the
Monte Carlo error of estimates from it.

```rust,ignore
{{#include ../../keanu/tests/book.rs:autocorrelation}}
```

For a tensor-valued node, pass the index of the element: for samples of shape `[1, 5]`, `&[0, 1]` is the second
element. `keanu::algorithms::statistics` has the same functions over plain `&[f64]` series, plus `summarize`.

## Diagnostics across chains

One chain can look well mixed while stuck in one part of the posterior. Run several independent chains and compare
them: `sample_chains` runs them (in parallel with the `parallel` feature) and returns a
`ChainSamples`, whose `summary()` gives each element's mean, standard deviation, 5%, 50% and 95% quantiles, Monte
Carlo standard error of the mean, bulk and tail ESS, and R-hat.

```rust,ignore
{{#include ../../keanu/tests/book.rs:multi_chain}}
```

The diagnostics follow Vehtari et al. (2021), as ArviZ computes them:

- **R-hat** compares the variance between chains with the variance within them, after splitting each chain in half
  and replacing the draws by the normal quantiles of their ranks. It reports the larger of that and the same
  statistic for the distances from the median, which catches chains with different spreads. It is 1 for chains
  that agree; trust the results only if it is below 1.01 (`RHAT_THRESHOLD`).
- **Bulk ESS** is the effective sample size of the rank-normalised draws, which governs estimates of the centre of
  the distribution. **Tail ESS** is the smaller effective sample size of the indicators of being below the 5% and
  95% quantiles, which governs interval estimates. Both should exceed 400 (`MIN_ESS`, 100 per chain for four
  chains).
- **MCSE** is the Monte Carlo standard error of the mean: the standard deviation over the square root of the ESS.

`ChainSamples::new` builds the same thing from chains you ran yourself, and checks they sampled the same nodes for
the same number of draws. `keanu::algorithms::diagnostics` has the functions over plain `&[Vec<f64>]` chains.
