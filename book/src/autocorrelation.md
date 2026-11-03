# Autocorrelation and effective sample size

Successive MCMC samples are correlated. The **autocorrelation** at each lag shows how quickly a chain forgets where
it was. The **effective sample size** (ESS) is the number of independent samples the chain is worth, which sets the
Monte Carlo error of estimates from it.

```rust,ignore
{{#include ../../keanu/tests/book.rs:autocorrelation}}
```

For a tensor-valued node, pass the index of the element: for samples of shape `[1, 5]`, `&[0, 1]` is the second
element. `keanu::algorithms::statistics` has the same functions over plain `&[f64]` series, plus `summarize`.
