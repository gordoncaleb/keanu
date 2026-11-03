# Custom distributions

When a model needs a distribution Keanu doesn't have, implement `CustomDistribution` and add the node with
`GraphBuilder::custom`:

- **`log_prob`** (required) is the log probability of a value, given the parameters' values.
- **`log_prob_gradient`** (optional) is its gradient with respect to the value and the parameters. Implement it to
  use the node with NUTS or the gradient optimizers.
- **`sample`** (optional) draws a value. Implement it to sample the node from its prior: prior proposals, forward
  sampling, probing for a starting state, or evaluating it without a value.
- **`value_type`** (optional) is the value's type, f64 by default.

```rust,ignore
{{#include ../../keanu/tests/book.rs:custom}}
```

The [custom distribution example](examples.md) uses one as a hierarchical prior.
