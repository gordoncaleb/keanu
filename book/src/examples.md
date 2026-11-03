# Examples

These live in `keanu/examples`. Run one with `cargo run --release --example <name>`. Each is also an
integration test (`keanu/tests/examples.rs`) that checks the same results the Java examples' tests check.

| Example | Question |
|---|---|
| `coal_mining_disasters` | When did the rate of coal mining disasters change? |
| `text_messaging` | When did someone's text messaging rate increase? |
| `ab_testing` | Is treatment A more likely to drive purchases than treatment B? |
| `cheating_students` | What proportion of students cheated, from a privacy-preserving survey? |
| `challenger_disaster` | Does temperature predict O-ring failure? |
| `custom_distribution` | A hierarchical model with a user-defined hyperprior |

Most come from *Bayesian Methods for Hackers* by Cameron Davidson-Pilon.

## Coal mining disasters

```rust,ignore
{{#include ../../keanu/examples/coal_mining_disasters.rs}}
```

## A custom hyperprior

```rust,ignore
{{#include ../../keanu/examples/custom_distribution.rs}}
```
