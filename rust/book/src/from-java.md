# Coming from Java Keanu

The model semantics are the same as the final Java version, and that's checked against vectors recorded from Java.
The API is idiomatic Rust rather than a translation.

| Java | Rust |
|---|---|
| Vertices hold their own values | An immutable `Graph`, with values in a separate, cloneable `State` |
| `new GaussianVertex(mu, sigma)` | `builder.gaussian(mu, sigma)` |
| `vertex.getValue()` | `state.f64(id)` (or `i64` / `bool`) |
| `setAndCascade`, `lazyEval`, `observe` | `State::set_and_cascade`, `State::lazy_eval`, `State::observe` |
| `If.isTrue(p).then(a).orElse(b)` | `builder.where_(p, a, b)` |
| `KeanuRandom.setDefaultRandomSeed(1)` | `KeanuRng::seed_from_u64(1)`, passed to each call (there's no global generator) |
| `Keanu.Sampling.NUTS.withDefaultConfig()` | `Nuts::default()` |
| `Keanu.Optimizer.of(network)` | `GradientOptimizer::new(&model)` |
| `IntegerTensor` (32-bit) | `IntTensor` (64-bit) |
| Exceptions | `Result` |
| Custom vertex subclasses | `CustomDistribution` |

## Behaviour differences

- **`tril(k)`** follows numpy's sign convention for `k`; Java's `triLower(k)` counted the other way.
- **`cholesky_inverse`** returns the full symmetric inverse; Java returned only its lower triangle.
- **The examples** check their results with longer chains than the Java examples used. With a different random
  number generator, the Java chain lengths only converged for particular seeds.
- **Not ported:**
  - the runtime-compiled computable graph (graphs are evaluated directly)
  - the CSV object binding
  - MIR files
  - external-process (lambda model) vertices
  - `PrintVertex`
  - generic object vertices (`Categorical` works over category indices)
- **BOBYQA** is translated from PRIMA rather than Apache Commons Math, so its path differs from Java's. It reaches
  the optimum in cases where Java's stops short with Keanu's defaults, and it accepts a single latent dimension.
