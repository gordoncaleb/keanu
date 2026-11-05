# Golden vectors

Reference inputs and outputs produced by the Java implementation of Keanu. They were the parity target for the
Rust port (see `RUST_MIGRATION_PLAN.md` at the repository root) and remain the regression suite that keeps the
Rust implementation equivalent to the Java one.

## Frozen

These files were generated for the last time by the Java `keanu-golden` exporter at the `java-final` tag, just
before the Java implementation was removed. **They are never regenerated.** The Rust tests in every crate check
them through `keanu_parity::golden_dir()`.

If a deliberate change to Rust's behaviour (for example a bug fix) changes a recorded result:

1. Make the change with a regression test in the affected crate.
2. Update only the affected case in the fixture, by hand, and keep the rest of the file untouched.
3. Record the change, and the case names it affected, in `CHANGELOG.md`.

To see how a case was produced, or to rerun the Java implementation, check out the `java-final` tag
(`git checkout java-final`, then `./gradlew :keanu-golden:generateGoldenVectors`).

## Files

| File | Contents |
|---|---|
| `manifest.json` | Format version, case counts per file, and known issues (see below) |
| `tensors.json` | Tensor operations on f64 / i32 / i64 / bool tensors: element-wise unary and binary ops (with broadcasting), reductions and scans, shape manipulation, numpy-style slicing, linear algebra, integer arithmetic, boolean logic, and their error cases |
| `distributions.json` | For each distribution and several parameter settings: element-wise `logProb` over a grid of x values, inside and outside the support, and the element-wise derivatives of the log prob with respect to x and each parameter |
| `vertices.json` | Small vertex graphs: the output value, plus forward-mode and reverse-mode partial derivatives of the output with respect to each input |
| `models.json`, `models/` | Whole models, as binary protobuf (`.pb`) and JSON files in Keanu's native save format (see below), with the expected log prob, log likelihood, their gradients with respect to each latent, and the value of every labelled vertex. `catalog` contains one labelled instance of every vertex type Java could save |
| `samplers.json` | Posterior summaries from fixed-seed Java runs of NUTS (adapt 500, drop 500 of 3000) and Metropolis–Hastings (Keanu's defaults, drop 2000 of 20,000) on each golden model: per latent element, the posterior mean, standard deviation and effective sample size (Geyer's initial monotone sequence). Random streams differ between implementations, so compare within the combined Monte Carlo error sd · sqrt(1 / ess_java + 1 / ess_rust) |
| `algorithms.json` | MAP and maximum likelihood results for each golden model, with Adam (Keanu's defaults, at most 200,000 iterations), conjugate gradient (Keanu's defaults) and BOBYQA (Keanu's defaults, and an initial trust region radius of 0.5 with a bounds range of 5): the optimized value of each latent (by label), the fitness, and whether Adam converged. Optimizers reach the optimum by different paths, so compare with optimization tolerances. The hierarchical model's MAP is unbounded (sigma → 0), so it has no meaningful optimum. Java's BOBYQA needs at least two latent dimensions (an error otherwise), and with Keanu's defaults it stops short of the optimum on the linear regression model |
| `regression.json` | Fitted regression models (Java's `model` package): for linear and logistic regression with each regularization (none, ridge, lasso), default and custom priors, and one and three features, on data generated from fixed formulas: the fitted weights and intercept, predictions on held out inputs, and R² or accuracy; plus `ModelScoring` cases. Optimizers stop at slightly different points, so compare fitness and then parameters with optimization tolerances. Lasso optima on the Laplace prior's kink at 0 are approached but not reached exactly by gradient optimizers |

## Format

Every case file is `{"formatVersion": 1, "cases": [...]}`. Each case has these fields:

```json
{
  "name": "binary/plus/[2,3]x[3]",
  "op": "plus",
  "args": [<value>, ...],
  "params": {...},
  "result": <value or error>
}
```

- **`name`** is unique within its file and stable across regenerations. Use it to report and skip cases.
- **`op`** is the Java method (tensors, vertices), the distribution name (distributions), or `graph` for composite
  vertex graphs.
- **`args`** holds the inputs:
  - For tensor cases, `args[0]` is the receiver and the rest are the method's arguments. Those can be tensors,
    numbers, strings (slices), booleans, or integer arrays (dimensions, permutations, shapes).
  - For distribution cases, `args[0]` is x.
  - For vertex cases, `args` holds the input values.
- **`params`** holds the distribution parameters, keyed by name.

### Values

- **Tensors** are `{"dtype": "f64" | "i32" | "i64" | "bool", "shape": [...], "data": [...]}`, with `data` in
  row-major order. A scalar has `"shape": []` and one element.
- **Doubles** are written with Java's `Double.toString`, which round-trips exactly.
- **Non-finite doubles** are the strings `"NaN"`, `"Infinity"` and `"-Infinity"`, because JSON has no literals for them.
- **Signed zero** is preserved (`-0.0`).
- **An error result** is `{"error": "<Java exception simple name>"}`. The Rust port must also fail in that case, as
  a `Result::Err` or panic as appropriate. The error type doesn't need to match.

### Distribution results

`result` is `{"logProb": <tensor or error>, "dLogProb": {<NAME>: <tensor>, ...} or error}`.

- `logProb` is element-wise, shaped like the broadcast of x and the parameters.
- Gradient names follow Keanu's `Diffs` names: `X` (with respect to x), `MU`, `SIGMA`, `A`/`B`/`C`, `THETA`, `K`
  and so on. The parameter each one refers to is documented per distribution in the Java sources.
- For `HalfGaussian` and `HalfCauchy`, which only exist as vertices, the result is `totalLogProb` (summed over x)
  and `dTotalLogProb` with respect to x (`X`) and the parameter (`PARAM`).
- **Gradients outside the support are unspecified.** Compare them only where `logProb` is finite.

### Vertex results

`result` is `{"value": <tensor>, "forwardPartials": [...], "reversePartials": [...]}`.

- There is one partial per input, in `args` order.
- Each partial has shape `[output shape..., input shape...]`.
- Forward and reverse partials should agree. Both are recorded so each mode can be checked independently.

### Model results

Each case is `{"name", "protobuf", "json", "expected"}`, where the two paths are relative to this directory.
`expected` contains:

- `latentValues`: the value of each latent, keyed by vertex label
- `logProb` and `logLikelihood`
- `dLogProb` and `dLogLikelihood`: gradients keyed by latent label
- `nodeValues`: the value of every labelled vertex, so deterministic vertices are checked too

Java exported the models in its own (Java-compatible) protobuf schema, reloading each from both formats during
export to check it reproduced `expected` (to 1e-12, because gradient contributions are summed in an order that
depends on vertex ids). The files were then converted once to Keanu's native schema (`keanu-io/proto/keanu/graph.proto`)
by loading them with the Java-format loader and saving them again; the conversion checked that every converted
model has exactly the same operations, values, log prob and gradients as the Java-format file, and `expected` is
unchanged. The Java files and loader are at the `java-final` tag. `keanu-io`'s `round_trip` test checks the
converted files against `expected` and that saving a loaded model reproduces its file byte for byte.

## Comparing

These tolerances are suggested for the Rust parity harness:

| Kind | Tolerance |
|---|---|
| Integer and boolean results, shapes, errors | Exact |
| Element-wise arithmetic (`plus`, `times`, `div`, comparisons, masks, slicing, reshapes) | Exact (bit-for-bit, except NaN payloads) |
| Transcendental functions (`exp`, `log`, `pow`, trig, `atan2`, `sigmoid`, `logAddExp`, ...) | ≤ 1e-14 relative. Java uses `FastMath` |
| Special functions (`logGamma`, `digamma`, `trigamma`), linear algebra, distributions, gradients | ≤ 1e-10 relative, or 1e-12 absolute near zero |
| Model log probs and gradients | ≤ 1e-10 relative |

## Known issues

`manifest.json` → `knownIssues` lists cases where the recorded Java result is a known defect, not intended
behaviour. For each entry it gives the case names, a description, and the intended behaviour. The parity
harness should skip those cases, or check them against the intended behaviour, and must not reproduce the
defect. When the Java side is fixed, its entry is removed and the vectors are regenerated.

## Not covered

- **Individual random draws.** Java and Rust random streams can't be compared value for value. Samplers are
  checked statistically instead: distribution moments and Kolmogorov–Smirnov tests in Rust, and posterior summaries
  against `samplers.json`.
- **The particle filter.** It's a search heuristic whose output depends on the random stream, so it's checked by
  properties in Rust rather than against Java.

## Scope decisions

Agreed on 2026-11-03. These features are not ported to Rust:

| Feature | Java location | Decision |
|---|---|---|
| Runtime-compiled computable graph (jOOR) | `backend/keanu/compiled` | **Not ported** (Java-specific). Rust evaluates graphs directly |
| CSV POJO reader (reflection-based binding) | `util/csv/pojo` | **Dropped** |
| MIR save/load | `util/io/MIRSaver`, `MIRLoader` | **Deferred** |
| Lambda model vertex (runs an external process) | `vertices/model/LambdaModelVertex` | **Deferred** |
| Loading Java's saved model files | `util/io` | **Dropped** (2026-11-05). Rust saves models in its own versioned schema |

These are kept and ported:

| Feature | Java location |
|---|---|
| DOT graph export | `util/io/DotSaver` |
| `AssertVertex` | `vertices/utility` |
| `PrintVertex` | `vertices/tensor/generic`; becomes tracing hooks in Rust |
