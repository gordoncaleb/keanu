# Known residuals: feature-gap work code review

Review of `e8d5a30c1..main` (plan `docs/plans/2026-11-04-001-feat-ppl-feature-gaps-plan.md`): correctness (three slices), testing, maintainability and adversarial reviewers plus an independent cross-model (Codex) adversarial pass. All P1 findings and every P2 correctness finding were fixed (see CHANGELOG "Code review fixes"). These lower-priority items remain.

## Robustness
- keanu-graph: ops shape-checked by dummy evaluation (e.g. `exp` of a huge broadcast, gather) still allocate their inputs' full size at build time; shapes under the `isize::MAX / 8` cap but enormous (e.g. `[2^40, 3]`) can exhaust memory when loading an untrusted file. Fix: explicit shape rules instead of dummy evaluation.
- keanu-graph: a scan of 2^40 steps over a scalar carry is within the cap, so a tiny file can make evaluation loop for a very long time.
- keanu-graph `ScanBody::jvp` multiplies batch × length × carry size unchecked.
- `add_proxy` / `add_distribution` with huge shapes from Rust code are not capped (only the loader caps them).
- Weibull sampling can return +inf with probability 2^-53 per draw (u = 0); NegativeBinomial gradients are NaN at mu = 0; Weibull `d_log_cdf` is NaN when (x/beta)^alpha underflows.
- OrderedLogistic, Mixture and HMM return `Err` (not −∞) when latent parameters leave their valid region, so Metropolis–Hastings aborts rather than rejects.
- `eigh` fixes eigenvector signs by the largest component; equal-magnitude components can flip under rounding while derivative rules assume smoothness.

## Sampling quality
- The stochastic volatility example still has a few divergences (< 1% of draws; the test tolerates that) and its test takes ~60 s without the `parallel` feature. NUTS trees nearly always hit 31 leapfrog steps there; Keanu's diagonal mass-matrix adaptation may be worth comparing with Stan's.

## Tests
- `keanu-algorithms/tests/laplace.rs`: the L-BFGS settings test only asserts `Ok`.
- `keanu-algorithms/tests/mcmc.rs`: the constrained-space NUTS regression compares floats bit-exactly.
- `keanu-graph/tests/hessian.rs`: the op and distribution coverage lists are hand-maintained (derive them from the enums).
- U18's "L-BFGS at or below CG's evaluation count" is printed in `golden_optimizers.rs`, not asserted.
- Test helpers (`assert_close`, `s`, `v`, …) are duplicated across ~25 test files with differing argument orders.

## Maintainability
- `keanu-distributions/src/timeseries.rs` has a private `Mat` (Cholesky, inverse) duplicating keanu-tensor linalg, and `keanu-algorithms/src/laplace.rs` hand-rolls triangular solves.
- `keanu-graph/src/transform.rs` (~1550 lines) and `autodiff/mod.rs` (~1900 lines) are large; `vars.rs`, `continuous.rs`, `tests/mcmc.rs` and `tests/book.rs` passed 1000 lines.
- ~69 doc comments still say "Java's X" although the Java code is gone.
