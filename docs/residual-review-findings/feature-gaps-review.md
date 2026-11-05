# Known residuals: feature-gap work code review

Review of `e8d5a30c1..main` (plan `docs/plans/2026-11-04-001-feat-ppl-feature-gaps-plan.md`): correctness (three slices), testing, maintainability and adversarial reviewers plus an independent cross-model (Codex) adversarial pass. All P1 findings and every P2 correctness finding were fixed (see CHANGELOG "Code review fixes").

Updated 2026-11-05. Every lower-priority item recorded here has since been addressed (the robustness items, the test and helper items and the maintainability items), except the two below, which stay open with their current status.

## Sampling quality
- The stochastic volatility example (`keanu/examples/stochastic_volatility.rs`) still has a few divergences, under 1% of draws, and its test allows that. NUTS now adapts on Stan's windowed schedule and counts divergences as Stan does (only energy increases). Stan's User's Guide parameterisation was tried as well: no day-0 state, with h_1 = mu + sigma / sqrt(1 − phi²) · eps_1. It doesn't reach zero. Measured with 4 chains of 500 draws over seeds 1–8:

  | Model | Target | Divergences per seed | Largest R-hat |
  |---|---:|---|---:|
  | This example | 0.95 | 2, 2, 2, 1, 1, 1, 0, 44 | 1.023 |
  | Stan's form, this example's priors | 0.8 | 4, 215, 2, 32, 3, 6, 4, 3 | 1.31 |
  | Stan's form, this example's priors | 0.95 | 5, 0, 2, 3, 9, 0, 3, 1 | 1.013 |
  | Stan's form, Stan's priors (Uniform phi, Cauchy mu) | 0.8 | 1, 91, 5, 4, 6, 2, 62, 28 | 1.089 |
  | Stan's form, Stan's priors (Uniform phi, Cauchy mu) | 0.95 | 2, 11, 9, 0, 0, 0, 1, 1 | 1.011 |

  The example, its tolerance (fewer than 1% of draws divergent, R-hat below 1.02) and its run time (about 25–40 s per run with the `parallel` feature, longer without it) are unchanged.

## Maintainability
- Large files left as they are:
  - `keanu-graph/src/autodiff/mod.rs` (about 1940 lines).
  - `keanu-graph/src/vars.rs` (about 1010 lines). Most of it is one `impl DoubleVar` block plus macros that all three variable types share, so it can't be split by family as a mechanical move.
  - `keanu/tests/book.rs` (about 1030 lines). The book includes its ANCHOR regions by path, 39 times.

  `transform.rs`, `continuous.rs` and `tests/mcmc.rs` were split into modules by family.
