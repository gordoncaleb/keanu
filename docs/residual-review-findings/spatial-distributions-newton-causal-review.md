# Known residuals: spatial models, distributions, Newton and causal tools review

Review of `687633475..05c586e2a` (plan `docs/plans/2026-11-06-001-feat-spatial-distributions-newton-causal-plan.md`, U1–U13), 2026-11-06/07: six correctness reviewers, one per slice (sparse and spatial; special functions; continuous families and mixtures; discrete and multivariate families; Wiener; finite-difference Hessians, trust region, `intervene` and imputation). They found two P1, three P2 and eight P3 problems, all fixed with regression tests in `23ad9d1a5..7d6fcaa61` (see CHANGELOG "Spatial models, distributions, Newton and causal tools", "Code review fixes for the above").

Remaining limits, by design or documented:

## Accuracy and sampling
- **The disease-mapping example uses a Beta(1, 1) prior on BYM2's ρ, not Morris et al.'s Beta(0.5, 0.5).** Infinite log densities at a bound are now impossible points, so Beta(0.5, 0.5) samples correctly. But ρ's R-hat exceeded 1.01 for 5 of 8 seeds (1.012–1.20) at the example's draw counts.
- **`hyp2f1` needs double-double arithmetic outside 0 ≤ z ≤ ½ with positive parameters.** It is accurate to a few ulps there, except near-degenerate polynomial cases such as ₂F₁(−21, 5.03; 1.03; 0.999), which is still off by 2e-3. `hyp2f1_derivatives` stays in f64 and loses accuracy where the series cancels.
- **Wishart draws are singular to working precision when ν − K + 1 is far below 1.** Their diagonal is never exactly 0.
- **Trust-region Newton is not cheaper than L-BFGS on radon by a fair cost count:** about 374 gradient-equivalents against 130. It needs fewer gradient evaluations and converges more tightly.
- **Laplace's φ (overdispersion) on the SIR example is 15–19% off NUTS,** because that posterior is strongly skewed.
- **The drift-diffusion example is written in centred form,** because the non-centred form diverged with only four subjects.

## Scope limits (documented)
- **Missing CDFs:** Rice, transformed distributions and the new discrete and multivariate families have none, so they can't be truncated or censored.
- **Not Gibbs-enumerable:** Dirichlet-multinomial and Plackett–Luce latents.
- **Mixture restrictions:** a `mixture_of` can't be a mixture component, truncated or censored. `transformed` takes element-wise bijectors only.
- **Unit-vector latents:** they use Stan's non-bijective auxiliary approach, and Laplace refuses them.
- **Restrictions on rewrites:**
  - `observe_masked` rejects multivariate kinds and `mixture_of`.
  - `intervene` drops only unobserved nodes whose every child is intervened or dropped.
- **ICAR and BYM2 setup cost:** they compute dense eigendecompositions at build time, O(n³), which is fine to a few thousand areas.
- **Constructor sizes:** `Adjacency::lattice` called directly with an enormous size isn't size-limited; loading saved files is.
- **Test time:** a full all-features test run now takes about 15 minutes.
