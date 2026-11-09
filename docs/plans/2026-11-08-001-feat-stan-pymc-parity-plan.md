---
title: Keanu Stan and PyMC Parity Plan
type: feat
date: 2026-11-08
execution: code
---

# Keanu Stan and PyMC parity plan

Use runnable models to drive the additions. Preserve existing Rust APIs and serialized models. Treat Stan and PyMC core as the main comparison and track PyMC Extras, PyMC-BART, and PyTensor backend capabilities separately.

| Phase | Work | Acceptance |
| --- | --- | --- |
| 1. Existing APIs | Hierarchical 2PL item response and Lotka–Volterra. Follow with measurement error, hierarchical binomial experiments, precomputed splines, Bayesian VAR, and an Euler–Maruyama example using pre-drawn innovations. | Seeded tests check recovery or held-out predictions. The book explains the model, how to run it, and the data source. |
| 2. Likelihoods | Add real-valued location-scale Student-t with backward-compatible constructors and serialization. Add native negative-binomial log mean. Extend stable tail and interval likelihoods to scalar transforms, mixtures, and ordered count families. Design multivariate partial observation separately. | Compare densities, tails, RNGs, and derivatives against independent references. Run robust regression and censored frailty examples through the public API. |
| 3. Prediction and model criticism | Bind posterior draws to an explicit prediction graph by validated latent mapping. Replay deterministic and stochastic targets on differently shaped data. Add per-event multivariate log likelihood and refit-based cross-validation. Then add labeled trace export and evaluate disk-backed traces. | Spline and VAR demos forecast on held-out data with different shapes. Compare event-level log likelihood, PSIS-LOO, and refits on small reference models. |
| 4. Samplers and optimizers | Add static HMC using existing Hamiltonian integration, then slice sampling and differential-evolution Metropolis. Benchmark L-BFGS, trust-region Newton, and BOBYQA before adding full BFGS or more VI update rules. | Compare moments against exact targets and NUTS. Cover adaptation, rejection, seeding, and diagnostics. Report ESS per second and objective cost on a gradient-free example and a correlated posterior. |
| 5. Structured models and ops | Add thin QR, then evaluate SVD and least-squares helpers, Kronecker and sparse solve paths, inducing-point and periodic HSGP, structured multivariate random walks, and packaged state-space models. Extend finite-discrete elimination to shared observations and evaluate conjugate rules. | Test shape and differentiation rules and serialization. Compare dense and structured implementations on the same inputs. Add correlated regression, scalable GP, and multivariate forecasting demos. |
| 6. Specialized coverage | Prioritize missing native families and parameterizations by a concrete model, including stable multinomial logits, precision-form Gaussian, Poisson-binomial, discrete Weibull, skew Student-t, Pólya–Gamma, and Wiener censoring and variability. Consider differentiable root solving and quadrature before DAEs or adjoint ODE sensitivities. | Require a representative example and an independent numerical reference for each addition. |
| 7. Larger projects | Assess embedded Laplace, SVGD, BART, within-chain parallel reductions, FFT and complex operations, and compiled backends separately. | Define a model that needs the capability and measure the baseline before choosing implementation and dependencies. |

## Execution sequence

1. Add the 2PL and Lotka–Volterra demos with seeded synthetic data and short integration tests, then finish the remaining existing-API examples.
2. Add the Student-t API and robust-regression demo. Follow with the stable negative-binomial log-mean likelihood and wrapper support.
3. Use the spline and VAR examples to specify held-out prediction across separate graphs.
4. Run each subsequent phase as independently reviewable changes. Add fixtures or analytic checks for numerical code, and run formatting, focused tests, then the default and all-features workspace suites before integration.

## Progress

- Phase 1 is implemented. The seven examples are indexed in the book and have seeded integration tests.
- Phase 2 has real-parameter Student-t, negative-binomial log mean, and censoring and truncation for scalar transforms with a base CDF and for logit-binomial counts. Robust regression and censored frailty exercise these APIs. Mixture and additional count tails, exact Hessians for wrapped transforms, and partial multivariate observation remain open.
- Phase 3 has explicit posterior replay across graphs, opt-in per-event likelihood for the multivariate Gaussian and shared event families, K-fold and leave-one-out refits, and labeled CSV export. The spline and VAR examples forecast from graphs separate from training.
- Streaming CSV export writes rows without building a second copy of the trace. Sampling still collects draws in memory. A disk-backed trace would require samplers to write draws as they arrive and diagnostics to read them without loading every chain. Defer that storage design until a workload exceeds the in-memory trace capacity and can guide the format and access pattern.
- Phase 4 adds static HMC with optional step-size adaptation, coordinate slice sampling, and differential-evolution Metropolis. Seeded tests cover exact posterior moments, NUTS comparison, transforms, reproducibility, and rejection.
- The [inference benchmark](../benchmarks.md#phase-4-inference-benchmark) compares sampler ESS per second and evaluation counts on correlated and discontinuous targets. L-BFGS, trust-region Newton, and BOBYQA reached the same optima on the Gaussian and Rosenbrock targets. Full BFGS and additional variational update rules remain deferred pending a representative workload.
- Phase 5 adds thin QR with batched first- and second-order graph derivatives and saved-model support, tensor least squares, a separable Kronecker solve, inducing-point and periodic GP priors, and multivariate walk and local-level builders. The examples exercise correlated regression, a held-out GP period, and correlated forecast uncertainty.
- Joint finite-discrete elimination supports shared likelihoods and dependent priors for scalar groups up to 1,000 assignments. Larger or tensor groups remain open.
- QR covers the full-rank least-squares use case. Defer SVD until a rank-deficient or spectral model needs it. CSR multiplication already provides a sparse graph path. A sparse direct solve needs a matrix class and a factorization reuse policy, so defer it until a sparse solve workload can measure the gain.
- Keep conjugate likelihoods explicit for now. The beta-binomial and Kalman likelihood already cover representative collapsed models. A general conjugacy rewrite needs to preserve exact observation normalizers and return the conditional posterior of eliminated variables.
- Phase 6 starts with stable multinomial logits and precision-form Gaussian distributions. Both have batched events, sampling, first- and second-order derivatives, and serialization. The examples exercise grouped vote-share regression and Gaussian signal denoising.
- Poisson-binomial and discrete Weibull add batched counts, sampling, exact first- and second-order derivatives, and serialization. Discrete Weibull supports censoring and truncation. The examples fit heterogeneous batch totals and censored device lifetimes.
- Poisson-binomial exposes differentiable CDF and survival methods at the distribution level. The calibration demo averages the predictive exceedance probability over posterior draws. Jones–Faddy skew Student-t adds density, sampling, log tails, exact derivatives, and serialization, with a regression example for asymmetric residuals.
- Poisson-binomial graph censoring and truncation use exact derivatives and direct log-space interval masses. The calibration demo fits top-coded counts and predicts exceedance risk for the underlying totals.
- Shape-one Pólya–Gamma PG(1, z) supports exact sampling, differentiable density, and saved graph models. The Bernoulli logistic Gibbs demo checks the augmented posterior against grid integration of the original model.
- General Pólya–Gamma shape and tails, and Wiener tails and additional variability parameters remain open. Differentiable root solving and quadrature remain under evaluation.

## Reference demos

- [Stan hierarchical item response](https://mc-stan.org/users/documentation/case-studies.html)
- [Stan Lotka–Volterra](https://mc-stan.org/users/documentation/case-studies/lotka-volterra-predator-prey.html)
- [PyMC example gallery](https://www.pymc.io/projects/examples/en/latest/gallery.html)
