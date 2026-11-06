# Known residuals: applied modelling gaps code review

Review of `d60e438a7..ba33329f8` (plan `docs/plans/2026-11-05-001-feat-applied-modelling-gaps-plan.md`, U1–U27), 2026-11-06: six correctness reviewers, one per slice (graph core; distributions; ODEs and the matrix exponential; MCMC; model comparison, export and VI; SMC, ABC, state space and HSGP), each verifying suspected defects with scratch tests. One P1, five P2 and seven P3 findings were reported, and all were fixed with regression tests in `91e56b0d2..c7d7b4b59` (see CHANGELOG "Applied modelling gaps", "Code review fixes for the above").

What remains is behaviour that is by design or documented, recorded here so follow-up work starts from it.

## Accuracy and performance

- **Single-path Pathfinder is approximate even on Gaussian targets.** The implementation was checked against the paper and Stan. The error comes from the default L-BFGS history of 6, which doesn't span the space near convergence, and from the noisy 5-draw ELBO choosing among late iterates that are nearly equal. Multi-path Pathfinder, or a longer history, is much closer. On radon, starting NUTS from Pathfinder draws didn't shorten the warm-up compared with the default start.
- **An ODE log-probability-and-gradient call solves twice:** once without sensitivities for the value and once with them for the gradient. On the SIR example the extra solve costs about 0.3 ms of 1.3 ms. The stiff solver's Jacobian for the combined system uses central differences of body JVPs. Adjoint sensitivities are out of scope.
- **The Kalman filter (structural time series) costs about 1–2 ms per gradient** on the CO₂ example, mostly overhead on small tensors.
- **SMC mixes slowly when discrete and continuous latents move together**, as in change-point models, so every stage then uses its full step budget.
- **Test-suite time grew.** The example tests take about 60–100 s, and the HSGP, ODE, structural time series and occupancy tests are the slowest.

## Scope limits (documented)

- ODE models have no second-order rule (`NoSecondOrderRule`), so the Laplace approximation doesn't support them. Partials with respect to constant parents of an ODE aren't computed.
- `PredictiveSampler::posterior` returns `SolverFailed` when a posterior draw's solve fails. Prior and forward sampling, and approximation draws, redraw instead, so their draws come from the distribution restricted to where the model solves. Pathfinder can return fewer draws than requested.
- Marginalisation supports the dependency patterns listed in the `marginalize` docs only. Latents that share observations can't be marginalised jointly.
- ABC supports continuous latents only, and its evidence only compares models with the same kernel, ε and statistics.
- Minibatch ADVI skips the N mod B rows left over at the end of each epoch, and Stan's η grid search is fragile on narrow posteriors with or without minibatches.
- The CmdStan CSV's `num_samples` is written as rows × thin, and `lp__` excludes the Jacobian that Stan includes.
- A batch of multivariate observations counts as one pointwise log-likelihood point (PyMC counts one per vector).
- With a regression component, structural time series assume the regression weights are 0 when choosing the initial level.
