# Changelog

All notable changes to Keanu. The Rust rewrite replaces the Java library; entries below "Rust rewrite" describe
the Java releases, kept for history.

## Unreleased: Rust rewrite

Keanu is now a pure Rust library: a Cargo workspace of the `keanu` facade crate and the `keanu-tensor`,
`keanu-distributions`, `keanu-graph`, `keanu-algorithms` and `keanu-io` crates. The Java implementation, its Gradle
build, the examples, benchmarks and documentation site were removed; the last Java version is at the `java-final`
tag. `RUST_MIGRATION_PLAN.md` describes the rewrite and `book/src/from-java.md` the differences for Java users.

- Same model semantics as the final Java version, checked against frozen golden vectors recorded from it
  (`keanu-parity/fixtures/golden-vectors`). Models saved by the final Java version load in Rust, and models saved by
  Rust use the same protobuf and JSON formats.
- Immutable, thread-safe graphs with a separate, cloneable `State`, so chains and particles can run in parallel.
- No global state and no native dependencies. BOBYQA is a translation of PRIMA's (BSD-3-Clause); everything else
  is MIT.
- The "Unreleased: Java" changes below were made to the Java implementation during the rewrite, mostly bug fixes
  the Rust port found, so that the Rust version is a faithful conversion.

### Closing the feature gaps with PyMC and TensorFlow Probability

Work from `docs/plans/2026-11-04-001-feat-ppl-feature-gaps-plan.md`. Java compatibility is no longer a goal, so
some of the "Same model semantics" guarantees above no longer hold where noted.

- keanu-graph: every distribution reports its `Support` (`DistributionKind::support`, `node_support`,
  `CustomDistribution::support`).
- keanu-graph: `transform` module of bijectors (lower bound, interval, ordered, simplex via the inverse isometric
  log-ratio as in Stan 2.37, correlation Cholesky, circular), and `Parameterisation` / `Space`, which map a flat
  unconstrained vector to a state and evaluate the log probability with or without the log-Jacobian.
- keanu-algorithms: **NUTS and the gradient optimizers now run in unconstrained space by default**
  (`Nuts::with_space`, `GradientOptimizer::with_space`; `Space::Constrained` restores the old behaviour exactly). NUTS
  includes the log-Jacobian; MAP and maximum likelihood don't, so interior optima are unchanged. A mode on a support
  boundary is reported in the new `OptimizedResult::boundary` instead of running to the iteration cap.
  Metropolis–Hastings stays in constrained space; BOBYQA derives default bounds from scalar supports.
  - Re-baselined tests: the golden Adam parity cases run in `Space::Constrained` (unconstrained Adam reached
    25.99994 on thermometers against Java's 25.99494); `conjugate_gradient_stops_at_an_impossible_start` and
    `an_impossible_optimum_is_not_converged` were adapted to the support checks.
- keanu-io: **new versioned Keanu-native save schema** (`proto/keanu/graph.proto`) replacing the Java-compatible
  format; Java-saved files no longer load. The golden models were converted with identical log probabilities.
  `Loaded::ids` is gone (saved node i is `NodeId` i); `IoError` gains `UnsupportedVersion` and `InvalidNode`;
  integers are saved as 64-bit; CPT and categorical nodes can now be saved.
- keanu-algorithms: multi-chain diagnostics following Vehtari et al. (2021), matching ArviZ: rank-normalised split
  R-hat, bulk and tail ESS, MCSE and a summary table (`ChainSamples`, `sample_chains`, `diagnostics`).
- keanu-algorithms: `PredictiveSampler` for prior- and posterior-predictive draws.
- keanu-tensor: special functions (erf, erfc, erfinv, normal CDF, log-CDF and quantile, softplus, logit,
  log1mexp, log-beta, regularised incomplete gamma and beta with shape derivatives, log-scaled Bessel I0 and I1),
  with the elementwise ones also available as graph ops.
- keanu-distributions, keanu-graph: `LkjCholesky` prior on Cholesky factors of correlation matrices.
- keanu-graph: indexing by tensor (`Op::Gather`, `Op::ScatterAdd`, `Op::Stack`) for hierarchical models, and stable
  logsumexp, softmax and log-softmax. The crate-private strided `Tensor::gather` is now `strided_read`.
- keanu-algorithms: NUTS records the energy of every step; `NutsStatistics::divergence_count`, `divergent_flags` and
  `diagnostics::e_bfmi`.
- keanu-distributions, keanu-graph: log CDFs and survival functions (`log_cdf`, `log_ccdf` and their gradients) for 16
  distributions; `Truncated` and `Censored` wrappers; `NegativeBinomial` (mean and dispersion), `Weibull`, `VonMises`
  (on a circular transform) and `OrderedLogistic`; same-family `Mixture`; `HiddenMarkovModel` (log-space forward
  algorithm) and `LinearGaussianStateSpace` (Kalman filter with the Joseph-form update). New `Support::LessThan` and
  `transform::UpperBound`. `parameter_types()` and `parameter_names()` now return `Cow`. The save schema gained a
  recursive `DistributionType` for nested kinds.
- keanu-graph: linear algebra ops `SolveTriangular`, `Solve`, `LogAbsDeterminant`, `DeterminantSign`, `Eigvalsh` and
  `EighVectors` (`DoubleVar::slogdet`, `eigh`).
- keanu-graph: `gp` module: covariance kernels (exponentiated quadratic, Matérn 1/2, 3/2 and 5/2, rational quadratic,
  periodic, linear, sums and products), the exact GP marginal likelihood and the conditional predictive distribution.
- keanu-graph: `Op::Scan` is now a general deterministic loop over a nested graph (`GraphBuilder::scan`,
  `scan_steps`), with forward, reverse and second-order autodiff and save support. **The old cumulative op is renamed
  `Op::Cumulative` (`ScanOp` → `CumulativeOp`).**
- keanu-graph: exact Hessian-vector products (`Parameterisation::hvp`, `hessian`) by forward-over-reverse
  differentiation, with second-order rules for every op, bijector and distribution, except custom distributions,
  lambdas, `LinearGaussianStateSpace`, and the shape parameters of CDFs built on incomplete gamma or beta functions,
  which return the new `GraphError::NoSecondOrderRule`.
- keanu-algorithms: **L-BFGS with a strong-Wolfe line search is the new default gradient optimizer** (Stan's settings).
  It needed far fewer evaluations than conjugate gradient on the golden models (36 against 320 on linear
  regression MAP); three tests that check CG-specific behaviour now name `ConjugateGradient`.
- keanu-algorithms: `Laplace` approximation at the mode in unconstrained space, using the exact Hessian; new
  `AlgorithmError::ModeOnBoundary` and `IndefiniteHessian`.
- keanu-algorithms: mean-field and full-rank `Advi` with Stan's defaults; new `AlgorithmError::NonFiniteElbo`.
- keanu-algorithms: the optimizers now compute deterministic nodes that depend only on constants before starting
  (they failed with `MissingValue` before).
- Code review fixes for the above:
  - keanu-io: malformed CPT nodes and shapes or scan lengths too large to allocate are load errors instead of panics;
    every node's element count is capped at `isize::MAX / 8`.
  - keanu-algorithms: constant (frozen) chains get NaN R-hat, ESS and MCSE, so they are never reported converged;
    draws count as constant only when exactly equal.
  - keanu-graph: latent mixture weights and HMM chain probabilities no longer read as a moving support; the bounded
    bijectors accept per-element mixed infinite bounds; far-tail truncated draws are inverted in log space instead of
    returning the bound.
  - keanu-tensor, keanu-distributions: incomplete gamma uses Temme's expansion for large shapes and returns NaN on
    non-convergence; new log-space `log_gamma_p`, `log_gamma_q`, `log_inc_beta` (and shape derivatives) keep far-tail
    log CDFs finite; public `special::pentagamma` and `special::log_sum_exp` (NaN-propagating, shared).
  - keanu-algorithms: the line search's minimum step is relative to its initial step (L-BFGS no longer stalls on huge
    gradients); a boundary is reported only when confirmed, so interior optima far from 1 converge normally; the
    Laplace approximation rejects circular latents. `Laplace::lbfgs` is now `with_lbfgs`.
  - keanu-graph: `DistributionKind::nested()` and `Nested` describe wrapped kinds; `Graph::describe` names a node.
- Follow-ups after the review:
  - keanu-distributions, keanu-graph: **exact second derivatives everywhere**: `LinearGaussianStateSpace::hvp`
    (forward mode through the Kalman filter, so the Laplace approximation works on Kalman models) and the shape
    parameters of truncated and censored gamma, beta and negative binomial distributions, from new second shape
    derivatives in `keanu_tensor::special` (`gamma_p_da2`, `inc_beta_da2`, `inc_beta_dadb`, `inc_beta_db2` and log
    variants). Only lambdas and custom distributions lack second-order rules. The Kalman filter now uses
    keanu-tensor's linear algebra.
  - keanu-tensor: DiDonato & Morris's asymptotic expansion for the incomplete beta at large shapes (1e-13 relative up to
    1e14); pentagamma to 1e-14; `eigh` signs each eigenvector so its first near-largest component is positive.
  - keanu-graph: every op's shape is inferred from its parents' shapes without evaluating; explicit proxy and
    distribution shapes are capped. **`GraphBuilder::add_proxy` now returns `Result<NodeId>`.**
  - keanu-io: **load limits for untrusted files** (`LoadOptions`, `load_protobuf_with`, `load_json_with`): by default at
    most 2^31 elements per node, 2^24 scan steps and 2^32 units of scan work. Larger trusted models need the `_with`
    functions.
  - keanu-graph, keanu-distributions: invalid parameters that come from the state (crossed truncation bounds,
    non-increasing cutpoints, mixture weights or transition rows off the simplex) give a log probability of −∞ instead
    of an error, so Metropolis–Hastings rejects them; Weibull sampling can no longer return +inf; NegativeBinomial and
    Weibull gradients have the correct limits at mu = 0 and x = 0.
  - keanu-algorithms: NUTS checks for U-turns across subtree boundaries, as Stan does (trees on a 100-dimensional
    Gaussian no longer run to the 1023-step maximum), and adapts its step size and diagonal metric on Stan's windowed
    schedule (75-step fast start, doubling slow windows from 25, 50-step fast end, regularised variance).
    `QuadraticPotential::adaptation_window` now means the first slow window (default 25, was 100), new
    `initial_buffer` and `terminal_buffer`; `initial_mean` and `initial_weight` are deprecated and ignored. NUTS draws
    for a given seed change.
  - keanu-algorithms: NUTS counts only energy *increases* beyond `max_energy_change` (or NaN) as divergences, as Stan
    does.
  - keanu-graph: public `Op::NAMES`, `DistributionKind::NAMES`, `DistributionKind::CDF_NAMES` and
    `DistributionKind::name`; `transform` and keanu-distributions' continuous distributions are split into modules
    by family (public paths unchanged).
- keanu: new examples `eight_schools`, `radon` (data from the PyMC examples repository, MIT), `gaussian_mixture` and
  `gp_regression`; new book chapters on variational inference and Gaussian processes.

### Applied modelling gaps

Work from `docs/plans/2026-11-05-001-feat-applied-modelling-gaps-plan.md`.

- keanu-graph: **factor nodes** (`Op::Factor`, `GraphBuilder::factor`), terms of the joint log probability like
  PyMC's `Potential` or Stan's `target +=`. A factor's value is the sum of its expression, and a value that is not
  finite makes the point impossible. Factors count in the joint log probability and the log likelihood (not the log
  prior), in their gradients and Hessian-vector products, and in Metropolis–Hastings' incremental updates;
  `Model::factors`, `Graph::factors`, `Node::is_factor` and `Op::is_factor` list them. Lambda sections include the
  factors they reach. Scan bodies can't hold factors. The save schema gained `factor`.
- keanu-graph: **log-density op** (`Op::LogDensity(DistributionKind)`, `GraphBuilder::log_density`): the
  element-wise log density of a value under an element-wise distribution (one value per element, or a truncated or
  censored one), as a deterministic f64 node of the broadcast shape of the value and the parameters. The value has
  the kind's type (f64, or i64 for counts); other kinds, types and parameter counts are rejected when the node is
  built. Its first and second derivatives reuse the distributions' element-wise rules. The save schema gained
  `log_density`, holding a `DistributionType`, and the DOT export names the kind.
- keanu-algorithms: **pointwise log likelihood** (`loglik::PointwiseLogLik`, `LogLikPoint`, also in the `keanu`
  prelude): the log likelihood of each observation point at each draw of each chain, `[chain][draw][point]`, from
  `ChainSamples` (`from_chains`) or one `NetworkSamples` (`from_samples`), computed after sampling so it works for
  every algorithm's draws. Points are the elements of an observed node whose distribution has one value per element
  (named as ArviZ labels them, `y[3]` or `y[1, 2]`) and whole nodes otherwise; factors have none. The samples must
  record every latent the observations depend on. With the `parallel` feature the draws are evaluated on rayon's
  thread pool. keanu-graph gained `node_log_prob_terms`, the unreduced terms of a node's log probability.
- keanu-algorithms: **Pareto-smoothed importance sampling** (`psis::psis`, `Psis`, `gpdfit`, `GeneralizedPareto`,
  `khat_threshold`, `tail_length`) following Vehtari et al. (2024) and R's `loo`: the ⌈min(0.2·S, 3·√(S / r_eff))⌉
  largest ratios are fitted with Zhang and Stephens' generalized Pareto estimate, its shape shrunk towards 0.5 by a
  prior worth 10 observations, and replaced by the fit's expected order statistics; weights are truncated at the
  largest raw weight. k̂ is reliable up to min(1 − 1 / log₁₀(S), 0.7). Tails too short to fit give k̂ = +∞, as in
  `loo`, and tails of equal ratios k̂ = −∞ (where `loo` reports +∞). Matches arviz-stats' `psislw` on fixed vectors.
- keanu-algorithms: **PSIS-LOO, WAIC and model comparison** (`loo::loo`, `waic`, `loo_from_values`,
  `waic_from_values`, `relative_efficiency`, `compare`, `stacking_weights`, `pseudo_bma_weights`, the `Loo`, `Waic`
  and `Comparison` results and the `ElpdEstimate` trait; `Loo`, `Waic`, `loo` and `waic` also in the `keanu`
  prelude), following Vehtari, Gelman and Gabry (2017) and R's `loo`: totals with standard errors, pointwise
  values, each point's Pareto k̂ and relative efficiency (from its chains' effective sample size), the Monte Carlo
  standard error of the LOO elpd, and warnings naming unreliable points. `compare` ranks models with `loo`'s
  negative `elpd_diff` and its paired standard error, and stacking weights (Yao et al., 2018, by L-BFGS on the
  softmax parameterisation); pseudo-BMA+ weights use a 1000-draw Bayesian bootstrap. The eight schools example
  reports LOO (elpd about −30.8, as ArviZ), and the guide has a new chapter, "Model comparison (LOO and WAIC)".
- keanu-algorithms: **CmdStan CSV export** (`export::CmdStanCsv`, also in the `keanu` prelude), one file per chain
  (`write_chain` to any `io::Write`, or `write_files`) in the layout ArviZ's `from_cmdstan` and Stan tools read:
  configuration comments, a header with Stan's dotted 1-based indices in column-major order, NUTS's adaptation
  results, and a line per draw with `lp__`, NUTS's `accept_stat__`, `stepsize__`, `treedepth__`, `n_leapfrog__`,
  `divergent__` and `energy__` when its statistics are given, the sampled nodes, and `log_lik.i` columns from a
  `PointwiseLogLik`. Other samplers' files have only `lp__`. `NutsStatistics` gained `accept_stat` and `tree_depth`
  (recorded with `save_statistics`) and `inverse_metric` (always), `NutsSampler` gained `inverse_metric()`, and
  `AlgorithmError` gained `Io`.
- keanu-algorithms: **NUTS block mode** (`Nuts::block_sampler`): NUTS over a chosen block of the continuous
  latents, with every other latent (continuous or discrete) held at its value in the state, so that NUTS can be one
  step of a compound sampler. Before each step the chain checks whether the state still holds the values it left
  there and, if another step changed them, refreshes its position, log probability and gradient from the state
  (`NutsSampler::refresh` does this on demand); adaptation carries on across those changes. A block's supports may
  not move with any latent. `NutsSampler` gained `log_prob`, `latents`, `is_block` and `into_statistics`.
  keanu-graph gained `Parameterisation::for_latents`, `Model::log_prob_and_gradient_wrt` (the gradient with respect
  to some continuous latents) and `Model::local_log_prob` (the log probability of some nodes only). Plain NUTS
  takes exactly the same trajectories as before.
- keanu-algorithms: **Gibbs and discrete Metropolis steps** (`mcmc::gibbs::Gibbs` and `DiscreteMetropolis`, with
  their `GibbsSampler` and `DiscreteMetropolisSampler`; both also in the `keanu` prelude), sampling algorithms for
  discrete latents that update one element at a time from its Markov blanket's log probability. `Gibbs` draws each
  element from its exact conditional distribution by evaluating every value it can take: Bernoulli, binomial,
  categorical, ordered logistic and discrete uniform latents, and truncated binomial, Poisson and negative binomial
  latents with a finite upper bound, with the supports read from the current parameter values at every step. An
  element with more than `max_support_size` values (default 1000) is an error that points to `DiscreteMetropolis`.
  `DiscreteMetropolis` is random-walk Metropolis–Hastings with steps drawn uniformly from ±1, …, ±`max_step` (default
  2), for integer latents with infinitely many values (Poisson, negative binomial, geometric) or any other integer
  latent except a multinomial. keanu-graph gained `DistributionKind::finite_support` (the values each element can
  take, given the parameters) and `DistributionKind::has_finite_support`.
- keanu-algorithms: **compound sampler** (`mcmc::compound::Compound`, `CompoundStep`, `CompoundSampler` and
  `CompoundStatistics`; `Compound` also in the `keanu` prelude): a cycle of steps over disjoint blocks of the latents
  (NUTS in block mode, Gibbs, discrete Metropolis and Metropolis–Hastings), built with
  `Compound::new().gibbs(..).nuts(..)` or chosen by `Compound::for_model`, which takes Gibbs for discrete latents
  with at most 1000 values per element, the random walk for other integer latents, Metropolis–Hastings for anything
  else, and NUTS last for the continuous latents when the log probability is differentiable with respect to them
  (the new `select::is_differentiable_wrt`, which ignores operations that depend only on other latents).
  `with_nuts_settings` replaces the NUTS steps' settings. Every latent must belong to exactly one step. NUTS adapts
  over the compound chain's first `adapt_count` steps, its statistics are returned in `CompoundStatistics::nuts`,
  and chains run in parallel with `sample_chains` as for the other samplers.
  **Default sampler change:** `Sampler::for_model` (with the new variants `Sampler::Gibbs` and
  `Sampler::Compound`) now returns `Gibbs` for a model whose latents are all discrete with finite supports, and the
  compound sampler for any other model with a discrete latent; both used to get Metropolis–Hastings for every
  latent, so their draws change. All-continuous models still get NUTS (or Metropolis–Hastings when not
  differentiable), and `MetropolisHastings` asked for explicitly is unchanged. `ChainSamples::element`, and with it
  `rhat`, `ess_bulk`, `ess_tail`, `mcse_mean` and `summarize`, now accepts integer and boolean nodes (as 0 and 1),
  so discrete latents can be checked for convergence. keanu-graph: the **categorical distribution now has a
  gradient** (and second derivatives) with respect to its probabilities, which gradient based samplers need when
  the probabilities are latent (a mixture's weights with explicit assignments); before, that term was silently left
  out of the gradient. The coal mining disasters, text messaging and cheating students (Bernoulli form) examples
  use the default compound sampler with four chains (the first two) instead of Metropolis–Hastings. Re-baselined
  tests: `coal_mining_disasters_switch_year` (now also every R-hat below 1.01 and the posterior means within four
  Monte Carlo standard errors of the exact conjugate posterior), `text_messaging_switch_point_and_rates` (the same
  checks; the late rate's mode, meaningful only for Metropolis–Hastings' repeated values, became its mean) and
  `cheating_students_with_bernoulli` (against the exact posterior mean on a grid) in `keanu/tests/examples.rs`;
  `metropolis_hastings_samples_a_discrete_posterior_exactly` (asks for Metropolis–Hastings explicitly, and checks
  that the default is now Gibbs) in `keanu-algorithms/tests/mcmc_metropolis.rs`;
  `chain_samples_summarise_every_element_of_every_f64_node` (an integer node's element is no longer an error) in
  `keanu-algorithms/tests/diagnostics.rs`; and
  `uniform_and_discrete_nodes_contribute_no_gradient_with_respect_to_themselves` (the categorical's gradient) in
  `keanu-graph/tests/logprob.rs`. The guide's sampling chapter has a new section on discrete latents.
- keanu-graph: **automatic marginalisation of discrete latents** (`marginalize`, `Marginalized`,
  `MarginalizedLatent`, `MAX_MARGINALIZED_VALUES`; `marginalize` and `Marginalized` also in the `keanu` prelude), a
  graph rewrite that sums Bernoulli, categorical and other finite-support element-wise discrete latents (at most
  1000 values, the same and fixed for every element) out of the model. For each value k it copies the deterministic
  path from the latent to its observed children with the latent replaced by k, sums each child's log density over
  its trailing axes, adds log p(z = k), and puts the log-sum-exp over k in a factor; the children become constants
  holding their observations and the latent and its path are removed, so the rewritten model is all continuous and
  gets NUTS by default. A vector latent's path may use only element-wise operations, axis-preserving reshapes,
  broadcasts and reductions, and `gather` by the latent along the first axis; a scalar latent's may use any
  deterministic operation. Several latents are marginalised one after another. Unsupported patterns fail with the
  new `GraphError::CannotMarginalize`, naming the node in the way. keanu-algorithms: `marginal::recover_marginals`
  (with `LatentMarginals`, both also in the `keanu` prelude) gives each draw's conditional probabilities
  p(z = k | draw), their mean over draws, and draws of the latent. `Op::LogDensity` now accepts the Bernoulli (a
  boolean value), and `Model::new` accepts a graph whose only log-probability terms are factors. The Gaussian
  mixture example gained `run_marginalized` (explicit assignments, marginalised), and the guide's "Building models"
  chapter a section on marginalising discrete latents.
- keanu-tensor and keanu-graph: **matrix exponential** (`DoubleTensor::matrix_exp`, `Op::MatrixExp`,
  `DoubleVar::matrix_exp`), batched over leading dimensions, by scaling and squaring with Padé approximants of
  degree 3, 5, 7, 9 or 13 chosen by the 1-norm (Higham, 2005, as MATLAB's `expm`). Its first derivatives use the
  Fréchet derivative L(A, E), the upper right block of exp([[A, E], [0, A]]) (`DoubleTensor::matrix_exp_frechet`),
  with L(Aᵀ, Ḡ) as the vector-Jacobian product, and its Hessian-vector products the second Fréchet derivative from
  3 × 3 block matrices (`DoubleTensor::matrix_exp_frechet_second`). A matrix with a non-finite element gives NaN.
  The save schema gained `matrix_exp`.
- keanu-graph: **ODE integrators** (the new `ode` module: `integrate`, `solve`, `solve_sensitivities`, `OdeSystem`,
  `SensitivitySystem`, `FnSystem`, `OdeOptions`, `OdeSolver`, `OdeSolution`, `OdeStatistics`, `OdeError`,
  `SolverFailed` and `FailureReason`) on plain `&[f64]` states: Dormand–Prince 5(4) with its dense output and PI
  step-size control (`OdeSolver::Rk45`), and a Rosenbrock method with the coefficients of Hairer and Wanner's Rodas4
  and its embedded order 3 error estimate, using the exact Jacobian and ∂f/∂t (`OdeSolver::Stiff`). Solutions are
  reported at output times that must be finite, strictly increasing and after the initial time. Tolerances
  default to Stan's (rtol and atol 10⁻⁶, at most 10⁶ steps). A non-finite initial state, the step limit and a step
  size underflow fail with `OdeError::SolverFailed`; a step with non-finite stages is retried with a smaller step.
  `solve_sensitivities` integrates forward sensitivities dS/dt = J S + ∂f/∂p with the state, with error control
  over both; the stiff solver uses the combined system's block triangular Jacobian, whose off-diagonal block comes
  from `SensitivitySystem::sensitivity_jacobian_product` (a central difference by default). keanu-tensor gained
  `LuFactorization`, an LU factorisation for repeated solves.
- keanu-graph: **ODE op** (`Op::Ode(Arc<OdeBody>)`, `GraphBuilder::ode`), the solution of dy/dt = f(t, y, θ) at
  given output times, shape `[T, n]`, where f is a nested deterministic body graph built like a scan's from
  placeholders for the time, the state and each parameter. Its parents are the initial state, the output times,
  the initial time and the parameters, all differentiable to first order by forward sensitivities whose right-hand
  side J S + ∂f/∂θ comes from one batched forward sweep through the body; the times' partials are f(tᵢ, y(tᵢ)).
  Reverse mode integrates one sensitivity column per element of each parent that is not a constant. The value is
  solved without sensitivities and the gradient solves again with them. Body nodes that depend only on the
  parameters are computed once per solve. There is no second-order rule (`NoSecondOrderRule`, so no Laplace
  approximation). Output times that are not strictly increasing after the initial time are an invalid node. A solve
  that fails is the new `GraphError::SolverFailed`, which `GraphError::is_impossible_point` reports and which is
  treated as an impossible point: `Model::log_prob_after`, `log_prob_and_gradient*` and
  `probe_for_non_zero_probability`, the new `Parameterisation::try_to_state` and `Parameterisation::log_prob`, and
  `Parameterisation::log_prob_and_gradient` give −∞; NUTS counts the trajectory as divergent, Metropolis–Hastings,
  Gibbs and discrete Metropolis reject the value, the optimizers reject the step, and ADVI reports a non-finite
  ELBO as for any impossible draw. The save schema gained `ode` and `OdeSolver`, `LoadOptions::max_ode_steps`
  (default 10⁷) caps a loaded step limit, and the DOT export draws ODE bodies as clusters.
- keanu: **ODE and multi-state examples** and a guide chapter, "Differential equations". `sir` fits the SIR
  epidemic model with a negative binomial likelihood to the 1978 boarding school influenza outbreak (BMJ), with the
  priors of the Stan case study by Grinsztajn et al. (2021), whose posterior it reproduces (β ≈ 1.74, γ ≈ 0.54,
  R₀ ≈ 3.2); its tests also recover β and γ from simulated data within their 90% intervals and check the log
  probability's gradient against finite differences to 10⁻⁶ relative. `pharmacokinetics` fits a one-compartment
  oral-dose model as an ODE and in closed form and checks that they agree. `multi_state` fits an illness–death
  continuous-time Markov chain to simulated interval-censored panel data through a batched matrix exponential.
- keanu-graph: **Hilbert space approximate GPs** (`gp::Hsgp`, `HsgpPrior`, `HsgpDomain`, `gp::hsgp`), following
  Riutort-Mayol et al. (2023): an exponentiated quadratic or Matérn 1/2, 3/2 or 5/2 kernel's GP as Φ (√S ⊙ β) with
  the Laplacian's eigenfunctions on a box that extends the inputs' range by a factor c, m basis functions per input
  dimension (the full tensor grid for multi-dimensional inputs, with a scalar or per-dimension length scale), the
  kernel's spectral density S and weights β ~ N(0, 1). `prior(x)` returns the function at the inputs and the weights,
  `predict` the same function at new inputs inside the box, and `covariance` the covariance the approximation implies.
  It is built from constants, element-wise ops and a matrix product, so gradients and Hessian-vector products come
  from the graph's autodiff. Other kernels and sums or products of kernels are rejected by `Hsgp::new`. The GP
  regression example gained an HSGP fit and the guide's GP chapter a section. keanu-tensor's small matrix products
  with one column (matrix-vector products) are about four times faster, with unchanged results.
- keanu-distributions, keanu-graph, keanu-io: **sum-to-zero vectors and the zero-sum normal**.
  `Support::SumToZero` and `transform::SumToZero` follow Stan's `sum_to_zero_vector` (2.36 and later): the N − 1
  unconstrained coordinates of each vector over the last axis are its coordinates in an orthonormal basis of the
  zero-sum subspace, so the map is linear and isometric, with log-Jacobian 0 and zero second derivatives.
  `ZeroSumNormal` (`DistributionKind::ZeroSumNormal`, `GraphBuilder::zero_sum_normal` and
  `zero_sum_normal_with_shape`) is PyMC's: N(0, σ² I) restricted to the subspace, with log density
  −Σx²/(2σ²) − (N − 1)(ln σ + ½ ln 2π), σ a scalar or one per vector of a batch, draws of N(0, σ² I) minus their
  mean, gradients and Hessian-vector products. A value that doesn't sum to zero (within 10⁻⁸ relative,
  `keanu_distributions::multivariate::ZERO_SUM_TOLERANCE`) has log probability −∞. The save schema gained
  `DISTRIBUTION_KIND_ZERO_SUM_NORMAL`, and the guide's "Building models" chapter a section on sum-to-zero group
  effects.
- keanu-distributions, keanu-graph, keanu-io: **count distributions and logit parameterisations**.
  `BetaBinomial(n, α, β)` (`GraphBuilder::beta_binomial`), with a log CDF and survival function summed term by term
  (and `BetaBinomial::log_tail`, `BetaBinomialTail`, their first and second shape derivatives), so it can be
  truncated and censored. `ZeroInflated(kind)` and `Hurdle(kind)` (`zero_inflated`, `hurdle` and their
  `_with_shape` forms) are kinds built around another one, with parameters the inner kind's followed by `psi`:
  zero-inflated counts have P(0) = ψ + (1 − ψ) p(0); hurdle counts P(0) = ψ and P(k) = (1 − ψ) p(k) / (1 − p(0)); a
  hurdle around a positive continuous kind (half Gaussian, half Cauchy, chi-squared, exponential, gamma, inverse
  gamma, log-normal, Weibull or beta) gives a value of exactly 0 probability ψ. `BernoulliLogit`, `BinomialLogit`,
  `PoissonLog` and `CategoricalLogit` (`bernoulli_logit`, `binomial_logit`, `poisson_log`, `categorical_logit`) take
  log odds, unnormalised log probabilities or the log mean, with log probabilities computed as −softplus(∓η),
  ηₓ − logsumexp(η) and kη − e^η, finite at logits of ±700. Each kind samples, and has exact gradients and
  Hessian-vector products; those with one value per element are mixture components, `log_density` kinds and
  children of marginalised latents, and those with finitely many values (`DistributionKind::finite_support`) are
  Gibbs-sampled and can be marginalised. The truncated and censored second-order rules no longer evaluate the base's
  CDF rule at an infinite bound's placeholder point (which only the beta-binomial's tail rule made nonzero). The
  save schema gained `DISTRIBUTION_KIND_BETA_BINOMIAL`, `_BERNOULLI_LOGIT`, `_BINOMIAL_LOGIT`, `_POISSON_LOG`,
  `_CATEGORICAL_LOGIT`, `_ZERO_INFLATED` and `_HURDLE`, and the guide's "Building models" chapter a section on them.
- keanu: **occupancy example** (`occupancy`), a site-occupancy model (MacKenzie et al., 2002) on 120 simulated
  sites with 4 visits each, written three ways: Bernoulli occupancy latents under the default compound sampler
  (Gibbs steps for the occupancies, NUTS for ψ and p), the occupancies marginalised with `marginalize` and recovered
  with `recover_marginals`, and each site's detection count as a zero-inflated binomial. Its test checks that each
  form covers the true ψ and p with its 90% interval and that the three agree on ψ, p and the posterior number of
  occupied sites; the guide's examples chapter describes it.
- keanu-distributions and keanu-algorithms: **smoothing, decoding and forecasting** of the state-space models.
  `LinearGaussianStateSpace` gained `smooth` (the Rauch–Tung–Striebel smoother after the same Joseph-form filter as
  the log density, with covariances kept positive semidefinite and a pseudo-inverse for a singular predicted
  covariance; `KalmanSmoother`), `sample_posterior_states` (forward filtering, backward sampling), `forecast` and
  `forecast_with` (each future step's state and observation means and covariances, `StateSpaceForecast`, for a
  time-invariant model or per-step `FutureStep` matrices), and `sample_forecast` and `sample_forecast_with` (one
  joint future path, `StateSpacePath`). `HiddenMarkovChain` gained `viterbi` (the most probable path and its log
  joint probability), `posterior_marginals` and `sample_posterior_states`. Observations of probability zero are an
  error for all of them. The new `keanu_algorithms::timeseries` module runs them at each posterior draw of a node:
  `StateSpacePosterior` (`smooth`, `mean_states`, `sample_states`, `forecast`, `forecast_with`, `sample_forecast`,
  `sample_forecast_with`) and `HiddenMarkovPosterior` (`viterbi`, `posterior_marginals`, `mean_marginals`,
  `sample_states`), from `NetworkSamples` or `ChainSamples`, with the node's parameters evaluated from each draw.
  keanu-graph gained `DistributionKind::hidden_markov_chain`. The HMM example decodes its regimes, and the guide's
  sequences chapter has a section on smoothing, decoding and forecasting.
- keanu-distributions, keanu-graph and keanu-io: **time-series distributions** (the new
  `keanu_distributions::processes` module), each over the last axis of a value [..., T] with a batch of series in
  the leading axes and parameters broadcasting to the batch shape, as PyMC's distributions of the same names:
  `GaussianRandomWalk(mu, sigma, init_mu, init_sigma)` (x₀ ~ N(init_mu, init_sigma), then steps of N(mu, sigma); T
  counts the initial value, so PyMC's `steps` is T − 1), `Autoregressive(constant, rho, sigma, init_mu, init_sigma)`
  (AR(p) with `rho` [..., p] the coefficients of lags 1 to p, conditioned on its first p values, which are
  N(init_mu, init_sigma); no stationarity is imposed) and `Garch11(omega, alpha, beta, initial_vol)` (r_t ~ N(0, σ_t)
  with σ_t² = ω + α r_{t−1}² + β σ_{t−1}²). Non-positive scales give a NaN log density; GARCH parameters outside
  ω > 0, α ≥ 0, β ≥ 0, α + β < 1 and initial_vol > 0 give −∞ and are an error to sample. Gradients are written out
  by hand and Hessian-vector products are that code on dual numbers, both exact. The graph has the kinds
  `DistributionKind::GaussianRandomWalk`, `Autoregressive` and `Garch11` (real support) with builder methods
  `gaussian_random_walk`, `autoregressive` and `garch11` and their `_with_shape` forms; the save schema gained
  `DISTRIBUTION_KIND_GAUSSIAN_RANDOM_WALK`, `_AUTOREGRESSIVE` and `_GARCH11`, and the guide's sequences chapter a
  section on them.
- keanu-graph, keanu-algorithms and keanu: **structural time series**. `keanu_graph::sts::StructuralModel` builds a
  model of an observed series from `Component`s (`local_level`, `local_linear_trend`, `seasonal` in the
  dummy-variable form whose S consecutive effects sum to zero apart from drift, `trigonometric_seasonal` with any
  period and 1 to ⌊period / 2⌋ harmonics, `autoregressive` of order p, and `regression` on known covariates) into
  one linear Gaussian state space node, with the components' states stacked and their transition and noise matrices
  block-diagonal, so the Kalman filter integrates the states out. Scales and coefficients are graph variables given
  with `with_scale`, `with_slope_scale`, `with_coefficients` and `with_observation_noise_scale`, or latents the
  builder creates with priors scaled by the series' standard deviation s (LogNormal(ln(0.05 s), 2) for component
  scales, LogNormal(ln(0.1 s), 2) for the noise, N(0, 1) for AR coefficients, N(0, 10 s / max |x_j|) for regression
  weights); the initial state is fixed at N(y₀, s²) for the level and N(0, s²) otherwise. A regression is an offset
  of the observations, so its likelihood is a factor of the state-space log density of the series minus X β:
  `Op::LogDensity` and `GraphBuilder::log_density` now accept the `LinearGaussianStateSpace` kind, as a scalar
  node with the distribution's gradient and Hessian-vector products. `build` returns a `StructuralTimeSeries`
  (each component's `ComponentLayout`, the parameters' nodes and the likelihood). `keanu_algorithms::sts::StructuralPosterior`
  evaluates it at each draw of `NetworkSamples` or `ChainSamples` and gives `decompose` (each component's
  smoothed contribution and their sum, `Decomposition`) and `forecast` (the series and each component h steps
  ahead, with future covariates for the regressions, `Forecast`), whose means are averaged over the draws and whose
  central intervals are quantiles of the mixture of each draw's Gaussian, and `sample_forecast` (one joint path per
  draw). `LinearGaussianStateSpace` gained `log_prob_and_d_log_prob`, one filter pass for both, which the graph now
  uses for the node's log probability and gradient together. The new `co2_forecast` example fits a local linear
  trend and a two-harmonic 12-month cycle to simulated monthly CO₂ with NUTS, decomposes it and forecasts 24 held-out
  months, and the guide's sequences chapter has a section on structural time series.
- keanu-algorithms: **dense mass matrix for NUTS** (`mcmc::nuts::Metric`, `QuadraticPotential::metric` and
  `with_metric`). `Metric::Dense` adapts the whole inverse mass matrix as Stan's `dense_e` does: in each slow window
  the draws' sample covariance Σ, regularised to (n / (n + 5)) · Σ + 10⁻³ · 5 / (n + 5) · I, becomes M⁻¹; momentum is
  drawn as L⁻ᵀ u with M⁻¹ = L Lᵀ, and the velocity M⁻¹ p enters the kinetic energy and the no-U-turn criterion. It
  works in block mode and so in compound samplers. `NutsStatistics::dense_inverse_metric` and
  `NutsSampler::dense_inverse_metric` hold the adapted [n, n] matrix (`inverse_metric` stays its diagonal), and
  `CmdStanCsv` writes `metric = dense_e` and the matrix one row per comment line, as CmdStan does. On a 2-D Gaussian
  with correlation 0.99 it reaches 11 to 34 times the diagonal metric's bulk ESS per gradient evaluation. The
  diagonal metric is the default and its trajectories are unchanged. The HSGP NUTS test now uses the dense metric
  (about 13 leapfrog steps per tree instead of about 100; 7 s instead of 18 s without the `parallel` feature), and
  allows for the HSGP's own deviation from the exact GP's posterior mean (up to 0.0051, measured with long runs of
  both metrics, which agree).
- keanu-algorithms: **Pathfinder** (`vi::Pathfinder`, `PathfinderResult`, `PathfinderPath`; `Pathfinder` and
  `PathfinderResult` also in the crate root and the `keanu` prelude), quasi-Newton variational inference following
  Zhang, Carpenter, Gelman and Vehtari (2022) with the paper's defaults. `single_path` runs L-BFGS (history 6, at
  most 1000 iterations) on the unconstrained log density from the state's values and, at each iterate, a Gaussian
  whose covariance is the L-BFGS inverse Hessian (a diagonal updated by Gilbert and Lemaréchal's rule plus the
  compact low-rank BFGS term from the pairs with positive curvature) and whose mean is the Newton step; draws use
  a thin QR factorisation of the low-rank part, or the Cholesky factor when it is not smaller. The approximation
  with the best 5-draw ELBO gives 1000 draws. `multi_path` runs 4 paths from uniform(−2, 2) starting points,
  skips failed ones, and resamples 1000 of the pooled draws with replacement by their PSIS weights, reporting k̂
  (`pareto_k`, `is_reliable`; −∞ when the importance ratios are equal to within 10⁻⁸, as for an exact
  approximation, where a fit to their rounding errors would be meaningless). Results hold the unconstrained draws with their log densities and the
  approximations' log densities, each path's mean, covariance, ELBO trace and chosen iteration, `sample` (the
  draws as `NetworkSamples`) and `initial_states` (starting states for chains from distinct draws), which the new
  `mcmc::chains::run_chains_from` and `sample_chains_from` run one chain from each. The radon example gained
  `model`, `sample` and `run_from_pathfinder`, whose chains converge with a 150-step warm-up, and the guide's VI
  chapter a Pathfinder section.
- keanu-graph, keanu-algorithms: **minibatch ADVI** (`Advi::with_minibatch` and the `minibatch` field;
  `vi::Minibatch`, `MinibatchGroup` and `Batches`; `Minibatch` and `MinibatchGroup` also in the crate root and the
  `keanu` prelude) and **log probability weights** (`Parameterisation::set_weight`, `with_weights` and `weight`,
  which multiply a latent's, observed node's or factor's term, its gradient and its Hessian-vector products by a
  finite non-negative weight; an impossible term stays impossible at any weight). A `MinibatchGroup` holds
  observed nodes and covariates (constants or observed nodes) with their full data of N rows along axis 0, a batch
  size B, and factors to weigh; each iteration sets every node of a group to the same B rows, drawn without
  replacement within an epoch and reshuffled each epoch (the N mod B rows left over are skipped), recomputes their
  descendants, and weighs the group's probabilistic nodes and factors by N / B. Batches are validated against the
  model: leading dimension B in the graph, N rows of the same trailing shape and type in the data, B ≤ N, no node
  twice, and every latent, observation or factor downstream of the data in the group (local latents are rejected).
  Each ELBO estimate uses a fresh batch and convergence is judged on the running mean of the last estimates; the
  step size adaptation runs on minibatches but compares ELBOs on one fixed batch. Without a minibatch, ADVI is
  unchanged, except that `Advi` is no longer `Copy`. On a 100 000-row regression, batches of 1000 rows give the
  exact posterior's coefficients to within 0.05 and their standard deviations to within a factor of 2, in about
  0.2 s. The guide's VI chapter has a Minibatches section.

- keanu-algorithms: **tempered sequential Monte Carlo** (`smc::Smc`, `SmcKernel`, `SmcResult`, `SmcRun`,
  `TemperedLikelihood` and `ModelLikelihood`; `Smc`, `SmcKernel` and `SmcResult` also in the crate root and the
  `keanu` prelude), following PyMC's `sample_smc`. Particles (2000 by default) are drawn from the prior and tempered
  from β = 0 to 1 by stages whose β is chosen by bisection so the normalised effective sample size of the
  incremental weights is 0.5, then resampled systematically and mutated at that β: the continuous latents in
  unconstrained space by independent Metropolis–Hastings with a multivariate normal proposal fitted to the weighted
  particles (the default) or by a random walk with their covariance and acceptance-tuned per-particle scales,
  taking steps until the mean correlation of the particles' coordinates before and after is at most 0.01 (at most
  25 steps); and the discrete latents by `Gibbs` or `DiscreteMetropolis` sweeps of the tempered conditionals. Each
  run estimates the log marginal likelihood as the sum of the stages' log mean incremental weights. `sample` runs 4
  independent runs (in parallel with the `parallel` feature) whose particles form `ChainSamples`, with each run's
  log marginal likelihood, temperatures, acceptance rates and steps per stage. Impossible particles (−∞, or a
  failed evaluation such as an ODE solve) have zero weight and moves to them are rejected; prior draws where the
  model can't be evaluated are redrawn. Models whose support bounds move with a latent are mutated in constrained
  space. The tempered likelihood is a trait, `TemperedLikelihood`, whose default is the model's own log likelihood
  (observed nodes and factors). Tests check the log evidence of a conjugate Gaussian to within 0.1, a Bayes factor,
  a symmetric bimodal posterior's modes in proportion, and mixed discrete models against exact posteriors, and the
  guide's sampling chapter has a section on SMC.
- keanu-graph, keanu-io, keanu-algorithms: **simulator nodes and SMC-ABC**. `GraphBuilder::simulator` makes a
  probabilistic node (`DistributionKind::Simulator`, `op::Simulator`, `op::SimulatorFn`) whose value of a given
  shape is drawn by a thread-safe user function of its parameters' values and a `KeanuRng`, like PyMC's
  `pm.Simulator`; it can be sampled from its prior (forward sampling, prior and posterior predictive draws) and
  observed, but has no likelihood: its log probability is the new `GraphError::NoLikelihood` naming the node, and a
  function that fails or returns the wrong type or shape is the new `GraphError::Simulator`. `Model::simulators`
  lists a model's simulator nodes and `Model::check_likelihood` fails if there are any; every inference algorithm
  but ABC (NUTS, block NUTS, Metropolis–Hastings, Gibbs, `DiscreteMetropolis`, compound and automatic samplers,
  simulated annealing, the gradient and BOBYQA optimizers, Laplace, ADVI, Pathfinder, the particle filter, SMC on
  the model's likelihood, and the pointwise log likelihood) rejects such a model with that error before it starts,
  and saving one fails with `IoError::NotSaveable` ("a simulator"). The new `abc` module (`Abc`, `AbcKernel`,
  `SummaryStatistics`, `SummaryFn`; the first three also in the crate root and the `keanu` prelude) is a
  `TemperedLikelihood` that simulates each observed simulator node at a particle and scores its summary statistics
  (the data, sorted data, mean, median, variance, or a user function) against the observation's with a Gaussian
  (−½‖Δ/ε‖², the default) or Laplace (−‖Δ/ε‖₁) kernel, with one ε or one per statistic, plus the exact log
  likelihood of the other observed nodes and factors; `Smc::sample_abc` runs it. As in PyMC, ε is fixed and SMC's
  tempering raises β to 1, each particle keeps the kernel value it was accepted with (a pseudo-marginal mutation),
  and the reported evidence is of the unnormalised kernel. `TemperedLikelihood` gained `check`, which by default
  rejects models with simulator nodes. ABC needs continuous latents only (discrete latents are mutated over their
  Markov blankets, where a simulator has no terms). On a Gaussian simulator with the mean as statistic, the ABC
  posterior's mean and variance and the ABC evidence match their closed forms for ε of 1, 0.3 and 0.05, and the
  posterior approaches the exact one as ε shrinks.
- keanu: **SMC and ABC examples and a simulation-based inference chapter**. `smc_model_comparison` compares a linear
  and a quadratic regression by their SMC log evidence, which its test checks against the closed form (to 0.15, and
  the log Bayes factor of about 8 to 0.2) and against PSIS-LOO on the particles, which also prefers the quadratic
  model. `abc_stochastic_sir` fits a stochastic SIR epidemic in 200 people, simulated by Gillespie's algorithm as a
  simulator node, by SMC-ABC on four summary statistics (final size, peak prevalence, peak day, person-days of
  infection) with one ε each; its test checks that the 90% intervals of β and γ cover the rates the outbreak was
  simulated with and are informative (both tests take about 6 s together without the `parallel` feature). The
  guide's new chapter "Simulation-based inference (SMC and ABC)" takes over the SMC section of the sampling
  chapter (which now links to it) and adds model comparison by evidence, simulator nodes, ABC and the stochastic
  epidemic; the examples page lists both examples.
- Code review fixes for the above:
  - keanu-graph: the gradient of a categorical with vector logits (`categorical_logit` with logits of shape [n])
    summed each logit's gradient as if it were batched, which overflowed in debug builds and failed with
    `InvalidReshape` in release builds; it is now summed to each logit's shape as the categorical's gradient is.
    Its Hessian-vector products were already correct and are now checked against finite differences.
  - keanu-graph: zero-inflated and hurdle distributions over a truncated or censored count have exact
    Hessian-vector products, as documented, taking the inner kind's per-element Hessians from its own second-order
    rule (they failed with `NoSecondOrderRule` before).
  - keanu-graph: a hurdle's positive counts are drawn by inverting the inner kind's mass above zero where the inner
    kind puts at least half its mass at zero, and by rejection elsewhere (or where the inversion takes more than
    1000 steps), so hurdles over small rates (a Poisson with μ = 0.002, say) can be sampled; they failed after
    1000 rejected draws before. Sampling fails only where the inner kind has no mass above zero and ψ < 1.

### Code review fixes

Fixes from the 2026-11-04 whole-codebase review. Where a fix changes a result recorded from Java, the golden
vector was edited by hand and the entry says so.

- keanu-graph: using a handle from one `GraphBuilder` with another now fails `build()` with the new
  `GraphError::WrongBuilder`; it used to silently wire the other builder's node with the same id (or panic).
- keanu-graph: auto diff now passes gradients through `set_with_mask` (the operand's where the mask isn't 1, the set
  value's where it is), `replace_nan` (where the input wasn't NaN) and `multiplexer` (the selected option's); they
  were treated as non-differentiable, silently dropping those paths.
- keanu-graph: a failing `add_distribution` (or other untyped builder call) no longer leaves a placeholder node
  behind, and returns its error even when an earlier deferred error is pending, which still fails `build()`.
- keanu-graph: `add_node` checks each op's number of parents and returns an error instead of panicking (e.g.
  `Op::Proxy` or `Op::Not` with no parents) or silently accepting extra parents.
- keanu-graph: building a graph's child lists is linear in fan-out (it was quadratic).
- keanu-graph: auto diff of `product`/`product_axes`, `cumsum`, `cumprod` and `tensor_multiply` uses closed-form
  forward and reverse rules (exclusive products, reverse scans, contraction with the other operand) instead of a
  dense Jacobian built by evaluating the op twice per input element; still exact where elements are zero. Reverse
  gradients of a 2000-element cumsum, product or cumprod drop from 12-40 ms to ~20-30 µs, and of a 30×30
  `tensor_multiply` from 31 ms to 90 µs.
- `Multinomial::log_prob` validates x: a count vector with a negative count or counts not summing to n has log probability -infinity, and an x whose last dimension is not the number of categories is an error.
- `Logistic` log probability and gradients use a stable softplus and sigmoid, so they no longer overflow to -infinity / NaN far in the tails.
- `MultivariateGaussian::d_log_prob` reduces each gradient to its parameter's own shape, summing size-1 broadcast dimensions as well as extra leading ones.
- `MultivariateGaussian::log_prob` no longer forms the inverse covariance: the Mahalanobis term is ‖z‖² with L z = x - mu (one triangular solve). `d_log_prob` gets Σ⁻¹ (x - mu) by two triangular solves and the covariance gradient as ½ (a aᵀ - Σ⁻¹), dropping two n³ matrix products (Σ⁻¹ itself is still formed: the gradient contains it). Results agree with the explicit-inverse form to ~1e-16; at n = 500, log_prob is ~20x and d_log_prob ~5x faster.
- keanu-tensor: new `DoubleTensor::solve_triangular(b, lower, transpose)` solves op(T) X = B for triangular T, broadcasting batch dimensions like `matmul` and accepting a vector right-hand side. It reads only the given triangle, returns `SingularMatrix` for a zero diagonal and `IncompatibleShapes` for mismatched sizes, and uses faer above 64 × 64.
- keanu-tensor: `cholesky_inverse` above 64 × 64 uses faer (a triangular solve against the identity, then L⁻ᵀ L⁻¹) instead of direct loops; the result is still exactly symmetric.
- Samplers whose requested shape is smaller than the parameters' broadcast shape (Gaussian, HalfGaussian, LogNormal, Cauchy, HalfCauchy, Beta, Exponential, Logistic, Pareto, StudentT, Triangular, Uniform, Bernoulli, Geometric, UniformInt, Categorical) draw one value per broadcast element instead of reusing draws. Sampling at the full broadcast shape uses the same random stream as before.
- `Kde` log probability and gradient use log-sum-exp, so they stay finite far from the samples. `Kde::new` rejects a bandwidth that is not positive and finite, and `Kde::with_scotts_bandwidth` returns an error for samples with no spread.
- `Dirichlet::log_prob` takes (c - 1) ln x as 0 where c = 1 (no NaN at x = 0) and is -infinity for negative x.
- `Poisson::log_prob` at mu = 0, k = 0 is 0 rather than NaN.
- The `Binomial` gradient takes 0 / p as 0, so it is finite at p = 0, k = 0 and p = 1, k = n. Golden vector `Binomial/p=0.0,n=4` dLogProb/P at k = 0 edited by hand from NaN to -4.
- `Pareto` and `Geometric` log probabilities treat invalid parameters per element (and return the broadcast shape) instead of making every element -infinity.
- `Pareto` support includes x = location. Golden vectors `Pareto/location=1.0,scale=2.0` (x = 1) and `Pareto/location=0.5,scale=1.0` (x = 0.5) logProb edited by hand from -infinity to ln 2.
- Multinomial and Categorical sampling never return a category with zero probability when rounding leaves the cumulative probability below the uniform draw; they return the last category with non-zero probability.
- keanu-distributions (breaking): distributions borrow or own their parameters. Each struct takes a lifetime
  (`Gaussian<'a>`), its parameter fields are `Param<'a>` (`Cow<'a, Tensor>`), and constructors take
  `impl IntoParam`, so `Gaussian::new(mu, sigma)` still works and `Gaussian::new(&mu, &sigma)` copies nothing;
  `Categorical::new` takes any iterator of owned or borrowed tensors. keanu-graph no longer copies every parameter
  tensor on each log prob and gradient evaluation (log prob plus gradient of a two-node, 100,000-element Gaussian
  model: 2.35 ms to 2.2 ms).
- keanu-graph: `Model::log_prob_and_gradient` returns the joint log probability and (unless it is -∞) its gradient
  in one pass that evaluates each node's distribution once, bit-for-bit equal to `log_prob` and
  `log_prob_gradient` (1.7 ms on the model above); Gaussian nodes use the new one-pass
  `Gaussian::log_prob_and_d_log_prob`.
- `Binomial` sampling with more than 50 trials uses inversion (small mean) or Hörmann's BTRS, and `Poisson`
  sampling with mu >= 10 uses Hörmann's PTRS, both O(1) expected per draw instead of O(n) / O(mu) (Binomial
  n = 1e8: 2.7 s to ~6 µs per draw). Smaller parameters draw exactly the same values as before.
- Gradients of Cauchy, Gamma, InverseGamma, Laplace, Logistic, LogNormal, Pareto, Beta and Exponential, and
  Triangular's (which evaluated every piece four times), are computed in one broadcast pass; Categorical log prob
  in one pass instead of one per category; Beta sampling in one pass. Results are unchanged, except that
  `Pareto`'s x gradient is now shaped like the broadcast of x and every parameter, as the module documents.
- Sampling rejects invalid parameters with `InvalidParameter` for every distribution: Gaussian, HalfGaussian and
  LogNormal sigma, Exponential rate, Logistic s, Pareto location and scale must be positive, Uniform, UniformInt,
  Beta and Triangular need x_max > x_min, Bernoulli and Binomial p in [0, 1], Binomial n >= 0, Geometric p in
  (0, 1), ChiSquared k > 0, Poisson mu finite. They used to return garbage (negative exponential draws, NaN).
- `Beta`, `Uniform` and `Triangular` log probabilities are -infinity everywhere for an empty or NaN range
  (x_max <= x_min, or a Triangular mode outside it) instead of NaN or +infinity; Categorical log prob and sampling
  reject negative or NaN probabilities instead of returning NaN.
- keanu-io: a `SliceVertex` whose `dimension` is not a dimension of its input is an `InvalidParameter` error
  instead of a capacity overflow panic.
- keanu-io: shapes whose element count overflows (tensor shapes, vertex shapes, `BroadcastVertex` `toShape`) are
  `Decode` errors instead of panicking (debug) or wrapping (release).
- keanu-io: an unlabelled proxy is saved with an empty label instead of an invented `proxy_<n>`, which could collide
  with another node's label (failing to load) and persisted after loading; invented labels in older files are
  ignored (proxies are labelled from the vertex label, which Java always wrote equal to the parameter).
- keanu-io: saving and loading repeatedly no longer grows the graph: loading folds back the zero location constant
  of half Gaussian / half Cauchy vertices (including Java's) and the subgraph saved for a Rust cholesky inverse.
- keanu-io: duplicate or missing vertex ids are `Decode` errors instead of being accepted silently.
- keanu-io: DOT node labels escape `"`, `\` and line breaks.
- keanu-io: JSON serialisation failures in `save_json` are the new `IoError::Encode`, not `Decode`.
- keanu-io: generic (object) stored values are the new `IoError::UnsupportedValue`, not `UnsupportedVertex`.
- keanu-io: values stored by label are matched with a label map built once, instead of a scan of every node each.
- keanu-io: `prost` is only a dependency with the `protobuf` or `json` feature; keanu-parity drops its unused
  `serde` dependency and inherits the workspace package fields.
- keanu-io: integer parameters loaded as axes, offsets or reshape dimensions are converted with one checked
  conversion (a `Decode` error where they don't fit an `isize`), and a `TriLowerVertex` `k` of `i64::MIN` is a
  `Decode` error instead of overflowing when negated (a panic in debug builds).
- keanu-io: saving borrows node labels, shapes, parents and constant / KDE data instead of copying them, and maps
  nodes to saved vertices with a vector instead of a hash map; DOT export names a distribution without formatting
  its data (it formatted a KDE's whole sample array to get the name "Kde").
- keanu-tensor: slice strings allow spaces inside items (`"1 : 3"`) and a trailing comma (`"1,"`), as numpy does.
- keanu-tensor: the owned scalar operators (`tensor + 1.0`, `-tensor`, ...) update the tensor's buffer in place
  instead of allocating a new one (four chained ops on 100k elements: 318 → 123 µs); integer `div`, `rem` and
  `pow` take the same-shape fast path and stop calling the element function after the first error (100k-element
  division: 433 → 129 µs; a broadcast remainder 437 → 89 µs); `tensor_multiply` no longer copies its permuted
  operands again to reshape them (20×30×40 · 40×30×10: 49 → 40 µs), and small matrix products zero their output
  once.
- `keanu-algorithms`: the particle filter treats a NaN log probability as impossible (−∞) when weighting, ranking
  and resampling; one NaN particle used to make every resampling draw pick the last particle.
- `keanu-algorithms`: `Samples::variance` uses a two-pass computation (still with an n − 1 denominator) instead of
  E[x²] − E[x]², which lost all precision for a large mean (samples of N(1e9, 1) had variance −896).
- `keanu-algorithms`: NUTS rejects a potential whose `initial_weight` or `initial_variance` is not finite and
  greater than 0 (`InvalidSetting`); a zero gave a zero mass matrix entry and a chain that never moved.
- `keanu-algorithms`: with `calculate_log_prob`, `ForwardSampler` recomputes the deterministic nodes downstream of
  the resampled latents before computing the log probability, which could otherwise use stale values. The docs of
  `ForwardSampler::new` no longer claim it records log probabilities by default.
- `keanu-algorithms`: `NetworkSamples::record` checks every value's type before recording any, so a mismatch no
  longer leaves the columns with different lengths.
- `keanu-algorithms`: simulated annealing rejects a schedule with zero iterations or a non-finite or non-positive
  start or end temperature (`InvalidSetting`, from the new `AnnealingSchedule::validate`).
- `keanu-algorithms`: conjugate gradient stops, without converging and at the best point found, when the fitness
  is impossible or the gradient NaN or infinite; it used to loop forever (with unlimited evaluations) and leave the
  latents at NaN. The initial-condition check now reports a NaN or infinite starting gradient as the new
  `AlgorithmError::NonFiniteGradient` (it used to pass, as `f64::max` skips NaN).
- `keanu-algorithms`: conjugate gradient keeps to `max_evaluations` inside its line search too
  (`with_max_evaluations(5)` used 19 evaluations).
- `keanu-algorithms`: BOBYQA no longer reports `converged: true` for an optimum with impossible (non-finite)
  fitness.
- `keanu-algorithms`: BOBYQA bound cleaning matches PRIMA: only a NaN or very low lower bound (or a NaN or very
  high upper bound) means "no bound", so a lower bound of +∞ or an upper bound of −∞ is now an error like any other
  lower bound above its upper bound. `NonGradientOptimizer` rejects NaN bounds (`InvalidArgument`).
- `keanu-algorithms`: BOBYQA no longer overflows checking a huge `interpolation_points`; it falls back to the
  default, as for any other out-of-range value.
- `keanu-algorithms`: BOBYQA stops as soon as the objective fails instead of iterating on to the final trust region
  radius (new `prima::minimize_with_termination` and `ExitStatus::CallbackTerminate`, PRIMA's callback
  termination), computes each step's Lagrange values once instead of up to seven times (bit-identical results), and
  documents the settings it adjusts silently (`max_evaluations` below n + 3, out-of-range `interpolation_points`).
- `keanu-algorithms`: the optimizers no longer evaluate points twice: conjugate gradient and Adam start from the
  initial-condition check's fitness and gradient, conjugate gradient takes the fitness at the line search's step
  from the line search, and neither it nor BOBYQA re-evaluates the final point (BOBYQA also reuses the check's
  value for its first point). Conjugate gradient's line search also stops once no step can change the fitness
  (a flat first step, steps too small to move the point, or a flat Brent bracket) instead of spending 40–60
  evaluations near the optimum. Reported `evaluations` drop: a one-dimensional Gaussian MAP 78 → 15, the golden
  models' conjugate gradient runs by 3–8× (e.g. thermometers MAP 80 → 24), BOBYQA by 2.
- `keanu-algorithms` (breaking): `MetropolisHastings` has one set of builder methods, `proposal`, `selector` and
  `rejection` (now `#[must_use]`), plus `shared_proposal` (was `with_shared_proposal`); `with_proposal`,
  `with_selector` and `with_rejection` are removed. A Metropolis–Hastings step whose proposal fails to evaluate
  (e.g. a cascade error) rolls the state back before returning the error instead of leaving the proposal applied.
  Rollback snapshots reuse the changeable nodes worked out for each proposed node set, and `GaussianProposal`
  (symmetric) returns 0 for both proposal densities instead of building two Gaussians per step (same random
  numbers and acceptances). A 20-latent chain with full proposals runs 1.6–1.9× faster.
- `keanu-algorithms`: the particle filter adds each increment's new latents' and observation's log probabilities to
  a particle's total instead of re-cascading the particle and recomputing every log probability so far (quadratic
  in the length of a time series), restores only the latents an increment depends on before sampling it (a
  200-step random walk with 100 particles drops from 2.8 s to 0.15 s), and
  `latent_increments` keeps a count per observation instead of every node's set of latent ancestors (quadratic
  memory on a chain). Same random numbers and ordering; particle log probabilities may differ in the last bits
  (summation order).
- `keanu-algorithms`: `Samples::mode` sorts instead of comparing each sample with every distinct value so far
  (20 000 samples with 15 000 distinct values: 450 ms → 2 ms), with the same answer and tie-breaking (the
  earliest-appearing of the most frequent values). It now requires `T: PartialOrd` (true of f64, i64 and bool).
- keanu-tensor: `digamma`'s asymptotic series had the wrong sign on its 1/(12x²) term (inherited from Apache Commons), so results were off by up to ~3e-9 (ψ(1) = -0.5772156677920679). Golden vectors edited by hand to mpmath values: `unary/digamma` and `unary/digamma/scalar` (tensors.json), `unary/digamma` and the `unary/logGamma` partials (vertices.json), the catalog model's `digamma` node (models.json), and the ψ-dependent gradients of `Beta/alpha=0.5,beta=0.5`, `Dirichlet/concentration=[0.5, 2.0, 3.0]` (both cases), the three `Gamma` cases and the two `InverseGamma` cases (distributions.json). The Rust-written `keanu-io` fixtures were regenerated, and the custom distribution example test uses seed 9, as its NUTS chain changed with the Beta gradient.
- keanu-tensor: `trigamma`'s asymptotic series subtracted its 1/(42z⁷) term instead of adding it.
- keanu-tensor: `digamma` and `trigamma` of negative arguments use the reflection formulas, so they no longer loop |x| times (or for ever from about -1e16). Non-positive integers are still poles (-∞ and +∞).
- keanu-tensor: shape lengths are computed with checked arithmetic. `Tensor::new` returns `LengthMismatch`, `reshape` returns `InvalidReshape`, and broadcasting and `matmul`/`tensor_multiply` return `IncompatibleShapes` when the length overflows `usize`, instead of panicking or wrapping. A shape with a zero dimension has length 0 whatever its other dimensions. `full`, `zeros`, `ones` and `from_fn` stay infallible and panic on overflow, as `vec!` does.
- keanu-tensor: slicing with a huge step (e.g. `"0:10:9223372036854775807"`) no longer overflows; it takes one element.
- keanu-tensor: `argmax_axis`/`argmin_axis` along an empty axis return an error instead of index 0.
- keanu-tensor (breaking): `DoubleTensor::arange` and `DoubleTensor::eye` return `Result`. `arange` rejects non-finite bounds and ranges too long to allocate, and `eye` rejects n where n × n overflows, instead of panicking.
- keanu-tensor: `tensor_multiply` over zero-length contracted axes returns zeros (e.g. [2, 0]·[0, 3] gives a [2, 3] zero tensor, as `matmul` does) instead of failing to reshape.
- Internal refactor, no behaviour or file format change: keanu-graph describes each distribution once, in a table that generates its name, parameter types, sampling, log probability and gradient; keanu-io keeps one table of distribution and simple-operation Java classes and parameter names, shared by saving, loading and DOT export (with hashed / binary search lookups instead of a scan per vertex), and saving parses slices with keanu-tensor's parser, now public as `keanu_tensor::ops::shape_ops::{parse_slice, SliceItem}`.
- keanu-graph (breaking): node ids that aren't nodes of the graph, builder or state (out of range, from another graph, or from `NodeId::from_index`) no longer panic. Lookups return `None`/`false` (`State::value`/`f64`/`has_value`/`is_observed`, `GraphBuilder::shape`/`dtype`/`label_of`, the first two now returning `Option`, and `is_differentiable`); everything else returns the new `GraphError::UnknownNode`: `State::set`/`set_and_cascade`/`observe`/`eval`/`lazy_eval`/`cascade`/`sample_prior`, `State::clear`/`unobserve`, `Model::affected_by`, `downstream_lambda_section`/`upstream_lambda_section` and `GraphBuilder::set_label_of` (these five now returning `Result`), the auto diff entry points, and handles from `double_var` (failing `build()`). Using a state with a graph of a different size is an error. `Graph::contains` checks an id (see the next entry for the graph's accessors).
- keanu-graph (breaking): no public method panics on an unknown node id. `Graph::node`, `Graph::parents` and
  `Graph::children` return `Option` (`Graph::get` is removed: `node` is the checked lookup), and `graph[id]`
  (`Index<NodeId>`, new) is the unchecked form that panics like slice indexing, for ids taken from the graph itself.
  `node_log_prob`, `node_log_prob_at`, `node_log_prob_gradient`, `log_prob` and `log_prob_gradient` return
  `GraphError::UnknownNode` for an unknown id (`node_log_prob_at` and `log_prob_gradient` panicked;
  `node_log_prob` reported a missing value). keanu-algorithms (breaking): `NetworkSamples::new` returns `Result`,
  failing with `UnknownNode`; Metropolis–Hastings and NUTS `sampler`/`sample` reject unknown nodes to sample up
  front, and `GaussianProposal::propose` returns an error instead of panicking.
- keanu-graph: `State::cascade`, `set_and_cascade`, `eval` and `lazy_eval` cost time proportional to the nodes they visit rather than to the size of the graph (they allocated and scanned a per-node plan on every call), so a Metropolis-Hastings step on one latent of a 110,800-node graph drops from ~460 µs to ~4 µs and a lazy eval with nothing to do from ~170 µs to ~0.04 µs.
- keanu-graph (breaking): `Model::set_and_cascade` takes its values by value, as any `IntoIterator<Item = (NodeId, Value)>`, instead of cloning each from a slice; pass `values.iter().cloned()` to keep borrowed values.
- keanu-graph: faster auto diff: element-wise binary rules no longer multiply by freshly allocated tensors of ones (nor `where` by one to broadcast its mask), reverse mode moves cotangents through proxies, CPTs and multiplexers instead of cloning them, and each shape op's index map (reshape, slice, concat, ...) is computed once per graph instead of on every gradient. Reverse mode through 41 element-wise ops on 1000-element vectors drops from ~200 µs to ~80 µs, forward mode from ~110 µs to ~23 µs, and reverse mode through 15 shape ops on a 100×100 matrix from ~1.7 ms to ~0.4 ms.
- keanu-graph (breaking): lambda functions and Jacobians (`GraphBuilder::lambda`, `lambda_with_jacobian`, `LambdaFn`, `LambdaJacobianFn`) take their parents' values as `&[&Value]`, borrowed from the state, instead of `&[Value]`, which copied every parent value on each evaluation and gradient.
- keanu-algorithms: NUTS computes each leapfrog step's log probability and gradient in one pass with
  `Model::log_prob_and_gradient`, moves the gradient out instead of copying it (new
  `LogProbGradient::take`), copies each position once instead of twice, and doesn't re-set the state when the
  chosen sample is the trajectory's last point. Samples are bit-for-bit unchanged; a fixed-seed chain on 300 scalar
  latents drops from ~6.35 ms to ~6.0 ms per sample, and on one 2000-element vector latent from ~1.55 ms to
  ~1.33 ms.
- keanu-algorithms: `ForwardSampler::sample` returns `GraphError::UnknownNode` (as `AlgorithmError::Graph`) for a
  node that isn't in the model's graph, instead of panicking.
- keanu examples: the custom distribution example (hierarchical Beta–Binomial) samples in BDA-style unconstrained
  coordinates with the same posterior: the hyperprior is a custom density over φ = (logit(a / (a + b)),
  ln(a + b)) including its Jacobian, and the rates are non-centred logit-scale offsets with a second custom
  distribution (whose gradient includes its parameter's). In (a, b, θ) the chains got stuck in the long
  large-a + b tail: across seeds 1–12, NUTS (10,000 samples) was off by up to 0.07, and the test passed only on a
  hand-picked seed. Now NUTS is within 0.0022 and Metropolis–Hastings (40,000 samples) within 0.0125 over seeds
  1–40, and the test checks both, with seed 1. The exact posterior means it checks against were recomputed by
  integration over a fine φ grid (the old 4-digit values were off by up to 0.0011).

## Unreleased: Java

#### Breaking changes
- Removed the ND4J tensor implementations and the ND4J dependency. Keanu's tensors are now pure JVM.
- Removed the OpenBLAS (JavaCPP) dependency. Matrix multiply, LU, inverse, determinant and Cholesky are implemented in pure Java (`KeanuLapack`).
- Removed the Python package (`keanu-python`), the Python code generator (`codegen`), the Py4j byte array converter and the `@ExportVertexToPythonBindings` annotation.
- `ExponentialVertex` now treats its parameter as a rate, as documented (pdf = rate * exp(-rate * x)). Previously it was treated as the mean (scale). To keep old models unchanged pass `1 / oldValue`.
- Element-wise floating point equality now uses IEEE 754 semantics everywhere (`NaN != NaN`, `0.0 == -0.0`).
- `argMax`/`argMin` on floating point tensors now return the index of the first NaN, as numpy does. Use `nanArgMax`/`nanArgMin` to ignore NaNs.
- Integer and long `pow` with a negative exponent now truncates `1 / base^|exponent|` (0 raised to a negative power throws), consistently across all code paths.
- `reshape` with `-1` now throws if the length doesn't divide evenly.
- Slicing now follows numpy for negative steps and steps greater than 1 (e.g. `"::-1"` reverses, `"3:3:2"` is empty).
- The build requires Gradle 9 (wrapper included) and compiles with a JDK 17 toolchain.

#### Bug fixes
- `RegressionModel` treated rank 1 training data as a single sample with many features; a vector is now samples
  of one feature (and is reshaped the same way for `predict`).
- The Laplace distribution's gradient with respect to x and mu was NaN at the mode (0 / 0); it is now the
  subgradient 0.
- Gamma sampling (and Beta, Dirichlet, ChiSquared and StudentT, which use it) was biased when the shape k < 1 because the wrong constant (Euler-Mascheroni instead of e) was used.
- Multinomial gradient with respect to p was wrong.
- UniformInt sampling was biased, and could return the exclusive upper bound, for ranges including negative numbers.
- Triangular logProb was -infinity at the mode and NaN when the mode equalled xMin or xMax. Triangular now also has a gradient.
- Poisson sampling did not broadcast mu correctly. Binomial and Poisson logProb were NaN instead of -infinity outside of their support.
- MultivariateGaussian logProbGraph used the wrong dimension when mu was batched, and compiled log prob graphs failed for non-scalar latents of most distributions.
- Metropolis Hastings with the prior proposal and several dependent variables proposed at once used the wrong Hastings correction.
- Particle filter: latents were sampled using another particle's parent values, resampling weights underflowed to zero, and building one failed on Java 9+.
- NUTS could not be reused for a second sampling run (all steps diverged). The first post-adaptation sample now uses the averaged step size.
- Adam's default convergence check never converged when the optimum was near zero, and the initial "flat gradient" check could reject steep negative gradients.
- `NetworkSamples.downSample` reported the wrong size.
- Forward mode auto diff with several requested vertices only returned the first one reached.
- Simulated annealing ignored the configured variable selector.
- Tensors: `atan2` with a scalar argument had its arguments swapped, `max`/`min` ignored infinities, `diagPart` was wrong for batches of non-square matrices, `matrixDeterminant` returned a single 0 for a whole batch with one singular matrix, `tensorMultiply` failed with negative dimensions and mutated its arguments, `logAddExp`/`logAddExp2` returned NaN for two matching infinities, `argMax`/`argMin` rejected negative axes.

- Gamma, ChiSquared, InverseGamma, LogNormal and Beta logProb returned NaN instead of -infinity outside of their support, and Gamma with k = 1 was NaN at x = 0.
- The Beta distribution's logProb and dLogProb ignored its xMin / xMax bounds (sampling honoured them). `BetaVertex` always uses [0, 1] so it is unaffected apart from the support fix.
- Loading a saved network (protobuf or JSON) dropped the values, including observations, of integer vertices.
- Networks containing a full reduction (`sum()`, `product()` or `mean()` over all dimensions) could not be saved.
- `RoundVertex` claimed to be differentiable and reported an identity gradient. floor, ceil, round and sign are now consistently non-differentiable, and asking for a partial derivative through a non-differentiable vertex throws an `IllegalArgumentException` instead of a `NullPointerException`.
- Unary `max()` / `min()` vertices reported their input's shape instead of a scalar, and had no auto diff. They now pass the gradient to the selected element (split equally between ties).
- Binary `max` / `min` vertices did not broadcast their operands, unlike the tensor operations.
- `logGamma(+infinity)` returned NaN; it now returns +infinity.
- Reverse mode autodiff lost one contribution when the same vertex was both operands of a binary op (e.g. `x.times(x)`), halving the gradient.
- Forward mode autodiff through a full `sum()` or `mean()` summed the wrong dimensions when the with-respect-to vertex's rank differed from the operand's, giving wrongly shaped partials.
- The gradient of `product()` was NaN where an element is zero; it is now exact.
- Forward mode through `where` (and so binary `max` / `min`), and both modes through a double CPT, reported no partial when only an unselected branch depended on the input; the partial is now zero.
- `Diffs.get` returned the gradient of the nearest smaller parameter name instead of failing when asked for a parameter that is not present.
- `Categorical.logProb` returned 0 (probability 1) for a value that is not one of the categories; it now returns -infinity.
- `Binomial`, `Gamma`, `Laplace` and `Poisson` sampling failed when the parameters had a larger shape than the requested sample shape; they now return the broadcast shape.
- The forward sampler rejected any model with two or more observed vertices, even when no random variable was upstream of them.
- Adam threw a `NullPointerException` when maximizing the likelihood of a model with a latent the likelihood doesn't depend on; that latent now has a zero gradient (as with conjugate gradient).
- Sequence labels used an identity hash code, so they changed every time a program ran. They now use a counter (`SequenceBuilder.resetSequenceIdentifiers()` restarts it).
- Networks could not be saved if any vertex had an array parameter: reshape, permute, broadcast, take, tensor multiply, sum / product / mean over given dimensions (`int[]` / `long[]`), or a string slice (`boolean[]`, now saved as a boolean tensor and converted back on load).
- `argMax()` / `argMin()` / `nanArgMax()` / `nanArgMin()` over every element, and string slices without an ellipsis, could not be saved (their null parameters were not nullable).
- `KDEVertex` failed to evaluate its log prob or gradient at its own (scalar) shape.
- `UniformIntVertex` threw when asked for its gradient, so a model containing one could not use gradient based algorithms (NUTS, gradient optimizers); its gradient contribution is now empty.
- The "No value returned from Save Function" error names the vertex type and parameter.

#### New
- Golden test vectors (`golden-vectors/`, generated by the `keanu-golden` module) recording the Java implementation's results as the parity reference for the Rust port. `./gradlew check` fails if they change.
- Auto diff for `cumSum`, `cumProd`, `tensorMultiply`, `choleskyDecomposition`, `choleskyInverse`, `standardize`, `standardDeviation`, `trigamma`, `logAddExp` and `logAddExp2`.

## Version 0.0.27

### Java

#### Breaking changes
- Moved `io.improbable.keanu.vertices.dbl` classes to `io.improbable.keanu.vertices.tensor.number.floating.dbl`
- Moved `io.improbable.keanu.vertices.int` classes to `io.improbable.keanu.vertices.tensor.number.fixed.int`
- Moved `io.improbable.keanu.vertices.bool` classes to `io.improbable.keanu.vertices.tensor.bool`
- renamed DoubleTensor `determinant()` method `matrixDeterminant()`
- renamed DoubleTensor `average()` to `mean()` and allowed it to be done on a given dimension

### Common

- Previously some operations were available on the tensor classes but not the vertex classes. All operations
on the tensor classes have been implemented with vertex operations now. A few operations have been added in order
to accommodate autodiff for the new vertex operations.

- All Tensor operations now have a corresponding Vertex. This brings the DoubleTensor, IntegerTensor and BooleanTensor
  in line with the DoubleVertex, IntegerVertex and BooleanVertex.   

- Along with this change, the vertices package has been changed:
io.improbable.keanu.vertices.* -> io.improbable.keanu.vertices.tensor.*

- There is now an experimental LongTensor implementation.
- There is now a native JVM implementation of the IntegerTensor. This will bring a significant performance increase 
to models with lots of small integer operations.

#### New operations

- sign
- strided slice
- trigamma
- triLower
- triUpper
- fillTriangular
- batch matrix multiply
- batch matrix inverse
- batch matrix determinant
- batch cholesky decomposition
- cholesky inverse (with batch)


## Version 0.0.26

### Common

* Made constructing the children set of a vertex more efficient by not constantly reconstructing the set.
* Upgrade to nd4j 1.0.0-beta4

### Java

* Fixed an error where a space was present in one of the dependencies, which broke some build systems fetching dependencies.

- Fixed issues with MultinomialVertex. The issues with it before were:
    - It was extremely strict on checking that p summed to 1.0
    - It disallowed scalar n. This is probably the most common use case.
    - It didn't support the usual batch sampling nor batch logProb (see MultinomialVertex.java for docs)

- MultinomialVertex Functional changes:
    - The shape of p expected k to be the far left dimension. It is now the far right in order to allow for broadcasting semantics.
    - n and p parameter validation was semi-controllable but is now completely toggleable with `vertex.setValidationEnabled(true);`

- Add more generic tensor slice that allows start stop and interval slicing.

- Change name on DoubleTensor compare methods by removing get (e.g. `getGreaterThanMask(...))` -> `greaterThanMask(...)`)

- BooleanTensor now uses boolean arrays instead of Nd4j doubles

- Added custom distribution example to examples

- Added to DoubleTensor operations inspired by numpy: 
    - cumSum
    - cumProd
    - product
    - logAddExp
    - logAddExp2
    - log10
    - log2
    - log1p
    - exp2
    - expM1
    - tanh
    - atanh
    - sinh
    - asinh
    - cosh
    - acosh

- Fixed bug in NUTS where NaN gradients do not cause a step to be divergent

### Python

* Faster tensor creation in Python


## Version 0.0.25

## Version 0.0.24

### Common

* Sequences save / load

## Version 0.0.23

### Common

* You can now build a `Sequence` using more than one factory, whose vertices may be dependent on another.
  * In Java: you can call `SequenceBuilder#withFactories` instead of `SequenceBuilder#withFactory`
  * In Python: the parameter has been renamed from `factory` to `factories` (BREAKING CHANGE). You can pass in one or a list of multiple factories. 

## Version 0.0.22

### Common

* Generalised `GaussianProposalDistribution` to take different sigmas for each latent.
  * Example in Python:
    * `MetropolisHastingsSampler(proposal_distribution='gaussian', latents=[v1, v2], proposal_distribution_sigma=[1., np.array([[1., 2.], [3., 4.]]))`
* `DoubleVertex#matrixMultiply` now performs dot product when given two vectors, and matrix-vector product when given a matrix and a vector.
* Autodiff for Slice Vertices has now been fixed (caused issues with NUTs when run on a graph containing Slices)
* `MultivariateGaussianVertex` now requires that mu is a vector instead of a matrix.
* `DoubleTensor.scalar` will throw an exception if the DoubleTensor is not actually a scalar. Please use .getValue(0) if you just
want the first element in a DoubleTensor.

### Python

* Improved performance of getting samples by coercing them into a `Tensor` in Java (instead of iterating through an `ArrayList`).

## Version 0.0.21 

### Common

* Using Buildkite for CI
* Can save dot files with disconnected vertices in BayesianNetwork. You can also just pass a list of vertices to DotSaver.
* Plates have been renamed
  * This was done primarily because we were using the term incorrectly, since Dynamic Bayes Nets (e.g. Hidden Markov) are not Plates.
  * `Plates` --> `Sequence`
  * `Plate` --> `SequenceItem`
  * `plate` package --> `template`
* `ForwardSampler`
  * Samples from the prior of a bayesian network. The network cannot have observations that are dependent on random variables.
  * It's accessible from the `Keanu` factory like so: `Keanu.Sampling.Forward.withDefaultConfig()`
  * There is a togglabe option to record the log prob of each sample. This is disabled by default for performance reasons.
  
### Python

* Expose `unobserve`
* Added `iter_all_vertices`
* Rename methods returning generators to `iter_*` from `get_*`
* Split the ND4J dependencies into their own pip package.

## Version 0.0.20

### Common

* Saving a network as a DOT file includes labels on constant vertices.

### Python

* Improved performance of getting samples by using byte streams.
* Added Python docstrings for sampling

## Version 0.0.19

* Added `get_vertex_by_label` to `BayesNet`
* Added optional param `label` for all vertices in Python (e.g. `Gaussian(0., 1., label="gaussian")`). Now you must label vertices you are sampling from, otherwise Keanu will throw an exception.
* Improved the structure of the Python sample dictionary
  * Scalar networks
    * If all sampled vertices are scalar then you are returned a dictionary keyed on vertex ID with the values containing a list of primitive values.
    * An example of a network of size 2 with three samples: `{"gaussian": [0., 1., 2.], "exponential": [0.5, 1.5, 3.5]}`
    * `samples["gaussian"] = [0., 1., 2.]`
  * Tensor networks
    * If any of the sampled vertices are tensors then you are returned a dictionary keyed on a tuple of vertex ID and tensor index. 
    * This makes it much easier to extract all the samples for a given index and for creating a multi-index dataframe
    * An example of a scalar exponential and a `2x2` tensor gaussian with three samples:
    ```
            exp  gaussian                               
            (0)    (0, 0)    (0, 1)    (1, 0)     (1, 1)
    0  4.231334  5.017627  5.734130  3.904472   9.909033
    1  4.231334  5.017627  5.734130  3.904472   9.909033
    2  4.676046  4.308018  5.035550  6.240894  10.112683
    ```
    * `samples[("gaussian", (0, 1))] = [5.734130, 5.734130, 5.035550] `
  * As a result, whenever keying into a sample dictionary, you are guaranteed to receive a list of primitives.
  * This greatly reduces the complexity of the `autocorrelation` and `traceplot` API's as they now simply expect a list of values.  
* Added Adam optimizer
* GradientOptimizer and NonGradientOptimizer now takes an algorithm argument that by default will use Conjugate Gradient and BOBYQA respectively.
* GradientOptimizer and NonGradientOptimizer return a OptimizedResult object instead of just the optimized fitness as a double
* Reorganised the factory methods for building `PosteriorSamplingAlgorithm` objects. The following are available and give you access to either a default implementation or, when you need more control over the configuration, a Builder object:
  * `Keanu.Sampling.MetropolisHastings`
  * `Keanu.Sampling.NUTS`
  * `Keanu.Sampling.SimulatedAnnealing`
  * `Keanu.Sampling.MCMC` (to automatically choose between `MetropolisHastings` and `NUTS`)
* The `PosteriorSamplingAlgorithm` objects are now immutable: you cannot set their values after construction.
* When you choose a custom configuration of `MetropolisHastings` or `SimulatedAnnealing`, you can specify:
  * the proposal distribution. Options are `PriorProposalDistribution` and `GaussianProposalDistribution`
  * a proposal rejection strategy - options are `RollBackToCachedValuesOnRejection` and `RollbackAndCascadeOnRejection`
    * (in Python, this is done for you)
* `Hamiltonian` Monte Carlo has been removed: use `NUTS` instead which is an auto-tuning implementation of Hamiltonian Monte Carlo.
* The `PosteriorSamplingAlgorithm` and `FitnessFunction` interfaces requires a `ProbabilisticModel` as its argument instead of a `BayesianNetwork`
  * You can create a `ProbabilisticModel` from a `BayesianNetwork`:
    * For `MetropolisHastings`: `new KeanuProbabilisticModel(bayesNet)`
    * For `NUTS`: `new KeanuProbabilisticModelWithGradient(bayesNet)`
* The `ProposalDistribution` interface uses `Variable` instead of `Vertex`
  * A `Variable` is an abstraction that does not assume any graph-like structure of your model.
* `KeanuRandom` has been moved to package `io.improbable.keanu`
* The `ProposalListener` interface has been changed: `onProposalApplied` is now called `onProposalCreated`.
  * This is because the `Proposal` interface no longer has `apply` and `reject` methods.
* Reversed the ordering of namespace for `VertexLabel` (e.g. from `new VertexLabel("keanu", "improbable", "io")` to `new VertexLabel("io", "improbable", "keanu"))
* Added better Python operator overloading support.
  - Includes automatically casting to the correct datatype where possible.
* Added Geometric Distribution vertex
* Fixed issue with certain vertices not taking a list properly in Kotlin
* Added `__version__` attribute to Python
* Added a permute vertex
* Added the MIRSaver/MIRLoader + the proposed MIR proto format
* Added the release notes text file to the repo
