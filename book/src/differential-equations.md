# Differential equations

Many models describe how a system changes rather than what it is: an epidemic spreading through a population, a
drug being absorbed and cleared, people moving between states of health. Their mean is the solution of an ordinary
differential equation (ODE)

```text
dy/dt = f(t, y, θ),    y(t₀) = y₀,
```

observed at some times, with parameters θ to infer. Keanu solves such equations inside a graph: an ODE node's value
is the state at each output time, and gradients flow through the solver to y₀, t₀, θ and the times, so NUTS, the
optimizers and ADVI work on ODE models as on any other.

## An ODE node

`GraphBuilder::ode(y0, times, t0, parameters, options, rhs)` adds the node. `y0` is the initial state, a vector of n
elements; `times` a vector of T output times, strictly increasing and after `t0`; `t0` the initial time, a scalar;
and `parameters` the graph values f depends on. `rhs` builds f as a *body* graph, like a
[scan](sequences.md#scan-a-loop-in-one-node)'s: it is given a fresh builder and placeholders for the time (a scalar),
the state (`[n]`) and each parameter, and returns dy/dt, of the state's shape. The node's value has shape `[T, n]`:
row i is y at the i-th output time.

Logistic growth, dN/dt = r N (1 − N / K), fitted to counts:

```rust,ignore
{{#include ../../keanu/tests/book.rs:ode_logistic}}
```

The body is deterministic and sees only its placeholders: a random variable or a factor inside it is an error, and
outer nodes (data, other latents) are passed in through `parameters`. The body may use the time (for forcing terms
such as seasonality) and any graph operation, including matrix products, scans and the
[matrix exponential](#the-matrix-exponential). A state of several compartments is one vector; take its elements with
`take` and assemble dy/dt with `stack`.

## Solvers and tolerances

`OdeOptions` chooses the integrator and its accuracy:

| Option | Default | Meaning |
|---|---|---|
| `solver` | `OdeSolver::Rk45` | `Rk45`: Dormand–Prince 5(4), explicit, for non-stiff problems (Stan's `ode_rk45`). `Stiff`: a Rosenbrock method (Rodas4's coefficients), linearly implicit and L-stable, for stiff problems (Stan's `ode_bdf` plays this role). |
| `rtol`, `atol` | 10⁻⁶, 10⁻⁶ | Each step's estimated local error is kept below `atol + rtol·abs(y)` in each component. |
| `max_steps` | 10⁶ | The most steps over the whole solve. |

The defaults are Stan's. A problem is stiff when it mixes very fast and very slow dynamics (chemical kinetics,
pharmacokinetics with fast absorption, epidemics at very high transmission rates): an explicit method is then
limited by stability rather than accuracy and takes a huge number of tiny steps. The stiff solver uses the body's
exact Jacobian ∂f/∂y and ∂f/∂t, from forward-mode autodiff of the body, and solves a linear system per stage.

```rust,ignore
{{#include ../../keanu/tests/book.rs:ode_options}}
```

Tighten the tolerances when comparing with a reference or taking finite differences; the defaults suit inference,
where the solver's error is far below the noise in the data.

## Gradients

The ODE node is differentiable in all its parents by *forward sensitivities*: alongside the state, the solver
integrates S = ∂y/∂p, which obeys dS/dt = J S + ∂f/∂p (J = ∂f/∂y), with S at t₀ the identity for y₀, zero for θ and
−f(t₀, y₀) for t₀. The right-hand side J S + ∂f/∂θ comes from one batched forward sweep through the body. The step
size control covers the sensitivities as well as the state (as CVODES with `errconS`), so gradients are as accurate
as values. The partial with respect to an output time tᵢ needs no integration: it is f(tᵢ, y(tᵢ)).

```rust,ignore
{{#include ../../keanu/tests/book.rs:ode_gradient}}
```

A gradient integrates n·(1 + p) equations, for p the number of parameter elements it is taken through; parents
that are constants (`b.constant`, `b.scalar`) are skipped. The value itself is solved without sensitivities, so a
log probability and gradient costs one plain solve and one with sensitivities. For large systems with many
parameters the cost grows like n·p per step; adjoint sensitivities, which would scale better, are not implemented.

There is no second-order rule: Hessian-vector products through an ODE fail with `GraphError::NoSecondOrderRule`, so
the [Laplace approximation](inference-map.md#the-laplace-approximation) and exact-Hessian methods don't apply to ODE models. NUTS, the gradient
optimizers and ADVI need first derivatives only.

## When the solver fails

At some parameter values the solution may not exist over the output times (it blows up), or the problem may be so
stiff that the step limit is reached. The solve then fails with `GraphError::SolverFailed`, saying where and why
(`FailureReason::MaxSteps`, `StepSizeUnderflow` or `NonFinite`). Models treat that as an *impossible point*, a
point of zero probability, rather than an error that stops inference (`GraphError::is_impossible_point`):

| Where | What happens |
|---|---|
| `Model::log_prob_after`, `log_prob_and_gradient`, `Parameterisation::log_prob(_and_gradient)` | −∞ (and no gradient) |
| NUTS | the trajectory is divergent, and sampling continues |
| Metropolis–Hastings, Gibbs, discrete Metropolis | the proposal or value is rejected |
| Gradient optimizers | the step is rejected (as one with −∞ log probability) |
| `probe_for_non_zero_probability` | resamples, as for any impossible state |
| ADVI | a draw at an impossible point makes the ELBO non-finite (`NonFiniteElbo`), as for any model with zero-probability regions |

```rust,ignore
{{#include ../../keanu/tests/book.rs:ode_failure}}
```

Output times that are not strictly increasing after t₀ are a malformed model, not an impossible point: evaluation
fails with an invalid-node error.

Priors that keep the parameters where the system is well behaved are the first remedy for frequent failures; if a
non-stiff problem keeps hitting the step limit, switch to `OdeSolver::Stiff`.

## The matrix exponential

A linear system with constant coefficients, dy/dt = A y, has the closed-form solution y(t) = exp(A t) y(0), and
`DoubleVar::matrix_exp` computes the matrix exponential (by scaling and squaring with Padé approximants), batched over
leading axes, with exact first and second derivatives. It is the workhorse of continuous-time Markov chains: with a
generator Q (off-diagonal rates, rows summing to zero), the transition probabilities over a time t are P(t) =
exp(Q t).

```rust,ignore
{{#include ../../keanu/tests/book.rs:matrix_exp}}
```

Unlike the ODE node it has a second-order rule, so it works with the Laplace approximation too.

## Solvers outside graphs

The integrators are also available on plain slices, in the `keanu::graph::ode` module: `integrate` with closures
for f and its Jacobian, `solve` for an `OdeSystem` whose evaluation can fail, and `solve_sensitivities` for a
`SensitivitySystem`.

```rust,ignore
{{#include ../../keanu/tests/book.rs:ode_integrate}}
```

## Worked examples

Three examples (see [Examples](examples.md)) fit ODE and continuous-time Markov models.

**`sir`**: the SIR epidemic model, fitted to the 1978 influenza outbreak at an English boarding school with a
negative binomial likelihood, following the Stan case study by Grinsztajn et al. (2021). The body is the three
compartments' rates:

```rust,ignore
{{#include ../../keanu/examples/sir.rs:sir_ode}}
```

and the daily counts in bed are negative binomial around the infected compartment:

```rust,ignore
{{#include ../../keanu/examples/sir.rs:sir_likelihood}}
```

Four NUTS chains give β ≈ 1.74, γ ≈ 0.54, R₀ = β / γ ≈ 3.2 and a recovery time of about 1.85 days, as in the case
study. The example's test also fits data simulated with known β and γ and checks that both fall in their 90%
posterior intervals, and checks the gradient of the log probability against finite differences to 10⁻⁶ relative.

**`pharmacokinetics`**: the one-compartment model with first-order absorption of an oral dose, written both as an
ODE and with its closed-form solution; the test checks that the two models' log probabilities and gradients agree
to the solver's tolerance and that their posteriors agree.

```rust,ignore
{{#include ../../keanu/examples/pharmacokinetics.rs:pk_ode}}
```

**`multi_state`**: an illness–death model, a three-state continuous-time Markov chain whose subjects are seen only at
irregular visits (interval-censored panel data). The transition matrices over the visit gaps are one batched matrix
exponential, and each visit's state is a categorical draw from the row of the state before it:

```rust,ignore
{{#include ../../keanu/examples/multi_state.rs:transition_matrices}}
```

```rust,ignore
{{#include ../../keanu/examples/multi_state.rs:panel_likelihood}}
```

## Saving

ODE nodes save and load with their body graph and options (see [Saving and loading](save-and-load.md)). When
loading untrusted files, `LoadOptions::max_ode_steps` (default 10⁷) caps the step limit a saved ODE may ask for.
