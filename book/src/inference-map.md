# Most probable values (MAP)

Given observations, the **maximum a posteriori** (MAP) estimate is the most probable value of every latent. The
**maximum likelihood** estimate ignores the latents' priors and maximizes only the observations' probability.

`GradientOptimizer` finds both using gradients computed by automatic differentiation. It works on continuous (f64)
latents; for models with discrete latents, use [sampling](inference-sampling.md) or simulated annealing instead.

```rust,ignore
{{#include ../../keanu/tests/book.rs:map}}
```

## Choosing an algorithm

Three gradient algorithms are available:

- **`Lbfgs`** (the default) is limited-memory BFGS with Stan's settings: a history of 5 steps
  (`history_size`) and a line search for the strong Wolfe conditions. It stops, converged, when the log
  probability changes by less than `tol_obj` (1e-12) or by less than `tol_rel_obj` (1e4) machine epsilons relative
  to its size, when the gradient's norm is below `tol_grad` (1e-8) or its size relative to the log probability is
  below `tol_rel_grad` (1e7) machine epsilons, or when the latents move less than `tol_param` (1e-8); and, without
  converging, after `max_iterations` (2000). On the golden models it needs between a half and a ninth of
  conjugate gradient's evaluations.
- **`ConjugateGradient`** is nonlinear conjugate gradient, Polak–Ribière+ with a line search.
  Settings: `max_evaluations`, `relative_threshold` and `absolute_threshold` (both 1e-8 by default; the optimizer
  stops when the change in log probability is below either), and `formula`.
- **`Adam`**, with its usual settings: `alpha` (the step size), `beta1`, `beta2`, `epsilon`, `max_evaluations`, and
  `convergence` (a relative or absolute checker on the change in position).

```rust,ignore
{{#include ../../keanu/tests/book.rs:map_settings}}
```

If an algorithm runs out of evaluations or iterations, it returns its current point with `converged: false` rather
than an error. L-BFGS shortens a step that reaches an impossible log probability or a NaN or infinite gradient, and
stops with `converged: false` if no step along the gradient improves the log probability. Conjugate gradient stops
with `converged: false`, at the best point found, if the log probability becomes impossible or the gradient NaN or
infinite. Unless the initial-condition check is turned off, an impossible start or a NaN or infinite starting
gradient is an error; for L-BFGS an impossible start is an error either way.

## Constrained latents

As [NUTS](inference-sampling.md#constrained-latents) does, the gradient optimizers work in **unconstrained space** by
default: each latent is mapped to the real line through a transform chosen from its support (log for a positive
scale, logit for a probability, and so on), so every point they try is inside the support. Unlike sampling, MAP and
maximum likelihood maximize the model's log probability at the latents' own values, *without* the log-Jacobian of
the transforms, so an optimum inside the support is exactly the one you would get without transforms:

```rust,ignore
{{#include ../../keanu/tests/book.rs:map_transforms}}
```

(With the Jacobian, the optimum would be the mode of the density of log σ, kθ = 1.)

When the optimum is on the boundary of a support, such as a scale whose most probable value is 0, it is at
infinity in unconstrained space. The optimizer stops once a bounded latent's unconstrained coordinate passes ±30
(also when it stalls on the way, with the coordinate past ±5 and the log probability no better than at ±31), and
reports the latent in `OptimizedResult::boundary`, with `converged: false` and the latent's value left next to the
boundary:

```rust,ignore
{{#include ../../keanu/tests/book.rs:map_boundary}}
```

`space(Space::Constrained)` opts out and optimizes the latents' own values, where a step can leave the support (the
log probability there is −∞) and Adam follows exactly Java Keanu's trajectory. In unconstrained space a starting
value outside its support, or on its boundary, is an impossible start even with the initial-condition check off.

## Without gradients: BOBYQA

`NonGradientOptimizer` finds the same optima without gradients, using BOBYQA (Powell's bound optimisation by
quadratic approximation, translated from [PRIMA](https://github.com/libprima/prima)). Use it when gradients are
unavailable or unreliable. Its settings are Java Keanu's:

- `initial_trust_region_radius` (default 10) and `stopping_trust_region_radius` (default 1e-8): the trust region
  starts at the first and shrinks to the second, which sets the accuracy of the result;
- `bounds_range` (default unbounded): latents without explicit bounds stay within this distance of their start;
- `max_evaluations` (default unlimited) and `interpolation_points` (default 2n + 1 for n latent dimensions).

As in PRIMA, settings BOBYQA can't use are adjusted rather than rejected: fewer than n + 3 evaluations are raised
to n + 3, an `interpolation_points` outside n + 2 to (n + 1)(n + 2) / 2 falls back to the default, and the initial
radius shrinks to fit the bounds. If evaluating the model fails, the optimization stops at once with that error.

`OptimizerBounds` bounds individual latents, with a scalar or a tensor shaped like the latent:

```rust,ignore
{{#include ../../keanu/tests/book.rs:bobyqa}}
```

A lower bound of −∞ or an upper bound of +∞ means no bound on that side. A NaN bound, or a lower bound that is
not below its upper bound (including a lower bound of +∞ or an upper bound of −∞), is an error.

BOBYQA always works on the latents' own values, but it keeps them inside their supports: each element of a latent
with a lower-bounded or interval support (a scale, a probability) is bounded by that support, intersected with any
explicit bounds and with `bounds_range` around the start. Explicit bounds that don't overlap the support at all are
an error. An optimum with
zero probability (possible only with the initial-condition check off) is never reported as converged.

Unlike Java's, it also works when there is only one latent dimension.
