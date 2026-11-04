# Most probable values (MAP)

Given observations, the **maximum a posteriori** (MAP) estimate is the most probable value of every latent. The
**maximum likelihood** estimate ignores the latents' priors and maximizes only the observations' probability.

`GradientOptimizer` finds both using gradients computed by automatic differentiation. It works on continuous (f64)
latents; for models with discrete latents, use [sampling](inference-sampling.md) or simulated annealing instead.

```rust,ignore
{{#include ../../keanu/tests/book.rs:map}}
```

## Choosing an algorithm

Two gradient algorithms are available:

- **`ConjugateGradient`** (the default) is nonlinear conjugate gradient, Polak–Ribière+ with a line search.
  Settings: `max_evaluations`, `relative_threshold` and `absolute_threshold` (both 1e-8 by default; the optimizer
  stops when the change in log probability is below either), and `formula`.
- **`Adam`**, with its usual settings: `alpha` (the step size), `beta1`, `beta2`, `epsilon`, `max_evaluations`, and
  `convergence` (a relative or absolute checker on the change in position).

```rust,ignore
{{#include ../../keanu/tests/book.rs:map_settings}}
```

If an algorithm runs out of evaluations, it returns its current point with `converged: false` rather than an error.
Conjugate gradient also stops with `converged: false`, at the best point found, if the log probability becomes
impossible or the gradient NaN or infinite. Unless the initial-condition check is turned off, an impossible start
or a NaN or infinite starting gradient is an error.

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
not below its upper bound (including a lower bound of +∞ or an upper bound of −∞), is an error. An optimum with
zero probability (possible only with the initial-condition check off) is never reported as converged.

Unlike Java's, it also works when there is only one latent dimension.
