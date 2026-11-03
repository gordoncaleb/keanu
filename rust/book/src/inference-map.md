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

## Without gradients: BOBYQA

`NonGradientOptimizer` finds the same optima without gradients, using BOBYQA (Powell's bound optimisation by
quadratic approximation, translated from [PRIMA](https://github.com/libprima/prima)). Use it when gradients are
unavailable or unreliable. Its settings are Java Keanu's:

- `initial_trust_region_radius` (default 10) and `stopping_trust_region_radius` (default 1e-8): the trust region
  starts at the first and shrinks to the second, which sets the accuracy of the result;
- `bounds_range` (default unbounded): latents without explicit bounds stay within this distance of their start;
- `max_evaluations` (default unlimited) and `interpolation_points` (default 2n + 1 for n latent dimensions).

`OptimizerBounds` bounds individual latents, with a scalar or a tensor shaped like the latent:

```rust,ignore
{{#include ../../keanu/tests/book.rs:bobyqa}}
```

Unlike Java's, it also works when there is only one latent dimension.
