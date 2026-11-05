# Gaussian processes

A Gaussian process (GP) is a prior over functions: any finite set of function values f(x₁), …, f(xₙ) is
multivariate Gaussian, with a mean m(x) and a covariance given by a kernel k(x, x'). With Gaussian observation noise,
y = f(x) + ε with ε ~ N(0, noise²), the targets are jointly Gaussian too, so the marginal likelihood of the data and
the predictive distribution at new inputs are both available in closed form. These are *exact* GPs: their cost is
cubic in the number of training points, so they suit up to a few thousand points.

The `keanu::graph::gp` module builds both from ordinary graph nodes. The kernel's hyperparameters (amplitudes,
length scales, periods) and the noise can be constants or latent variables, and gradients flow through the GP to
them, so they can be fitted by MAP or sampled by NUTS like any other latent.

```rust,ignore
{{#include ../../keanu/tests/book.rs:gaussian_processes}}
```

## Kernels

A `Kernel` builds the covariance matrix between two sets of inputs. Inputs are `[n, d]` matrices with one row per
point, or `[n]` vectors for one feature; `kernel.covariance(x1, x2)` with x1 `[n, d]` and x2 `[m, d]` is `[n, m]`.
In the table, σ is the amplitude and r² = Σ_d ((x_d − x'_d) / ℓ_d)² is the squared distance scaled by the length
scale ℓ, which is a scalar or one value per feature (automatic relevance determination).

| Constructor | k(x, x') |
|---|---|
| `Kernel::exp_quad(&b, σ, ℓ)` | σ² exp(−r²/2) (squared exponential, RBF) |
| `Kernel::matern12(&b, σ, ℓ)` | σ² exp(−r) |
| `Kernel::matern32(&b, σ, ℓ)` | σ² (1 + √3 r) exp(−√3 r) |
| `Kernel::matern52(&b, σ, ℓ)` | σ² (1 + √5 r + 5r²/3) exp(−√5 r) |
| `Kernel::rational_quadratic(&b, σ, ℓ, α)` | σ² (1 + r²/(2α))^(−α) |
| `Kernel::periodic(&b, σ, ℓ, p)` | σ² exp(−2 Σ_d sin²(π (x_d − x'_d)/p) / ℓ_d²) |
| `Kernel::linear(&b, σ_b, σ_s, c)` | σ_b² + σ_s² (x − c)·(x' − c) |

Kernels combine with `+` and `*`: `Kernel::exp_quad(&b, 1.0, 10.0) * Kernel::periodic(&b, 1.0, 1.0, 12.0)` is a
locally periodic kernel, and adding `Kernel::linear(..)` adds a trend.

## Marginal likelihood and prediction

`GaussianProcess::new(kernel)` is a zero-mean, noise-free GP; `with_noise(sd)` sets the observation noise,
`with_mean(c)` a constant mean and `with_mean_fn(|x| ...)` a mean function of the inputs (which may use other nodes,
e.g. latent regression coefficients).

- `gp.marginal(x, y)` adds a multivariate Gaussian over y with covariance K(x, x) + (noise² + jitter)·I and observes
  it, so the GP marginal likelihood joins the model's log likelihood. `gp.prior(x)` is the same node unobserved.
- `gp.predict(x_train, y_train, x_new)` returns the mean and covariance nodes of the latent function at `x_new`,
  computed with a Cholesky factor L of the training covariance and triangular solves: the mean is
  m(x_new) + K*ᵀ L⁻ᵀ L⁻¹ (y − m(x_train)) and the covariance K(x_new, x_new) − VᵀV with V = L⁻¹ K*. No matrix is
  inverted. `prediction.variance()` is the diagonal. The predictive excludes the observation noise; add noise² to
  the variance for the spread of new observations.

The free functions `gp_marginal(x, y, mean, &kernel, noise)` and `gp_predict(x_train, y_train, x_new, mean, &kernel,
noise)` do the same with a constant mean in one call.

The jitter, 10⁻⁶ by default (`with_jitter` changes it), keeps the Cholesky factorisation stable when the noise is zero
or tiny. With zero noise the predictive at the training inputs returns the training targets with variance of the
order of the jitter.

Shape errors (inputs with different feature counts, a length scale with the wrong number of elements, targets of the
wrong length) fail `build` like any other invalid node. Hyperparameter values are not checked: give them priors on the
right support, such as `half_gaussian`, and the samplers and optimizers keep them positive.

Parts of the kernel that depend only on constant inputs are deterministic nodes with no latent ancestor, so compute
every node once (as above, with `lazy_eval` over `graph.ids()`) before running an optimizer on the model.
