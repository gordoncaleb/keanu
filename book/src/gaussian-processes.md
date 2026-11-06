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
| `Kernel::matern(&b, ν, σ, ℓ)` | σ² 2^(1−ν)/Γ(ν) (√(2ν) r)^ν K_ν(√(2ν) r), any constant ν > 0 |
| `Kernel::rational_quadratic(&b, σ, ℓ, α)` | σ² (1 + r²/(2α))^(−α) |
| `Kernel::periodic(&b, σ, ℓ, p)` | σ² exp(−2 Σ_d sin²(π (x_d − x'_d)/p) / ℓ_d²) |
| `Kernel::linear(&b, σ_b, σ_s, c)` | σ_b² + σ_s² (x − c)·(x' − c) |

Kernels combine with `+` and `*`: `Kernel::exp_quad(&b, 1.0, 10.0) * Kernel::periodic(&b, 1.0, 1.0, 12.0)` is a
locally periodic kernel, and adding `Kernel::linear(..)` adds a trend.

The general Matérn kernel takes its smoothness ν as a number, not a node, and computes the modified Bessel function
K_ν in the graph (`DoubleVar::matern_correlation`), with first and second derivatives in the distance, so its
amplitude and length scale may be latent. At ν = 1/2, 3/2 and 5/2 it equals the closed forms above, which are
cheaper; it tends to the exponentiated quadratic as ν grows.

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

## Hilbert space approximate GPs (HSGP)

An exact GP's cost grows with the cube of the number of points. For long series or large spatial data sets,
`Hsgp` approximates a GP with an exponentiated quadratic or Matérn kernel by a weighted sum of fixed basis functions
(Riutort-Mayol, Bürkner, Andersen, Solin and Vehtari, 2023):

f(x) ≈ Σⱼ φⱼ(x) · √S(√λⱼ) · βⱼ, βⱼ ~ N(0, 1).

The φⱼ are the eigenfunctions of the Laplacian on a box [centre − L, centre + L] that contains the inputs, λⱼ their
eigenvalues and S the kernel's spectral density, which carries the amplitude and length scale. The basis functions
depend only on the inputs, so they are a constant matrix Φ, and f = Φ (√S ⊙ β) costs O(n·m) to evaluate. The
function is an ordinary latent of the model (through the weights β), so any likelihood can sit on top of it, not
only a Gaussian one, and gradients and Hessian-vector products in the weights and hyperparameters come from the
graph's autodiff.

```rust,ignore
{{#include ../../keanu/tests/book.rs:hsgp}}
```

- `Hsgp::new(&kernel, m, c)` takes the kernel (`exp_quad`, `matern12`, `matern32`, `matern52` or `matern`; other
  kernels and sums or products of kernels have no spectral density here and are an error: for a sum, add one HSGP
  per term), the number of basis functions m per input dimension and the factor c ≥ 1 by which the box extends the
  inputs' range: L = c · S, with S half the range, centred on its midpoint.
  `with_domain(HsgpDomain::new(centre, L))` sets the box instead, for instance to predict beyond the data.
- `hsgp.prior(x)` adds the weights (`prior.beta`, of length m^d) and returns `prior.f`, the function at the inputs
  (`[n]`, or `[n, d]` with one row per point). Inputs with d dimensions use the full tensor grid of m functions per
  dimension, so m^d weights; the length scale is a scalar or one per dimension.
- `prior.predict(x_new)` is the same function, with the same weights, at new inputs inside the box: its draws are
  draws of the posterior predictive function. `prior.covariance(x1, x2)` is the covariance the approximation implies,
  Φ(x1) diag(S) Φ(x2)ᵀ, to compare with `kernel.covariance(x1, x2)`.

The approximation is accurate when the box is wide enough and there are enough basis functions for the length
scale. Riutort-Mayol et al. recommend c ≥ 1.2 (at least 3.2 ℓ/S for the exponentiated quadratic) and m ≥ 1.75 c / (ℓ/S)
for the exponentiated quadratic; the Matérn kernels' spectra decay more slowly and need more (m ≥ 3.42 c / (ℓ/S) for
Matérn 3/2). The error is largest near the inputs' ends, where the box's edge (at which every basis function
vanishes) is closest: with ℓ = 0.3 on [−1, 1] and m = 40, c = 2 approximates the exponentiated quadratic's
covariance to better than 10⁻⁹, but c = 1.5 only to 4·10⁻³. With latent hyperparameters, give the length scale a
prior that keeps it above the smallest value m can represent.

The GP regression example (`cargo run --release --example gp_regression`) fits the same model with an exact GP and
with an HSGP, sampling the HSGP's weights with its hyperparameters by NUTS.
