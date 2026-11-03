# Regression models

`RegressionModel` fits a generalised linear model to training data without you building the graph by hand. The
outputs depend on a linear combination of the inputs, `x · weights + intercept`:

- **Linear regression** (`RegressionModel::linear`) observes `Gaussian(x · w + b, 1)` and predicts the mean.
- **Logistic regression** (`RegressionModel::logistic`) observes `Bernoulli(sigmoid(x · w + b))` and predicts
  `true` where the probability is above 0.5.

Inputs are matrices of shape `[samples, features]`. A vector is taken to be samples of a single feature, and outputs
can be a vector or a `[samples, 1]` column.

```rust,ignore
{{#include ../../keanu/tests/book.rs:regression}}
```

## Regularization and priors

The weights and intercept have priors, which the regularization chooses:

| Regularization | Prior | Default fitter |
|---|---|---|
| `None` (the default) | Gaussian | maximum likelihood, so the priors have no effect |
| `Ridge` (L2) | Gaussian | maximum a posteriori |
| `Lasso` (L1) | Laplace | maximum a posteriori |

The priors default to mean 0 and scale 1 (the sigma of a Gaussian, the beta of a Laplace). Set them with
`prior_on_weights` (means and scales broadcast to the weights' shape `[features, 1]`), `prior_on_intercept` or
`prior_on_weights_and_intercept`. Smaller scales shrink the fit more towards the means.

Lasso optima often sit exactly on the Laplace prior's kink at 0. The gradient optimizers approach such a point but
can stop just short of it.

## Fitting by sampling

Instead of optimizing, a model can be fitted by sampling its posterior: `sampling(SamplingFitting::new(sampler, count,
seed))` runs the chain, sets the weights and intercept to the most probable sample, and keeps the samples (see
`samples()`) so you can look at the posterior's spread.

```rust,ignore
{{#include ../../keanu/tests/book.rs:regression_sampling}}
```

`fitter(Fitter::MaxAPosteriori(...))` and `Fitter::MaxLikelihood(...)` choose an optimizer explicitly. A `Fitter`
can also fit any `Model`, not only regressions.

## Scores

`coefficient_of_determination` (R²) scores continuous predictions and `accuracy` scores boolean ones.

## Coming from Java

`RegressionModel.withTrainingData(x, y)...build()` becomes `RegressionModel::linear(&x, &y)?...fit()?` (or
`logistic` for boolean outputs; `build()` gives an unfitted model). `getWeightVertex().getValue()` becomes
`weights()`, `getInterceptVertex().getValue().scalar()` becomes `intercept()`, and `SamplingModelFitting` becomes
`SamplingFitting`, whose samples are on the model rather than on the fitting object. Fitting is deterministic: the
weights and intercept start at their prior means rather than at a random draw from the prior.
