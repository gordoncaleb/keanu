//! Probabilistic nodes: a distribution whose parameters are the node's parents.

use crate::error::{GraphError, Result};
use crate::value::{DType, Value};
use keanu_distributions as d;
use keanu_distributions::KeanuRng;
use keanu_tensor::{DoubleTensor, Shape};
use std::sync::Arc;

/// The set of values a distribution gives positive density (or mass), which decides how a gradient based algorithm
/// maps the node to unconstrained space.
///
/// Bounds are tensors that broadcast with the node's value, so a distribution whose bounds are parameters (Uniform,
/// Triangular, Pareto) reports them element-wise. Bounds are reported as the parameters give them, valid or not (a
/// Uniform with `x_max <= x_min` reports that interval anyway). Whether a bound itself is in the support (Uniform's
/// `[x_min, x_max)`) is not recorded: it has probability zero for a continuous distribution.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Support {
    /// Every real number.
    Real,
    /// Real numbers greater than the lower bound.
    GreaterThan(DoubleTensor),
    /// Real numbers between the lower and the upper bound.
    Interval(DoubleTensor, DoubleTensor),
    /// Non-negative vectors summing to 1, over the value's last dimension.
    Simplex,
    /// Strictly increasing vectors, over the value's last dimension.
    Ordered,
    /// Lower triangular Cholesky factors of correlation matrices, over the value's last two dimensions: a positive
    /// diagonal and rows of unit length.
    CorrelationCholesky,
    /// Angles, for distributions of directions such as the von Mises: a value and the same value plus any multiple
    /// of 2π are the same point, so the value is mapped to unconstrained space with a circular transform.
    Circular,
    /// Integer or boolean values: the node is not continuous, so it has no unconstrained form.
    Discrete,
}

impl Support {
    /// Real numbers greater than a scalar lower bound.
    pub fn greater_than(lower: f64) -> Self {
        Support::GreaterThan(DoubleTensor::scalar(lower))
    }

    /// Real numbers between scalar lower and upper bounds.
    pub fn interval(lower: f64, upper: f64) -> Self {
        Support::Interval(DoubleTensor::scalar(lower), DoubleTensor::scalar(upper))
    }
}

/// The distribution of a probabilistic node. The node's parents are the parameters, in the order listed.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum DistributionKind {
    /// Parameters: `mu`, `sigma`. Support: real.
    Gaussian,
    /// Parameters: `sigma`. Support: greater than 0.
    HalfGaussian,
    /// Parameters: `location`, `scale`. Support: real.
    Cauchy,
    /// Parameters: `scale`. Support: greater than 0.
    HalfCauchy,
    /// Parameters: `alpha`, `beta`. Support: the interval [0, 1].
    Beta,
    /// Parameters: `k (i64)`. Support: greater than 0.
    ChiSquared,
    /// Parameters: `concentration`; the value has the concentration's shape. Support: the simplex over the last
    /// dimension.
    Dirichlet,
    /// Parameters: `rate`. Support: greater than 0.
    Exponential,
    /// Parameters: `theta (scale)`, `k (shape)`. Support: greater than 0.
    Gamma,
    /// Parameters: `alpha`, `beta`. Support: greater than 0.
    InverseGamma,
    /// Parameters: `mu`, `beta`. Support: real.
    Laplace,
    /// Parameters: `mu`, `s`. Support: real.
    Logistic,
    /// Parameters: `mu`, `sigma`. Support: greater than 0.
    LogNormal,
    /// Parameters: `mu`, `covariance`; the value's last dimension is the event dimension. Support: real.
    MultivariateGaussian,
    /// Parameters: `location` (the minimum value), `scale` (the shape). Support: greater than `location`.
    Pareto,
    /// Parameters: `v (i64)`. Support: real.
    StudentT,
    /// Parameters: `x_min`, `x_max`, `c`. Support: the interval [`x_min`, `x_max`].
    Triangular,
    /// Parameters: `x_min`, `x_max`. Support: the interval [`x_min`, `x_max`).
    Uniform,
    /// No parameters: a kernel density estimate from fixed samples. Support: real.
    Kde(Arc<d::Kde>),
    /// Parameters: `probability of true` -> bool. Support: discrete.
    Bernoulli,
    /// Parameters: `p`, `n (i64)` -> i64. Support: discrete.
    Binomial,
    /// Parameters: `probability of each category...` -> i64 category index. Support: discrete.
    Categorical,
    /// Parameters: `p` -> i64. Support: discrete.
    Geometric,
    /// Parameters: `n (i64)`, `p` -> i64. Support: discrete.
    Multinomial,
    /// Parameters: `mu` -> i64. Support: discrete.
    Poisson,
    /// Parameters: `x_min (i64)`, `x_max (i64)` -> i64 in [x_min, x_max). Support: discrete.
    UniformInt,
    /// A user-defined distribution; its parameters are whatever nodes it was given. Support: whatever
    /// [`CustomDistribution::support`] returns.
    Custom(Arc<dyn CustomDistribution>),
}

/// A user-defined distribution for a probabilistic node, the equivalent of implementing a custom probabilistic
/// vertex in Java.
///
/// Only [`log_prob`](Self::log_prob) is required. Implement [`log_prob_gradient`](Self::log_prob_gradient) to use
/// the node with gradient based algorithms (NUTS, gradient optimizers), and [`sample`](Self::sample) to sample it
/// from its prior (forward sampling, prior proposals, evaluating a graph without a value for it).
pub trait CustomDistribution: Send + Sync + std::fmt::Debug {
    /// The type of the node's value. f64 by default.
    fn value_type(&self) -> DType {
        DType::F64
    }

    /// The log probability of `value` (summed over its elements) given the parameters' values.
    fn log_prob(&self, value: &Value, parameters: &[&Value]) -> std::result::Result<f64, String>;

    /// The gradient of [`log_prob`](Self::log_prob) with respect to the value (f64 values only) and each f64
    /// parameter, shaped like the value and the parameters. `None` (the default) means the distribution has no
    /// gradient.
    fn log_prob_gradient(
        &self,
        value: &Value,
        parameters: &[&Value],
    ) -> std::result::Result<Option<CustomGradient>, String> {
        let _ = (value, parameters);
        Ok(None)
    }

    /// Draws a value of the given shape. By default the distribution can't be sampled.
    fn sample(
        &self,
        shape: &[usize],
        parameters: &[&Value],
        rng: &mut KeanuRng,
    ) -> std::result::Result<Value, String> {
        let _ = (shape, parameters, rng);
        Err("this custom distribution cannot be sampled".into())
    }

    /// The support of the distribution given the parameters' values. By default every real number for f64 values
    /// and [`Support::Discrete`] for i64 and bool values. Override it for a bounded distribution so that gradient
    /// based algorithms keep the node inside its support.
    fn support(&self, parameters: &[&Value]) -> std::result::Result<Support, String> {
        let _ = parameters;
        Ok(match self.value_type() {
            DType::F64 => Support::Real,
            DType::I64 | DType::Bool => Support::Discrete,
        })
    }
}

/// The gradient of a [`CustomDistribution`]'s log probability.
#[derive(Debug, Clone, Default)]
pub struct CustomGradient {
    /// With respect to the value, if it is f64 and differentiable.
    pub value: Option<keanu_tensor::DoubleTensor>,
    /// With respect to each parameter, in order (`None` where there is no gradient).
    pub parameters: Vec<Option<keanu_tensor::DoubleTensor>>,
}

/// Per-parameter log prob gradients, inline for up to four parameters.
pub(crate) type ParameterGradients = smallvec::SmallVec<[Option<DoubleTensor>; 4]>;

/// A node's log prob gradient with respect to its value (None for non-f64 values and distributions without one)
/// and to each parameter (None where there is no derivative).
pub(crate) type LogProbGradientParts = (Option<DoubleTensor>, ParameterGradients);

/// The first bracketed tokens if there are any (an optional override), else the second.
macro_rules! override_or {
    ([] [$($default:tt)*]) => { $($default)* };
    ([$($given:tt)+] [$($default:tt)*]) => { $($given)+ };
}

/// The tensor of a value of the given [`DType`] variant.
macro_rules! tensor {
    (F64, $value:expr) => {
        $value.f64()?
    };
    (I64, $value:expr) => {
        $value.i64()?
    };
    (Bool, $value:expr) => {
        $value.bool()?
    };
}

/// A log prob gradient with respect to a value or parameter of the given [`DType`] variant: only f64 ones have one.
macro_rules! gradient {
    (F64, $gradient:expr) => {
        Some($gradient)
    };
    ($dtype:ident, $gradient:expr) => {
        None
    };
}

/// Describes each distribution with a fixed parameter list once, as
/// `Kind "name" (parameter: dtype, ...) -> value dtype => constructor`, and generates [`DistributionKind`]'s name,
/// types, sampling, log probability and gradient from the table (the remaining kinds are written out here).
///
/// The parameters are bound to references to their tensors, which the distribution borrows (nothing is copied). By
/// default a distribution is sampled with `sample(shape, rng)`, and its gradient is the `d_log_prob` field named after each f64 parameter, and `x`
/// for an f64 value. A row can override either: `sample = f` is called as `f(&distribution, shape, rng)` and
/// `gradient = f` as `f(&distribution, x)` (both non-capturing closures or functions). `fused = f`, called as
/// `f(&distribution, x)`, computes the summed log probability and the gradient together for
/// `log_prob_and_gradient`; it must return exactly what the separate methods do. Every row states its support as
/// `support = expression`, an expression of the parameters' tensors (bound to the parameters' names) giving a
/// [`Support`].
macro_rules! distributions {
    ($(
        $kind:ident $name:literal ($($param:ident: $ptype:ident),+) -> $vtype:ident => $build:expr,
        support = $support:expr
        $(, sample = $sample:expr)? $(, gradient = $gradient:expr)? $(, fused = $fused:expr)?;
    )+) => {
        impl DistributionKind {
            /// The parameter types, or None for a variable number of f64 parameters (Categorical).
            pub(crate) fn parameter_types(&self) -> Option<&'static [DType]> {
                match self {
                    $(DistributionKind::$kind => Some(&[$(DType::$ptype),+]),)+
                    DistributionKind::Kde(_) => Some(&[]),
                    DistributionKind::Categorical | DistributionKind::Custom(_) => None,
                }
            }

            /// The type of the node's value.
            pub fn value_type(&self) -> DType {
                match self {
                    $(DistributionKind::$kind => DType::$vtype,)+
                    DistributionKind::Kde(_) => DType::F64,
                    DistributionKind::Categorical => DType::I64,
                    DistributionKind::Custom(c) => c.value_type(),
                }
            }

            pub(crate) fn name(&self) -> &'static str {
                match self {
                    $(DistributionKind::$kind => $name,)+
                    DistributionKind::Kde(_) => "kde",
                    DistributionKind::Categorical => "categorical",
                    DistributionKind::Custom(_) => "custom",
                }
            }

            /// The distribution's support given its parameters' values (see [`Support`]). Bounds that are parameters
            /// are the parameters' values.
            pub fn support(&self, parameters: &[&Value]) -> Result<Support> {
                Ok(match self {
                    $(DistributionKind::$kind => {
                        let &[$($param),+] = parameters else { return Err(self.wrong_parameters()) };
                        $(let $param = tensor!($ptype, $param);)+
                        let _ = ($($param,)+);
                        $support
                    })+
                    DistributionKind::Kde(_) => Support::Real,
                    DistributionKind::Categorical => Support::Discrete,
                    DistributionKind::Custom(c) => c.support(parameters).map_err(GraphError::Custom)?,
                })
            }

            /// Draws a value of the given shape from the distribution given its parameters' values.
            fn sample(&self, shape: &[usize], parameters: &[&Value], rng: &mut KeanuRng) -> Result<Value> {
                Ok(match self {
                    $(DistributionKind::$kind => {
                        let &[$($param),+] = parameters else { return Err(self.wrong_parameters()) };
                        $(let $param = tensor!($ptype, $param);)+
                        let dist = $build;
                        Value::$vtype(override_or!(
                            [$({
                                let sample: fn(&_, &[usize], &mut KeanuRng) -> d::Result<_> = $sample;
                                sample(&dist, shape, rng)
                            })?]
                            [dist.sample(shape, rng)]
                        )?)
                    })+
                    DistributionKind::Kde(kde) => {
                        let count: usize = shape.iter().product();
                        let signed: Vec<isize> = shape.iter().map(|&s| s as isize).collect();
                        Value::F64(kde.sample(count, rng).reshape(&signed)?)
                    }
                    DistributionKind::Categorical => {
                        Value::I64(d::Categorical::new(f64_tensors(parameters)?)?.sample(shape, rng)?)
                    }
                    DistributionKind::Custom(c) => {
                        let value = c
                            .sample(shape, parameters, rng)
                            .map_err(GraphError::Custom)?;
                        if value.dtype() != c.value_type() || value.shape() != shape {
                            return Err(GraphError::Custom(format!(
                                "sample returned a {:?} value of shape {:?}, expected {:?} of shape {shape:?}",
                                value.dtype(),
                                value.shape(),
                                c.value_type()
                            )));
                        }
                        value
                    }
                })
            }

            /// The log probability of `value` given the parameters' values, summed over its elements.
            pub(crate) fn log_prob(&self, value: &Value, parameters: &[&Value]) -> Result<f64> {
                Ok(match self {
                    $(DistributionKind::$kind => {
                        let &[$($param),+] = parameters else { return Err(self.wrong_parameters()) };
                        $(let $param = tensor!($ptype, $param);)+
                        $build.log_prob(tensor!($vtype, value))?.total()
                    })+
                    DistributionKind::Kde(kde) => kde.log_prob(value.f64()?).sum(),
                    DistributionKind::Categorical => {
                        d::Categorical::new(f64_tensors(parameters)?)?.log_prob(value.i64()?)?.sum()
                    }
                    DistributionKind::Custom(c) => c.log_prob(value, parameters).map_err(GraphError::Custom)?,
                })
            }

            /// The gradient of [`Self::log_prob`] with respect to `value` and each parameter, following the Java
            /// vertices' `dLogProb`.
            pub(crate) fn log_prob_gradient(
                &self,
                value: &Value,
                parameters: &[&Value],
            ) -> Result<LogProbGradientParts> {
                Ok(match self {
                    $(DistributionKind::$kind => {
                        let &[$($param),+] = parameters else { return Err(self.wrong_parameters()) };
                        $(let $param = tensor!($ptype, $param);)+
                        let dist = $build;
                        let x = tensor!($vtype, value);
                        distribution_gradient!(dist, x, $vtype, [$($param: $ptype),+], [$($gradient)?])
                    })+
                    DistributionKind::Kde(kde) => (Some(kde.d_log_prob(value.f64()?)?.x), smallvec::smallvec![]),
                    DistributionKind::Categorical => no_gradient(parameters.len()),
                    DistributionKind::Custom(c) => {
                        match c.log_prob_gradient(value, parameters).map_err(GraphError::Custom)? {
                            Some(g) => {
                                let mut gradients: ParameterGradients = g.parameters.into_iter().collect();
                                gradients.resize(parameters.len(), None);
                                (g.value, gradients)
                            }
                            None => no_gradient(parameters.len()),
                        }
                    }
                })
            }

            /// [`Self::log_prob`] and, unless it is impossible (-infinity or NaN), [`Self::log_prob_gradient`],
            /// sharing the work of reading the parameters and building the distribution. Each is exactly what the
            /// separate method returns.
            pub(crate) fn log_prob_and_gradient(
                &self,
                value: &Value,
                parameters: &[&Value],
            ) -> Result<(f64, Option<LogProbGradientParts>)> {
                Ok(match self {
                    $(DistributionKind::$kind => {
                        let &[$($param),+] = parameters else { return Err(self.wrong_parameters()) };
                        $(let $param = tensor!($ptype, $param);)+
                        let dist = $build;
                        let x = tensor!($vtype, value);
                        override_or!(
                            [$({
                                let fused: fn(&_, &_) -> Result<(f64, LogProbGradientParts)> = $fused;
                                let (log_prob, gradient) = fused(&dist, x)?;
                                (log_prob, (!is_impossible(log_prob)).then_some(gradient))
                            })?]
                            [{
                                let log_prob = dist.log_prob(x)?.total();
                                if is_impossible(log_prob) {
                                    return Ok((log_prob, None));
                                }
                                let gradient = distribution_gradient!(
                                    dist, x, $vtype, [$($param: $ptype),+], [$($gradient)?]
                                );
                                (log_prob, Some(gradient))
                            }]
                        )
                    })+
                    DistributionKind::Kde(_)
                    | DistributionKind::Categorical
                    | DistributionKind::Custom(_) => {
                        let log_prob = self.log_prob(value, parameters)?;
                        if is_impossible(log_prob) {
                            return Ok((log_prob, None));
                        }
                        (log_prob, Some(self.log_prob_gradient(value, parameters)?))
                    }
                })
            }
        }
    };
}

/// The log prob gradient parts of a built distribution `$dist` at `$x`: the row's `gradient` override if it has
/// one, else the `d_log_prob` field named after each f64 parameter (and `x` for an f64 value).
macro_rules! distribution_gradient {
    ($dist:ident, $x:ident, $vtype:ident, [$($param:ident: $ptype:ident),+], [$($gradient:expr)?]) => {
        override_or!(
            [$({
                let gradient: fn(&_, &_) -> Result<LogProbGradientParts> = $gradient;
                gradient(&$dist, $x)?
            })?]
            [{
                let g = $dist.d_log_prob($x)?;
                (gradient!($vtype, g.x), smallvec::smallvec![$(gradient!($ptype, g.$param)),+])
            }]
        )
    };
}

/// Whether a log probability makes the state impossible: -infinity or NaN.
pub(crate) fn is_impossible(log_prob: f64) -> bool {
    log_prob == f64::NEG_INFINITY || log_prob.is_nan()
}

distributions! {
    Gaussian "gaussian" (mu: F64, sigma: F64) -> F64 => d::Gaussian::new(mu, sigma),
        support = Support::Real,
        fused = gaussian_log_prob_and_gradient;
    HalfGaussian "half_gaussian" (sigma: F64) -> F64 => d::HalfGaussian::new(sigma),
        support = Support::greater_than(0.0);
    Cauchy "cauchy" (location: F64, scale: F64) -> F64 => d::Cauchy::new(location, scale),
        support = Support::Real;
    HalfCauchy "half_cauchy" (scale: F64) -> F64 => d::HalfCauchy::new(scale),
        support = Support::greater_than(0.0);
    Beta "beta" (alpha: F64, beta: F64) -> F64 => d::Beta::standard(alpha, beta),
        support = Support::interval(0.0, 1.0);
    ChiSquared "chi_squared" (k: I64) -> F64 => d::ChiSquared::new(k),
        support = Support::greater_than(0.0);
    Dirichlet "dirichlet" (concentration: F64) -> F64 => d::Dirichlet::new(concentration),
        support = Support::Simplex,
        sample = |dist: &d::Dirichlet<'_>, _, rng| dist.sample(rng);
    Exponential "exponential" (rate: F64) -> F64 => d::Exponential::new(rate),
        support = Support::greater_than(0.0);
    Gamma "gamma" (theta: F64, k: F64) -> F64 => d::Gamma::new(theta, k),
        support = Support::greater_than(0.0);
    InverseGamma "inverse_gamma" (alpha: F64, beta: F64) -> F64 => d::InverseGamma::new(alpha, beta),
        support = Support::greater_than(0.0);
    Laplace "laplace" (mu: F64, beta: F64) -> F64 => d::Laplace::new(mu, beta),
        support = Support::Real;
    Logistic "logistic" (mu: F64, s: F64) -> F64 => d::Logistic::new(mu, s),
        support = Support::Real;
    LogNormal "log_normal" (mu: F64, sigma: F64) -> F64 => d::LogNormal::new(mu, sigma),
        support = Support::greater_than(0.0);
    MultivariateGaussian "multivariate_gaussian" (mu: F64, covariance: F64) -> F64 =>
        d::MultivariateGaussian::new(mu, covariance),
        support = Support::Real;
    Pareto "pareto" (location: F64, scale: F64) -> F64 => d::Pareto::new(location, scale),
        support = Support::GreaterThan(location.clone());
    StudentT "student_t" (v: I64) -> F64 => d::StudentT::new(v),
        support = Support::Real;
    Triangular "triangular" (x_min: F64, x_max: F64, c: F64) -> F64 => d::Triangular::new(x_min, x_max, c),
        support = Support::Interval(x_min.clone(), x_max.clone());
    Uniform "uniform" (x_min: F64, x_max: F64) -> F64 => d::Uniform::new(x_min, x_max),
        support = Support::Interval(x_min.clone(), x_max.clone()),
        gradient = uniform_gradient;
    Bernoulli "bernoulli" (prob_true: F64) -> Bool => d::Bernoulli::new(prob_true),
        support = Support::Discrete;
    Binomial "binomial" (p: F64, n: I64) -> I64 => d::Binomial::new(p, n),
        support = Support::Discrete;
    Geometric "geometric" (p: F64) -> I64 => d::Geometric::new(p),
        support = Support::Discrete;
    Multinomial "multinomial" (n: I64, p: F64) -> I64 => d::Multinomial::new(n, p)?,
        support = Support::Discrete;
    Poisson "poisson" (mu: F64) -> I64 => d::Poisson::new(mu),
        support = Support::Discrete;
    UniformInt "uniform_int" (x_min: I64, x_max: I64) -> I64 => d::UniformInt::new(x_min, x_max),
        support = Support::Discrete,
        gradient = |_, _| Ok(no_gradient(2));
}

impl DistributionKind {
    /// The error for a parameter list that doesn't match the distribution (checked when the node is built).
    fn wrong_parameters(&self) -> GraphError {
        GraphError::InvalidNode {
            op: self.name(),
            message: format!("needs parameters of types {:?}", self.parameter_types()),
        }
    }
}

/// The sum of an element-wise log probability (distributions that reduce over their event dimension return the
/// sum already).
trait Total {
    fn total(self) -> f64;
}

impl Total for DoubleTensor {
    fn total(self) -> f64 {
        self.sum()
    }
}

impl Total for f64 {
    fn total(self) -> f64 {
        self
    }
}

/// Every parameter's f64 tensor (Categorical's probabilities).
fn f64_tensors<'v>(parameters: &[&'v Value]) -> Result<Vec<&'v DoubleTensor>> {
    parameters.iter().map(|p| p.f64()).collect()
}

/// No gradient with respect to the value or any of `parameters` parameters.
fn no_gradient(parameters: usize) -> LogProbGradientParts {
    (None, smallvec::smallvec![None; parameters])
}

/// The Gaussian log probability and gradient in one pass.
fn gaussian_log_prob_and_gradient(
    gaussian: &d::Gaussian<'_>,
    x: &DoubleTensor,
) -> Result<(f64, LogProbGradientParts)> {
    let (log_prob, g) = gaussian.log_prob_and_d_log_prob(x)?;
    Ok((
        log_prob.sum(),
        (Some(g.x), smallvec::smallvec![Some(g.mu), Some(g.sigma)]),
    ))
}

/// As UniformVertex: zero inside the support, -inf above x_max and +inf at or below x_min, and no gradient with
/// respect to the bounds.
fn uniform_gradient(uniform: &d::Uniform<'_>, x: &DoubleTensor) -> Result<LogProbGradientParts> {
    let dx = x
        .zip_with(&*uniform.x_max, |x, max| {
            if x > max { f64::NEG_INFINITY } else { 0.0 }
        })?
        .zip_with(&x.le(&uniform.x_min)?, |d, below| {
            if below { f64::INFINITY } else { d }
        })?;
    Ok((Some(dx), smallvec::smallvec![None, None]))
}

/// A probabilistic node's distribution and the shape of its value.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DistributionOp {
    /// The distribution.
    pub kind: DistributionKind,
    /// The shape of the node's value.
    pub shape: Shape,
}

impl DistributionOp {
    /// The node's value shape given the requested shape and the parameter shapes: the broadcast of all of them,
    /// except for distributions whose value shape is determined by a parameter.
    pub(crate) fn value_shape(
        kind: &DistributionKind,
        requested: Option<&[usize]>,
        parameters: &[(DType, &[usize])],
    ) -> Result<Shape> {
        let invalid = |message: String| GraphError::InvalidNode {
            op: kind.name(),
            message,
        };
        if let DistributionKind::Custom(_) = kind {
            return Ok(Shape::from_slice(requested.unwrap_or(&[])));
        }
        match kind.parameter_types() {
            Some(types) => {
                if parameters.len() != types.len()
                    || parameters.iter().zip(types).any(|((t, _), e)| t != e)
                {
                    return Err(invalid(format!("needs parameters of types {types:?}")));
                }
            }
            None => {
                if parameters.is_empty()
                    || parameters
                        .iter()
                        .any(|(t, s)| *t != DType::F64 || *s != parameters[0].1)
                {
                    return Err(invalid(
                        "needs at least one f64 probability, all of the same shape".into(),
                    ));
                }
            }
        }
        match kind {
            DistributionKind::Dirichlet => Ok(Shape::from_slice(parameters[0].1)),
            DistributionKind::MultivariateGaussian => {
                let (mu, covariance) = (parameters[0].1, parameters[1].1);
                if mu.is_empty() || covariance.len() < 2 {
                    return Err(invalid(
                        "mu must be a vector and the covariance a matrix".into(),
                    ));
                }
                let n = mu[mu.len() - 1];
                let c = covariance.len();
                if covariance[c - 1] != n || covariance[c - 2] != n {
                    return Err(invalid(format!(
                        "covariance {covariance:?} does not match mu {mu:?}"
                    )));
                }
                let batch =
                    keanu_tensor::broadcast_shape(&mu[..mu.len() - 1], &covariance[..c - 2])?;
                let mut shape = batch;
                shape.push(n);
                Ok(match requested {
                    Some(r) => keanu_tensor::broadcast_shape(r, &shape)?,
                    None => shape,
                })
            }
            DistributionKind::Kde(_) => Ok(Shape::from_slice(requested.unwrap_or(&[]))),
            _ => {
                let mut shape = Shape::from_slice(requested.unwrap_or(&[]));
                for (_, p) in parameters {
                    shape = keanu_tensor::broadcast_shape(&shape, p)?;
                }
                Ok(shape)
            }
        }
    }

    /// Draws a value from the distribution given its parameters' values.
    pub(crate) fn sample(&self, parameters: &[&Value], rng: &mut KeanuRng) -> Result<Value> {
        self.kind.sample(self.shape.as_slice(), parameters, rng)
    }
}
