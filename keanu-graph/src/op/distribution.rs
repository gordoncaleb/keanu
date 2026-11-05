//! Probabilistic nodes: a distribution whose parameters are the node's parents.

use super::mixture;
use super::wrappers::{self, Wrapper};
use crate::error::{GraphError, Result};
use crate::value::{DType, Value};
use keanu_distributions as d;
use keanu_distributions::KeanuRng;
use keanu_tensor::{DoubleTensor, Shape};
use std::borrow::Cow;
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
    /// Real numbers less than the upper bound.
    LessThan(DoubleTensor),
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

    /// Real numbers less than a scalar upper bound.
    pub fn less_than(upper: f64) -> Self {
        Support::LessThan(DoubleTensor::scalar(upper))
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
    /// Parameters: `eta`; the value is a K × K lower triangular Cholesky factor of a correlation matrix (or a batch
    /// of them, shape [..., K, K], with `eta` broadcasting to the batch shape), K set by the node's shape. Support:
    /// correlation Cholesky factors. See [`keanu_distributions::LkjCholesky`].
    LkjCholesky,
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
    /// Parameters: `alpha` (shape), `beta` (scale). Support: greater than 0.
    Weibull,
    /// Parameters: `mu` (mean direction), `kappa` (concentration). Support: circular (angles; the density is
    /// periodic with period 2π).
    VonMises,
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
    /// Parameters: `mu` (mean), `alpha` (dispersion) -> i64, with variance mu + mu²/alpha. Support: discrete.
    NegativeBinomial,
    /// Parameters: `eta`, `cutpoints` (strictly increasing along the last axis, K - 1 of them) -> i64 category index
    /// in 0..K. The value's shape is the broadcast of eta's shape and the cutpoints' leading axes. Support:
    /// discrete.
    OrderedLogistic,
    /// Parameters: `x_min (i64)`, `x_max (i64)` -> i64 in [x_min, x_max). Support: discrete.
    UniformInt,
    /// A user-defined distribution; its parameters are whatever nodes it was given. Support: whatever
    /// [`CustomDistribution::support`] returns.
    Custom(Arc<dyn CustomDistribution>),
    /// The base distribution restricted to `[lower, upper]` and renormalised by the probability it gives that
    /// range. Parameters: the base's parameters, then `lower`, `upper` (f64, either may be infinite). Support: the
    /// base's support intersected with the bounds (discrete for a discrete base). See
    /// [`GraphBuilder::truncated`](crate::GraphBuilder::truncated).
    Truncated(Arc<DistributionKind>),
    /// The base distribution with values at or beyond the bounds recorded as the bounds: a value equal to `lower`
    /// has the probability P(X <= lower) and a value equal to `upper` the probability P(X >= upper). Parameters:
    /// the base's parameters, then `lower`, `upper` (f64, either may be infinite). Support: as for
    /// [`Truncated`](Self::Truncated). See [`GraphBuilder::censored`](crate::GraphBuilder::censored).
    Censored(Arc<DistributionKind>),
    /// A mixture of C distributions of the component kind: density Σ_c w_c p_c(x). Parameters: the component's
    /// parameters, each with a trailing component axis of length C (or 1, or no axes at all, to share the value
    /// between components), then `weights` (non-negative and summing to 1 along their last axis, of length C).
    /// The component is an element-wise univariate kind (a distribution of one value per element, or a truncated
    /// or censored one); the value type is the component's. Support: the component's support widened to hold
    /// every component's. See [`GraphBuilder::mixture`](crate::GraphBuilder::mixture).
    Mixture(Arc<DistributionKind>),
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
///
/// A univariate distribution with `log_cdf`, `log_ccdf`, `d_log_cdf` and `d_log_ccdf` methods (which can then be
/// truncated or censored) says so with `cdf = inverse` if it also has `inverse_cdf` and `inverse_ccdf` (so a
/// truncation is sampled by inverting the CDF), or `cdf = rejection` if not (so it is sampled by rejection). A
/// distribution of one value per element whose log density and gradient are element-wise but which has no CDF says
/// `cdf = none`: it can't be truncated or censored, but it can be a mixture component.
macro_rules! distributions {
    ($(
        $kind:ident $name:literal ($($param:ident: $ptype:ident),+) -> $vtype:ident => $build:expr,
        support = $support:expr
        $(, cdf = $cdf:ident)?
        $(, sample = $sample:expr)? $(, gradient = $gradient:expr)? $(, fused = $fused:expr)?;
    )+) => {
        impl DistributionKind {
            /// The parameter types, or None for a variable number of f64 parameters (Categorical).
            pub(crate) fn parameter_types(&self) -> Option<Cow<'static, [DType]>> {
                match self {
                    $(DistributionKind::$kind => Some(Cow::Borrowed(&[$(DType::$ptype),+])),)+
                    DistributionKind::Kde(_) => Some(Cow::Borrowed(&[])),
                    DistributionKind::Categorical | DistributionKind::Custom(_) => None,
                    DistributionKind::Truncated(base) | DistributionKind::Censored(base) => {
                        let mut types = base.parameter_types()?.into_owned();
                        types.extend([DType::F64; wrappers::OWN_PARAMETERS]);
                        Some(Cow::Owned(types))
                    }
                    DistributionKind::Mixture(component) => {
                        let mut types = component.parameter_types()?.into_owned();
                        types.extend([DType::F64; mixture::OWN_PARAMETERS]);
                        Some(Cow::Owned(types))
                    }
                }
            }

            /// The parameter names, in parent order (e.g. `["mu", "sigma"]` for a Gaussian), or None for a variable
            /// number of parameters (Categorical, Custom). A kind built around another one has its base's names
            /// followed by its own (`lower` and `upper` for a truncated or censored kind, `weights` for a mixture).
            pub fn parameter_names(&self) -> Option<Cow<'static, [&'static str]>> {
                match self {
                    $(DistributionKind::$kind => Some(Cow::Borrowed(&[$(stringify!($param)),+])),)+
                    DistributionKind::Kde(_) => Some(Cow::Borrowed(&[])),
                    DistributionKind::Categorical | DistributionKind::Custom(_) => None,
                    DistributionKind::Truncated(base) | DistributionKind::Censored(base) => {
                        let mut names = base.parameter_names()?.into_owned();
                        names.extend(["lower", "upper"]);
                        Some(Cow::Owned(names))
                    }
                    DistributionKind::Mixture(component) => {
                        let mut names = component.parameter_names()?.into_owned();
                        names.push("weights");
                        Some(Cow::Owned(names))
                    }
                }
            }

            /// Whether the distribution is univariate and has a CDF, so that it can be truncated or censored.
            pub(crate) fn has_cdf(&self) -> bool {
                match self {
                    $(DistributionKind::$kind => has_cdf!([$($cdf)?]),)+
                    _ => false,
                }
            }

            /// Whether the distribution has one value per element and an element-wise log density and gradient
            /// (see [`Self::univariate`]), so that it can be a mixture component.
            pub(crate) fn is_elementwise(&self) -> bool {
                match self {
                    $(DistributionKind::$kind => is_elementwise!([$($cdf)?]),)+
                    DistributionKind::Truncated(_) | DistributionKind::Censored(_) => true,
                    _ => false,
                }
            }

            /// `function` of an element-wise distribution at `x`, a value of the distribution's type, element-wise
            /// and broadcast with the parameters; with `gradient`, also its derivatives with respect to x and each
            /// parameter, element-wise. The log density works for every [element-wise](Self::is_elementwise) kind,
            /// the CDF and survival function only for those [with a CDF](Self::has_cdf).
            pub(crate) fn univariate(
                &self,
                function: Univariate,
                x: &Value,
                parameters: &[&Value],
                gradient: bool,
            ) -> Result<(DoubleTensor, Option<LogProbGradientParts>)> {
                Ok(match self {
                    $(DistributionKind::$kind => univariate!(
                        [$($cdf)?], self, function, x, parameters, gradient, $vtype,
                        [$($param: $ptype),+], [$($gradient)?], $build
                    ),)+
                    DistributionKind::Truncated(base) if function == Univariate::Density => {
                        wrappers::elementwise(Wrapper::Truncated, base, x, parameters, gradient)?
                    }
                    DistributionKind::Censored(base) if function == Univariate::Density => {
                        wrappers::elementwise(Wrapper::Censored, base, x, parameters, gradient)?
                    }
                    _ => return Err(self.no_cdf()),
                })
            }

            /// The inverse CDF at probabilities `p` (or the inverse survival function if `upper`), element-wise
            /// and broadcast with the parameters, if the distribution has one in closed form.
            pub(crate) fn inverse_cdf(
                &self,
                p: &DoubleTensor,
                upper: bool,
                parameters: &[&Value],
            ) -> Result<Option<DoubleTensor>> {
                Ok(match self {
                    $(DistributionKind::$kind => inverse_cdf!(
                        [$($cdf)?], self, p, upper, parameters, [$($param: $ptype),+], $build
                    ),)+
                    _ => None,
                })
            }

            /// The type of the node's value.
            pub fn value_type(&self) -> DType {
                match self {
                    $(DistributionKind::$kind => DType::$vtype,)+
                    DistributionKind::Kde(_) => DType::F64,
                    DistributionKind::Categorical => DType::I64,
                    DistributionKind::Custom(c) => c.value_type(),
                    DistributionKind::Truncated(base)
                    | DistributionKind::Censored(base)
                    | DistributionKind::Mixture(base) => base.value_type(),
                }
            }

            pub(crate) fn name(&self) -> &'static str {
                match self {
                    $(DistributionKind::$kind => $name,)+
                    DistributionKind::Kde(_) => "kde",
                    DistributionKind::Categorical => "categorical",
                    DistributionKind::Custom(_) => "custom",
                    DistributionKind::Truncated(_) => "truncated",
                    DistributionKind::Censored(_) => "censored",
                    DistributionKind::Mixture(_) => "mixture",
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
                    DistributionKind::Truncated(base) | DistributionKind::Censored(base) => {
                        wrappers::support(base, parameters)?
                    }
                    DistributionKind::Mixture(component) => mixture::support(component, parameters)?,
                })
            }

            /// Draws a value of the given shape from the distribution given its parameters' values.
            pub(crate) fn sample(&self, shape: &[usize], parameters: &[&Value], rng: &mut KeanuRng) -> Result<Value> {
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
                    DistributionKind::Truncated(base) => wrappers::sample_truncated(base, shape, parameters, rng)?,
                    DistributionKind::Censored(base) => wrappers::sample_censored(base, shape, parameters, rng)?,
                    DistributionKind::Mixture(component) => mixture::sample(component, shape, parameters, rng)?,
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
                    DistributionKind::Truncated(base) => {
                        wrappers::log_prob(Wrapper::Truncated, base, value, parameters, false)?.0
                    }
                    DistributionKind::Censored(base) => {
                        wrappers::log_prob(Wrapper::Censored, base, value, parameters, false)?.0
                    }
                    DistributionKind::Mixture(component) => {
                        mixture::log_prob(component, value, parameters, false)?.0
                    }
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
                    DistributionKind::Truncated(base) => {
                        wrapper_gradient(Wrapper::Truncated, base, value, parameters)?
                    }
                    DistributionKind::Censored(base) => {
                        wrapper_gradient(Wrapper::Censored, base, value, parameters)?
                    }
                    DistributionKind::Mixture(component) => {
                        let (_, gradient) = mixture::log_prob(component, value, parameters, true)?;
                        gradient.unwrap_or_else(|| no_gradient(parameters.len()))
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
                    DistributionKind::Truncated(base) => {
                        let (log_prob, gradient) =
                            wrappers::log_prob(Wrapper::Truncated, base, value, parameters, true)?;
                        (log_prob, gradient.filter(|_| !is_impossible(log_prob)))
                    }
                    DistributionKind::Censored(base) => {
                        let (log_prob, gradient) =
                            wrappers::log_prob(Wrapper::Censored, base, value, parameters, true)?;
                        (log_prob, gradient.filter(|_| !is_impossible(log_prob)))
                    }
                    DistributionKind::Mixture(component) => {
                        let (log_prob, gradient) = mixture::log_prob(component, value, parameters, true)?;
                        (log_prob, gradient.filter(|_| !is_impossible(log_prob)))
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

/// `true` for a row with a CDF.
macro_rules! has_cdf {
    ([]) => {
        false
    };
    ([none]) => {
        false
    };
    ([$cdf:ident]) => {
        true
    };
}

/// `true` for an element-wise row: one with a `cdf` entry, even `none`.
macro_rules! is_elementwise {
    ([]) => {
        false
    };
    ([$cdf:ident]) => {
        true
    };
}

/// The body of [`DistributionKind::univariate`] for a row: an error without a CDF, else the requested function of
/// the built distribution and, if asked for, its gradient parts (from `d_log_prob`, or the row's gradient override,
/// for the log density, and from `d_log_cdf` / `d_log_ccdf` for the others).
macro_rules! univariate {
    ([], $kind:ident, $($rest:tt)*) => {
        return Err($kind.no_cdf())
    };
    ([none], $kind:ident, $function:ident, $x:ident, $parameters:ident, $want:ident, $vtype:ident,
     [$($param:ident: $ptype:ident),+], [$($gradient:expr)?], $build:expr) => {{
        if $function != Univariate::Density {
            return Err($kind.no_cdf());
        }
        let &[$($param),+] = $parameters else { return Err($kind.wrong_parameters()) };
        $(let $param = tensor!($ptype, $param);)+
        let dist = $build;
        let x = tensor!($vtype, $x);
        (
            dist.log_prob(x)?,
            if $want {
                Some(distribution_gradient!(dist, x, $vtype, [$($param: $ptype),+], [$($gradient)?]))
            } else {
                None
            },
        )
    }};
    ([$cdf:ident], $kind:ident, $function:ident, $x:ident, $parameters:ident, $want:ident, $vtype:ident,
     [$($param:ident: $ptype:ident),+], [$($gradient:expr)?], $build:expr) => {{
        let &[$($param),+] = $parameters else { return Err($kind.wrong_parameters()) };
        $(let $param = tensor!($ptype, $param);)+
        let dist = $build;
        let x = tensor!($vtype, $x);
        match $function {
            Univariate::Density => (
                dist.log_prob(x)?,
                if $want {
                    Some(distribution_gradient!(dist, x, $vtype, [$($param: $ptype),+], [$($gradient)?]))
                } else {
                    None
                },
            ),
            Univariate::Cdf => (
                dist.log_cdf(x)?,
                if $want {
                    let g = dist.d_log_cdf(x)?;
                    Some((gradient!($vtype, g.x), smallvec::smallvec![$(gradient!($ptype, g.$param)),+]))
                } else {
                    None
                },
            ),
            Univariate::Ccdf => (
                dist.log_ccdf(x)?,
                if $want {
                    let g = dist.d_log_ccdf(x)?;
                    Some((gradient!($vtype, g.x), smallvec::smallvec![$(gradient!($ptype, g.$param)),+]))
                } else {
                    None
                },
            ),
        }
    }};
}

/// The body of [`DistributionKind::inverse_cdf`] for a row.
macro_rules! inverse_cdf {
    ([inverse], $kind:ident, $p:ident, $upper:ident, $parameters:ident, [$($param:ident: $ptype:ident),+],
     $build:expr) => {{
        let &[$($param),+] = $parameters else { return Err($kind.wrong_parameters()) };
        $(let $param = tensor!($ptype, $param);)+
        let dist = $build;
        Some(if $upper { dist.inverse_ccdf($p)? } else { dist.inverse_cdf($p)? })
    }};
    ([$($other:ident)?], $($rest:tt)*) => {
        None
    };
}

/// A function of a univariate distribution, evaluated element-wise by [`DistributionKind::univariate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Univariate {
    /// The log density (or mass), ln p(x).
    Density,
    /// The log CDF, ln P(X <= x).
    Cdf,
    /// The log survival function, ln P(X > x).
    Ccdf,
}

/// A wrapper's log prob gradient (the separate method, which [`DistributionKind::log_prob_gradient`] returns even
/// where the log probability is impossible).
fn wrapper_gradient(
    wrapper: Wrapper,
    base: &DistributionKind,
    value: &Value,
    parameters: &[&Value],
) -> Result<LogProbGradientParts> {
    let (_, gradient) = wrappers::log_prob(wrapper, base, value, parameters, true)?;
    Ok(gradient.unwrap_or_else(|| no_gradient(parameters.len())))
}

/// Whether a log probability makes the state impossible: -infinity or NaN.
pub(crate) fn is_impossible(log_prob: f64) -> bool {
    log_prob == f64::NEG_INFINITY || log_prob.is_nan()
}

distributions! {
    Gaussian "gaussian" (mu: F64, sigma: F64) -> F64 => d::Gaussian::new(mu, sigma),
        support = Support::Real,
        cdf = inverse,
        fused = gaussian_log_prob_and_gradient;
    HalfGaussian "half_gaussian" (sigma: F64) -> F64 => d::HalfGaussian::new(sigma),
        support = Support::greater_than(0.0),
        cdf = none;
    Cauchy "cauchy" (location: F64, scale: F64) -> F64 => d::Cauchy::new(location, scale),
        support = Support::Real,
        cdf = inverse;
    HalfCauchy "half_cauchy" (scale: F64) -> F64 => d::HalfCauchy::new(scale),
        support = Support::greater_than(0.0),
        cdf = none;
    Beta "beta" (alpha: F64, beta: F64) -> F64 => d::Beta::standard(alpha, beta),
        support = Support::interval(0.0, 1.0),
        cdf = rejection;
    ChiSquared "chi_squared" (k: I64) -> F64 => d::ChiSquared::new(k),
        support = Support::greater_than(0.0),
        cdf = none;
    Dirichlet "dirichlet" (concentration: F64) -> F64 => d::Dirichlet::new(concentration),
        support = Support::Simplex,
        sample = |dist: &d::Dirichlet<'_>, _, rng| dist.sample(rng);
    Exponential "exponential" (rate: F64) -> F64 => d::Exponential::new(rate),
        support = Support::greater_than(0.0),
        cdf = inverse;
    Gamma "gamma" (theta: F64, k: F64) -> F64 => d::Gamma::new(theta, k),
        support = Support::greater_than(0.0),
        cdf = rejection;
    InverseGamma "inverse_gamma" (alpha: F64, beta: F64) -> F64 => d::InverseGamma::new(alpha, beta),
        support = Support::greater_than(0.0),
        cdf = none;
    Laplace "laplace" (mu: F64, beta: F64) -> F64 => d::Laplace::new(mu, beta),
        support = Support::Real,
        cdf = inverse;
    Logistic "logistic" (mu: F64, s: F64) -> F64 => d::Logistic::new(mu, s),
        support = Support::Real,
        cdf = inverse;
    LogNormal "log_normal" (mu: F64, sigma: F64) -> F64 => d::LogNormal::new(mu, sigma),
        support = Support::greater_than(0.0),
        cdf = inverse;
    LkjCholesky "lkj_cholesky" (eta: F64) -> F64 => d::LkjCholesky::new(eta),
        support = Support::CorrelationCholesky;
    MultivariateGaussian "multivariate_gaussian" (mu: F64, covariance: F64) -> F64 =>
        d::MultivariateGaussian::new(mu, covariance),
        support = Support::Real;
    Pareto "pareto" (location: F64, scale: F64) -> F64 => d::Pareto::new(location, scale),
        support = Support::GreaterThan(location.clone()),
        cdf = none;
    StudentT "student_t" (v: I64) -> F64 => d::StudentT::new(v),
        support = Support::Real,
        cdf = rejection;
    Triangular "triangular" (x_min: F64, x_max: F64, c: F64) -> F64 => d::Triangular::new(x_min, x_max, c),
        support = Support::Interval(x_min.clone(), x_max.clone()),
        cdf = none;
    Uniform "uniform" (x_min: F64, x_max: F64) -> F64 => d::Uniform::new(x_min, x_max),
        support = Support::Interval(x_min.clone(), x_max.clone()),
        cdf = inverse,
        gradient = uniform_gradient;
    Weibull "weibull" (alpha: F64, beta: F64) -> F64 => d::Weibull::new(alpha, beta),
        support = Support::greater_than(0.0),
        cdf = inverse;
    VonMises "von_mises" (mu: F64, kappa: F64) -> F64 => d::VonMises::new(mu, kappa),
        support = Support::Circular,
        cdf = none;
    Bernoulli "bernoulli" (prob_true: F64) -> Bool => d::Bernoulli::new(prob_true),
        support = Support::Discrete;
    Binomial "binomial" (p: F64, n: I64) -> I64 => d::Binomial::new(p, n),
        support = Support::Discrete,
        cdf = rejection;
    Geometric "geometric" (p: F64) -> I64 => d::Geometric::new(p),
        support = Support::Discrete,
        cdf = none;
    Multinomial "multinomial" (n: I64, p: F64) -> I64 => d::Multinomial::new(n, p)?,
        support = Support::Discrete;
    Poisson "poisson" (mu: F64) -> I64 => d::Poisson::new(mu),
        support = Support::Discrete,
        cdf = rejection;
    NegativeBinomial "negative_binomial" (mu: F64, alpha: F64) -> I64 => d::NegativeBinomial::new(mu, alpha),
        support = Support::Discrete,
        cdf = rejection;
    OrderedLogistic "ordered_logistic" (eta: F64, cutpoints: F64) -> I64 =>
        d::OrderedLogistic::new(eta, cutpoints),
        support = Support::Discrete;
    UniformInt "uniform_int" (x_min: I64, x_max: I64) -> I64 => d::UniformInt::new(x_min, x_max),
        support = Support::Discrete,
        cdf = none,
        gradient = |_, _| Ok(no_gradient(2));
}

impl DistributionKind {
    /// The error for a distribution without a CDF used where one is needed.
    fn no_cdf(&self) -> GraphError {
        GraphError::InvalidNode {
            op: self.name(),
            message: "the distribution has no CDF, so it cannot be truncated or censored".into(),
        }
    }

    /// The error for a parameter list that doesn't match the distribution (checked when the node is built).
    pub(crate) fn wrong_parameters(&self) -> GraphError {
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
        if let DistributionKind::Truncated(base) | DistributionKind::Censored(base) = kind {
            if !base.has_cdf() {
                return Err(invalid(format!(
                    "the base distribution ({}) has no CDF, so it cannot be truncated or censored",
                    base.name()
                )));
            }
        }
        if let DistributionKind::Mixture(component) = kind {
            if !component.is_elementwise() {
                return Err(invalid(format!(
                    "the component distribution ({}) does not have one value per element with an element-wise \
                     density, so it cannot be mixed",
                    component.name()
                )));
            }
        }
        match kind.parameter_types() {
            Some(types) => {
                if parameters.len() != types.len()
                    || parameters
                        .iter()
                        .zip(types.iter())
                        .any(|((t, _), e)| t != e)
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
            DistributionKind::LkjCholesky => {
                let shape = requested.unwrap_or(&[]);
                let [batch @ .., rows, cols] = shape else {
                    return Err(invalid(format!(
                        "needs a shape [..., K, K] for its K × K Cholesky factors, not {shape:?}"
                    )));
                };
                if rows != cols || *rows == 0 {
                    return Err(invalid(format!(
                        "needs a shape [..., K, K] for its K × K Cholesky factors, with K > 0, not {shape:?}"
                    )));
                }
                let eta = parameters[0].1;
                if keanu_tensor::broadcast_shape(batch, eta)?.as_slice() != batch {
                    return Err(invalid(format!(
                        "eta of shape {eta:?} does not broadcast to the batch shape {batch:?}"
                    )));
                }
                Ok(Shape::from_slice(shape))
            }
            DistributionKind::Kde(_) => Ok(Shape::from_slice(requested.unwrap_or(&[]))),
            DistributionKind::Mixture(_) => {
                mixture::value_shape(requested, parameters).map_err(invalid)
            }
            DistributionKind::OrderedLogistic => {
                let (eta, cutpoints) = (parameters[0].1, parameters[1].1);
                let Some((&count, leading)) = cutpoints.split_last() else {
                    return Err(invalid("the cutpoints must have at least one axis".into()));
                };
                if count == 0 {
                    return Err(invalid("there must be at least one cutpoint".into()));
                }
                let mut shape = keanu_tensor::broadcast_shape(eta, leading)?;
                if let Some(r) = requested {
                    shape = keanu_tensor::broadcast_shape(r, &shape)?;
                }
                Ok(shape)
            }
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
