//! Probabilistic nodes: a distribution whose parameters are the node's parents.

use crate::error::{GraphError, Result};
use crate::value::{DType, Value};
use keanu_distributions as d;
use keanu_distributions::KeanuRng;
use keanu_tensor::Shape;
use std::sync::Arc;

/// The distribution of a probabilistic node. The node's parents are the parameters, in the order listed.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum DistributionKind {
    /// Parameters: `mu`, `sigma`
    Gaussian,
    /// Parameters: `sigma`
    HalfGaussian,
    /// Parameters: `location`, `scale`
    Cauchy,
    /// Parameters: `scale`
    HalfCauchy,
    /// Parameters: `alpha`, `beta`, on [0, 1]
    Beta,
    /// Parameters: `k (i64)`
    ChiSquared,
    /// Parameters: `concentration`; the value has the concentration's shape
    Dirichlet,
    /// Parameters: `rate`
    Exponential,
    /// Parameters: `theta (scale)`, `k (shape)`
    Gamma,
    /// Parameters: `alpha`, `beta`
    InverseGamma,
    /// Parameters: `mu`, `beta`
    Laplace,
    /// Parameters: `mu`, `s`
    Logistic,
    /// Parameters: `mu`, `sigma`
    LogNormal,
    /// Parameters: `mu`, `covariance`; the value's last dimension is the event dimension
    MultivariateGaussian,
    /// Parameters: `location`, `scale`
    Pareto,
    /// Parameters: `v (i64)`
    StudentT,
    /// Parameters: `x_min`, `x_max`, `c`
    Triangular,
    /// Parameters: `x_min`, `x_max`
    Uniform,
    /// No parameters: a kernel density estimate from fixed samples.
    Kde(Arc<d::Kde>),
    /// Parameters: `probability of true` -> bool
    Bernoulli,
    /// Parameters: `p`, `n (i64)` -> i64
    Binomial,
    /// Parameters: `probability of each category...` -> i64 category index
    Categorical,
    /// Parameters: `p` -> i64
    Geometric,
    /// Parameters: `n (i64)`, `p` -> i64
    Multinomial,
    /// Parameters: `mu` -> i64
    Poisson,
    /// Parameters: `x_min (i64)`, `x_max (i64)` -> i64 in [x_min, x_max)
    UniformInt,
    /// A user-defined distribution; its parameters are whatever nodes it was given.
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
}

/// The gradient of a [`CustomDistribution`]'s log probability.
#[derive(Debug, Clone, Default)]
pub struct CustomGradient {
    /// With respect to the value, if it is f64 and differentiable.
    pub value: Option<keanu_tensor::DoubleTensor>,
    /// With respect to each parameter, in order (`None` where there is no gradient).
    pub parameters: Vec<Option<keanu_tensor::DoubleTensor>>,
}

impl DistributionKind {
    /// The parameter types, or None for a variable number of f64 parameters (Categorical).
    pub(crate) fn parameter_types(&self) -> Option<&'static [DType]> {
        use DType::*;
        use DistributionKind::*;
        Some(match self {
            Gaussian | Cauchy | Beta | Gamma | InverseGamma | Laplace | Logistic | LogNormal
            | Pareto | Uniform => &[F64, F64],
            MultivariateGaussian => &[F64, F64],
            HalfGaussian | HalfCauchy | Exponential | Dirichlet | Bernoulli | Geometric
            | Poisson => &[F64],
            ChiSquared | StudentT => &[I64],
            Triangular => &[F64, F64, F64],
            Kde(_) => &[],
            Binomial => &[F64, I64],
            Multinomial => &[I64, F64],
            UniformInt => &[I64, I64],
            Categorical | Custom(_) => return None,
        })
    }

    /// The type of the node's value.
    pub fn value_type(&self) -> DType {
        use DistributionKind::*;
        match self {
            Bernoulli => DType::Bool,
            Binomial | Categorical | Geometric | Multinomial | Poisson | UniformInt => DType::I64,
            Custom(c) => c.value_type(),
            _ => DType::F64,
        }
    }

    pub(crate) fn name(&self) -> &'static str {
        use DistributionKind::*;
        match self {
            Gaussian => "gaussian",
            HalfGaussian => "half_gaussian",
            Cauchy => "cauchy",
            HalfCauchy => "half_cauchy",
            Beta => "beta",
            ChiSquared => "chi_squared",
            Dirichlet => "dirichlet",
            Exponential => "exponential",
            Gamma => "gamma",
            InverseGamma => "inverse_gamma",
            Laplace => "laplace",
            Logistic => "logistic",
            LogNormal => "log_normal",
            MultivariateGaussian => "multivariate_gaussian",
            Pareto => "pareto",
            StudentT => "student_t",
            Triangular => "triangular",
            Uniform => "uniform",
            Kde(_) => "kde",
            Bernoulli => "bernoulli",
            Binomial => "binomial",
            Categorical => "categorical",
            Geometric => "geometric",
            Multinomial => "multinomial",
            Poisson => "poisson",
            UniformInt => "uniform_int",
            Custom(_) => "custom",
        }
    }
}

/// A probabilistic node's distribution and the shape of its value.
#[derive(Debug, Clone)]
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
        use DistributionKind::*;
        let f = |i: usize| parameters[i].f64().cloned();
        let i = |i: usize| parameters[i].i64().cloned();
        let shape = self.shape.as_slice();
        Ok(match &self.kind {
            Gaussian => Value::F64(d::Gaussian::new(f(0)?, f(1)?).sample(shape, rng)?),
            HalfGaussian => Value::F64(d::HalfGaussian::new(f(0)?).sample(shape, rng)?),
            Cauchy => Value::F64(d::Cauchy::new(f(0)?, f(1)?).sample(shape, rng)?),
            HalfCauchy => Value::F64(d::HalfCauchy::new(f(0)?).sample(shape, rng)?),
            Beta => Value::F64(d::Beta::standard(f(0)?, f(1)?).sample(shape, rng)?),
            ChiSquared => Value::F64(d::ChiSquared::new(i(0)?).sample(shape, rng)?),
            Dirichlet => Value::F64(d::Dirichlet::new(f(0)?).sample(rng)?),
            Exponential => Value::F64(d::Exponential::new(f(0)?).sample(shape, rng)?),
            Gamma => Value::F64(d::Gamma::new(f(0)?, f(1)?).sample(shape, rng)?),
            InverseGamma => Value::F64(d::InverseGamma::new(f(0)?, f(1)?).sample(shape, rng)?),
            Laplace => Value::F64(d::Laplace::new(f(0)?, f(1)?).sample(shape, rng)?),
            Logistic => Value::F64(d::Logistic::new(f(0)?, f(1)?).sample(shape, rng)?),
            LogNormal => Value::F64(d::LogNormal::new(f(0)?, f(1)?).sample(shape, rng)?),
            MultivariateGaussian => {
                Value::F64(d::MultivariateGaussian::new(f(0)?, f(1)?).sample(shape, rng)?)
            }
            Pareto => Value::F64(d::Pareto::new(f(0)?, f(1)?).sample(shape, rng)?),
            StudentT => Value::F64(d::StudentT::new(i(0)?).sample(shape, rng)?),
            Triangular => Value::F64(d::Triangular::new(f(0)?, f(1)?, f(2)?).sample(shape, rng)?),
            Uniform => Value::F64(d::Uniform::new(f(0)?, f(1)?).sample(shape, rng)?),
            Kde(kde) => {
                let count: usize = shape.iter().product();
                let signed: Vec<isize> = shape.iter().map(|&s| s as isize).collect();
                Value::F64(kde.sample(count, rng).reshape(&signed)?)
            }
            Bernoulli => Value::Bool(d::Bernoulli::new(f(0)?).sample(shape, rng)?),
            Binomial => Value::I64(d::Binomial::new(f(0)?, i(1)?).sample(shape, rng)?),
            Categorical => {
                let probabilities = (0..parameters.len()).map(f).collect::<Result<Vec<_>>>()?;
                Value::I64(d::Categorical::new(probabilities)?.sample(shape, rng)?)
            }
            Geometric => Value::I64(d::Geometric::new(f(0)?).sample(shape, rng)?),
            Multinomial => Value::I64(d::Multinomial::new(i(0)?, f(1)?)?.sample(shape, rng)?),
            Poisson => Value::I64(d::Poisson::new(f(0)?).sample(shape, rng)?),
            UniformInt => Value::I64(d::UniformInt::new(i(0)?, i(1)?).sample(shape, rng)?),
            Custom(c) => {
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
}
