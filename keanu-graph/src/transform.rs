//! Bijectors between unconstrained real vectors and the constrained values of a distribution's [`Support`].
//!
//! Gradient based algorithms (NUTS, the gradient optimizers, variational inference) move freely in ℝⁿ, but a
//! latent with a bounded support, such as a scale parameter or a probability vector, cannot take every real value.
//! A [`Bijector`] maps a flat unconstrained vector `y` to a value `x = forward(y)` in the support, and back with
//! [`Bijector::inverse`]. A density over `x` becomes a density over `y` by adding the log absolute determinant of
//! the forward Jacobian, [`Bijector::log_det_jacobian`].
//!
//! The transforms follow the Stan reference manual ("Constraint transforms"):
//!
//! | Support | Bijector | Forward map | log \|det J\| |
//! |---|---|---|---|
//! | [`Support::Real`] | [`Identity`] | `x = y` | 0 |
//! | [`Support::GreaterThan`] | [`LowerBound`] | `x = a + exp(y)` | `y` |
//! | [`Support::LessThan`] | [`UpperBound`] | `x = b − exp(y)` | `y` |
//! | [`Support::Interval`] | [`Interval`] | `x = a + (b − a)·logistic(y)` | `log(b − a) + log s + log(1 − s)`, `s = logistic(y)` |
//! | [`Support::Ordered`] | [`Ordered`] | `x₁ = y₁`, `xₖ = xₖ₋₁ + exp(yₖ)` | `Σₖ₌₂ yₖ` |
//! | [`Support::Simplex`] | [`Simplex`] | inverse isometric log-ratio, then softmax | `Σ log xᵢ + ½ log K` |
//! | [`Support::CorrelationCholesky`] | [`CorrelationCholesky`] | tanh canonical partial correlations | see the type |
//! | [`Support::Circular`] | [`Circular`] | `θ = atan2(v, u)` | `−½(u² + v²)` (not a Jacobian, see the type) |
//!
//! Every log-Jacobian is summed over the whole value, and every value is laid out row-major, so the unconstrained
//! vector of a batch is the concatenation of the unconstrained vectors of its members.
//!
//! The unconstrained length can differ from the constrained value's size: a simplex of K elements has K − 1 free
//! coordinates, a K × K correlation Cholesky factor has K(K − 1)/2, and an angle has 2. For the simplex and the
//! correlation Cholesky factor the Jacobian, and so the log-determinant, is taken with respect to the free
//! constrained coordinates: the first K − 1 elements of a simplex, and the strictly lower triangle (row by row) of a
//! Cholesky factor, which is what the Dirichlet and LKJ densities are densities of.
//!
//! [`bijector_for`] chooses the bijector for a support and a value shape:
//!
//! ```
//! use keanu_graph::{Support, transform};
//! use keanu_tensor::DoubleTensor;
//!
//! let b = transform::bijector_for(&Support::Simplex, &[3])?;
//! assert_eq!(b.unconstrained_len(), 2);
//! let x = b.forward(&[0.5, -1.0])?;
//! assert!((x.as_slice().iter().sum::<f64>() - 1.0).abs() < 1e-12);
//! let y = b.inverse(&x)?;
//! assert!((y[0] - 0.5).abs() < 1e-12 && (y[1] + 1.0).abs() < 1e-12);
//! # Ok::<(), keanu_graph::GraphError>(())
//! ```
//!
//! First derivatives are provided as Jacobian-vector and vector-Jacobian products of the forward map and the
//! gradient of the log-Jacobian. Second derivatives, which exact Hessian-vector products in unconstrained space need
//! ([`Parameterisation::hvp`](crate::Parameterisation::hvp)), are the derivative of the vector-Jacobian product
//! along a direction ([`Bijector::forward_vjp_jvp`]) and the log-Jacobian's Hessian-vector product
//! ([`Bijector::log_det_jacobian_hvp`]). Every bijector here implements them; for other implementations of the
//! trait they default to a [`GraphError::NoSecondOrderRule`] error.

use crate::error::{GraphError, Result};
use crate::op::Support;
use keanu_tensor::DoubleTensor;
use std::fmt;

/// A smooth invertible map from unconstrained flat vectors to constrained values of a fixed shape.
///
/// Every method checks the sizes of its arguments and returns [`GraphError::Transform`] for a wrong size or, for
/// [`inverse`](Self::inverse), a value outside or on the boundary of the support. None of them panics.
pub trait Bijector: fmt::Debug + Send + Sync {
    /// The shape of constrained values.
    fn constrained_shape(&self) -> &[usize];

    /// The length of unconstrained vectors.
    fn unconstrained_len(&self) -> usize;

    /// Maps an unconstrained vector to a constrained value.
    fn forward(&self, y: &[f64]) -> Result<DoubleTensor>;

    /// Maps a constrained value to its unconstrained vector. Fails if the value is outside the support or on its
    /// boundary, where the unconstrained value would be infinite.
    fn inverse(&self, x: &DoubleTensor) -> Result<Vec<f64>>;

    /// The log absolute determinant of the forward Jacobian at `y`, summed over the value.
    fn log_det_jacobian(&self, y: &[f64]) -> Result<f64>;

    /// The gradient of [`log_det_jacobian`](Self::log_det_jacobian) with respect to `y`.
    fn log_det_jacobian_gradient(&self, y: &[f64]) -> Result<Vec<f64>>;

    /// The Jacobian-vector product of the forward map: the directional derivative of `forward` at `y` along
    /// `tangent` (an unconstrained vector), as a constrained-shaped tensor.
    fn forward_jvp(&self, y: &[f64], tangent: &[f64]) -> Result<DoubleTensor>;

    /// The vector-Jacobian product of the forward map: given the gradient of a function with respect to the
    /// constrained value (`cotangent`, of the constrained shape), the gradient of the same function with respect to
    /// `y`. Elements of the constrained value that are constant (a Cholesky factor's upper triangle) receive no
    /// gradient.
    fn forward_vjp(&self, y: &[f64], cotangent: &DoubleTensor) -> Result<Vec<f64>>;

    /// The derivative of [`forward_vjp`](Self::forward_vjp) at `y` along `tangent`, with the cotangent held fixed:
    /// `Σᵢ cᵢ ∇²xᵢ(y) · tangent`, where `xᵢ` are the elements of the constrained value. This is the curvature term of
    /// the chain rule when a Hessian-vector product is pulled back through the forward map.
    ///
    /// The default returns [`GraphError::NoSecondOrderRule`] naming the type, so implementations written before
    /// second derivatives existed keep compiling; every bijector of this module implements it.
    fn forward_vjp_jvp(
        &self,
        y: &[f64],
        tangent: &[f64],
        cotangent: &DoubleTensor,
    ) -> Result<Vec<f64>> {
        let _ = (y, tangent, cotangent);
        Err(no_second_order_rule::<Self>())
    }

    /// The Hessian-vector product of [`log_det_jacobian`](Self::log_det_jacobian) at `y` along `tangent`.
    ///
    /// The default returns [`GraphError::NoSecondOrderRule`] naming the type; every bijector of this module
    /// implements it.
    fn log_det_jacobian_hvp(&self, y: &[f64], tangent: &[f64]) -> Result<Vec<f64>> {
        let _ = (y, tangent);
        Err(no_second_order_rule::<Self>())
    }
}

/// The error of a bijector type without second derivatives.
fn no_second_order_rule<T: ?Sized>() -> GraphError {
    GraphError::NoSecondOrderRule(format!("the bijector {}", std::any::type_name::<T>()))
}

/// Checks the arguments of the second-order methods: `y` and `tangent` of the unconstrained length.
fn check_second_order(y: &[f64], tangent: &[f64], len: usize) -> Result<()> {
    check_len("the unconstrained vector", y.len(), len)?;
    check_len("the tangent", tangent.len(), len)
}

/// The bijector for values of `shape` in `support`.
///
/// Bounds of [`Support::GreaterThan`], [`Support::LessThan`] and [`Support::Interval`] are broadcast to `shape` and apply element-wise;
/// they must be finite, and each lower bound must be less than its upper bound. [`Support::Ordered`] and
/// [`Support::Simplex`] act on the last axis and [`Support::CorrelationCholesky`] on the last two (which must be
/// equal), with any leading axes as a batch.
///
/// # Errors
///
/// [`GraphError::Transform`] for [`Support::Discrete`] (a discrete value has no unconstrained form), for bounds
/// that are NaN, infinite, out of order or don't broadcast to `shape`, and for a shape with too few axes for the
/// support.
pub fn bijector_for(support: &Support, shape: &[usize]) -> Result<Box<dyn Bijector>> {
    Ok(match support {
        Support::Real => Box::new(Identity::new(shape)),
        Support::GreaterThan(lower) => Box::new(LowerBound::new(lower, shape)?),
        Support::LessThan(upper) => Box::new(UpperBound::new(upper, shape)?),
        Support::Interval(lower, upper) => Box::new(Interval::new(lower, upper, shape)?),
        Support::Simplex => Box::new(Simplex::new(shape)?),
        Support::Ordered => Box::new(Ordered::new(shape)?),
        Support::CorrelationCholesky => Box::new(CorrelationCholesky::new(shape)?),
        Support::Circular => Box::new(Circular::new(shape)),
        Support::Discrete => {
            return Err(transform_error(
                "a discrete support has no unconstrained transform",
            ));
        }
    })
}

fn transform_error(message: impl Into<String>) -> GraphError {
    GraphError::Transform(message.into())
}

fn check_len(what: &str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(transform_error(format!(
            "{what} has length {actual}, expected {expected}"
        )))
    }
}

fn check_shape(what: &str, x: &DoubleTensor, shape: &[usize]) -> Result<()> {
    if x.shape() == shape {
        Ok(())
    } else {
        Err(transform_error(format!(
            "{what} has shape {:?}, expected {shape:?}",
            x.shape()
        )))
    }
}

fn tensor(shape: &[usize], data: Vec<f64>) -> Result<DoubleTensor> {
    Ok(DoubleTensor::new(shape, data)?)
}

/// Broadcasts a bound to the value's shape and checks it is finite.
fn bound(name: &str, bound: &DoubleTensor, shape: &[usize]) -> Result<Vec<f64>> {
    let values = bound.broadcast_to(shape).map_err(|_| {
        transform_error(format!(
            "a {name} bound of shape {:?} does not broadcast to the value's shape {shape:?}",
            bound.shape()
        ))
    })?;
    if let Some(v) = values.as_slice().iter().find(|v| !v.is_finite()) {
        return Err(transform_error(format!(
            "a {name} bound must be finite, not {v}"
        )));
    }
    Ok(values.into_vec())
}

/// The last axis of a shape, as (batch, length), failing for a scalar.
fn last_axis(what: &str, shape: &[usize]) -> Result<(usize, usize)> {
    match shape.last() {
        Some(&k) => Ok((shape[..shape.len() - 1].iter().product(), k)),
        None => Err(transform_error(format!(
            "a {what} value needs at least one axis, not a scalar"
        ))),
    }
}

/// `log(1 + exp(x))` without overflow.
fn log1p_exp(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// `1 / (1 + exp(-x))` without overflow.
fn logistic(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// `log(cosh(x))` without overflow.
fn log_cosh(x: f64) -> f64 {
    let a = x.abs();
    a + (-2.0 * a).exp().ln_1p() - std::f64::consts::LN_2
}

/// Splits a slice into rows of length `k`, giving no rows for `k = 0` (when the slice is empty too).
fn rows(data: &[f64], k: usize) -> std::slice::ChunksExact<'_, f64> {
    data.chunks_exact(k.max(1))
}

fn rows_mut(data: &mut [f64], k: usize) -> std::slice::ChunksExactMut<'_, f64> {
    data.chunks_exact_mut(k.max(1))
}

/// The identity, for [`Support::Real`]: `x = y`.
#[derive(Debug, Clone, PartialEq)]
pub struct Identity {
    shape: Vec<usize>,
}

impl Identity {
    /// The identity on values of `shape`.
    pub fn new(shape: &[usize]) -> Self {
        Identity {
            shape: shape.to_vec(),
        }
    }

    fn len(&self) -> usize {
        self.shape.iter().product()
    }
}

impl Bijector for Identity {
    fn constrained_shape(&self) -> &[usize] {
        &self.shape
    }

    fn unconstrained_len(&self) -> usize {
        self.len()
    }

    fn forward(&self, y: &[f64]) -> Result<DoubleTensor> {
        check_len("the unconstrained vector", y.len(), self.len())?;
        tensor(&self.shape, y.to_vec())
    }

    fn inverse(&self, x: &DoubleTensor) -> Result<Vec<f64>> {
        check_shape("the value", x, &self.shape)?;
        if let Some(v) = x.as_slice().iter().find(|v| !v.is_finite()) {
            return Err(transform_error(format!(
                "a real value must be finite, not {v}"
            )));
        }
        Ok(x.as_slice().to_vec())
    }

    fn log_det_jacobian(&self, y: &[f64]) -> Result<f64> {
        check_len("the unconstrained vector", y.len(), self.len())?;
        Ok(0.0)
    }

    fn log_det_jacobian_gradient(&self, y: &[f64]) -> Result<Vec<f64>> {
        check_len("the unconstrained vector", y.len(), self.len())?;
        Ok(vec![0.0; y.len()])
    }

    fn forward_jvp(&self, y: &[f64], tangent: &[f64]) -> Result<DoubleTensor> {
        check_len("the unconstrained vector", y.len(), self.len())?;
        check_len("the tangent", tangent.len(), self.len())?;
        tensor(&self.shape, tangent.to_vec())
    }

    fn forward_vjp(&self, y: &[f64], cotangent: &DoubleTensor) -> Result<Vec<f64>> {
        check_len("the unconstrained vector", y.len(), self.len())?;
        check_shape("the cotangent", cotangent, &self.shape)?;
        Ok(cotangent.as_slice().to_vec())
    }

    fn forward_vjp_jvp(
        &self,
        y: &[f64],
        tangent: &[f64],
        cotangent: &DoubleTensor,
    ) -> Result<Vec<f64>> {
        check_second_order(y, tangent, self.len())?;
        check_shape("the cotangent", cotangent, &self.shape)?;
        Ok(vec![0.0; y.len()])
    }

    fn log_det_jacobian_hvp(&self, y: &[f64], tangent: &[f64]) -> Result<Vec<f64>> {
        check_second_order(y, tangent, self.len())?;
        Ok(vec![0.0; y.len()])
    }
}

/// A lower bound, for [`Support::GreaterThan`]: `x = a + exp(y)` element-wise, with log-Jacobian `Σ y`.
#[derive(Debug, Clone, PartialEq)]
pub struct LowerBound {
    shape: Vec<usize>,
    lower: Vec<f64>,
}

impl LowerBound {
    /// Values of `shape` greater than `lower`, which is broadcast to `shape`.
    ///
    /// # Errors
    ///
    /// [`GraphError::Transform`] if the bound doesn't broadcast to `shape` or isn't finite.
    pub fn new(lower: &DoubleTensor, shape: &[usize]) -> Result<Self> {
        Ok(LowerBound {
            shape: shape.to_vec(),
            lower: bound("lower", lower, shape)?,
        })
    }
}

impl Bijector for LowerBound {
    fn constrained_shape(&self) -> &[usize] {
        &self.shape
    }

    fn unconstrained_len(&self) -> usize {
        self.lower.len()
    }

    fn forward(&self, y: &[f64]) -> Result<DoubleTensor> {
        check_len("the unconstrained vector", y.len(), self.lower.len())?;
        let x = y
            .iter()
            .zip(&self.lower)
            .map(|(y, a)| a + y.exp())
            .collect();
        tensor(&self.shape, x)
    }

    fn inverse(&self, x: &DoubleTensor) -> Result<Vec<f64>> {
        check_shape("the value", x, &self.shape)?;
        x.as_slice()
            .iter()
            .zip(&self.lower)
            .map(|(&x, &a)| {
                let d = x - a;
                if d > 0.0 && d.is_finite() {
                    Ok(d.ln())
                } else {
                    Err(transform_error(format!(
                        "{x} is not a finite value greater than the lower bound {a}"
                    )))
                }
            })
            .collect()
    }

    fn log_det_jacobian(&self, y: &[f64]) -> Result<f64> {
        check_len("the unconstrained vector", y.len(), self.lower.len())?;
        Ok(y.iter().sum())
    }

    fn log_det_jacobian_gradient(&self, y: &[f64]) -> Result<Vec<f64>> {
        check_len("the unconstrained vector", y.len(), self.lower.len())?;
        Ok(vec![1.0; y.len()])
    }

    fn forward_jvp(&self, y: &[f64], tangent: &[f64]) -> Result<DoubleTensor> {
        check_len("the unconstrained vector", y.len(), self.lower.len())?;
        check_len("the tangent", tangent.len(), self.lower.len())?;
        let dx = y.iter().zip(tangent).map(|(y, t)| y.exp() * t).collect();
        tensor(&self.shape, dx)
    }

    fn forward_vjp(&self, y: &[f64], cotangent: &DoubleTensor) -> Result<Vec<f64>> {
        check_len("the unconstrained vector", y.len(), self.lower.len())?;
        check_shape("the cotangent", cotangent, &self.shape)?;
        Ok(y.iter()
            .zip(cotangent.as_slice())
            .map(|(y, c)| y.exp() * c)
            .collect())
    }

    fn forward_vjp_jvp(
        &self,
        y: &[f64],
        tangent: &[f64],
        cotangent: &DoubleTensor,
    ) -> Result<Vec<f64>> {
        check_second_order(y, tangent, self.lower.len())?;
        check_shape("the cotangent", cotangent, &self.shape)?;
        // the vector-Jacobian product is exp(y) c, element-wise
        Ok(y.iter()
            .zip(tangent)
            .zip(cotangent.as_slice())
            .map(|((y, t), c)| y.exp() * c * t)
            .collect())
    }

    fn log_det_jacobian_hvp(&self, y: &[f64], tangent: &[f64]) -> Result<Vec<f64>> {
        // log |J| = Σ y is linear
        check_second_order(y, tangent, self.lower.len())?;
        Ok(vec![0.0; y.len()])
    }
}

/// An upper bound, for [`Support::LessThan`]: `x = b − exp(y)` element-wise, with log-Jacobian `Σ y`.
#[derive(Debug, Clone, PartialEq)]
pub struct UpperBound {
    shape: Vec<usize>,
    upper: Vec<f64>,
}

impl UpperBound {
    /// Values of `shape` less than `upper`, which is broadcast to `shape`.
    ///
    /// # Errors
    ///
    /// [`GraphError::Transform`] if the bound doesn't broadcast to `shape` or isn't finite.
    pub fn new(upper: &DoubleTensor, shape: &[usize]) -> Result<Self> {
        Ok(UpperBound {
            shape: shape.to_vec(),
            upper: bound("upper", upper, shape)?,
        })
    }
}

impl Bijector for UpperBound {
    fn constrained_shape(&self) -> &[usize] {
        &self.shape
    }

    fn unconstrained_len(&self) -> usize {
        self.upper.len()
    }

    fn forward(&self, y: &[f64]) -> Result<DoubleTensor> {
        check_len("the unconstrained vector", y.len(), self.upper.len())?;
        let x = y
            .iter()
            .zip(&self.upper)
            .map(|(y, b)| b - y.exp())
            .collect();
        tensor(&self.shape, x)
    }

    fn inverse(&self, x: &DoubleTensor) -> Result<Vec<f64>> {
        check_shape("the value", x, &self.shape)?;
        x.as_slice()
            .iter()
            .zip(&self.upper)
            .map(|(&x, &b)| {
                let d = b - x;
                if d > 0.0 && d.is_finite() {
                    Ok(d.ln())
                } else {
                    Err(transform_error(format!(
                        "{x} is not a finite value less than the upper bound {b}"
                    )))
                }
            })
            .collect()
    }

    fn log_det_jacobian(&self, y: &[f64]) -> Result<f64> {
        check_len("the unconstrained vector", y.len(), self.upper.len())?;
        Ok(y.iter().sum())
    }

    fn log_det_jacobian_gradient(&self, y: &[f64]) -> Result<Vec<f64>> {
        check_len("the unconstrained vector", y.len(), self.upper.len())?;
        Ok(vec![1.0; y.len()])
    }

    fn forward_jvp(&self, y: &[f64], tangent: &[f64]) -> Result<DoubleTensor> {
        check_len("the unconstrained vector", y.len(), self.upper.len())?;
        check_len("the tangent", tangent.len(), self.upper.len())?;
        let dx = y.iter().zip(tangent).map(|(y, t)| -y.exp() * t).collect();
        tensor(&self.shape, dx)
    }

    fn forward_vjp(&self, y: &[f64], cotangent: &DoubleTensor) -> Result<Vec<f64>> {
        check_len("the unconstrained vector", y.len(), self.upper.len())?;
        check_shape("the cotangent", cotangent, &self.shape)?;
        Ok(y.iter()
            .zip(cotangent.as_slice())
            .map(|(y, c)| -y.exp() * c)
            .collect())
    }

    fn forward_vjp_jvp(
        &self,
        y: &[f64],
        tangent: &[f64],
        cotangent: &DoubleTensor,
    ) -> Result<Vec<f64>> {
        check_second_order(y, tangent, self.upper.len())?;
        check_shape("the cotangent", cotangent, &self.shape)?;
        // the vector-Jacobian product is -exp(y) c, element-wise
        Ok(y.iter()
            .zip(tangent)
            .zip(cotangent.as_slice())
            .map(|((y, t), c)| -y.exp() * c * t)
            .collect())
    }

    fn log_det_jacobian_hvp(&self, y: &[f64], tangent: &[f64]) -> Result<Vec<f64>> {
        // log |J| = Σ y is linear
        check_second_order(y, tangent, self.upper.len())?;
        Ok(vec![0.0; y.len()])
    }
}

/// A finite interval, for [`Support::Interval`]: `x = a + (b − a)·logistic(y)` element-wise, with log-Jacobian
/// `Σ log(b − a) + log s + log(1 − s)` where `s = logistic(y)`.
///
/// For positive `y` the value is computed from the upper bound, `x = b − (b − a)·logistic(−y)`, so values near
/// either bound keep their precision.
#[derive(Debug, Clone, PartialEq)]
pub struct Interval {
    shape: Vec<usize>,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl Interval {
    /// Values of `shape` between `lower` and `upper`, which are broadcast to `shape`.
    ///
    /// # Errors
    ///
    /// [`GraphError::Transform`] if a bound doesn't broadcast to `shape` or isn't finite, or a lower bound isn't
    /// less than its upper bound.
    pub fn new(lower: &DoubleTensor, upper: &DoubleTensor, shape: &[usize]) -> Result<Self> {
        let lower = bound("lower", lower, shape)?;
        let upper = bound("upper", upper, shape)?;
        if let Some((a, b)) = lower.iter().zip(&upper).find(|(a, b)| a >= b) {
            return Err(transform_error(format!(
                "an interval's lower bound must be less than its upper bound, not [{a}, {b}]"
            )));
        }
        Ok(Interval {
            shape: shape.to_vec(),
            lower,
            upper,
        })
    }

    fn bounds(&self) -> impl Iterator<Item = (f64, f64)> + '_ {
        self.lower.iter().copied().zip(self.upper.iter().copied())
    }
}

impl Bijector for Interval {
    fn constrained_shape(&self) -> &[usize] {
        &self.shape
    }

    fn unconstrained_len(&self) -> usize {
        self.lower.len()
    }

    fn forward(&self, y: &[f64]) -> Result<DoubleTensor> {
        check_len("the unconstrained vector", y.len(), self.lower.len())?;
        let x = y
            .iter()
            .zip(self.bounds())
            .map(|(&y, (a, b))| {
                if y > 0.0 {
                    b - (b - a) * logistic(-y)
                } else {
                    a + (b - a) * logistic(y)
                }
            })
            .collect();
        tensor(&self.shape, x)
    }

    fn inverse(&self, x: &DoubleTensor) -> Result<Vec<f64>> {
        check_shape("the value", x, &self.shape)?;
        x.as_slice()
            .iter()
            .zip(self.bounds())
            .map(|(&x, (a, b))| {
                if x > a && x < b {
                    Ok((x - a).ln() - (b - x).ln())
                } else {
                    Err(transform_error(format!(
                        "{x} is not strictly inside the interval ({a}, {b})"
                    )))
                }
            })
            .collect()
    }

    fn log_det_jacobian(&self, y: &[f64]) -> Result<f64> {
        check_len("the unconstrained vector", y.len(), self.lower.len())?;
        Ok(y.iter()
            .zip(self.bounds())
            .map(|(&y, (a, b))| (b - a).ln() - log1p_exp(-y) - log1p_exp(y))
            .sum())
    }

    fn log_det_jacobian_gradient(&self, y: &[f64]) -> Result<Vec<f64>> {
        check_len("the unconstrained vector", y.len(), self.lower.len())?;
        Ok(y.iter().map(|&y| logistic(-y) - logistic(y)).collect())
    }

    fn forward_jvp(&self, y: &[f64], tangent: &[f64]) -> Result<DoubleTensor> {
        check_len("the unconstrained vector", y.len(), self.lower.len())?;
        check_len("the tangent", tangent.len(), self.lower.len())?;
        let dx = y
            .iter()
            .zip(tangent)
            .zip(self.bounds())
            .map(|((&y, t), (a, b))| (b - a) * logistic(y) * logistic(-y) * t)
            .collect();
        tensor(&self.shape, dx)
    }

    fn forward_vjp(&self, y: &[f64], cotangent: &DoubleTensor) -> Result<Vec<f64>> {
        check_len("the unconstrained vector", y.len(), self.lower.len())?;
        check_shape("the cotangent", cotangent, &self.shape)?;
        Ok(y.iter()
            .zip(cotangent.as_slice())
            .zip(self.bounds())
            .map(|((&y, c), (a, b))| (b - a) * logistic(y) * logistic(-y) * c)
            .collect())
    }

    fn forward_vjp_jvp(
        &self,
        y: &[f64],
        tangent: &[f64],
        cotangent: &DoubleTensor,
    ) -> Result<Vec<f64>> {
        check_second_order(y, tangent, self.lower.len())?;
        check_shape("the cotangent", cotangent, &self.shape)?;
        // x' = (b − a) s (1 − s), so x'' = (b − a) s (1 − s) (1 − 2s)
        Ok(y.iter()
            .zip(tangent)
            .zip(cotangent.as_slice())
            .zip(self.bounds())
            .map(|(((&y, t), c), (a, b))| {
                let (s, r) = (logistic(y), logistic(-y));
                (b - a) * s * r * (r - s) * c * t
            })
            .collect())
    }

    fn log_det_jacobian_hvp(&self, y: &[f64], tangent: &[f64]) -> Result<Vec<f64>> {
        check_second_order(y, tangent, self.lower.len())?;
        // the gradient is 1 − 2s, so the Hessian is diagonal, −2 s (1 − s)
        Ok(y.iter()
            .zip(tangent)
            .map(|(&y, t)| -2.0 * logistic(y) * logistic(-y) * t)
            .collect())
    }
}

/// Strictly increasing vectors over the last axis, for [`Support::Ordered`]: `x₁ = y₁` and `xₖ = xₖ₋₁ + exp(yₖ)`,
/// with log-Jacobian `Σₖ₌₂ yₖ`. Leading axes are a batch.
#[derive(Debug, Clone, PartialEq)]
pub struct Ordered {
    shape: Vec<usize>,
    k: usize,
}

impl Ordered {
    /// Ordered vectors over the last axis of `shape`.
    ///
    /// # Errors
    ///
    /// [`GraphError::Transform`] if `shape` is a scalar shape.
    pub fn new(shape: &[usize]) -> Result<Self> {
        let (_, k) = last_axis("ordered", shape)?;
        Ok(Ordered {
            shape: shape.to_vec(),
            k,
        })
    }

    fn len(&self) -> usize {
        self.shape.iter().product()
    }
}

impl Bijector for Ordered {
    fn constrained_shape(&self) -> &[usize] {
        &self.shape
    }

    fn unconstrained_len(&self) -> usize {
        self.len()
    }

    fn forward(&self, y: &[f64]) -> Result<DoubleTensor> {
        check_len("the unconstrained vector", y.len(), self.len())?;
        let mut x = y.to_vec();
        for row in rows_mut(&mut x, self.k) {
            for i in 1..row.len() {
                row[i] = row[i - 1] + row[i].exp();
            }
        }
        tensor(&self.shape, x)
    }

    fn inverse(&self, x: &DoubleTensor) -> Result<Vec<f64>> {
        check_shape("the value", x, &self.shape)?;
        let mut y = Vec::with_capacity(self.len());
        for row in rows(x.as_slice(), self.k) {
            if !row[0].is_finite() {
                return Err(transform_error(format!(
                    "an ordered value must be finite, not {}",
                    row[0]
                )));
            }
            y.push(row[0]);
            for pair in row.windows(2) {
                let d = pair[1] - pair[0];
                if !(d > 0.0 && d.is_finite()) {
                    return Err(transform_error(format!(
                        "{row:?} is not a finite strictly increasing vector"
                    )));
                }
                y.push(d.ln());
            }
        }
        Ok(y)
    }

    fn log_det_jacobian(&self, y: &[f64]) -> Result<f64> {
        check_len("the unconstrained vector", y.len(), self.len())?;
        Ok(rows(y, self.k)
            .map(|row| row[1..].iter().sum::<f64>())
            .sum())
    }

    fn log_det_jacobian_gradient(&self, y: &[f64]) -> Result<Vec<f64>> {
        check_len("the unconstrained vector", y.len(), self.len())?;
        let mut g = vec![1.0; y.len()];
        for row in rows_mut(&mut g, self.k) {
            row[0] = 0.0;
        }
        Ok(g)
    }

    fn forward_jvp(&self, y: &[f64], tangent: &[f64]) -> Result<DoubleTensor> {
        check_len("the unconstrained vector", y.len(), self.len())?;
        check_len("the tangent", tangent.len(), self.len())?;
        let mut dx = Vec::with_capacity(self.len());
        for (y, t) in rows(y, self.k).zip(rows(tangent, self.k)) {
            let mut acc = t[0];
            dx.push(acc);
            for i in 1..y.len() {
                acc += y[i].exp() * t[i];
                dx.push(acc);
            }
        }
        tensor(&self.shape, dx)
    }

    fn forward_vjp(&self, y: &[f64], cotangent: &DoubleTensor) -> Result<Vec<f64>> {
        check_len("the unconstrained vector", y.len(), self.len())?;
        check_shape("the cotangent", cotangent, &self.shape)?;
        let mut g = vec![0.0; self.len()];
        for ((y, c), g) in rows(y, self.k)
            .zip(rows(cotangent.as_slice(), self.k))
            .zip(rows_mut(&mut g, self.k))
        {
            // g₁ = Σ cₖ and gⱼ = exp(yⱼ) Σₖ₌ⱼ cₖ, from the suffix sums of the cotangent
            let mut suffix = 0.0;
            for i in (0..y.len()).rev() {
                suffix += c[i];
                g[i] = if i == 0 { suffix } else { y[i].exp() * suffix };
            }
        }
        Ok(g)
    }

    fn forward_vjp_jvp(
        &self,
        y: &[f64],
        tangent: &[f64],
        cotangent: &DoubleTensor,
    ) -> Result<Vec<f64>> {
        check_second_order(y, tangent, self.len())?;
        check_shape("the cotangent", cotangent, &self.shape)?;
        let mut g = vec![0.0; self.len()];
        for (((y, t), c), g) in rows(y, self.k)
            .zip(rows(tangent, self.k))
            .zip(rows(cotangent.as_slice(), self.k))
            .zip(rows_mut(&mut g, self.k))
        {
            // the derivative of gⱼ = exp(yⱼ) Σₖ₌ⱼ cₖ (j > 0) along t; g₁ = Σ cₖ is constant
            let mut suffix = 0.0;
            for i in (0..y.len()).rev() {
                suffix += c[i];
                g[i] = if i == 0 {
                    0.0
                } else {
                    y[i].exp() * t[i] * suffix
                };
            }
        }
        Ok(g)
    }

    fn log_det_jacobian_hvp(&self, y: &[f64], tangent: &[f64]) -> Result<Vec<f64>> {
        // log |J| = Σₖ₌₂ yₖ is linear
        check_second_order(y, tangent, self.len())?;
        Ok(vec![0.0; y.len()])
    }
}

/// Probability vectors over the last axis, for [`Support::Simplex`], as in Stan 2.37 and later: the K − 1
/// unconstrained coordinates are mapped by the inverse isometric log-ratio transform to a zero-sum K-vector `z`,
/// then `x = softmax(z)`. Leading axes are a batch.
///
/// The isometric log-ratio uses Stan's orthonormal basis of the zero-sum subspace (column `i` of `V`, for
/// `i = 1..K−1`, has `1/√(i(i+1))` in its first `i` entries and `−i/√(i(i+1))` in entry `i + 1`), so `z = V y` and
/// `y = Vᵀ log x`. The log-Jacobian of the map from `y` to the first K − 1 elements of `x` is
/// `Σᵢ log xᵢ + ½ log K`.
#[derive(Debug, Clone, PartialEq)]
pub struct Simplex {
    shape: Vec<usize>,
    k: usize,
}

/// The tolerance on a simplex value's sum.
const SIMPLEX_SUM_TOLERANCE: f64 = 1e-8;

impl Simplex {
    /// Simplices over the last axis of `shape`.
    ///
    /// # Errors
    ///
    /// [`GraphError::Transform`] if `shape` is a scalar shape or its last axis is empty.
    pub fn new(shape: &[usize]) -> Result<Self> {
        let (_, k) = last_axis("simplex", shape)?;
        if k == 0 {
            return Err(transform_error("a simplex needs at least one element"));
        }
        Ok(Simplex {
            shape: shape.to_vec(),
            k,
        })
    }

    fn batch(&self) -> usize {
        self.shape[..self.shape.len() - 1].iter().product()
    }

    /// `z = V y`, for `y` of length K − 1 and `z` of length K.
    fn basis_times(y: &[f64], z: &mut [f64]) {
        z.fill(0.0);
        let mut suffix = 0.0;
        for i in (1..=y.len()).rev() {
            let n = i as f64;
            let w = y[i - 1] / (n * (n + 1.0)).sqrt();
            suffix += w;
            z[i - 1] += suffix;
            z[i] -= n * w;
        }
    }

    /// `y = Vᵀ z`, for `z` of length K and `y` of length K − 1.
    fn basis_transpose_times(z: &[f64], y: &mut [f64]) {
        let mut prefix = 0.0;
        for i in 1..z.len() {
            prefix += z[i - 1];
            let n = i as f64;
            y[i - 1] = (prefix - n * z[i]) / (n * (n + 1.0)).sqrt();
        }
    }

    /// Calls `f(row, z, x, log_sum)` for each row's unconstrained coordinates `y`, with `z = V y`,
    /// `log_sum = log Σ exp(z)` and `x = softmax(z)`.
    fn for_each_row(&self, y: &[f64], mut f: impl FnMut(usize, &[f64], &[f64], f64)) {
        let k = self.k;
        let mut z = vec![0.0; k];
        let mut x = vec![0.0; k];
        for (row, y) in self.rows(y).enumerate() {
            Self::basis_times(y, &mut z);
            let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let log_sum = max + z.iter().map(|z| (z - max).exp()).sum::<f64>().ln();
            for (x, z) in x.iter_mut().zip(&z) {
                *x = (z - log_sum).exp();
            }
            f(row, &z, &x, log_sum);
        }
    }

    /// The unconstrained coordinates of each row. For K = 1 a row has none (and `chunks_exact(0)` is invalid), so
    /// this yields one empty row per batch member.
    fn rows<'a>(&self, y: &'a [f64]) -> Box<dyn Iterator<Item = &'a [f64]> + 'a> {
        if self.k == 1 {
            Box::new(std::iter::repeat_n(&[][..], self.batch()))
        } else {
            Box::new(y.chunks_exact(self.k - 1))
        }
    }
}

impl Bijector for Simplex {
    fn constrained_shape(&self) -> &[usize] {
        &self.shape
    }

    fn unconstrained_len(&self) -> usize {
        self.batch() * (self.k - 1)
    }

    fn forward(&self, y: &[f64]) -> Result<DoubleTensor> {
        check_len(
            "the unconstrained vector",
            y.len(),
            self.unconstrained_len(),
        )?;
        let mut out = Vec::with_capacity(self.batch() * self.k);
        self.for_each_row(y, |_, _, x, _| out.extend_from_slice(x));
        tensor(&self.shape, out)
    }

    fn inverse(&self, x: &DoubleTensor) -> Result<Vec<f64>> {
        check_shape("the value", x, &self.shape)?;
        let k = self.k;
        let mut y = vec![0.0; self.unconstrained_len()];
        let mut log_x = vec![0.0; k];
        for (row, x) in x.as_slice().chunks_exact(k).enumerate() {
            let sum: f64 = x.iter().sum();
            if !x.iter().all(|&v| v > 0.0 && v.is_finite())
                || (sum - 1.0).abs() > SIMPLEX_SUM_TOLERANCE
            {
                return Err(transform_error(format!(
                    "{x:?} is not a vector of positive elements summing to 1"
                )));
            }
            for (l, x) in log_x.iter_mut().zip(x) {
                *l = x.ln();
            }
            // Vᵀ is orthogonal to the ones vector, so the log-ratio's normaliser drops out.
            Self::basis_transpose_times(&log_x, &mut y[row * (k - 1)..(row + 1) * (k - 1)]);
        }
        Ok(y)
    }

    fn log_det_jacobian(&self, y: &[f64]) -> Result<f64> {
        check_len(
            "the unconstrained vector",
            y.len(),
            self.unconstrained_len(),
        )?;
        let half_log_k = 0.5 * (self.k as f64).ln();
        let mut total = 0.0;
        self.for_each_row(y, |_, z, _, log_sum| {
            // Σ log xᵢ with log xᵢ = zᵢ − log Σ exp(z), finite even where x underflows
            total += z.iter().map(|z| z - log_sum).sum::<f64>() + half_log_k;
        });
        Ok(total)
    }

    fn log_det_jacobian_gradient(&self, y: &[f64]) -> Result<Vec<f64>> {
        check_len(
            "the unconstrained vector",
            y.len(),
            self.unconstrained_len(),
        )?;
        let k = self.k;
        let mut g = vec![0.0; y.len()];
        let mut gz = vec![0.0; k];
        self.for_each_row(y, |row, _, x, _| {
            // ∂/∂z Σᵢ (zᵢ − log Σ exp z) = 1 − K x
            for (g, x) in gz.iter_mut().zip(x) {
                *g = 1.0 - k as f64 * x;
            }
            Self::basis_transpose_times(&gz, &mut g[row * (k - 1)..(row + 1) * (k - 1)]);
        });
        Ok(g)
    }

    fn forward_jvp(&self, y: &[f64], tangent: &[f64]) -> Result<DoubleTensor> {
        check_len(
            "the unconstrained vector",
            y.len(),
            self.unconstrained_len(),
        )?;
        check_len("the tangent", tangent.len(), self.unconstrained_len())?;
        let k = self.k;
        let tangents: Vec<&[f64]> = self.rows(tangent).collect();
        let mut dz = vec![0.0; k];
        let mut out = Vec::with_capacity(self.batch() * k);
        self.for_each_row(y, |row, _, x, _| {
            Self::basis_times(tangents[row], &mut dz);
            let m: f64 = x.iter().zip(&dz).map(|(x, d)| x * d).sum();
            out.extend(x.iter().zip(&dz).map(|(x, d)| x * (d - m)));
        });
        tensor(&self.shape, out)
    }

    fn forward_vjp(&self, y: &[f64], cotangent: &DoubleTensor) -> Result<Vec<f64>> {
        check_len(
            "the unconstrained vector",
            y.len(),
            self.unconstrained_len(),
        )?;
        check_shape("the cotangent", cotangent, &self.shape)?;
        let k = self.k;
        let c = cotangent.as_slice();
        let mut g = vec![0.0; y.len()];
        let mut gz = vec![0.0; k];
        self.for_each_row(y, |row, _, x, _| {
            let c = &c[row * k..(row + 1) * k];
            let m: f64 = x.iter().zip(c).map(|(x, c)| x * c).sum();
            for ((g, x), c) in gz.iter_mut().zip(x).zip(c) {
                *g = x * (c - m);
            }
            Self::basis_transpose_times(&gz, &mut g[row * (k - 1)..(row + 1) * (k - 1)]);
        });
        Ok(g)
    }

    fn forward_vjp_jvp(
        &self,
        y: &[f64],
        tangent: &[f64],
        cotangent: &DoubleTensor,
    ) -> Result<Vec<f64>> {
        check_second_order(y, tangent, self.unconstrained_len())?;
        check_shape("the cotangent", cotangent, &self.shape)?;
        let k = self.k;
        let c = cotangent.as_slice();
        let tangents: Vec<&[f64]> = self.rows(tangent).collect();
        let mut g = vec![0.0; y.len()];
        let (mut dz, mut dx, mut gz) = (vec![0.0; k], vec![0.0; k], vec![0.0; k]);
        self.for_each_row(y, |row, _, x, _| {
            // the vector-Jacobian product is Vᵀ (x ∘ (c − x·c)); along t, ẋ = x ∘ (ż − x·ż) with ż = V t, and its
            // derivative is Vᵀ (ẋ ∘ (c − x·c) − x (ẋ·c))
            let c = &c[row * k..(row + 1) * k];
            Self::basis_times(tangents[row], &mut dz);
            let xdz: f64 = x.iter().zip(&dz).map(|(x, d)| x * d).sum();
            for ((dx, x), dz) in dx.iter_mut().zip(x).zip(&dz) {
                *dx = x * (dz - xdz);
            }
            let m: f64 = x.iter().zip(c).map(|(x, c)| x * c).sum();
            let dm: f64 = dx.iter().zip(c).map(|(d, c)| d * c).sum();
            for (((g, x), dx), c) in gz.iter_mut().zip(x).zip(&dx).zip(c) {
                *g = dx * (c - m) - x * dm;
            }
            Self::basis_transpose_times(&gz, &mut g[row * (k - 1)..(row + 1) * (k - 1)]);
        });
        Ok(g)
    }

    fn log_det_jacobian_hvp(&self, y: &[f64], tangent: &[f64]) -> Result<Vec<f64>> {
        check_second_order(y, tangent, self.unconstrained_len())?;
        let k = self.k;
        let tangents: Vec<&[f64]> = self.rows(tangent).collect();
        let mut g = vec![0.0; y.len()];
        let (mut dz, mut gz) = (vec![0.0; k], vec![0.0; k]);
        self.for_each_row(y, |row, _, x, _| {
            // the gradient is Vᵀ (1 − K x), so the product is Vᵀ (−K ẋ)
            Self::basis_times(tangents[row], &mut dz);
            let xdz: f64 = x.iter().zip(&dz).map(|(x, d)| x * d).sum();
            for ((g, x), dz) in gz.iter_mut().zip(x).zip(&dz) {
                *g = -(k as f64) * x * (dz - xdz);
            }
            Self::basis_transpose_times(&gz, &mut g[row * (k - 1)..(row + 1) * (k - 1)]);
        });
        Ok(g)
    }
}

/// Lower triangular Cholesky factors of correlation matrices over the last two axes, for
/// [`Support::CorrelationCholesky`], as in the Stan reference manual ("Cholesky factors of correlation matrices").
/// Leading axes are a batch.
///
/// The K(K − 1)/2 unconstrained coordinates of each matrix, in the row-major order of its strictly lower triangle,
/// are mapped to canonical partial correlations `zᵢⱼ = tanh(yᵢⱼ)`, and then row by row
/// `Lᵢ₀ = zᵢ₀`, `Lᵢⱼ = zᵢⱼ √(1 − Σₘ₍ⱼ Lᵢₘ²)` and `Lᵢᵢ = √(1 − Σₘ₍ᵢ Lᵢₘ²)`, with `L₀₀ = 1`. Each row has unit
/// length and the diagonal is positive. The log-Jacobian of the map from `y` to the strictly lower triangle is
/// `−Σᵢⱼ (i + 1 − j) log cosh yᵢⱼ` (zero-based `i > j`), Stan's `Σ log(1 − z²) + ½ Σ log(1 − Σ L²)` written in
/// `y`.
#[derive(Debug, Clone, PartialEq)]
pub struct CorrelationCholesky {
    shape: Vec<usize>,
    k: usize,
}

/// The tolerance on a correlation Cholesky factor's row lengths and upper triangle.
const CHOLESKY_TOLERANCE: f64 = 1e-8;

impl CorrelationCholesky {
    /// Correlation Cholesky factors over the last two axes of `shape`.
    ///
    /// # Errors
    ///
    /// [`GraphError::Transform`] if `shape` has fewer than two axes or its last two axes differ.
    pub fn new(shape: &[usize]) -> Result<Self> {
        match shape {
            [.., rows, cols] if rows == cols => Ok(CorrelationCholesky {
                shape: shape.to_vec(),
                k: *rows,
            }),
            _ => Err(transform_error(format!(
                "a correlation Cholesky factor needs a shape ending in two equal axes, not {shape:?}"
            ))),
        }
    }

    fn batch(&self) -> usize {
        self.shape[..self.shape.len() - 2].iter().product()
    }

    fn per_matrix(&self) -> usize {
        self.k * self.k.saturating_sub(1) / 2
    }

    /// Calls `f(matrix, row, y)` with the unconstrained coordinates of each row of each matrix (row `i` has `i`).
    fn for_each_row<'a>(&self, y: &'a [f64], mut f: impl FnMut(usize, usize, &'a [f64])) {
        let n = self.per_matrix();
        for m in 0..self.batch() {
            let y = &y[m * n..(m + 1) * n];
            for i in 0..self.k {
                let start = i * i.saturating_sub(1) / 2;
                f(m, i, &y[start..start + i]);
            }
        }
    }
}

impl Bijector for CorrelationCholesky {
    fn constrained_shape(&self) -> &[usize] {
        &self.shape
    }

    fn unconstrained_len(&self) -> usize {
        self.batch() * self.per_matrix()
    }

    fn forward(&self, y: &[f64]) -> Result<DoubleTensor> {
        check_len(
            "the unconstrained vector",
            y.len(),
            self.unconstrained_len(),
        )?;
        let k = self.k;
        let mut l = vec![0.0; self.batch() * k * k];
        self.for_each_row(y, |m, i, y| {
            let row = &mut l[(m * k + i) * k..(m * k + i + 1) * k];
            // s is the length still available to the row: √(1 − Σ L²) = Π sech yₘ
            let mut s = 1.0;
            for (j, &y) in y.iter().enumerate() {
                row[j] = y.tanh() * s;
                s /= y.cosh();
            }
            row[i] = s;
        });
        tensor(&self.shape, l)
    }

    fn inverse(&self, x: &DoubleTensor) -> Result<Vec<f64>> {
        check_shape("the value", x, &self.shape)?;
        let k = self.k;
        let mut y = Vec::with_capacity(self.unconstrained_len());
        for (index, row) in rows(x.as_slice(), k).enumerate() {
            let i = index % k;
            let invalid = || {
                transform_error(format!(
                    "row {i} of a correlation Cholesky factor, {row:?}, does not have unit length, a positive \
                     diagonal and zeros above the diagonal"
                ))
            };
            let norm: f64 = row.iter().map(|v| v * v).sum();
            // written so that NaN fails every check
            let valid = row[i] > 0.0
                && (norm - 1.0).abs() <= CHOLESKY_TOLERANCE
                && row[i + 1..].iter().all(|v| v.abs() <= CHOLESKY_TOLERANCE);
            if !valid {
                return Err(invalid());
            }
            // zᵢⱼ = Lᵢⱼ / √(Σₘ₌ⱼ..ᵢ Lᵢₘ²), from tail sums so the diagonal's share stays accurate
            let mut tail: f64 = row[..=i].iter().map(|v| v * v).sum();
            for &v in &row[..i] {
                let z = v / tail.sqrt();
                let yij = z.atanh();
                if !yij.is_finite() {
                    return Err(invalid());
                }
                y.push(yij);
                tail -= v * v;
            }
        }
        Ok(y)
    }

    fn log_det_jacobian(&self, y: &[f64]) -> Result<f64> {
        check_len(
            "the unconstrained vector",
            y.len(),
            self.unconstrained_len(),
        )?;
        let mut total = 0.0;
        self.for_each_row(y, |_, i, y| {
            total -= y
                .iter()
                .enumerate()
                .map(|(j, &y)| (i + 1 - j) as f64 * log_cosh(y))
                .sum::<f64>();
        });
        Ok(total)
    }

    fn log_det_jacobian_gradient(&self, y: &[f64]) -> Result<Vec<f64>> {
        check_len(
            "the unconstrained vector",
            y.len(),
            self.unconstrained_len(),
        )?;
        let mut g = Vec::with_capacity(y.len());
        self.for_each_row(y, |_, i, y| {
            g.extend(
                y.iter()
                    .enumerate()
                    .map(|(j, &y)| -((i + 1 - j) as f64) * y.tanh()),
            );
        });
        Ok(g)
    }

    fn forward_jvp(&self, y: &[f64], tangent: &[f64]) -> Result<DoubleTensor> {
        check_len(
            "the unconstrained vector",
            y.len(),
            self.unconstrained_len(),
        )?;
        check_len("the tangent", tangent.len(), self.unconstrained_len())?;
        let k = self.k;
        let mut dl = vec![0.0; self.batch() * k * k];
        let mut offset = 0;
        self.for_each_row(y, |m, i, y| {
            let t = &tangent[offset..offset + i];
            offset += i;
            let row = &mut dl[(m * k + i) * k..(m * k + i + 1) * k];
            // With c = sech y: dz = c² dy, dc = −c z dy, and s ← s c.
            let (mut s, mut ds) = (1.0, 0.0);
            for (j, (&y, &t)) in y.iter().zip(t).enumerate() {
                let (z, c) = (y.tanh(), 1.0 / y.cosh());
                row[j] = c * c * t * s + z * ds;
                ds = ds * c - s * c * z * t;
                s *= c;
            }
            row[i] = ds;
        });
        tensor(&self.shape, dl)
    }

    fn forward_vjp(&self, y: &[f64], cotangent: &DoubleTensor) -> Result<Vec<f64>> {
        check_len(
            "the unconstrained vector",
            y.len(),
            self.unconstrained_len(),
        )?;
        check_shape("the cotangent", cotangent, &self.shape)?;
        let k = self.k;
        let c = cotangent.as_slice();
        let mut g = vec![0.0; y.len()];
        let mut offset = 0;
        let mut s_before = vec![0.0; k];
        self.for_each_row(y, |m, i, y| {
            let cot = &c[(m * k + i) * k..(m * k + i + 1) * k];
            let g = &mut g[offset..offset + i];
            offset += i;
            // forward pass: the remaining length before each element
            let mut s = 1.0;
            for (j, &y) in y.iter().enumerate() {
                s_before[j] = s;
                s /= y.cosh();
            }
            // reverse pass: Lᵢᵢ = s, Lᵢⱼ = zⱼ s₍ⱼ₋₁₎, sⱼ = s₍ⱼ₋₁₎ cⱼ
            let mut gs = cot[i];
            for j in (0..i).rev() {
                let (z, sech) = (y[j].tanh(), 1.0 / y[j].cosh());
                let before = s_before[j];
                g[j] = cot[j] * before * sech * sech - gs * before * sech * z;
                gs = gs * sech + cot[j] * z;
            }
        });
        Ok(g)
    }

    fn forward_vjp_jvp(
        &self,
        y: &[f64],
        tangent: &[f64],
        cotangent: &DoubleTensor,
    ) -> Result<Vec<f64>> {
        check_second_order(y, tangent, self.unconstrained_len())?;
        check_shape("the cotangent", cotangent, &self.shape)?;
        let k = self.k;
        let c = cotangent.as_slice();
        let mut g = vec![0.0; y.len()];
        let mut offset = 0;
        let mut s_before = vec![Dual::constant(0.0); k];
        let mut duals = Vec::with_capacity(k);
        self.for_each_row(y, |m, i, y| {
            // forward_vjp's recurrences evaluated on dual numbers yⱼ + ε tⱼ: the ε parts are the derivative
            let cot = &c[(m * k + i) * k..(m * k + i + 1) * k];
            let t = &tangent[offset..offset + i];
            let g = &mut g[offset..offset + i];
            offset += i;
            duals.clear();
            duals.extend(y.iter().zip(t).map(|(&y, &t)| {
                let (z, sech) = (y.tanh(), 1.0 / y.cosh());
                (Dual(z, (1.0 - z * z) * t), Dual(sech, -sech * z * t))
            }));
            let mut s = Dual::constant(1.0);
            for (j, &(_, sech)) in duals.iter().enumerate() {
                s_before[j] = s;
                s = s * sech;
            }
            let mut gs = Dual::constant(cot[i]);
            for j in (0..i).rev() {
                let (z, sech) = duals[j];
                let before = s_before[j];
                let gj = before * sech * sech * cot[j] - gs * before * sech * z;
                g[j] = gj.1;
                gs = gs * sech + z * cot[j];
            }
        });
        Ok(g)
    }

    fn log_det_jacobian_hvp(&self, y: &[f64], tangent: &[f64]) -> Result<Vec<f64>> {
        check_second_order(y, tangent, self.unconstrained_len())?;
        let mut h = Vec::with_capacity(y.len());
        let mut offset = 0;
        self.for_each_row(y, |_, i, y| {
            // the gradient is −(i + 1 − j) tanh yᵢⱼ, so the Hessian is diagonal, −(i + 1 − j) sech² yᵢⱼ
            let t = &tangent[offset..offset + i];
            offset += i;
            h.extend(y.iter().zip(t).enumerate().map(|(j, (&y, &t))| {
                let sech = 1.0 / y.cosh();
                -((i + 1 - j) as f64) * sech * sech * t
            }));
        });
        Ok(h)
    }
}

/// Angles, for [`Support::Circular`], as Stan's unit vectors: each angle has two unconstrained coordinates `(u, v)`
/// (interleaved, `[u₀, v₀, u₁, v₁, …]`) and `θ = atan2(v, u)`, in (−π, π].
///
/// This is not a bijection: every point on a ray from the origin maps to the same angle. A standard normal density
/// on the radius identifies it, so [`log_det_jacobian`](Bijector::log_det_jacobian) returns `Σ −½(u² + v²)`, which
/// is not a Jacobian determinant but the term that makes the density over `(u, v)` proper and its marginal over
/// `θ` the angle's density (up to a constant). [`inverse`](Bijector::inverse) maps `θ` to `(cos θ, sin θ)`, so
/// `forward(inverse(θ))` is `θ` wrapped into (−π, π], and `inverse(forward(y))` is `y` scaled to unit length. The
/// derivatives of `θ` don't exist at the origin, where the products return an error.
#[derive(Debug, Clone, PartialEq)]
pub struct Circular {
    shape: Vec<usize>,
}

impl Circular {
    /// Angles of `shape`.
    pub fn new(shape: &[usize]) -> Self {
        Circular {
            shape: shape.to_vec(),
        }
    }

    fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// The squared radius of a point, failing at the origin.
    fn radius_squared(u: f64, v: f64) -> Result<f64> {
        let r2 = u * u + v * v;
        if r2 > 0.0 {
            Ok(r2)
        } else {
            Err(transform_error(
                "an angle has no derivative at the origin of its unconstrained coordinates",
            ))
        }
    }
}

impl Bijector for Circular {
    fn constrained_shape(&self) -> &[usize] {
        &self.shape
    }

    fn unconstrained_len(&self) -> usize {
        2 * self.len()
    }

    fn forward(&self, y: &[f64]) -> Result<DoubleTensor> {
        check_len("the unconstrained vector", y.len(), 2 * self.len())?;
        let theta = y.chunks_exact(2).map(|p| p[1].atan2(p[0])).collect();
        tensor(&self.shape, theta)
    }

    fn inverse(&self, x: &DoubleTensor) -> Result<Vec<f64>> {
        check_shape("the value", x, &self.shape)?;
        let mut y = Vec::with_capacity(2 * self.len());
        for &theta in x.as_slice() {
            if !theta.is_finite() {
                return Err(transform_error(format!(
                    "an angle must be finite, not {theta}"
                )));
            }
            let (sin, cos) = theta.sin_cos();
            y.push(cos);
            y.push(sin);
        }
        Ok(y)
    }

    fn log_det_jacobian(&self, y: &[f64]) -> Result<f64> {
        check_len("the unconstrained vector", y.len(), 2 * self.len())?;
        Ok(-0.5 * y.iter().map(|v| v * v).sum::<f64>())
    }

    fn log_det_jacobian_gradient(&self, y: &[f64]) -> Result<Vec<f64>> {
        check_len("the unconstrained vector", y.len(), 2 * self.len())?;
        Ok(y.iter().map(|v| -v).collect())
    }

    fn forward_jvp(&self, y: &[f64], tangent: &[f64]) -> Result<DoubleTensor> {
        check_len("the unconstrained vector", y.len(), 2 * self.len())?;
        check_len("the tangent", tangent.len(), 2 * self.len())?;
        let dtheta = y
            .chunks_exact(2)
            .zip(tangent.chunks_exact(2))
            .map(|(p, t)| Ok((p[0] * t[1] - p[1] * t[0]) / Self::radius_squared(p[0], p[1])?))
            .collect::<Result<Vec<f64>>>()?;
        tensor(&self.shape, dtheta)
    }

    fn forward_vjp(&self, y: &[f64], cotangent: &DoubleTensor) -> Result<Vec<f64>> {
        check_len("the unconstrained vector", y.len(), 2 * self.len())?;
        check_shape("the cotangent", cotangent, &self.shape)?;
        let mut g = Vec::with_capacity(y.len());
        for (p, &c) in y.chunks_exact(2).zip(cotangent.as_slice()) {
            let r2 = Self::radius_squared(p[0], p[1])?;
            g.push(-p[1] * c / r2);
            g.push(p[0] * c / r2);
        }
        Ok(g)
    }

    fn forward_vjp_jvp(
        &self,
        y: &[f64],
        tangent: &[f64],
        cotangent: &DoubleTensor,
    ) -> Result<Vec<f64>> {
        check_second_order(y, tangent, 2 * self.len())?;
        check_shape("the cotangent", cotangent, &self.shape)?;
        let mut g = Vec::with_capacity(y.len());
        for ((p, t), &c) in y
            .chunks_exact(2)
            .zip(tangent.chunks_exact(2))
            .zip(cotangent.as_slice())
        {
            // the derivatives of c (−v, u) / r² along (tu, tv), with the derivative of r² being 2 (u tu + v tv)
            let r2 = Self::radius_squared(p[0], p[1])?;
            let dr2 = 2.0 * (p[0] * t[0] + p[1] * t[1]);
            g.push(c * (-t[1] / r2 + p[1] * dr2 / (r2 * r2)));
            g.push(c * (t[0] / r2 - p[0] * dr2 / (r2 * r2)));
        }
        Ok(g)
    }

    fn log_det_jacobian_hvp(&self, y: &[f64], tangent: &[f64]) -> Result<Vec<f64>> {
        // −½ Σ (u² + v²) has Hessian −I
        check_second_order(y, tangent, 2 * self.len())?;
        Ok(tangent.iter().map(|t| -t).collect())
    }
}

/// A dual number `a + b ε` with `ε² = 0`: the value and directional derivative of a quantity, for derivatives of
/// scalar recurrences.
#[derive(Debug, Clone, Copy)]
struct Dual(f64, f64);

impl Dual {
    fn constant(value: f64) -> Self {
        Dual(value, 0.0)
    }
}

impl std::ops::Mul for Dual {
    type Output = Dual;
    fn mul(self, other: Dual) -> Dual {
        Dual(self.0 * other.0, self.0 * other.1 + self.1 * other.0)
    }
}

impl std::ops::Mul<f64> for Dual {
    type Output = Dual;
    fn mul(self, other: f64) -> Dual {
        Dual(self.0 * other, self.1 * other)
    }
}

impl std::ops::Add for Dual {
    type Output = Dual;
    fn add(self, other: Dual) -> Dual {
        Dual(self.0 + other.0, self.1 + other.1)
    }
}

impl std::ops::Sub for Dual {
    type Output = Dual;
    fn sub(self, other: Dual) -> Dual {
        Dual(self.0 - other.0, self.1 - other.1)
    }
}
