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
//! | [`Support::SumToZero`] | [`SumToZero`] | `x = V y`, an orthonormal basis of the zero-sum subspace | 0 (isometric) |
//! | [`Support::SumToZeroComponents`] | [`SumToZeroComponents`] | [`SumToZero`]'s map on each group of elements, the others free | 0 (isometric) |
//! | [`Support::CorrelationCholesky`] | [`CorrelationCholesky`] | tanh canonical partial correlations | see the type |
//! | [`Support::CovarianceMatrix`] | [`CovarianceMatrix`] | `Σ = L Lᵀ`, `L` lower triangular with `Lᵢᵢ = exp(yᵢᵢ)` | `K ln 2 + Σᵢ (K − i + 1) yᵢᵢ` |
//! | [`Support::CovarianceCholesky`] | [`CovarianceCholesky`] | `L` lower triangular with `Lᵢᵢ = exp(yᵢᵢ)` | `Σᵢ yᵢᵢ` |
//! | [`Support::UnitVector`] | [`UnitVector`] | `x = y / ‖y‖` | `−½‖y‖²` (not a Jacobian, see the type) |
//! | [`Support::Circular`] | [`Circular`] | `θ = atan2(v, u)` | `−½(u² + v²)` (not a Jacobian, see the type) |
//!
//! The bounds of the three bounded supports apply element-wise and may be infinite on their own side, so one value
//! can mix kinds: each element uses the row of the table its own bounds call for (the identity where both are
//! infinite).
//!
//! Every log-Jacobian is summed over the whole value, and every value is laid out row-major, so the unconstrained
//! vector of a batch is the concatenation of the unconstrained vectors of its members.
//!
//! The unconstrained length can differ from the constrained value's size: a simplex or a zero-sum vector of K
//! elements has K − 1 free coordinates, a K × K correlation Cholesky factor has K(K − 1)/2, a K × K covariance
//! matrix or its Cholesky factor K(K + 1)/2, and an angle has 2. For the simplex, the correlation Cholesky factor,
//! the covariance matrix and the covariance Cholesky factor the Jacobian, and so the log-determinant, is taken with
//! respect to the free constrained coordinates: the first K − 1 elements of a simplex, the strictly lower triangle
//! (row by row) of a correlation Cholesky factor and the lower triangle of a covariance matrix or its factor, which
//! is what the Dirichlet, LKJ and Wishart densities are densities of.
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

mod bounded;
mod circular;
mod correlation_cholesky;
mod covariance;
mod identity;
mod ordered;
mod simplex;
mod sum_to_zero;
mod sum_to_zero_components;
mod unit_vector;

pub use bounded::{Interval, LowerBound, UpperBound};
pub use circular::Circular;
pub use correlation_cholesky::CorrelationCholesky;
pub use covariance::{CovarianceCholesky, CovarianceMatrix};
pub use identity::Identity;
pub use ordered::Ordered;
pub use simplex::Simplex;
pub use sum_to_zero::SumToZero;
pub use sum_to_zero_components::SumToZeroComponents;
pub use unit_vector::UnitVector;

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

    /// [`forward`](Self::forward) written into `out` (the constrained value's elements in row-major order). The
    /// default calls `forward`; bijectors on the hot path of samplers write directly.
    fn forward_into(&self, y: &[f64], out: &mut [f64]) -> Result<()> {
        let x = self.forward(y)?;
        check_len("the output", out.len(), x.len())?;
        out.copy_from_slice(x.as_slice());
        Ok(())
    }

    /// [`forward_vjp`](Self::forward_vjp) with the cotangent given as the constrained value's elements, written into
    /// `out` (of the unconstrained length). The default calls `forward_vjp`.
    fn forward_vjp_into(&self, y: &[f64], cotangent: &[f64], out: &mut [f64]) -> Result<()> {
        let cotangent = DoubleTensor::new(self.constrained_shape(), cotangent.to_vec())
            .map_err(|e| transform_error(e.to_string()))?;
        let gradient = self.forward_vjp(y, &cotangent)?;
        check_len("the output", out.len(), gradient.len())?;
        out.copy_from_slice(&gradient);
        Ok(())
    }

    /// [`log_det_jacobian`](Self::log_det_jacobian) at `y`, with its [gradient](Self::log_det_jacobian_gradient)
    /// added to `gradient` element by element. The default calls both.
    fn add_log_det_jacobian(&self, y: &[f64], gradient: &mut [f64]) -> Result<f64> {
        let log_det = self.log_det_jacobian(y)?;
        let jacobian_gradient = self.log_det_jacobian_gradient(y)?;
        check_len("the gradient", gradient.len(), jacobian_gradient.len())?;
        for (g, j) in gradient.iter_mut().zip(jacobian_gradient) {
            *g += j;
        }
        Ok(log_det)
    }

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
/// each lower bound must be less than its upper bound. A bound may be infinite on its own side (−∞ below, +∞ above),
/// leaving that element unbounded there, so the elements of one value may differ in kind: each is transformed by
/// the identity, `a + exp(y)`, `b − exp(y)` or the scaled logistic as its own bounds require. [`Support::Ordered`],
/// [`Support::Simplex`], [`Support::SumToZero`] and [`Support::UnitVector`] act on the last axis and
/// [`Support::CorrelationCholesky`], [`Support::CovarianceMatrix`] and [`Support::CovarianceCholesky`] on the last two
/// (which must be equal), with any leading axes as a batch.
///
/// # Errors
///
/// [`GraphError::Transform`] for [`Support::Discrete`] (a discrete value has no unconstrained form), for bounds
/// that are NaN, infinite on the wrong side (a lower bound of +∞, an upper bound of −∞), out of order or don't
/// broadcast to `shape`, and for a shape with too few axes for the
/// support.
pub fn bijector_for(support: &Support, shape: &[usize]) -> Result<Box<dyn Bijector>> {
    Ok(match support {
        Support::Real => Box::new(Identity::new(shape)),
        Support::GreaterThan(lower) => Box::new(LowerBound::new(lower, shape)?),
        Support::LessThan(upper) => Box::new(UpperBound::new(upper, shape)?),
        Support::Interval(lower, upper) => Box::new(Interval::new(lower, upper, shape)?),
        Support::Simplex => Box::new(Simplex::new(shape)?),
        Support::SumToZero => Box::new(SumToZero::new(shape)?),
        Support::SumToZeroComponents(groups) => Box::new(SumToZeroComponents::new(groups, shape)?),
        Support::Ordered => Box::new(Ordered::new(shape)?),
        Support::CorrelationCholesky => Box::new(CorrelationCholesky::new(shape)?),
        Support::CovarianceMatrix => Box::new(CovarianceMatrix::new(shape)?),
        Support::CovarianceCholesky => Box::new(CovarianceCholesky::new(shape)?),
        Support::UnitVector => Box::new(UnitVector::new(shape)?),
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

/// Broadcasts a bound to the value's shape and checks it is neither NaN nor `forbidden`, the infinity that would
/// leave no room for a value (+∞ for a lower bound, −∞ for an upper bound). The other infinity leaves the element
/// unbounded on that side.
fn bound(name: &str, bound: &DoubleTensor, shape: &[usize], forbidden: f64) -> Result<Vec<f64>> {
    let values = bound.broadcast_to(shape).map_err(|_| {
        transform_error(format!(
            "a {name} bound of shape {:?} does not broadcast to the value's shape {shape:?}",
            bound.shape()
        ))
    })?;
    if let Some(v) = values
        .as_slice()
        .iter()
        .find(|&&v| v.is_nan() || v == forbidden)
    {
        return Err(transform_error(format!(
            "a {name} bound must be neither NaN nor {forbidden}, not {v}"
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
