//! The element-wise bounded bijectors: lower and upper bounds and intervals.

use super::*;

/// The transform of one element between a lower and an upper bound, either of which may be infinite.
#[derive(Debug, Clone, Copy, PartialEq)]
enum ElementBounds {
    /// Both bounds infinite: `x = y`.
    Free,
    /// A finite lower bound only: `x = a + exp(y)`.
    Lower(f64),
    /// A finite upper bound only: `x = b − exp(y)`.
    Upper(f64),
    /// Both bounds finite: `x = a + (b − a)·logistic(y)`.
    Both(f64, f64),
}

impl ElementBounds {
    fn new(a: f64, b: f64) -> Self {
        match (a.is_finite(), b.is_finite()) {
            (false, false) => ElementBounds::Free,
            (true, false) => ElementBounds::Lower(a),
            (false, true) => ElementBounds::Upper(b),
            (true, true) => ElementBounds::Both(a, b),
        }
    }

    /// The value, strictly inside the bounds: where the exact value rounds to a bound (for (0, 1), y above about
    /// 37 or below about −745), the nearest float inside it. A density can be +∞ at a bound (a Beta's with a shape
    /// below 1), so a value on it would be an impossible point the sampler could never leave.
    fn forward(self, y: f64) -> f64 {
        match self {
            ElementBounds::Free => y,
            ElementBounds::Lower(a) => inside(a + y.exp(), next_up(a), f64::MAX),
            ElementBounds::Upper(b) => inside(b - y.exp(), f64::MIN, next_down(b)),
            // for positive y the value is computed from the upper bound, so values near either bound keep their
            // precision
            ElementBounds::Both(a, b) if y > 0.0 => {
                inside(b - (b - a) * logistic(-y), next_up(a), next_down(b))
            }
            ElementBounds::Both(a, b) => {
                inside(a + (b - a) * logistic(y), next_up(a), next_down(b))
            }
        }
    }

    fn inverse(self, x: f64) -> Result<f64> {
        let (y, message) = match self {
            ElementBounds::Free => (x, format!("a real value must be finite, not {x}")),
            ElementBounds::Lower(a) => (
                (x - a).ln(),
                format!("{x} is not a finite value greater than the lower bound {a}"),
            ),
            ElementBounds::Upper(b) => (
                (b - x).ln(),
                format!("{x} is not a finite value less than the upper bound {b}"),
            ),
            ElementBounds::Both(a, b) => (
                if x > a && x < b {
                    (x - a).ln() - (b - x).ln()
                } else {
                    f64::NAN
                },
                format!("{x} is not strictly inside the interval ({a}, {b})"),
            ),
        };
        if y.is_finite() {
            Ok(y)
        } else {
            Err(transform_error(message))
        }
    }

    fn log_det_jacobian(self, y: f64) -> f64 {
        match self {
            ElementBounds::Free => 0.0,
            ElementBounds::Lower(_) | ElementBounds::Upper(_) => y,
            ElementBounds::Both(a, b) => (b - a).ln() - log1p_exp(-y) - log1p_exp(y),
        }
    }

    fn log_det_jacobian_gradient(self, y: f64) -> f64 {
        match self {
            ElementBounds::Free => 0.0,
            ElementBounds::Lower(_) | ElementBounds::Upper(_) => 1.0,
            ElementBounds::Both(..) => logistic(-y) - logistic(y),
        }
    }

    /// The second derivative of the log-Jacobian: −2 s (1 − s) for an interval, else 0.
    fn log_det_jacobian_second(self, y: f64) -> f64 {
        match self {
            ElementBounds::Both(..) => -2.0 * logistic(y) * logistic(-y),
            _ => 0.0,
        }
    }

    /// dx/dy.
    fn derivative(self, y: f64) -> f64 {
        match self {
            ElementBounds::Free => 1.0,
            ElementBounds::Lower(_) => y.exp(),
            ElementBounds::Upper(_) => -y.exp(),
            ElementBounds::Both(a, b) => (b - a) * logistic(y) * logistic(-y),
        }
    }

    /// d²x/dy²: for an interval x' = (b − a) s (1 − s), so x'' = (b − a) s (1 − s) (1 − 2s).
    fn second_derivative(self, y: f64) -> f64 {
        match self {
            ElementBounds::Free => 0.0,
            ElementBounds::Lower(_) => y.exp(),
            ElementBounds::Upper(_) => -y.exp(),
            ElementBounds::Both(a, b) => {
                let (s, r) = (logistic(y), logistic(-y));
                (b - a) * s * r * (r - s)
            }
        }
    }
}

/// The least float greater than `x` (`f64::next_up`, which needs a newer Rust than the MSRV), for finite `x`.
fn next_up(x: f64) -> f64 {
    if x == 0.0 {
        f64::from_bits(1)
    } else if x > 0.0 {
        f64::from_bits(x.to_bits() + 1)
    } else {
        f64::from_bits(x.to_bits() - 1)
    }
}

/// The greatest float less than `x`, for finite `x`.
fn next_down(x: f64) -> f64 {
    -next_up(-x)
}

/// `x` moved into [lo, hi] (never panicking, unlike `clamp`, when the bounds are adjacent floats and so cross).
fn inside(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

/// Values of a fixed shape between element-wise bounds, each element transformed by its [`ElementBounds`]: the
/// shared implementation of [`LowerBound`], [`UpperBound`] and [`Interval`].
#[derive(Debug, Clone, PartialEq)]
struct Bounded {
    shape: Vec<usize>,
    elements: Vec<ElementBounds>,
}

impl Bounded {
    /// Checks the bounds (already broadcast and checked against NaN and the forbidden infinities) are in order.
    fn new(lower: &[f64], upper: &[f64], shape: &[usize]) -> Result<Self> {
        if let Some((a, b)) = lower.iter().zip(upper).find(|(a, b)| a >= b) {
            return Err(transform_error(format!(
                "an interval's lower bound must be less than its upper bound, not [{a}, {b}]"
            )));
        }
        Ok(Bounded {
            shape: shape.to_vec(),
            elements: lower
                .iter()
                .zip(upper)
                .map(|(&a, &b)| ElementBounds::new(a, b))
                .collect(),
        })
    }

    fn len(&self) -> usize {
        self.elements.len()
    }

    fn check_y(&self, y: &[f64]) -> Result<()> {
        check_len("the unconstrained vector", y.len(), self.len())
    }

    fn map(&self, y: &[f64], f: impl Fn(ElementBounds, f64) -> f64) -> Vec<f64> {
        self.elements
            .iter()
            .zip(y)
            .map(|(&e, &y)| f(e, y))
            .collect()
    }

    fn forward(&self, y: &[f64]) -> Result<DoubleTensor> {
        self.check_y(y)?;
        tensor(&self.shape, self.map(y, ElementBounds::forward))
    }

    fn inverse(&self, x: &DoubleTensor) -> Result<Vec<f64>> {
        check_shape("the value", x, &self.shape)?;
        self.elements
            .iter()
            .zip(x.as_slice())
            .map(|(&e, &x)| e.inverse(x))
            .collect()
    }

    fn log_det_jacobian(&self, y: &[f64]) -> Result<f64> {
        self.check_y(y)?;
        Ok(self.map(y, ElementBounds::log_det_jacobian).iter().sum())
    }

    fn log_det_jacobian_gradient(&self, y: &[f64]) -> Result<Vec<f64>> {
        self.check_y(y)?;
        Ok(self.map(y, ElementBounds::log_det_jacobian_gradient))
    }

    fn forward_jvp(&self, y: &[f64], tangent: &[f64]) -> Result<DoubleTensor> {
        self.check_y(y)?;
        check_len("the tangent", tangent.len(), self.len())?;
        let dx = self
            .map(y, ElementBounds::derivative)
            .iter()
            .zip(tangent)
            .map(|(d, t)| d * t)
            .collect();
        tensor(&self.shape, dx)
    }

    fn forward_vjp(&self, y: &[f64], cotangent: &DoubleTensor) -> Result<Vec<f64>> {
        self.check_y(y)?;
        check_shape("the cotangent", cotangent, &self.shape)?;
        Ok(self
            .map(y, ElementBounds::derivative)
            .iter()
            .zip(cotangent.as_slice())
            // where the value is pinned next to a bound the derivative underflows to zero, and the log density's
            // derivative there may overflow: the map is flat, so the product is zero rather than NaN
            .map(|(&d, c)| if d == 0.0 { 0.0 } else { d * c })
            .collect())
    }

    fn forward_vjp_jvp(
        &self,
        y: &[f64],
        tangent: &[f64],
        cotangent: &DoubleTensor,
    ) -> Result<Vec<f64>> {
        check_second_order(y, tangent, self.len())?;
        check_shape("the cotangent", cotangent, &self.shape)?;
        // the vector-Jacobian product is x'(y) c, element-wise
        Ok(self
            .map(y, ElementBounds::second_derivative)
            .iter()
            .zip(tangent)
            .zip(cotangent.as_slice())
            .map(|((d2, t), c)| d2 * c * t)
            .collect())
    }

    fn log_det_jacobian_hvp(&self, y: &[f64], tangent: &[f64]) -> Result<Vec<f64>> {
        check_second_order(y, tangent, self.len())?;
        // the log-Jacobian is a sum of element terms, so its Hessian is diagonal
        Ok(self
            .map(y, ElementBounds::log_det_jacobian_second)
            .iter()
            .zip(tangent)
            .map(|(h, t)| h * t)
            .collect())
    }
}

/// Implements [`Bijector`] for a wrapper of [`Bounded`] by delegation.
macro_rules! bounded_bijector {
    ($name:ident) => {
        impl Bijector for $name {
            fn constrained_shape(&self) -> &[usize] {
                &self.0.shape
            }

            fn unconstrained_len(&self) -> usize {
                self.0.len()
            }

            fn forward(&self, y: &[f64]) -> Result<DoubleTensor> {
                self.0.forward(y)
            }

            fn inverse(&self, x: &DoubleTensor) -> Result<Vec<f64>> {
                self.0.inverse(x)
            }

            fn log_det_jacobian(&self, y: &[f64]) -> Result<f64> {
                self.0.log_det_jacobian(y)
            }

            fn log_det_jacobian_gradient(&self, y: &[f64]) -> Result<Vec<f64>> {
                self.0.log_det_jacobian_gradient(y)
            }

            fn forward_jvp(&self, y: &[f64], tangent: &[f64]) -> Result<DoubleTensor> {
                self.0.forward_jvp(y, tangent)
            }

            fn forward_vjp(&self, y: &[f64], cotangent: &DoubleTensor) -> Result<Vec<f64>> {
                self.0.forward_vjp(y, cotangent)
            }

            fn forward_vjp_jvp(
                &self,
                y: &[f64],
                tangent: &[f64],
                cotangent: &DoubleTensor,
            ) -> Result<Vec<f64>> {
                self.0.forward_vjp_jvp(y, tangent, cotangent)
            }

            fn log_det_jacobian_hvp(&self, y: &[f64], tangent: &[f64]) -> Result<Vec<f64>> {
                self.0.log_det_jacobian_hvp(y, tangent)
            }
        }
    };
}

/// A lower bound, for [`Support::GreaterThan`]: `x = a + exp(y)` element-wise, with log-Jacobian `Σ y`. An element
/// whose bound is −∞ is unbounded, and transformed by the identity.
#[derive(Debug, Clone, PartialEq)]
pub struct LowerBound(Bounded);

impl LowerBound {
    /// Values of `shape` greater than `lower`, which is broadcast to `shape`.
    ///
    /// # Errors
    ///
    /// [`GraphError::Transform`] if the bound doesn't broadcast to `shape` or is NaN or +∞.
    pub fn new(lower: &DoubleTensor, shape: &[usize]) -> Result<Self> {
        let lower = bound("lower", lower, shape, f64::INFINITY)?;
        let upper = vec![f64::INFINITY; lower.len()];
        Ok(LowerBound(Bounded::new(&lower, &upper, shape)?))
    }
}

bounded_bijector!(LowerBound);

/// An upper bound, for [`Support::LessThan`]: `x = b − exp(y)` element-wise, with log-Jacobian `Σ y`. An element
/// whose bound is +∞ is unbounded, and transformed by the identity.
#[derive(Debug, Clone, PartialEq)]
pub struct UpperBound(Bounded);

impl UpperBound {
    /// Values of `shape` less than `upper`, which is broadcast to `shape`.
    ///
    /// # Errors
    ///
    /// [`GraphError::Transform`] if the bound doesn't broadcast to `shape` or is NaN or −∞.
    pub fn new(upper: &DoubleTensor, shape: &[usize]) -> Result<Self> {
        let upper = bound("upper", upper, shape, f64::NEG_INFINITY)?;
        let lower = vec![f64::NEG_INFINITY; upper.len()];
        Ok(UpperBound(Bounded::new(&lower, &upper, shape)?))
    }
}

bounded_bijector!(UpperBound);

/// An interval, for [`Support::Interval`]: `x = a + (b − a)·logistic(y)` element-wise, with log-Jacobian
/// `Σ log(b − a) + log s + log(1 − s)` where `s = logistic(y)`.
///
/// For positive `y` the value is computed from the upper bound, `x = b − (b − a)·logistic(−y)`, so values near
/// either bound keep their precision.
///
/// The bounds may differ in kind from element to element (as a truncation's do when only some of its elements are
/// truncated on a side): an element with only a finite lower bound is transformed as by [`LowerBound`], one with
/// only a finite upper bound as by [`UpperBound`], and one with neither by the identity.
#[derive(Debug, Clone, PartialEq)]
pub struct Interval(Bounded);

impl Interval {
    /// Values of `shape` between `lower` and `upper`, which are broadcast to `shape`.
    ///
    /// # Errors
    ///
    /// [`GraphError::Transform`] if a bound doesn't broadcast to `shape` or is NaN, a lower bound is +∞ or an upper
    /// bound −∞, or a lower bound isn't less than its upper bound.
    pub fn new(lower: &DoubleTensor, upper: &DoubleTensor, shape: &[usize]) -> Result<Self> {
        let lower = bound("lower", lower, shape, f64::INFINITY)?;
        let upper = bound("upper", upper, shape, f64::NEG_INFINITY)?;
        Ok(Interval(Bounded::new(&lower, &upper, shape)?))
    }
}

bounded_bijector!(Interval);
