//! The transform for unit vectors.

use super::*;

/// The tolerance on a unit vector's squared length.
const TOLERANCE: f64 = 1e-8;

/// Unit vectors over the last axis, for [`Support::UnitVector`], as Stan's `unit_vector`: each vector of K elements
/// has K unconstrained coordinates `y` and `x = y / ‖y‖`. Leading axes are a batch.
///
/// This is not a bijection: every point on a ray from the origin maps to the same unit vector. A standard normal
/// density on `y` identifies it, so [`log_det_jacobian`](Bijector::log_det_jacobian) returns `Σ −½‖y‖²`, which is not
/// a Jacobian determinant but the term that makes the density over `y` proper and its marginal over `x` the unit
/// vector's density (up to a constant), as Stan does. [`inverse`](Bijector::inverse) maps `x` to itself, so
/// `inverse(forward(y))` is `y` scaled to unit length. The derivatives don't exist at the origin, where the forward
/// map and the products return an error.
#[derive(Debug, Clone, PartialEq)]
pub struct UnitVector {
    shape: Vec<usize>,
    k: usize,
}

impl UnitVector {
    /// Unit vectors over the last axis of `shape`.
    ///
    /// # Errors
    ///
    /// [`GraphError::Transform`] for a scalar shape.
    pub fn new(shape: &[usize]) -> Result<Self> {
        let (_, k) = last_axis("unit vector", shape)?;
        Ok(Self {
            shape: shape.to_vec(),
            k,
        })
    }

    fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// The length of each vector of `y`, failing at the origin.
    fn radius(y: &[f64]) -> Result<f64> {
        let r = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        if r > 0.0 && r.is_finite() {
            Ok(r)
        } else {
            Err(transform_error(
                "a unit vector has no direction (or derivative) at the origin of its unconstrained coordinates",
            ))
        }
    }
}

impl Bijector for UnitVector {
    fn constrained_shape(&self) -> &[usize] {
        &self.shape
    }

    fn unconstrained_len(&self) -> usize {
        self.len()
    }

    fn forward(&self, y: &[f64]) -> Result<DoubleTensor> {
        check_len("the unconstrained vector", y.len(), self.len())?;
        let mut x = Vec::with_capacity(y.len());
        for v in rows(y, self.k) {
            let r = Self::radius(v)?;
            x.extend(v.iter().map(|v| v / r));
        }
        tensor(&self.shape, x)
    }

    fn inverse(&self, x: &DoubleTensor) -> Result<Vec<f64>> {
        check_shape("the value", x, &self.shape)?;
        for v in rows(x.as_slice(), self.k) {
            let squared: f64 = v.iter().map(|v| v * v).sum();
            if squared.is_nan() || (squared - 1.0).abs() > TOLERANCE {
                return Err(transform_error(format!(
                    "a unit vector must have length 1, not {}",
                    squared.sqrt()
                )));
            }
        }
        Ok(x.as_slice().to_vec())
    }

    fn log_det_jacobian(&self, y: &[f64]) -> Result<f64> {
        check_len("the unconstrained vector", y.len(), self.len())?;
        Ok(-0.5 * y.iter().map(|v| v * v).sum::<f64>())
    }

    fn log_det_jacobian_gradient(&self, y: &[f64]) -> Result<Vec<f64>> {
        check_len("the unconstrained vector", y.len(), self.len())?;
        Ok(y.iter().map(|v| -v).collect())
    }

    fn forward_jvp(&self, y: &[f64], tangent: &[f64]) -> Result<DoubleTensor> {
        check_second_order(y, tangent, self.len())?;
        let mut dx = Vec::with_capacity(y.len());
        for (v, t) in rows(y, self.k).zip(rows(tangent, self.k)) {
            // dx = (t − x (x·t)) / r
            let r = Self::radius(v)?;
            let along: f64 = v.iter().zip(t).map(|(v, t)| v * t).sum::<f64>() / r;
            dx.extend(v.iter().zip(t).map(|(v, t)| (t - v / r * along) / r));
        }
        tensor(&self.shape, dx)
    }

    fn forward_vjp(&self, y: &[f64], cotangent: &DoubleTensor) -> Result<Vec<f64>> {
        check_len("the unconstrained vector", y.len(), self.len())?;
        check_shape("the cotangent", cotangent, &self.shape)?;
        let mut g = Vec::with_capacity(y.len());
        for (v, c) in rows(y, self.k).zip(rows(cotangent.as_slice(), self.k)) {
            // g = (c − x (x·c)) / r
            let r = Self::radius(v)?;
            let along: f64 = v.iter().zip(c).map(|(v, c)| v * c).sum::<f64>() / r;
            g.extend(v.iter().zip(c).map(|(v, c)| (c - v / r * along) / r));
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
        let mut out = Vec::with_capacity(y.len());
        for ((v, t), c) in rows(y, self.k)
            .zip(rows(tangent, self.k))
            .zip(rows(cotangent.as_slice(), self.k))
        {
            // with x = y/r, dr = x·t and dx = (t − x dr)/r, the derivative of g = (c − x (x·c))/r is
            // (−dx (x·c) − x (dx·c))/r − g dr/r
            let r = Self::radius(v)?;
            let x: Vec<f64> = v.iter().map(|v| v / r).collect();
            let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
            let dr = dot(&x, t);
            let dx: Vec<f64> = t.iter().zip(&x).map(|(t, x)| (t - x * dr) / r).collect();
            let (xc, dxc) = (dot(&x, c), dot(&dx, c));
            out.extend((0..self.k).map(|i| {
                let g = (c[i] - x[i] * xc) / r;
                (-dx[i] * xc - x[i] * dxc) / r - g * dr / r
            }));
        }
        Ok(out)
    }

    fn log_det_jacobian_hvp(&self, y: &[f64], tangent: &[f64]) -> Result<Vec<f64>> {
        // −½ ‖y‖² has Hessian −I
        check_second_order(y, tangent, self.len())?;
        Ok(tangent.iter().map(|t| -t).collect())
    }
}
