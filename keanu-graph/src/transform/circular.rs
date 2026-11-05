//! The bijector for angles.

use super::*;

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
