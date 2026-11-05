//! The bijector for strictly increasing vectors.

use super::*;

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
