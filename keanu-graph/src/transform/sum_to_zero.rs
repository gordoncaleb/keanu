//! The bijector for zero-sum vectors.

use super::simplex::Simplex;
use super::*;
use keanu_distributions::multivariate::sums_to_zero;

/// Vectors over the last axis whose elements sum to zero, for [`Support::SumToZero`], as Stan's
/// `sum_to_zero_vector` (2.36 and later): the N − 1 unconstrained coordinates `y` map to `x = V y`, where the columns
/// of `V` are the orthonormal basis of the zero-sum subspace that [`Simplex`] uses (column `i`, for `i = 1..N−1`,
/// has `1/√(i(i+1))` in its first `i` entries and `−i/√(i(i+1))` in entry `i + 1`). Leading axes are a batch.
///
/// The map is linear and isometric (`Vᵀ V = I`), so `y = Vᵀ x` and `‖x‖ = ‖y‖`. Its log-Jacobian is taken to be 0:
/// a density over the zero-sum subspace (such as [`ZeroSumNormal`](crate::op::DistributionKind::ZeroSumNormal)'s) is
/// a density with respect to the subspace's own volume, which the isometry preserves. Its second derivatives are
/// zero too.
#[derive(Debug, Clone, PartialEq)]
pub struct SumToZero {
    shape: Vec<usize>,
    n: usize,
}

impl SumToZero {
    /// Zero-sum vectors over the last axis of `shape`.
    ///
    /// # Errors
    ///
    /// [`GraphError::Transform`] if `shape` is a scalar shape or its last axis is empty.
    pub fn new(shape: &[usize]) -> Result<Self> {
        let (_, n) = last_axis("sum-to-zero", shape)?;
        if n == 0 {
            return Err(transform_error(
                "a zero-sum vector needs at least one element",
            ));
        }
        Ok(SumToZero {
            shape: shape.to_vec(),
            n,
        })
    }

    fn batch(&self) -> usize {
        self.shape[..self.shape.len() - 1].iter().product()
    }

    /// `V y` for each row of `y`, as a constrained-shaped tensor.
    fn apply(&self, y: &[f64]) -> Result<DoubleTensor> {
        let n = self.n;
        let mut out = vec![0.0; self.batch() * n];
        if n > 1 {
            for (y, x) in y.chunks_exact(n - 1).zip(out.chunks_exact_mut(n)) {
                Simplex::basis_times(y, x);
            }
        }
        tensor(&self.shape, out)
    }

    /// `Vᵀ x` for each row of `x`.
    fn apply_transpose(&self, x: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut y = vec![0.0; self.unconstrained_len()];
        if n > 1 {
            for (x, y) in x.chunks_exact(n).zip(y.chunks_exact_mut(n - 1)) {
                Simplex::basis_transpose_times(x, y);
            }
        }
        y
    }

    fn check(&self, what: &str, y: &[f64]) -> Result<()> {
        check_len(what, y.len(), self.unconstrained_len())
    }
}

impl Bijector for SumToZero {
    fn constrained_shape(&self) -> &[usize] {
        &self.shape
    }

    fn unconstrained_len(&self) -> usize {
        self.batch() * (self.n - 1)
    }

    fn forward(&self, y: &[f64]) -> Result<DoubleTensor> {
        self.check("the unconstrained vector", y)?;
        self.apply(y)
    }

    fn inverse(&self, x: &DoubleTensor) -> Result<Vec<f64>> {
        check_shape("the value", x, &self.shape)?;
        for row in x.as_slice().chunks_exact(self.n) {
            if !row.iter().all(|v| v.is_finite()) || !sums_to_zero(row) {
                return Err(transform_error(format!(
                    "{row:?} is not a vector of finite elements summing to 0 (within a relative {})",
                    keanu_distributions::multivariate::ZERO_SUM_TOLERANCE
                )));
            }
        }
        Ok(self.apply_transpose(x.as_slice()))
    }

    fn log_det_jacobian(&self, y: &[f64]) -> Result<f64> {
        self.check("the unconstrained vector", y)?;
        Ok(0.0)
    }

    fn log_det_jacobian_gradient(&self, y: &[f64]) -> Result<Vec<f64>> {
        self.check("the unconstrained vector", y)?;
        Ok(vec![0.0; y.len()])
    }

    fn forward_jvp(&self, y: &[f64], tangent: &[f64]) -> Result<DoubleTensor> {
        self.check("the unconstrained vector", y)?;
        self.check("the tangent", tangent)?;
        self.apply(tangent)
    }

    fn forward_vjp(&self, y: &[f64], cotangent: &DoubleTensor) -> Result<Vec<f64>> {
        self.check("the unconstrained vector", y)?;
        check_shape("the cotangent", cotangent, &self.shape)?;
        Ok(self.apply_transpose(cotangent.as_slice()))
    }

    fn forward_vjp_jvp(
        &self,
        y: &[f64],
        tangent: &[f64],
        cotangent: &DoubleTensor,
    ) -> Result<Vec<f64>> {
        check_second_order(y, tangent, self.unconstrained_len())?;
        check_shape("the cotangent", cotangent, &self.shape)?;
        // the map is linear
        Ok(vec![0.0; y.len()])
    }

    fn log_det_jacobian_hvp(&self, y: &[f64], tangent: &[f64]) -> Result<Vec<f64>> {
        check_second_order(y, tangent, self.unconstrained_len())?;
        Ok(vec![0.0; y.len()])
    }
}
