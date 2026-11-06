//! The bijector for values that sum to zero on each of several groups of their elements.

use super::simplex::Simplex;
use super::*;
use keanu_distributions::multivariate::sums_to_zero;

/// Values whose elements sum to zero within each of several disjoint groups (indices into the flattened value),
/// the other elements being free, for [`Support::SumToZeroComponents`]: an ICAR's support, which sums to zero on
/// each connected component of two or more areas.
///
/// Each group of m elements is mapped as [`SumToZero`] maps a vector: its m − 1 unconstrained coordinates are the
/// group's coordinates in the orthonormal basis of the zero-sum subspace, scattered to the group's elements. Free
/// elements are their own coordinates. The unconstrained vector holds the free elements in index order, then each
/// group's coordinates in turn, so it has the value's length minus the number of groups. The map is linear and
/// isometric, so its log-Jacobian is 0 (a density with respect to the subspace's own volume) and its second
/// derivatives are zero.
#[derive(Debug, Clone, PartialEq)]
pub struct SumToZeroComponents {
    shape: Vec<usize>,
    groups: Vec<Vec<usize>>,
    free: Vec<usize>,
}

impl SumToZeroComponents {
    /// Values of `shape` whose elements sum to zero within each group (of flat indices).
    ///
    /// # Errors
    ///
    /// [`GraphError::Transform`] if an index is out of range, an element is in two groups, or a group is empty.
    pub fn new(groups: &[Vec<usize>], shape: &[usize]) -> Result<Self> {
        let len: usize = shape.iter().product();
        let mut grouped = vec![false; len];
        for group in groups {
            if group.is_empty() {
                return Err(transform_error(
                    "a sum-to-zero group needs at least one element",
                ));
            }
            for &i in group {
                if i >= len {
                    return Err(transform_error(format!(
                        "the sum-to-zero group element {i} is out of range for a value of shape {shape:?}"
                    )));
                }
                if grouped[i] {
                    return Err(transform_error(format!(
                        "the element {i} is in more than one sum-to-zero group"
                    )));
                }
                grouped[i] = true;
            }
        }
        Ok(SumToZeroComponents {
            shape: shape.to_vec(),
            groups: groups.to_vec(),
            free: (0..len).filter(|&i| !grouped[i]).collect(),
        })
    }

    fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// The constrained value of unconstrained coordinates (or a tangent of them).
    fn apply(&self, y: &[f64]) -> Result<DoubleTensor> {
        let mut x = vec![0.0; self.len()];
        for (&i, &v) in self.free.iter().zip(y) {
            x[i] = v;
        }
        let mut offset = self.free.len();
        let mut buffer = Vec::new();
        for group in &self.groups {
            let m = group.len();
            buffer.clear();
            buffer.resize(m, 0.0);
            if m > 1 {
                Simplex::basis_times(&y[offset..offset + m - 1], &mut buffer);
            }
            for (&i, &v) in group.iter().zip(&buffer) {
                x[i] = v;
            }
            offset += m - 1;
        }
        tensor(&self.shape, x)
    }

    /// The transpose of [`Self::apply`].
    fn apply_transpose(&self, x: &[f64]) -> Vec<f64> {
        let mut y: Vec<f64> = self.free.iter().map(|&i| x[i]).collect();
        let mut gathered = Vec::new();
        for group in &self.groups {
            let m = group.len();
            gathered.clear();
            gathered.extend(group.iter().map(|&i| x[i]));
            let start = y.len();
            y.resize(start + m - 1, 0.0);
            if m > 1 {
                Simplex::basis_transpose_times(&gathered, &mut y[start..]);
            }
        }
        y
    }

    fn check(&self, what: &str, y: &[f64]) -> Result<()> {
        check_len(what, y.len(), self.unconstrained_len())
    }
}

impl Bijector for SumToZeroComponents {
    fn constrained_shape(&self) -> &[usize] {
        &self.shape
    }

    fn unconstrained_len(&self) -> usize {
        self.len() - self.groups.len()
    }

    fn forward(&self, y: &[f64]) -> Result<DoubleTensor> {
        self.check("the unconstrained vector", y)?;
        self.apply(y)
    }

    fn inverse(&self, x: &DoubleTensor) -> Result<Vec<f64>> {
        check_shape("the value", x, &self.shape)?;
        let xs = x.as_slice();
        if let Some(v) = xs.iter().find(|v| !v.is_finite()) {
            return Err(transform_error(format!(
                "a sum-to-zero value must be finite, not {v}"
            )));
        }
        for group in &self.groups {
            let values: Vec<f64> = group.iter().map(|&i| xs[i]).collect();
            if !sums_to_zero(&values) {
                return Err(transform_error(format!(
                    "the elements {group:?} of the value, {values:?}, do not sum to 0 (within a relative {})",
                    keanu_distributions::multivariate::ZERO_SUM_TOLERANCE
                )));
            }
        }
        Ok(self.apply_transpose(xs))
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
