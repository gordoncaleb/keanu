//! The identity bijector, for real values.

use super::*;

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

    fn forward_into(&self, y: &[f64], out: &mut [f64]) -> Result<()> {
        check_len("the unconstrained vector", y.len(), self.len())?;
        check_len("the output", out.len(), self.len())?;
        out.copy_from_slice(y);
        Ok(())
    }

    fn forward_vjp_into(&self, y: &[f64], cotangent: &[f64], out: &mut [f64]) -> Result<()> {
        check_len("the unconstrained vector", y.len(), self.len())?;
        check_len("the cotangent", cotangent.len(), self.len())?;
        check_len("the output", out.len(), self.len())?;
        out.copy_from_slice(cotangent);
        Ok(())
    }

    fn add_log_det_jacobian(&self, y: &[f64], gradient: &mut [f64]) -> Result<f64> {
        check_len("the unconstrained vector", y.len(), self.len())?;
        check_len("the gradient", gradient.len(), self.len())?;
        // adding the zero gradient, as the general path does
        for g in gradient.iter_mut() {
            *g += 0.0;
        }
        Ok(0.0)
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
