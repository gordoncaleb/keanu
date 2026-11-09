//! Low-rank feature representations of Gaussian process priors.

use super::{DEFAULT_JITTER, Kernel, Kind, as_matrix};
use crate::{DoubleVar, GraphError, IntoDoubleVar, Result, Var};
use keanu_tensor::DoubleTensor;
use std::f64::consts::TAU;

/// A zero-mean inducing-point GP prior. It uses the conditional mean given the inducing values,
/// with covariance K(x, z) (K(z, z) + jitter I)⁻¹ K(z, x). The conditional residual is omitted.
/// Use the same instance for training and prediction to share the inducing draws.
#[derive(Debug, Clone)]
pub struct InducingGp<'b> {
    kernel: Kernel<'b>,
    inducing: DoubleVar<'b>,
    factor: DoubleVar<'b>,
    weights: DoubleVar<'b>,
    coefficients: DoubleVar<'b>,
}

impl<'b> InducingGp<'b> {
    /// Constructs the prior from finite, nonempty inducing locations shaped [m] or [m, d].
    pub fn new(kernel: &Kernel<'b>, inducing: DoubleTensor) -> Result<Self> {
        let m = match inducing.shape() {
            [m] if *m > 0 => *m,
            [m, d] if *m > 0 && *d > 0 => *m,
            _ => {
                return Err(GraphError::InvalidValue(
                    "inducing locations need shape [m] or [m, d], with m and d positive".into(),
                ));
            }
        };
        if inducing.as_slice().iter().any(|x| !x.is_finite()) {
            return Err(GraphError::InvalidValue(
                "inducing locations must be finite".into(),
            ));
        }
        let b = kernel.builder();
        let inducing = b.constant(inducing);
        let jitter = b.constant(DoubleTensor::eye(m)? * DEFAULT_JITTER);
        let factor = (kernel.covariance(inducing, inducing) + jitter).cholesky();
        let coefficients = b.gaussian_with_shape(&[m], 0.0, 1.0);
        let weights = factor.solve_triangular(coefficients, true, true);
        Ok(Self {
            kernel: kernel.clone(),
            inducing,
            factor,
            weights,
            coefficients,
        })
    }

    /// The whitened standard-normal inducing coefficients.
    pub fn coefficients(&self) -> DoubleVar<'b> {
        self.coefficients
    }

    /// The whitened features `K(x, z) L⁻ᵀ`, with one row per input.
    pub fn basis(&self, x: impl IntoDoubleVar<'b>) -> DoubleVar<'b> {
        let cross = self.kernel.covariance(self.inducing, x);
        self.factor.solve_triangular(cross, true, false).transpose()
    }

    /// The latent function values at training or prediction inputs.
    pub fn predict(&self, x: impl IntoDoubleVar<'b>) -> DoubleVar<'b> {
        self.kernel
            .covariance(x, self.inducing)
            .matmul(self.weights.reshape(&[-1, 1]))
            .reshape(&[-1])
    }
}

/// A one-dimensional periodic GP prior using the first Fourier harmonics of the periodic kernel.
/// It has one constant feature and two features per harmonic. The coefficients are shared
/// across all training and prediction inputs.
///
/// The weights follow [Solin and Särkkä (2014)](https://proceedings.mlr.press/v33/solin14.html).
#[derive(Debug, Clone)]
pub struct PeriodicHsgp<'b> {
    period: DoubleVar<'b>,
    scales: Vec<DoubleVar<'b>>,
    coefficients: DoubleVar<'b>,
}

impl<'b> PeriodicHsgp<'b> {
    /// Constructs a Fourier prior from a periodic kernel with scalar parameters and at least one harmonic.
    pub fn new(kernel: &Kernel<'b>, harmonics: usize) -> Result<Self> {
        let Kind::Periodic {
            amplitude,
            length_scale,
            period,
        } = kernel.kind
        else {
            return Err(GraphError::InvalidValue(
                "periodic HSGP needs a periodic kernel".into(),
            ));
        };
        let count = harmonics
            .checked_mul(2)
            .and_then(|m| m.checked_add(1))
            .filter(|&m| m <= super::hsgp::MAX_BASIS_FUNCTIONS && m > 1)
            .ok_or_else(|| {
                GraphError::InvalidValue("invalid number of periodic HSGP harmonics".into())
            })?;
        if [amplitude, length_scale, period]
            .iter()
            .any(|p| !p.shape().is_empty())
        {
            return Err(GraphError::InvalidValue(
                "periodic HSGP needs scalar kernel parameters".into(),
            ));
        }
        let b = kernel.builder();
        let inverse_concentration = length_scale * length_scale;
        let log_leading = length_scale.abs().ln() - 0.5 * TAU.ln();
        let scales = (0..=harmonics)
            .map(|j| {
                let multiplicity = if j == 0 { 1.0 } else { 2.0_f64.sqrt() };
                let order = j as f64;
                let cutoff = 1e-6 / (1.0 + order * order);
                let concentration = 1.0 / inverse_concentration.maximum(cutoff);
                let ordinary = concentration.log_bessel_i(order) - concentration;
                // DLMF 10.40.1. At large concentration, subtracting it from log I loses the log weight.
                let correction =
                    inverse_concentration.minimum(cutoff) * ((1.0 - 4.0 * order * order) / 8.0);
                let asymptotic = log_leading + correction.ln_1p();
                let log_weight = b.where_(inverse_concentration.lt(cutoff), asymptotic, ordinary);
                amplitude * (log_weight * 0.5).exp() * multiplicity
            })
            .collect();
        let coefficients = b.gaussian_with_shape(&[count], 0.0, 1.0);
        Ok(Self {
            period,
            scales,
            coefficients,
        })
    }

    /// The standard-normal Fourier coefficients, ordered constant, cosine 1, sine 1, and so on.
    pub fn coefficients(&self) -> DoubleVar<'b> {
        self.coefficients
    }

    /// The scaled Fourier features, with one row per input and one column per coefficient.
    pub fn basis(&self, x: impl IntoDoubleVar<'b>) -> DoubleVar<'b> {
        let b = self.period.builder();
        let x = x.into_var(b);
        let Some((x, n, dimensions)) = as_matrix(b, x, "periodic inputs") else {
            return b.scalar(0.0);
        };
        if dimensions != 1 {
            b.record(GraphError::InvalidValue(
                "periodic HSGP needs one input dimension".into(),
            ));
            return b.scalar(0.0);
        }
        let angle = x.reshape(&[n as isize]) * TAU / self.period;
        let mut features = Vec::with_capacity(self.coefficients.shape()[0]);
        features.push(self.scales[0].broadcast_to(&[n]));
        for (j, scale) in self.scales.iter().copied().enumerate().skip(1) {
            let harmonic = angle * j as f64;
            features.extend([harmonic.cos() * scale, harmonic.sin() * scale]);
        }
        b.stack(-1, &features)
    }

    /// The latent function values at training or prediction inputs.
    pub fn predict(&self, x: impl IntoDoubleVar<'b>) -> DoubleVar<'b> {
        self.basis(x)
            .matmul(self.coefficients.reshape(&[-1, 1]))
            .reshape(&[-1])
    }
}
