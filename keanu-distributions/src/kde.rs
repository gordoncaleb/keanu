//! Gaussian kernel density estimate.

use crate::error::{Result, invalid_parameter};
use crate::rng::KeanuRng;
use keanu_tensor::DoubleTensor;

/// A Gaussian kernel density estimate from samples.
#[derive(Debug, Clone, PartialEq)]
pub struct Kde {
    /// The samples the estimate is built from.
    pub samples: Vec<f64>,
    /// The kernel bandwidth.
    pub bandwidth: f64,
}

/// Derivative of [`Kde::log_prob`] with respect to x.
#[derive(Debug, Clone, PartialEq)]
pub struct KdeGradient {
    /// With respect to x.
    pub x: DoubleTensor,
}

fn kernel(d: f64) -> f64 {
    (-0.5 * d * d).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

impl Kde {
    /// Creates the distribution from its parameters. The bandwidth must be positive and finite.
    pub fn new(samples: Vec<f64>, bandwidth: f64) -> Result<Self> {
        if samples.is_empty() {
            return invalid_parameter("a KDE needs at least one sample");
        }
        if !(bandwidth > 0.0 && bandwidth.is_finite()) {
            return invalid_parameter(format!(
                "the KDE bandwidth must be positive and finite, got {bandwidth}"
            ));
        }
        Ok(Self { samples, bandwidth })
    }

    /// Uses Scott's rule: 1.06 σ n^(-1/5). An error if the samples have no spread (e.g. are all the same), as
    /// the bandwidth would then be zero.
    pub fn with_scotts_bandwidth(samples: Vec<f64>) -> Result<Self> {
        if samples.is_empty() {
            return invalid_parameter("a KDE needs at least one sample");
        }
        let n = samples.len();
        let sd = DoubleTensor::vector(samples.clone()).std_dev();
        let bandwidth = 1.06 * sd * (n as f64).powf(-1.0 / 5.0);
        if !(bandwidth > 0.0 && bandwidth.is_finite()) {
            return invalid_parameter(format!(
                "Scott's rule gives bandwidth {bandwidth}: the samples need a positive, finite standard deviation"
            ));
        }
        Self::new(samples, bandwidth)
    }

    /// The probability density at x.
    pub fn pdf(&self, x: &DoubleTensor) -> DoubleTensor {
        let (n, h) = (self.samples.len() as f64, self.bandwidth);
        x.map(|x| {
            self.samples
                .iter()
                .map(|&s| kernel((x - s) / h))
                .sum::<f64>()
                / (n * h)
        })
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support).
    ///
    /// Computed with log-sum-exp over the kernels, so it stays finite far from the samples.
    pub fn log_prob(&self, x: &DoubleTensor) -> DoubleTensor {
        let (n, h) = (self.samples.len() as f64, self.bandwidth);
        let normaliser = (n * h).ln() + 0.5 * (2.0 * std::f64::consts::PI).ln();
        x.map(|x| {
            let max = self.max_exponent(x);
            if !max.is_finite() {
                // x is infinite or NaN
                return max - normaliser;
            }
            let sum: f64 = self
                .samples
                .iter()
                .map(|&s| (-0.5 * ((x - s) / h).powi(2) - max).exp())
                .sum();
            max + sum.ln() - normaliser
        })
    }

    /// The largest kernel exponent -((x - s) / h)^2 / 2 over the samples (that of the nearest sample).
    fn max_exponent(&self, x: f64) -> f64 {
        let h = self.bandwidth;
        self.samples
            .iter()
            .map(|&s| -0.5 * ((x - s) / h).powi(2))
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    ///
    /// d/dx ln p = -sum_i w_i (x - s_i) / h^2 with w_i the kernels' softmax weights, so it is stable far from
    /// the samples.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<KdeGradient> {
        let h = self.bandwidth;
        Ok(KdeGradient {
            x: x.map(|x| {
                let max = self.max_exponent(x);
                let (weighted, total) =
                    self.samples
                        .iter()
                        .fold((0.0, 0.0), |(weighted, total), &s| {
                            let d = (x - s) / h;
                            let w = (-0.5 * d * d - max).exp();
                            (weighted + w * d, total + w)
                        });
                -weighted / (total * h)
            }),
        })
    }

    /// Picks a sample uniformly and adds Gaussian noise with the bandwidth as standard deviation.
    pub fn sample(&self, count: usize, rng: &mut KeanuRng) -> DoubleTensor {
        let n = self.samples.len();
        let means: Vec<f64> = (0..count)
            .map(|_| self.samples[((rng.next_f64() * n as f64).floor() as usize).min(n - 1)])
            .collect();
        DoubleTensor::vector(
            means
                .iter()
                .map(|&m| m + rng.next_gaussian() * self.bandwidth)
                .collect(),
        )
    }
}
