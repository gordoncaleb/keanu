//! Gaussian kernel density estimate.

use crate::error::{Result, invalid_parameter};
use crate::rng::KeanuRng;
use keanu_tensor::DoubleTensor;

/// A Gaussian kernel density estimate from samples.
#[derive(Debug, Clone, PartialEq)]
pub struct Kde {
    pub samples: Vec<f64>,
    pub bandwidth: f64,
}

/// Derivative of [`Kde::log_prob`] with respect to x.
#[derive(Debug, Clone, PartialEq)]
pub struct KdeGradient {
    pub x: DoubleTensor,
}

fn kernel(d: f64) -> f64 {
    (-0.5 * d * d).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

impl Kde {
    pub fn new(samples: Vec<f64>, bandwidth: f64) -> Result<Self> {
        if samples.is_empty() {
            return invalid_parameter("a KDE needs at least one sample");
        }
        Ok(Self { samples, bandwidth })
    }

    /// Uses Scott's rule: 1.06 σ n^(-1/5).
    pub fn with_scotts_bandwidth(samples: Vec<f64>) -> Result<Self> {
        let n = samples.len();
        let sd = DoubleTensor::vector(samples.clone()).std_dev();
        Self::new(samples, 1.06 * sd * (n as f64).powf(-1.0 / 5.0))
    }

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

    pub fn log_prob(&self, x: &DoubleTensor) -> DoubleTensor {
        self.pdf(x).ln()
    }

    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<KdeGradient> {
        let (n, h) = (self.samples.len() as f64, self.bandwidth);
        let d_pdf = x.map(|x| {
            -self
                .samples
                .iter()
                .map(|&s| {
                    let d = (x - s) / h;
                    kernel(d) * d
                })
                .sum::<f64>()
                / (h * h * n)
        });
        Ok(KdeGradient {
            x: d_pdf.div(&self.pdf(x))?,
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
