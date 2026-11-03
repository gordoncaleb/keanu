//! Univariate continuous distributions.
//!
//! `log_prob` is element-wise and broadcasts x with the parameters. Gradients are element-wise too, shaped like
//! the broadcast of x and the parameters; they are unspecified where `log_prob` is -infinity.

use crate::error::{Result, invalid_parameter};
use crate::rng::KeanuRng;
use crate::util::{digamma, ln_gamma, map2, map3, map4, safe_ln_times, sample_each, scalar};
use keanu_tensor::{DoubleTensor, IntTensor};
use std::f64::consts::{LN_2, PI};

const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_8;

fn require_positive(name: &str, t: &DoubleTensor) -> Result<()> {
    if t.as_slice().iter().all(|&v| v > 0.0) {
        Ok(())
    } else {
        invalid_parameter(format!("{name} must be positive"))
    }
}

/// Normal distribution with mean `mu` and standard deviation `sigma`.
#[derive(Debug, Clone, PartialEq)]
pub struct Gaussian {
    /// The location (mean).
    pub mu: DoubleTensor,
    /// The scale (standard deviation).
    pub sigma: DoubleTensor,
}

/// Derivatives of [`Gaussian::log_prob`].
#[derive(Debug, Clone, PartialEq)]
pub struct GaussianGradient {
    /// With respect to x.
    pub x: DoubleTensor,
    /// With respect to `mu`.
    pub mu: DoubleTensor,
    /// With respect to `sigma`.
    pub sigma: DoubleTensor,
}

impl Gaussian {
    /// Creates the distribution from its parameters.
    pub fn new(mu: DoubleTensor, sigma: DoubleTensor) -> Self {
        Self { mu, sigma }
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support).
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map3(x, &self.mu, &self.sigma, |x, mu, sigma| {
            -((x - mu).powi(2) / (2.0 * sigma * sigma) + sigma.ln() + LN_SQRT_2PI)
        })
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<GaussianGradient> {
        let d_mu = map3(x, &self.mu, &self.sigma, |x, mu, sigma| {
            (x - mu) / (sigma * sigma)
        })?;
        let d_sigma = map3(x, &self.mu, &self.sigma, |x, mu, sigma| {
            (x - mu).powi(2) / (sigma * sigma * sigma) - 1.0 / sigma
        })?;
        Ok(GaussianGradient {
            x: d_mu.neg(),
            mu: d_mu,
            sigma: d_sigma,
        })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes).
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        Ok(rng.gaussian(shape).mul(&self.sigma)?.add(&self.mu)?)
    }
}

/// Gaussian folded at 0: the density of |X| for X ~ N(0, sigma).
#[derive(Debug, Clone, PartialEq)]
pub struct HalfGaussian {
    /// The scale (standard deviation).
    pub sigma: DoubleTensor,
}

/// Derivatives of [`HalfGaussian::log_prob`].
#[derive(Debug, Clone, PartialEq)]
pub struct HalfGaussianGradient {
    /// With respect to x.
    pub x: DoubleTensor,
    /// With respect to `sigma`.
    pub sigma: DoubleTensor,
}

impl HalfGaussian {
    /// Creates the distribution from its parameters.
    pub fn new(sigma: DoubleTensor) -> Self {
        Self { sigma }
    }

    fn gaussian(&self) -> Gaussian {
        Gaussian::new(scalar(0.0), self.sigma.clone())
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support).
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        let gaussian = self.gaussian().log_prob(x)?;
        Ok(gaussian.zip_with(x, |lp, x| {
            if x < 0.0 {
                f64::NEG_INFINITY
            } else {
                lp + LN_2
            }
        })?)
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<HalfGaussianGradient> {
        let g = self.gaussian().d_log_prob(x)?;
        let inside = |t: DoubleTensor| t.zip_with(x, |d, x| if x < 0.0 { 0.0 } else { d });
        Ok(HalfGaussianGradient {
            x: inside(g.x)?,
            sigma: inside(g.sigma)?,
        })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes).
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        Ok(self.gaussian().sample(shape, rng)?.abs())
    }
}

/// Cauchy distribution with the given location and scale.
#[derive(Debug, Clone, PartialEq)]
pub struct Cauchy {
    /// The location.
    pub location: DoubleTensor,
    /// The scale.
    pub scale: DoubleTensor,
}

/// Derivatives of [`Cauchy::log_prob`].
#[derive(Debug, Clone, PartialEq)]
pub struct CauchyGradient {
    /// With respect to x.
    pub x: DoubleTensor,
    /// With respect to `location`.
    pub location: DoubleTensor,
    /// With respect to `scale`.
    pub scale: DoubleTensor,
}

impl Cauchy {
    /// Creates the distribution from its parameters.
    pub fn new(location: DoubleTensor, scale: DoubleTensor) -> Self {
        Self { location, scale }
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support).
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map3(x, &self.location, &self.scale, |x, l, s| {
            (-s.ln() - PI.ln()) - (((x - l) / s).powi(2) + 1.0).ln()
        })
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<CauchyGradient> {
        let (l, s) = (&self.location, &self.scale);
        Ok(CauchyGradient {
            location: map3(x, l, s, |x, l, s| 2.0 * (x - l) / (s * s + (x - l).powi(2)))?,
            scale: map3(x, l, s, |x, l, s| {
                ((x - l).powi(2) - s * s) / (s * ((x - l).powi(2) + s * s))
            })?,
            x: map3(x, l, s, |x, l, s| {
                -2.0 * (x - l) / (l * l - 2.0 * l * x + s * s + x * x)
            })?,
        })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes).
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        require_positive("scale", &self.scale)?;
        let unit = rng.uniform(shape).map(|u| ((u - 0.5) * PI).tan());
        Ok(unit.mul(&self.scale)?.add(&self.location)?)
    }
}

/// Cauchy folded at 0: the density of |X| for X ~ Cauchy(0, scale).
#[derive(Debug, Clone, PartialEq)]
pub struct HalfCauchy {
    /// The scale.
    pub scale: DoubleTensor,
}

/// Derivatives of [`HalfCauchy::log_prob`].
#[derive(Debug, Clone, PartialEq)]
pub struct HalfCauchyGradient {
    /// With respect to x.
    pub x: DoubleTensor,
    /// With respect to `scale`.
    pub scale: DoubleTensor,
}

impl HalfCauchy {
    /// Creates the distribution from its parameters.
    pub fn new(scale: DoubleTensor) -> Self {
        Self { scale }
    }

    fn cauchy(&self) -> Cauchy {
        Cauchy::new(scalar(0.0), self.scale.clone())
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support).
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        let cauchy = self.cauchy().log_prob(x)?;
        Ok(cauchy.zip_with(x, |lp, x| {
            if x < 0.0 {
                f64::NEG_INFINITY
            } else {
                lp + LN_2
            }
        })?)
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<HalfCauchyGradient> {
        let g = self.cauchy().d_log_prob(x)?;
        let inside = |t: DoubleTensor| t.zip_with(x, |d, x| if x < 0.0 { 0.0 } else { d });
        Ok(HalfCauchyGradient {
            x: inside(g.x)?,
            scale: inside(g.scale)?,
        })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes).
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        Ok(self.cauchy().sample(shape, rng)?.abs())
    }
}

/// Beta distribution with shape parameters `alpha` and `beta`, on the interval [x_min, x_max].
#[derive(Debug, Clone, PartialEq)]
pub struct Beta {
    /// The alpha (first shape) parameter.
    pub alpha: DoubleTensor,
    /// The beta parameter.
    pub beta: DoubleTensor,
    /// The lower bound.
    pub x_min: DoubleTensor,
    /// The upper bound.
    pub x_max: DoubleTensor,
}

/// Derivatives of [`Beta::log_prob`] (with respect to x and the shape parameters).
#[derive(Debug, Clone, PartialEq)]
pub struct BetaGradient {
    /// With respect to x.
    pub x: DoubleTensor,
    /// With respect to `alpha`.
    pub alpha: DoubleTensor,
    /// With respect to `beta`.
    pub beta: DoubleTensor,
}

impl Beta {
    /// Beta on [x_min, x_max].
    pub fn new(
        alpha: DoubleTensor,
        beta: DoubleTensor,
        x_min: DoubleTensor,
        x_max: DoubleTensor,
    ) -> Self {
        Self {
            alpha,
            beta,
            x_min,
            x_max,
        }
    }

    /// The standard Beta on [0, 1].
    pub fn standard(alpha: DoubleTensor, beta: DoubleTensor) -> Self {
        Self::new(alpha, beta, scalar(0.0), scalar(1.0))
    }

    /// With y = (x - x_min) / (x_max - x_min):
    /// (alpha - 1) ln y + (beta - 1) ln(1 - y) - ln B(alpha, beta) - ln(x_max - x_min), -infinity outside the bounds.
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        crate::util::map5(
            x,
            &self.alpha,
            &self.beta,
            &self.x_min,
            &self.x_max,
            |x, a, b, min, max| {
                if x < min || x > max {
                    return f64::NEG_INFINITY;
                }
                let range = max - min;
                let y = (x - min) / range;
                let beta_function = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
                safe_ln_times(y, a - 1.0) + safe_ln_times(1.0 - y, b - 1.0)
                    - beta_function
                    - range.ln()
            },
        )
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<BetaGradient> {
        let (a, b, min, max) = (&self.alpha, &self.beta, &self.x_min, &self.x_max);
        let y = map3(x, min, max, |x, min, max| (x - min) / (max - min))?;
        let range = max.sub(min)?;
        Ok(BetaGradient {
            x: map4(&y, a, b, &range, |y, a, b, r| {
                ((a - 1.0) / y - (b - 1.0) / (1.0 - y)) / r
            })?,
            alpha: map3(&y, a, b, |y, a, b| y.ln() + (digamma(a + b) - digamma(a)))?,
            beta: map3(&y, a, b, |y, a, b| {
                (1.0 - y).ln() + (digamma(a + b) - digamma(b))
            })?,
        })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes).
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        require_positive("alpha", &self.alpha)?;
        require_positive("beta", &self.beta)?;
        let y1 = Gamma::new(scalar(1.0), self.alpha.clone()).sample(shape, rng)?;
        let y2 = Gamma::new(scalar(1.0), self.beta.clone()).sample(shape, rng)?;
        let ratio1 = map2(&y1, &y2, |y1, y2| y1 / (y1 + y2))?;
        let ratio2 = map2(&y1, &y2, |y1, y2| y2 / (y1 + y2))?;
        // same arithmetic as Java: from the upper bound when alpha < beta, from the lower bound otherwise
        let from_max = map3(&self.x_max, &self.x_min, &ratio2, |max, min, r| {
            max - r * (max - min)
        })?;
        let from_min = map3(&self.x_min, &self.x_max, &ratio1, |min, max, r| {
            min + r * (max - min)
        })?;
        let choose_max = self.alpha.lt(&self.beta)?;
        map3(&choose_max, &from_max, &from_min, |use_max, a, b| {
            if use_max { a } else { b }
        })
    }
}

/// Chi-squared distribution with `k` degrees of freedom.
#[derive(Debug, Clone, PartialEq)]
pub struct ChiSquared {
    /// The shape parameter k.
    pub k: IntTensor,
}

/// Derivatives of [`ChiSquared::log_prob`].
#[derive(Debug, Clone, PartialEq)]
pub struct ChiSquaredGradient {
    /// With respect to x.
    pub x: DoubleTensor,
}

impl ChiSquared {
    /// Creates the distribution from its parameters.
    pub fn new(k: IntTensor) -> Self {
        Self { k }
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support).
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map2(x, &self.k, |x, k| {
            if x < 0.0 {
                return f64::NEG_INFINITY;
            }
            let half_k = k as f64 / 2.0;
            (safe_ln_times(x, half_k - 1.0) - x / 2.0) - (half_k * LN_2 + ln_gamma(half_k))
        })
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<ChiSquaredGradient> {
        Ok(ChiSquaredGradient {
            x: map2(x, &self.k, |x, k| (k as f64 / 2.0 - 1.0) / x - 0.5)?,
        })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes).
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        Gamma::new(scalar(2.0), self.k.to_double() / 2.0).sample(shape, rng)
    }
}

/// Exponential distribution with the given rate (pdf = rate e^(-rate x)).
#[derive(Debug, Clone, PartialEq)]
pub struct Exponential {
    /// The rate (1 / mean).
    pub rate: DoubleTensor,
}

/// Derivatives of [`Exponential::log_prob`].
#[derive(Debug, Clone, PartialEq)]
pub struct ExponentialGradient {
    /// With respect to x.
    pub x: DoubleTensor,
    /// With respect to `rate`.
    pub rate: DoubleTensor,
}

impl Exponential {
    /// Creates the distribution from its parameters.
    pub fn new(rate: DoubleTensor) -> Self {
        Self { rate }
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support).
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map2(x, &self.rate, |x, rate| {
            if x < 0.0 {
                f64::NEG_INFINITY
            } else {
                rate.ln() - x * rate
            }
        })
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<ExponentialGradient> {
        Ok(ExponentialGradient {
            x: map2(x, &self.rate, |_, rate| -rate)?,
            rate: map2(x, &self.rate, |x, rate| 1.0 / rate - x)?,
        })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes).
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        Ok(rng.uniform(shape).map(|u| -u.ln()).div(&self.rate)?)
    }
}

/// Gamma distribution with scale `theta` and shape `k`.
#[derive(Debug, Clone, PartialEq)]
pub struct Gamma {
    /// The scale parameter θ.
    pub theta: DoubleTensor,
    /// The shape parameter k.
    pub k: DoubleTensor,
}

/// Derivatives of [`Gamma::log_prob`].
#[derive(Debug, Clone, PartialEq)]
pub struct GammaGradient {
    /// With respect to x.
    pub x: DoubleTensor,
    /// With respect to `theta`.
    pub theta: DoubleTensor,
    /// With respect to `k`.
    pub k: DoubleTensor,
}

impl Gamma {
    /// Creates the distribution from its parameters.
    pub fn new(theta: DoubleTensor, k: DoubleTensor) -> Self {
        Self { theta, k }
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support).
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map3(x, &self.theta, &self.k, |x, theta, k| {
            if x < 0.0 {
                return f64::NEG_INFINITY;
            }
            safe_ln_times(x, k - 1.0) - ln_gamma(k) - x / theta - k * theta.ln()
        })
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<GammaGradient> {
        let (t, k) = (&self.theta, &self.k);
        Ok(GammaGradient {
            x: map3(x, t, k, |x, t, k| (k - 1.0) / x - 1.0 / t)?,
            theta: map3(x, t, k, |x, t, k| -((t * k - x) / (t * t)))?,
            k: map3(x, t, k, |x, t, k| x.ln() - t.ln() - digamma(k))?,
        })
    }

    /// Cheng's algorithm for k > 1 and Ahrens–Dieter GS for k < 1.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        sample_each(shape, &[&self.theta, &self.k], |p| {
            sample_gamma(p[0], p[1], rng)
        })
    }
}

pub(crate) fn sample_gamma(theta: f64, k: f64, rng: &mut KeanuRng) -> Result<f64> {
    if theta <= 0.0 || k <= 0.0 || theta.is_nan() || k.is_nan() {
        return invalid_parameter(format!(
            "theta and k must be positive, got theta {theta} and k {k}"
        ));
    }
    if k < 1.0 {
        let c = 1.0 + k / std::f64::consts::E;
        loop {
            let p = c * rng.next_f64();
            if p > 1.0 {
                let y = -((c - p) / k).ln();
                if rng.next_f64() <= y.powf(k - 1.0) {
                    return Ok(theta * y);
                }
            } else {
                let y = p.powf(1.0 / k);
                if rng.next_f64() <= (-y).exp() {
                    return Ok(theta * y);
                }
            }
        }
    }
    if k == 1.0 {
        return Ok(-theta * rng.next_f64().ln());
    }
    let a = 1.0 / (2.0 * k - 1.0).sqrt();
    let b = k - 4f64.ln();
    let q = k + 1.0 / a;
    let t: f64 = 4.5;
    let d = 1.0 + t.ln();
    loop {
        let p1 = rng.next_f64();
        let p2 = rng.next_f64();
        let v = a * (p1 / (1.0 - p1)).ln();
        let y = k * v.exp();
        let z = p1 * p1 * p2;
        let w = b + q * v - y;
        if w + d - t * z >= 0.0 || w >= z.ln() {
            return Ok(theta * y);
        }
    }
}

/// Inverse gamma distribution with shape `alpha` and scale `beta`.
#[derive(Debug, Clone, PartialEq)]
pub struct InverseGamma {
    /// The alpha (first shape) parameter.
    pub alpha: DoubleTensor,
    /// The beta parameter.
    pub beta: DoubleTensor,
}

/// Derivatives of [`InverseGamma::log_prob`].
#[derive(Debug, Clone, PartialEq)]
pub struct InverseGammaGradient {
    /// With respect to x.
    pub x: DoubleTensor,
    /// With respect to `alpha`.
    pub alpha: DoubleTensor,
    /// With respect to `beta`.
    pub beta: DoubleTensor,
}

impl InverseGamma {
    /// Creates the distribution from its parameters.
    pub fn new(alpha: DoubleTensor, beta: DoubleTensor) -> Self {
        Self { alpha, beta }
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support).
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map3(x, &self.alpha, &self.beta, |x, a, b| {
            if x <= 0.0 {
                return f64::NEG_INFINITY;
            }
            a * b.ln() + x.ln() * (-a - 1.0) - ln_gamma(a) - b / x
        })
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<InverseGammaGradient> {
        let (a, b) = (&self.alpha, &self.beta);
        Ok(InverseGammaGradient {
            alpha: map3(x, a, b, |x, a, b| -x.ln() - digamma(a) + b.ln())?,
            beta: map3(x, a, b, |x, a, b| -1.0 / x + a / b)?,
            x: map3(x, a, b, |x, a, b| (x * -(a + 1.0) + b) / (x * x))?,
        })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes).
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        let gamma = Gamma::new(self.beta.reciprocal(), self.alpha.clone()).sample(shape, rng)?;
        Ok(gamma.reciprocal())
    }
}

/// Laplace distribution with location `mu` and scale `beta`.
#[derive(Debug, Clone, PartialEq)]
pub struct Laplace {
    /// The location (mean).
    pub mu: DoubleTensor,
    /// The beta parameter.
    pub beta: DoubleTensor,
}

/// Derivatives of [`Laplace::log_prob`].
#[derive(Debug, Clone, PartialEq)]
pub struct LaplaceGradient {
    /// With respect to x.
    pub x: DoubleTensor,
    /// With respect to `mu`.
    pub mu: DoubleTensor,
    /// With respect to `beta`.
    pub beta: DoubleTensor,
}

impl Laplace {
    /// Creates the distribution from its parameters.
    pub fn new(mu: DoubleTensor, beta: DoubleTensor) -> Self {
        Self { mu, beta }
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support).
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map3(x, &self.mu, &self.beta, |x, mu, beta| {
            -((mu - x).abs() / beta + (2.0 * beta).ln())
        })
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<LaplaceGradient> {
        let (m, b) = (&self.mu, &self.beta);
        Ok(LaplaceGradient {
            // sign(mu - x) / beta, with the subgradient at the mode taken to be 0 (not 0 / 0)
            x: map3(x, m, b, |x, m, b| sign(m - x) / b)?,
            mu: map3(x, m, b, |x, m, b| sign(x - m) / b)?,
            beta: map3(x, m, b, |x, m, b| ((m - x).abs() - b) / (b * b))?,
        })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes).
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        sample_each(shape, &[&self.mu, &self.beta], |p| {
            let (mu, beta) = (p[0], p[1]);
            if beta <= 0.0 {
                return invalid_parameter(format!("beta must be positive, got {beta}"));
            }
            Ok(if rng.next_f64() > 0.5 {
                mu + beta * rng.next_f64().ln()
            } else {
                mu - beta * rng.next_f64().ln()
            })
        })
    }
}

/// Logistic distribution with location `mu` and scale `s`.
#[derive(Debug, Clone, PartialEq)]
pub struct Logistic {
    /// The location (mean).
    pub mu: DoubleTensor,
    /// The scale.
    pub s: DoubleTensor,
}

/// Derivatives of [`Logistic::log_prob`].
#[derive(Debug, Clone, PartialEq)]
pub struct LogisticGradient {
    /// With respect to x.
    pub x: DoubleTensor,
    /// With respect to `mu`.
    pub mu: DoubleTensor,
    /// With respect to `s`.
    pub s: DoubleTensor,
}

impl Logistic {
    /// Creates the distribution from its parameters.
    pub fn new(mu: DoubleTensor, s: DoubleTensor) -> Self {
        Self { mu, s }
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support).
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map3(x, &self.mu, &self.s, |x, mu, s| {
            let m = (mu - x) / s;
            m - s.ln() - (m.exp() + 1.0).ln() * 2.0
        })
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<LogisticGradient> {
        let a_term = |m: f64| 2.0 * m.exp() / (m.exp() + 1.0);
        let d_x = map3(x, &self.mu, &self.s, |x, mu, s| {
            (a_term((mu - x) / s) - 1.0) / s
        })?;
        let d_s = map3(x, &self.mu, &self.s, |x, mu, s| {
            let m = (mu - x) / s;
            (a_term(m) * m - m - 1.0) / s
        })?;
        Ok(LogisticGradient {
            mu: d_x.neg(),
            x: d_x,
            s: d_s,
        })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes).
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        let unit = rng.uniform(shape).map(|u| (1.0 / u - 1.0).ln());
        Ok(unit.mul(&self.s)?.add(&self.mu)?)
    }
}

/// Log-normal distribution: ln X ~ N(mu, sigma).
#[derive(Debug, Clone, PartialEq)]
pub struct LogNormal {
    /// The location (mean).
    pub mu: DoubleTensor,
    /// The scale (standard deviation).
    pub sigma: DoubleTensor,
}

/// Derivatives of [`LogNormal::log_prob`].
#[derive(Debug, Clone, PartialEq)]
pub struct LogNormalGradient {
    /// With respect to x.
    pub x: DoubleTensor,
    /// With respect to `mu`.
    pub mu: DoubleTensor,
    /// With respect to `sigma`.
    pub sigma: DoubleTensor,
}

impl LogNormal {
    /// Creates the distribution from its parameters.
    pub fn new(mu: DoubleTensor, sigma: DoubleTensor) -> Self {
        Self { mu, sigma }
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support).
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map3(x, &self.mu, &self.sigma, |x, mu, sigma| {
            if x <= 0.0 {
                return f64::NEG_INFINITY;
            }
            -((x.ln() - mu).powi(2) / (2.0 * sigma * sigma) + (sigma * x).ln() + LN_SQRT_2PI)
        })
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<LogNormalGradient> {
        let (m, s) = (&self.mu, &self.sigma);
        Ok(LogNormalGradient {
            mu: map3(x, m, s, |x, m, s| (x.ln() - m) / (s * s))?,
            x: map3(x, m, s, |x, m, s| -((x.ln() - m) / (s * s) + 1.0) / x)?,
            sigma: map3(x, m, s, |x, m, s| {
                (x.ln() - m).powi(2) / (s * s * s) - 1.0 / s
            })?,
        })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes).
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        Ok(Gaussian::new(self.mu.clone(), self.sigma.clone())
            .sample(shape, rng)?
            .exp())
    }
}

/// Pareto distribution with minimum value `location` and shape `scale`.
#[derive(Debug, Clone, PartialEq)]
pub struct Pareto {
    /// The location.
    pub location: DoubleTensor,
    /// The scale.
    pub scale: DoubleTensor,
}

/// Derivatives of [`Pareto::log_prob`].
#[derive(Debug, Clone, PartialEq)]
pub struct ParetoGradient {
    /// With respect to x.
    pub x: DoubleTensor,
    /// With respect to `location`.
    pub location: DoubleTensor,
    /// With respect to `scale`.
    pub scale: DoubleTensor,
}

impl Pareto {
    /// Creates the distribution from its parameters.
    pub fn new(location: DoubleTensor, scale: DoubleTensor) -> Self {
        Self { location, scale }
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support).
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        let valid = self.location.as_slice().iter().all(|&l| l > 0.0)
            && self.scale.as_slice().iter().all(|&s| s > 0.0);
        if !valid {
            return Ok(DoubleTensor::full(x.shape(), f64::NEG_INFINITY));
        }
        map3(x, &self.location, &self.scale, |x, l, s| {
            if x <= l {
                f64::NEG_INFINITY
            } else {
                s.ln() + l.ln() * s - (s + 1.0) * x.ln()
            }
        })
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<ParetoGradient> {
        let (l, s) = (&self.location, &self.scale);
        Ok(ParetoGradient {
            x: map2(x, s, |x, s| -((s + 1.0) / x))?,
            location: map3(x, l, s, |_, l, s| s / l)?,
            scale: map3(x, l, s, |x, l, s| 1.0 / s + l.ln() - x.ln())?,
        })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes).
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        let u = rng.uniform(shape);
        map3(&u, &self.location, &self.scale, |u, l, s| {
            l / (1.0 - u).powf(1.0 / s)
        })
    }
}

/// Student's t distribution with `v` degrees of freedom.
#[derive(Debug, Clone, PartialEq)]
pub struct StudentT {
    /// The degrees of freedom.
    pub v: IntTensor,
}

/// Derivatives of [`StudentT::log_prob`].
#[derive(Debug, Clone, PartialEq)]
pub struct StudentTGradient {
    /// With respect to x.
    pub x: DoubleTensor,
}

impl StudentT {
    /// Creates the distribution from its parameters.
    pub fn new(v: IntTensor) -> Self {
        Self { v }
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support).
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        let half_ln_pi = PI.ln() / 2.0;
        map2(x, &self.v, |t, v| {
            let v = v as f64;
            let half_v_plus_one = (v + 1.0) / 2.0;
            ln_gamma(half_v_plus_one)
                - v.ln() / 2.0
                - half_ln_pi
                - ln_gamma(v / 2.0)
                - half_v_plus_one * (t * t / v + 1.0).ln()
        })
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<StudentTGradient> {
        Ok(StudentTGradient {
            x: map2(x, &self.v, |t, v| {
                -t * (v as f64 + 1.0) / (t * t + v as f64)
            })?,
        })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes).
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        let chi2 = ChiSquared::new(self.v.clone()).sample(shape, rng)?;
        let normal = rng.gaussian(shape);
        let denominator = map2(&chi2, &self.v, |c, v| (c / v as f64).sqrt())?;
        Ok(normal.div(&denominator)?)
    }
}

/// Triangular distribution on [x_min, x_max] with mode `c`.
#[derive(Debug, Clone, PartialEq)]
pub struct Triangular {
    /// The lower bound.
    pub x_min: DoubleTensor,
    /// The upper bound.
    pub x_max: DoubleTensor,
    /// The mode.
    pub c: DoubleTensor,
}

/// Derivatives of [`Triangular::log_prob`].
#[derive(Debug, Clone, PartialEq)]
pub struct TriangularGradient {
    /// With respect to x.
    pub x: DoubleTensor,
    /// With respect to `x_min`.
    pub x_min: DoubleTensor,
    /// With respect to `x_max`.
    pub x_max: DoubleTensor,
    /// With respect to `c`.
    pub c: DoubleTensor,
}

impl Triangular {
    /// Creates the distribution from its parameters.
    pub fn new(x_min: DoubleTensor, x_max: DoubleTensor, c: DoubleTensor) -> Self {
        Self { x_min, x_max, c }
    }

    /// 2(x - a)/((b - a)(c - a)) below the mode, 2/(b - a) at it, 2(b - x)/((b - a)(b - c)) above it.
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map4(x, &self.x_min, &self.x_max, &self.c, |x, a, b, c| {
            let range = b - a;
            let pdf = if x >= a && x < c {
                2.0 * (x - a) / (range * (c - a))
            } else if x == c {
                2.0 / range
            } else if x > c && x <= b {
                2.0 * (b - x) / (range * (b - c))
            } else {
                0.0
            };
            pdf.ln()
        })
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<TriangularGradient> {
        // (dx, da, db, dc) for each piece of the pdf, zero outside the support
        let gradient = |x: f64, a: f64, b: f64, c: f64| -> [f64; 4] {
            let inv_range = 1.0 / (b - a);
            if x >= a && x < c {
                let inv_x_minus_a = if x - a > 0.0 { 1.0 / (x - a) } else { 1.0 };
                let inv_c_minus_a = 1.0 / (c - a);
                [
                    inv_x_minus_a,
                    -inv_x_minus_a + inv_range + inv_c_minus_a,
                    -inv_range,
                    -inv_c_minus_a,
                ]
            } else if x == c {
                [0.0, inv_range, -inv_range, 0.0]
            } else if x > c && x <= b {
                let inv_b_minus_x = if b - x > 0.0 { 1.0 / (b - x) } else { 1.0 };
                let inv_b_minus_c = 1.0 / (b - c);
                [
                    -inv_b_minus_x,
                    inv_range,
                    inv_b_minus_x - inv_range - inv_b_minus_c,
                    inv_b_minus_c,
                ]
            } else {
                [0.0; 4]
            }
        };
        let (a, b, c) = (&self.x_min, &self.x_max, &self.c);
        Ok(TriangularGradient {
            x: map4(x, a, b, c, |x, a, b, c| gradient(x, a, b, c)[0])?,
            x_min: map4(x, a, b, c, |x, a, b, c| gradient(x, a, b, c)[1])?,
            x_max: map4(x, a, b, c, |x, a, b, c| gradient(x, a, b, c)[2])?,
            c: map4(x, a, b, c, |x, a, b, c| gradient(x, a, b, c)[3])?,
        })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes).
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        let valid = self.c.ge(&self.x_min)?.all() && self.c.le(&self.x_max)?.all();
        if !valid {
            return invalid_parameter("the mode c must be between x_min and x_max");
        }
        let p = rng.uniform(shape);
        map4(&p, &self.x_min, &self.x_max, &self.c, |p, a, b, c| {
            let range = b - a;
            if p <= (c - a) / range {
                a + (range * (c - a) * p).sqrt()
            } else {
                b - (range * (b - c) * (1.0 - p)).sqrt()
            }
        })
    }
}

/// Uniform distribution on [x_min, x_max).
#[derive(Debug, Clone, PartialEq)]
pub struct Uniform {
    /// The lower bound.
    pub x_min: DoubleTensor,
    /// The upper bound.
    pub x_max: DoubleTensor,
}

impl Uniform {
    /// Creates the distribution from its parameters.
    pub fn new(x_min: DoubleTensor, x_max: DoubleTensor) -> Self {
        Self { x_min, x_max }
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support).
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map3(x, &self.x_min, &self.x_max, |x, min, max| {
            if x >= max || x < min {
                f64::NEG_INFINITY
            } else {
                -(max - min).ln()
            }
        })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes).
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        Ok(rng
            .uniform(shape)
            .mul(&self.x_max.sub(&self.x_min)?)?
            .add(&self.x_min)?)
    }
}

/// -1, 0 or 1 (0 for both signed zeros).
fn sign(v: f64) -> f64 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}
