//! Univariate continuous distributions.
//!
//! `log_prob` is element-wise and broadcasts x with the parameters. Gradients are element-wise too, shaped like
//! the broadcast of x and the parameters; they are unspecified where `log_prob` is -infinity.

use crate::error::{Result, invalid_parameter};
use crate::param::{IntoParam, Param};
use crate::rng::KeanuRng;
use crate::util::{
    digamma, ln_gamma, map, map_n, require_ordered, require_positive, safe_ln_times, sample_each,
    sample_shape, scalar, sigmoid, softplus,
};
use keanu_tensor::DoubleTensor;
use std::f64::consts::{LN_2, PI};

const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_8;

/// Normal distribution with mean `mu` and standard deviation `sigma`.
#[derive(Debug, Clone, PartialEq)]
pub struct Gaussian<'a> {
    /// The location (mean).
    pub mu: Param<'a>,
    /// The scale (standard deviation).
    pub sigma: Param<'a>,
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

impl<'a> Gaussian<'a> {
    /// Creates the distribution from its parameters.
    pub fn new(mu: impl IntoParam<'a, f64>, sigma: impl IntoParam<'a, f64>) -> Self {
        Self {
            mu: mu.into_param(),
            sigma: sigma.into_param(),
        }
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support).
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map((x, &*self.mu, &*self.sigma), |(x, mu, sigma)| {
            gaussian_log_prob(x, mu, sigma)
        })
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<GaussianGradient> {
        let [x, mu, sigma] = map_n((x, &*self.mu, &*self.sigma), |(x, mu, sigma)| {
            gaussian_gradient(x, mu, sigma)
        })?;
        Ok(GaussianGradient { x, mu, sigma })
    }

    /// [`Self::log_prob`] and [`Self::d_log_prob`] (exactly), computed in one pass over the broadcast operands.
    pub fn log_prob_and_d_log_prob(
        &self,
        x: &DoubleTensor,
    ) -> Result<(DoubleTensor, GaussianGradient)> {
        let [log_prob, x, mu, sigma] = map_n((x, &*self.mu, &*self.sigma), |(x, mu, sigma)| {
            let [d_x, d_mu, d_sigma] = gaussian_gradient(x, mu, sigma);
            [gaussian_log_prob(x, mu, sigma), d_x, d_mu, d_sigma]
        })?;
        Ok((log_prob, GaussianGradient { x, mu, sigma }))
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes). An error unless sigma is
    /// positive.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        require_positive("sigma", &self.sigma)?;
        let shape = sample_shape(shape, &[self.mu.shape(), self.sigma.shape()])?;
        Ok(rng.gaussian(&shape).mul(&self.sigma)?.add(&self.mu)?)
    }
}

#[inline]
fn gaussian_log_prob(x: f64, mu: f64, sigma: f64) -> f64 {
    -((x - mu).powi(2) / (2.0 * sigma * sigma) + sigma.ln() + LN_SQRT_2PI)
}

/// The derivatives of [`gaussian_log_prob`] with respect to x, mu and sigma.
#[inline]
fn gaussian_gradient(x: f64, mu: f64, sigma: f64) -> [f64; 3] {
    let d_mu = (x - mu) / (sigma * sigma);
    [
        -d_mu,
        d_mu,
        (x - mu).powi(2) / (sigma * sigma * sigma) - 1.0 / sigma,
    ]
}

/// A distribution symmetric about 0 folded at 0 (the distribution of |X|): twice the density for x >= 0 and
/// -infinity below.
fn folded_log_prob(log_prob: DoubleTensor, x: &DoubleTensor) -> Result<DoubleTensor> {
    Ok(log_prob.zip_with(x, |lp, x| {
        if x < 0.0 {
            f64::NEG_INFINITY
        } else {
            lp + LN_2
        }
    })?)
}

/// A gradient of a folded distribution's log probability: the unfolded one, zero below 0.
fn folded_gradient(gradient: DoubleTensor, x: &DoubleTensor) -> Result<DoubleTensor> {
    Ok(gradient.zip_with(x, |d, x| if x < 0.0 { 0.0 } else { d })?)
}

/// Gaussian folded at 0: the density of |X| for X ~ N(0, sigma).
#[derive(Debug, Clone, PartialEq)]
pub struct HalfGaussian<'a> {
    /// The scale (standard deviation).
    pub sigma: Param<'a>,
}

/// Derivatives of [`HalfGaussian::log_prob`].
#[derive(Debug, Clone, PartialEq)]
pub struct HalfGaussianGradient {
    /// With respect to x.
    pub x: DoubleTensor,
    /// With respect to `sigma`.
    pub sigma: DoubleTensor,
}

impl<'a> HalfGaussian<'a> {
    /// Creates the distribution from its parameters.
    pub fn new(sigma: impl IntoParam<'a, f64>) -> Self {
        Self {
            sigma: sigma.into_param(),
        }
    }

    /// The unfolded distribution, borrowing sigma.
    fn gaussian(&self) -> Gaussian<'_> {
        Gaussian::new(scalar(0.0), &*self.sigma)
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support).
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        folded_log_prob(self.gaussian().log_prob(x)?, x)
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<HalfGaussianGradient> {
        let g = self.gaussian().d_log_prob(x)?;
        Ok(HalfGaussianGradient {
            x: folded_gradient(g.x, x)?,
            sigma: folded_gradient(g.sigma, x)?,
        })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes). An error unless sigma is
    /// positive.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        Ok(self.gaussian().sample(shape, rng)?.abs())
    }
}

/// Cauchy distribution with the given location and scale.
#[derive(Debug, Clone, PartialEq)]
pub struct Cauchy<'a> {
    /// The location.
    pub location: Param<'a>,
    /// The scale.
    pub scale: Param<'a>,
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

impl<'a> Cauchy<'a> {
    /// Creates the distribution from its parameters.
    pub fn new(location: impl IntoParam<'a, f64>, scale: impl IntoParam<'a, f64>) -> Self {
        Self {
            location: location.into_param(),
            scale: scale.into_param(),
        }
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support).
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map((x, &*self.location, &*self.scale), |(x, l, s)| {
            (-s.ln() - PI.ln()) - (((x - l) / s).powi(2) + 1.0).ln()
        })
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<CauchyGradient> {
        let [x, location, scale] = map_n((x, &*self.location, &*self.scale), |(x, l, s)| {
            [
                -2.0 * (x - l) / (l * l - 2.0 * l * x + s * s + x * x),
                2.0 * (x - l) / (s * s + (x - l).powi(2)),
                ((x - l).powi(2) - s * s) / (s * ((x - l).powi(2) + s * s)),
            ]
        })?;
        Ok(CauchyGradient { x, location, scale })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes). An error unless the scale is
    /// positive.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        require_positive("scale", &self.scale)?;
        let shape = sample_shape(shape, &[self.location.shape(), self.scale.shape()])?;
        let unit = rng.uniform(&shape).map(|u| ((u - 0.5) * PI).tan());
        Ok(unit.mul(&self.scale)?.add(&self.location)?)
    }
}

/// Cauchy folded at 0: the density of |X| for X ~ Cauchy(0, scale).
#[derive(Debug, Clone, PartialEq)]
pub struct HalfCauchy<'a> {
    /// The scale.
    pub scale: Param<'a>,
}

/// Derivatives of [`HalfCauchy::log_prob`].
#[derive(Debug, Clone, PartialEq)]
pub struct HalfCauchyGradient {
    /// With respect to x.
    pub x: DoubleTensor,
    /// With respect to `scale`.
    pub scale: DoubleTensor,
}

impl<'a> HalfCauchy<'a> {
    /// Creates the distribution from its parameters.
    pub fn new(scale: impl IntoParam<'a, f64>) -> Self {
        Self {
            scale: scale.into_param(),
        }
    }

    /// The unfolded distribution, borrowing the scale.
    fn cauchy(&self) -> Cauchy<'_> {
        Cauchy::new(scalar(0.0), &*self.scale)
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support).
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        folded_log_prob(self.cauchy().log_prob(x)?, x)
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<HalfCauchyGradient> {
        let g = self.cauchy().d_log_prob(x)?;
        Ok(HalfCauchyGradient {
            x: folded_gradient(g.x, x)?,
            scale: folded_gradient(g.scale, x)?,
        })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes). An error unless the scale is
    /// positive.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        Ok(self.cauchy().sample(shape, rng)?.abs())
    }
}

/// Beta distribution with shape parameters `alpha` and `beta`, on the interval [x_min, x_max].
#[derive(Debug, Clone, PartialEq)]
pub struct Beta<'a> {
    /// The alpha (first shape) parameter.
    pub alpha: Param<'a>,
    /// The beta parameter.
    pub beta: Param<'a>,
    /// The lower bound.
    pub x_min: Param<'a>,
    /// The upper bound.
    pub x_max: Param<'a>,
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

impl<'a> Beta<'a> {
    /// Beta on [x_min, x_max].
    pub fn new(
        alpha: impl IntoParam<'a, f64>,
        beta: impl IntoParam<'a, f64>,
        x_min: impl IntoParam<'a, f64>,
        x_max: impl IntoParam<'a, f64>,
    ) -> Self {
        Self {
            alpha: alpha.into_param(),
            beta: beta.into_param(),
            x_min: x_min.into_param(),
            x_max: x_max.into_param(),
        }
    }

    /// The standard Beta on [0, 1].
    pub fn standard(alpha: impl IntoParam<'a, f64>, beta: impl IntoParam<'a, f64>) -> Self {
        Self::new(alpha, beta, scalar(0.0), scalar(1.0))
    }

    /// With y = (x - x_min) / (x_max - x_min):
    /// (alpha - 1) ln y + (beta - 1) ln(1 - y) - ln B(alpha, beta) - ln(x_max - x_min), -infinity outside the bounds
    /// (everywhere if x_max <= x_min).
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map(
            (x, &*self.alpha, &*self.beta, &*self.x_min, &*self.x_max),
            |(x, a, b, min, max)| {
                // `contains` is false for NaN bounds too
                if max <= min || !(min..=max).contains(&x) {
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
        let [x, alpha, beta] = map_n(
            (x, &*self.alpha, &*self.beta, &*self.x_min, &*self.x_max),
            |(x, a, b, min, max)| {
                let range = max - min;
                let y = (x - min) / range;
                [
                    ((a - 1.0) / y - (b - 1.0) / (1.0 - y)) / range,
                    y.ln() + (digamma(a + b) - digamma(a)),
                    (1.0 - y).ln() + (digamma(a + b) - digamma(b)),
                ]
            },
        )?;
        Ok(BetaGradient { x, alpha, beta })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes). An error unless alpha and beta
    /// are positive and x_max > x_min.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        require_positive("alpha", &self.alpha)?;
        require_positive("beta", &self.beta)?;
        require_ordered(&self.x_min, &self.x_max, "x_max must be greater than x_min")?;
        let shape = sample_shape(
            shape,
            &[
                self.alpha.shape(),
                self.beta.shape(),
                self.x_min.shape(),
                self.x_max.shape(),
            ],
        )?;
        let y1 = Gamma::new(scalar(1.0), &*self.alpha).sample(&shape, rng)?;
        let y2 = Gamma::new(scalar(1.0), &*self.beta).sample(&shape, rng)?;
        map(
            (
                &y1,
                &y2,
                &*self.alpha,
                &*self.beta,
                &*self.x_min,
                &*self.x_max,
            ),
            |(y1, y2, alpha, beta, min, max)| {
                // same arithmetic as Java: from the upper bound when alpha < beta, from the lower bound otherwise
                if alpha < beta {
                    max - y2 / (y1 + y2) * (max - min)
                } else {
                    min + y1 / (y1 + y2) * (max - min)
                }
            },
        )
    }
}

/// Chi-squared distribution with `k` degrees of freedom.
#[derive(Debug, Clone, PartialEq)]
pub struct ChiSquared<'a> {
    /// The shape parameter k.
    pub k: Param<'a, i64>,
}

/// Derivatives of [`ChiSquared::log_prob`].
#[derive(Debug, Clone, PartialEq)]
pub struct ChiSquaredGradient {
    /// With respect to x.
    pub x: DoubleTensor,
}

impl<'a> ChiSquared<'a> {
    /// Creates the distribution from its parameters.
    pub fn new(k: impl IntoParam<'a, i64>) -> Self {
        Self { k: k.into_param() }
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support).
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map((x, &*self.k), |(x, k)| {
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
            x: map((x, &*self.k), |(x, k)| (k as f64 / 2.0 - 1.0) / x - 0.5)?,
        })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes). An error unless k is positive.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        crate::util::require(&self.k, |k| k > 0, || "k must be positive".into())?;
        Gamma::new(scalar(2.0), self.k.to_double() / 2.0).sample(shape, rng)
    }
}

/// Exponential distribution with the given rate (pdf = rate e^(-rate x)).
#[derive(Debug, Clone, PartialEq)]
pub struct Exponential<'a> {
    /// The rate (1 / mean).
    pub rate: Param<'a>,
}

/// Derivatives of [`Exponential::log_prob`].
#[derive(Debug, Clone, PartialEq)]
pub struct ExponentialGradient {
    /// With respect to x.
    pub x: DoubleTensor,
    /// With respect to `rate`.
    pub rate: DoubleTensor,
}

impl<'a> Exponential<'a> {
    /// Creates the distribution from its parameters.
    pub fn new(rate: impl IntoParam<'a, f64>) -> Self {
        Self {
            rate: rate.into_param(),
        }
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support).
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map((x, &*self.rate), |(x, rate)| {
            if x < 0.0 {
                f64::NEG_INFINITY
            } else {
                rate.ln() - x * rate
            }
        })
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<ExponentialGradient> {
        let [x, rate] = map_n((x, &*self.rate), |(x, rate)| [-rate, 1.0 / rate - x])?;
        Ok(ExponentialGradient { x, rate })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes). An error unless the rate is
    /// positive.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        require_positive("rate", &self.rate)?;
        let shape = sample_shape(shape, &[self.rate.shape()])?;
        Ok(rng.uniform(&shape).map(|u| -u.ln()).div(&self.rate)?)
    }
}

/// Gamma distribution with scale `theta` and shape `k`.
#[derive(Debug, Clone, PartialEq)]
pub struct Gamma<'a> {
    /// The scale parameter θ.
    pub theta: Param<'a>,
    /// The shape parameter k.
    pub k: Param<'a>,
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

impl<'a> Gamma<'a> {
    /// Creates the distribution from its parameters.
    pub fn new(theta: impl IntoParam<'a, f64>, k: impl IntoParam<'a, f64>) -> Self {
        Self {
            theta: theta.into_param(),
            k: k.into_param(),
        }
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support).
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map((x, &*self.theta, &*self.k), |(x, theta, k)| {
            if x < 0.0 {
                return f64::NEG_INFINITY;
            }
            safe_ln_times(x, k - 1.0) - ln_gamma(k) - x / theta - k * theta.ln()
        })
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<GammaGradient> {
        let [x, theta, k] = map_n((x, &*self.theta, &*self.k), |(x, t, k)| {
            [
                (k - 1.0) / x - 1.0 / t,
                -((t * k - x) / (t * t)),
                x.ln() - t.ln() - digamma(k),
            ]
        })?;
        Ok(GammaGradient { x, theta, k })
    }

    /// Cheng's algorithm for k > 1 and Ahrens–Dieter GS for k < 1. An error unless theta and k are positive.
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
pub struct InverseGamma<'a> {
    /// The alpha (first shape) parameter.
    pub alpha: Param<'a>,
    /// The beta parameter.
    pub beta: Param<'a>,
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

impl<'a> InverseGamma<'a> {
    /// Creates the distribution from its parameters.
    pub fn new(alpha: impl IntoParam<'a, f64>, beta: impl IntoParam<'a, f64>) -> Self {
        Self {
            alpha: alpha.into_param(),
            beta: beta.into_param(),
        }
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support).
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map((x, &*self.alpha, &*self.beta), |(x, a, b)| {
            if x <= 0.0 {
                return f64::NEG_INFINITY;
            }
            a * b.ln() + x.ln() * (-a - 1.0) - ln_gamma(a) - b / x
        })
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<InverseGammaGradient> {
        let [x, alpha, beta] = map_n((x, &*self.alpha, &*self.beta), |(x, a, b)| {
            [
                (x * -(a + 1.0) + b) / (x * x),
                -x.ln() - digamma(a) + b.ln(),
                -1.0 / x + a / b,
            ]
        })?;
        Ok(InverseGammaGradient { x, alpha, beta })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes). An error unless alpha and beta
    /// are positive.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        require_positive("alpha", &self.alpha)?;
        require_positive("beta", &self.beta)?;
        let gamma = Gamma::new(self.beta.reciprocal(), &*self.alpha).sample(shape, rng)?;
        Ok(gamma.reciprocal())
    }
}

/// Laplace distribution with location `mu` and scale `beta`.
#[derive(Debug, Clone, PartialEq)]
pub struct Laplace<'a> {
    /// The location (mean).
    pub mu: Param<'a>,
    /// The beta parameter.
    pub beta: Param<'a>,
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

impl<'a> Laplace<'a> {
    /// Creates the distribution from its parameters.
    pub fn new(mu: impl IntoParam<'a, f64>, beta: impl IntoParam<'a, f64>) -> Self {
        Self {
            mu: mu.into_param(),
            beta: beta.into_param(),
        }
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support).
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map((x, &*self.mu, &*self.beta), |(x, mu, beta)| {
            -((mu - x).abs() / beta + (2.0 * beta).ln())
        })
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<LaplaceGradient> {
        let [x, mu, beta] = map_n((x, &*self.mu, &*self.beta), |(x, m, b)| {
            [
                // sign(mu - x) / beta, with the subgradient at the mode taken to be 0 (not 0 / 0)
                sign(m - x) / b,
                sign(x - m) / b,
                ((m - x).abs() - b) / (b * b),
            ]
        })?;
        Ok(LaplaceGradient { x, mu, beta })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes). An error unless beta is
    /// positive.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        sample_each(shape, &[&self.mu, &self.beta], |p| {
            let (mu, beta) = (p[0], p[1]);
            if beta <= 0.0 || beta.is_nan() {
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
pub struct Logistic<'a> {
    /// The location (mean).
    pub mu: Param<'a>,
    /// The scale.
    pub s: Param<'a>,
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

impl<'a> Logistic<'a> {
    /// Creates the distribution from its parameters.
    pub fn new(mu: impl IntoParam<'a, f64>, s: impl IntoParam<'a, f64>) -> Self {
        Self {
            mu: mu.into_param(),
            s: s.into_param(),
        }
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support).
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map((x, &*self.mu, &*self.s), |(x, mu, s)| {
            let m = (mu - x) / s;
            m - s.ln() - softplus(m) * 2.0
        })
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<LogisticGradient> {
        let [x, mu, s] = map_n((x, &*self.mu, &*self.s), |(x, mu, s)| {
            let m = (mu - x) / s;
            let a_term = 2.0 * sigmoid(m);
            let d_x = (a_term - 1.0) / s;
            [d_x, -d_x, (a_term * m - m - 1.0) / s]
        })?;
        Ok(LogisticGradient { x, mu, s })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes). An error unless s is positive.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        require_positive("s", &self.s)?;
        let shape = sample_shape(shape, &[self.mu.shape(), self.s.shape()])?;
        let unit = rng.uniform(&shape).map(|u| (1.0 / u - 1.0).ln());
        Ok(unit.mul(&self.s)?.add(&self.mu)?)
    }
}

/// Log-normal distribution: ln X ~ N(mu, sigma).
#[derive(Debug, Clone, PartialEq)]
pub struct LogNormal<'a> {
    /// The location (mean).
    pub mu: Param<'a>,
    /// The scale (standard deviation).
    pub sigma: Param<'a>,
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

impl<'a> LogNormal<'a> {
    /// Creates the distribution from its parameters.
    pub fn new(mu: impl IntoParam<'a, f64>, sigma: impl IntoParam<'a, f64>) -> Self {
        Self {
            mu: mu.into_param(),
            sigma: sigma.into_param(),
        }
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support).
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map((x, &*self.mu, &*self.sigma), |(x, mu, sigma)| {
            if x <= 0.0 {
                return f64::NEG_INFINITY;
            }
            -((x.ln() - mu).powi(2) / (2.0 * sigma * sigma) + (sigma * x).ln() + LN_SQRT_2PI)
        })
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<LogNormalGradient> {
        let [x, mu, sigma] = map_n((x, &*self.mu, &*self.sigma), |(x, m, s)| {
            [
                -((x.ln() - m) / (s * s) + 1.0) / x,
                (x.ln() - m) / (s * s),
                (x.ln() - m).powi(2) / (s * s * s) - 1.0 / s,
            ]
        })?;
        Ok(LogNormalGradient { x, mu, sigma })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes). An error unless sigma is
    /// positive.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        Ok(Gaussian::new(&*self.mu, &*self.sigma)
            .sample(shape, rng)?
            .exp())
    }
}

/// Pareto distribution with minimum value `location` and shape `scale`.
#[derive(Debug, Clone, PartialEq)]
pub struct Pareto<'a> {
    /// The location.
    pub location: Param<'a>,
    /// The scale.
    pub scale: Param<'a>,
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

impl<'a> Pareto<'a> {
    /// Creates the distribution from its parameters.
    pub fn new(location: impl IntoParam<'a, f64>, scale: impl IntoParam<'a, f64>) -> Self {
        Self {
            location: location.into_param(),
            scale: scale.into_param(),
        }
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support).
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map((x, &*self.location, &*self.scale), |(x, l, s)| {
            // `!(.. > 0)` also rejects NaN parameters
            if !(l > 0.0 && s > 0.0) || x < l {
                f64::NEG_INFINITY
            } else {
                s.ln() + l.ln() * s - (s + 1.0) * x.ln()
            }
        })
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<ParetoGradient> {
        let [x, location, scale] = map_n((x, &*self.location, &*self.scale), |(x, l, s)| {
            [-((s + 1.0) / x), s / l, 1.0 / s + l.ln() - x.ln()]
        })?;
        Ok(ParetoGradient { x, location, scale })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes). An error unless the location
    /// and scale are positive.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        require_positive("location", &self.location)?;
        require_positive("scale", &self.scale)?;
        let shape = sample_shape(shape, &[self.location.shape(), self.scale.shape()])?;
        let u = rng.uniform(&shape);
        map((&u, &*self.location, &*self.scale), |(u, l, s)| {
            l / (1.0 - u).powf(1.0 / s)
        })
    }
}

/// Student's t distribution with `v` degrees of freedom.
#[derive(Debug, Clone, PartialEq)]
pub struct StudentT<'a> {
    /// The degrees of freedom.
    pub v: Param<'a, i64>,
}

/// Derivatives of [`StudentT::log_prob`].
#[derive(Debug, Clone, PartialEq)]
pub struct StudentTGradient {
    /// With respect to x.
    pub x: DoubleTensor,
}

impl<'a> StudentT<'a> {
    /// Creates the distribution from its parameters.
    pub fn new(v: impl IntoParam<'a, i64>) -> Self {
        Self { v: v.into_param() }
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support).
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        let half_ln_pi = PI.ln() / 2.0;
        map((x, &*self.v), |(t, v)| {
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
            x: map((x, &*self.v), |(t, v)| {
                -t * (v as f64 + 1.0) / (t * t + v as f64)
            })?,
        })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes). An error unless v is positive.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        let shape = sample_shape(shape, &[self.v.shape()])?;
        let chi2 = ChiSquared::new(&*self.v).sample(&shape, rng)?;
        let normal = rng.gaussian(&shape);
        let denominator = map((&chi2, &*self.v), |(c, v)| (c / v as f64).sqrt())?;
        Ok(normal.div(&denominator)?)
    }
}

/// Triangular distribution on [x_min, x_max] with mode `c`.
#[derive(Debug, Clone, PartialEq)]
pub struct Triangular<'a> {
    /// The lower bound.
    pub x_min: Param<'a>,
    /// The upper bound.
    pub x_max: Param<'a>,
    /// The mode.
    pub c: Param<'a>,
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

impl<'a> Triangular<'a> {
    /// Creates the distribution from its parameters.
    pub fn new(
        x_min: impl IntoParam<'a, f64>,
        x_max: impl IntoParam<'a, f64>,
        c: impl IntoParam<'a, f64>,
    ) -> Self {
        Self {
            x_min: x_min.into_param(),
            x_max: x_max.into_param(),
            c: c.into_param(),
        }
    }

    /// 2(x - a)/((b - a)(c - a)) below the mode, 2/(b - a) at it, 2(b - x)/((b - a)(b - c)) above it; -infinity
    /// everywhere unless a < b and a <= c <= b.
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map((x, &*self.x_min, &*self.x_max, &*self.c), |(x, a, b, c)| {
            // written so that NaN parameters are rejected too
            if !(a < b && a <= c && c <= b) {
                return f64::NEG_INFINITY;
            }
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
        let [x, x_min, x_max, c] =
            map_n((x, &*self.x_min, &*self.x_max, &*self.c), |(x, a, b, c)| {
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
            })?;
        Ok(TriangularGradient { x, x_min, x_max, c })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes). An error unless x_min < x_max
    /// and x_min <= c <= x_max.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        require_ordered(&self.x_min, &self.x_max, "x_max must be greater than x_min")?;
        let valid = self.c.ge(&self.x_min)?.all() && self.c.le(&self.x_max)?.all();
        if !valid {
            return invalid_parameter("the mode c must be between x_min and x_max");
        }
        let shape = sample_shape(
            shape,
            &[self.x_min.shape(), self.x_max.shape(), self.c.shape()],
        )?;
        let p = rng.uniform(&shape);
        map(
            (&p, &*self.x_min, &*self.x_max, &*self.c),
            |(p, a, b, c)| {
                let range = b - a;
                if p <= (c - a) / range {
                    a + (range * (c - a) * p).sqrt()
                } else {
                    b - (range * (b - c) * (1.0 - p)).sqrt()
                }
            },
        )
    }
}

/// Uniform distribution on [x_min, x_max).
#[derive(Debug, Clone, PartialEq)]
pub struct Uniform<'a> {
    /// The lower bound.
    pub x_min: Param<'a>,
    /// The upper bound.
    pub x_max: Param<'a>,
}

impl<'a> Uniform<'a> {
    /// Creates the distribution from its parameters.
    pub fn new(x_min: impl IntoParam<'a, f64>, x_max: impl IntoParam<'a, f64>) -> Self {
        Self {
            x_min: x_min.into_param(),
            x_max: x_max.into_param(),
        }
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support,
    /// so everywhere if x_max <= x_min).
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map((x, &*self.x_min, &*self.x_max), |(x, min, max)| {
            // `contains` is false for NaN bounds too
            if max <= min || !(min..max).contains(&x) {
                f64::NEG_INFINITY
            } else {
                -(max - min).ln()
            }
        })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes). An error unless
    /// x_max > x_min.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        require_ordered(&self.x_min, &self.x_max, "x_max must be greater than x_min")?;
        let shape = sample_shape(shape, &[self.x_min.shape(), self.x_max.shape()])?;
        Ok(rng
            .uniform(&shape)
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
