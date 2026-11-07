//! Distributions on the positive reals (or above a location).

use super::*;

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

/// A Gamma(k, θ) draw (shape k, scale θ). A draw below the least positive float (likely for k well below 1, whose
/// draws are mostly far below it) is that float rather than 0, which is outside the support. Where the draws' ratios
/// matter, use [`sample_log_gamma`], which doesn't underflow.
pub(crate) fn sample_gamma(theta: f64, k: f64, rng: &mut KeanuRng) -> Result<f64> {
    Ok(sample_gamma_or_zero(theta, k, rng)?.max(f64::from_bits(1)))
}

/// The log of a Gamma(k, 1) draw, accurate where the draw itself would underflow: for k < 1, as
/// ln G′ + ln U / k with G′ ~ Gamma(k + 1, 1) and U ~ Uniform(0, 1) (Marsaglia & Tsang 2000), which stays finite
/// for any positive k; for k ≥ 1 the log of [`sample_gamma`]'s draw, from the same random numbers.
pub(crate) fn sample_log_gamma(k: f64, rng: &mut KeanuRng) -> Result<f64> {
    if k >= 1.0 {
        return Ok(sample_gamma(1.0, k, rng)?.ln());
    }
    let g = sample_gamma(1.0, k + 1.0, rng)?;
    Ok(g.ln() + rng.next_f64_non_zero().ln() / k)
}

/// A Dirichlet(α) draw, the normalised Gamma(α_i, 1) draws, formed from their logs: a probability vector (finite,
/// summing to 1) even when every Gamma draw would underflow (every α well below 1).
pub(crate) fn sample_dirichlet(alpha: &[f64], rng: &mut KeanuRng) -> Result<Vec<f64>> {
    let logs = alpha
        .iter()
        .map(|&a| sample_log_gamma(a, rng))
        .collect::<Result<Vec<_>>>()?;
    let max = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = logs.iter().map(|l| (l - max).exp()).collect();
    let total: f64 = weights.iter().sum();
    Ok(weights.iter().map(|w| w / total).collect())
}

/// A Beta(a, b) draw, G_a / (G_a + G_b), formed from the Gamma draws' logs so that it doesn't become 0/0 when both
/// underflow (a and b well below 1). It may round to 0 or 1.
pub(crate) fn sample_beta_ratio(a: f64, b: f64, rng: &mut KeanuRng) -> Result<f64> {
    let x = sample_log_gamma(a, rng)?;
    let y = sample_log_gamma(b, rng)?;
    Ok(sigmoid(x - y))
}

/// √(ν / χ²_ν), the scale of a Student-t draw over a normal one, formed from the χ² draw's log so that it stays finite
/// (unless the true value exceeds the largest float) when the draw underflows (ν well below 1).
pub(crate) fn sample_student_scale(nu: f64, rng: &mut KeanuRng) -> Result<f64> {
    if nu.is_nan() || nu <= 0.0 {
        return invalid_parameter(format!("nu must be positive, got {nu}"));
    }
    let log_chi2 = std::f64::consts::LN_2 + sample_log_gamma(nu / 2.0, rng)?;
    Ok((0.5 * (nu.ln() - log_chi2)).exp())
}

fn sample_gamma_or_zero(theta: f64, k: f64, rng: &mut KeanuRng) -> Result<f64> {
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

/// Weibull distribution with shape `alpha` and scale `beta`: pdf = (α/β)(x/β)^(α-1) e^(-(x/β)^α) for x >= 0.
#[derive(Debug, Clone, PartialEq)]
pub struct Weibull<'a> {
    /// The shape parameter α.
    pub alpha: Param<'a>,
    /// The scale parameter β.
    pub beta: Param<'a>,
}

/// Derivatives of [`Weibull::log_prob`] (and of [`Weibull::log_cdf`] and [`Weibull::log_ccdf`]).
#[derive(Debug, Clone, PartialEq)]
pub struct WeibullGradient {
    /// With respect to x.
    pub x: DoubleTensor,
    /// With respect to `alpha`.
    pub alpha: DoubleTensor,
    /// With respect to `beta`.
    pub beta: DoubleTensor,
}

impl<'a> Weibull<'a> {
    /// Creates the distribution from its parameters.
    pub fn new(alpha: impl IntoParam<'a, f64>, beta: impl IntoParam<'a, f64>) -> Self {
        Self {
            alpha: alpha.into_param(),
            beta: beta.into_param(),
        }
    }

    /// ln α - ln β + (α - 1) ln(x/β) - (x/β)^α element-wise, broadcasting x with the parameters (-infinity for
    /// x < 0).
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map((x, &*self.alpha, &*self.beta), |(x, a, b)| {
            if x < 0.0 {
                return f64::NEG_INFINITY;
            }
            let z = x / b;
            a.ln() - b.ln() + safe_ln_times(z, a - 1.0) - z.powf(a)
        })
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    ///
    /// They are computed from ln(x/β) = ln x - ln β, so they stay accurate where x/β underflows. At x = 0 they
    /// are the limits as x → 0: with respect to x, -1/β for α = 1 and ±infinity (the sign of α - 1) otherwise;
    /// with respect to α, -infinity.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<WeibullGradient> {
        let [x, alpha, beta] = map_n((x, &*self.alpha, &*self.beta), |(x, a, b)| {
            let ln_z = x.ln() - b.ln();
            let za = (a * ln_z).exp();
            // d/dx = (α - 1)/x - (α/β)(x/β)^(α-1), with each term's limit at x = 0 for α = 1
            let d_ln_x = if a == 1.0 { 0.0 } else { (a - 1.0) / x };
            let d_power = if a == 1.0 {
                a / b
            } else {
                a / b * ((a - 1.0) * ln_z).exp()
            };
            [
                d_ln_x - d_power,
                1.0 / a + ln_z * (1.0 - za),
                a / b * (za - 1.0),
            ]
        })?;
        Ok(WeibullGradient { x, alpha, beta })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes) by inverting the CDF. An error
    /// unless alpha and beta are positive.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        require_positive("alpha", &self.alpha)?;
        require_positive("beta", &self.beta)?;
        let shape = sample_shape(shape, &[self.alpha.shape(), self.beta.shape()])?;
        let u = rng.uniform(&shape);
        map((&u, &*self.alpha, &*self.beta), |(u, a, b)| {
            weibull_from_uniform(u, a, b)
        })
    }
}

/// A Weibull(α, β) draw from a uniform draw u in [0, 1): the inverse CDF at u, β (-ln(1 - u))^(1/α). 1 - u is in
/// (0, 1], so the draw is finite even at u = 0 (where inverting the survival function, at 1 - u, would give +∞).
fn weibull_from_uniform(u: f64, a: f64, b: f64) -> f64 {
    b * (-(-u).ln_1p()).powf(1.0 / a)
}

#[cfg(test)]
mod tests {
    use super::weibull_from_uniform;

    #[test]
    fn weibull_draws_are_finite_at_the_ends_of_the_uniform_draw() {
        // KeanuRng::next_f64 is in [0, 1), so u = 0 is a possible draw (with probability 2^-53)
        let largest_below_one = 1.0 - f64::EPSILON / 2.0;
        for (a, b) in [(0.5, 1.0), (1.0, 2.0), (3.0, 0.5)] {
            for u in [0.0, f64::MIN_POSITIVE, 0.5, largest_below_one] {
                let x = weibull_from_uniform(u, a, b);
                assert!(
                    x.is_finite() && x >= 0.0,
                    "u = {u}, alpha = {a}, beta = {b}: {x}"
                );
            }
        }
    }
}
