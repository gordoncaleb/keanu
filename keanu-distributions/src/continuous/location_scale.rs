//! Location-scale distributions on the real line, and the half (folded) distributions built from them.

use super::*;

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

derived_distribution! {
    /// Student-t with real degrees of freedom, location and positive scale.
    LocationScaleStudentT, LocationScaleStudentTGradient, inputs = 4,
    (
        /// The positive degrees of freedom.
        nu,
        /// The location.
        mu,
        /// The positive scale.
        sigma,
    ),
    log_prob = location_scale_student_t_log_prob,
    log_cdf = location_scale_student_t_log_cdf,
}

fn student_t_log_normalizer<const K: usize, const H: usize>(nu: Jet<K, H>) -> Jet<K, H> {
    if nu.v >= 1e5 {
        // The next term in the log-gamma ratio is 1/(24ν³), below f64 precision here.
        -LN_SQRT_2PI - 0.25 / nu
    } else {
        ((nu + 1.0) * 0.5).ln_gamma() - (nu * 0.5).ln_gamma() - 0.5 * (nu.ln() + PI.ln())
    }
}

fn location_scale_student_t_log_prob<const K: usize, const H: usize>(
    x: Jet<K, H>,
    [nu, mu, sigma]: [Jet<K, H>; 3],
) -> Jet<K, H> {
    if !(nu.v.is_finite() && nu.v > 0.0 && mu.v.is_finite() && sigma.v.is_finite() && sigma.v > 0.0)
    {
        return Jet::constant(f64::NAN);
    }
    let centred = x - mu;
    let z = centred / sigma;
    let log_kernel = if z.v.abs() < 1e150 * nu.v.sqrt().min(1.0) {
        (z.square() / nu).ln_1p()
    } else {
        let magnitude = if centred.v < 0.0 { -centred } else { centred };
        (2.0 * (magnitude.ln() - sigma.ln()) - nu.ln()).softplus()
    };
    student_t_log_normalizer(nu) - sigma.ln() - (nu + 1.0) * 0.5 * log_kernel
}

fn location_scale_student_t_log_cdf<const K: usize, const H: usize>(
    x: Jet<K, H>,
    [nu, mu, sigma]: [Jet<K, H>; 3],
    upper: bool,
) -> Jet<K, H> {
    if !(nu.v.is_finite() && nu.v > 0.0 && mu.v.is_finite() && sigma.v.is_finite() && sigma.v > 0.0)
    {
        return Jet::constant(f64::NAN);
    }
    let centred = x - mu;
    let z = centred / sigma;
    if z.v.abs() < 1e-4 * nu.v.sqrt().min(1.0) {
        // Integrate the density's Taylor series at the centre, where the beta argument loses precision.
        let mass =
            z * student_t_log_normalizer(nu).exp() * (1.0 - (nu + 1.0) * z.square() / (6.0 * nu));
        return (0.5 + if upper { -mass } else { mass }).ln();
    }
    let magnitude = if centred.v < 0.0 { -centred } else { centred };
    let half_nu = nu * 0.5;
    let half = Jet::constant(0.5);
    let log_odds = 2.0 * (magnitude.ln() - sigma.ln()) - nu.ln();
    let log_w = -log_odds.softplus();
    let w = log_w.exp();
    let log_tail = if w.v < 1e-100 && log_odds.v.is_finite() {
        // The leading term of the incomplete beta integral avoids underflow in its argument.
        half_nu * log_w + student_t_log_normalizer(nu) - 0.5 * nu.ln()
    } else if log_odds.v < 0.0 {
        let complement = (-(-log_odds).softplus()).exp();
        complement.log_inc_beta(w.v, half, half_nu).log1mexp() - LN_2
    } else {
        w.log_inc_beta((-(-log_odds).softplus()).exp().v, half_nu, half) - LN_2
    };
    if (z.v > 0.0) == upper {
        log_tail
    } else {
        log_tail.log1mexp()
    }
}

impl LocationScaleStudentT<'_> {
    /// Draws a sample of the requested shape, broadcast with the parameters. Degrees of freedom and scale
    /// must be positive and all parameters finite.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        sample_each(shape, &[&self.nu, &self.mu, &self.sigma], |p| {
            let (nu, mu, sigma) = (p[0], p[1], p[2]);
            if !(nu.is_finite() && nu > 0.0 && mu.is_finite() && sigma.is_finite() && sigma > 0.0) {
                return invalid_parameter(format!(
                    "Student-t requires finite parameters and positive nu and sigma, got {p:?}"
                ));
            }
            Ok(mu + sigma * rng.next_gaussian() * super::positive::sample_student_scale(nu, rng)?)
        })
    }
}
