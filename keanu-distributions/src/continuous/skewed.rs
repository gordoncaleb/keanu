//! Skewed families on the real line: the skew normal, the exponentially modified Gaussian and the asymmetric
//! Laplace.

// the negated comparisons (`!(sigma > 0.0)`) are deliberate: they also catch NaN
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use super::derived::{check_all, log_complement, log_from_either, uniform_open};
use super::*;

derived_distribution! {
    /// Skew normal distribution with location `xi` (ξ), scale `omega` (ω > 0) and shape `alpha` (α), as Stan's
    /// `skew_normal(ξ, ω, α)`: with z = (x − ξ)/ω,
    ///
    /// ```text
    /// p(x) = (2/ω) φ(z) Φ(αz)
    /// ```
    ///
    /// where φ and Φ are the standard normal density and CDF. α = 0 is the normal N(ξ, ω); α > 0 skews to the
    /// right. The CDF is Φ(z) − 2T(z, α), with T Owen's T function ([`keanu_tensor::special::owens_t`]), as in
    /// Stan: in the far tail on the short side (the left for α > 0) that difference cancels, so the log CDF there
    /// has an absolute rather than a relative error of about 1e-16, and both tails underflow to −∞ once their
    /// probability is below the smallest f64.
    SkewNormal, SkewNormalGradient, inputs = 4,
    (
        /// The location ξ.
        xi,
        /// The scale ω.
        omega,
        /// The shape α.
        alpha,
    ),
    log_prob = skew_normal_log_prob,
    log_cdf = skew_normal_log_cdf,
}

fn skew_normal_log_prob<const K: usize>(x: Jet<K>, [xi, omega, alpha]: [Jet<K>; 3]) -> Jet<K> {
    if !(omega.v > 0.0) || alpha.v.is_nan() {
        return Jet::constant(f64::NAN);
    }
    let z = (x - xi) / omega;
    (LN_2 - LN_SQRT_2PI) - omega.ln() - 0.5 * z.square() + (alpha * z).log_ndtr()
}

fn skew_normal_log_cdf<const K: usize>(
    x: Jet<K>,
    [xi, omega, alpha]: [Jet<K>; 3],
    upper: bool,
) -> Jet<K> {
    if !(omega.v > 0.0) || alpha.v.is_nan() {
        return Jet::constant(f64::NAN);
    }
    let z = (x - xi) / omega;
    let t = z.owens_t(alpha) * 2.0;
    let cdf = z.ndtr() - t;
    let ccdf = (-z).ndtr() + t;
    if upper {
        log_from_either(ccdf, cdf)
    } else {
        log_from_either(cdf, ccdf)
    }
}

impl SkewNormal<'_> {
    /// Draws a sample of the given shape (broadcast with the parameters' shapes), as ξ + ω(δ|U₀| + √(1 − δ²) U₁)
    /// with δ = α/√(1 + α²) and U₀, U₁ standard normal. An error unless ω is positive.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        sample_each(shape, &[&self.xi, &self.omega, &self.alpha], |p| {
            let (xi, omega, alpha) = (p[0], p[1], p[2]);
            if !(omega > 0.0) || alpha.is_nan() {
                return invalid_parameter(format!(
                    "omega must be positive and alpha a number, got {omega} and {alpha}"
                ));
            }
            let delta = alpha / alpha.hypot(1.0);
            let (u0, u1) = (rng.next_gaussian(), rng.next_gaussian());
            Ok(xi + omega * (delta * u0.abs() + (1.0 - delta * delta).sqrt() * u1))
        })
    }
}

derived_distribution! {
    /// Exponentially modified Gaussian distribution, the sum of N(μ, σ) and an independent exponential with rate
    /// λ, as Stan's `exp_mod_normal(μ, σ, λ)`:
    ///
    /// ```text
    /// p(x) = λ exp(λ(μ − x) + λ²σ²/2) Φ((x − μ)/σ − λσ)
    /// ```
    ///
    /// for σ > 0 and λ > 0 (the ex-Gaussian of response-time modelling, PyMC's `ExGaussian` with ν = 1/λ). The CDF
    /// is Φ(u) − p(x)/λ with u = (x − μ)/σ, evaluated in logs; in the far left tail that difference cancels, so
    /// the log CDF there loses relative precision (about 1e-16 × u²/2 in absolute terms).
    ExGaussian, ExGaussianGradient, inputs = 4,
    (
        /// The mean μ of the normal part.
        mu,
        /// The standard deviation σ of the normal part.
        sigma,
        /// The rate λ of the exponential part.
        lambda,
    ),
    log_prob = ex_gaussian_log_prob,
    log_cdf = ex_gaussian_log_cdf,
}

/// The parameters are valid: σ > 0 and λ > 0.
fn ex_gaussian_valid<const K: usize>(sigma: Jet<K>, lambda: Jet<K>) -> bool {
    sigma.v > 0.0 && lambda.v > 0.0
}

fn ex_gaussian_log_prob<const K: usize>(x: Jet<K>, [mu, sigma, lambda]: [Jet<K>; 3]) -> Jet<K> {
    if !ex_gaussian_valid(sigma, lambda) {
        return Jet::constant(f64::NAN);
    }
    let u = (x - mu) / sigma;
    lambda.ln()
        + lambda * (mu - x)
        + 0.5 * (lambda * sigma).square()
        + (u - lambda * sigma).log_ndtr()
}

fn ex_gaussian_log_cdf<const K: usize>(
    x: Jet<K>,
    [mu, sigma, lambda]: [Jet<K>; 3],
    upper: bool,
) -> Jet<K> {
    if !ex_gaussian_valid(sigma, lambda) {
        return Jet::constant(f64::NAN);
    }
    let u = (x - mu) / sigma;
    // ln(p(x)/λ)
    let log_tail =
        lambda * (mu - x) + 0.5 * (lambda * sigma).square() + (u - lambda * sigma).log_ndtr();
    // the survival function Φ(−u) + p(x)/λ is a sum, accurate in logs; the CDF Φ(u) − p(x)/λ a difference
    let log_ccdf = (-u).log_ndtr().log_add_exp(log_tail);
    if upper {
        return log_ccdf;
    }
    let log_phi = u.log_ndtr();
    log_complement(log_phi + (log_tail - log_phi).log1mexp(), || log_ccdf)
}

impl ExGaussian<'_> {
    /// Draws a sample of the given shape (broadcast with the parameters' shapes), as a normal plus an exponential
    /// draw. An error unless σ and λ are positive.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        sample_each(shape, &[&self.mu, &self.sigma, &self.lambda], |p| {
            let (mu, sigma, lambda) = (p[0], p[1], p[2]);
            if !(sigma > 0.0 && lambda > 0.0) {
                return invalid_parameter(format!(
                    "sigma and lambda must be positive, got {sigma} and {lambda}"
                ));
            }
            let exponential = -rng.next_f64_non_zero().ln() / lambda;
            Ok(mu + sigma * rng.next_gaussian() + exponential)
        })
    }
}

derived_distribution! {
    /// Asymmetric Laplace distribution in the quantile parameterisation of Stan's
    /// `skew_double_exponential(μ, σ, τ)`: location μ, scale σ > 0 and skewness τ in (0, 1),
    ///
    /// ```text
    /// p(x) = 2τ(1 − τ)/σ · exp(−(2/σ) ρ_τ(x − μ)),   ρ_τ(u) = u (τ − 1[u < 0])
    /// ```
    ///
    /// so μ is the τ-quantile (the check loss ρ_τ of quantile regression). τ = ½ is the Laplace distribution with
    /// scale σ/2. PyMC's `AsymmetricLaplace(b, κ, μ)` is the same family with σ = 2/(b(κ + 1/κ)) and τ = κ²/(1 + κ²).
    /// At x = μ the density has a kink, where the gradient and second derivatives take the right-hand side's values.
    AsymmetricLaplace, AsymmetricLaplaceGradient, inputs = 4,
    (
        /// The location μ (the τ-quantile).
        mu,
        /// The scale σ.
        sigma,
        /// The skewness τ in (0, 1).
        tau,
    ),
    log_prob = asymmetric_laplace_log_prob,
    log_cdf = asymmetric_laplace_log_cdf,
}

fn asymmetric_laplace_valid(sigma: f64, tau: f64) -> bool {
    sigma > 0.0 && tau > 0.0 && tau < 1.0
}

fn asymmetric_laplace_log_prob<const K: usize>(x: Jet<K>, [mu, sigma, tau]: [Jet<K>; 3]) -> Jet<K> {
    if !asymmetric_laplace_valid(sigma.v, tau.v) {
        return Jet::constant(f64::NAN);
    }
    let d = x - mu;
    let base = LN_2 + tau.ln() + (1.0 - tau).ln() - sigma.ln();
    if d.v < 0.0 {
        base + 2.0 * (1.0 - tau) * d / sigma
    } else {
        base - 2.0 * tau * d / sigma
    }
}

fn asymmetric_laplace_log_cdf<const K: usize>(
    x: Jet<K>,
    [mu, sigma, tau]: [Jet<K>; 3],
    upper: bool,
) -> Jet<K> {
    if !asymmetric_laplace_valid(sigma.v, tau.v) {
        return Jet::constant(f64::NAN);
    }
    let d = x - mu;
    if d.v < 0.0 {
        let log_cdf = tau.ln() + 2.0 * (1.0 - tau) * d / sigma;
        if upper { log_cdf.log1mexp() } else { log_cdf }
    } else {
        let log_ccdf = (1.0 - tau).ln() - 2.0 * tau * d / sigma;
        if upper { log_ccdf } else { log_ccdf.log1mexp() }
    }
}

impl AsymmetricLaplace<'_> {
    /// The element-wise inverse CDF: the x with P(X <= x) = p, for p in [0, 1].
    pub fn inverse_cdf(&self, p: &DoubleTensor) -> Result<DoubleTensor> {
        map(
            (p, &*self.mu, &*self.sigma, &*self.tau),
            |(p, mu, sigma, tau)| {
                if p < tau {
                    mu + sigma / (2.0 * (1.0 - tau)) * (p / tau).ln()
                } else {
                    mu - sigma / (2.0 * tau) * ((-p).ln_1p() - (-tau).ln_1p())
                }
            },
        )
    }

    /// The element-wise inverse survival function: the x with P(X > x) = q, for q in [0, 1].
    pub fn inverse_ccdf(&self, q: &DoubleTensor) -> Result<DoubleTensor> {
        map(
            (q, &*self.mu, &*self.sigma, &*self.tau),
            |(q, mu, sigma, tau)| {
                if q > 1.0 - tau {
                    mu + sigma / (2.0 * (1.0 - tau)) * ((-q).ln_1p() - tau.ln())
                } else {
                    mu - sigma / (2.0 * tau) * (q.ln() - (-tau).ln_1p())
                }
            },
        )
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes) by inverting the CDF. An error
    /// unless σ is positive and τ in (0, 1).
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        check_all(
            &[&self.sigma, &self.tau],
            |p| asymmetric_laplace_valid(p[0], p[1]),
            || "sigma must be positive and tau in (0, 1)".into(),
        )?;
        let shape = sample_shape(
            shape,
            &[self.mu.shape(), self.sigma.shape(), self.tau.shape()],
        )?;
        self.inverse_cdf(&uniform_open(&shape, rng))
    }
}
