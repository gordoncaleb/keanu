//! Positive families for durations, magnitudes and scales: the inverse Gaussian, the half Student-t, the
//! log-logistic, the Rice and the generalized gamma distributions.

// the negated comparisons (`!(sigma > 0.0)`) are deliberate: they also catch NaN
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use super::derived::{check_all, log_complement, outside, uniform_open};
use super::*;
use crate::continuous::positive::{sample_log_gamma, sample_student_scale};
use keanu_tensor::special::logit;

derived_distribution! {
    /// Inverse Gaussian (Wald) distribution with mean `mu` (μ > 0) and shape `lambda` (λ > 0), as PyMC's
    /// `Wald(μ, λ)` (Stan has no built-in inverse Gaussian):
    ///
    /// ```text
    /// p(x) = √(λ/(2πx³)) exp(−λ(x − μ)²/(2μ²x))
    /// P(X <= x) = Φ(√(λ/x)(x/μ − 1)) + e^{2λ/μ} Φ(−√(λ/x)(x/μ + 1))
    /// ```
    ///
    /// on x > 0. The first passage time of a Brownian motion with drift; its variance is μ³/λ.
    InverseGaussian, InverseGaussianGradient, inputs = 3,
    (
        /// The mean μ.
        mu,
        /// The shape λ.
        lambda,
    ),
    log_prob = inverse_gaussian_log_prob,
    log_cdf = inverse_gaussian_log_cdf,
}

fn inverse_gaussian_log_prob<const K: usize, const H: usize>(
    x: Jet<K, H>,
    [mu, lambda]: [Jet<K, H>; 2],
) -> Jet<K, H> {
    if !(mu.v > 0.0 && lambda.v > 0.0) {
        return Jet::constant(f64::NAN);
    }
    if !(x.v > 0.0) {
        return Jet::constant(f64::NEG_INFINITY);
    }
    0.5 * lambda.ln()
        - LN_SQRT_2PI
        - 1.5 * x.ln()
        - lambda * (x - mu).square() / (2.0 * mu.square() * x)
}

fn inverse_gaussian_log_cdf<const K: usize, const H: usize>(
    x: Jet<K, H>,
    [mu, lambda]: [Jet<K, H>; 2],
    upper: bool,
) -> Jet<K, H> {
    if !(mu.v > 0.0 && lambda.v > 0.0) {
        return Jet::constant(f64::NAN);
    }
    if !(x.v > 0.0) {
        return outside(true, upper);
    }
    let r = (lambda / x).sqrt();
    let a = r * (x / mu - 1.0);
    // ln(e^{2λ/μ} Φ(−√(λ/x)(x/μ + 1)))
    let second = 2.0 * lambda / mu + (-(r * (x / mu + 1.0))).log_ndtr();
    let log_cdf = a.log_ndtr().log_add_exp(second);
    if !upper {
        return log_cdf;
    }
    // the survival function Φ(−a) − e^{2λ/μ}Φ(b) directly where it is below ½, else from the CDF
    let first = (-a).log_ndtr();
    let direct = first + (second - first).log1mexp();
    log_complement(direct, || log_cdf)
}

impl InverseGaussian<'_> {
    /// Draws a sample of the given shape (broadcast with the parameters' shapes) by the transformation with
    /// multiple roots of Michael, Schucany and Haas (1976). An error unless μ and λ are positive.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        sample_each(shape, &[&self.mu, &self.lambda], |p| {
            let (mu, lambda) = (p[0], p[1]);
            if !(mu > 0.0 && lambda > 0.0) {
                return invalid_parameter(format!(
                    "mu and lambda must be positive, got {mu} and {lambda}"
                ));
            }
            let n = rng.next_gaussian();
            let y = n * n;
            let my = mu * y;
            let x = mu + mu * my / (2.0 * lambda)
                - mu / (2.0 * lambda) * (4.0 * lambda * my + my * my).sqrt();
            Ok(if rng.next_f64() <= mu / (mu + x) {
                x
            } else {
                mu * mu / x
            })
        })
    }
}

derived_distribution! {
    /// Half Student-t distribution with `nu` (ν > 0, real) degrees of freedom and scale `sigma` (σ > 0), as PyMC's
    /// `HalfStudentT(ν, σ)`: the Student-t folded at 0, with t = x/σ,
    ///
    /// ```text
    /// p(x) = 2 Γ((ν + 1)/2) / (Γ(ν/2) √(νπ) σ) · (1 + t²/ν)^{−(ν+1)/2}
    /// P(X <= x) = I_{t²/(ν+t²)}(½, ν/2)
    /// ```
    ///
    /// on x >= 0, with I the regularised incomplete beta function (whose shape derivatives give the CDF's
    /// derivatives in ν). A weakly informative prior for scales.
    HalfStudentT, HalfStudentTGradient, inputs = 3,
    (
        /// The degrees of freedom ν.
        nu,
        /// The scale σ.
        sigma,
    ),
    log_prob = half_student_t_log_prob,
    log_cdf = half_student_t_log_cdf,
}

fn half_student_t_log_prob<const K: usize, const H: usize>(
    x: Jet<K, H>,
    [nu, sigma]: [Jet<K, H>; 2],
) -> Jet<K, H> {
    if !(nu.v > 0.0 && sigma.v > 0.0) {
        return Jet::constant(f64::NAN);
    }
    if x.v < 0.0 {
        return Jet::constant(f64::NEG_INFINITY);
    }
    let t = x / sigma;
    let half = (nu + 1.0) * 0.5;
    LN_2 + half.ln_gamma()
        - (nu * 0.5).ln_gamma()
        - 0.5 * (nu * PI).ln()
        - sigma.ln()
        - half * (t.square() / nu).ln_1p()
}

fn half_student_t_log_cdf<const K: usize, const H: usize>(
    x: Jet<K, H>,
    [nu, sigma]: [Jet<K, H>; 2],
    upper: bool,
) -> Jet<K, H> {
    if !(nu.v > 0.0 && sigma.v > 0.0) {
        return Jet::constant(f64::NAN);
    }
    if !(x.v > 0.0) {
        return outside(true, upper);
    }
    let t2 = (x / sigma).square();
    let w = t2 / (nu + t2);
    let w_complement = nu / (nu + t2);
    let half = Jet::constant(0.5);
    if upper {
        w_complement.log_inc_beta(w.v, nu * 0.5, half)
    } else {
        w.log_inc_beta(w_complement.v, half, nu * 0.5)
    }
}

impl HalfStudentT<'_> {
    /// Draws a sample of the given shape (broadcast with the parameters' shapes), as σ|Z|/√(G/ν) with Z standard
    /// normal and G chi-squared with ν degrees of freedom. An error unless ν and σ are positive.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        sample_each(shape, &[&self.nu, &self.sigma], |p| {
            let (nu, sigma) = (p[0], p[1]);
            if !(nu > 0.0 && sigma > 0.0) {
                return invalid_parameter(format!(
                    "nu and sigma must be positive, got {nu} and {sigma}"
                ));
            }
            Ok(sigma * rng.next_gaussian().abs() * sample_student_scale(nu, rng)?)
        })
    }
}

derived_distribution! {
    /// Log-logistic distribution with scale `alpha` (α > 0) and shape `beta` (β > 0), as Stan's
    /// `loglogistic(α, β)`:
    ///
    /// ```text
    /// p(x) = (β/α)(x/α)^{β−1} / (1 + (x/α)^β)²,   P(X <= x) = 1 / (1 + (x/α)^{−β})
    /// ```
    ///
    /// on x > 0: ln X is logistic with location ln α and scale 1/β. α is the median.
    LogLogistic, LogLogisticGradient, inputs = 3,
    (
        /// The scale α (the median).
        alpha,
        /// The shape β.
        beta,
    ),
    log_prob = log_logistic_log_prob,
    log_cdf = log_logistic_log_cdf,
}

fn log_logistic_log_prob<const K: usize, const H: usize>(
    x: Jet<K, H>,
    [alpha, beta]: [Jet<K, H>; 2],
) -> Jet<K, H> {
    if !(alpha.v > 0.0 && beta.v > 0.0) {
        return Jet::constant(f64::NAN);
    }
    if !(x.v > 0.0) {
        return Jet::constant(f64::NEG_INFINITY);
    }
    let l = x.ln() - alpha.ln();
    beta.ln() - alpha.ln() + (beta - 1.0) * l - 2.0 * (beta * l).softplus()
}

fn log_logistic_log_cdf<const K: usize, const H: usize>(
    x: Jet<K, H>,
    [alpha, beta]: [Jet<K, H>; 2],
    upper: bool,
) -> Jet<K, H> {
    if !(alpha.v > 0.0 && beta.v > 0.0) {
        return Jet::constant(f64::NAN);
    }
    if !(x.v > 0.0) {
        return outside(true, upper);
    }
    let y = beta * (x.ln() - alpha.ln());
    if upper {
        -y.softplus()
    } else {
        -(-y).softplus()
    }
}

impl LogLogistic<'_> {
    /// The element-wise inverse CDF: the x with P(X <= x) = p, for p in [0, 1].
    pub fn inverse_cdf(&self, p: &DoubleTensor) -> Result<DoubleTensor> {
        map((p, &*self.alpha, &*self.beta), |(p, alpha, beta)| {
            alpha * (logit(p) / beta).exp()
        })
    }

    /// The element-wise inverse survival function: the x with P(X > x) = q, for q in [0, 1].
    pub fn inverse_ccdf(&self, q: &DoubleTensor) -> Result<DoubleTensor> {
        map((q, &*self.alpha, &*self.beta), |(q, alpha, beta)| {
            alpha * (-logit(q) / beta).exp()
        })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes) by inverting the CDF. An error
    /// unless α and β are positive.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        check_all(
            &[&self.alpha, &self.beta],
            |p| p[0] > 0.0 && p[1] > 0.0,
            || "alpha and beta must be positive".into(),
        )?;
        let shape = sample_shape(shape, &[self.alpha.shape(), self.beta.shape()])?;
        self.inverse_cdf(&uniform_open(&shape, rng))
    }
}

derived_distribution! {
    /// Rice distribution with noncentrality `nu` (ν >= 0) and scale `sigma` (σ > 0), as PyMC's `Rice(ν, σ)` (Stan
    /// has no built-in Rice): the length of a bivariate normal vector with mean at distance ν from the origin and
    /// independent N(0, σ²) components,
    ///
    /// ```text
    /// p(x) = (x/σ²) exp(−(x² + ν²)/(2σ²)) I₀(xν/σ²)
    /// ```
    ///
    /// on x > 0, with I₀ the modified Bessel function of order 0 (evaluated in logs). ν = 0 is the Rayleigh
    /// distribution. It has no CDF here (that needs Marcum's Q function), so it can't be truncated or censored.
    Rice, RiceGradient, inputs = 3,
    (
        /// The noncentrality ν.
        nu,
        /// The scale σ.
        sigma,
    ),
    log_prob = rice_log_prob,
}

fn rice_log_prob<const K: usize, const H: usize>(
    x: Jet<K, H>,
    [nu, sigma]: [Jet<K, H>; 2],
) -> Jet<K, H> {
    if !(nu.v >= 0.0 && sigma.v > 0.0) {
        return Jet::constant(f64::NAN);
    }
    if !(x.v > 0.0) {
        return Jet::constant(f64::NEG_INFINITY);
    }
    let variance = sigma.square();
    x.ln() - variance.ln() - (x.square() + nu.square()) / (2.0 * variance)
        + (x * nu / variance).log_bessel_i0()
}

impl Rice<'_> {
    /// Draws a sample of the given shape (broadcast with the parameters' shapes), as the length of
    /// (ν + σZ₁, σZ₂) with Z₁, Z₂ standard normal. An error unless ν is non-negative and σ positive.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        sample_each(shape, &[&self.nu, &self.sigma], |p| {
            let (nu, sigma) = (p[0], p[1]);
            if !(nu >= 0.0 && sigma > 0.0) {
                return invalid_parameter(format!(
                    "nu must be non-negative and sigma positive, got {nu} and {sigma}"
                ));
            }
            let (z1, z2) = (rng.next_gaussian(), rng.next_gaussian());
            Ok((nu + sigma * z1).hypot(sigma * z2))
        })
    }
}

derived_distribution! {
    /// Generalized gamma distribution with scale `theta` (θ > 0), shape `k` (k > 0) and power `p` (p > 0):
    ///
    /// ```text
    /// p(x) = p x^{kp−1} e^{−(x/θ)^p} / (θ^{kp} Γ(k)),   P(X <= x) = P(k, (x/θ)^p)
    /// ```
    ///
    /// on x > 0, with P the regularised lower incomplete gamma function: (X/θ)^p is Gamma(k, 1). Neither Stan nor
    /// PyMC has it built in; this is Stacy's (1962) family with Stacy's d = kp (SciPy's `gengamma(a = k, c = p,
    /// scale = θ)`), chosen so that p = 1 is the [`Gamma`] with the same θ and k, and k = 1 the [`Weibull`] with shape
    /// p and scale θ.
    GeneralizedGamma, GeneralizedGammaGradient, inputs = 4,
    (
        /// The scale θ.
        theta,
        /// The shape k.
        k,
        /// The power p.
        p,
    ),
    log_prob = generalized_gamma_log_prob,
    log_cdf = generalized_gamma_log_cdf,
}

fn generalized_gamma_valid(theta: f64, k: f64, p: f64) -> bool {
    theta > 0.0 && k > 0.0 && p > 0.0
}

fn generalized_gamma_log_prob<const K: usize, const H: usize>(
    x: Jet<K, H>,
    [theta, k, p]: [Jet<K, H>; 3],
) -> Jet<K, H> {
    if !generalized_gamma_valid(theta.v, k.v, p.v) {
        return Jet::constant(f64::NAN);
    }
    if !(x.v > 0.0) {
        return Jet::constant(f64::NEG_INFINITY);
    }
    let ln_x = x.ln();
    // ln (x/θ)^p
    let power = p * (ln_x - theta.ln());
    p.ln() - ln_x + k * power - power.exp() - k.ln_gamma()
}

fn generalized_gamma_log_cdf<const K: usize, const H: usize>(
    x: Jet<K, H>,
    [theta, k, p]: [Jet<K, H>; 3],
    upper: bool,
) -> Jet<K, H> {
    if !generalized_gamma_valid(theta.v, k.v, p.v) {
        return Jet::constant(f64::NAN);
    }
    if !(x.v > 0.0) {
        return outside(true, upper);
    }
    let y = (p * (x.ln() - theta.ln())).exp();
    if y.v == 0.0 {
        return outside(true, upper);
    }
    y.log_gamma_p(k, upper)
}

impl GeneralizedGamma<'_> {
    /// Draws a sample of the given shape (broadcast with the parameters' shapes), as θ G^{1/p} with G ~ Gamma(k, 1).
    /// An error unless θ, k and p are positive.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        check_all(
            &[&self.theta, &self.k, &self.p],
            |q| generalized_gamma_valid(q[0], q[1], q[2]),
            || "theta, k and p must be positive".into(),
        )?;
        sample_each(shape, &[&self.theta, &self.k, &self.p], |q| {
            // from the Gamma draw's log, which doesn't underflow for a small k; the draw stays positive
            Ok((q[0] * (sample_log_gamma(q[1], rng)? / q[2]).exp()).max(f64::from_bits(1)))
        })
    }
}
