//! Extreme-value families: the Gumbel, the generalized extreme value and the generalized Pareto distributions.
//!
//! The generalized families are written in s = ln(1 + ξz)/ξ = z·L(ξz), with L(u) = ln(1 + u)/u continued smoothly
//! through u = 0 (where L = 1) and its derivatives taken from its power series near 0, so the value and every
//! derivative are continuous and accurate as the shape ξ passes through 0, where they become the Gumbel and the
//! exponential distributions.

// the negated comparisons (`!(sigma > 0.0)`) are deliberate: they also catch NaN
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use super::derived::{check_all, outside, uniform_open};
use super::*;

derived_distribution! {
    /// Gumbel (type I extreme value) distribution with location `mu` (μ) and scale `beta` (β > 0), as Stan's
    /// `gumbel(μ, β)`: with z = (x − μ)/β,
    ///
    /// ```text
    /// p(x) = (1/β) exp(−z − e^{−z}),   P(X <= x) = exp(−e^{−z})
    /// ```
    Gumbel, GumbelGradient, inputs = 3,
    (
        /// The location μ (the mode).
        mu,
        /// The scale β.
        beta,
    ),
    log_prob = gumbel_log_prob,
    log_cdf = gumbel_log_cdf,
}

fn gumbel_log_prob<const K: usize>(x: Jet<K>, [mu, beta]: [Jet<K>; 2]) -> Jet<K> {
    if !(beta.v > 0.0) {
        return Jet::constant(f64::NAN);
    }
    let z = (x - mu) / beta;
    -beta.ln() - z - (-z).exp()
}

fn gumbel_log_cdf<const K: usize>(x: Jet<K>, [mu, beta]: [Jet<K>; 2], upper: bool) -> Jet<K> {
    if !(beta.v > 0.0) {
        return Jet::constant(f64::NAN);
    }
    let z = (x - mu) / beta;
    if upper {
        z.log_gumbel_ccdf()
    } else {
        -(-z).exp()
    }
}

impl Gumbel<'_> {
    /// The element-wise inverse CDF: the x with P(X <= x) = p, for p in [0, 1].
    pub fn inverse_cdf(&self, p: &DoubleTensor) -> Result<DoubleTensor> {
        map((p, &*self.mu, &*self.beta), |(p, mu, beta)| {
            mu - beta * (-p.ln()).ln()
        })
    }

    /// The element-wise inverse survival function: the x with P(X > x) = q, for q in [0, 1].
    pub fn inverse_ccdf(&self, q: &DoubleTensor) -> Result<DoubleTensor> {
        map((q, &*self.mu, &*self.beta), |(q, mu, beta)| {
            mu - beta * (-(-q).ln_1p()).ln()
        })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes) by inverting the CDF. An error
    /// unless β is positive.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        require_positive("beta", &self.beta)?;
        let shape = sample_shape(shape, &[self.mu.shape(), self.beta.shape()])?;
        self.inverse_cdf(&uniform_open(&shape, rng))
    }
}

/// ln(1 + ξz)/ξ (z itself at ξ = 0), or None where 1 + ξz <= 0, outside the support.
fn generalized_log<const K: usize>(z: Jet<K>, xi: Jet<K>) -> Option<Jet<K>> {
    let u = xi * z;
    (1.0 + u.v > 0.0).then(|| z * u.log1p_over())
}

/// The standardised point z with ln(1 + ξz)/ξ = s: (e^{ξs} − 1)/ξ, or s at ξ = 0.
fn generalized_exp(s: f64, xi: f64) -> f64 {
    if xi == 0.0 { s } else { (xi * s).exp_m1() / xi }
}

derived_distribution! {
    /// Generalized extreme value distribution with location `mu` (μ), scale `sigma` (σ > 0) and shape `xi` (ξ), as
    /// PyMC's `GenExtreme(μ, σ, ξ)` (Stan has no built-in GEV), in the sign convention in which ξ > 0 is the
    /// heavy-tailed Fréchet type and ξ < 0 the bounded Weibull type (SciPy's `genextreme` uses c = −ξ): with z = (x − μ)/σ and t = (1 + ξz)^{−1/ξ} (t = e^{−z} at ξ = 0),
    ///
    /// ```text
    /// p(x) = (1/σ) t^{ξ+1} e^{−t},   P(X <= x) = e^{−t}
    /// ```
    ///
    /// on 1 + ξz > 0: x > μ − σ/ξ for ξ > 0, x < μ − σ/ξ for ξ < 0 and every x for ξ = 0, where it is the
    /// [`Gumbel`]. The density, CDF and their derivatives are continuous through ξ = 0: they are written in
    /// s = ln(1 + ξz)/ξ = z·L(ξz), with L(u) = ln(1 + u)/u continued through u = 0 and its derivatives taken from
    /// its power series near 0.
    GeneralizedExtremeValue, GeneralizedExtremeValueGradient, inputs = 4,
    (
        /// The location μ.
        mu,
        /// The scale σ.
        sigma,
        /// The shape ξ.
        xi,
    ),
    log_prob = gev_log_prob,
    log_cdf = gev_log_cdf,
}

fn gev_log_prob<const K: usize>(x: Jet<K>, [mu, sigma, xi]: [Jet<K>; 3]) -> Jet<K> {
    if !(sigma.v > 0.0) || xi.v.is_nan() {
        return Jet::constant(f64::NAN);
    }
    match generalized_log((x - mu) / sigma, xi) {
        // ln t = −s
        Some(s) => -sigma.ln() - (xi + 1.0) * s - (-s).exp(),
        None => Jet::constant(f64::NEG_INFINITY),
    }
}

fn gev_log_cdf<const K: usize>(x: Jet<K>, [mu, sigma, xi]: [Jet<K>; 3], upper: bool) -> Jet<K> {
    if !(sigma.v > 0.0) || xi.v.is_nan() {
        return Jet::constant(f64::NAN);
    }
    match generalized_log((x - mu) / sigma, xi) {
        Some(s) => {
            if upper {
                s.log_gumbel_ccdf()
            } else {
                -(-s).exp()
            }
        }
        // below the lower end for ξ > 0, above the upper end for ξ < 0
        None => outside(xi.v > 0.0, upper),
    }
}

impl GeneralizedExtremeValue<'_> {
    /// The element-wise inverse CDF: the x with P(X <= x) = p, for p in [0, 1].
    pub fn inverse_cdf(&self, p: &DoubleTensor) -> Result<DoubleTensor> {
        map(
            (p, &*self.mu, &*self.sigma, &*self.xi),
            |(p, mu, sigma, xi)| {
                // t = −ln p and s = −ln t
                mu + sigma * generalized_exp(-(-p.ln()).ln(), xi)
            },
        )
    }

    /// The element-wise inverse survival function: the x with P(X > x) = q, for q in [0, 1].
    pub fn inverse_ccdf(&self, q: &DoubleTensor) -> Result<DoubleTensor> {
        map(
            (q, &*self.mu, &*self.sigma, &*self.xi),
            |(q, mu, sigma, xi)| mu + sigma * generalized_exp(-(-(-q).ln_1p()).ln(), xi),
        )
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes) by inverting the CDF. An error
    /// unless σ is positive and ξ a number.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        check_all(
            &[&self.sigma, &self.xi],
            |p| p[0] > 0.0 && !p[1].is_nan(),
            || "sigma must be positive and xi a number".into(),
        )?;
        let shape = sample_shape(
            shape,
            &[self.mu.shape(), self.sigma.shape(), self.xi.shape()],
        )?;
        self.inverse_cdf(&uniform_open(&shape, rng))
    }
}

derived_distribution! {
    /// Generalized Pareto distribution with location `mu` (μ), scale `sigma` (σ > 0) and shape `xi` (ξ), the
    /// peaks-over-threshold distribution in its usual parameterisation (Coles 2001; neither Stan nor PyMC has it
    /// built in; it is SciPy's `genpareto(c = ξ, loc = μ, scale = σ)`): with z = (x − μ)/σ,
    ///
    /// ```text
    /// p(x) = (1/σ) (1 + ξz)^{−1/ξ − 1},   P(X > x) = (1 + ξz)^{−1/ξ}
    /// ```
    ///
    /// on z >= 0 (and z <= −1/ξ for ξ < 0); at ξ = 0 it is the exponential distribution e^{−z}/σ shifted to μ. The
    /// density, CDF and their derivatives are continuous through ξ = 0, as the [`GeneralizedExtremeValue`]'s are.
    GeneralizedPareto, GeneralizedParetoGradient, inputs = 4,
    (
        /// The location μ (the threshold, the lower end of the support).
        mu,
        /// The scale σ.
        sigma,
        /// The shape ξ.
        xi,
    ),
    log_prob = gpd_log_prob,
    log_cdf = gpd_log_cdf,
}

fn gpd_log_prob<const K: usize>(x: Jet<K>, [mu, sigma, xi]: [Jet<K>; 3]) -> Jet<K> {
    if !(sigma.v > 0.0) || xi.v.is_nan() {
        return Jet::constant(f64::NAN);
    }
    let z = (x - mu) / sigma;
    if z.v < 0.0 {
        return Jet::constant(f64::NEG_INFINITY);
    }
    match generalized_log(z, xi) {
        Some(s) => -sigma.ln() - (xi + 1.0) * s,
        None => Jet::constant(f64::NEG_INFINITY),
    }
}

fn gpd_log_cdf<const K: usize>(x: Jet<K>, [mu, sigma, xi]: [Jet<K>; 3], upper: bool) -> Jet<K> {
    if !(sigma.v > 0.0) || xi.v.is_nan() {
        return Jet::constant(f64::NAN);
    }
    let z = (x - mu) / sigma;
    if z.v < 0.0 {
        return outside(true, upper);
    }
    match generalized_log(z, xi) {
        Some(s) => {
            let log_ccdf = -s;
            if upper { log_ccdf } else { log_ccdf.log1mexp() }
        }
        None => outside(false, upper),
    }
}

impl GeneralizedPareto<'_> {
    /// The element-wise inverse CDF: the x with P(X <= x) = p, for p in [0, 1].
    pub fn inverse_cdf(&self, p: &DoubleTensor) -> Result<DoubleTensor> {
        map(
            (p, &*self.mu, &*self.sigma, &*self.xi),
            |(p, mu, sigma, xi)| mu + sigma * generalized_exp(-(-p).ln_1p(), xi),
        )
    }

    /// The element-wise inverse survival function: the x with P(X > x) = q, for q in [0, 1].
    pub fn inverse_ccdf(&self, q: &DoubleTensor) -> Result<DoubleTensor> {
        map(
            (q, &*self.mu, &*self.sigma, &*self.xi),
            |(q, mu, sigma, xi)| mu + sigma * generalized_exp(-q.ln(), xi),
        )
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes) by inverting the CDF. An error
    /// unless σ is positive and ξ a number.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        check_all(
            &[&self.sigma, &self.xi],
            |p| p[0] > 0.0 && !p[1].is_nan(),
            || "sigma must be positive and xi a number".into(),
        )?;
        let shape = sample_shape(
            shape,
            &[self.mu.shape(), self.sigma.shape(), self.xi.shape()],
        )?;
        self.inverse_ccdf(&uniform_open(&shape, rng))
    }
}
