//! Families on the unit interval (0, 1): the Kumaraswamy and the logit-normal distributions.

// the negated comparisons (`!(sigma > 0.0)`) are deliberate: they also catch NaN
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use super::derived::{check_all, outside, uniform_open};
use super::*;
use keanu_tensor::special::ndtri;

/// A quantile for the probability `p`: the bound itself for p = 0 or 1 (or NaN), otherwise moved strictly inside
/// (0, 1) if it rounded onto a bound, as it does far into either tail.
fn interior_quantile(x: f64, p: f64) -> f64 {
    if p > 0.0 && p < 1.0 {
        strictly_inside(x, 0.0, 1.0)
    } else {
        x
    }
}

/// The log CDF (or survival function) outside (0, 1), or None inside.
fn unit_outside<const K: usize>(x: f64, upper: bool) -> Option<Jet<K>> {
    if x <= 0.0 {
        Some(outside(true, upper))
    } else if x >= 1.0 {
        Some(outside(false, upper))
    } else {
        None
    }
}

derived_distribution! {
    /// Kumaraswamy distribution with shapes `a` > 0 and `b` > 0, as PyMC's `Kumaraswamy(a, b)` (Stan has no built-in
    /// Kumaraswamy):
    ///
    /// ```text
    /// p(x) = a b x^{a−1} (1 − x^a)^{b−1},   P(X > x) = (1 − x^a)^b
    /// ```
    ///
    /// on (0, 1). Like the beta distribution but with a closed-form CDF and quantile function.
    Kumaraswamy, KumaraswamyGradient, inputs = 3,
    (
        /// The first shape a.
        a,
        /// The second shape b.
        b,
    ),
    log_prob = kumaraswamy_log_prob,
    log_cdf = kumaraswamy_log_cdf,
}

fn kumaraswamy_log_prob<const K: usize>(x: Jet<K>, [a, b]: [Jet<K>; 2]) -> Jet<K> {
    if !(a.v > 0.0 && b.v > 0.0) {
        return Jet::constant(f64::NAN);
    }
    if !(x.v > 0.0 && x.v < 1.0) {
        return Jet::constant(f64::NEG_INFINITY);
    }
    let ln_x = x.ln();
    // ln(1 − x^a) = ln(1 − e^{a ln x})
    a.ln() + b.ln() + (a - 1.0) * ln_x + (b - 1.0) * (a * ln_x).log1mexp()
}

fn kumaraswamy_log_cdf<const K: usize>(x: Jet<K>, [a, b]: [Jet<K>; 2], upper: bool) -> Jet<K> {
    if !(a.v > 0.0 && b.v > 0.0) {
        return Jet::constant(f64::NAN);
    }
    if let Some(limit) = unit_outside(x.v, upper) {
        return limit;
    }
    let log_ccdf = b * (a * x.ln()).log1mexp();
    if upper { log_ccdf } else { log_ccdf.log1mexp() }
}

impl Kumaraswamy<'_> {
    /// The element-wise inverse CDF: the x with P(X <= x) = p, for p in [0, 1]; strictly inside (0, 1) for p
    /// strictly inside it.
    pub fn inverse_cdf(&self, p: &DoubleTensor) -> Result<DoubleTensor> {
        map((p, &*self.a, &*self.b), |(p, a, b)| {
            // 1 − x^a = (1 − p)^{1/b}
            interior_quantile(((-((-p).ln_1p() / b).exp_m1()).ln() / a).exp(), p)
        })
    }

    /// The element-wise inverse survival function: the x with P(X > x) = q, for q in [0, 1]; strictly inside
    /// (0, 1) for q strictly inside it.
    pub fn inverse_ccdf(&self, q: &DoubleTensor) -> Result<DoubleTensor> {
        map((q, &*self.a, &*self.b), |(q, a, b)| {
            interior_quantile(((-(q.ln() / b).exp_m1()).ln() / a).exp(), q)
        })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes) by inverting the CDF. An error
    /// unless a and b are positive.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        check_all(
            &[&self.a, &self.b],
            |p| p[0] > 0.0 && p[1] > 0.0,
            || "a and b must be positive".into(),
        )?;
        let shape = sample_shape(shape, &[self.a.shape(), self.b.shape()])?;
        self.inverse_ccdf(&uniform_open(&shape, rng))
    }
}

derived_distribution! {
    /// Logit-normal distribution: the logistic sigmoid of N(μ, σ), as PyMC's `LogitNormal(μ, σ)` (Stan writes it as
    /// a transformed normal):
    ///
    /// ```text
    /// p(x) = 1/(σ√(2π) x(1 − x)) exp(−(logit x − μ)²/(2σ²)),   P(X <= x) = Φ((logit x − μ)/σ)
    /// ```
    ///
    /// on (0, 1), for σ > 0.
    LogitNormal, LogitNormalGradient, inputs = 3,
    (
        /// The mean μ of the logit.
        mu,
        /// The standard deviation σ of the logit.
        sigma,
    ),
    log_prob = logit_normal_log_prob,
    log_cdf = logit_normal_log_cdf,
}

fn logit_normal_log_prob<const K: usize>(x: Jet<K>, [mu, sigma]: [Jet<K>; 2]) -> Jet<K> {
    if !(sigma.v > 0.0) {
        return Jet::constant(f64::NAN);
    }
    if !(x.v > 0.0 && x.v < 1.0) {
        return Jet::constant(f64::NEG_INFINITY);
    }
    let (ln_x, ln_1mx) = (x.ln(), (-x).ln_1p());
    let z = (ln_x - ln_1mx - mu) / sigma;
    -sigma.ln() - LN_SQRT_2PI - 0.5 * z.square() - ln_x - ln_1mx
}

fn logit_normal_log_cdf<const K: usize>(
    x: Jet<K>,
    [mu, sigma]: [Jet<K>; 2],
    upper: bool,
) -> Jet<K> {
    if !(sigma.v > 0.0) {
        return Jet::constant(f64::NAN);
    }
    if let Some(limit) = unit_outside(x.v, upper) {
        return limit;
    }
    let z = (x.ln() - (-x).ln_1p() - mu) / sigma;
    if upper { (-z).log_ndtr() } else { z.log_ndtr() }
}

impl LogitNormal<'_> {
    /// The element-wise inverse CDF: the x with P(X <= x) = p, for p in [0, 1]; strictly inside (0, 1) for p
    /// strictly inside it.
    pub fn inverse_cdf(&self, p: &DoubleTensor) -> Result<DoubleTensor> {
        map((p, &*self.mu, &*self.sigma), |(p, mu, sigma)| {
            interior_quantile(sigmoid(mu + sigma * ndtri(p)), p)
        })
    }

    /// The element-wise inverse survival function: the x with P(X > x) = q, for q in [0, 1]; strictly inside
    /// (0, 1) for q strictly inside it.
    pub fn inverse_ccdf(&self, q: &DoubleTensor) -> Result<DoubleTensor> {
        map((q, &*self.mu, &*self.sigma), |(q, mu, sigma)| {
            interior_quantile(sigmoid(mu - sigma * ndtri(q)), q)
        })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes), as the sigmoid of a normal draw,
    /// strictly inside (0, 1) (a draw that rounded onto a bound, as one far into a tail does, is moved just inside
    /// it). An error unless σ is positive.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        require_positive("sigma", &self.sigma)?;
        let shape = sample_shape(shape, &[self.mu.shape(), self.sigma.shape()])?;
        let normal = rng.gaussian(&shape).mul(&self.sigma)?.add(&self.mu)?;
        Ok(normal.map(|z| strictly_inside(sigmoid(z), 0.0, 1.0)))
    }
}
