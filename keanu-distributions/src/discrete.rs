//! Discrete distributions. Values are integers (`IntTensor`) or booleans (`BoolTensor`).

use crate::continuous::sample_gamma;
use crate::error::{Result, invalid_parameter};
use crate::param::{IntoParam, Param};
use crate::rng::KeanuRng;
use crate::util::{
    Operand, broadcast_all, digamma, ln_gamma, map, map_n, operand, require, require_ordered,
    require_probability, safe_ln_times, sample_each, sample_shape, sigmoid, softplus,
};
use keanu_tensor::special::{log_beta, log1mexp};
use keanu_tensor::{BoolTensor, DoubleTensor, IntTensor, Shape};

/// Bernoulli distribution: true with probability `prob_true`.
#[derive(Debug, Clone, PartialEq)]
pub struct Bernoulli<'a> {
    /// The probability of true.
    pub prob_true: Param<'a>,
}

/// Derivative of [`Bernoulli::log_prob`] with respect to `prob_true`.
#[derive(Debug, Clone, PartialEq)]
pub struct BernoulliGradient {
    /// With respect to `prob_true`.
    pub prob_true: DoubleTensor,
}

impl<'a> Bernoulli<'a> {
    /// Creates the distribution from its parameters.
    pub fn new(prob_true: impl IntoParam<'a, f64>) -> Self {
        Self {
            prob_true: prob_true.into_param(),
        }
    }

    /// ln p where x is true and ln(1 - p) where false, with p clamped to [0, 1].
    pub fn log_prob(&self, x: &BoolTensor) -> Result<DoubleTensor> {
        map((x, &*self.prob_true), |(x, p)| {
            let p = p.clamp(0.0, 1.0);
            (if x { p } else { 1.0 - p }).ln()
        })
    }

    /// 1/p where true and 1/(p - 1) where false; 0 where p is outside (0, 1].
    pub fn d_log_prob(&self, x: &BoolTensor) -> Result<BernoulliGradient> {
        let gradient = map((x, &*self.prob_true), |(x, p)| {
            if p > 1.0 || p <= 0.0 {
                0.0
            } else if x {
                1.0 / p
            } else {
                1.0 / (p - 1.0)
            }
        })?;
        Ok(BernoulliGradient {
            prob_true: gradient,
        })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes). An error unless the probability
    /// is in [0, 1].
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<BoolTensor> {
        require_probability("prob_true", &self.prob_true)?;
        let shape = sample_shape(shape, &[self.prob_true.shape()])?;
        Ok(rng.uniform(&shape).lt(&self.prob_true)?)
    }
}

/// Binomial distribution: the number of successes in `n` trials with success probability `p`.
#[derive(Debug, Clone, PartialEq)]
pub struct Binomial<'a> {
    /// The probability parameter.
    pub p: Param<'a>,
    /// The number of trials.
    pub n: Param<'a, i64>,
}

/// Derivative of [`Binomial::log_prob`] with respect to `p`.
#[derive(Debug, Clone, PartialEq)]
pub struct BinomialGradient {
    /// With respect to `p`.
    pub p: DoubleTensor,
}

impl<'a> Binomial<'a> {
    /// Creates the distribution from its parameters.
    pub fn new(p: impl IntoParam<'a, f64>, n: impl IntoParam<'a, i64>) -> Self {
        Self {
            p: p.into_param(),
            n: n.into_param(),
        }
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support).
    pub fn log_prob(&self, k: &IntTensor) -> Result<DoubleTensor> {
        map((k, &*self.p, &*self.n), |(k, p, n)| {
            if k < 0 || k > n {
                return f64::NEG_INFINITY;
            }
            let (k, n) = (k as f64, n as f64);
            let log_binomial_coefficient =
                ln_gamma(n + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0);
            log_binomial_coefficient + safe_ln_times(p, k) + safe_ln_times(1.0 - p, n - k)
        })
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    pub fn d_log_prob(&self, k: &IntTensor) -> Result<BinomialGradient> {
        Ok(BinomialGradient {
            p: map((k, &*self.p, &*self.n), |(k, p, n)| {
                // 0 / p taken as 0, as for the multinomial, so p = 0 with k = 0 (or p = 1 with k = n) is finite
                let ratio = |count: i64, q: f64| if count == 0 { 0.0 } else { count as f64 / q };
                ratio(k, p) - ratio(n - k, 1.0 - p)
            })?,
        })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes). An error unless p is in [0, 1]
    /// and n is not negative.
    ///
    /// Each draw counts successes of n uniform draws for n up to 50; above that it is by inversion when
    /// n min(p, 1 - p) < 10 and by Hörmann's BTRS transformed rejection otherwise, so the expected cost is O(1)
    /// for large n.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<IntTensor> {
        require_probability("p", &self.p)?;
        require(&self.n, |n| n >= 0, || "n must not be negative".into())?;
        let n = self.n.to_double();
        sample_each(shape, &[&self.p, &n], |params| {
            Ok(sample_binomial(params[1] as i64, params[0], rng))
        })
    }
}

/// Up to this many trials, binomial draws count successes of uniform draws one by one.
const BINOMIAL_COUNTING_MAX_N: i64 = 50;
/// From this mean (of the less likely outcome), binomial draws use BTRS.
const BTRS_MIN_MEAN: f64 = 10.0;

/// A Binomial(n, p) draw, for n >= 0 and p in [0, 1].
pub(crate) fn sample_binomial(n: i64, p: f64, rng: &mut KeanuRng) -> i64 {
    if n <= BINOMIAL_COUNTING_MAX_N {
        return (0..n).filter(|_| rng.next_f64() < p).count() as i64;
    }
    // draw the count of the less likely outcome
    let (q, flipped) = if p > 0.5 { (1.0 - p, true) } else { (p, false) };
    let k = if q == 0.0 {
        0
    } else if n as f64 * q < BTRS_MIN_MEAN {
        binomial_inversion(n, q, rng)
    } else {
        binomial_btrs(n, q, rng)
    };
    if flipped { n - k } else { k }
}

/// Inversion by sequential search from 0, for p <= 0.5 and a small mean np: O(np) expected.
fn binomial_inversion(n: i64, p: f64, rng: &mut KeanuRng) -> i64 {
    let q = 1.0 - p;
    let s = p / q;
    let a = (n as f64 + 1.0) * s;
    let p0 = q.powf(n as f64);
    'draw: loop {
        let mut u = rng.next_f64();
        let mut k = 0;
        let mut pk = p0;
        while u > pk {
            u -= pk;
            k += 1;
            if k > n {
                // rounding left u above the total probability: draw again
                continue 'draw;
            }
            // P(k) = P(k - 1) (n - k + 1) p / (k q)
            pk *= a / k as f64 - s;
        }
        return k;
    }
}

/// ln k!
fn ln_factorial(k: f64) -> f64 {
    ln_gamma(k + 1.0)
}

/// Hörmann's BTRS (transformed rejection with squeeze), for p <= 0.5 and np >= 10: W. Hörmann, "The generation
/// of binomial random variates", Journal of Statistical Computation and Simulation 46 (1993).
fn binomial_btrs(n: i64, p: f64, rng: &mut KeanuRng) -> i64 {
    let n_f = n as f64;
    let q = 1.0 - p;
    let spq = (n_f * p * q).sqrt();
    let b = 1.15 + 2.53 * spq;
    let a = -0.0873 + 0.0248 * b + 0.01 * p;
    let c = n_f * p + 0.5;
    let v_r = 0.92 - 4.2 / b;
    let alpha = (2.83 + 5.1 / b) * spq;
    let ln_p_over_q = (p / q).ln();
    let m = ((n_f + 1.0) * p).floor();
    let h = ln_factorial(m) + ln_factorial(n_f - m);
    loop {
        let u = rng.next_f64() - 0.5;
        let v = rng.next_f64();
        let us = 0.5 - u.abs();
        let k = ((2.0 * a / us + b) * u + c).floor();
        // `contains` also rejects the NaN of us = 0
        if !(0.0..=n_f).contains(&k) {
            continue;
        }
        if us >= 0.07 && v <= v_r {
            return k as i64;
        }
        let v = (v * alpha / (a / (us * us) + b)).ln();
        if v <= h - ln_factorial(k) - ln_factorial(n_f - k) + (k - m) * ln_p_over_q {
            return k as i64;
        }
    }
}

/// Geometric distribution: the number of trials up to and including the first success (k >= 1).
#[derive(Debug, Clone, PartialEq)]
pub struct Geometric<'a> {
    /// The probability parameter.
    pub p: Param<'a>,
}

/// Derivative of [`Geometric::log_prob`] with respect to `p`.
#[derive(Debug, Clone, PartialEq)]
pub struct GeometricGradient {
    /// With respect to `p`.
    pub p: DoubleTensor,
}

impl<'a> Geometric<'a> {
    /// Creates the distribution from its parameters.
    pub fn new(p: impl IntoParam<'a, f64>) -> Self {
        Self { p: p.into_param() }
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support,
    /// and everywhere unless 0 < p < 1).
    pub fn log_prob(&self, k: &IntTensor) -> Result<DoubleTensor> {
        map((k, &*self.p), |(k, p)| {
            // `!(..)` also rejects a NaN p
            if !(p > 0.0 && p < 1.0) || k < 1 {
                f64::NEG_INFINITY
            } else {
                safe_ln_times(1.0 - p, k as f64 - 1.0) + p.ln()
            }
        })
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    pub fn d_log_prob(&self, k: &IntTensor) -> Result<GeometricGradient> {
        Ok(GeometricGradient {
            p: map((k, &*self.p), |(k, p)| {
                if k >= 1 {
                    1.0 / p - (k as f64 - 1.0) / (1.0 - p)
                } else {
                    0.0
                }
            })?,
        })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes). An error unless 0 < p < 1 (as
    /// for [`Self::log_prob`]).
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<IntTensor> {
        require(
            &self.p,
            |p| p > 0.0 && p < 1.0,
            || "p must be in (0, 1)".into(),
        )?;
        let shape = sample_shape(shape, &[self.p.shape()])?;
        let u = rng.uniform(&shape);
        let samples = map((&u, &*self.p), |(u, p)| (u.ln() / (1.0 - p).ln()).floor())?;
        Ok(&samples.to_int() + 1)
    }
}

/// Poisson distribution with mean `mu`.
#[derive(Debug, Clone, PartialEq)]
pub struct Poisson<'a> {
    /// The location (mean).
    pub mu: Param<'a>,
}

/// Derivative of [`Poisson::log_prob`] with respect to `mu`.
#[derive(Debug, Clone, PartialEq)]
pub struct PoissonGradient {
    /// With respect to `mu`.
    pub mu: DoubleTensor,
}

impl<'a> Poisson<'a> {
    /// Creates the distribution from its parameters.
    pub fn new(mu: impl IntoParam<'a, f64>) -> Self {
        Self {
            mu: mu.into_param(),
        }
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support).
    pub fn log_prob(&self, k: &IntTensor) -> Result<DoubleTensor> {
        map((k, &*self.mu), |(k, mu)| {
            if k < 0 {
                f64::NEG_INFINITY
            } else {
                safe_ln_times(mu, k as f64) - mu - ln_gamma(k as f64 + 1.0)
            }
        })
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    pub fn d_log_prob(&self, k: &IntTensor) -> Result<PoissonGradient> {
        Ok(PoissonGradient {
            mu: map((k, &*self.mu), |(k, mu)| k as f64 / mu - 1.0)?,
        })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes). An error unless mu is positive.
    ///
    /// Each draw is by Knuth's multiplication method for mu < 10 and by Hörmann's PTRS transformed rejection
    /// above, so the expected cost is O(1) for large mu.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<IntTensor> {
        sample_each(shape, &[&self.mu], |p| sample_poisson(p[0], rng))
    }
}

/// From this mean, Poisson draws use PTRS.
const PTRS_MIN_MU: f64 = 10.0;

fn sample_poisson(mu: f64, rng: &mut KeanuRng) -> Result<i64> {
    // `!(..)` also rejects NaN
    if !(mu > 0.0 && mu.is_finite()) {
        return invalid_parameter(format!("mu must be positive and finite, got {mu}"));
    }
    if mu >= PTRS_MIN_MU {
        return Ok(poisson_ptrs(mu, rng));
    }
    // Knuth's multiplication method: the number of uniform draws whose product with e^mu stays above 1
    let mut p = mu.exp();
    let mut k = 0;
    loop {
        p *= rng.next_f64_non_zero();
        if p <= 1.0 {
            return Ok(k);
        }
        k += 1;
    }
}

/// Hörmann's PTRS (transformed rejection with squeeze), for mu >= 10: W. Hörmann, "The transformed rejection
/// method for generating Poisson random variables", Insurance: Mathematics and Economics 12 (1993).
fn poisson_ptrs(mu: f64, rng: &mut KeanuRng) -> i64 {
    let ln_mu = mu.ln();
    let b = 0.931 + 2.53 * mu.sqrt();
    let a = -0.059 + 0.02483 * b;
    let ln_inv_alpha = (1.1239 + 1.1328 / (b - 3.4)).ln();
    let v_r = 0.9277 - 3.6224 / (b - 2.0);
    loop {
        let u = rng.next_f64() - 0.5;
        let v = rng.next_f64();
        let us = 0.5 - u.abs();
        let k = ((2.0 * a / us + b) * u + mu + 0.43).floor();
        if us >= 0.07 && v <= v_r {
            return k as i64;
        }
        // the NaN of us = 0 is rejected too
        if k < 0.0 || k.is_nan() || (us < 0.013 && v > us) {
            continue;
        }
        if v.ln() + ln_inv_alpha - (a / (us * us) + b).ln() <= -mu + k * ln_mu - ln_gamma(k + 1.0) {
            return k as i64;
        }
    }
}

/// Uniform distribution over the integers in [x_min, x_max).
#[derive(Debug, Clone, PartialEq)]
pub struct UniformInt<'a> {
    /// The lower bound.
    pub x_min: Param<'a, i64>,
    /// The upper bound.
    pub x_max: Param<'a, i64>,
}

impl<'a> UniformInt<'a> {
    /// Creates the distribution from its parameters.
    pub fn new(x_min: impl IntoParam<'a, i64>, x_max: impl IntoParam<'a, i64>) -> Self {
        Self {
            x_min: x_min.into_param(),
            x_max: x_max.into_param(),
        }
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support).
    pub fn log_prob(&self, x: &IntTensor) -> Result<DoubleTensor> {
        map((x, &*self.x_min, &*self.x_max), |(x, min, max)| {
            if x >= max || x < min {
                f64::NEG_INFINITY
            } else {
                -((max - min) as f64).ln()
            }
        })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes). An error unless
    /// x_max > x_min.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<IntTensor> {
        require_ordered(&self.x_min, &self.x_max, "x_max must be greater than x_min")?;
        let shape = sample_shape(shape, &[self.x_min.shape(), self.x_max.shape()])?;
        let u = rng.uniform(&shape);
        let (min, max) = (self.x_min.to_double(), self.x_max.to_double());
        // floor rather than truncate so negative ranges are sampled uniformly
        Ok(map((&u, &min, &max), |(u, min, max)| {
            ((max - min) * u + min).floor()
        })?
        .to_int())
    }
}

/// Categorical distribution over category indices 0..k, with (unnormalised) probabilities for each category.
/// Each probability tensor gives the probability of that category element-wise, so a batch of categorical
/// variables can have different probabilities.
#[derive(Debug, Clone, PartialEq)]
pub struct Categorical<'a> {
    /// The probability of each category.
    pub probabilities: Vec<Param<'a>>,
}

/// The error for a negative (or NaN) categorical probability.
fn negative_probability<T>() -> Result<T> {
    invalid_parameter("categorical probabilities must not be negative")
}

/// The error for categorical probabilities that sum to zero.
fn zero_total<T>() -> Result<T> {
    invalid_parameter("the probabilities sum to zero")
}

impl<'a> Categorical<'a> {
    /// Creates the distribution from the probability of each category (owned or borrowed tensors). An error if
    /// there are no categories.
    pub fn new<P: IntoParam<'a, f64>>(probabilities: impl IntoIterator<Item = P>) -> Result<Self> {
        let probabilities: Vec<_> = probabilities
            .into_iter()
            .map(IntoParam::into_param)
            .collect();
        if probabilities.is_empty() {
            return invalid_parameter("a categorical distribution needs at least one category");
        }
        Ok(Self { probabilities })
    }

    /// The probabilities broadcast to `shape`.
    fn broadcast(&self, shape: &[usize]) -> Result<Vec<Operand<'_, f64>>> {
        self.probabilities
            .iter()
            .map(|p| operand(p, shape))
            .collect()
    }

    /// The sum of the probabilities at each index, checking they are not negative and sum to more than zero.
    fn total(probabilities: &[Operand<'_, f64>], i: usize) -> Result<f64> {
        let mut total = 0.0;
        for p in probabilities {
            let p = p.at(i);
            if p < 0.0 || p.is_nan() {
                return negative_probability();
            }
            total += p;
        }
        if total <= 0.0 {
            return zero_total();
        }
        Ok(total)
    }

    /// ln(p_x / sum(p)) for each category index x; -infinity for an index that is not a category. An error if a
    /// probability is negative or they sum to zero.
    pub fn log_prob(&self, x: &IntTensor) -> Result<DoubleTensor> {
        let mut shapes = vec![x.shape()];
        shapes.extend(self.probabilities.iter().map(|p| p.shape()));
        let shape = broadcast_all(&shapes)?;
        let probabilities = self.broadcast(&shape)?;
        let x = operand(x, &shape)?;
        let len: usize = shape.iter().product();
        let mut log_prob = Vec::with_capacity(len);
        for i in 0..len {
            let total = Self::total(&probabilities, i)?;
            log_prob.push(match usize::try_from(x.at(i)) {
                Ok(category) if category < probabilities.len() => {
                    (probabilities[category].at(i) / total).ln()
                }
                _ => f64::NEG_INFINITY,
            });
        }
        Ok(DoubleTensor::new(&shape, log_prob)?)
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes). An error if a probability is
    /// negative or they sum to zero.
    ///
    /// Each element is the first category whose cumulative normalised probability exceeds a uniform draw. If
    /// rounding leaves the cumulative sum at or below the draw, it is the last category with non-zero probability.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<IntTensor> {
        let shapes: Vec<&[usize]> = self.probabilities.iter().map(|p| p.shape()).collect();
        let shape = sample_shape(shape, &shapes)?;
        let probabilities = self.broadcast(&shape)?;
        let len: usize = shape.iter().product();
        let totals = (0..len)
            .map(|i| Self::total(&probabilities, i))
            .collect::<Result<Vec<_>>>()?;
        let u = rng.uniform(&shape);
        let samples = u
            .as_slice()
            .iter()
            .zip(&totals)
            .enumerate()
            .map(|(i, (&u, &total))| {
                select_category(u, probabilities.iter().map(|p| p.at(i) / total)) as i64
            })
            .collect();
        Ok(IntTensor::new(&shape, samples)?)
    }
}

/// The first category whose cumulative probability exceeds `u`; if rounding leaves the cumulative sum at or
/// below `u`, the last category with non-zero probability (the last category if none is positive).
fn select_category(u: f64, probabilities: impl Iterator<Item = f64>) -> usize {
    let mut cumulative = 0.0;
    let mut last_possible = None;
    let mut count = 0;
    for (category, p) in probabilities.enumerate() {
        count = category + 1;
        cumulative += p;
        if cumulative > u {
            return category;
        }
        if p > 0.0 {
            last_possible = Some(category);
        }
    }
    last_possible.unwrap_or(count.saturating_sub(1))
}

/// Negative binomial distribution in the mean–dispersion form (as PyMC's `NegativeBinomial(mu, alpha)`): a Poisson
/// whose rate is Gamma distributed with mean `mu` and shape `alpha`, so the mean is mu and the variance is
/// mu + mu²/alpha. It approaches Poisson(mu) as alpha grows.
///
/// P(k) = Γ(k + α) / (Γ(α) k!) (α / (α + μ))^α (μ / (α + μ))^k for k >= 0.
#[derive(Debug, Clone, PartialEq)]
pub struct NegativeBinomial<'a> {
    /// The mean μ.
    pub mu: Param<'a>,
    /// The dispersion (shape) α: smaller is more over-dispersed.
    pub alpha: Param<'a>,
}

/// Derivatives of [`NegativeBinomial::log_prob`] (and of its log CDF and survival function).
#[derive(Debug, Clone, PartialEq)]
pub struct NegativeBinomialGradient {
    /// With respect to `mu`.
    pub mu: DoubleTensor,
    /// With respect to `alpha`.
    pub alpha: DoubleTensor,
}

/// ln Γ(k + α) - ln Γ(α) - ln k!, computed as -ln k - ln B(α, k) for k >= 1 so that it stays accurate for large α.
fn negative_binomial_log_coefficient(k: f64, alpha: f64) -> f64 {
    if k == 0.0 {
        0.0
    } else {
        -k.ln() - log_beta(alpha, k)
    }
}

impl<'a> NegativeBinomial<'a> {
    /// Creates the distribution from its parameters.
    pub fn new(mu: impl IntoParam<'a, f64>, alpha: impl IntoParam<'a, f64>) -> Self {
        Self {
            mu: mu.into_param(),
            alpha: alpha.into_param(),
        }
    }

    /// The element-wise log probability of k, broadcasting k with the parameters (-infinity for k < 0).
    pub fn log_prob(&self, k: &IntTensor) -> Result<DoubleTensor> {
        map((k, &*self.mu, &*self.alpha), |(k, mu, alpha)| {
            if k < 0 {
                return f64::NEG_INFINITY;
            }
            let k = k as f64;
            // α ln(α / (α + μ)) = -α ln(1 + μ/α), accurate for large α
            negative_binomial_log_coefficient(k, alpha) - alpha * (mu / alpha).ln_1p()
                + safe_ln_times(mu / (alpha + mu), k)
        })
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to each parameter.
    ///
    /// At mu = 0 they are the limits as mu → 0: for k = 0 (probability 1) -1 and 0; for k > 0 (probability 0)
    /// +infinity with respect to mu (the log probability is ≈ k ln mu) and a finite value with respect to alpha.
    pub fn d_log_prob(&self, k: &IntTensor) -> Result<NegativeBinomialGradient> {
        let [mu, alpha] = map_n((k, &*self.mu, &*self.alpha), |(k, mu, alpha)| {
            if k < 0 {
                return [0.0; 2];
            }
            let k = k as f64;
            let total = alpha + mu;
            // the k ln μ term, which contributes nothing for k = 0 (also at μ = 0)
            let d_ln_mu = if k == 0.0 { 0.0 } else { k / mu };
            [
                d_ln_mu - (k + alpha) / total,
                digamma(k + alpha) - digamma(alpha) - (mu / alpha).ln_1p() + (mu - k) / total,
            ]
        })?;
        Ok(NegativeBinomialGradient { mu, alpha })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes) as a Poisson draw whose rate is a
    /// Gamma(shape α, scale μ/α) draw. An error unless mu and alpha are positive.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<IntTensor> {
        sample_each(shape, &[&self.mu, &self.alpha], |p| {
            let (mu, alpha) = (p[0], p[1]);
            // `!(..)` also rejects NaN
            if !(mu > 0.0 && alpha > 0.0 && mu.is_finite()) {
                return invalid_parameter(format!(
                    "mu and alpha must be positive, got mu {mu} and alpha {alpha}"
                ));
            }
            let rate = sample_gamma(mu / alpha, alpha, rng)?;
            if rate <= 0.0 {
                return Ok(0);
            }
            sample_poisson(rate, rng)
        })
    }
}

/// Ordered logistic distribution (Stan's `ordered_logistic`) over categories 0..K: a latent η is compared with K - 1
/// increasing cutpoints c, giving P(0) = 1 - logit⁻¹(η - c₁), P(k) = logit⁻¹(η - cₖ) - logit⁻¹(η - cₖ₊₁) and
/// P(K - 1) = logit⁻¹(η - c_(K-1)).
///
/// The cutpoints are a tensor whose last axis is the K - 1 cutpoints; its leading axes and η's shape broadcast with
/// the value's shape. Probabilities are computed in log space, with logit⁻¹(a) - logit⁻¹(b) =
/// logit⁻¹(a) logit⁻¹(-b) (1 - e^(b - a)), so no difference of nearly equal probabilities is taken.
#[derive(Debug, Clone, PartialEq)]
pub struct OrderedLogistic<'a> {
    /// The location η.
    pub eta: Param<'a>,
    /// The cutpoints, strictly increasing along the last axis.
    pub cutpoints: Param<'a>,
}

/// Derivatives of [`OrderedLogistic::log_prob`].
#[derive(Debug, Clone, PartialEq)]
pub struct OrderedLogisticGradient {
    /// With respect to `eta`, shaped like the log probability.
    pub eta: DoubleTensor,
    /// With respect to `cutpoints`, shaped like the log probability with the cutpoints' last axis appended.
    pub cutpoints: DoubleTensor,
}

/// Whether a set of cutpoints is strictly increasing (and has no NaN).
fn increasing(cutpoints: &[f64]) -> bool {
    cutpoints.windows(2).all(|w| w[0] < w[1]) && cutpoints.iter().all(|c| !c.is_nan())
}

/// ln logit⁻¹(t) = -softplus(-t).
fn log_sigmoid(t: f64) -> f64 {
    -softplus(-t)
}

/// The log probability of category `k` and its derivatives with respect to η and each cutpoint (written into
/// `d_cutpoints`, which is zeroed first).
fn ordered_logistic(k: i64, eta: f64, cutpoints: &[f64], d_cutpoints: &mut [f64]) -> (f64, f64) {
    d_cutpoints.iter_mut().for_each(|d| *d = 0.0);
    let last = cutpoints.len();
    let Ok(k) = usize::try_from(k) else {
        return (f64::NEG_INFINITY, 0.0);
    };
    if k > last {
        return (f64::NEG_INFINITY, 0.0);
    }
    if k == 0 {
        let a = eta - cutpoints[0];
        d_cutpoints[0] = sigmoid(a);
        return (log_sigmoid(-a), -sigmoid(a));
    }
    if k == last {
        let a = eta - cutpoints[last - 1];
        d_cutpoints[last - 1] = -sigmoid(-a);
        return (log_sigmoid(a), sigmoid(-a));
    }
    // between cutpoints k - 1 and k (0-based): a > b
    let (a, b) = (eta - cutpoints[k - 1], eta - cutpoints[k]);
    let log_prob = log_sigmoid(a) + log_sigmoid(-b) + log1mexp(b - a);
    let gap = 1.0 / (a - b).exp_m1();
    let d_a = sigmoid(-a) + gap;
    let d_b = -sigmoid(b) - gap;
    d_cutpoints[k - 1] = -d_a;
    d_cutpoints[k] = -d_b;
    (log_prob, d_a + d_b)
}

impl<'a> OrderedLogistic<'a> {
    /// Creates the distribution from its parameters.
    pub fn new(eta: impl IntoParam<'a, f64>, cutpoints: impl IntoParam<'a, f64>) -> Self {
        Self {
            eta: eta.into_param(),
            cutpoints: cutpoints.into_param(),
        }
    }

    /// The batch shape (the broadcast of the value's shape, η's shape and the cutpoints' leading axes) and the
    /// number of cutpoints, checking the cutpoints are a non-empty last axis.
    fn layout(&self, value: &[usize]) -> Result<(Shape, usize)> {
        let Some((&count, leading)) = self.cutpoints.shape().split_last() else {
            return invalid_parameter("the cutpoints must have at least one axis");
        };
        if count == 0 {
            return invalid_parameter("there must be at least one cutpoint");
        }
        Ok((broadcast_all(&[value, self.eta.shape(), leading])?, count))
    }

    /// An error unless every set of cutpoints is strictly increasing.
    fn require_increasing(&self, count: usize) -> Result<()> {
        if !self
            .cutpoints
            .as_slice()
            .chunks_exact(count)
            .all(increasing)
        {
            return invalid_parameter("the cutpoints must be strictly increasing");
        }
        Ok(())
    }

    /// The cutpoints broadcast to `batch` with their last axis appended.
    fn cutpoints_for(&self, batch: &[usize], count: usize) -> Result<Vec<f64>> {
        let mut shape = batch.to_vec();
        shape.push(count);
        Ok(self.cutpoints.broadcast_to(&shape)?.into_vec())
    }

    /// The log probability of each category index k, broadcasting k with η and the cutpoints' leading axes
    /// (-infinity for an index that is not a category). Where the cutpoints are not strictly increasing the
    /// parameters are outside their valid region and the log probability is -infinity (with a zero gradient), so
    /// that a sampler proposing such cutpoints rejects the proposal; an error if the cutpoints have no last axis
    /// or it is empty.
    pub fn log_prob(&self, k: &IntTensor) -> Result<DoubleTensor> {
        Ok(self.log_prob_and_gradient(k)?.0)
    }

    /// The derivatives of [`Self::log_prob`] with respect to each parameter.
    pub fn d_log_prob(&self, k: &IntTensor) -> Result<OrderedLogisticGradient> {
        Ok(self.log_prob_and_gradient(k)?.1)
    }

    fn log_prob_and_gradient(
        &self,
        k: &IntTensor,
    ) -> Result<(DoubleTensor, OrderedLogisticGradient)> {
        let (batch, count) = self.layout(k.shape())?;
        let cutpoints = self.cutpoints_for(&batch, count)?;
        let k = operand(k, &batch)?;
        let eta = operand(&self.eta, &batch)?;
        let len: usize = batch.iter().product();
        let mut log_prob = Vec::with_capacity(len);
        let mut d_eta = Vec::with_capacity(len);
        let mut d_cutpoints = vec![0.0; len * count];
        for i in 0..len {
            let range = i * count..(i + 1) * count;
            if !increasing(&cutpoints[range.clone()]) {
                log_prob.push(f64::NEG_INFINITY);
                d_eta.push(0.0);
                continue;
            }
            let (lp, de) = ordered_logistic(
                k.at(i),
                eta.at(i),
                &cutpoints[range.clone()],
                &mut d_cutpoints[range],
            );
            log_prob.push(lp);
            d_eta.push(de);
        }
        let mut cut_shape = batch.to_vec();
        cut_shape.push(count);
        Ok((
            DoubleTensor::new(&batch, log_prob)?,
            OrderedLogisticGradient {
                eta: DoubleTensor::new(&batch, d_eta)?,
                cutpoints: DoubleTensor::new(&cut_shape, d_cutpoints)?,
            },
        ))
    }

    /// The probability of each category, shaped like the batch with a last axis of the K categories.
    pub fn probabilities(&self) -> Result<DoubleTensor> {
        let (batch, count) = self.layout(&[])?;
        self.require_increasing(count)?;
        let len: usize = batch.iter().product();
        let mut out = Vec::with_capacity(len * (count + 1));
        let cutpoints = self.cutpoints_for(&batch, count)?;
        let eta = operand(&self.eta, &batch)?;
        let mut scratch = vec![0.0; count];
        for i in 0..len {
            for k in 0..=count {
                let (lp, _) = ordered_logistic(
                    k as i64,
                    eta.at(i),
                    &cutpoints[i * count..(i + 1) * count],
                    &mut scratch,
                );
                out.push(lp.exp());
            }
        }
        let mut shape = batch.to_vec();
        shape.push(count + 1);
        Ok(DoubleTensor::new(&shape, out)?)
    }

    /// Draws a sample of the given shape (broadcast with η's shape and the cutpoints' leading axes). An error
    /// unless the cutpoints are strictly increasing.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<IntTensor> {
        let (batch, count) = self.layout(shape)?;
        self.require_increasing(count)?;
        let cutpoints = self.cutpoints_for(&batch, count)?;
        let eta = operand(&self.eta, &batch)?;
        let len: usize = batch.iter().product();
        let mut scratch = vec![0.0; count];
        let samples = (0..len)
            .map(|i| {
                let cut = &cutpoints[i * count..(i + 1) * count];
                let probabilities = (0..=count).map(|k| {
                    ordered_logistic(k as i64, eta.at(i), cut, &mut scratch)
                        .0
                        .exp()
                });
                let probabilities: Vec<f64> = probabilities.collect();
                select_category(rng.next_f64(), probabilities.into_iter()) as i64
            })
            .collect();
        Ok(IntTensor::new(&batch, samples)?)
    }
}

#[cfg(test)]
mod tests {
    use super::{sample_binomial, select_category};
    use crate::rng::KeanuRng;

    #[test]
    fn rounding_shortfall_selects_the_last_possible_category() {
        // cumulative probabilities that fall short of u (as rounding can leave them) must not select a
        // category with zero probability
        assert_eq!(select_category(0.75, [0.5, 0.0].into_iter()), 0);
        assert_eq!(select_category(0.99, [0.2, 0.7, 0.0, 0.0].into_iter()), 1);
        assert_eq!(select_category(0.3, [0.2, 0.7, 0.0].into_iter()), 1);
        assert_eq!(select_category(0.1, [0.0, 0.2, 0.7].into_iter()), 1);
    }

    #[test]
    fn binomial_draws_stay_in_range_for_every_method() {
        let mut rng = KeanuRng::seed_from_u64(3);
        for (n, p) in [
            (0, 0.5),
            (60, 0.0),
            (60, 1.0),
            (60, 0.01),
            (60, 0.99),
            (1000, 0.5),
            (1000, 0.995),
            (1_000_000_000, 1e-9),
        ] {
            for _ in 0..200 {
                let k = sample_binomial(n, p, &mut rng);
                assert!((0..=n).contains(&k), "Binomial({n}, {p}) drew {k}");
            }
        }
        assert_eq!(sample_binomial(60, 0.0, &mut rng), 0);
        assert_eq!(sample_binomial(60, 1.0, &mut rng), 60);
    }
}
