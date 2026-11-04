//! Discrete distributions. Values are integers (`IntTensor`) or booleans (`BoolTensor`).

use crate::error::{Result, invalid_parameter};
use crate::rng::KeanuRng;
use crate::util::{ln_gamma, map2, map3, safe_ln_times, sample_each, sample_shape};
use keanu_tensor::{BoolTensor, DoubleTensor, IntTensor};

/// Bernoulli distribution: true with probability `prob_true`.
#[derive(Debug, Clone, PartialEq)]
pub struct Bernoulli {
    /// The probability of true.
    pub prob_true: DoubleTensor,
}

/// Derivative of [`Bernoulli::log_prob`] with respect to `prob_true`.
#[derive(Debug, Clone, PartialEq)]
pub struct BernoulliGradient {
    /// With respect to `prob_true`.
    pub prob_true: DoubleTensor,
}

impl Bernoulli {
    /// Creates the distribution from its parameters.
    pub fn new(prob_true: DoubleTensor) -> Self {
        Self { prob_true }
    }

    /// ln p where x is true and ln(1 - p) where false, with p clamped to [0, 1].
    pub fn log_prob(&self, x: &BoolTensor) -> Result<DoubleTensor> {
        map2(x, &self.prob_true, |x, p| {
            let p = p.clamp(0.0, 1.0);
            (if x { p } else { 1.0 - p }).ln()
        })
    }

    /// 1/p where true and 1/(p - 1) where false; 0 where p is outside (0, 1].
    pub fn d_log_prob(&self, x: &BoolTensor) -> Result<BernoulliGradient> {
        let gradient = map2(x, &self.prob_true, |x, p| {
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

    /// Draws a sample of the given shape (broadcast with the parameters' shapes).
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<BoolTensor> {
        let shape = sample_shape(shape, &[self.prob_true.shape()])?;
        Ok(rng.uniform(&shape).lt(&self.prob_true)?)
    }
}

/// Binomial distribution: the number of successes in `n` trials with success probability `p`.
#[derive(Debug, Clone, PartialEq)]
pub struct Binomial {
    /// The probability parameter.
    pub p: DoubleTensor,
    /// The number of trials.
    pub n: IntTensor,
}

/// Derivative of [`Binomial::log_prob`] with respect to `p`.
#[derive(Debug, Clone, PartialEq)]
pub struct BinomialGradient {
    /// With respect to `p`.
    pub p: DoubleTensor,
}

impl Binomial {
    /// Creates the distribution from its parameters.
    pub fn new(p: DoubleTensor, n: IntTensor) -> Self {
        Self { p, n }
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support).
    pub fn log_prob(&self, k: &IntTensor) -> Result<DoubleTensor> {
        map3(k, &self.p, &self.n, |k, p, n| {
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
            p: map3(k, &self.p, &self.n, |k, p, n| {
                // 0 / p taken as 0, as for the multinomial, so p = 0 with k = 0 (or p = 1 with k = n) is finite
                let ratio = |count: i64, q: f64| if count == 0 { 0.0 } else { count as f64 / q };
                ratio(k, p) - ratio(n - k, 1.0 - p)
            })?,
        })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes).
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<IntTensor> {
        let n = self.n.to_double();
        sample_each(shape, &[&self.p, &n], |params| {
            let (p, n) = (params[0], params[1] as i64);
            Ok((0..n).filter(|_| rng.next_f64() < p).count() as i64)
        })
    }
}

/// Geometric distribution: the number of trials up to and including the first success (k >= 1).
#[derive(Debug, Clone, PartialEq)]
pub struct Geometric {
    /// The probability parameter.
    pub p: DoubleTensor,
}

/// Derivative of [`Geometric::log_prob`] with respect to `p`.
#[derive(Debug, Clone, PartialEq)]
pub struct GeometricGradient {
    /// With respect to `p`.
    pub p: DoubleTensor,
}

impl Geometric {
    /// Creates the distribution from its parameters.
    pub fn new(p: DoubleTensor) -> Self {
        Self { p }
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support).
    pub fn log_prob(&self, k: &IntTensor) -> Result<DoubleTensor> {
        map2(k, &self.p, |k, p| {
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
            p: map2(k, &self.p, |k, p| {
                if k >= 1 {
                    1.0 / p - (k as f64 - 1.0) / (1.0 - p)
                } else {
                    0.0
                }
            })?,
        })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes).
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<IntTensor> {
        let shape = sample_shape(shape, &[self.p.shape()])?;
        let u = rng.uniform(&shape);
        let samples = map2(&u, &self.p, |u, p| (u.ln() / (1.0 - p).ln()).floor())?;
        Ok(&samples.to_int() + 1)
    }
}

/// Poisson distribution with mean `mu`.
#[derive(Debug, Clone, PartialEq)]
pub struct Poisson {
    /// The location (mean).
    pub mu: DoubleTensor,
}

/// Derivative of [`Poisson::log_prob`] with respect to `mu`.
#[derive(Debug, Clone, PartialEq)]
pub struct PoissonGradient {
    /// With respect to `mu`.
    pub mu: DoubleTensor,
}

impl Poisson {
    /// Creates the distribution from its parameters.
    pub fn new(mu: DoubleTensor) -> Self {
        Self { mu }
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support).
    pub fn log_prob(&self, k: &IntTensor) -> Result<DoubleTensor> {
        map2(k, &self.mu, |k, mu| {
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
            mu: map2(k, &self.mu, |k, mu| k as f64 / mu - 1.0)?,
        })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes).
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<IntTensor> {
        sample_each(shape, &[&self.mu], |p| sample_poisson(p[0], rng))
    }
}

/// Knuth's multiplication method, introducing mu in steps of 500 to avoid underflow for large mu.
fn sample_poisson(mu: f64, rng: &mut KeanuRng) -> Result<i64> {
    if mu <= 0.0 || mu.is_nan() {
        return invalid_parameter(format!("mu must be positive, got {mu}"));
    }
    const STEP: f64 = 500.0;
    let mut mu_left = mu;
    let mut k = 0i64;
    let mut p = 1.0;
    loop {
        k += 1;
        p *= rng.next_f64_non_zero();
        while p < 1.0 && mu_left > 0.0 {
            if mu_left > STEP {
                p *= STEP.exp();
                mu_left -= STEP;
            } else {
                p *= mu_left.exp();
                mu_left = 0.0;
            }
        }
        if p <= 1.0 {
            return Ok(k - 1);
        }
    }
}

/// Uniform distribution over the integers in [x_min, x_max).
#[derive(Debug, Clone, PartialEq)]
pub struct UniformInt {
    /// The lower bound.
    pub x_min: IntTensor,
    /// The upper bound.
    pub x_max: IntTensor,
}

impl UniformInt {
    /// Creates the distribution from its parameters.
    pub fn new(x_min: IntTensor, x_max: IntTensor) -> Self {
        Self { x_min, x_max }
    }

    /// The element-wise log probability of x, broadcasting x with the parameters (-infinity outside the support).
    pub fn log_prob(&self, x: &IntTensor) -> Result<DoubleTensor> {
        map3(x, &self.x_min, &self.x_max, |x, min, max| {
            if x >= max || x < min {
                f64::NEG_INFINITY
            } else {
                -((max - min) as f64).ln()
            }
        })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes).
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<IntTensor> {
        let shape = sample_shape(shape, &[self.x_min.shape(), self.x_max.shape()])?;
        let u = rng.uniform(&shape);
        let (min, max) = (self.x_min.to_double(), self.x_max.to_double());
        // floor rather than truncate so negative ranges are sampled uniformly
        Ok(map3(&u, &min, &max, |u, min, max| {
            ((max - min) * u + min).floor()
        })?
        .to_int())
    }
}

/// Categorical distribution over category indices 0..k, with (unnormalised) probabilities for each category.
/// Each probability tensor gives the probability of that category element-wise, so a batch of categorical
/// variables can have different probabilities.
#[derive(Debug, Clone, PartialEq)]
pub struct Categorical {
    /// The probability of each category.
    pub probabilities: Vec<DoubleTensor>,
}

impl Categorical {
    /// Creates the distribution from its parameters.
    pub fn new(probabilities: Vec<DoubleTensor>) -> Result<Self> {
        if probabilities.is_empty() {
            return invalid_parameter("a categorical distribution needs at least one category");
        }
        Ok(Self { probabilities })
    }

    fn total(&self, shape: &[usize]) -> Result<DoubleTensor> {
        let mut total = DoubleTensor::zeros(shape);
        for p in &self.probabilities {
            total = total.add(p)?;
        }
        if total.as_slice().iter().any(|&t| t <= 0.0) {
            return invalid_parameter("the probabilities sum to zero");
        }
        Ok(total)
    }

    /// ln(p_x / sum(p)) for each category index x; -infinity for an index that is not a category.
    pub fn log_prob(&self, x: &IntTensor) -> Result<DoubleTensor> {
        let total = self.total(x.shape())?;
        let mut log_prob = DoubleTensor::full(x.shape(), f64::NEG_INFINITY);
        for (category, p) in self.probabilities.iter().enumerate() {
            let category_log_prob = p.div(&total)?.ln();
            log_prob = map3(&log_prob, x, &category_log_prob, |lp, x, c| {
                if x == category as i64 { c } else { lp }
            })?;
        }
        Ok(log_prob)
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes).
    ///
    /// Each element is the first category whose cumulative normalised probability exceeds a uniform draw. If
    /// rounding leaves the cumulative sum at or below the draw, it is the last category with non-zero probability.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<IntTensor> {
        let shapes: Vec<&[usize]> = self.probabilities.iter().map(|p| p.shape()).collect();
        let shape = sample_shape(shape, &shapes)?;
        let total = self.total(&shape)?;
        let u = rng.uniform(&shape);
        let normalised = self
            .probabilities
            .iter()
            .map(|p| Ok(p.div(&total)?.broadcast_to(&shape)?))
            .collect::<Result<Vec<_>>>()?;
        let samples = u
            .as_slice()
            .iter()
            .enumerate()
            .map(|(i, &u)| select_category(u, normalised.iter().map(|p| p.as_slice()[i])) as i64)
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

#[cfg(test)]
mod tests {
    use super::select_category;

    #[test]
    fn rounding_shortfall_selects_the_last_possible_category() {
        // cumulative probabilities that fall short of u (as rounding can leave them) must not select a
        // category with zero probability
        assert_eq!(select_category(0.75, [0.5, 0.0].into_iter()), 0);
        assert_eq!(select_category(0.99, [0.2, 0.7, 0.0, 0.0].into_iter()), 1);
        assert_eq!(select_category(0.3, [0.2, 0.7, 0.0].into_iter()), 1);
        assert_eq!(select_category(0.1, [0.0, 0.2, 0.7].into_iter()), 1);
    }
}
