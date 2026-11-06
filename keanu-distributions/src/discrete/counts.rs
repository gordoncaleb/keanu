//! The hypergeometric and Skellam distributions.

use super::{sample_poisson, select_category};
use crate::error::{Result, invalid_parameter};
use crate::jet::{ElementDerivatives, Jet};
use crate::param::{IntoParam, Param};
use crate::rng::KeanuRng;
use crate::util::{map, map_n, sample_each};
use keanu_tensor::special::ln_gamma;
use keanu_tensor::{DoubleTensor, IntTensor};

/// The hypergeometric distribution, as PyMC's `HyperGeometric(N, k, n)`: the number of successes x in `draws` (n)
/// draws without replacement from a population of `population` (N) items of which `successes` (K) are successes,
///
/// ```text
/// P(x) = C(K, x) C(N − K, n − x) / C(N, n),  max(0, n + K − N) <= x <= min(K, n)
/// ```
///
/// with mean nK/N. (Stan's `hypergeometric(n | N, a, b)` counts the successes n in N draws from a successes and b
/// failures, so its (N, a, b) is (n, K, N − K) here.) Every parameter is an integer and they broadcast with the
/// value; parameters outside 0 <= K <= N and 0 <= n <= N give a NaN log probability and a value outside the range
/// above −∞. Draws invert the CDF over the range, O(min(K, n)) each.
#[derive(Debug, Clone, PartialEq)]
pub struct HyperGeometric<'a> {
    /// The population size N >= 0.
    pub population: Param<'a, i64>,
    /// The number of successes K in the population, 0 <= K <= N.
    pub successes: Param<'a, i64>,
    /// The number of draws n, 0 <= n <= N.
    pub draws: Param<'a, i64>,
}

/// ln C(n, k) for 0 <= k <= n.
fn log_choose(n: f64, k: f64) -> f64 {
    ln_gamma(n + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0)
}

impl<'a> HyperGeometric<'a> {
    /// Creates the distribution from its parameters.
    pub fn new(
        population: impl IntoParam<'a, i64>,
        successes: impl IntoParam<'a, i64>,
        draws: impl IntoParam<'a, i64>,
    ) -> Self {
        Self {
            population: population.into_param(),
            successes: successes.into_param(),
            draws: draws.into_param(),
        }
    }

    /// Whether the parameters are valid: 0 <= K <= N and 0 <= n <= N.
    fn valid(population: i64, successes: i64, draws: i64) -> bool {
        (0..=population).contains(&successes) && (0..=population).contains(&draws)
    }

    /// The values a hypergeometric count can take, [max(0, n + K − N), min(K, n)] (an empty range for invalid
    /// parameters).
    #[allow(clippy::reversed_empty_ranges)] // empty on purpose
    pub fn range(population: i64, successes: i64, draws: i64) -> std::ops::RangeInclusive<i64> {
        if Self::valid(population, successes, draws) {
            (draws + successes - population).max(0)..=successes.min(draws)
        } else {
            1..=0
        }
    }

    /// The log probability of `x` successes given the parameters.
    fn log_pmf(x: i64, population: i64, successes: i64, draws: i64) -> f64 {
        if !Self::valid(population, successes, draws) {
            return f64::NAN;
        }
        if !Self::range(population, successes, draws).contains(&x) {
            return f64::NEG_INFINITY;
        }
        let (n, k, d, x) = (population as f64, successes as f64, draws as f64, x as f64);
        log_choose(k, x) + log_choose(n - k, d - x) - log_choose(n, d)
    }

    /// The element-wise log probability of x, broadcasting x with the parameters.
    pub fn log_prob(&self, x: &IntTensor) -> Result<DoubleTensor> {
        map(
            (x, &*self.population, &*self.successes, &*self.draws),
            |(x, n, k, d)| Self::log_pmf(x, n, k, d),
        )
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes). An error unless
    /// 0 <= K <= N and 0 <= n <= N.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<IntTensor> {
        let parameters = [
            self.population.to_double(),
            self.successes.to_double(),
            self.draws.to_double(),
        ];
        let refs: Vec<&DoubleTensor> = parameters.iter().collect();
        sample_each(shape, &refs, |p| {
            let (n, k, d) = (p[0] as i64, p[1] as i64, p[2] as i64);
            if !Self::valid(n, k, d) {
                return invalid_parameter(format!(
                    "a hypergeometric needs 0 <= successes <= population and 0 <= draws <= population, got \
                     {n}, {k}, {d}"
                ));
            }
            let range = Self::range(n, k, d);
            let logs: Vec<f64> = range.clone().map(|x| Self::log_pmf(x, n, k, d)).collect();
            let max = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let total: f64 = logs.iter().map(|l| (l - max).exp()).sum();
            let probabilities = logs.iter().map(|l| (l - max).exp() / total);
            Ok(range.start() + select_category(rng.next_f64(), probabilities) as i64)
        })
    }
}

/// The Skellam distribution, the difference of independent Poisson(μ₁) and Poisson(μ₂) counts, over all integers:
///
/// ```text
/// P(k) = e^(−(μ₁ + μ₂)) (μ₁/μ₂)^(k/2) I_|k|(2√(μ₁μ₂))
/// ```
///
/// with I_ν the modified Bessel function of the first kind, taken in logs
/// ([`keanu_tensor::special::log_bessel_i`]), so it neither overflows nor underflows; the mean is μ₁ − μ₂ and the
/// variance μ₁ + μ₂. Both means must be positive (else the log probability is NaN). Draws are differences of
/// Poisson draws.
#[derive(Debug, Clone, PartialEq)]
pub struct Skellam<'a> {
    /// The mean μ₁ > 0 of the added count.
    pub mu1: Param<'a>,
    /// The mean μ₂ > 0 of the subtracted count.
    pub mu2: Param<'a>,
}

/// Derivatives of [`Skellam::log_prob`].
#[derive(Debug, Clone, PartialEq)]
pub struct SkellamGradient {
    /// With respect to `mu1`.
    pub mu1: DoubleTensor,
    /// With respect to `mu2`.
    pub mu2: DoubleTensor,
}

fn skellam_log_prob<const N: usize>(k: f64, [mu1, mu2]: [Jet<N>; 2]) -> Jet<N> {
    if !(mu1.v > 0.0 && mu2.v > 0.0 && mu1.v.is_finite() && mu2.v.is_finite()) {
        return Jet::constant(f64::NAN);
    }
    let z = (mu1 * mu2).sqrt() * 2.0;
    -(mu1 + mu2) + (mu1.ln() - mu2.ln()) * (k / 2.0) + z.log_bessel_i(k.abs())
}

impl<'a> Skellam<'a> {
    /// Creates the distribution from its parameters.
    pub fn new(mu1: impl IntoParam<'a, f64>, mu2: impl IntoParam<'a, f64>) -> Self {
        Self {
            mu1: mu1.into_param(),
            mu2: mu2.into_param(),
        }
    }

    /// The log probability of one count with its gradient and Hessian with respect to the count (index 0, whose
    /// entries are zero: the count is discrete) and the means (indices 1 and 2), computed exactly by forward-mode
    /// differentiation to second order.
    pub fn log_prob_derivatives(k: f64, parameters: [f64; 2]) -> ElementDerivatives<3> {
        crate::jet::derivatives(k, parameters, |x, p| skellam_log_prob(x.v, p))
    }

    /// The element-wise log probability of k, broadcasting k with the parameters.
    pub fn log_prob(&self, k: &IntTensor) -> Result<DoubleTensor> {
        map((k, &*self.mu1, &*self.mu2), |(k, mu1, mu2)| {
            crate::jet::value(k as f64, [mu1, mu2], |x, p| skellam_log_prob(x.v, p))
        })
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to each parameter.
    pub fn d_log_prob(&self, k: &IntTensor) -> Result<SkellamGradient> {
        let [mu1, mu2] = map_n((k, &*self.mu1, &*self.mu2), |(k, mu1, mu2)| {
            let d = Self::log_prob_derivatives(k as f64, [mu1, mu2]);
            [d.gradient[1], d.gradient[2]]
        })?;
        Ok(SkellamGradient { mu1, mu2 })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes). An error unless both means are
    /// positive and finite.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<IntTensor> {
        sample_each(shape, &[&self.mu1, &self.mu2], |p| {
            Ok(sample_poisson(p[0], rng)? - sample_poisson(p[1], rng)?)
        })
    }
}
