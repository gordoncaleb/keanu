//! Pareto-smoothed importance sampling (PSIS) and its k̂ diagnostic.
//!
//! Importance sampling estimates expectations under one distribution from draws of another by weighting each draw
//! by the ratio of their densities. When the ratios have a heavy right tail a few draws dominate and the estimate is
//! noisy or, worse, biased with a variance that is infinite. PSIS (Vehtari, Simpson, Gelman, Yao and Gabry, 2024,
//! "Pareto smoothed importance sampling", *JMLR* 25, arXiv:1507.02646) fits a generalized Pareto distribution to the
//! largest ratios and replaces them by the expected order statistics of the fit. The fitted shape k̂ says how
//! reliable the result is: below [`khat_threshold`] (at most 0.7) the smoothed estimate is reliable; above it, not.
//!
//! The algorithm follows the paper and the R package `loo` (2.8):
//!
//! 1. The S log ratios are shifted so that the largest is 0.
//! 2. The tail is the M = ⌈min(0.2·S, 3·√(S / r_eff))⌉ largest ratios ([`tail_length`]), where r_eff is the relative
//!    efficiency of the draws (1 for independent draws).
//! 3. The tail's excesses over the largest ratio not in the tail (the cutoff), on the ratio scale, are fitted with
//!    [`gpdfit`]: Zhang and Stephens' (2009) empirical Bayes estimate, with k̂ then shrunk towards 0.5 by a weakly
//!    informative prior worth 10 observations, k̂ ← (M·k̂ + 10·0.5) / (M + 10).
//! 4. If k̂ is finite, the tail ratios are replaced, in their order, by the cutoff plus the fitted distribution's
//!    quantiles at (j − ½) / M for j = 1, …, M.
//! 5. Every smoothed log ratio above the largest raw one (0 after the shift) is truncated to it, and the shift is
//!    undone.
//!
//! The edge cases also follow `loo`, with one exception: with fewer than five tail draws there is no fit and k̂ is
//! +∞ (the weights are the raw ones, truncated); a tail whose values are all the same, which `loo` also reports as
//! +∞, has k̂ = −∞ here, since its ratios are bounded and the raw weights need no smoothing.
//!
//! ```
//! use keanu_algorithms::psis::{khat_threshold, psis};
//!
//! // log ratios with a heavy tail: logs of generalized Pareto quantiles with shape 0.5
//! let log_ratios: Vec<f64> = (0..2000)
//!     .map(|i| {
//!         let u = (i as f64 + 0.5) / 2000.0;
//!         (((1.0 - u).powf(-0.5) - 1.0) / 0.5).ln()
//!     })
//!     .collect();
//! let result = psis(&log_ratios, 1.0)?;
//! assert!((result.pareto_k() - 0.5).abs() < 0.1);
//! assert!(result.pareto_k() < khat_threshold(2000));
//! let weights = result.weights();
//! assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
//! # Ok::<(), keanu_algorithms::AlgorithmError>(())
//! ```

use crate::error::{AlgorithmError, Result};
use keanu_tensor::special::log_sum_exp;

/// The weight of the prior on the shape k̂ of the generalized Pareto fit, in observations (as in `loo`).
const PRIOR_OBSERVATIONS: f64 = 10.0;
/// The shape the prior shrinks k̂ towards.
const PRIOR_SHAPE: f64 = 0.5;
/// The fewest tail draws for which the generalized Pareto distribution is fitted.
const MIN_TAIL_LENGTH: usize = 5;

/// The result of Pareto-smoothed importance sampling: smoothed log weights and the k̂ diagnostic.
#[derive(Debug, Clone, PartialEq)]
pub struct Psis {
    log_weights: Vec<f64>,
    pareto_k: f64,
    tail_length: usize,
}

impl Psis {
    /// The smoothed log weights, one per draw, in the draws' order. They are not normalised: as in `loo`, they are
    /// on the scale of the log ratios given, so draws outside the tail keep their log ratios.
    pub fn log_weights(&self) -> &[f64] {
        &self.log_weights
    }

    /// The smoothed log weights normalised so that the weights sum to 1.
    pub fn normalized_log_weights(&self) -> Vec<f64> {
        let total = log_sum_exp(self.log_weights.iter().copied());
        self.log_weights.iter().map(|w| w - total).collect()
    }

    /// The smoothed weights normalised to sum to 1.
    pub fn weights(&self) -> Vec<f64> {
        self.normalized_log_weights()
            .into_iter()
            .map(f64::exp)
            .collect()
    }

    /// The estimated shape k̂ of the generalized Pareto distribution fitted to the tail: the reliability of the
    /// importance sampling estimate. +∞ when there are too few draws to fit the tail, and −∞ when the tail ratios
    /// are all the same (see the [module documentation](self)).
    pub fn pareto_k(&self) -> f64 {
        self.pareto_k
    }

    /// The number of draws in the tail ([`tail_length`]).
    pub fn tail_length(&self) -> usize {
        self.tail_length
    }

    /// Whether k̂ is at most [`khat_threshold`] for this number of draws, so that the estimate is reliable.
    pub fn is_reliable(&self) -> bool {
        self.pareto_k <= khat_threshold(self.log_weights.len())
    }
}

/// The k̂ above which PSIS estimates from `draws` draws are unreliable: min(1 − 1 / log₁₀(S), 0.7) (Vehtari et al.,
/// 2024, §3.2.4). With few draws the threshold is lower, because the tail estimate is noisier; from 2200 draws on it
/// is 0.7. Fewer than ten draws give a threshold that is not positive (−∞ for one draw).
pub fn khat_threshold(draws: usize) -> f64 {
    (1.0 - 1.0 / (draws as f64).log10()).min(0.7)
}

/// The number of draws in the tail PSIS fits, ⌈min(0.2·S, 3·√(S / r_eff))⌉, for S draws with relative efficiency
/// `r_eff`.
pub fn tail_length(draws: usize, r_eff: f64) -> usize {
    let s = draws as f64;
    (0.2 * s).min(3.0 * (s / r_eff).sqrt()).ceil() as usize
}

/// Pareto-smoothed importance sampling of the log importance ratios `log_ratios` (log target density minus log
/// proposal density, up to a constant), one per draw, whose relative efficiency is `r_eff` (the effective sample
/// size over the number of draws; 1 for independent draws). See the [module documentation](self).
///
/// # Errors
///
/// [`AlgorithmError::InvalidArgument`] if there are no ratios, any is NaN or +∞, all are −∞, or `r_eff` is not
/// finite and positive.
pub fn psis(log_ratios: &[f64], r_eff: f64) -> Result<Psis> {
    if log_ratios.is_empty() {
        return Err(AlgorithmError::InvalidArgument(
            "PSIS needs at least one log ratio".into(),
        ));
    }
    if !(r_eff.is_finite() && r_eff > 0.0) {
        return Err(AlgorithmError::InvalidArgument(format!(
            "the relative efficiency must be finite and positive, not {r_eff}"
        )));
    }
    if log_ratios.iter().any(|r| r.is_nan() || *r == f64::INFINITY) {
        return Err(AlgorithmError::InvalidArgument(
            "the log ratios must not be NaN or +∞".into(),
        ));
    }
    let max = log_ratios.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max == f64::NEG_INFINITY {
        return Err(AlgorithmError::InvalidArgument(
            "every log ratio is −∞, so every weight is zero".into(),
        ));
    }
    let draws = log_ratios.len();
    let tail_len = tail_length(draws, r_eff);
    let mut log_weights: Vec<f64> = log_ratios.iter().map(|r| r - max).collect();
    let mut pareto_k = f64::INFINITY;
    if tail_len >= MIN_TAIL_LENGTH && tail_len < draws {
        let mut order: Vec<usize> = (0..draws).collect();
        order.sort_by(|&a, &b| log_weights[a].total_cmp(&log_weights[b]));
        let tail = &order[draws - tail_len..];
        let tail_values: Vec<f64> = tail.iter().map(|&i| log_weights[i]).collect();
        let (lowest, highest) = (tail_values[0], tail_values[tail_len - 1]);
        if (highest - lowest).abs() < f64::EPSILON / 100.0 {
            pareto_k = f64::NEG_INFINITY;
        } else {
            let cutoff = log_weights[order[draws - tail_len - 1]];
            let exp_cutoff = cutoff.exp();
            let excesses: Vec<f64> = tail_values.iter().map(|w| w.exp() - exp_cutoff).collect();
            let fit = gpdfit_sorted(&excesses);
            pareto_k = fit.k;
            if fit.k.is_finite() && fit.sigma.is_finite() && fit.sigma > 0.0 {
                let m = tail_len as f64;
                for (j, &i) in tail.iter().enumerate() {
                    let p = (j as f64 + 0.5) / m;
                    log_weights[i] = (fit.quantile(p) + exp_cutoff).ln();
                }
            }
        }
    }
    for w in &mut log_weights {
        // truncate at the largest raw weight, then undo the shift
        *w = w.min(0.0) + max;
    }
    Ok(Psis {
        log_weights,
        pareto_k,
        tail_length: tail_len,
    })
}

/// A generalized Pareto distribution with location 0, shape `k` and scale `sigma`: for x ≥ 0 (and x ≤ −σ / k when
/// k < 0), P(X > x) = (1 + k·x / σ)^(−1/k), or e^(−x / σ) when k = 0.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct GeneralizedPareto {
    /// The shape: positive for a heavy tail, whose moments of order 1 / k and above are infinite.
    pub k: f64,
    /// The scale.
    pub sigma: f64,
}

impl GeneralizedPareto {
    /// The distribution with shape `k` and scale `sigma`.
    pub fn new(k: f64, sigma: f64) -> Self {
        Self { k, sigma }
    }

    /// The quantile function: σ·((1 − p)^(−k) − 1) / k, or −σ·log(1 − p) when k = 0. NaN unless σ is positive.
    pub fn quantile(&self, p: f64) -> f64 {
        if self.sigma.is_nan() || self.sigma <= 0.0 {
            return f64::NAN;
        }
        if self.k == 0.0 {
            -self.sigma * (-p).ln_1p()
        } else {
            self.sigma * (-self.k * (-p).ln_1p()).exp_m1() / self.k
        }
    }
}

/// Fits a generalized Pareto distribution to positive values (in any order), as PSIS fits its tail: the empirical
/// Bayes estimate of Zhang and Stephens (2009), "A new and efficient estimation method for the generalized Pareto
/// distribution", *Technometrics* 51(3), §4, with the weakly informative prior of Vehtari et al. (2024) on the shape.
///
/// With n values x₍₁₎ ≤ … ≤ x₍ₙ₎, the estimate averages over a grid of m = 30 + ⌊√n⌋ values of θ = −k / σ,
/// θⱼ = 1 / x₍ₙ₎ + (1 − √(m / (j − ½))) / (3·x*) for j = 1, …, m, with x* the first quartile x₍⌊n/4 + ½⌋₎. Each θⱼ is
/// weighted by its profile likelihood exp(n·(log(−θⱼ / kⱼ) − kⱼ − 1)), where kⱼ = mean(log(1 − θⱼ·xᵢ)); the
/// weighted mean θ̂ gives k = mean(log(1 − θ̂·xᵢ)) and σ = −k / θ̂. Finally the shape is shrunk towards 0.5 as
/// k ← (n·k + 10·0.5) / (n + 10). A shape that is not a number (from degenerate values) is reported as +∞, as `loo`
/// does.
///
/// An empty input gives a NaN shape and scale.
pub fn gpdfit(values: &[f64]) -> GeneralizedPareto {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    gpdfit_sorted(&sorted)
}

/// [`gpdfit`] of values sorted in increasing order.
fn gpdfit_sorted(x: &[f64]) -> GeneralizedPareto {
    let n = x.len();
    if n == 0 {
        return GeneralizedPareto::new(f64::NAN, f64::NAN);
    }
    let nf = n as f64;
    let prior = 3.0;
    let m = 30 + nf.sqrt().floor() as usize;
    // the first quartile, x[⌊n/4 + 1/2⌋] counting from 1
    let quartile_index = ((nf / 4.0 + 0.5).floor() as usize).clamp(1, n) - 1;
    let x_star = x[quartile_index];
    let x_max = x[n - 1];
    let mean_log1p = |theta: f64| x.iter().map(|&xi| (-theta * xi).ln_1p()).sum::<f64>() / nf;
    let thetas: Vec<f64> = (1..=m)
        .map(|j| 1.0 / x_max + (1.0 - (m as f64 / (j as f64 - 0.5)).sqrt()) / prior / x_star)
        .collect();
    let profile: Vec<f64> = thetas
        .iter()
        .map(|&theta| {
            let k = mean_log1p(theta);
            let l = nf * ((-theta / k).ln() - k - 1.0);
            // a grid point whose profile likelihood is undefined carries no weight
            if l.is_nan() { f64::NEG_INFINITY } else { l }
        })
        .collect();
    let total = log_sum_exp(profile.iter().copied());
    let theta_hat: f64 = thetas
        .iter()
        .zip(&profile)
        .map(|(theta, l)| theta * (l - total).exp())
        .sum();
    let k = mean_log1p(theta_hat);
    let sigma = -k / theta_hat;
    let k = (nf * k + PRIOR_OBSERVATIONS * PRIOR_SHAPE) / (nf + PRIOR_OBSERVATIONS);
    GeneralizedPareto::new(if k.is_nan() { f64::INFINITY } else { k }, sigma)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thresholds_and_tail_lengths() {
        assert_eq!(khat_threshold(10_000), 0.7);
        assert!((khat_threshold(100) - 0.5).abs() < 1e-15);
        assert_eq!(tail_length(10_000, 1.0), 300);
        assert_eq!(tail_length(100, 1.0), 20);
        // 3·√(4000) = 189.7
        assert_eq!(tail_length(4000, 1.0), 190);
        // a lower relative efficiency lengthens the tail
        assert_eq!(tail_length(4000, 0.25), 380);
    }

    #[test]
    fn quantiles_invert_the_survival_function() {
        let d = GeneralizedPareto::new(0.4, 2.0);
        let x = d.quantile(0.9);
        let survival = (1.0 + 0.4 * x / 2.0).powf(-1.0 / 0.4);
        assert!((survival - 0.1).abs() < 1e-14);
        let exponential = GeneralizedPareto::new(0.0, 2.0);
        assert!((exponential.quantile(0.5) - 2.0 * 2f64.ln()).abs() < 1e-14);
        assert!(GeneralizedPareto::new(0.4, 0.0).quantile(0.5).is_nan());
    }
}
