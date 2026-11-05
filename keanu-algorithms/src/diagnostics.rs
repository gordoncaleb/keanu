//! Convergence diagnostics for several MCMC chains: rank-normalised split R-hat, bulk and tail effective sample
//! size (ESS) and the Monte Carlo standard error (MCSE) of the mean.
//!
//! The estimators follow Vehtari, Gelman, Simpson, Carpenter and Bürkner (2021), "Rank-normalization, folding, and
//! localization: an improved R-hat for assessing convergence of MCMC", *Bayesian Analysis* 16(2), as implemented
//! in ArviZ (arviz-stats `_rhat_rank`, `_ess_bulk`, `_ess_tail` and `_mcse_mean`), and match it to rounding error:
//!
//! - **Split chains.** Each chain of n draws becomes two of ⌊n/2⌋: its first and last halves (an odd chain's
//!   middle draw is dropped). Splitting lets R-hat see a trend within a chain.
//! - **Rank normalisation.** The S pooled draws are ranked (ties get the average rank r) and replaced by
//!   z = Φ⁻¹((r − 3/8) / (S + 1/4)) (Blom's offsets), which makes the diagnostics work for heavy tails.
//! - **R-hat.** With W the mean within-chain variance and B/N the variance of the chain means over N draws per
//!   chain, var⁺ = (N − 1)/N · W + B/N and R-hat = √(var⁺ / W). The bulk R-hat is that of the rank-normalised split
//!   chains, the folded R-hat that of the rank-normalised |θ − median(θ)|, which detects chains with different
//!   scales; [`rhat`] reports the larger.
//! - **ESS.** The autocorrelations of all chains are pooled: ρ̂ₜ = 1 − (W − mean of the chain autocovariances at lag
//!   t) / var⁺. They are summed with Geyer's initial positive sequence (pairs of consecutive lags are added while
//!   their sum stays positive) made monotone (no pair may exceed the previous one), giving τ̂ = −1 + 2 Σ ρ̂ₜ, floored
//!   at 1 / log₁₀(S) so the ESS is at most S·log₁₀(S). The ESS is S / τ̂. The **bulk ESS** is that of the
//!   rank-normalised split chains; the **tail ESS** is the smaller ESS of the indicators I(θ ≤ q₀.₀₅) and
//!   I(θ ≤ q₀.₉₅) of the split chains.
//! - **MCSE of the mean.** sd(θ) / √ESS, with the ESS of the raw (not rank-normalised) split chains.
//!
//! Quantiles use linear interpolation between order statistics (NumPy's default, type 7 of Hyndman and Fan).
//!
//! # Thresholds
//!
//! Vehtari et al. recommend trusting the results only if R-hat < [`RHAT_THRESHOLD`] (1.01) and both the bulk and
//! tail ESS exceed [`MIN_ESS`] (400, that is 100 per chain for four chains).
//!
//! # Edge cases
//!
//! - R-hat needs at least two chains and every diagnostic needs at least four draws per chain; with fewer, or with
//!   any NaN or infinite draw, they are NaN (as in ArviZ, which checks only NaN).
//! - When every draw is the same (to within 10⁻¹⁵), R-hat is 1, both ESS are S and the MCSE is 0, so a fixed
//!   quantity does not count as unconverged. (ArviZ returns a NaN R-hat here.) When each chain is constant but the
//!   chains disagree, the within-chain variance is zero and R-hat is infinite.
//!
//! The functions take the draws of one scalar quantity as one slice per chain, which must all be the same length;
//! [`ChainSamples`](crate::ChainSamples) applies them to every element of sampled nodes.
//!
//! ```
//! use keanu_algorithms::diagnostics::{self, RHAT_THRESHOLD};
//!
//! // two chains that agree
//! let chains = vec![vec![0.1, -0.4, 1.2, 0.3, -0.8, 0.5], vec![-0.2, 0.9, -1.1, 0.4, 0.0, -0.6]];
//! let summary = diagnostics::summarize(&chains)?;
//! assert!(summary.r_hat < 1.1);
//! # Ok::<(), keanu_algorithms::AlgorithmError>(())
//! ```

use crate::error::{AlgorithmError, Result};
use crate::statistics::autocovariance;

/// Trust the samples only if R-hat is below this (Vehtari et al., 2021).
pub const RHAT_THRESHOLD: f64 = 1.01;

/// Trust the samples only if the bulk and tail effective sample sizes are above this (Vehtari et al., 2021,
/// recommend 100 per chain with four chains).
pub const MIN_ESS: f64 = 400.0;

/// The probabilities of the quantiles whose indicators give the tail ESS.
const TAIL_PROBABILITIES: (f64, f64) = (0.05, 0.95);

/// Draws closer together than this count as constant (NumPy's `finfo(float).resolution`, as ArviZ uses).
const CONSTANT_RESOLUTION: f64 = 1e-15;

/// The fewest draws per chain for which the diagnostics are computed (as in ArviZ).
const MIN_DRAWS: usize = 4;

/// Diagnostics and summary statistics of one scalar quantity over several chains.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ElementSummary {
    /// The mean of all draws.
    pub mean: f64,
    /// The sample standard deviation of all draws (n − 1 denominator).
    pub sd: f64,
    /// The 5% quantile of all draws.
    pub q5: f64,
    /// The median of all draws.
    pub q50: f64,
    /// The 95% quantile of all draws.
    pub q95: f64,
    /// The Monte Carlo standard error of the mean ([`mcse_mean`]).
    pub mcse_mean: f64,
    /// The bulk effective sample size ([`ess_bulk`]).
    pub ess_bulk: f64,
    /// The tail effective sample size ([`ess_tail`]).
    pub ess_tail: f64,
    /// The rank-normalised split R-hat ([`rhat`]).
    pub r_hat: f64,
}

impl ElementSummary {
    /// Whether R-hat is below [`RHAT_THRESHOLD`] and both effective sample sizes are above [`MIN_ESS`]. False if
    /// any of them is NaN.
    pub fn is_converged(&self) -> bool {
        self.r_hat < RHAT_THRESHOLD && self.ess_bulk > MIN_ESS && self.ess_tail > MIN_ESS
    }
}

/// The rank-normalised split R-hat: the larger of the bulk and folded R-hats (see the [module docs](self)).
///
/// Fails if there are no chains or they have different lengths.
pub fn rhat<S: AsRef<[f64]>>(chains: &[S]) -> Result<f64> {
    Ok(Draws::new(chains)?.rhat())
}

/// The bulk effective sample size: the ESS of the rank-normalised split chains.
///
/// Fails if there are no chains or they have different lengths.
pub fn ess_bulk<S: AsRef<[f64]>>(chains: &[S]) -> Result<f64> {
    Ok(Draws::new(chains)?.ess_bulk())
}

/// The tail effective sample size: the smaller ESS of the indicators of the draws being at most the 5% and at most
/// the 95% quantiles.
///
/// Fails if there are no chains or they have different lengths.
pub fn ess_tail<S: AsRef<[f64]>>(chains: &[S]) -> Result<f64> {
    Ok(Draws::new(chains)?.ess_tail())
}

/// The Monte Carlo standard error of the mean: the standard deviation of the draws over the square root of the
/// ESS of the split chains.
///
/// Fails if there are no chains or they have different lengths.
pub fn mcse_mean<S: AsRef<[f64]>>(chains: &[S]) -> Result<f64> {
    Ok(Draws::new(chains)?.mcse_mean())
}

/// The summary statistics and diagnostics of one scalar quantity.
///
/// Fails if there are no chains or they have different lengths.
pub fn summarize<S: AsRef<[f64]>>(chains: &[S]) -> Result<ElementSummary> {
    let draws = Draws::new(chains)?;
    let mut sorted = draws.data.clone();
    sorted.sort_by(f64::total_cmp);
    let (mean, variance) = mean_and_variance(&draws.data);
    Ok(ElementSummary {
        mean,
        sd: variance.sqrt(),
        q5: quantile(&sorted, 0.05),
        q50: quantile(&sorted, 0.5),
        q95: quantile(&sorted, 0.95),
        mcse_mean: draws.mcse_mean(),
        ess_bulk: draws.ess_bulk(),
        ess_tail: draws.ess_tail(),
        r_hat: draws.rhat(),
    })
}

/// Draws of one quantity: `chains` chains of `draws` each, chain by chain.
#[derive(Debug, Clone)]
struct Draws {
    chains: usize,
    draws: usize,
    data: Vec<f64>,
}

impl Draws {
    fn new<S: AsRef<[f64]>>(chains: &[S]) -> Result<Self> {
        let Some(first) = chains.first() else {
            return Err(AlgorithmError::InvalidArgument(
                "diagnostics need at least one chain".into(),
            ));
        };
        let draws = first.as_ref().len();
        if let Some(other) = chains.iter().find(|c| c.as_ref().len() != draws) {
            return Err(AlgorithmError::InvalidArgument(format!(
                "every chain must have the same number of draws, but there are chains of {draws} and {}",
                other.as_ref().len()
            )));
        }
        Ok(Self {
            chains: chains.len(),
            draws,
            data: chains
                .iter()
                .flat_map(|c| c.as_ref().iter().copied())
                .collect(),
        })
    }

    fn chain(&self, index: usize) -> &[f64] {
        &self.data[index * self.draws..(index + 1) * self.draws]
    }

    fn map(&self, f: impl Fn(f64) -> f64) -> Self {
        Self {
            data: self.data.iter().map(|&x| f(x)).collect(),
            ..*self
        }
    }

    /// Whether there are enough chains and draws, and every draw is finite.
    fn valid(&self, min_chains: usize) -> bool {
        self.chains >= min_chains
            && self.draws >= MIN_DRAWS
            && self.data.iter().all(|x| x.is_finite())
    }

    fn is_constant(&self) -> bool {
        let (lo, hi) = self
            .data
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| {
                (lo.min(x), hi.max(x))
            });
        hi - lo < CONSTANT_RESOLUTION
    }

    /// Each chain's first and last halves as separate chains: every first half, then every last half.
    fn split(&self) -> Self {
        let half = self.draws / 2;
        let data = (0..self.chains)
            .flat_map(|c| self.chain(c)[..half].iter().copied())
            .chain(
                (0..self.chains).flat_map(|c| self.chain(c)[self.draws - half..].iter().copied()),
            )
            .collect();
        Self {
            chains: 2 * self.chains,
            draws: half,
            data,
        }
    }

    /// The pooled draws replaced by the normal quantiles of their (average) ranks.
    fn z_scale(&self) -> Self {
        let size = self.data.len() as f64;
        let ranks = average_ranks(&self.data);
        Self {
            data: ranks
                .iter()
                .map(|r| inverse_normal_cdf((r - 3.0 / 8.0) / (size + 0.25)))
                .collect(),
            ..*self
        }
    }

    fn rhat(&self) -> f64 {
        if !self.valid(2) {
            return f64::NAN;
        }
        if self.is_constant() {
            return 1.0;
        }
        let split = self.split();
        let bulk = split.z_scale().basic_rhat();
        let mut sorted = split.data.clone();
        sorted.sort_by(f64::total_cmp);
        let median = quantile(&sorted, 0.5);
        let folded = split.map(|x| (x - median).abs()).z_scale().basic_rhat();
        bulk.max(folded)
    }

    fn ess_bulk(&self) -> f64 {
        if !self.valid(1) {
            return f64::NAN;
        }
        self.split().z_scale().basic_ess()
    }

    fn ess_tail(&self) -> f64 {
        if !self.valid(1) {
            return f64::NAN;
        }
        let mut sorted = self.data.clone();
        sorted.sort_by(f64::total_cmp);
        let quantile_ess = |p: f64| {
            let q = quantile(&sorted, p);
            self.map(|x| if x <= q { 1.0 } else { 0.0 })
                .split()
                .basic_ess()
        };
        let (low, high) = TAIL_PROBABILITIES;
        quantile_ess(low).min(quantile_ess(high))
    }

    fn mcse_mean(&self) -> f64 {
        if !self.valid(1) {
            return f64::NAN;
        }
        let (_, variance) = mean_and_variance(&self.data);
        variance.sqrt() / self.split().basic_ess().sqrt()
    }

    /// The chain means and the mean of the within-chain variances (n − 1 denominators).
    fn chain_means_and_within_variance(&self) -> (Vec<f64>, f64) {
        let (means, variances): (Vec<f64>, Vec<f64>) = (0..self.chains)
            .map(|c| mean_and_variance(self.chain(c)))
            .unzip();
        let within = variances.iter().sum::<f64>() / self.chains as f64;
        (means, within)
    }

    /// R-hat of these chains as they are: √(var⁺ / W).
    fn basic_rhat(&self) -> f64 {
        let n = self.draws as f64;
        let (means, within) = self.chain_means_and_within_variance();
        // B / n: the variance of the chain means
        let (_, between_over_n) = mean_and_variance(&means);
        if within == 0.0 {
            // every chain is constant: they agree only if they are at the same value
            return if between_over_n == 0.0 {
                1.0
            } else {
                f64::INFINITY
            };
        }
        let var_plus = (n - 1.0) / n * within + between_over_n;
        (var_plus / within).sqrt()
    }

    /// The ESS of these chains as they are, from the pooled autocorrelations with Geyer's initial monotone
    /// sequence. A port of arviz-stats' `_ess`.
    fn basic_ess(&self) -> f64 {
        let size = self.data.len() as f64;
        if self.is_constant() {
            return size;
        }
        let n_draw = self.draws;
        let n = n_draw as f64;
        let autocovariances: Vec<Vec<f64>> = (0..self.chains)
            .map(|c| autocovariance(self.chain(c)))
            .collect();
        let mean_autocovariance =
            |t: usize| autocovariances.iter().map(|a| a[t]).sum::<f64>() / self.chains as f64;
        let (means, _) = self.chain_means_and_within_variance();
        // W, from the 1/n autocovariance at lag 0
        let mean_var = mean_autocovariance(0) * n / (n - 1.0);
        let mut var_plus = mean_var * (n - 1.0) / n;
        if self.chains > 1 {
            var_plus += mean_and_variance(&means).1;
        }
        let rho = |t: usize| 1.0 - (mean_var - mean_autocovariance(t)) / var_plus;

        let mut rho_hat = vec![0.0; n_draw];
        let mut rho_even = 1.0;
        rho_hat[0] = rho_even;
        let mut rho_odd = rho(1);
        rho_hat[1] = rho_odd;

        // Geyer's initial positive sequence
        let mut t = 1;
        while t + 3 < n_draw && rho_even + rho_odd > 0.0 {
            rho_even = rho(t + 1);
            rho_odd = rho(t + 2);
            if rho_even + rho_odd >= 0.0 {
                rho_hat[t + 1] = rho_even;
                rho_hat[t + 2] = rho_odd;
            }
            t += 2;
        }
        // the last odd lag kept; -1 if the loop never ran
        let max_t = t as isize - 2;
        let last = (max_t + 1) as usize;
        if rho_even > 0.0 {
            rho_hat[last] = rho_even;
        }

        // Geyer's initial monotone sequence
        let mut t = 1;
        while t as isize <= max_t - 2 {
            if rho_hat[t + 1] + rho_hat[t + 2] > rho_hat[t - 1] + rho_hat[t] {
                rho_hat[t + 1] = (rho_hat[t - 1] + rho_hat[t]) / 2.0;
                rho_hat[t + 2] = rho_hat[t + 1];
            }
            t += 2;
        }

        let tau = -1.0 + 2.0 * rho_hat[..last].iter().sum::<f64>() + rho_hat[last];
        let tau = tau.max(1.0 / size.log10());
        if rho_hat.iter().any(|r| r.is_nan()) {
            return f64::NAN;
        }
        size / tau
    }
}

/// The mean and sample variance (n − 1 denominator). A series of identical values has exactly that mean and zero
/// variance, whatever the rounding of their sum.
fn mean_and_variance(series: &[f64]) -> (f64, f64) {
    let n = series.len() as f64;
    if let Some(&first) = series.first() {
        if series.iter().all(|&x| x == first) {
            return (first, if series.len() > 1 { 0.0 } else { f64::NAN });
        }
    }
    let mean = series.iter().sum::<f64>() / n;
    let variance = series.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0);
    (mean, variance)
}

/// The quantile with linear interpolation between the order statistics of `sorted` (NumPy's default). NaN for no
/// values.
fn quantile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let h = (sorted.len() - 1) as f64 * p;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let fraction = h - lo as f64;
    let (a, b) = (sorted[lo], sorted[hi]);
    // NumPy's form, exact at both ends of each interval
    if fraction >= 0.5 {
        b - (b - a) * (1.0 - fraction)
    } else {
        a + (b - a) * fraction
    }
}

/// The ranks of the values, 1 to n, with tied values given the average of their ranks.
fn average_ranks(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut ranks = vec![0.0; values.len()];
    let mut i = 0;
    while i < order.len() {
        let mut j = i + 1;
        while j < order.len() && values[order[j]] == values[order[i]] {
            j += 1;
        }
        // positions i..j (ranks i + 1..=j) are tied
        let rank = (i + 1 + j) as f64 / 2.0;
        for &k in &order[i..j] {
            ranks[k] = rank;
        }
        i = j;
    }
    ranks
}

/// The standard normal quantile function Φ⁻¹(p), by Wichura's algorithm AS 241 (PPND16), "The percentage points
/// of the normal distribution", *Applied Statistics* 37 (1988), which is accurate to about 1 part in 10¹⁶.
/// −∞ at 0, +∞ at 1 and NaN outside [0, 1].
///
/// TODO(U12): replace with the public `ndtri` that U12 adds to keanu-tensor.
fn inverse_normal_cdf(p: f64) -> f64 {
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180625 - q * q;
        return q * polynomial(r, &CENTRAL_NUMERATOR) / polynomial(r, &CENTRAL_DENOMINATOR);
    }
    let r = (-(if q < 0.0 { p } else { 1.0 - p }).ln()).sqrt();
    let value = if r <= 5.0 {
        let r = r - 1.6;
        polynomial(r, &INTERMEDIATE_NUMERATOR) / polynomial(r, &INTERMEDIATE_DENOMINATOR)
    } else {
        let r = r - 5.0;
        polynomial(r, &TAIL_NUMERATOR) / polynomial(r, &TAIL_DENOMINATOR)
    };
    if q < 0.0 { -value } else { value }
}

/// Σ coefficients[i] · xⁱ by Horner's rule (coefficients from the constant term up).
fn polynomial(x: f64, coefficients: &[f64; 8]) -> f64 {
    coefficients.iter().rev().fold(0.0, |sum, &c| sum * x + c)
}

// AS 241's coefficients as published, constant term first.
#[allow(clippy::excessive_precision)]
const CENTRAL_NUMERATOR: [f64; 8] = [
    3.387132872796366608,
    133.14166789178437745,
    1971.5909503065514427,
    13731.693765509461125,
    45921.953931549871457,
    67265.770927008700853,
    33430.575583588128105,
    2509.0809287301226727,
];
#[allow(clippy::excessive_precision)]
const CENTRAL_DENOMINATOR: [f64; 8] = [
    1.0,
    42.313330701600911252,
    687.1870074920579083,
    5394.1960214247511077,
    21213.794301586595867,
    39307.89580009271061,
    28729.085735721942674,
    5226.495278852545925,
];
#[allow(clippy::excessive_precision)]
const INTERMEDIATE_NUMERATOR: [f64; 8] = [
    1.42343711074968357734,
    4.6303378461565452959,
    5.7694972214606914055,
    3.64784832476320460504,
    1.27045825245236838258,
    0.24178072517745061177,
    0.0227238449892691845833,
    7.7454501427834140764e-4,
];
#[allow(clippy::excessive_precision)]
const INTERMEDIATE_DENOMINATOR: [f64; 8] = [
    1.0,
    2.05319162663775882187,
    1.6763848301838038494,
    0.68976733498510000455,
    0.14810397642748007459,
    0.0151986665636164571966,
    5.475938084995344946e-4,
    1.05075007164441684324e-9,
];
#[allow(clippy::excessive_precision)]
const TAIL_NUMERATOR: [f64; 8] = [
    6.6579046435011037772,
    5.4637849111641143699,
    1.7848265399172913358,
    0.29656057182850489123,
    0.026532189526576123093,
    0.0012426609473880784386,
    2.71155556874348757815e-5,
    2.01033439929228813265e-7,
];
#[allow(clippy::excessive_precision)]
const TAIL_DENOMINATOR: [f64; 8] = [
    1.0,
    0.59983220655588793769,
    0.13692988092273580531,
    0.0148753612908506148525,
    7.868691311456132591e-4,
    1.8463183175100546818e-5,
    1.4215117583164458887e-7,
    2.04426310338993978564e-15,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_normal_cdf_matches_reference_values() {
        // reference values from CPython's statistics.NormalDist().inv_cdf. The ArviZ fixture test
        // (tests/diagnostics.rs) also checks the rank normalisation against SciPy's independent Cephes ndtri.
        let cases = [
            (1e-300, -37.0470962993612),
            (1e-20, -9.262340089798405),
            (1e-10, -6.361340902404056),
            (0.001, -3.090232306167813),
            (0.02425, -1.9729610513118845),
            (0.1, -1.2815515655446008),
            (0.3, -0.5244005127080407),
            (0.5, 0.0),
            (0.7, 0.5244005127080407),
            (0.975, 1.9599639845400536),
            (0.999999, 4.753424308817089),
        ];
        for (p, expected) in cases {
            let z = inverse_normal_cdf(p);
            assert!(
                (z - expected).abs() <= 4.0 * f64::EPSILON * expected.abs().max(1.0),
                "{p}: {z} vs {expected}"
            );
        }
        assert_eq!(inverse_normal_cdf(0.0), f64::NEG_INFINITY);
        assert_eq!(inverse_normal_cdf(1.0), f64::INFINITY);
        assert!(inverse_normal_cdf(-0.1).is_nan() && inverse_normal_cdf(f64::NAN).is_nan());
        for p in [0.01, 0.2, 0.4, 0.45] {
            assert!((inverse_normal_cdf(p) + inverse_normal_cdf(1.0 - p)).abs() < 1e-15);
        }
    }

    #[test]
    fn ties_get_average_ranks() {
        assert_eq!(
            average_ranks(&[3.0, 1.0, 3.0, 2.0, 1.0, 3.0]),
            vec![5.0, 1.5, 5.0, 3.0, 1.5, 5.0]
        );
        assert!(average_ranks(&[]).is_empty());
    }

    #[test]
    fn split_drops_the_middle_of_an_odd_chain() {
        let draws = Draws::new(&[
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![6.0, 7.0, 8.0, 9.0, 10.0],
        ])
        .unwrap();
        let split = draws.split();
        assert_eq!((split.chains, split.draws), (4, 2));
        assert_eq!(split.data, vec![1.0, 2.0, 6.0, 7.0, 4.0, 5.0, 9.0, 10.0]);
    }

    #[test]
    fn quantiles_interpolate_linearly() {
        let sorted = [1.0, 2.0, 4.0, 8.0];
        assert_eq!(quantile(&sorted, 0.0), 1.0);
        assert_eq!(quantile(&sorted, 1.0), 8.0);
        assert_eq!(quantile(&sorted, 0.5), 3.0);
        // h = 3 · 0.05 = 0.15: 1 + 0.15 · (2 − 1)
        assert!((quantile(&sorted, 0.05) - 1.15).abs() < 1e-15);
        assert!(quantile(&[], 0.5).is_nan());
    }

    /// Two identical chains 1, 2, 3, 4 split into [1, 2], [1, 2], [3, 4], [3, 4]. The pooled ranks are 1.5, 3.5,
    /// 5.5 and 7.5 (each value twice) of S = 8, so z = Φ⁻¹((r − 3/8) / 8.25) gives a, b, −b, −a with
    /// a = Φ⁻¹(1.125 / 8.25) and b = Φ⁻¹(3.125 / 8.25). With n = 2 draws per split chain, W = (a − b)² / 2 and
    /// B / n = (a + b)² / 3 (the variance of the means ±(a + b) / 2 over four chains), so
    /// R-hat = √((W / 2 + B / n) / W). Folding about the median 2.5 gives chains that all have mean 0, a smaller
    /// R-hat of √(1/2), so the bulk R-hat is reported.
    #[test]
    fn hand_computed_rhat() {
        let a = inverse_normal_cdf(1.125 / 8.25);
        let b = inverse_normal_cdf(3.125 / 8.25);
        let w = (a - b).powi(2) / 2.0;
        let between_over_n = (a + b).powi(2) / 3.0;
        let expected = ((w / 2.0 + between_over_n) / w).sqrt();
        let r = rhat(&[vec![1.0, 2.0, 3.0, 4.0], vec![1.0, 2.0, 3.0, 4.0]]).unwrap();
        assert!((r - expected).abs() < 1e-14, "{r} vs {expected}");
        assert!(expected > 1.5); // about 1.62
        // |x − 2.5| of the split chains
        let folded = Draws::new(&[[1.5, 0.5], [1.5, 0.5], [0.5, 1.5], [0.5, 1.5]])
            .unwrap()
            .z_scale()
            .basic_rhat();
        assert!((folded - 0.5f64.sqrt()).abs() < 1e-14, "{folded}");
    }

    /// Independent draws whose lag-1 autocorrelation is very negative: τ̂ would go below its floor, so the ESS is
    /// S · log₁₀(S).
    #[test]
    fn ess_is_capped_at_s_log10_s() {
        let alternating: Vec<f64> = (0..100)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let draws = Draws::new(&[alternating]).unwrap();
        let ess = draws.basic_ess();
        assert!((ess - 100.0 * 2.0).abs() < 1e-9, "{ess}");
    }
}
