//! Model comparison by predictive accuracy: leave-one-out cross-validation with Pareto-smoothed importance sampling
//! (PSIS-LOO), the widely applicable information criterion (WAIC), comparison tables, and stacking and pseudo-BMA+
//! weights.
//!
//! Both estimate the expected log pointwise predictive density (elpd) for new data, Σᵢ log p(ỹᵢ | y), from the
//! [pointwise log likelihood](crate::loglik) of the posterior draws; higher is better. They follow Vehtari, Gelman
//! and Gabry (2017), "Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC",
//! *Statistics and Computing* 27, and the R package `loo` (2.8):
//!
//! - **PSIS-LOO** ([`loo`]). For each point i, the posterior without it is reached by importance sampling the full
//!   posterior's draws with ratios 1 / p(yᵢ | θₛ), smoothed by [PSIS](crate::psis) with the point's relative
//!   efficiency (`loo`'s `relative_eff`: the effective sample size of p(yᵢ | θₛ) over the chains, divided by the
//!   number of draws). Then elpdᵢ = log Σₛ wₛ p(yᵢ | θₛ) with normalised weights wₛ, p_looᵢ = lpdᵢ − elpdᵢ where
//!   lpdᵢ = log mean p(yᵢ | θₛ), and looicᵢ = −2·elpdᵢ. Each point's k̂ says whether its estimate is reliable (at
//!   most [`khat_threshold`], 0.7 from 2200 draws): a point above it is too
//!   influential for importance sampling, and [`Loo::warnings`] names it.
//! - **WAIC** ([`waic`]). elpdᵢ = lpdᵢ − p_waicᵢ with p_waicᵢ the sample variance of log p(yᵢ | θₛ) over the draws.
//!   A point with p_waicᵢ above 0.4 makes WAIC unreliable, which [`Waic::warnings`] reports; prefer PSIS-LOO.
//!
//! Totals sum the points; the standard error of a total is √(n · var(pointwise)) for n points (with the n − 1
//! sample variance, as `loo`). The Monte Carlo standard error of the LOO elpd ([`Loo::mcse_elpd`]) follows `loo`'s
//! `mcse_elpd`: for each point, the variance of the importance sampling estimate of p(yᵢ | y₋ᵢ), Σₛ wₛ²·(p(yᵢ | θₛ) −
//! E)² / r_eff, is carried to the log scale through 1000 normal quantiles (where `loo` uses 1000 random normal
//! draws), and the variances are summed over the points.
//!
//! [`compare`] ranks models by elpd, as `loo_compare` and ArviZ's `compare` do: elpd_diff is each model's elpd
//! minus the best model's (so 0 for the best and negative for the others, as in `loo`), with standard error
//! √(n · var(elpdᵢ − elpdᵢ,best)), which accounts for the correlation between models evaluated on the same points.
//! A difference of a few standard errors or more is meaningful. The table also holds [`stacking_weights`] (Yao,
//! Vehtari, Simpson and Gelman, 2018, "Using stacking to average Bayesian predictive distributions", *Bayesian
//! Analysis* 13(3)): the weights on the simplex that maximise the log score Σᵢ log Σₖ wₖ exp(elpdₖᵢ) of the LOO
//! predictive densities, found by L-BFGS on a softmax parameterisation. [`pseudo_bma_weights`] gives the
//! alternative pseudo-BMA+ weights of the same paper, softmax weights of n·Σᵢ αᵢ·elpdₖᵢ averaged over 1000 Bayesian
//! bootstrap draws α ~ Dirichlet(1, …, 1).
//!
//! ```
//! use keanu_algorithms::loglik::PointwiseLogLik;
//! use keanu_algorithms::loo::{compare, loo};
//! use keanu_algorithms::mcmc::chains::sample_chains;
//! use keanu_algorithms::mcmc::nuts::Nuts;
//! use keanu_graph::{GraphBuilder, Model};
//! use keanu_tensor::DoubleTensor;
//!
//! let data = vec![0.4, 1.3, 0.9, 1.8, 0.2, 1.1];
//! // y ~ N(mu, sigma), with sigma known and either 1 or (wrongly) 0.2
//! let fit = |sigma: f64| -> Result<_, Box<dyn std::error::Error>> {
//!     let b = GraphBuilder::new();
//!     let mu = b.gaussian(0.0, 5.0);
//!     let y = b.gaussian_with_shape(&[6], mu, sigma).label("y");
//!     b.observe(y, DoubleTensor::vector(data.clone()));
//!     let mu = mu.id();
//!     let (graph, state) = b.build()?;
//!     let model = Model::new(graph, &state)?;
//!     let nuts = Nuts::default().with_adapt_count(200);
//!     let chains = sample_chains(2, 1, &state, |state, rng| nuts.sample(&model, state, &[mu], 500, rng))?;
//!     Ok(loo(&PointwiseLogLik::from_chains(&model, &state, &chains)?)?)
//! };
//! let (good, bad) = (fit(1.0)?, fit(0.2)?);
//! assert_eq!(good.points.len(), 6);
//! let table = compare(&[("sigma 1", &good), ("sigma 0.2", &bad)])?;
//! assert_eq!(table.rows()[0].name, "sigma 1");
//! assert!(table.rows()[1].elpd_diff < 0.0);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::diagnostics::ess_unsplit;
use crate::error::{AlgorithmError, Result};
use crate::loglik::PointwiseLogLik;
use crate::optimizer::Lbfgs;
use crate::psis::{khat_threshold, psis};
use keanu_distributions::KeanuRng;
use keanu_tensor::special::{log_sum_exp, ndtri};
use std::fmt;

/// The p_waic of a point above which WAIC is unreliable (Vehtari, Gelman and Gabry, 2017).
pub const P_WAIC_THRESHOLD: f64 = 0.4;

/// The number of Bayesian bootstrap draws behind [`pseudo_bma_weights`] (as in `loo` and ArviZ).
pub const PSEUDO_BMA_BOOTSTRAP: usize = 1000;

/// The number of normal quantiles carrying each point's Monte Carlo variance to the log scale (`loo`'s default).
const MCSE_QUANTILES: usize = 1000;

/// An estimate of the expected log pointwise predictive density, as [`compare`] and the weights need it.
pub trait ElpdEstimate {
    /// The estimated elpd, summed over the points.
    fn elpd(&self) -> f64;
    /// The standard error of [`elpd`](Self::elpd).
    fn se(&self) -> f64;
    /// The effective number of parameters.
    fn p(&self) -> f64;
    /// Each point's elpd, in order.
    fn pointwise_elpd(&self) -> Vec<f64>;
    /// Whether the estimate has warnings about its reliability.
    fn has_warnings(&self) -> bool;
}

/// One point's PSIS-LOO estimates.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct LooPoint {
    /// The point's name ([`PointwiseLogLik::names`]).
    pub name: String,
    /// log p(yᵢ | y₋ᵢ), estimated by PSIS.
    pub elpd: f64,
    /// The point's effective number of parameters, lpdᵢ − elpdᵢ.
    pub p: f64,
    /// −2·elpd.
    pub looic: f64,
    /// The Monte Carlo standard error of `elpd`.
    pub mcse_elpd: f64,
    /// The Pareto k̂ of the point's importance weights.
    pub pareto_k: f64,
    /// The relative efficiency the point's PSIS used.
    pub r_eff: f64,
}

/// PSIS-LOO estimates; see the [module documentation](self).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Loo {
    /// The expected log pointwise predictive density.
    pub elpd: f64,
    /// The standard error of `elpd`.
    pub se: f64,
    /// The effective number of parameters, p_loo.
    pub p: f64,
    /// The standard error of `p`.
    pub p_se: f64,
    /// The LOO information criterion, −2·elpd.
    pub looic: f64,
    /// The standard error of `looic`, 2·se.
    pub looic_se: f64,
    /// The Monte Carlo standard error of `elpd`.
    pub mcse_elpd: f64,
    /// The estimates for each point.
    pub points: Vec<LooPoint>,
    /// The k̂ above which a point's estimate is unreliable, for this number of draws.
    pub khat_threshold: f64,
    /// The number of draws, over all chains.
    pub draws: usize,
    /// Warnings about the reliability of the estimates (points whose k̂ is above the threshold); empty if there are
    /// none.
    pub warnings: Vec<String>,
}

impl Loo {
    /// Each point's Pareto k̂, in order.
    pub fn pareto_k(&self) -> Vec<f64> {
        self.points.iter().map(|p| p.pareto_k).collect()
    }

    /// The points whose k̂ is above [`khat_threshold`](Self::khat_threshold) (or NaN).
    pub fn bad_points(&self) -> Vec<&LooPoint> {
        self.points
            .iter()
            .filter(|p| p.pareto_k.is_nan() || p.pareto_k > self.khat_threshold)
            .collect()
    }
}

impl ElpdEstimate for Loo {
    fn elpd(&self) -> f64 {
        self.elpd
    }
    fn se(&self) -> f64 {
        self.se
    }
    fn p(&self) -> f64 {
        self.p
    }
    fn pointwise_elpd(&self) -> Vec<f64> {
        self.points.iter().map(|p| p.elpd).collect()
    }
    fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }
}

impl fmt::Display for Loo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "PSIS-LOO from {} draws and {} points",
            self.draws,
            self.points.len()
        )?;
        writeln!(f, "{:<10} {:>10} {:>8}", "", "estimate", "se")?;
        writeln!(f, "{:<10} {:>10.2} {:>8.2}", "elpd_loo", self.elpd, self.se)?;
        writeln!(f, "{:<10} {:>10.2} {:>8.2}", "p_loo", self.p, self.p_se)?;
        writeln!(
            f,
            "{:<10} {:>10.2} {:>8.2}",
            "looic", self.looic, self.looic_se
        )?;
        writeln!(f, "MCSE of elpd_loo: {:.2}", self.mcse_elpd)?;
        let bad = self.bad_points().len();
        if bad == 0 {
            write!(f, "every Pareto k̂ is at most {:.2}", self.khat_threshold)?;
        } else {
            write!(
                f,
                "{bad} of {} Pareto k̂ are above {:.2}",
                self.points.len(),
                self.khat_threshold
            )?;
        }
        Ok(())
    }
}

/// One point's WAIC estimates.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct WaicPoint {
    /// The point's name.
    pub name: String,
    /// lpdᵢ − p_waicᵢ.
    pub elpd: f64,
    /// p_waicᵢ, the sample variance of the point's log likelihood over the draws.
    pub p: f64,
    /// −2·elpd.
    pub waic: f64,
}

/// WAIC estimates; see the [module documentation](self).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Waic {
    /// The expected log pointwise predictive density.
    pub elpd: f64,
    /// The standard error of `elpd`.
    pub se: f64,
    /// The effective number of parameters, p_waic.
    pub p: f64,
    /// The standard error of `p`.
    pub p_se: f64,
    /// The information criterion, −2·elpd.
    pub waic: f64,
    /// The standard error of `waic`, 2·se.
    pub waic_se: f64,
    /// The estimates for each point.
    pub points: Vec<WaicPoint>,
    /// The number of draws, over all chains.
    pub draws: usize,
    /// Warnings about the reliability of the estimates (points whose p_waic is above [`P_WAIC_THRESHOLD`]); empty
    /// if there are none.
    pub warnings: Vec<String>,
}

impl ElpdEstimate for Waic {
    fn elpd(&self) -> f64 {
        self.elpd
    }
    fn se(&self) -> f64 {
        self.se
    }
    fn p(&self) -> f64 {
        self.p
    }
    fn pointwise_elpd(&self) -> Vec<f64> {
        self.points.iter().map(|p| p.elpd).collect()
    }
    fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }
}

impl fmt::Display for Waic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "WAIC from {} draws and {} points",
            self.draws,
            self.points.len()
        )?;
        writeln!(f, "{:<10} {:>10} {:>8}", "", "estimate", "se")?;
        writeln!(
            f,
            "{:<10} {:>10.2} {:>8.2}",
            "elpd_waic", self.elpd, self.se
        )?;
        writeln!(f, "{:<10} {:>10.2} {:>8.2}", "p_waic", self.p, self.p_se)?;
        write!(
            f,
            "{:<10} {:>10.2} {:>8.2}",
            "waic", self.waic, self.waic_se
        )
    }
}

/// PSIS-LOO estimates from a pointwise log likelihood, with each point's relative efficiency from its draws'
/// chains ([`relative_efficiency`]).
///
/// # Errors
///
/// As [`loo_from_values`].
pub fn loo(loglik: &PointwiseLogLik) -> Result<Loo> {
    loo_from_values(loglik.values(), loglik.names(), None)
}

/// PSIS-LOO estimates from log likelihood values `[chain][draw][point]` and the points' names, with the given
/// relative efficiency of each point, or by default each point's from its draws ([`relative_efficiency`]).
///
/// # Errors
///
/// [`AlgorithmError::InvalidArgument`] if there are fewer than two draws or no points, the chains or draws differ
/// in length, the names or relative efficiencies don't match the points, or a value is NaN or +∞ (or −∞, which
/// would make the point's importance ratio infinite).
pub fn loo_from_values(
    values: &[Vec<Vec<f64>>],
    names: &[String],
    r_eff: Option<&[f64]>,
) -> Result<Loo> {
    let columns = Columns::new(values, names)?;
    let r_eff = match r_eff {
        Some(r_eff) if r_eff.len() == columns.points() => r_eff.to_vec(),
        Some(r_eff) => {
            return Err(AlgorithmError::InvalidArgument(format!(
                "{} relative efficiencies for {} points",
                r_eff.len(),
                columns.points()
            )));
        }
        None => relative_efficiency(values)?,
    };
    let draws = columns.draws();
    let threshold = khat_threshold(draws);
    let points = columns
        .columns
        .iter()
        .zip(names)
        .zip(&r_eff)
        .map(|((ll, name), &r_eff)| loo_point(ll, name, r_eff))
        .collect::<Result<Vec<_>>>()?;
    let elpd: Vec<f64> = points.iter().map(|p| p.elpd).collect();
    let p: Vec<f64> = points.iter().map(|p| p.p).collect();
    let (elpd_total, se) = total_and_se(&elpd);
    let (p_total, p_se) = total_and_se(&p);
    let mcse_elpd = points
        .iter()
        .map(|p| p.mcse_elpd * p.mcse_elpd)
        .sum::<f64>()
        .sqrt();
    let bad: Vec<&str> = points
        .iter()
        .filter(|p| p.pareto_k.is_nan() || p.pareto_k > threshold)
        .map(|p| p.name.as_str())
        .collect();
    let warnings = if bad.is_empty() {
        Vec::new()
    } else {
        vec![format!(
            "{} of {} points have a Pareto k̂ above {threshold:.2}, so their PSIS-LOO estimates are unreliable: {}. \
             Such points are very influential; consider K-fold cross-validation, or a model that predicts them better",
            bad.len(),
            points.len(),
            bad.join(", ")
        )]
    };
    Ok(Loo {
        elpd: elpd_total,
        se,
        p: p_total,
        p_se,
        looic: -2.0 * elpd_total,
        looic_se: 2.0 * se,
        mcse_elpd,
        points,
        khat_threshold: threshold,
        draws,
        warnings,
    })
}

/// One point's PSIS-LOO estimates from its log likelihood at every draw.
fn loo_point(ll: &[f64], name: &str, r_eff: f64) -> Result<LooPoint> {
    let log_ratios: Vec<f64> = ll.iter().map(|l| -l).collect();
    let smoothed = psis(&log_ratios, r_eff).map_err(|e| {
        AlgorithmError::InvalidArgument(format!("PSIS of point {name} failed: {e}"))
    })?;
    let lw = smoothed.normalized_log_weights();
    let elpd = log_sum_exp(lw.iter().zip(ll).map(|(w, l)| w + l));
    let lpd = log_sum_exp(ll.iter().copied()) - (ll.len() as f64).ln();
    Ok(LooPoint {
        name: name.to_owned(),
        elpd,
        p: lpd - elpd,
        looic: -2.0 * elpd,
        mcse_elpd: mcse_elpd(ll, &lw, elpd, r_eff),
        pareto_k: smoothed.pareto_k(),
        r_eff,
    })
}

/// The Monte Carlo standard error of one point's elpd, as `loo`'s `mcse_elpd` (with normal quantiles instead of
/// random normal draws). Everything is scaled by the largest likelihood, which the variance of a log ignores.
fn mcse_elpd(ll: &[f64], normalized_log_weights: &[f64], elpd: f64, r_eff: f64) -> f64 {
    let scale = ll.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let expected = (elpd - scale).exp();
    let variance: f64 = normalized_log_weights
        .iter()
        .zip(ll)
        .map(|(w, l)| {
            let deviation = (l - scale).exp() - expected;
            (2.0 * w).exp() * deviation * deviation
        })
        .sum::<f64>()
        / r_eff;
    let sd = variance.sqrt();
    let logs: Vec<f64> = (0..MCSE_QUANTILES)
        .map(|j| expected + sd * ndtri((j as f64 + 0.5) / MCSE_QUANTILES as f64))
        .filter(|&z| z > 0.0)
        .map(f64::ln)
        .collect();
    sample_variance(&logs).sqrt()
}

/// WAIC estimates from a pointwise log likelihood.
///
/// # Errors
///
/// As [`waic_from_values`].
pub fn waic(loglik: &PointwiseLogLik) -> Result<Waic> {
    waic_from_values(loglik.values(), loglik.names())
}

/// WAIC estimates from log likelihood values `[chain][draw][point]` and the points' names.
///
/// # Errors
///
/// [`AlgorithmError::InvalidArgument`] if there are fewer than two draws or no points, the chains or draws differ
/// in length, the names don't match the points, or a value is NaN or infinite.
pub fn waic_from_values(values: &[Vec<Vec<f64>>], names: &[String]) -> Result<Waic> {
    let columns = Columns::new(values, names)?;
    if let Some(name) = columns
        .columns
        .iter()
        .zip(names)
        .find(|(ll, _)| ll.contains(&f64::NEG_INFINITY))
        .map(|(_, name)| name)
    {
        return Err(AlgorithmError::InvalidArgument(format!(
            "the log likelihood of point {name} is −∞ at a draw"
        )));
    }
    let draws = columns.draws();
    let points: Vec<WaicPoint> = columns
        .columns
        .iter()
        .zip(names)
        .map(|(ll, name)| {
            let lpd = log_sum_exp(ll.iter().copied()) - (draws as f64).ln();
            let p = sample_variance(ll);
            WaicPoint {
                name: name.clone(),
                elpd: lpd - p,
                p,
                waic: -2.0 * (lpd - p),
            }
        })
        .collect();
    let elpd: Vec<f64> = points.iter().map(|p| p.elpd).collect();
    let p: Vec<f64> = points.iter().map(|p| p.p).collect();
    let (elpd_total, se) = total_and_se(&elpd);
    let (p_total, p_se) = total_and_se(&p);
    let high: Vec<&str> = points
        .iter()
        .filter(|p| p.p > P_WAIC_THRESHOLD)
        .map(|p| p.name.as_str())
        .collect();
    let warnings = if high.is_empty() {
        Vec::new()
    } else {
        vec![format!(
            "{} of {} points have a p_waic above {P_WAIC_THRESHOLD}, so WAIC may be unreliable (PSIS-LOO is more \
             robust): {}",
            high.len(),
            points.len(),
            high.join(", ")
        )]
    };
    Ok(Waic {
        elpd: elpd_total,
        se,
        p: p_total,
        p_se,
        waic: -2.0 * elpd_total,
        waic_se: 2.0 * se,
        points,
        draws,
        warnings,
    })
}

/// The relative efficiency of each point's draws, as `loo`'s `relative_eff`: the effective sample size of the
/// likelihood p(yᵢ | θₛ) over the chains (neither split nor rank-normalised), divided by the number of draws.
/// Draws too few for an effective sample size (under four per chain) or giving a non-finite one have relative
/// efficiency 1.
///
/// # Errors
///
/// [`AlgorithmError::InvalidArgument`] if there are no chains or draws, or the chains or draws differ in length.
pub fn relative_efficiency(values: &[Vec<Vec<f64>>]) -> Result<Vec<f64>> {
    let points = check_shape(values)?;
    let draws = values.len() * values[0].len();
    (0..points)
        .map(|i| {
            let scale = values
                .iter()
                .flatten()
                .map(|draw| draw[i])
                .fold(f64::NEG_INFINITY, f64::max);
            let likelihood: Vec<Vec<f64>> = values
                .iter()
                .map(|chain| chain.iter().map(|draw| (draw[i] - scale).exp()).collect())
                .collect();
            let ess = ess_unsplit(&likelihood)?;
            Ok(if ess.is_finite() && ess > 0.0 {
                ess / draws as f64
            } else {
                1.0
            })
        })
        .collect()
}

/// One row of a [`Comparison`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ComparisonRow {
    /// The model's name.
    pub name: String,
    /// Its rank, 0 for the best (highest elpd).
    pub rank: usize,
    /// Its elpd.
    pub elpd: f64,
    /// Its effective number of parameters.
    pub p: f64,
    /// The standard error of its elpd.
    pub se: f64,
    /// Its elpd minus the best model's: 0 for the best, negative for the others.
    pub elpd_diff: f64,
    /// The standard error of `elpd_diff`, from the pointwise differences (0 for the best).
    pub se_diff: f64,
    /// Its stacking weight ([`stacking_weights`]).
    pub weight: f64,
    /// Whether its estimate has warnings.
    pub warning: bool,
}

/// Models ranked by their estimated elpd, best first; see the [module documentation](self).
#[derive(Debug, Clone, PartialEq)]
pub struct Comparison {
    rows: Vec<ComparisonRow>,
}

impl Comparison {
    /// The rows, best model first.
    pub fn rows(&self) -> &[ComparisonRow] {
        &self.rows
    }

    /// The row of the model called `name`.
    pub fn get(&self, name: &str) -> Option<&ComparisonRow> {
        self.rows.iter().find(|r| r.name == name)
    }

    /// The best model's row.
    pub fn best(&self) -> &ComparisonRow {
        &self.rows[0]
    }
}

impl fmt::Display for Comparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let width = self
            .rows
            .iter()
            .map(|r| r.name.chars().count())
            .max()
            .unwrap_or(0)
            .max(5);
        write!(
            f,
            "{:<width$} {:>4} {:>9} {:>7} {:>9} {:>7} {:>8} {:>7} {:>7}",
            "model", "rank", "elpd", "p", "elpd_diff", "se", "se_diff", "weight", "warning"
        )?;
        for r in &self.rows {
            write!(
                f,
                "\n{:<width$} {:>4} {:>9.2} {:>7.2} {:>9.2} {:>7.2} {:>8.2} {:>7.3} {:>7}",
                r.name, r.rank, r.elpd, r.p, r.elpd_diff, r.se, r.se_diff, r.weight, r.warning
            )?;
        }
        Ok(())
    }
}

/// Compares models by their estimated elpd ([`Loo`] or [`Waic`]), each given with a name: ranks them best first,
/// with each one's difference from the best and its standard error, and stacking weights. See the
/// [module documentation](self).
///
/// # Errors
///
/// [`AlgorithmError::InvalidArgument`] if there are no models or they have different numbers of points (they must
/// be evaluated on the same observations), and any error from [`stacking_weights`].
pub fn compare<E: ElpdEstimate>(models: &[(&str, &E)]) -> Result<Comparison> {
    let estimates: Vec<&E> = models.iter().map(|(_, e)| *e).collect();
    let pointwise = check_models(&estimates)?;
    let weights = stacking(&pointwise)?;
    let mut order: Vec<usize> = (0..models.len()).collect();
    order.sort_by(|&a, &b| models[b].1.elpd().total_cmp(&models[a].1.elpd()));
    let best = order[0];
    let rows = order
        .iter()
        .enumerate()
        .map(|(rank, &m)| {
            let (name, estimate) = models[m];
            let differences: Vec<f64> = pointwise[m]
                .iter()
                .zip(&pointwise[best])
                .map(|(a, b)| a - b)
                .collect();
            let (elpd_diff, se_diff) = if m == best {
                (0.0, 0.0)
            } else {
                total_and_se(&differences)
            };
            ComparisonRow {
                name: name.to_owned(),
                rank,
                elpd: estimate.elpd(),
                p: estimate.p(),
                se: estimate.se(),
                elpd_diff,
                se_diff,
                weight: weights[m],
                warning: estimate.has_warnings(),
            }
        })
        .collect();
    Ok(Comparison { rows })
}

/// Stacking weights of models (Yao et al., 2018): the weights w on the simplex that maximise
/// Σᵢ log Σₖ wₖ exp(elpdₖᵢ), the log score of the weighted mixture of the models' leave-one-out predictive
/// densities, in the models' order.
///
/// # Errors
///
/// [`AlgorithmError::InvalidArgument`] if there are no models, they have different numbers of points, or a
/// pointwise elpd is NaN or +∞.
pub fn stacking_weights<E: ElpdEstimate>(models: &[&E]) -> Result<Vec<f64>> {
    stacking(&check_models(models)?)
}

/// Pseudo-BMA+ weights of models (Yao et al., 2018): with n points, the mean over [`PSEUDO_BMA_BOOTSTRAP`] draws
/// of α ~ Dirichlet(1, …, 1) of the softmax over the models of n·Σᵢ αᵢ·elpdₖᵢ, in the models' order. The Bayesian
/// bootstrap accounts for the uncertainty in the elpd estimates, so the weights are less extreme than those of
/// plain pseudo-BMA (the softmax of the elpds).
///
/// # Errors
///
/// As [`stacking_weights`].
pub fn pseudo_bma_weights<E: ElpdEstimate>(models: &[&E], rng: &mut KeanuRng) -> Result<Vec<f64>> {
    let pointwise = check_models(models)?;
    let n = pointwise[0].len();
    let mut weights = vec![0.0; models.len()];
    for _ in 0..PSEUDO_BMA_BOOTSTRAP {
        // Dirichlet(1, ..., 1) as normalised exponential draws
        let mut alpha: Vec<f64> = (0..n).map(|_| -rng.next_f64_non_zero().ln()).collect();
        let total: f64 = alpha.iter().sum();
        alpha.iter_mut().for_each(|a| *a /= total);
        let z: Vec<f64> = pointwise
            .iter()
            .map(|elpd| n as f64 * elpd.iter().zip(&alpha).map(|(e, a)| e * a).sum::<f64>())
            .collect();
        for (w, s) in weights.iter_mut().zip(softmax(&z)) {
            *w += s;
        }
    }
    weights
        .iter_mut()
        .for_each(|w| *w /= PSEUDO_BMA_BOOTSTRAP as f64);
    Ok(weights)
}

/// The models' pointwise elpds, `[model][point]`, checked.
fn check_models<E: ElpdEstimate>(models: &[&E]) -> Result<Vec<Vec<f64>>> {
    let Some(first) = models.first() else {
        return Err(AlgorithmError::InvalidArgument(
            "there are no models to compare".into(),
        ));
    };
    let n = first.pointwise_elpd().len();
    let pointwise: Vec<Vec<f64>> = models.iter().map(|m| m.pointwise_elpd()).collect();
    if n == 0 || pointwise.iter().any(|p| p.len() != n) {
        return Err(AlgorithmError::InvalidArgument(format!(
            "the models must have the same, positive, number of points, but they have {}",
            pointwise
                .iter()
                .map(|p| p.len().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    if pointwise
        .iter()
        .flatten()
        .any(|e| e.is_nan() || *e == f64::INFINITY)
    {
        return Err(AlgorithmError::InvalidArgument(
            "a pointwise elpd is NaN or +∞".into(),
        ));
    }
    Ok(pointwise)
}

/// Stacking weights of pointwise elpds `[model][point]`, by L-BFGS on the softmax parameterisation (the last
/// model's parameter fixed at 0).
fn stacking(pointwise: &[Vec<f64>]) -> Result<Vec<f64>> {
    let models = pointwise.len();
    if models == 1 {
        return Ok(vec![1.0]);
    }
    let n = pointwise[0].len();
    // each point's densities relative to its best model's, which changes the log score by a constant
    let densities: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let best = pointwise
                .iter()
                .map(|e| e[i])
                .fold(f64::NEG_INFINITY, f64::max);
            pointwise.iter().map(|e| (e[i] - best).exp()).collect()
        })
        .collect();
    let weights_of = |beta: &[f64]| {
        let mut z = beta.to_vec();
        z.push(0.0);
        softmax(&z)
    };
    let negative_log_score = |beta: &[f64]| -> Result<(f64, Vec<f64>)> {
        let w = weights_of(beta);
        let mut value = 0.0;
        // g_k = Σᵢ pᵢₖ / rᵢ, the derivative of the log score by w_k
        let mut g = vec![0.0; models];
        for p in &densities {
            let r: f64 = p.iter().zip(&w).map(|(p, w)| p * w).sum();
            value -= r.ln();
            for (gk, pk) in g.iter_mut().zip(p) {
                *gk += pk / r;
            }
        }
        let mean: f64 = g.iter().zip(&w).map(|(g, w)| g * w).sum();
        let gradient = (0..models - 1).map(|k| -w[k] * (g[k] - mean)).collect();
        Ok((value, gradient))
    };
    let (beta, _, _) =
        Lbfgs::default().minimize_function(vec![0.0; models - 1], negative_log_score)?;
    Ok(weights_of(&beta))
}

fn softmax(z: &[f64]) -> Vec<f64> {
    let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = z.iter().map(|v| (v - max).exp()).collect();
    let total: f64 = exp.iter().sum();
    exp.iter().map(|e| e / total).collect()
}

/// The sum of pointwise values and its standard error, √(n · var) with the n − 1 sample variance.
fn total_and_se(values: &[f64]) -> (f64, f64) {
    let total = values.iter().sum();
    let n = values.len() as f64;
    let se = if values.len() > 1 {
        (n * sample_variance(values)).sqrt()
    } else {
        0.0
    };
    (total, se)
}

/// The sample variance (n − 1 denominator); NaN for fewer than two values.
fn sample_variance(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0)
}

/// Checks that values `[chain][draw][point]` have at least one chain and draw and the same number of draws in each
/// chain and points in each draw, and returns the number of points.
fn check_shape(values: &[Vec<Vec<f64>>]) -> Result<usize> {
    let draws = values.first().map_or(0, Vec::len);
    if draws == 0 {
        return Err(AlgorithmError::InvalidArgument(
            "there are no draws of the log likelihood".into(),
        ));
    }
    let points = values[0][0].len();
    if values.iter().any(|chain| chain.len() != draws) {
        return Err(AlgorithmError::InvalidArgument(
            "every chain must have the same number of draws".into(),
        ));
    }
    if values.iter().flatten().any(|draw| draw.len() != points) {
        return Err(AlgorithmError::InvalidArgument(
            "every draw must have the same number of points".into(),
        ));
    }
    Ok(points)
}

/// The log likelihood values of each point over all draws, chain after chain.
struct Columns {
    columns: Vec<Vec<f64>>,
}

impl Columns {
    fn new(values: &[Vec<Vec<f64>>], names: &[String]) -> Result<Self> {
        let points = check_shape(values)?;
        if points == 0 {
            return Err(AlgorithmError::InvalidArgument(
                "there are no points in the log likelihood".into(),
            ));
        }
        if names.len() != points {
            return Err(AlgorithmError::InvalidArgument(format!(
                "{} names for {points} points",
                names.len()
            )));
        }
        let columns: Vec<Vec<f64>> = (0..points)
            .map(|i| values.iter().flatten().map(|draw| draw[i]).collect())
            .collect();
        if columns[0].len() < 2 {
            return Err(AlgorithmError::InvalidArgument(
                "the estimates need at least two draws".into(),
            ));
        }
        if let Some((_, name)) = columns
            .iter()
            .zip(names)
            .find(|(ll, _)| ll.iter().any(|l| l.is_nan() || *l == f64::INFINITY))
        {
            return Err(AlgorithmError::InvalidArgument(format!(
                "the log likelihood of point {name} is NaN or +∞ at a draw"
            )));
        }
        Ok(Self { columns })
    }

    fn points(&self) -> usize {
        self.columns.len()
    }

    fn draws(&self) -> usize {
        self.columns[0].len()
    }
}
