//! When will this fleet start failing? Equipment reliability from failures and units still running.
//!
//! A fleet of pumps was installed over the past five years, some under high load. A few have failed, at a known
//! age; most are still running, so all we know of their lifetime is that it is longer than their age so far. Each
//! pump's lifetime is Weibull, with a shape shared by the fleet (above 1 means wear-out: failures speed up with age)
//! and a scale that high load shortens:
//!
//! ```text
//! log k ~ N(ln 1.5, 0.5)    log η ~ N(ln 60, 1)    γ ~ N(0, 0.5)
//! lifetime_i ~ Weibull(k, η exp(γ high_i)),   observed at failure, or right-censored at its age if still running
//! ```
//!
//! A running pump contributes the probability it has survived to its age, S(age) = exp(−(age/η)^k), through
//! Keanu's censored distribution. Dropping the running pumps instead, the classic mistake, keeps only the lifetimes
//! short enough to have ended already, so the fleet looks far less reliable than it is; [`naive_fit`] fits that
//! mistake by maximum likelihood for comparison.
//!
//! The posterior then prices a warranty: for each draw, a new pump fails within the warranty with probability
//! F(w), averaged over the load mix, so the claims on a new fleet are binomial. The warranty length, fleet size and
//! cost per claim are applied to the stored draws, so they update without refitting.
//!
//! The fleet is synthetic, generated from a seed with a known lifetime distribution, so a demo can show how well it
//! is recovered.

use crate::run::{ChainRun, Histogram, Interval, mean, quantile, scalar_draws};
use keanu::distributions::Binomial;
use keanu::graph::op::DistributionKind;
use keanu::prelude::*;
use serde::Serialize;

/// The Weibull shape the fleet's lifetimes were generated with (above 1: wear-out).
pub const TRUE_SHAPE: f64 = 2.0;
/// The Weibull scale, in months, of a pump under normal load.
pub const TRUE_SCALE: f64 = 72.0;
/// The factor high load multiplies the scale by.
pub const TRUE_HIGH_LOAD_FACTOR: f64 = 0.6;
/// The share of pumps under high load, in the history and in a new fleet.
pub const HIGH_LOAD_SHARE: f64 = 0.35;
/// The oldest pump's age, in months: installations are spread evenly over this many months.
pub const INSTALL_WINDOW: f64 = 60.0;
/// The survival curve's horizon, in months.
pub const CURVE_MONTHS: usize = 120;

/// One pump in the history: its age, load, and whether it has failed (at `time`) or is still running (aged `time`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Unit {
    /// The age at failure if failed, otherwise the current age, in months.
    pub time: f64,
    /// Whether the pump has failed.
    pub failed: bool,
    /// Whether the pump runs under high load.
    pub high_load: bool,
}

/// The true Weibull scale for a load.
pub fn true_scale(high_load: bool) -> f64 {
    if high_load {
        TRUE_SCALE * TRUE_HIGH_LOAD_FACTOR
    } else {
        TRUE_SCALE
    }
}

/// A synthetic fleet of `count` pumps installed evenly over the last [`INSTALL_WINDOW`] months.
pub fn fleet(count: usize, seed: u64) -> Vec<Unit> {
    let mut rng = KeanuRng::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            let age = rng.next_f64_between(1.0, INSTALL_WINDOW);
            let high_load = rng.next_f64() < HIGH_LOAD_SHARE;
            // inverse CDF: t = η (−ln U)^(1/k)
            let lifetime =
                true_scale(high_load) * (-rng.next_f64_non_zero().ln()).powf(1.0 / TRUE_SHAPE);
            Unit {
                time: lifetime.min(age),
                failed: lifetime < age,
                high_load,
            }
        })
        .collect()
}

/// Weibull lifetime parameters: a shared shape, and a scale for each load.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Weibull {
    /// The shape k.
    pub shape: f64,
    /// The scale under normal load, in months.
    pub scale_normal: f64,
    /// The scale under high load, in months.
    pub scale_high: f64,
}

impl Weibull {
    /// The truth the synthetic fleet was generated with.
    pub fn truth() -> Self {
        Self {
            shape: TRUE_SHAPE,
            scale_normal: true_scale(false),
            scale_high: true_scale(true),
        }
    }

    fn survival_at(&self, scale: f64, t: f64) -> f64 {
        (-(t / scale).powf(self.shape)).exp()
    }

    /// The probability a pump of the load mix survives past `t` months.
    pub fn survival(&self, t: f64) -> f64 {
        (1.0 - HIGH_LOAD_SHARE) * self.survival_at(self.scale_normal, t)
            + HIGH_LOAD_SHARE * self.survival_at(self.scale_high, t)
    }

    /// The age by which a `q` share of pumps with `scale` has failed.
    fn life_at(&self, scale: f64, q: f64) -> f64 {
        scale * (-(1.0 - q).ln()).powf(1.0 / self.shape)
    }

    /// The age by which a `q` share of the load mix has failed (B10 life for q = 0.1), by bisection.
    pub fn fleet_life(&self, q: f64) -> f64 {
        let (mut lo, mut hi) = (
            self.life_at(self.scale_normal.min(self.scale_high), q),
            self.life_at(self.scale_normal.max(self.scale_high), q),
        );
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if 1.0 - self.survival(mid) < q {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }
}

/// The maximum likelihood Weibull fit to the failed pumps alone, ignoring the ones still running: the classic
/// mistake. The scales are profiled out for each shape, which is found by golden-section search on its log.
pub fn naive_fit(units: &[Unit]) -> Weibull {
    let failed = |high: bool| -> Vec<f64> {
        units
            .iter()
            .filter(|u| u.failed && u.high_load == high)
            .map(|u| u.time)
            .collect()
    };
    let (normal, high) = (failed(false), failed(true));
    let scale = |times: &[f64], k: f64| {
        (times.iter().map(|t| t.powf(k)).sum::<f64>() / times.len().max(1) as f64).powf(1.0 / k)
    };
    // profile log likelihood: n ln k − k Σ ln η_g + (k − 1) Σ ln t − n
    let log_likelihood = |k: f64| {
        let mut total = 0.0;
        for times in [&normal, &high] {
            if times.is_empty() {
                continue;
            }
            let eta = scale(times, k);
            let n = times.len() as f64;
            total += n * k.ln() - n * k * eta.ln()
                + (k - 1.0) * times.iter().map(|t| t.ln()).sum::<f64>()
                - n;
        }
        total
    };
    let (mut a, mut b) = (0.1f64.ln(), 20f64.ln());
    let ratio = (5f64.sqrt() - 1.0) / 2.0;
    for _ in 0..80 {
        let c = b - ratio * (b - a);
        let d = a + ratio * (b - a);
        if log_likelihood(c.exp()) > log_likelihood(d.exp()) {
            b = d;
        } else {
            a = c;
        }
    }
    let k = (0.5 * (a + b)).exp();
    Weibull {
        shape: k,
        scale_normal: scale(&normal, k),
        scale_high: scale(&high, k),
    }
}

/// A fit in progress.
#[derive(Debug)]
pub struct ReliabilityModel {
    chain: ChainRun,
    shape: NodeId,
    scale_normal: NodeId,
    scale_high: NodeId,
    units: Vec<Unit>,
    naive: Weibull,
}

/// A point on the survival curve: the share of pumps still running at `month`.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct CurvePoint {
    /// Months since installation.
    pub month: f64,
    /// The posterior mean and 90% interval.
    pub mean: f64,
    /// The 5% quantile.
    pub low: f64,
    /// The 95% quantile.
    pub high: f64,
    /// The true survival probability.
    pub truth: f64,
    /// The survival probability if the running pumps were ignored.
    pub naive: f64,
}

/// A lifetime summary for one load: the posterior, the truth, and the naive estimate.
#[derive(Debug, Clone, Serialize)]
pub struct LifeReport {
    /// The load: "Normal load", "High load" or "Whole fleet".
    pub load: &'static str,
    /// The posterior B10 life, in months.
    pub b10: Interval,
    /// The true B10 life.
    pub true_b10: f64,
    /// The B10 life if the running pumps were ignored.
    pub naive_b10: f64,
}

/// What the history says, and the warranty's cost.
#[derive(Debug, Clone, Serialize)]
pub struct ReliabilityReport {
    /// Pumps in the history.
    pub units: usize,
    /// Pumps that have failed.
    pub failed: usize,
    /// The posterior Weibull shape.
    pub shape: Interval,
    /// The posterior scale under normal load, in months.
    pub scale_normal: Interval,
    /// The posterior scale under high load, in months.
    pub scale_high: Interval,
    /// The posterior factor high load multiplies the scale by.
    pub high_load_factor: Interval,
    /// The truth the history was generated with.
    pub truth: Weibull,
    /// The fit to the failed pumps alone.
    pub naive: Weibull,
    /// The share of the load mix still running each month, from 0 to [`CURVE_MONTHS`].
    pub survival: Vec<CurvePoint>,
    /// The B10 life for each load and the whole fleet.
    pub lives: Vec<LifeReport>,
    /// The warranty length, in months.
    pub warranty_months: f64,
    /// Pumps in the new fleet.
    pub fleet_size: i64,
    /// The cost of one claim.
    pub cost_per_claim: f64,
    /// The claims budget.
    pub budget: f64,
    /// The probability a new pump fails within the warranty.
    pub failure_probability: Interval,
    /// The expected number of claims (posterior mean of fleet size × failure probability).
    pub expected_claims: f64,
    /// The 5% and 95% quantiles of the predicted number of claims, with posterior and binomial uncertainty.
    pub claims_low: f64,
    /// The 95% quantile of the predicted claims.
    pub claims_high: f64,
    /// The expected warranty cost.
    pub expected_cost: f64,
    /// The probability the claims cost more than the budget.
    pub probability_over_budget: f64,
    /// The distribution of the number of claims.
    pub claims_histogram: Histogram,
    /// The expected claims if the running pumps were ignored.
    pub naive_expected_claims: f64,
    /// The expected claims under the true lifetimes.
    pub true_expected_claims: f64,
    /// Posterior draws used.
    pub draws: usize,
}

impl ReliabilityModel {
    /// Starts fitting the history with `draws` NUTS draws after as many adaptation steps.
    pub fn new(units: Vec<Unit>, draws: usize, seed: u64) -> keanu::Result<Self> {
        let b = GraphBuilder::new();
        let log_shape = b.gaussian(1.5f64.ln(), 0.5);
        let log_scale = b.gaussian(60f64.ln(), 1.0);
        let load_effect = b.gaussian(0.0, 0.5);
        b.set_value(log_shape, DoubleTensor::scalar(1.5f64.ln()));
        b.set_value(log_scale, DoubleTensor::scalar(60f64.ln()));
        b.set_value(load_effect, DoubleTensor::scalar(0.0));
        let high = DoubleTensor::vector(
            units
                .iter()
                .map(|u| if u.high_load { 1.0 } else { 0.0 })
                .collect(),
        );
        let shape = log_shape.exp();
        let scales = (log_scale + load_effect * b.constant(high)).exp();
        // a running pump's age is its censoring point; a failed pump has no upper bound, so its time has the density
        let upper = DoubleTensor::vector(
            units
                .iter()
                .map(|u| if u.failed { f64::INFINITY } else { u.time })
                .collect(),
        );
        let lifetimes: DoubleVar = b.censored(
            DistributionKind::Weibull,
            &[shape.id(), scales.id()],
            f64::NEG_INFINITY,
            b.constant(upper),
        );
        b.observe(
            lifetimes,
            DoubleTensor::vector(units.iter().map(|u| u.time).collect()),
        );
        let scale_normal = log_scale.exp();
        let scale_high = (log_scale + load_effect).exp();
        let ids = [shape.id(), scale_normal.id(), scale_high.id()];
        let (graph, state) = b.build()?;
        let model = Model::new(graph, &state)?;
        let chain = ChainRun::new(model, state, &ids, Nuts::default(), draws, draws, seed)?;
        let naive = naive_fit(&units);
        Ok(Self {
            chain,
            shape: ids[0],
            scale_normal: ids[1],
            scale_high: ids[2],
            units,
            naive,
        })
    }

    /// The chain.
    pub fn chain(&mut self) -> &mut ChainRun {
        &mut self.chain
    }

    /// The history.
    pub fn units(&self) -> &[Unit] {
        &self.units
    }

    /// The posterior draws of the lifetime parameters.
    pub fn posterior(&self) -> Vec<Weibull> {
        let samples = self.chain.samples();
        let shapes = scalar_draws(samples, self.shape);
        let normal = scalar_draws(samples, self.scale_normal);
        let high = scalar_draws(samples, self.scale_high);
        shapes
            .into_iter()
            .zip(normal)
            .zip(high)
            .map(|((shape, scale_normal), scale_high)| Weibull {
                shape,
                scale_normal,
                scale_high,
            })
            .collect()
    }

    /// The fleet's reliability, and the claims on a warranty of `warranty_months` for `fleet_size` new pumps at
    /// `cost_per_claim` each against `budget`, from `scenarios` simulated claim counts per posterior draw.
    pub fn report(
        &self,
        warranty_months: f64,
        fleet_size: i64,
        cost_per_claim: f64,
        budget: f64,
        scenarios: usize,
    ) -> keanu::Result<ReliabilityReport> {
        let invalid = |message: &str| {
            keanu::Error::from(keanu::graph::GraphError::InvalidValue(message.into()))
        };
        if !(warranty_months.is_finite() && warranty_months >= 0.0) {
            return Err(invalid(
                "the warranty must be a non-negative number of months",
            ));
        }
        if fleet_size < 0 {
            return Err(invalid("the fleet size must not be negative"));
        }
        let posterior = self.posterior();
        if posterior.is_empty() {
            return Err(invalid("the model has no posterior draws yet"));
        }
        let truth = Weibull::truth();
        let naive = self.naive;
        let pick =
            |f: fn(&Weibull) -> f64| Interval::of(&posterior.iter().map(f).collect::<Vec<_>>());

        let survival = (0..=CURVE_MONTHS)
            .map(|month| {
                let t = month as f64;
                let values: Vec<f64> = posterior.iter().map(|w| w.survival(t)).collect();
                let summary = Interval::of(&values);
                CurvePoint {
                    month: t,
                    mean: summary.mean,
                    low: summary.low,
                    high: summary.high,
                    truth: truth.survival(t),
                    naive: naive.survival(t),
                }
            })
            .collect();

        let life = |load: &'static str, f: &dyn Fn(&Weibull) -> f64| LifeReport {
            load,
            b10: Interval::of(&posterior.iter().map(f).collect::<Vec<_>>()),
            true_b10: f(&truth),
            naive_b10: f(&naive),
        };
        let lives = vec![
            life("Normal load", &|w| w.life_at(w.scale_normal, 0.1)),
            life("High load", &|w| w.life_at(w.scale_high, 0.1)),
            life("Whole fleet", &|w| w.fleet_life(0.1)),
        ];

        let failure: Vec<f64> = posterior
            .iter()
            .map(|w| 1.0 - w.survival(warranty_months))
            .collect();
        let m = fleet_size as f64;
        let claims = Binomial::new(
            DoubleTensor::vector(failure.clone()),
            IntTensor::scalar(fleet_size),
        )
        .sample(
            &[scenarios.max(1), failure.len()],
            &mut self.chain.rng_copy(),
        )?;
        let claims: Vec<f64> = claims.as_slice().iter().map(|&c| c as f64).collect();
        let expected_claims = m * mean(&failure);
        let over = claims
            .iter()
            .filter(|&&c| c * cost_per_claim > budget)
            .count();

        Ok(ReliabilityReport {
            units: self.units.len(),
            failed: self.units.iter().filter(|u| u.failed).count(),
            shape: pick(|w| w.shape),
            scale_normal: pick(|w| w.scale_normal),
            scale_high: pick(|w| w.scale_high),
            high_load_factor: pick(|w| w.scale_high / w.scale_normal),
            truth,
            naive,
            survival,
            lives,
            warranty_months,
            fleet_size,
            cost_per_claim,
            budget,
            failure_probability: Interval::of(&failure),
            expected_claims,
            claims_low: quantile(&claims, 0.05),
            claims_high: quantile(&claims, 0.95),
            expected_cost: expected_claims * cost_per_claim,
            probability_over_budget: over as f64 / claims.len() as f64,
            claims_histogram: Histogram::of(&claims, 40),
            naive_expected_claims: m * (1.0 - naive.survival(warranty_months)),
            true_expected_claims: m * (1.0 - truth.survival(warranty_months)),
            draws: posterior.len(),
        })
    }
}
