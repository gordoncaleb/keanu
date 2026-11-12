//! Will the hospitals run out of beds? Forecasting an outbreak from its first weeks of reported cases.
//!
//! A town of [`POPULATION`] people has an outbreak of a new infection. Its course follows a discrete-time SIR
//! (susceptible, infected, recovered) model, one step per day: infected people stay infectious for `D` days on
//! average, and each infects R₀ / D people a day while almost everyone is susceptible, so R₀ is the number one
//! infected person infects at the start. Only a fraction ρ of new infections are reported, and the counts are
//! overdispersed (negative binomial with dispersion φ):
//!
//! ```text
//! new_t = S_{t−1} (1 − exp(−(R₀ / D) I_{t−1} / N))
//! S_t = S_{t−1} − new_t        I_t = I_{t−1} (1 − 1 / D) + new_t        S_0 = N − I₀, I_0 = I₀
//! reported_t ~ NegativeBinomial(mean ρ new_t, dispersion φ)
//!
//! R₀ ~ LogNormal(ln 2, 0.5)    D ~ LogNormal(ln 7, 0.2)    ρ ~ Beta(12, 28)
//! I₀ ~ LogNormal(ln 10, 1)     1 / φ ~ Exponential(5)
//! ```
//!
//! The recursion is a `scan` node: the graph holds one step, and NUTS takes gradients through every day. In the
//! growth phase the counts pin down the growth rate, about (R₀ − 1) / D, and the product ρ I₀; the priors on the
//! recovery time and the reporting rate (from clinical and survey studies, in a real outbreak) separate them.
//! Those trade-offs make the posterior a pair of tilted ridges (R₀ against D, ρ against I₀), so NUTS adapts a dense
//! mass matrix: measured in WebAssembly, it fits 500 draws about three times faster than the diagonal default, with
//! no divergences. An ODE solver would give a continuous-time model, but the daily recursion is what the data are
//! (daily counts) and is cheaper to differentiate.
//!
//! The forecast runs each posterior draw forward in plain Rust: a fixed [`HOSPITALISED`] share of infections need a
//! bed [`ADMISSION_DELAY`] days after infection and stay [`STAY`] days on average, so the beds in use fill and empty
//! like a leaky bucket. An intervention scales transmission down by a fraction from a chosen day; it reuses the
//! draws, so it updates without refitting.
//!
//! The case counts are synthetic, generated from a seed with known R₀, recovery time and reporting rate, so a demo
//! can show how well they are recovered.

use crate::run::{ChainRun, Histogram, Interval, quantile, scalar_draws};
use keanu::algorithms::mcmc::nuts::{Metric, QuadraticPotential};
use keanu::distributions::NegativeBinomial;
use keanu::prelude::*;
use serde::Serialize;

/// The number of people in the town.
pub const POPULATION: f64 = 300_000.0;
/// The share of infections that need a hospital bed.
pub const HOSPITALISED: f64 = 0.02;
/// Days from infection to hospital admission.
pub const ADMISSION_DELAY: usize = 7;
/// The average hospital stay, in days.
pub const STAY: f64 = 8.0;
/// The number of days the forecast covers, counting the observed ones.
pub const HORIZON: usize = 200;

/// The basic reproduction number the synthetic cases were generated with.
pub const TRUE_R0: f64 = 1.8;
/// The mean infectious period the synthetic cases were generated with, in days.
pub const TRUE_RECOVERY_DAYS: f64 = 6.0;
/// The share of new infections reported in the synthetic cases.
pub const TRUE_REPORTING: f64 = 0.25;
/// The number infected on day 0 of the synthetic outbreak.
pub const TRUE_INITIAL: f64 = 20.0;
/// The negative binomial dispersion of the synthetic counts.
pub const TRUE_DISPERSION: f64 = 10.0;

/// The parameters of one outbreak.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Outbreak {
    /// The basic reproduction number R₀.
    pub r0: f64,
    /// The mean infectious period D, in days.
    pub recovery_days: f64,
    /// The share of new infections reported.
    pub reporting: f64,
    /// The number infected on day 0.
    pub initial: f64,
    /// The negative binomial dispersion of the reported counts.
    pub dispersion: f64,
}

/// The parameters the synthetic cases were generated with.
pub const TRUTH: Outbreak = Outbreak {
    r0: TRUE_R0,
    recovery_days: TRUE_RECOVERY_DAYS,
    reporting: TRUE_REPORTING,
    initial: TRUE_INITIAL,
    dispersion: TRUE_DISPERSION,
};

/// An intervention: transmission scaled by 1 − `cut` from day `from_day` on.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Intervention {
    /// The fraction by which contacts fall, 0 to 1.
    pub cut: f64,
    /// The first day with fewer contacts (day 1 is the first reported day).
    pub from_day: usize,
}

impl Intervention {
    /// No intervention.
    pub const NONE: Self = Self {
        cut: 0.0,
        from_day: usize::MAX,
    };
}

/// An outbreak's course, day by day from day 1.
#[derive(Debug, Clone, PartialEq)]
pub struct Course {
    /// New infections each day.
    pub new_infections: Vec<f64>,
    /// Hospital beds in use at the end of each day.
    pub occupancy: Vec<f64>,
}

impl Outbreak {
    /// The course over `days` days under `intervention`: the same recursion as the fitted model, then hospital
    /// admissions and discharges.
    pub fn course(&self, days: usize, intervention: Intervention) -> Course {
        let g = 1.0 / self.recovery_days;
        let beta = self.r0 * g;
        let (mut s, mut i) = (POPULATION - self.initial, self.initial);
        let mut new_infections = Vec::with_capacity(days);
        let mut occupancy = Vec::with_capacity(days);
        let mut beds = 0.0;
        for day in 1..=days {
            let contact = if day >= intervention.from_day {
                1.0 - intervention.cut
            } else {
                1.0
            };
            let new = s * (1.0 - (-beta * contact * i / POPULATION).exp());
            s -= new;
            i = i * (1.0 - g) + new;
            new_infections.push(new);
            let admitted = if day > ADMISSION_DELAY {
                HOSPITALISED * new_infections[day - 1 - ADMISSION_DELAY]
            } else {
                0.0
            };
            beds = beds * (1.0 - 1.0 / STAY) + admitted;
            occupancy.push(beds);
        }
        Course {
            new_infections,
            occupancy,
        }
    }
}

/// Reported cases on days 1 to `days` of the synthetic outbreak, from a generator seeded with `seed`.
pub fn reported_cases(days: usize, seed: u64) -> Vec<i64> {
    let course = TRUTH.course(days, Intervention::NONE);
    let mut rng = KeanuRng::seed_from_u64(seed);
    report_noise(
        &course.new_infections,
        TRUTH.reporting,
        TRUTH.dispersion,
        &mut rng,
    )
}

/// Negative binomial counts with mean `reporting` × each day's new infections.
fn report_noise(new: &[f64], reporting: f64, dispersion: f64, rng: &mut KeanuRng) -> Vec<i64> {
    let means = new.iter().map(|n| (reporting * n).max(1e-9)).collect();
    NegativeBinomial::new(
        DoubleTensor::vector(means),
        DoubleTensor::scalar(dispersion),
    )
    .sample(&[new.len()], rng)
    .map(|t| t.as_slice().to_vec())
    .unwrap_or_else(|_| vec![0; new.len()])
}

/// A fit in progress.
#[derive(Debug)]
pub struct EpidemicModel {
    chain: ChainRun,
    cases: Vec<i64>,
    r0: NodeId,
    recovery_days: NodeId,
    reporting: NodeId,
    initial: NodeId,
    dispersion: NodeId,
}

/// The forecast under one intervention against a number of beds.
#[derive(Debug, Clone, Serialize)]
pub struct EpidemicReport {
    /// The people in the town.
    pub population: f64,
    /// The number of reported days.
    pub days_observed: usize,
    /// The reported cases on each observed day.
    pub observed: Vec<i64>,
    /// Reported cases each day from day 1 to the horizon, as the model predicts them (with reporting noise).
    pub cases: Vec<Interval>,
    /// Beds in use each day from day 1 to the horizon.
    pub occupancy: Vec<Interval>,
    /// The number of beds.
    pub beds: f64,
    /// The intervention's cut in contacts, 0 to 1.
    pub cut: f64,
    /// The intervention's first day.
    pub from_day: usize,
    /// The probability that the beds in use exceed the number of beds on some day.
    pub probability_exceeds: f64,
    /// The highest number of beds in use.
    pub peak_occupancy: Interval,
    /// The day of the highest number of beds in use.
    pub peak_day: Interval,
    /// The distribution of the peak number of beds in use.
    pub peak_histogram: Histogram,
    /// The expected number of days with more patients than beds.
    pub days_over_capacity: f64,
    /// The probability of exceeding the beds without the intervention.
    pub probability_exceeds_without: f64,
    /// The peak beds in use without the intervention.
    pub peak_occupancy_without: Interval,
    /// The share of the town infected by the horizon.
    pub attack_rate: Interval,
    /// The basic reproduction number.
    pub r0: Interval,
    /// The mean infectious period, in days.
    pub recovery_days: Interval,
    /// The share of new infections reported.
    pub reporting: Interval,
    /// The number infected on day 0.
    pub initial: Interval,
    /// The values the synthetic data were generated with.
    pub truth: Outbreak,
    /// The share of infections that need a bed.
    pub hospitalised: f64,
    /// Days from infection to admission.
    pub admission_delay: usize,
    /// The average stay in days.
    pub stay: f64,
    /// The number of posterior draws used.
    pub draws: usize,
    /// Divergent transitions after adaptation.
    pub divergences: usize,
}

impl EpidemicModel {
    /// Starts fitting the reported `cases` with `draws` NUTS draws after as many adaptation steps.
    pub fn new(cases: &[i64], draws: usize, seed: u64) -> keanu::Result<Self> {
        let days = cases.len();
        let b = GraphBuilder::new();
        let r0 = b.log_normal(2f64.ln(), 0.5).label("r0");
        let recovery_days = b.log_normal(7f64.ln(), 0.2).label("recovery_days");
        let reporting = b.beta(12.0, 28.0).label("reporting");
        let initial = b.log_normal(10f64.ln(), 1.0).label("initial");
        let dispersion_inv = b.exponential(5.0);
        b.set_value(r0, DoubleTensor::scalar(2.0));
        b.set_value(recovery_days, DoubleTensor::scalar(7.0));
        b.set_value(reporting, DoubleTensor::scalar(0.3));
        b.set_value(initial, DoubleTensor::scalar(10.0));
        b.set_value(dispersion_inv, DoubleTensor::scalar(0.2));
        let dispersion = dispersion_inv.reciprocal().label("dispersion");

        let g = recovery_days.reciprocal();
        let beta = r0 * g;
        let start = b.stack(0, &[b.scalar(POPULATION) - initial, initial, b.scalar(0.0)]);
        // the carry is [S, I, new infections]
        let states = b.scan_steps(days, start, &[beta, g], |body, y, c| {
            let (s, i) = (y.take(&[0]), y.take(&[1]));
            let (beta, g) = (c[0], c[1]);
            let new = s * (1.0 - (-(beta * i) / POPULATION).exp());
            body.stack(0, &[s - new, i * (1.0 - g) + new, new])
        });
        let new_infections = states.slice(":, 2");
        let reported = b.negative_binomial(reporting * new_infections, dispersion);
        b.observe(reported, IntTensor::vector(cases.to_vec()));

        let nodes = [
            r0.id(),
            recovery_days.id(),
            reporting.id(),
            initial.id(),
            dispersion.id(),
        ];
        let (graph, mut state) = b.build()?;
        let all: Vec<NodeId> = graph.ids().collect();
        state.lazy_eval(&graph, &all, &mut KeanuRng::seed_from_u64(0))?;
        let model = Model::new(graph, &state)?;
        // a dense mass matrix for the tilted ridges (see the module documentation)
        let nuts = Nuts::default()
            .with_potential(QuadraticPotential::default().with_metric(Metric::Dense));
        let chain = ChainRun::new(model, state, &nodes, nuts, draws, draws, seed)?;
        let [r0, recovery_days, reporting, initial, dispersion] = nodes;
        Ok(Self {
            chain,
            cases: cases.to_vec(),
            r0,
            recovery_days,
            reporting,
            initial,
            dispersion,
        })
    }

    /// The chain.
    pub fn chain(&mut self) -> &mut ChainRun {
        &mut self.chain
    }

    /// The posterior draws of the outbreak's parameters.
    pub fn outbreaks(&self) -> Vec<Outbreak> {
        let samples = self.chain.samples();
        let [r0, d, rho, i0, phi] = [
            self.r0,
            self.recovery_days,
            self.reporting,
            self.initial,
            self.dispersion,
        ]
        .map(|node| scalar_draws(samples, node));
        (0..r0.len())
            .map(|k| Outbreak {
                r0: r0[k],
                recovery_days: d[k],
                reporting: rho[k],
                initial: i0[k],
                dispersion: phi[k],
            })
            .collect()
    }

    /// The forecast with `beds` beds under `intervention`.
    pub fn report(&self, beds: f64, intervention: Intervention) -> keanu::Result<EpidemicReport> {
        let outbreaks = self.outbreaks();
        if outbreaks.is_empty() {
            return Err(keanu::Error::from(keanu::graph::GraphError::InvalidValue(
                "the model has no posterior draws yet".into(),
            )));
        }
        let n = outbreaks.len() as f64;
        let mut rng = self.chain.rng_copy();
        let by_day = || -> Vec<Vec<f64>> {
            (0..HORIZON)
                .map(|_| Vec::with_capacity(outbreaks.len()))
                .collect()
        };
        let mut cases_by_day = by_day();
        let mut occupancy_by_day = by_day();
        let (mut peaks, mut peak_days, mut attack) = (vec![], vec![], vec![]);
        let (mut exceeded, mut days_over) = (0usize, 0usize);
        let (mut exceeded_without, mut peaks_without) = (0usize, vec![]);
        for outbreak in &outbreaks {
            let course = outbreak.course(HORIZON, intervention);
            let reported = report_noise(
                &course.new_infections,
                outbreak.reporting,
                outbreak.dispersion,
                &mut rng,
            );
            for (day, (&c, &o)) in reported.iter().zip(&course.occupancy).enumerate() {
                cases_by_day[day].push(c as f64);
                occupancy_by_day[day].push(o);
            }
            let (peak_day, peak) = argmax(&course.occupancy);
            peaks.push(peak);
            peak_days.push((peak_day + 1) as f64);
            exceeded += usize::from(peak > beds);
            days_over += course.occupancy.iter().filter(|&&o| o > beds).count();
            attack.push(course.new_infections.iter().sum::<f64>() / POPULATION);
            let without = outbreak.course(HORIZON, Intervention::NONE);
            let (_, peak) = argmax(&without.occupancy);
            peaks_without.push(peak);
            exceeded_without += usize::from(peak > beds);
        }
        let band = |by_day: &[Vec<f64>]| by_day.iter().map(|v| Interval::of(v)).collect();
        let field =
            |f: fn(&Outbreak) -> f64| Interval::of(&outbreaks.iter().map(f).collect::<Vec<_>>());
        Ok(EpidemicReport {
            population: POPULATION,
            days_observed: self.cases.len(),
            observed: self.cases.clone(),
            cases: band(&cases_by_day),
            occupancy: band(&occupancy_by_day),
            beds,
            cut: intervention.cut,
            from_day: intervention.from_day,
            probability_exceeds: exceeded as f64 / n,
            peak_occupancy: Interval::of(&peaks),
            peak_day: Interval {
                // the median day reads better than the mean of a skewed day count
                mean: quantile(&peak_days, 0.5),
                ..Interval::of(&peak_days)
            },
            peak_histogram: Histogram::of(&peaks, 30),
            days_over_capacity: days_over as f64 / n,
            probability_exceeds_without: exceeded_without as f64 / n,
            peak_occupancy_without: Interval::of(&peaks_without),
            attack_rate: Interval::of(&attack),
            r0: field(|o| o.r0),
            recovery_days: field(|o| o.recovery_days),
            reporting: field(|o| o.reporting),
            initial: field(|o| o.initial),
            truth: TRUTH,
            hospitalised: HOSPITALISED,
            admission_delay: ADMISSION_DELAY,
            stay: STAY,
            draws: outbreaks.len(),
            divergences: self.chain.divergences(),
        })
    }
}

/// The index and value of the largest of `values`.
fn argmax(values: &[f64]) -> (usize, f64) {
    values
        .iter()
        .copied()
        .enumerate()
        .fold((0, f64::NEG_INFINITY), |best, (i, v)| {
            if v > best.1 { (i, v) } else { best }
        })
}
