//! What dose keeps this patient in the safe range? Individual drug dosing from a few blood samples.
//!
//! A fictional drug is taken by mouth. The standard one-compartment model with first-order absorption describes its
//! blood concentration: a dose D moves from the gut into the body (volume V) at the absorption rate kₐ and is cleared
//! at the elimination rate kₑ = CL / V, for clearance CL, so after one dose
//!
//! ```text
//! C(t) = D kₐ / (V (kₐ − kₑ)) · (e^(−kₑ t) − e^(−kₐ t))
//! ```
//!
//! Patients differ, so the population's spread of each parameter is the prior for this patient's, and a handful of
//! blood samples after the first dose, each measured with a proportional assay error, update it:
//!
//! ```text
//! CL ~ LogNormal(ln 5, 0.35)    V ~ LogNormal(ln 50, 0.25)    kₐ ~ LogNormal(ln 1.2, 0.4)
//! C_obs(tᵢ) ~ LogNormal(ln C(tᵢ), 0.15)
//! ```
//!
//! The model samples each parameter's standard-normal deviation from the population median (a non-centred
//! parameterisation), which NUTS handles well. Repeated doses of D every τ hours add up (the model is linear in the
//! dose), and at steady state the geometric series of past doses has a closed form:
//!
//! ```text
//! C_ss(t) = D kₐ / (V (kₐ − kₑ)) · (e^(−kₑ t) / (1 − e^(−kₑ τ)) − e^(−kₐ t) / (1 − e^(−kₐ τ)))    0 ≤ t < τ
//! ```
//!
//! For each posterior draw that profile is computed once for a unit dose and scaled, so a dosing decision (the time
//! in the target range, the chance the trough falls below the minimum effective level or the peak rises above the
//! toxic level, and the dose that keeps the level in range longest) is cheap to recompute for any dose. The same
//! decision from population draws alone shows what the patient's own samples are worth.
//!
//! The drug, its target range and the patient are fictional; the samples are simulated from a seed with known
//! parameters, so the demo can show how well they are recovered. None of this is medical advice.

use crate::run::{ChainRun, Interval, mean, scalar_draws};
use keanu::prelude::*;
use serde::Serialize;

/// A log-normal spread of a parameter across patients: its median and the standard deviation of its logarithm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spread {
    /// The population median.
    pub median: f64,
    /// The standard deviation of the logarithm.
    pub log_sd: f64,
}

/// The population's clearance (L per hour).
pub const POPULATION_CLEARANCE: Spread = Spread {
    median: 5.0,
    log_sd: 0.35,
};
/// The population's volume of distribution (L).
pub const POPULATION_VOLUME: Spread = Spread {
    median: 50.0,
    log_sd: 0.25,
};
/// The population's absorption rate (per hour).
pub const POPULATION_ABSORPTION: Spread = Spread {
    median: 1.2,
    log_sd: 0.4,
};
/// The assay's proportional measurement error, on the log scale.
pub const ASSAY_ERROR: f64 = 0.15;
/// The first dose (mg), after which the samples are taken.
pub const FIRST_DOSE: f64 = 300.0;
/// The sampling times (hours after the first dose).
pub const SAMPLE_HOURS: [f64; 6] = [1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
/// The minimum effective concentration (mg/L).
pub const MIN_EFFECTIVE: f64 = 4.0;
/// The concentration above which the drug is toxic (mg/L).
pub const TOXIC: f64 = 12.0;
/// The doses the suggestion searches (mg): 25 to 1,500 in steps of 25.
pub const DOSE_STEP: f64 = 25.0;
const DOSE_GRID: usize = 60;
/// Points across one dosing interval at which the steady-state profile is evaluated.
const PROFILE_POINTS: usize = 96;
/// The hours after the first dose that the first-dose curve covers.
const FIRST_DOSE_HOURS: f64 = 36.0;
const FIRST_DOSE_POINTS: usize = 72;

/// One patient's pharmacokinetic parameters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Params {
    /// Clearance (L per hour).
    pub clearance: f64,
    /// Volume of distribution (L).
    pub volume: f64,
    /// Absorption rate (per hour).
    pub absorption: f64,
}

/// The synthetic patient's true parameters: they clear the drug more slowly than the typical patient.
pub const TRUE_PATIENT: Params = Params {
    clearance: 3.2,
    volume: 42.0,
    absorption: 0.9,
};

/// The typical patient: the population's median of each parameter.
pub const TYPICAL_PATIENT: Params = Params {
    clearance: POPULATION_CLEARANCE.median,
    volume: POPULATION_VOLUME.median,
    absorption: POPULATION_ABSORPTION.median,
};

impl Params {
    fn elimination(&self) -> f64 {
        self.clearance / self.volume
    }

    /// The elimination half-life (hours).
    pub fn half_life(&self) -> f64 {
        std::f64::consts::LN_2 / self.elimination()
    }

    /// The absorption rate, nudged away from the elimination rate where the closed form is 0/0.
    fn safe_absorption(&self) -> f64 {
        let (ka, ke) = (self.absorption, self.elimination());
        if (ka - ke).abs() < 1e-9 * ke {
            ka * (1.0 + 1e-6)
        } else {
            ka
        }
    }

    /// The concentration (mg/L) `hours` after a single dose of `dose` mg.
    pub fn single_dose(&self, dose: f64, hours: f64) -> f64 {
        let (ka, ke) = (self.safe_absorption(), self.elimination());
        dose * ka / (self.volume * (ka - ke)) * ((-ke * hours).exp() - (-ka * hours).exp())
    }

    /// The concentration (mg/L) at steady state, `hours` (0 to `interval`) after a dose, with `dose` mg given every
    /// `interval` hours.
    pub fn steady_state(&self, dose: f64, interval: f64, hours: f64) -> f64 {
        let (ka, ke) = (self.safe_absorption(), self.elimination());
        dose * ka / (self.volume * (ka - ke))
            * ((-ke * hours).exp() / (1.0 - (-ke * interval).exp())
                - (-ka * hours).exp() / (1.0 - (-ka * interval).exp()))
    }
}

/// A measured blood concentration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Sample {
    /// Hours after the first dose.
    pub hours: f64,
    /// The measured concentration (mg/L).
    pub concentration: f64,
}

/// The synthetic patient's blood samples after [`FIRST_DOSE`], measured with the assay error from `seed`.
pub fn patient_samples(seed: u64) -> Vec<Sample> {
    let mut rng = KeanuRng::seed_from_u64(seed);
    SAMPLE_HOURS
        .iter()
        .map(|&hours| {
            let truth = TRUE_PATIENT.single_dose(FIRST_DOSE, hours);
            let measured = truth * (ASSAY_ERROR * rng.next_gaussian()).exp();
            Sample {
                hours,
                concentration: (measured * 100.0).round() / 100.0,
            }
        })
        .collect()
}

/// `count` patients drawn from the population.
pub fn population_draws(count: usize, seed: u64) -> Vec<Params> {
    let mut rng = KeanuRng::seed_from_u64(seed ^ 0x9e37_79b9);
    let mut draw = |s: Spread| s.median * (s.log_sd * rng.next_gaussian()).exp();
    (0..count)
        .map(|_| Params {
            clearance: draw(POPULATION_CLEARANCE),
            volume: draw(POPULATION_VOLUME),
            absorption: draw(POPULATION_ABSORPTION),
        })
        .collect()
}

/// A fit in progress.
#[derive(Debug)]
pub struct DosingModel {
    chain: ChainRun,
    clearance: NodeId,
    volume: NodeId,
    absorption: NodeId,
    samples: Vec<Sample>,
    population: Vec<Params>,
}

/// A curve over time with a 90% band, and the true patient's curve.
#[derive(Debug, Clone, Serialize)]
pub struct Curve {
    /// Hours (after the first dose, or after a dose at steady state).
    pub hours: Vec<f64>,
    /// The mean and 90% interval of the concentration at each hour.
    pub concentration: Vec<Interval>,
    /// The true patient's concentration at each hour.
    pub truth: Vec<f64>,
}

/// One parameter: what the samples say about this patient, what the population says, and the truth.
#[derive(Debug, Clone, Serialize)]
pub struct ParameterReport {
    /// The parameter's name, in sentence case.
    pub name: &'static str,
    /// Its unit.
    pub unit: &'static str,
    /// The posterior for this patient.
    pub patient: Interval,
    /// The population's spread (the prior).
    pub population: Interval,
    /// The value for the typical patient, whose parameters are the population medians.
    pub typical: f64,
    /// The value the samples were simulated with.
    pub truth: f64,
}

/// How a dosing regimen is expected to work out at steady state.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Outcome {
    /// The dose (mg).
    pub dose: f64,
    /// The expected fraction of the dosing interval with the level in the target range.
    pub time_in_range: f64,
    /// The probability the trough falls below the minimum effective level.
    pub probability_trough_low: f64,
    /// The probability the peak rises above the toxic level.
    pub probability_peak_high: f64,
}

/// The fitted patient and a dosing decision.
#[derive(Debug, Clone, Serialize)]
pub struct DosingReport {
    /// Whether the decision uses the population alone, ignoring the patient's samples.
    pub population_only: bool,
    /// The first dose (mg).
    pub first_dose: f64,
    /// The minimum effective concentration (mg/L).
    pub min_effective: f64,
    /// The toxic concentration (mg/L).
    pub toxic: f64,
    /// The patient's measured samples.
    pub samples: Vec<Sample>,
    /// The concentration after the first dose.
    pub first_dose_curve: Curve,
    /// Clearance, volume, absorption rate and half-life.
    pub parameters: Vec<ParameterReport>,
    /// The dosing interval (hours).
    pub interval: f64,
    /// The chosen dose's outcome.
    pub chosen: Outcome,
    /// The steady-state concentration over one dosing interval of the chosen dose.
    pub steady_state: Curve,
    /// The peak at steady state.
    pub peak: Interval,
    /// The trough at steady state.
    pub trough: Interval,
    /// The fraction of the interval the true patient would spend in range on the chosen dose.
    pub true_time_in_range: f64,
    /// The dose on the grid with the longest expected time in range (the lower one on a tie).
    pub suggested: Outcome,
    /// Every dose on the grid.
    pub doses: Vec<Outcome>,
    /// The number of posterior draws.
    pub draws: usize,
    /// Divergent transitions after adaptation.
    pub divergences: usize,
}

fn evenly(to: f64, points: usize) -> Vec<f64> {
    (0..=points)
        .map(|i| to * i as f64 / points as f64)
        .collect()
}

fn fraction(flags: impl Iterator<Item = bool>) -> f64 {
    let (mut hits, mut total) = (0usize, 0usize);
    for flag in flags {
        hits += usize::from(flag);
        total += 1;
    }
    hits as f64 / total.max(1) as f64
}

fn band(draws: &[Params], hours: &[f64], at: impl Fn(&Params, f64) -> f64) -> Vec<Interval> {
    let mut column = vec![0.0; draws.len()];
    hours
        .iter()
        .map(|&t| {
            for (c, p) in column.iter_mut().zip(draws) {
                *c = at(p, t);
            }
            Interval::of(&column)
        })
        .collect()
}

/// The unit-dose steady-state profile of each draw, at `PROFILE_POINTS` points across the interval, with its
/// minimum and maximum.
struct Profiles {
    unit: Vec<Vec<f64>>,
    low: Vec<f64>,
    high: Vec<f64>,
}

impl Profiles {
    fn new(draws: &[Params], interval: f64) -> Self {
        let hours = evenly(interval, PROFILE_POINTS);
        let unit: Vec<Vec<f64>> = draws
            .iter()
            .map(|p| {
                hours[..PROFILE_POINTS]
                    .iter()
                    .map(|&t| p.steady_state(1.0, interval, t))
                    .collect()
            })
            .collect();
        let low = unit
            .iter()
            .map(|u| u.iter().copied().fold(f64::INFINITY, f64::min))
            .collect();
        let high = unit
            .iter()
            .map(|u| u.iter().copied().fold(f64::NEG_INFINITY, f64::max))
            .collect();
        Self { unit, low, high }
    }

    fn outcome(&self, dose: f64) -> Outcome {
        let in_range = |c: f64| (MIN_EFFECTIVE..=TOXIC).contains(&c);
        let per_draw: Vec<f64> = self
            .unit
            .iter()
            .map(|u| fraction(u.iter().map(|&c| in_range(dose * c))))
            .collect();
        Outcome {
            dose,
            time_in_range: mean(&per_draw),
            probability_trough_low: fraction(self.low.iter().map(|&l| dose * l < MIN_EFFECTIVE)),
            probability_peak_high: fraction(self.high.iter().map(|&h| dose * h > TOXIC)),
        }
    }
}

impl DosingModel {
    /// Starts fitting the patient's `samples` with `draws` NUTS draws after as many adaptation steps. The population
    /// comparison uses as many draws from the population.
    pub fn new(samples: Vec<Sample>, draws: usize, seed: u64) -> keanu::Result<Self> {
        if samples.is_empty()
            || samples
                .iter()
                .any(|s| !(s.concentration > 0.0 && s.hours > 0.0))
        {
            return Err(invalid(
                "every sample needs a positive time and concentration",
            ));
        }
        let b = GraphBuilder::new();
        let parameter = |s: Spread| {
            let z = b.gaussian(0.0, 1.0);
            b.set_value(z, DoubleTensor::scalar(0.0));
            (z * s.log_sd + s.median.ln()).exp()
        };
        let clearance = parameter(POPULATION_CLEARANCE);
        let volume = parameter(POPULATION_VOLUME);
        let absorption = parameter(POPULATION_ABSORPTION);
        let elimination = clearance / volume;
        let hours = b.constant(DoubleTensor::vector(
            samples.iter().map(|s| s.hours).collect(),
        ));
        let concentration = absorption * FIRST_DOSE / (volume * (absorption - elimination))
            * ((-(elimination * hours)).exp() - (-(absorption * hours)).exp());
        let measured = b.log_normal(concentration.ln(), ASSAY_ERROR);
        b.observe(
            measured,
            DoubleTensor::vector(samples.iter().map(|s| s.concentration).collect()),
        );
        let (clearance, volume, absorption) = (clearance.id(), volume.id(), absorption.id());
        let (graph, state) = b.build()?;
        let model = Model::new(graph, &state)?;
        let chain = ChainRun::new(
            model,
            state,
            &[clearance, volume, absorption],
            Nuts::default(),
            draws,
            draws,
            seed,
        )?;
        Ok(Self {
            chain,
            clearance,
            volume,
            absorption,
            samples,
            population: population_draws(draws, seed),
        })
    }

    /// The chain.
    pub fn chain(&mut self) -> &mut ChainRun {
        &mut self.chain
    }

    /// The patient's posterior draws.
    pub fn posterior(&self) -> Vec<Params> {
        let samples = self.chain.samples();
        let clearance = scalar_draws(samples, self.clearance);
        let volume = scalar_draws(samples, self.volume);
        let absorption = scalar_draws(samples, self.absorption);
        clearance
            .into_iter()
            .zip(volume)
            .zip(absorption)
            .map(|((clearance, volume), absorption)| Params {
                clearance,
                volume,
                absorption,
            })
            .collect()
    }

    /// The fit, and the decision for `dose` mg every `interval` hours, from the patient's posterior or, with
    /// `population_only`, from the population alone.
    pub fn report(
        &self,
        dose: f64,
        interval: f64,
        population_only: bool,
    ) -> keanu::Result<DosingReport> {
        if !(dose > 0.0 && dose.is_finite()) {
            return Err(invalid("the dose must be a positive number of mg"));
        }
        if !(1.0..=72.0).contains(&interval) {
            return Err(invalid(
                "the dosing interval must be between 1 and 72 hours",
            ));
        }
        let posterior = self.posterior();
        if posterior.is_empty() {
            return Err(invalid("the model has no posterior draws yet"));
        }
        let draws: &[Params] = if population_only {
            &self.population
        } else {
            &posterior
        };

        let first_hours = evenly(FIRST_DOSE_HOURS, FIRST_DOSE_POINTS);
        let first_dose_curve = Curve {
            concentration: band(draws, &first_hours, |p, t| p.single_dose(FIRST_DOSE, t)),
            truth: first_hours
                .iter()
                .map(|&t| TRUE_PATIENT.single_dose(FIRST_DOSE, t))
                .collect(),
            hours: first_hours,
        };

        let summary = |values: &dyn Fn(&Params) -> f64, of: &[Params]| {
            Interval::of(&of.iter().map(values).collect::<Vec<_>>())
        };
        let parameters = [
            (
                "Clearance",
                "L/h",
                &(|p: &Params| p.clearance) as &dyn Fn(&Params) -> f64,
            ),
            ("Volume", "L", &|p: &Params| p.volume),
            ("Absorption rate", "per hour", &|p: &Params| p.absorption),
            ("Half-life", "hours", &|p: &Params| p.half_life()),
        ]
        .into_iter()
        .map(|(name, unit, value)| ParameterReport {
            name,
            unit,
            patient: summary(value, &posterior),
            population: summary(value, &self.population),
            typical: value(&TYPICAL_PATIENT),
            truth: value(&TRUE_PATIENT),
        })
        .collect();

        let profiles = Profiles::new(draws, interval);
        let chosen = profiles.outcome(dose);
        let doses: Vec<Outcome> = (1..=DOSE_GRID)
            .map(|i| profiles.outcome(i as f64 * DOSE_STEP))
            .collect();
        let suggested = suggest(&doses);

        let ss_hours = evenly(interval, PROFILE_POINTS);
        let mut column = vec![0.0; draws.len()];
        let mut concentration: Vec<Interval> = (0..PROFILE_POINTS)
            .map(|j| {
                for (c, u) in column.iter_mut().zip(&profiles.unit) {
                    *c = dose * u[j];
                }
                Interval::of(&column)
            })
            .collect();
        // the level at the end of the interval is the level at its start
        concentration.push(concentration[0]);
        let steady_state = Curve {
            concentration,
            truth: ss_hours
                .iter()
                .map(|&t| TRUE_PATIENT.steady_state(dose, interval, t % interval))
                .collect(),
            hours: ss_hours,
        };
        let true_profile = Profiles::new(&[TRUE_PATIENT], interval);
        let peaks: Vec<f64> = profiles.high.iter().map(|h| dose * h).collect();
        let troughs: Vec<f64> = profiles.low.iter().map(|l| dose * l).collect();
        Ok(DosingReport {
            population_only,
            first_dose: FIRST_DOSE,
            min_effective: MIN_EFFECTIVE,
            toxic: TOXIC,
            samples: self.samples.clone(),
            first_dose_curve,
            parameters,
            interval,
            chosen,
            steady_state,
            peak: Interval::of(&peaks),
            trough: Interval::of(&troughs),
            true_time_in_range: true_profile.outcome(dose).time_in_range,
            suggested,
            doses,
            draws: draws.len(),
            divergences: self.chain.divergences(),
        })
    }
}

/// The dose with the longest expected time in range; among doses within one percentage point of it, the one with the
/// lowest total risk of a low trough or a high peak, then the lowest dose.
fn suggest(doses: &[Outcome]) -> Outcome {
    let best = doses
        .iter()
        .map(|o| o.time_in_range)
        .fold(f64::NEG_INFINITY, f64::max);
    let risk = |o: &Outcome| o.probability_trough_low + o.probability_peak_high;
    doses
        .iter()
        .copied()
        .filter(|o| o.time_in_range >= best - 0.01)
        .fold(None, |chosen: Option<Outcome>, o| match chosen {
            Some(c) if risk(&c) <= risk(&o) + 1e-12 => Some(c),
            _ => Some(o),
        })
        .unwrap_or(doses[0])
}

fn invalid(message: &str) -> keanu::Error {
    keanu::Error::from(keanu::graph::GraphError::InvalidValue(message.into()))
}
