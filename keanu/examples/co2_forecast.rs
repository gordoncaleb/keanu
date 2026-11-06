//! What will the atmospheric CO₂ concentration be over the next two years, and how much of each month's value is
//! trend and how much season?
//!
//! A structural time-series model in the manner of TensorFlow Probability's Mauna Loa CO₂ example: a local linear
//! trend (a level whose slope drifts slowly) plus a 12-month seasonal cycle plus observation noise,
//!
//! ```text
//! level_{t+1} = level_t + slope_t + N(0, level_scale²)
//! slope_{t+1} = slope_t + N(0, slope_scale²)
//! (c_j, s_j)_{t+1} = R(2πj / 12) (c_j, s_j)_t + N(0, drift_scale² I)    for the harmonics j = 1, 2
//! y_t = level_t + c_{1,t} + c_{2,t} + N(0, observation_noise_scale²)
//! ```
//!
//! built with `keanu::graph::sts` into one linear-Gaussian state-space node, so the six states (level, slope and a
//! rotating pair per harmonic) are integrated out by the Kalman filter and NUTS samples only the four scales, with
//! the builder's default priors (LogNormal around 5% of the series' standard deviation for the component scales and
//! 10% for the noise). After sampling, `keanu::algorithms::sts::StructuralPosterior` decomposes the series into its
//! trend and seasonal parts and forecasts it 24 months ahead, both over the posterior draws, with intervals that
//! include the scales' uncertainty as well as the states'.
//!
//! The seasonal cycle is the trigonometric form with two harmonics (a yearly sinusoid and its first overtone, which
//! gives the cycle its asymmetric shape). TFP's example uses the dummy-variable form, `Component::seasonal(12)`,
//! which works the same way here but has 11 seasonal states instead of 4: each Kalman step then costs about three
//! times as much, for a cycle that a smooth curve describes as well.
//!
//! Data: simulated, shaped like the monthly Mauna Loa record of the 1960s (about 315 ppm rising by about 0.1 ppm a
//! month, a seasonal cycle of about ±3 ppm peaking in May, and a few tenths of a ppm of noise), from the model
//! itself with the scales below, so that the forecast can be checked against the held-out truth: [`simulate`] draws
//! [`TRAINING_MONTHS`] + [`HORIZON`] months from seed [`DATA_SEED`], the model is fitted to the first
//! [`TRAINING_MONTHS`] and the last [`HORIZON`] are compared with the forecast.
//!
//! Run with `cargo run --release --example co2_forecast`.

use keanu::algorithms::ChainSamples;
use keanu::algorithms::mcmc::chains::run_chains;
use keanu::algorithms::sts::{Decomposition, Forecast, StructuralPosterior};
use keanu::graph::sts::{Component, StructuralModel};
use keanu::prelude::*;

/// The level scale the data are simulated with (ppm per month).
pub const TRUE_LEVEL_SCALE: f64 = 0.05;
/// The slope scale the data are simulated with (ppm per month per month).
pub const TRUE_SLOPE_SCALE: f64 = 0.004;
/// The seasonal drift scale the data are simulated with (ppm).
pub const TRUE_DRIFT_SCALE: f64 = 0.04;
/// The observation noise scale the data are simulated with (ppm).
pub const TRUE_NOISE_SCALE: f64 = 0.2;
/// The months the model is fitted to (8 years).
pub const TRAINING_MONTHS: usize = 96;
/// The months forecast and held out (2 years).
pub const HORIZON: usize = 24;
/// The seed of the simulated series.
pub const DATA_SEED: u64 = 3;
/// The number of chains.
pub const CHAINS: usize = 4;
/// The probability of the reported intervals.
pub const LEVEL: f64 = 0.9;
/// The seasonal harmonics.
pub const HARMONICS: usize = 2;

/// Simulates `months` months of the model with the true scales, starting at 315 ppm rising 0.1 ppm a month, with
/// a seasonal cycle of a yearly harmonic of amplitude 3 ppm peaking in May (month 4 of each year, counting from 0)
/// and a second harmonic of amplitude 0.8 ppm.
pub fn simulate(months: usize, seed: u64) -> Vec<f64> {
    use std::f64::consts::PI;
    let mut rng = KeanuRng::seed_from_u64(seed);
    let (mut level, mut slope) = (315.0, 0.1);
    // each harmonic's (cosine, sine) pair: its contribution is the cosine, A cos(λ (t − peak))
    let mut pairs: Vec<(f64, f64)> = [(3.0, 4.0), (0.8, 1.0)]
        .iter()
        .enumerate()
        .map(|(j, &(amplitude, peak))| {
            let phase = -2.0 * PI * (j + 1) as f64 * peak / 12.0;
            (amplitude * phase.cos(), -amplitude * phase.sin())
        })
        .collect();
    let mut y = Vec::with_capacity(months);
    for _ in 0..months {
        let season: f64 = pairs.iter().map(|(c, _)| c).sum();
        y.push(level + season + TRUE_NOISE_SCALE * rng.next_gaussian());
        level += slope + TRUE_LEVEL_SCALE * rng.next_gaussian();
        slope += TRUE_SLOPE_SCALE * rng.next_gaussian();
        for (j, (c, s)) in pairs.iter_mut().enumerate() {
            let (sin, cos) = (2.0 * PI * (j + 1) as f64 / 12.0).sin_cos();
            let next = (
                cos * *c + sin * *s + TRUE_DRIFT_SCALE * rng.next_gaussian(),
                -sin * *c + cos * *s + TRUE_DRIFT_SCALE * rng.next_gaussian(),
            );
            (*c, *s) = next;
        }
    }
    y
}

/// The fitted model: its draws, the decomposition of the training months and the forecast.
#[derive(Debug)]
pub struct Co2Forecast {
    /// The draws of the scales, every chain.
    pub samples: ChainSamples,
    /// The number of divergent transitions after adaptation, over all chains.
    pub divergences: usize,
    /// The scales' nodes, by name (`"trend.level_scale"`, ...).
    pub parameters: Vec<(String, NodeId)>,
    /// The trend and seasonal parts of the training months.
    pub decomposition: Decomposition,
    /// The next [`HORIZON`] months.
    pub forecast: Forecast,
    /// The model at each draw, for other decompositions and forecasts.
    pub posterior: StructuralPosterior,
}

/// Fits the model to `series` with `chains` NUTS chains of `draws` draws each (after as many adaptation steps),
/// then decomposes it and forecasts `horizon` months, with intervals of probability [`LEVEL`].
pub fn fit(
    series: &[f64],
    horizon: usize,
    chains: usize,
    seed: u64,
    draws: usize,
) -> keanu::Result<Co2Forecast> {
    let b = GraphBuilder::new();
    let sts = StructuralModel::new(&b, series)
        .with(Component::local_linear_trend())
        .with(Component::trigonometric_seasonal(12.0, HARMONICS))
        .build()?;
    let parameters = sts.parameters();
    let nodes: Vec<NodeId> = parameters.iter().map(|(_, id)| *id).collect();
    let (graph, state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let nuts = Nuts::default().with_adapt_count(draws);
    let runs = run_chains(chains, seed, &state, |state, rng| {
        nuts.sample_with_statistics(
            &model,
            state,
            &nodes,
            2 * draws,
            SamplingOptions::default().with_drop(draws),
            rng,
        )
    })?;
    let divergences = runs.iter().map(|(_, s)| s.divergence_count()).sum();
    let samples = ChainSamples::new(runs.into_iter().map(|(samples, _)| samples).collect())?;
    let posterior = StructuralPosterior::from_chains(&model, &state, &sts, &samples)?;
    let decomposition = posterior.decompose(LEVEL)?;
    let forecast = posterior.forecast(horizon, &[], LEVEL)?;
    Ok(Co2Forecast {
        samples,
        divergences,
        parameters,
        decomposition,
        forecast,
        posterior,
    })
}

/// Simulates the series, fits the first [`TRAINING_MONTHS`] months with [`CHAINS`] chains of `draws` draws and
/// forecasts the [`HORIZON`] held-out months. Returns the fit and the held-out months.
pub fn run(seed: u64, draws: usize) -> keanu::Result<(Co2Forecast, Vec<f64>)> {
    let series = simulate(TRAINING_MONTHS + HORIZON, DATA_SEED);
    let (training, held_out) = series.split_at(TRAINING_MONTHS);
    Ok((
        fit(training, HORIZON, CHAINS, seed, draws)?,
        held_out.to_vec(),
    ))
}

/// How many of `values` lie in their step's interval of `forecast` (the series' interval, with noise).
pub fn covered(forecast: &Forecast, values: &[f64]) -> usize {
    let o = &forecast.observations;
    values
        .iter()
        .enumerate()
        .filter(|&(k, v)| o.lower[k] <= *v && *v <= o.upper[k])
        .count()
}

/// The number of draws per chain (after as many adaptation steps).
pub const DEFAULT_SAMPLES: usize = 1000;

#[allow(dead_code)]
fn main() -> keanu::Result<()> {
    let (r, held_out) = run(1, DEFAULT_SAMPLES)?;
    println!("{} divergences", r.divergences);
    let truths = [
        TRUE_LEVEL_SCALE,
        TRUE_SLOPE_SCALE,
        TRUE_DRIFT_SCALE,
        TRUE_NOISE_SCALE,
    ];
    for ((name, node), truth) in r.parameters.iter().zip(truths) {
        let s = r.samples.summarize(*node, &[])?;
        println!(
            "{name:<24} mean {:.4}  sd {:.4}  (true {truth:.3})  R-hat {:.3}",
            s.mean, s.sd, s.r_hat
        );
    }
    let trend = r.decomposition.component("trend").unwrap();
    let seasonal = r.decomposition.component("seasonal").unwrap();
    println!("\nthe last year of the training months: trend + seasonal");
    for t in TRAINING_MONTHS - 12..TRAINING_MONTHS {
        println!(
            "month {t:>3}  trend {:>7.2} ({:.2} to {:.2})  seasonal {:>5.2} ({:.2} to {:.2})",
            trend.mean[t],
            trend.lower[t],
            trend.upper[t],
            seasonal.mean[t],
            seasonal.lower[t],
            seasonal.upper[t]
        );
    }
    println!("\nforecast (90% interval) and the held-out month");
    let o = &r.forecast.observations;
    for (k, truth) in held_out.iter().enumerate() {
        println!(
            "month {:>3}  {:>7.2} ({:.2} to {:.2})  held out {truth:.2}",
            TRAINING_MONTHS + k,
            o.mean[k],
            o.lower[k],
            o.upper[k]
        );
    }
    println!(
        "{} of {HORIZON} held-out months in their 90% interval",
        covered(&r.forecast, &held_out)
    );
    Ok(())
}
