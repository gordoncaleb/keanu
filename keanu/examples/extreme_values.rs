//! How high will the river rise once in a hundred years?
//!
//! Annual maxima (the largest flow in each year) follow, approximately, a generalized extreme value distribution
//! GEV(μ, σ, ξ) by the Fisher–Tippett–Gnedenko theorem, whatever the daily flows' distribution. The T-year return
//! level z_T is the level exceeded in any one year with probability 1/T, the GEV's 1 − 1/T quantile:
//!
//! z_T = μ + σ ((−ln(1 − 1/T))^(−ξ) − 1) / ξ,
//!
//! (μ − σ ln(−ln(1 − 1/T)) in the limit ξ → 0). A positive shape ξ gives a heavy upper tail (the Fréchet type),
//! so the return levels of rare events grow quickly with T and are uncertain mainly through ξ.
//!
//! Sixty years of annual maxima are simulated from known values ([`TRUE_MU`], [`TRUE_SIGMA`], [`TRUE_XI`]), so
//! the fit can be checked against them. The priors are weakly informative: μ ~ N(30, 20), σ ~ half-N(20) and
//! ξ ~ N(0, 0.3), the last keeping ξ in the range seen in hydrology. The return levels are deterministic nodes of
//! the parameters, so each posterior draw gives a draw of each return level. The GEV's support depends on its
//! parameters (x > μ − σ/ξ for ξ > 0), so parameters that leave a datum outside it have log probability −∞ and
//! NUTS rejects them. Run with `cargo run --release --example extreme_values`.

use keanu::algorithms::ChainSamples;
use keanu::algorithms::mcmc::chains::sample_chains;
use keanu::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The number of years of annual maxima.
pub const YEARS: usize = 60;
/// The location the data are simulated with (a flow, in m³/s).
pub const TRUE_MU: f64 = 30.0;
/// The scale the data are simulated with.
pub const TRUE_SIGMA: f64 = 6.0;
/// The shape the data are simulated with.
pub const TRUE_XI: f64 = 0.15;
/// The return periods, in years.
pub const PERIODS: [f64; 3] = [10.0, 50.0, 100.0];
/// The seed the example's data are simulated with.
pub const DATA_SEED: u64 = 1;

/// The T-year return level of a GEV(μ, σ, ξ): its 1 − 1/T quantile.
pub fn return_level(mu: f64, sigma: f64, xi: f64, period: f64) -> f64 {
    let y = -(1.0 - 1.0 / period).ln();
    if xi.abs() < 1e-12 {
        mu - sigma * y.ln()
    } else {
        mu + sigma * ((-xi * y.ln()).exp_m1()) / xi
    }
}

/// Simulates the annual maxima by inverting the GEV's CDF, exp(−(1 + ξ(x − μ)/σ)^(−1/ξ)).
pub fn data(seed: u64) -> Vec<f64> {
    let mut rng = KeanuRng::seed_from_u64(seed);
    (0..YEARS)
        .map(|_| return_level(TRUE_MU, TRUE_SIGMA, TRUE_XI, 1.0 / (1.0 - rng.next_f64())))
        .collect()
}

/// A fit: every chain's draws of the parameters and the return levels (one per period in [`PERIODS`]).
#[derive(Debug)]
pub struct Fit {
    pub chains: ChainSamples,
    pub mu: NodeId,
    pub sigma: NodeId,
    pub xi: NodeId,
    pub return_levels: NodeId,
    /// The number of divergent transitions after warm-up, over every chain.
    pub divergences: usize,
}

/// Fits the GEV by NUTS, with `chains` chains of `draws` draws after as many warm-up steps.
pub fn run(maxima: &[f64], chains: usize, draws: usize, seed: u64) -> keanu::Result<Fit> {
    let b = GraphBuilder::new();
    let mu = b.gaussian(30.0, 20.0).label("mu");
    let sigma = b.half_gaussian(20.0).label("sigma");
    let xi = b.gaussian(0.0, 0.3).label("xi");
    let flow = b.generalized_extreme_value_with_shape(&[maxima.len()], mu, sigma, xi);
    b.observe(flow, DoubleTensor::vector(maxima.to_vec()));
    // z_T = μ + σ expm1(−ξ ln y_T) / ξ with y_T = −ln(1 − 1/T), for every period at once
    let log_y = b.constant(DoubleTensor::vector(
        PERIODS
            .iter()
            .map(|t| (-(1.0 - 1.0 / t).ln()).ln())
            .collect(),
    ));
    let return_levels = (mu + sigma * (-xi * log_y).exp_m1() / xi).label("return_levels");
    // start at the data's location and spread, with a light tail
    let mean = maxima.iter().sum::<f64>() / maxima.len() as f64;
    b.set_value(mu, DoubleTensor::scalar(mean));
    b.set_value(sigma, DoubleTensor::scalar(5.0));
    b.set_value(xi, DoubleTensor::scalar(0.05));
    let (mu, sigma, xi, return_levels) = (mu.id(), sigma.id(), xi.id(), return_levels.id());
    let (graph, state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let nuts = Nuts::default().with_adapt_count(draws);
    let divergences = AtomicUsize::new(0);
    let chains = sample_chains(chains, seed, &state, |state, rng| {
        nuts.sample_with_statistics(
            &model,
            state,
            &[mu, sigma, xi, return_levels],
            2 * draws,
            SamplingOptions::default().with_drop(draws),
            rng,
        )
        .map(|(samples, statistics)| {
            divergences.fetch_add(statistics.divergent_steps.len(), Ordering::Relaxed);
            samples
        })
    })?;
    Ok(Fit {
        chains,
        mu,
        sigma,
        xi,
        return_levels,
        divergences: divergences.into_inner(),
    })
}

#[allow(dead_code)]
fn main() -> keanu::Result<()> {
    let maxima = data(DATA_SEED);
    let largest = maxima.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    println!("{YEARS} annual maxima, the largest {largest:.1}");
    let fit = run(&maxima, 4, 1000, 1)?;
    println!("{} divergent transitions", fit.divergences);
    let report = |name: String, node: NodeId, index: &[usize], truth: f64| -> keanu::Result<()> {
        let s = fit.chains.summarize(node, index)?;
        println!(
            "{name}: mean {:.3}, 90% interval [{:.3}, {:.3}], R-hat {:.3} (truth {truth:.3})",
            s.mean, s.q5, s.q95, s.r_hat
        );
        Ok(())
    };
    report("mu".into(), fit.mu, &[], TRUE_MU)?;
    report("sigma".into(), fit.sigma, &[], TRUE_SIGMA)?;
    report("xi".into(), fit.xi, &[], TRUE_XI)?;
    for (i, &period) in PERIODS.iter().enumerate() {
        report(
            format!("{period}-year return level"),
            fit.return_levels,
            &[i],
            return_level(TRUE_MU, TRUE_SIGMA, TRUE_XI, period),
        )?;
    }
    Ok(())
}
