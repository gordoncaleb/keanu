//! How volatile is an asset from one day to the next, and how persistent is its volatility?
//!
//! The stochastic volatility model of the Stan User's Guide (§2.5) and the PyMC example: the daily return r_t has
//! mean 0 and standard deviation exp(h_t / 2), and the log variance h_t follows a stationary AR(1) process around a
//! level mu with persistence phi and innovation scale sigma:
//!
//! ```text
//! mu ~ N(0, 10)    (phi + 1) / 2 ~ Beta(20, 1.5)    sigma ~ HalfCauchy(5)
//! h_0 = mu + sigma / sqrt(1 − phi²) · eps_0    (the day before the first return)
//! h_t = mu + phi (h_{t−1} − mu) + sigma · eps_t    eps_t ~ N(0, 1)    for t = 1, ..., T
//! r_t ~ N(0, exp(h_t / 2))
//! ```
//!
//! The innovations are non-centred: the latents are the standard normal eps (one scalar for the stationary start
//! and one vector of length T), and h is a deterministic function of them and the parameters. The recursion is a
//! scan (see "Scan" in the book's Sequences chapter): one node holding the loop body, so the graph stays the same
//! size whatever T is. The body is written as h_t = mu (1 − phi) + phi h_{t−1} + (sigma eps)_t, the same recursion
//! with fewer operations per step: the level mu (1 − phi) is a constant of the loop and the scaled innovations
//! sigma · eps are one vector operation outside it, passed in as the per-step input.
//!
//! Even non-centred, the posterior has an awkward corner: as phi approaches 1 the level mu and the slow movements
//! of h trade off against each other along a curved ridge, which NUTS's diagonal mass matrix fits poorly. A target
//! acceptance probability of 0.95 (smaller steps) keeps the chains mixing there, but a few trajectories in a
//! thousand still diverge, and raising it to 0.99 doesn't remove them. They come from the posterior's shape, not from
//! how NUTS tunes itself: near phi = 1 the slow movements of h are set by many innovations together, so the
//! curvature along that direction grows like 1 / (1 − phi)² and no single step size suits it; longer adaptation
//! doesn't remove them either. The same corner keeps NUTS's step size near 0.1, so most trajectories take 31 to 63
//! leapfrog steps. The example reports the count; its test checks that it stays below 1% of the draws.
//!
//! The Stan User's Guide's own form, with no day-0 state (h_1 = mu + sigma / sqrt(1 − phi²) · eps_1 and the
//! recursion from day 2), doesn't remove them either. Measured on 2026-11-05 with 4 chains of 500 draws, over seeds
//! 1 to 8: this example (target 0.95) had 2, 2, 2, 1, 1, 1, 0 and 44 divergences; Stan's form with this example's
//! priors had 4, 215, 2, 32, 3, 6, 4, 3 at target 0.8 and 5, 0, 2, 3, 9, 0, 3, 1 at 0.95; and with Stan's priors
//! (phi ~ Uniform(−1, 1), mu ~ Cauchy(0, 10)) 1, 91, 5, 4, 6, 2, 62, 28 at 0.8 and 2, 11, 9, 0, 0, 0, 1, 1 at 0.95.
//!
//! Data: `data/returns.csv` holds T = 200 daily returns simulated from the model with mu = −1, phi = 0.9 and
//! sigma = 0.4, by [`simulate`] with seed [`DATA_SEED`]. Its columns are the day, the true log variance h_t and the
//! return r_t; the example's test checks that [`simulate`] still reproduces the file (regenerate it with
//! `KEANU_REGENERATE=1 cargo test -p keanu --test examples stochastic_volatility_data`).
//!
//! Run with `cargo run --release --example stochastic_volatility`.

use keanu::algorithms::ChainSamples;
use keanu::algorithms::mcmc::chains::run_chains;
use keanu::prelude::*;

/// The level of the log variance the data were simulated with.
pub const TRUE_MU: f64 = -1.0;
/// The persistence the data were simulated with.
pub const TRUE_PHI: f64 = 0.9;
/// The innovation scale the data were simulated with.
pub const TRUE_SIGMA: f64 = 0.4;
/// The number of days in `data/returns.csv`.
pub const DAYS: usize = 200;
/// The seed [`simulate`] was given to write `data/returns.csv`.
pub const DATA_SEED: u64 = 2;

/// The number of chains.
pub const CHAINS: usize = 4;

/// A simulated series: each day's log variance and return.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Series {
    /// The log variance h_t of each day.
    pub log_variance: Vec<f64>,
    /// The return r_t of each day.
    pub returns: Vec<f64>,
}

/// Simulates `days` days of the model with the true parameters, from a generator seeded with `seed`. The first day's
/// log variance is drawn from the stationary distribution of the AR(1) process.
pub fn simulate(days: usize, seed: u64) -> Series {
    let mut rng = KeanuRng::seed_from_u64(seed);
    let stationary_sd = TRUE_SIGMA / (1.0 - TRUE_PHI * TRUE_PHI).sqrt();
    let mut h = TRUE_MU + stationary_sd * rng.next_gaussian();
    let mut series = Series::default();
    for _ in 0..days {
        h = TRUE_MU + TRUE_PHI * (h - TRUE_MU) + TRUE_SIGMA * rng.next_gaussian();
        series.log_variance.push(h);
        series.returns.push((h / 2.0).exp() * rng.next_gaussian());
    }
    series
}

/// The series in the bundled CSV.
pub fn data() -> Series {
    let mut series = Series::default();
    for line in include_str!("data/returns.csv").lines().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        let [_day, log_variance, ret] = fields[..] else {
            panic!("expected day,log_variance,return: {line}");
        };
        series
            .log_variance
            .push(log_variance.trim().parse().unwrap());
        series.returns.push(ret.trim().parse().unwrap());
    }
    series
}

/// The series as CSV, as written to `data/returns.csv`.
pub fn to_csv(series: &Series) -> String {
    let mut csv = String::from("day,log_variance,return\n");
    for (t, (h, r)) in series.log_variance.iter().zip(&series.returns).enumerate() {
        csv.push_str(&format!("{t},{h:.12},{r:.12}\n"));
    }
    csv
}

/// The posterior draws and where to find each parameter in them.
#[derive(Debug)]
pub struct StochasticVolatility {
    /// The draws of every chain.
    pub samples: ChainSamples,
    /// The number of divergent transitions after adaptation, over all chains.
    pub divergences: usize,
    /// The level of the log variance.
    pub mu: NodeId,
    /// The persistence of the log variance.
    pub phi: NodeId,
    /// The scale of the log variance's innovations.
    pub sigma: NodeId,
    /// The log variance of each day, shape [T].
    pub log_variance: NodeId,
}

/// Fits the model to `returns` with `chains` NUTS chains of `draws` draws each, after as many adaptation steps.
pub fn fit(
    returns: &[f64],
    chains: usize,
    seed: u64,
    draws: usize,
) -> keanu::Result<StochasticVolatility> {
    let days = returns.len();
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 10.0).label("mu");
    // phi on (−1, 1), with most of its prior mass near 1 as for daily volatility
    let phi_raw = b.beta(20.0, 1.5);
    let phi = (phi_raw * 2.0 - 1.0).label("phi");
    let sigma = b.half_cauchy(5.0).label("sigma");
    // the non-centred innovations: one for the stationary start and one per day
    let eps_0 = b.gaussian(0.0, 1.0);
    let eps = b.gaussian_with_shape(&[days], 0.0, 1.0);
    b.set_value(mu, DoubleTensor::scalar(0.0));
    b.set_value(phi_raw, DoubleTensor::scalar(0.9));
    b.set_value(sigma, DoubleTensor::scalar(0.5));
    b.set_value(eps_0, DoubleTensor::scalar(0.0));
    b.set_value(eps, DoubleTensor::vector(vec![0.0; days]));

    let h_0 = mu + sigma / (1.0 - phi * phi).sqrt() * eps_0;
    // h_t = mu + phi (h_{t−1} − mu) + sigma eps_t, written as level + phi h_{t−1} + shock_t
    let level = mu * (1.0 - phi);
    let shocks = sigma * eps;
    let h = b
        .scan(h_0, &[shocks], &[level, phi], |_, h, shock, c| {
            let (level, phi) = (c[0], c[1]);
            level + phi * h + shock[0]
        })
        .label("log_variance");
    let r = b.gaussian_with_shape(&[days], 0.0, (h * 0.5).exp());
    b.observe(r, DoubleTensor::vector(returns.to_vec()));
    let (mu, phi, sigma, h) = (mu.id(), phi.id(), sigma.id(), h.id());

    let (graph, state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let adapt = draws;
    let nuts = Nuts::default()
        .with_adapt_count(adapt)
        .with_target_acceptance_prob(0.95);
    let chains = run_chains(chains, seed, &state, |state, rng| {
        nuts.sample_with_statistics(
            &model,
            state,
            &[mu, phi, sigma, h],
            adapt + draws,
            SamplingOptions::default().with_drop(adapt),
            rng,
        )
    })?;
    let divergences = chains.iter().map(|(_, s)| s.divergence_count()).sum();
    let samples = ChainSamples::new(chains.into_iter().map(|(samples, _)| samples).collect())?;
    Ok(StochasticVolatility {
        samples,
        divergences,
        mu,
        phi,
        sigma,
        log_variance: h,
    })
}

/// Fits the model to the bundled returns with `CHAINS` chains of `draws` draws each.
pub fn run(seed: u64, draws: usize) -> keanu::Result<StochasticVolatility> {
    fit(&data().returns, CHAINS, seed, draws)
}

/// The number of draws per chain (after as many adaptation steps).
pub const DEFAULT_SAMPLES: usize = 1000;

#[allow(dead_code)]
fn main() -> keanu::Result<()> {
    let series = data();
    let r = run(1, DEFAULT_SAMPLES)?;
    println!("{} divergences", r.divergences);
    for (name, node, truth) in [
        ("mu", r.mu, TRUE_MU),
        ("phi", r.phi, TRUE_PHI),
        ("sigma", r.sigma, TRUE_SIGMA),
    ] {
        let s = r.samples.summarize(node, &[])?;
        println!(
            "{name:<6} mean {:>7.3}  sd {:.3}  (true {truth:>5.2})  R-hat {:.3}  bulk ESS {:.0}",
            s.mean, s.sd, s.r_hat, s.ess_bulk
        );
    }
    println!("\nday  true h   posterior h (mean ± sd)");
    for t in (0..series.returns.len()).step_by(30) {
        let s = r.samples.summarize(r.log_variance, &[t])?;
        println!(
            "{t:>3}  {:>6.2}   {:>6.2} ± {:.2}",
            series.log_variance[t], s.mean, s.sd
        );
    }
    Ok(())
}
