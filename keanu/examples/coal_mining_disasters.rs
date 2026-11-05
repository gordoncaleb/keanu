//! When did the rate of coal mining disasters change?
//!
//! Each year's disaster count is Poisson, with one rate before an unknown switch year and another after it. The
//! model has a discrete latent (the switch year) and two continuous ones (the rates), so the default sampler is the
//! compound sampler: Gibbs sampling of the switch year, by enumerating every year, and NUTS for the rates. Four
//! chains are run and checked with R-hat.
//!
//! Run with `cargo run --release --example coal_mining_disasters`.

use keanu::algorithms::ChainSamples;
use keanu::algorithms::mcmc::chains::sample_chains;
use keanu::prelude::*;

/// (year, disasters) pairs from the bundled CSV.
pub fn data() -> Vec<(i64, i64)> {
    include_str!("data/coal-mining-disaster-data.csv")
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let (year, count) = line.split_once(',').expect("year,count");
            (year.trim().parse().unwrap(), count.trim().parse().unwrap())
        })
        .collect()
}

/// What the example reports.
#[derive(Debug)]
pub struct Results {
    /// The draws of every chain.
    pub chains: ChainSamples,
    /// The switch year: the first year with the late rate.
    pub switch_point: NodeId,
    /// The rate before the switch.
    pub early_rate: NodeId,
    /// The rate from the switch on.
    pub late_rate: NodeId,
    /// The most frequently sampled switch year.
    pub switch_year_mode: i64,
}

/// The number of chains.
pub const CHAINS: usize = 4;

/// Samples the model with `CHAINS` chains of `draws` draws each, after the default sampler's 1000 adaptation steps.
pub fn run(seed: u64, draws: usize) -> keanu::Result<Results> {
    let data = data();
    let years: Vec<i64> = data.iter().map(|&(year, _)| year).collect();
    let disasters: Vec<i64> = data.iter().map(|&(_, count)| count).collect();
    let (start, end) = (*years.iter().min().unwrap(), *years.iter().max().unwrap());

    let b = GraphBuilder::new();
    let switch_point = b.uniform_int(start, end + 1);
    let early_rate = b.exponential(1.0);
    let late_rate = b.exponential(1.0);
    let years = b.int_constant(IntTensor::vector(years));
    let rate_for_year = b.where_(switch_point.gt(years), early_rate, late_rate);
    let disasters_for_year = b.poisson(rate_for_year);
    b.observe(disasters_for_year, IntTensor::vector(disasters));
    let (switch_point, early_rate, late_rate) =
        (switch_point.id(), early_rate.id(), late_rate.id());

    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;
    model.probe_for_non_zero_probability(&mut state, 100, &mut KeanuRng::seed_from_u64(seed))?;

    // Gibbs for the switch year and NUTS (adapting for 1000 steps) for the rates
    let sampler = Sampler::for_model(&model, &state);
    let warmup = 1000;
    let chains = sample_chains(CHAINS, seed, &state, |state, rng| {
        sampler.sample(
            &model,
            state,
            model.latents(),
            warmup + draws,
            SamplingOptions::default().with_drop(warmup),
            rng,
        )
    })?;
    let mut years_drawn: Vec<i64> = Vec::new();
    for chain in chains.chains() {
        years_drawn.extend(
            chain
                .i64(switch_point)
                .unwrap()
                .as_slice()
                .iter()
                .map(|t| t.as_slice()[0]),
        );
    }
    let switch_year_mode = mode(&years_drawn);
    Ok(Results {
        chains,
        switch_point,
        early_rate,
        late_rate,
        switch_year_mode,
    })
}

/// The most common value (the smallest, if several are).
fn mode(values: &[i64]) -> i64 {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let mut best = (0, sorted[0]);
    let mut i = 0;
    while i < sorted.len() {
        let j = sorted[i..].iter().take_while(|&&v| v == sorted[i]).count();
        if j > best.0 {
            best = (j, sorted[i]);
        }
        i += j;
    }
    best.1
}

/// The default number of draws per chain.
pub const DEFAULT_SAMPLES: usize = 1000;

#[allow(dead_code)]
fn main() -> keanu::Result<()> {
    let results = run(1, DEFAULT_SAMPLES)?;
    println!("Switch year found: {}", results.switch_year_mode);
    for (name, node) in [
        ("switch year", results.switch_point),
        ("early rate", results.early_rate),
        ("late rate", results.late_rate),
    ] {
        let summary = results.chains.summarize(node, &[])?;
        println!(
            "{name}: mean {:.3}, R-hat {:.4}",
            summary.mean, summary.r_hat
        );
    }
    Ok(())
}
