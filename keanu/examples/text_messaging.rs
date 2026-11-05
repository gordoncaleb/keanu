//! When did the author's text messaging rate increase, based on daily message counts?
//!
//! From "Bayesian Methods for Hackers", chapter 1. The switch day is discrete and the rates continuous, so the
//! default sampler is the compound sampler (Gibbs for the day, NUTS for the rates), run here with four chains.
//! Run with `cargo run --release --example text_messaging`.

use keanu::algorithms::ChainSamples;
use keanu::algorithms::mcmc::chains::sample_chains;
use keanu::prelude::*;

/// The results the example reports.
#[derive(Debug)]
pub struct Results {
    /// The most frequently sampled switch day.
    pub switch_point_mode: i64,
    /// The posterior mean of the rate before the switch.
    pub early_rate_mean: f64,
    /// The posterior mean of the rate after the switch.
    pub late_rate_mean: f64,
    /// The draws of every chain.
    pub chains: ChainSamples,
    /// The switch day: the first day with the late rate.
    pub switch_point: NodeId,
    /// The rate before the switch.
    pub early_rate: NodeId,
    /// The rate from the switch on.
    pub late_rate: NodeId,
}

/// The number of chains.
pub const CHAINS: usize = 4;

/// (day, number of messages) from the bundled CSV.
pub fn data() -> (Vec<i64>, Vec<i64>) {
    include_str!("data/text_messaging_data.csv")
        .lines()
        .skip(1)
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let (day, count) = line.split_once(',').expect("day,numberOfMessages");
            (
                day.trim().parse::<i64>().unwrap(),
                count.trim().parse::<i64>().unwrap(),
            )
        })
        .unzip()
}

/// Samples the model with `CHAINS` chains of `draws` draws each, after the default sampler's 1000 adaptation steps.
pub fn run(seed: u64, draws: usize) -> keanu::Result<Results> {
    let (days, messages) = data();
    let number_of_days = days.len() as i64;

    // Exponential takes a rate, so a prior with mean alpha has rate 1 / alpha.
    let alpha = 10.0;
    let b = GraphBuilder::new();
    let early_rate = b.exponential(1.0 / alpha);
    let late_rate = b.exponential(1.0 / alpha);
    let switch_point = b.uniform_int(0, number_of_days);
    let days = b.int_constant(IntTensor::vector(days));
    let rate_for_day = b.where_(switch_point.gt(days), early_rate, late_rate);
    let texts_for_day = b.poisson(rate_for_day);
    b.observe(texts_for_day, IntTensor::vector(messages));
    let (early_rate, late_rate, switch_point) =
        (early_rate.id(), late_rate.id(), switch_point.id());

    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;
    model.probe_for_non_zero_probability(&mut state, 1000, &mut KeanuRng::seed_from_u64(seed))?;

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

    let mut switch_points: Vec<i64> = Vec::new();
    for chain in chains.chains() {
        switch_points.extend(
            chain
                .i64(switch_point)
                .unwrap()
                .as_slice()
                .iter()
                .map(|t| t.as_slice()[0]),
        );
    }
    let mut counts = std::collections::BTreeMap::new();
    for &day in &switch_points {
        *counts.entry(day).or_insert(0usize) += 1;
    }
    let switch_point_mode = counts
        .iter()
        .max_by_key(|&(day, count)| (*count, std::cmp::Reverse(*day)))
        .map(|(&day, _)| day)
        .unwrap();
    let mean = |id: NodeId| -> keanu::Result<f64> { Ok(chains.summarize(id, &[])?.mean) };
    Ok(Results {
        switch_point_mode,
        early_rate_mean: mean(early_rate)?,
        late_rate_mean: mean(late_rate)?,
        switch_point,
        early_rate,
        late_rate,
        chains,
    })
}

/// The default number of draws per chain.
pub const DEFAULT_SAMPLES: usize = 1000;

#[allow(dead_code)]
fn main() -> keanu::Result<()> {
    let r = run(1, DEFAULT_SAMPLES)?;
    println!(
        "switch day {}, early rate {:.2}, late rate {:.2}",
        r.switch_point_mode, r.early_rate_mean, r.late_rate_mean
    );
    for (name, node) in [
        ("switch day", r.switch_point),
        ("early rate", r.early_rate),
        ("late rate", r.late_rate),
    ] {
        println!("{name}: R-hat {:.4}", r.chains.rhat(node, &[])?);
    }
    Ok(())
}
