//! When did the rate of coal mining disasters change?
//!
//! Each year's disaster count is Poisson, with one rate before an unknown switch year and another after it.
//! Metropolis–Hastings samples the switch year and both rates.
//!
//! Run with `cargo run --example coal_mining_disasters`.

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

/// The most frequently sampled switch year.
pub fn run(seed: u64, sample_count: usize) -> keanu::Result<i64> {
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
    let switch_point = switch_point.id();

    let (graph, mut state) = b.build()?;
    let mut rng = KeanuRng::seed_from_u64(seed);
    let model = Model::new(graph, &state)?;
    model.probe_for_non_zero_probability(&mut state, 100, &mut rng)?;

    let samples = MetropolisHastings::default().sample_with(
        &model,
        &mut state,
        model.latents(),
        sample_count,
        SamplingOptions::default()
            .with_drop(sample_count / 5)
            .with_down_sample_interval(3),
        &mut rng,
    )?;
    let mode = samples.i64(switch_point).unwrap().mode().unwrap();
    Ok(mode.to_scalar().unwrap())
}

/// The default number of samples (as many as Keanu's original example took).
pub const DEFAULT_SAMPLES: usize = 500;

#[allow(dead_code)]
fn main() -> keanu::Result<()> {
    println!("Switch year found: {}", run(1, DEFAULT_SAMPLES)?);
    Ok(())
}
