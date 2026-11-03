//! When did the author's text messaging rate increase, based on daily message counts?
//!
//! From "Bayesian Methods for Hackers", chapter 1. Run with `cargo run --example text_messaging`.

use keanu::prelude::*;

/// The results the Java example reports.
#[derive(Debug)]
pub struct Results {
    /// The most frequently sampled switch day.
    pub switch_point_mode: i64,
    /// The posterior mean of the rate before the switch.
    pub early_rate_mean: f64,
    /// The most frequently sampled rate after the switch.
    pub late_rate_mode: f64,
}

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

pub fn run(seed: u64, sample_count: usize) -> keanu::Result<Results> {
    let (days, messages) = data();
    let number_of_days = days.len() as i64;

    // The book uses a wider prior; Metropolis–Hastings with prior proposals works better with a tighter one.
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
    let mut rng = KeanuRng::seed_from_u64(seed);
    let model = Model::new(graph, &state)?;
    model.probe_for_non_zero_probability(&mut state, 1000, &mut rng)?;

    let samples = MetropolisHastings::default().sample_with(
        &model,
        &mut state,
        model.latents(),
        sample_count,
        SamplingOptions::default()
            .with_drop(sample_count / 2)
            .with_down_sample_interval(model.latents().len()),
        &mut rng,
    )?;

    Ok(Results {
        switch_point_mode: samples
            .i64(switch_point)
            .unwrap()
            .mode()
            .unwrap()
            .to_scalar()
            .unwrap(),
        early_rate_mean: samples
            .f64(early_rate)
            .unwrap()
            .mean()?
            .to_scalar()
            .unwrap(),
        late_rate_mode: samples
            .f64(late_rate)
            .unwrap()
            .mode()
            .unwrap()
            .to_scalar()
            .unwrap(),
    })
}

/// The number of samples the Java example takes.
pub const DEFAULT_SAMPLES: usize = 1000;

#[allow(dead_code)]
fn main() -> keanu::Result<()> {
    println!("{:?}", run(1, DEFAULT_SAMPLES)?);
    Ok(())
}
