//! Is treatment A more likely to drive purchases than treatment B?
//!
//! From "Bayesian Methods for Hackers", chapter 2. Run with `cargo run --example ab_testing`.

use keanu::prelude::*;

/// Posterior means of the two purchase probabilities.
#[derive(Debug)]
pub struct Posteriors {
    pub p_a: f64,
    pub p_b: f64,
}

pub fn run(seed: u64, sample_count: usize) -> keanu::Result<Posteriors> {
    let mut rng = KeanuRng::seed_from_u64(seed);

    // manufacture observations: 1500 visitors see A (true rate 5%), 750 see B (true rate 4%)
    let observations_a = rng.uniform(&[1, 1500]).lt(&DoubleTensor::scalar(0.05))?;
    let observations_b = rng.uniform(&[1, 750]).lt(&DoubleTensor::scalar(0.04))?;

    let b = GraphBuilder::new();
    let probability_a = b.uniform(0.0, 1.0);
    let probability_b = b.uniform(0.0, 1.0);
    let delta = probability_a - probability_b;
    let observation_a = b.bernoulli_with_shape(&[1, 1500], probability_a);
    let observation_b = b.bernoulli_with_shape(&[1, 750], probability_b);
    b.observe(observation_a, observations_a);
    b.observe(observation_b, observations_b);
    let (probability_a, probability_b, delta) =
        (probability_a.id(), probability_b.id(), delta.id());

    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;
    model.probe_for_non_zero_probability(&mut state, 100, &mut rng)?;

    let samples = MetropolisHastings::default().sample_with(
        &model,
        &mut state,
        &[probability_a, probability_b, delta],
        sample_count,
        SamplingOptions {
            drop: sample_count / 2,
            down_sample_interval: model.latents().len(),
        },
        &mut rng,
    )?;

    // the most probable probabilities are the averages of the walk in this case
    Ok(Posteriors {
        p_a: samples
            .f64(probability_a)
            .unwrap()
            .mean()?
            .to_scalar()
            .unwrap(),
        p_b: samples
            .f64(probability_b)
            .unwrap()
            .mean()?
            .to_scalar()
            .unwrap(),
    })
}

/// The number of samples the Java example takes.
pub const DEFAULT_SAMPLES: usize = 100;

#[allow(dead_code)]
fn main() -> keanu::Result<()> {
    println!("{:?}", run(1, DEFAULT_SAMPLES)?);
    Ok(())
}
