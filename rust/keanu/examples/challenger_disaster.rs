//! Does temperature correlate with O-ring defects?
//!
//! A logistic regression of O-ring failure on launch temperature, from "Bayesian Methods for Hackers", chapter 2.
//! Run with `cargo run --example challenger_disaster`.

use keanu::prelude::*;

/// Posterior means of the logistic regression's intercept and slope.
#[derive(Debug)]
pub struct Posteriors {
    pub alpha: f64,
    pub beta: f64,
}

/// (temperature, O-ring failure) from the bundled CSV.
pub fn data() -> (Vec<f64>, Vec<bool>) {
    include_str!("data/challenger_data.csv")
        .lines()
        .skip(1)
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let (temp, failure) = line.split_once(',').expect("temps,oRingFailure");
            (temp.trim().parse::<f64>().unwrap(), failure.trim() == "1")
        })
        .unzip()
}

fn tau_to_sigma(tau: f64) -> f64 {
    (1.0 / tau).sqrt()
}

pub fn run(seed: u64, sample_count: usize) -> keanu::Result<Posteriors> {
    let (temps, failures) = data();

    // Tighter priors than the book's, because Metropolis–Hastings proposes from the prior.
    let b = GraphBuilder::new();
    let alpha = b.gaussian(0.0, tau_to_sigma(0.005));
    let beta = b.gaussian(0.0, tau_to_sigma(0.01));
    let temps = b.constant(DoubleTensor::vector(temps));
    let probability_of_defect = (-(beta * temps + alpha)).sigmoid();
    let defect = b.bernoulli(probability_of_defect);
    b.observe(defect, BoolTensor::vector(failures));
    let (alpha, beta) = (alpha.id(), beta.id());

    let (graph, mut state) = b.build()?;
    let mut rng = KeanuRng::seed_from_u64(seed);
    let model = Model::new(graph, &state)?;
    model.probe_for_non_zero_probability(&mut state, 1000, &mut rng)?;

    let samples = MetropolisHastings::default().sample_with(
        &model,
        &mut state,
        model.latents(),
        sample_count,
        SamplingOptions {
            drop: sample_count / 2,
            down_sample_interval: model.latents().len(),
        },
        &mut rng,
    )?;
    Ok(Posteriors {
        alpha: samples.f64(alpha).unwrap().mean()?.to_scalar().unwrap(),
        beta: samples.f64(beta).unwrap().mean()?.to_scalar().unwrap(),
    })
}

/// The same model sampled with NUTS, which handles the strong correlation between alpha and beta far better than
/// Metropolis–Hastings with prior proposals.
pub fn run_with_nuts(seed: u64, sample_count: usize) -> keanu::Result<Posteriors> {
    let (temps, failures) = data();
    let b = GraphBuilder::new();
    let alpha = b.gaussian(0.0, tau_to_sigma(0.005));
    let beta = b.gaussian(0.0, tau_to_sigma(0.01));
    let temps = b.constant(DoubleTensor::vector(temps));
    let probability_of_defect = (-(beta * temps + alpha)).sigmoid();
    let defect = b.bernoulli(probability_of_defect);
    b.observe(defect, BoolTensor::vector(failures));
    b.set_value(alpha, DoubleTensor::scalar(0.0));
    b.set_value(beta, DoubleTensor::scalar(0.0));
    let (alpha, beta) = (alpha.id(), beta.id());

    let (graph, mut state) = b.build()?;
    let mut rng = KeanuRng::seed_from_u64(seed);
    let model = Model::new(graph, &state)?;
    model.probe_for_non_zero_probability(&mut state, 1000, &mut rng)?;
    let (samples, _) = Nuts::default().sample_with_statistics(
        &model,
        &mut state,
        &[alpha, beta],
        sample_count,
        SamplingOptions {
            drop: sample_count / 2,
            down_sample_interval: 1,
        },
        &mut rng,
    )?;
    Ok(Posteriors {
        alpha: samples.f64(alpha).unwrap().mean()?.to_scalar().unwrap(),
        beta: samples.f64(beta).unwrap().mean()?.to_scalar().unwrap(),
    })
}

/// The number of samples the Java example takes.
pub const DEFAULT_SAMPLES: usize = 3000;

#[allow(dead_code)]
fn main() -> keanu::Result<()> {
    println!("Metropolis–Hastings: {:?}", run(1, DEFAULT_SAMPLES)?);
    println!("NUTS: {:?}", run_with_nuts(1, 4000)?);
    Ok(())
}
