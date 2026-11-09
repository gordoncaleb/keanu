//! Bayesian linear-spline regression with held-out predictions at a different set of inputs.
//!
//! Triangular basis functions interpolate between neighbouring knots. A prior on their coefficients keeps the
//! curve from tracking observation noise. Train and prediction basis matrices share the same coefficients.
//!
//! Run with `cargo run --release --example spline_regression`.

use keanu::prelude::*;

const KNOTS: usize = 9;
const NOISE: f64 = 0.12;

fn basis(inputs: &[f64]) -> Result<DoubleTensor> {
    let values = inputs
        .iter()
        .flat_map(|&x| {
            let position = x * (KNOTS - 1) as f64;
            (0..KNOTS).map(move |k| (1.0 - (position - k as f64).abs()).max(0.0))
        })
        .collect();
    Ok(DoubleTensor::new(&[inputs.len(), KNOTS], values)?)
}

fn signal(x: f64) -> f64 {
    (std::f64::consts::TAU * x).sin() + 0.4 * x
}

/// Fit noisy observations and return the held-out RMSE against the simulated curve.
pub fn run(draws: usize) -> Result<f64> {
    let mut rng = KeanuRng::seed_from_u64(32);
    let train: Vec<_> = (0..=64).map(|i| i as f64 / 64.0).collect();
    let test: Vec<_> = (0..16).map(|i| (i as f64 + 0.5) / 16.0).collect();
    let y = train
        .iter()
        .map(|&x| signal(x) + NOISE * rng.next_gaussian())
        .collect();

    let b = GraphBuilder::new();
    let coefficients = b.gaussian_with_shape(&[KNOTS, 1], 0.0, 1.0);
    b.set_value(coefficients, DoubleTensor::zeros(&[KNOTS, 1]));
    let fitted = b
        .constant(basis(&train)?)
        .matmul(coefficients)
        .reshape(&[-1]);
    b.observe(b.gaussian(fitted, NOISE), DoubleTensor::vector(y));
    let coefficients = coefficients.id();
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let (samples, _) = Nuts::default()
        .with_adapt_count(draws)
        .sample_with_statistics(
            &model,
            &mut state,
            &[coefficients],
            2 * draws,
            SamplingOptions::default().with_drop(draws),
            &mut rng,
        )?;
    let prediction = GraphBuilder::new();
    let weights = prediction.gaussian_with_shape(&[KNOTS, 1], 0.0, 1.0);
    let mean = prediction
        .constant(basis(&test)?)
        .matmul(weights)
        .reshape(&[-1]);
    let outcome = prediction.gaussian(mean, NOISE);
    let (weights, mean, outcome) = (weights.id(), mean.id(), outcome.id());
    let (graph, state) = prediction.build()?;
    let held_out = PredictiveSampler::new().predict(
        &graph,
        &state,
        &samples,
        &[(coefficients, weights)],
        &[mean, outcome],
        &mut rng,
    )?;
    let predictions = held_out.f64(mean).unwrap().mean()?;
    Ok((predictions
        .as_slice()
        .iter()
        .zip(&test)
        .map(|(&prediction, &x)| (prediction - signal(x)).powi(2))
        .sum::<f64>()
        / test.len() as f64)
        .sqrt())
}

fn main() -> Result<()> {
    let rmse = run(500)?;
    println!("Held-out curve RMSE {rmse:.3}.");
    Ok(())
}
