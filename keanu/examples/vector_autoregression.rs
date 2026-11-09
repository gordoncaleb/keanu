//! A Bayesian VAR(1) models how two series jointly depend on their previous values.
//!
//! Each observation has two independent Gaussian innovations with known scales. Fit one transition matrix
//! and forecast both series recursively, propagating uncertainty in the matrix through each forecast horizon.
//!
//! Run with `cargo run --release --example vector_autoregression`.

use keanu::prelude::*;

const TRANSITIONS: usize = 220;
const HORIZON: usize = 4;
const NOISE: f64 = 0.3;
pub const TRUE_COEFFICIENTS: [f64; 4] = [0.6, -0.25, 0.3, 0.55];

fn transition([x, y]: [f64; 2]) -> [f64; 2] {
    let [a, b, c, d] = TRUE_COEFFICIENTS;
    [a * x + b * y, c * x + d * y]
}

/// Return the posterior mean coefficient matrix and the error of the conditional mean forecast.
pub fn run(draws: usize) -> Result<(DoubleTensor, f64)> {
    let mut rng = KeanuRng::seed_from_u64(44);
    let mut current = [0.0, 0.0];
    let mut previous = Vec::with_capacity(2 * TRANSITIONS);
    let mut observed = Vec::with_capacity(2 * TRANSITIONS);
    for _ in 0..TRANSITIONS {
        previous.extend(current);
        current = transition(current).map(|value| value + NOISE * rng.next_gaussian());
        observed.extend(current);
    }

    let b = GraphBuilder::new();
    let coefficients = b.gaussian_with_shape(&[2, 2], 0.0, 0.6);
    b.set_value(coefficients, DoubleTensor::zeros(&[2, 2]));
    let design = b.constant(DoubleTensor::new(&[TRANSITIONS, 2], previous)?);
    let mean = design.matmul(coefficients.transpose());
    b.observe(
        b.gaussian(mean, NOISE),
        DoubleTensor::new(&[TRANSITIONS, 2], observed)?,
    );
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
    let weights = prediction.gaussian_with_shape(&[2, 2], 0.0, 0.6);
    let forecast = prediction
        .scan_steps(
            HORIZON,
            prediction.constant(DoubleTensor::new(&[2, 1], current.to_vec())?),
            &[weights],
            |_, carry, parameters| parameters[0].matmul(carry),
        )
        .id();
    let weights = weights.id();
    let (graph, state) = prediction.build()?;
    let held_out = PredictiveSampler::new().predict(
        &graph,
        &state,
        &samples,
        &[(coefficients, weights)],
        &[forecast],
        &mut rng,
    )?;
    let predicted = held_out.f64(forecast).unwrap().mean()?;
    let squared_error = predicted
        .as_slice()
        .chunks_exact(2)
        .map(|predicted| {
            current = transition(current);
            (predicted[0] - current[0]).powi(2) + (predicted[1] - current[1]).powi(2)
        })
        .sum::<f64>();
    Ok((
        samples.f64(coefficients).unwrap().mean()?,
        (squared_error / (2 * HORIZON) as f64).sqrt(),
    ))
}

fn main() -> Result<()> {
    let (coefficients, rmse) = run(500)?;
    println!(
        "Posterior mean coefficient matrix {:?}",
        coefficients.as_slice()
    );
    println!("Conditional mean forecast RMSE {rmse:.3}.");
    Ok(())
}
