//! Estimate a regression slope when the predictor is measured with error.
//!
//! The data are centred. The true predictor x is latent, w is its noisy measurement, and y is the outcome.
//! Both noise scales are known. The model estimates the slope jointly with the true predictors, then compares
//! the posterior mean slope to ordinary least squares on the measured predictors.
//!
//! Run with `cargo run --release --example measurement_error`.

use keanu::prelude::*;

const COUNT: usize = 160;
const PREDICTOR_NOISE: f64 = 1.0;
const OUTCOME_NOISE: f64 = 0.5;
pub const TRUE_SLOPE: f64 = 1.5;

/// Return the naive slope and the posterior mean slope from the measurement-error model.
pub fn run(draws: usize) -> keanu::Result<(f64, f64)> {
    let mut rng = KeanuRng::seed_from_u64(81);
    let (measured, outcomes): (Vec<_>, Vec<_>) = (0..COUNT)
        .map(|_| {
            let x = rng.next_gaussian();
            (
                x + PREDICTOR_NOISE * rng.next_gaussian(),
                TRUE_SLOPE * x + OUTCOME_NOISE * rng.next_gaussian(),
            )
        })
        .unzip();
    let naive = measured
        .iter()
        .zip(&outcomes)
        .map(|(w, y)| w * y)
        .sum::<f64>()
        / measured.iter().map(|w| w * w).sum::<f64>();

    let b = GraphBuilder::new();
    let x = b.gaussian_with_shape(&[COUNT], 0.0, 1.0);
    let slope = b.gaussian(0.0, 2.0).label("slope");
    let measured = DoubleTensor::vector(measured);
    b.set_value(x, &measured * 0.5);
    b.set_value(slope, DoubleTensor::scalar(naive));
    b.observe(b.gaussian(x, PREDICTOR_NOISE), measured);
    b.observe(
        b.gaussian(x * slope, OUTCOME_NOISE),
        DoubleTensor::vector(outcomes),
    );
    let slope = slope.id();
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let (samples, _) = Nuts::default()
        .with_adapt_count(draws)
        .sample_with_statistics(
            &model,
            &mut state,
            &[slope],
            2 * draws,
            SamplingOptions::default().with_drop(draws),
            &mut rng,
        )?;
    Ok((
        naive,
        samples.f64(slope).unwrap().mean()?.to_scalar().unwrap(),
    ))
}

fn main() -> keanu::Result<()> {
    let (naive, corrected) = run(500)?;
    println!(
        "Naive slope {naive:.3}. Posterior mean slope {corrected:.3}. Simulated slope {TRUE_SLOPE}."
    );
    Ok(())
}
