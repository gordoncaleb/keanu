//! Denoise observations of a signal whose Gaussian prior is specified by neighboring precision terms.
//!
//! Run with `cargo run --release --example precision_gaussian_denoising`.

use keanu::prelude::*;

const DIMENSION: usize = 8;
const NOISE: f64 = 0.35;

/// Returns the posterior mean from NUTS and the exact Gaussian posterior mean.
pub fn run(draws: usize) -> Result<(DoubleTensor, DoubleTensor)> {
    let precision = DoubleTensor::new(
        &[DIMENSION, DIMENSION],
        (0..DIMENSION * DIMENSION)
            .map(|i| match (i / DIMENSION).abs_diff(i % DIMENSION) {
                0 => 2.6,
                1 => -1.0,
                _ => 0.0,
            })
            .collect(),
    )?;
    let measurements = DoubleTensor::vector(vec![0.0, 0.55, 0.9, 1.35, 0.8, 0.3, -0.3, -0.4]);
    let noise_precision = 1.0 / NOISE.powi(2);
    let posterior_precision = precision.add(&(DoubleTensor::eye(DIMENSION)? * noise_precision))?;
    let expected = posterior_precision.solve(&(&measurements * noise_precision))?;

    let b = GraphBuilder::new();
    let signal = b.multivariate_gaussian_precision(DoubleTensor::zeros(&[DIMENSION]), precision);
    b.set_value(signal, DoubleTensor::zeros(&[DIMENSION]));
    b.observe(b.gaussian(signal, NOISE), measurements);
    let signal = signal.id();
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let (samples, _) = Nuts::default()
        .with_adapt_count(200)
        .sample_with_statistics(
            &model,
            &mut state,
            &[signal],
            200 + draws,
            SamplingOptions::default().with_drop(200),
            &mut KeanuRng::seed_from_u64(62),
        )?;
    Ok((samples.f64(signal).unwrap().mean()?, expected))
}

fn main() -> Result<()> {
    let (estimated, exact) = run(700)?;
    println!("Estimated signal {:?}", estimated.as_slice());
    println!("Exact posterior mean {:?}", exact.as_slice());
    Ok(())
}
