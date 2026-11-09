//! Forecast a periodic signal with a small Fourier basis and with inducing points.
//!
//! Run with `cargo run --release --example structured_gp`.

use keanu::graph::gp::{InducingGp, Kernel, PeriodicHsgp};
use keanu::prelude::*;
use std::f64::consts::TAU;

fn signal(x: f64) -> f64 {
    (TAU * x).sin() + 0.25 * (2.0 * TAU * x).cos()
}

fn fit(fourier: bool, draws: usize) -> Result<f64> {
    let mut rng = KeanuRng::seed_from_u64(54);
    let train: Vec<f64> = (0..128).map(|i| i as f64 / 64.0).collect();
    let forecast: Vec<f64> = (0..32).map(|i| 2.0 + i as f64 / 32.0).collect();
    let response = DoubleTensor::vector(
        train
            .iter()
            .map(|&x| signal(x) + 0.08 * rng.next_gaussian())
            .collect(),
    );
    let b = GraphBuilder::new();
    let kernel = Kernel::periodic(&b, 1.0, 0.8, 1.0);
    let train = DoubleTensor::vector(train);
    let future = DoubleTensor::vector(forecast.clone());
    let (mean, prediction) = if fourier {
        let gp = PeriodicHsgp::new(&kernel, 8)?;
        (gp.predict(train), gp.predict(future))
    } else {
        let locations = DoubleTensor::vector((0..12).map(|i| i as f64 / 12.0).collect());
        let gp = InducingGp::new(&kernel, locations)?;
        (gp.predict(train), gp.predict(future))
    };
    b.observe(b.gaussian(mean, 0.08), response);
    let prediction = prediction.id();
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let (samples, _) = Nuts::default()
        .with_adapt_count(250)
        .sample_with_statistics(
            &model,
            &mut state,
            &[prediction],
            250 + draws,
            SamplingOptions::default().with_drop(250),
            &mut rng,
        )?;
    let mean = samples.f64(prediction).unwrap().mean()?;
    let squared_error = mean
        .as_slice()
        .iter()
        .zip(forecast)
        .map(|(y, x)| (y - signal(x)).powi(2))
        .sum::<f64>();
    Ok((squared_error / mean.len() as f64).sqrt())
}

/// Held-out root-mean-square errors for Fourier and inducing-point predictions.
pub fn run(draws: usize) -> Result<[f64; 2]> {
    Ok([fit(true, draws)?, fit(false, draws)?])
}

fn main() -> Result<()> {
    let [fourier, inducing] = run(500)?;
    println!("Forecast RMSE for Fourier {fourier:.3} and inducing points {inducing:.3}");
    Ok(())
}
