//! Forecast two related series with correlated Gaussian innovations.
//!
//! Run with `cargo run --release --example multivariate_walk_forecast`.

use keanu::prelude::*;

const DRIFT: [f64; 2] = [0.2, -0.1];
const START: [f64; 2] = [1.0, 2.0];
const HORIZON: usize = 4;

/// Mean and cross-covariance of the forecast at the last horizon.
pub fn run(draws: usize) -> Result<([f64; 2], f64)> {
    let b = GraphBuilder::new();
    let path = b
        .multivariate_gaussian_random_walk(
            HORIZON + 1,
            DoubleTensor::vector(DRIFT.to_vec()),
            DoubleTensor::new(&[2, 2], vec![0.3, 0.0, 0.2, 0.25])?,
            DoubleTensor::vector(START.to_vec()),
        )
        .id();
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let mut rng = KeanuRng::seed_from_u64(55);
    let samples = ForwardSampler::default().sample(&model, &mut state, &[path], draws, &mut rng)?;
    let paths = samples.f64(path).unwrap();
    let x = paths.element(&[HORIZON, 0])?;
    let y = paths.element(&[HORIZON, 1])?;
    let mx = x.iter().sum::<f64>() / draws as f64;
    let my = y.iter().sum::<f64>() / draws as f64;
    let covariance = x
        .iter()
        .zip(y)
        .map(|(x, y)| (x - mx) * (y - my))
        .sum::<f64>()
        / (draws - 1) as f64;
    Ok(([mx, my], covariance))
}

fn main() -> Result<()> {
    let (mean, covariance) = run(4000)?;
    println!("Horizon {HORIZON} mean {mean:?} and cross-covariance {covariance:.3}");
    Ok(())
}
