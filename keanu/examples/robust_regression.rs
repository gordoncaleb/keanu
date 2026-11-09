//! Fit a regression with heavy-tailed observation noise and learn its degrees of freedom.
//!
//! Run with `cargo run --release --example robust_regression`.

use keanu::prelude::*;

const POINTS: usize = 100;
pub const TRUE_SLOPE: f64 = 1.5;

/// Return the posterior mean slope and degrees of freedom.
pub fn run(draws: usize) -> Result<(f64, f64)> {
    let mut rng = KeanuRng::seed_from_u64(42);
    let x: Vec<_> = (0..POINTS)
        .map(|i| 4.0 * i as f64 / (POINTS - 1) as f64 - 2.0)
        .collect();
    let y = x
        .iter()
        .map(|&x| {
            // Independent t(3) draw from a normal and a sum of three squared normals.
            let variance = (0..3).map(|_| rng.next_gaussian().powi(2)).sum::<f64>() / 3.0;
            0.5 + TRUE_SLOPE * x + 0.35 * rng.next_gaussian() / variance.sqrt()
        })
        .collect();
    let b = GraphBuilder::new();
    let intercept = b.gaussian(0.0, 1.0);
    let slope = b.gaussian(0.0, 2.0);
    let scale = b.log_normal(0.5_f64.ln(), 0.7);
    let nu = b.log_normal(5.0_f64.ln(), 0.9);
    for (node, value) in [(intercept, 0.0), (slope, 1.0), (scale, 0.5), (nu, 5.0)] {
        b.set_value(node, DoubleTensor::scalar(value));
    }
    let mean = intercept + slope * b.constant(DoubleTensor::vector(x));
    b.observe(
        b.location_scale_student_t(nu, mean, scale),
        DoubleTensor::vector(y),
    );
    let (slope, nu) = (slope.id(), nu.id());
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let (samples, _) = Nuts::default()
        .with_adapt_count(draws)
        .sample_with_statistics(
            &model,
            &mut state,
            &[slope, nu],
            2 * draws,
            SamplingOptions::default().with_drop(draws),
            &mut rng,
        )?;
    Ok((
        samples.f64(slope).unwrap().mean()?.to_scalar().unwrap(),
        samples.f64(nu).unwrap().mean()?.to_scalar().unwrap(),
    ))
}

fn main() -> Result<()> {
    let (slope, nu) = run(500)?;
    println!("Posterior mean slope {slope:.3}, degrees of freedom {nu:.3}.");
    Ok(())
}
