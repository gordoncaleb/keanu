//! Fit a regression with correlated predictors in QR coordinates.
//!
//! Run with `cargo run --release --example correlated_regression`.

use keanu::prelude::*;

pub const TRUE_COEFFICIENTS: [f64; 2] = [1.4, -0.8];

/// Returns the posterior mean coefficients in the original predictor coordinates.
pub fn run(draws: usize) -> Result<DoubleTensor> {
    let mut rng = KeanuRng::seed_from_u64(53);
    let mut design = Vec::new();
    let mut response = Vec::new();
    let n = 128;
    for i in 0..n {
        let x = -1.0 + 2.0 * i as f64 / (n - 1) as f64;
        let correlated = x + 0.17 * (i as f64 * 1.3).sin();
        design.extend([x, correlated]);
        response.push(
            TRUE_COEFFICIENTS[0] * x
                + TRUE_COEFFICIENTS[1] * correlated
                + 0.1 * rng.next_gaussian(),
        );
    }

    let b = GraphBuilder::new();
    let x = b.constant(DoubleTensor::new(&[n, 2], design)?);
    let (q, r) = x.thin_qr();
    let theta = b.gaussian_with_shape(&[2], 0.0, 5.0);
    let beta = r.solve_triangular(theta, false, false).label("beta");
    let mean = q.matmul(theta.reshape(&[2, 1])).reshape(&[n as isize]);
    b.observe(b.gaussian(mean, 0.1), DoubleTensor::vector(response));
    let beta = beta.id();
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let (samples, _) = Nuts::default()
        .with_adapt_count(200)
        .sample_with_statistics(
            &model,
            &mut state,
            &[beta],
            200 + draws,
            SamplingOptions::default().with_drop(200),
            &mut rng,
        )?;
    Ok(samples.f64(beta).unwrap().mean()?)
}

fn main() -> Result<()> {
    println!("Posterior mean coefficients {:?}", run(700)?.as_slice());
    Ok(())
}
