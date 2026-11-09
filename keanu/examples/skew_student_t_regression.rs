//! Fit a linear response with heavy-tailed errors that favor one direction.
//!
//! Run with `cargo run --release --example skew_student_t_regression`.

use keanu::distributions::SkewStudentT;
use keanu::prelude::*;

/// True slope for the synthetic regression.
pub const SLOPE: f64 = 1.35;

/// Returns the posterior means of the slope and the left-tail shape.
pub fn run(draws: usize) -> Result<[f64; 2]> {
    let mut rng = KeanuRng::seed_from_u64(73);
    let n = 320;
    let x = DoubleTensor::vector(
        (0..n)
            .map(|i| 2.0 * i as f64 / (n - 1) as f64 - 1.0)
            .collect(),
    );
    let noise = SkewStudentT::new(
        DoubleTensor::scalar(4.5),
        DoubleTensor::scalar(2.0),
        DoubleTensor::scalar(0.0),
        DoubleTensor::scalar(0.35),
    )
    .sample(&[n], &mut rng)?;
    let response = (&x * SLOPE).add(&noise)?;
    let b = GraphBuilder::new();
    let slope = b.gaussian(0.0, 2.0).label("slope");
    let shape = b.log_normal(1.4, 0.5).label("left_shape");
    let location = b.constant(x) * slope;
    b.observe(b.skew_student_t(shape, 2.0, location, 0.35), response);
    let ids = [slope.id(), shape.id()];
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let (samples, _) = Nuts::default()
        .with_adapt_count(240)
        .sample_with_statistics(
            &model,
            &mut state,
            &ids,
            240 + draws,
            SamplingOptions::default().with_drop(240),
            &mut rng,
        )?;
    let mean = |id| -> Result<f64> { Ok(samples.f64(id).unwrap().mean()?.to_scalar().unwrap()) };
    Ok([mean(ids[0])?, mean(ids[1])?])
}

fn main() -> Result<()> {
    let [slope, shape] = run(600)?;
    println!("Posterior mean slope {slope:.3} and left-tail shape {shape:.3}");
    Ok(())
}
