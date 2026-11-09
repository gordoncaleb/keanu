//! Infer the distribution of device lifetimes counted in completed cycles, with administrative censoring.
//!
//! Run with `cargo run --release --example discrete_weibull_reliability`.

use keanu::prelude::*;

const CAP: i64 = 4;
/// Survival probability past zero used to simulate lifetimes.
pub const Q: f64 = 0.81;
/// Shape used to simulate lifetimes.
pub const BETA: f64 = 1.45;

/// Posterior means of the survival and shape parameters.
pub fn run(draws: usize) -> Result<[f64; 2]> {
    let mut rng = KeanuRng::seed_from_u64(72);
    let cycles: Vec<i64> = (0..500)
        .map(|_| {
            let uniform = rng.next_f64();
            // The first threshold crossed is the count; any device still running at CAP is censored.
            (0..CAP)
                .find(|&k| uniform < 1.0 - Q.powf(((k + 1) as f64).powf(BETA)))
                .unwrap_or(CAP)
        })
        .collect();
    let b = GraphBuilder::new();
    let q = b.beta(5.0, 2.0).label("q");
    let beta = b.log_normal(0.0, 0.5).label("beta");
    let y: IntVar = b.censored(
        keanu::graph::op::DistributionKind::DiscreteWeibull,
        &[q.id(), beta.id()],
        f64::NEG_INFINITY,
        b.constant(DoubleTensor::full(&[cycles.len()], CAP as f64)),
    );
    b.observe(y, IntTensor::vector(cycles));
    let ids = [q.id(), beta.id()];
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let (samples, _) = Nuts::default()
        .with_adapt_count(200)
        .sample_with_statistics(
            &model,
            &mut state,
            &ids,
            200 + draws,
            SamplingOptions::default().with_drop(200),
            &mut rng,
        )?;
    let mean = |id| -> Result<f64> { Ok(samples.f64(id).unwrap().mean()?.to_scalar().unwrap()) };
    Ok([mean(ids[0])?, mean(ids[1])?])
}

fn main() -> Result<()> {
    let [q, beta] = run(600)?;
    println!("Posterior mean q {q:.3}, shape {beta:.3}");
    Ok(())
}
