//! Infer a common calibration offset from top-coded batch success totals with heterogeneous probabilities.
//!
//! Run with `cargo run --release --example poisson_binomial_calibration`.

use keanu::distributions::PoissonBinomial;
use keanu::prelude::*;

const GROUPS: usize = 100;
const CAP: i64 = 4;
const BASE: [f64; 6] = [-1.7, -0.8, -0.3, 0.5, 1.1, 1.8];
/// Calibration offset used to generate the synthetic batches.
pub const OFFSET: f64 = 0.45;

/// Posterior mean offset and predictive probability of more than four successes.
pub fn run(draws: usize) -> Result<(f64, f64)> {
    let mut rng = KeanuRng::seed_from_u64(71);
    let counts = (0..GROUPS)
        .map(|_| {
            let successes = BASE
                .iter()
                .filter(|&&base| rng.next_f64() < 1.0 / (1.0 + (-base - OFFSET).exp()))
                .count() as i64;
            successes.min(CAP)
        })
        .collect();
    let b = GraphBuilder::new();
    let offset = b.gaussian(0.0, 1.0).label("offset");
    let probabilities = (offset + b.constant(DoubleTensor::vector(BASE.to_vec()))).sigmoid();
    let totals: IntVar<'_> = b.censored(
        keanu::graph::op::DistributionKind::PoissonBinomial,
        &[probabilities.id()],
        f64::NEG_INFINITY,
        DoubleTensor::full(&[GROUPS], CAP as f64),
    );
    b.observe(totals, IntTensor::vector(counts));
    let offset = offset.id();
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let (samples, _) = Nuts::default()
        .with_adapt_count(180)
        .sample_with_statistics(
            &model,
            &mut state,
            &[offset],
            180 + draws,
            SamplingOptions::default().with_drop(180),
            &mut rng,
        )?;
    let draws = samples.f64(offset).unwrap();
    let mean = draws.mean()?.to_scalar().unwrap();
    let column = DoubleTensor::new(&[samples.len(), 1], draws.element(&[])?)?;
    let probabilities = column.add(&DoubleTensor::vector(BASE.to_vec()))?.sigmoid();
    let log_tail = PoissonBinomial::new(probabilities)?.log_ccdf(&IntTensor::scalar(CAP))?;
    Ok((mean, log_tail.exp().sum() / samples.len() as f64))
}

fn main() -> Result<()> {
    let (offset, risk) = run(600)?;
    println!("Estimated offset from top-coded totals {offset:.3} (simulated {OFFSET})");
    println!("Predictive probability of more than four successes {risk:.3}");
    Ok(())
}
