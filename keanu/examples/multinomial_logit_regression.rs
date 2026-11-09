//! Estimate how a precinct-level covariate shifts vote shares among three candidates.
//!
//! Run with `cargo run --release --example multinomial_logit_regression`.

use keanu::distributions::MultinomialLogit;
use keanu::prelude::*;

const PRECINCTS: usize = 48;
const VOTERS: i64 = 80;
const SLOPE: [f64; 2] = [0.8, -0.55];

/// Returns the posterior mean slopes for candidates one and two relative to candidate three.
pub fn run(draws: usize) -> Result<DoubleTensor> {
    let mut rng = KeanuRng::seed_from_u64(61);
    let x = (0..PRECINCTS)
        .map(|i| 2.0 * i as f64 / (PRECINCTS - 1) as f64 - 1.0)
        .collect::<Vec<_>>();
    let logits = DoubleTensor::new(
        &[PRECINCTS, 3],
        x.iter()
            .flat_map(|&x| [0.2 + SLOPE[0] * x, -0.3 + SLOPE[1] * x, 0.0])
            .collect(),
    )?;
    let votes = MultinomialLogit::new(IntTensor::scalar(VOTERS), logits)?
        .sample(&[PRECINCTS, 3], &mut rng)?;

    let b = GraphBuilder::new();
    let intercept = b.gaussian_with_shape(&[2], 0.0, 1.0);
    let slope = b.gaussian_with_shape(&[2], 0.0, 1.0);
    let x = b.constant(DoubleTensor::new(&[PRECINCTS, 1], x)?);
    let reference = b.constant(DoubleTensor::zeros(&[PRECINCTS, 1]));
    let logits = b.concat(-1, &[intercept + x * slope, reference]);
    b.observe(b.multinomial_logit(VOTERS, logits), votes);
    let slope = slope.id();
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let (samples, _) = Nuts::default()
        .with_adapt_count(200)
        .sample_with_statistics(
            &model,
            &mut state,
            &[slope],
            200 + draws,
            SamplingOptions::default().with_drop(200),
            &mut rng,
        )?;
    Ok(samples.f64(slope).unwrap().mean()?)
}

fn main() -> Result<()> {
    println!("Posterior mean slopes {:?}", run(700)?.as_slice());
    Ok(())
}
