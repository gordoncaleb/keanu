//! Bernoulli logistic regression via Pólya–Gamma augmentation and two Gibbs updates.
//!
//! Run with `cargo run --release --example polya_gamma_logistic`.

use keanu::distributions::PolyaGamma1;
use keanu::prelude::*;
use keanu::tensor::special::{log_sum_exp, softplus};

const OBSERVATIONS: usize = 80;
const PRIOR_SD: f64 = 1.5;
const SLOPE: f64 = 1.2;

/// Returns the Gibbs and grid posterior summaries, each as [mean, standard deviation].
pub fn run(draws: usize) -> Result<([f64; 2], [f64; 2])> {
    assert!(draws > 0);
    let mut rng = KeanuRng::seed_from_u64(563);
    let features = DoubleTensor::vector(
        (0..OBSERVATIONS)
            .map(|i| 4.0 * i as f64 / (OBSERVATIONS - 1) as f64 - 2.0)
            .collect(),
    );
    let outcomes: Vec<_> = features
        .as_slice()
        .iter()
        .map(|&x| rng.next_f64_non_zero().ln() < -softplus(-SLOPE * x))
        .collect();
    let linear_term: f64 = features
        .as_slice()
        .iter()
        .zip(&outcomes)
        .map(|(&x, &y)| x * (f64::from(y) - 0.5))
        .sum();
    let squared_features = features.map(|x| x * x);

    let mut slope = 0.0;
    let (mut sum, mut squares) = (0.0, 0.0);
    for step in 0..draws + 200 {
        // Conditional on the current slope, each Bernoulli observation gets an independent PG(1, xβ).
        let omega = PolyaGamma1::new(features.map(|x| x * slope)).sample(&[], &mut rng)?;
        // Conditional on omega, the likelihood is Gaussian in the slope.
        let precision = PRIOR_SD.powi(-2) + omega.mul(&squared_features)?.sum();
        slope = linear_term / precision + rng.next_gaussian() / precision.sqrt();
        if step >= 200 {
            sum += slope;
            squares += slope * slope;
        }
    }
    let mean = sum / draws as f64;
    let gibbs = [mean, (squares / draws as f64 - mean * mean).sqrt()];

    // Integrate the unaugmented model on a grid as an independent posterior reference.
    let b = GraphBuilder::new();
    let slope = b.gaussian(0.0, PRIOR_SD);
    let observed = b.bernoulli_logit(slope * b.constant(features));
    b.observe(observed, BoolTensor::vector(outcomes));
    b.set_value(slope, DoubleTensor::scalar(0.0));
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let param = Parameterisation::new(&model, &state)?;
    let grid: Vec<_> = (-800..=800)
        .map(|i| {
            let beta = f64::from(i) * 0.01;
            Ok((beta, param.log_prob(&[beta], false, &mut state)?))
        })
        .collect::<Result<_>>()?;
    let log_total = log_sum_exp(grid.iter().map(|&(_, lp)| lp));
    let (mean, second_moment) = grid
        .into_iter()
        .fold((0.0, 0.0), |(mean, second), (beta, lp)| {
            let w = (lp - log_total).exp();
            (mean + w * beta, second + w * beta * beta)
        });
    Ok((gibbs, [mean, (second_moment - mean * mean).sqrt()]))
}

fn main() -> Result<()> {
    let ([mean, sd], [reference_mean, reference_sd]) = run(3000)?;
    println!("Gibbs slope mean {mean:.3}, posterior SD {sd:.3}");
    println!("Grid slope mean {reference_mean:.3}, posterior SD {reference_sd:.3}");
    Ok(())
}
