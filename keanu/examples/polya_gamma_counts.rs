//! Negative-binomial regression with noninteger Pólya–Gamma augmentation.
//!
//! Run with `cargo run --release --example polya_gamma_counts`.

use keanu::distributions::{NegativeBinomialLog, PolyaGamma};
use keanu::prelude::*;
use keanu::tensor::special::log_sum_exp;

const DISPERSION: f64 = 1.7;
const PRIOR_SD: f64 = 1.5;

/// Gibbs and grid estimates of the slope posterior's mean and standard deviation.
pub fn run(draws: usize) -> Result<([f64; 2], [f64; 2])> {
    assert!(draws > 0);
    let mut rng = KeanuRng::seed_from_u64(634);
    let x = DoubleTensor::vector((0..64).map(|i| f64::from(i) / 21.0 - 1.5).collect());
    let dispersion = DoubleTensor::scalar(DISPERSION);
    let log_mu = |slope| x.map(|x| DISPERSION.ln() + slope * x);
    let y = NegativeBinomialLog::new(log_mu(0.75), &dispersion).sample(&[], &mut rng)?;
    let b = y.to_double().map(|y| y + DISPERSION);
    let kappa = y.to_double().map(|y| 0.5 * (y - DISPERSION));
    let linear = kappa.mul(&x)?.sum();
    let x_squared = x.map(|x| x * x);
    let mut slope = 0.0;
    let mut mean = 0.0;
    let mut second = 0.0;
    for i in 0..draws + 300 {
        let omega = PolyaGamma::new(&b, x.map(|x| x * slope)).sample(&[], &mut rng)?;
        let precision = PRIOR_SD.powi(-2) + omega.mul(&x_squared)?.sum();
        slope = linear / precision + rng.next_gaussian() / precision.sqrt();
        if i >= 300 {
            mean += slope / draws as f64;
            second += slope * slope / draws as f64;
        }
    }
    let gibbs = [mean, (second - mean * mean).sqrt()];

    // The original likelihood gives an independent reference for the augmented fit.
    let grid: Vec<_> = (-300..=300)
        .map(|i| {
            let slope = f64::from(i) * 0.01;
            let likelihood = NegativeBinomialLog::new(log_mu(slope), &dispersion)
                .log_prob(&y)?
                .sum();
            Ok((slope, likelihood - 0.5 * (slope / PRIOR_SD).powi(2)))
        })
        .collect::<Result<_>>()?;
    let total = log_sum_exp(grid.iter().map(|&(_, lp)| lp));
    let (mean, second) = grid.into_iter().fold((0.0, 0.0), |(m, s), (x, lp)| {
        let p = (lp - total).exp();
        (m + p * x, s + p * x * x)
    });
    Ok((gibbs, [mean, (second - mean * mean).sqrt()]))
}

fn main() -> Result<()> {
    let ([mean, sd], [reference_mean, reference_sd]) = run(4000)?;
    println!("Gibbs slope mean {mean:.3}, posterior SD {sd:.3}");
    println!("Grid slope mean {reference_mean:.3}, posterior SD {reference_sd:.3}");
    Ok(())
}
