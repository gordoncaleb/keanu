//! Pool binomial A/B tests across markets and predict the treatment effect in a new market.
//!
//! Market treatment effects are normal on the log-odds scale, with a common mean and an unknown
//! between-market standard deviation. The new market inherits uncertainty about both quantities.
//!
//! Run with `cargo run --release --example hierarchical_experiments`.

use keanu::prelude::*;

const TRIALS: i64 = 350;
const CONTROL_LOGIT: f64 = -1.2;
const EFFECTS: [f64; 10] = [-0.4, -0.1, 0.1, 0.35, 0.4, 0.55, 0.7, 0.8, 1.0, 1.1];
pub const TRUE_MEAN_EFFECT: f64 = 0.45;

/// Return the population mean effect and the posterior standard deviations for the mean and a new market.
pub fn run(draws: usize) -> Result<(f64, f64, f64)> {
    let mut rng = KeanuRng::seed_from_u64(94);
    let mut count = |logit: f64| {
        let probability = 1.0 / (1.0 + (-logit).exp());
        (0..TRIALS).filter(|_| rng.next_f64() < probability).count() as i64
    };
    let control = EFFECTS.iter().map(|_| count(CONTROL_LOGIT)).collect();
    let treated = EFFECTS
        .iter()
        .map(|&effect| count(CONTROL_LOGIT + effect))
        .collect();

    let b = GraphBuilder::new();
    let baseline = b.gaussian(0.0, 2.0);
    let mean = b.gaussian(0.0, 1.0);
    let spread = b.log_normal(0.5_f64.ln(), 0.6);
    let offset = b.gaussian_with_shape(&[EFFECTS.len()], 0.0, 1.0);
    b.set_value(baseline, DoubleTensor::scalar(-1.0));
    b.set_value(mean, DoubleTensor::scalar(0.0));
    b.set_value(spread, DoubleTensor::scalar(0.5));
    b.set_value(offset, DoubleTensor::zeros(&[EFFECTS.len()]));
    b.observe(
        b.binomial_logit_with_shape(&[EFFECTS.len()], baseline, TRIALS),
        IntTensor::vector(control),
    );
    b.observe(
        b.binomial_logit(baseline + mean + spread * offset, TRIALS),
        IntTensor::vector(treated),
    );
    let (mean, spread) = (mean.id(), spread.id());
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let (samples, _) = Nuts::default()
        .with_adapt_count(draws)
        .sample_with_statistics(
            &model,
            &mut state,
            &[mean, spread],
            2 * draws,
            SamplingOptions::default().with_drop(draws),
            &mut rng,
        )?;
    let means = samples.f64(mean).unwrap();
    let spreads = samples.f64(spread).unwrap();
    // Var(new effect | data) = Var(population mean | data) + E[spread² | data].
    let mean_variance = means.variance()?.to_scalar().unwrap();
    let expected_spread_squared = spreads
        .as_slice()
        .iter()
        .map(|s| s.as_slice()[0].powi(2))
        .sum::<f64>()
        / spreads.as_slice().len() as f64;
    Ok((
        means.mean()?.to_scalar().unwrap(),
        mean_variance.sqrt(),
        (mean_variance + expected_spread_squared).sqrt(),
    ))
}

fn main() -> Result<()> {
    let (mean, mean_sd, new_market_sd) = run(600)?;
    println!("Population mean log odds ratio {mean:.3} (posterior SD {mean_sd:.3}).");
    println!("New-market log odds ratio predictive SD {new_market_sd:.3}.");
    Ok(())
}
