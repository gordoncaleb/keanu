//! Infer the drift and diffusion of a noisy mean-reverting stochastic process.
//!
//! Euler–Maruyama discretizes dX = −rate × X dt + diffusion dW. Standard-normal innovations are drawn as
//! graph latents before the deterministic scan. The scan then propagates each proposed path.
//!
//! Run with `cargo run --release --example euler_maruyama`.

use keanu::prelude::*;

const STEPS: usize = 100;
const DT: f64 = 0.25;
const OBS_NOISE: f64 = 0.07;
pub const RATE: f64 = 0.9;
pub const DIFFUSION: f64 = 0.55;

/// Fit the simulated observations and return posterior mean rate and diffusion.
pub fn run(draws: usize) -> Result<(f64, f64)> {
    let mut rng = KeanuRng::seed_from_u64(81);
    let mut position = 0.0;
    let observed = (0..STEPS)
        .map(|_| {
            position = (1.0 - RATE * DT) * position + DIFFUSION * DT.sqrt() * rng.next_gaussian();
            position + OBS_NOISE * rng.next_gaussian()
        })
        .collect();

    let b = GraphBuilder::new();
    let rate = b.log_normal(0.0, 0.45).label("rate");
    let diffusion = b.log_normal(0.5_f64.ln(), 0.35).label("diffusion");
    let innovations = b.gaussian_with_shape(&[STEPS], 0.0, 1.0);
    b.set_value(rate, DoubleTensor::scalar(1.0));
    b.set_value(diffusion, DoubleTensor::scalar(0.5));
    b.set_value(innovations, DoubleTensor::zeros(&[STEPS]));
    let path = b.scan(
        b.scalar(0.0),
        &[innovations],
        &[rate, diffusion],
        |_, x, noise, parameters| {
            (1.0 - parameters[0] * DT) * x + parameters[1] * DT.sqrt() * noise[0]
        },
    );
    b.observe(b.gaussian(path, OBS_NOISE), DoubleTensor::vector(observed));
    let (rate, diffusion) = (rate.id(), diffusion.id());
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let (samples, _) = Nuts::default()
        .with_adapt_count(draws)
        .sample_with_statistics(
            &model,
            &mut state,
            &[rate, diffusion],
            2 * draws,
            SamplingOptions::default().with_drop(draws),
            &mut rng,
        )?;
    Ok((
        samples.f64(rate).unwrap().mean()?.to_scalar().unwrap(),
        samples.f64(diffusion).unwrap().mean()?.to_scalar().unwrap(),
    ))
}

fn main() -> Result<()> {
    let (rate, diffusion) = run(500)?;
    println!("Posterior mean reversion rate {rate:.3}, diffusion {diffusion:.3}.");
    Ok(())
}
