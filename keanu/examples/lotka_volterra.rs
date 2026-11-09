//! Fit predator and prey populations with a differentiable Lotka–Volterra ODE.
//!
//! Prey grow at rate alpha and are consumed at rate beta per predator. Predators die at rate gamma and grow
//! at rate delta per prey. Observations have multiplicative noise. The last five observation times are held out.
//!
//! Run with `cargo run --release --example lotka_volterra`.

use keanu::graph::ode::OdeOptions;
use keanu::prelude::*;

const INITIAL: [f64; 2] = [1.2, 0.7];
pub const TRUE_RATES: [f64; 4] = [0.9, 0.55, 0.65, 0.45];
const TIMES: usize = 30;
const TRAIN: usize = 25;
const NOISE: f64 = 0.04;

fn derivative([prey, predator]: [f64; 2]) -> [f64; 2] {
    let [alpha, beta, gamma, delta] = TRUE_RATES;
    [
        prey * (alpha - beta * predator),
        predator * (delta * prey - gamma),
    ]
}

/// Reference trajectory from a fixed-step fourth-order Runge–Kutta integrator.
fn reference() -> Vec<f64> {
    let step = 0.0025;
    let mut state = INITIAL;
    let mut result = Vec::with_capacity(2 * TIMES);
    for _ in 0..TIMES {
        for _ in 0..100 {
            let k1 = derivative(state);
            let offset = |rate: [f64; 2], scale: f64| {
                std::array::from_fn(|i| state[i] + step * scale * rate[i])
            };
            let k2 = derivative(offset(k1, 0.5));
            let k3 = derivative(offset(k2, 0.5));
            let k4 = derivative(offset(k3, 1.0));
            state = std::array::from_fn(|i| {
                state[i] + step * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0
            });
        }
        result.extend(state);
    }
    result
}

/// Fit the observed prefix and return the posterior mean rates and held-out trajectory RMSE.
pub fn run(draws: usize) -> keanu::Result<(DoubleTensor, f64)> {
    let mut rng = KeanuRng::seed_from_u64(21);
    let truth = reference();
    let observed = truth[..2 * TRAIN]
        .iter()
        .map(|value| value * (NOISE * rng.next_gaussian()).exp())
        .collect();

    let b = GraphBuilder::new();
    let centres = DoubleTensor::vector(vec![0.8, 0.6, 0.6, 0.5]);
    let rates = b.log_normal(centres.ln(), 0.35).label("rates");
    b.set_value(rates, centres);
    let times = b.constant(DoubleTensor::vector(
        (1..=TIMES).map(|i| 0.25 * i as f64).collect(),
    ));
    let populations = b.ode(
        b.constant(DoubleTensor::vector(INITIAL.to_vec())),
        times,
        b.scalar(0.0),
        &[rates],
        OdeOptions::default(),
        |body, _, population, parameters| {
            let rates = parameters[0];
            let prey = population.take(&[0]);
            let predator = population.take(&[1]);
            body.stack(
                0,
                &[
                    prey * (rates.take(&[0]) - rates.take(&[1]) * predator),
                    predator * (rates.take(&[3]) * prey - rates.take(&[2])),
                ],
            )
        },
    );
    let observation = b.log_normal(populations.slice(&format!(":{TRAIN}, :")).ln(), NOISE);
    b.observe(observation, DoubleTensor::new(&[TRAIN, 2], observed)?);
    let forecast = populations.slice(&format!("{TRAIN}:, :")).id();
    let rates = rates.id();
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let (samples, _) = Nuts::default()
        .with_adapt_count(draws)
        .with_target_acceptance_prob(0.9)
        .sample_with_statistics(
            &model,
            &mut state,
            &[rates, forecast],
            2 * draws,
            SamplingOptions::default().with_drop(draws),
            &mut rng,
        )?;
    let forecast = samples.f64(forecast).unwrap().mean()?;
    let held_out = &truth[2 * TRAIN..];
    let rmse = (forecast
        .as_slice()
        .iter()
        .zip(held_out)
        .map(|(&estimate, &actual)| (estimate - actual).powi(2))
        .sum::<f64>()
        / held_out.len() as f64)
        .sqrt();
    Ok((samples.f64(rates).unwrap().mean()?, rmse))
}

fn main() -> keanu::Result<()> {
    let (rates, rmse) = run(400)?;
    println!(
        "Posterior mean rates [alpha, beta, gamma, delta] {:?}",
        rates.as_slice()
    );
    println!("Held-out trajectory RMSE {rmse:.3}.");
    Ok(())
}
