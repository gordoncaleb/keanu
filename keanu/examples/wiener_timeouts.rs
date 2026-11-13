//! Fit a drift-diffusion model when a deadline leaves some response choices unobserved.
//!
//! Run with `cargo run --release --example wiener_timeouts`.

use keanu::distributions::sample_wiener;
use keanu::graph::op::DistributionKind;
use keanu::prelude::*;
use keanu::tensor::special::log_sum_exp;
use std::sync::Arc;

const TRIALS: usize = 500;
const DEADLINE: f64 = 0.50;
const SW: f64 = 0.18;
const ST0: f64 = 0.10;
/// Mean drift used to simulate the trials.
pub const DRIFT: f64 = 1.1;

/// Posterior mean drift, observed timeout fraction, and posterior predictive timeout probability.
pub fn run() -> Result<(f64, f64, f64)> {
    let s = DoubleTensor::scalar;
    let (a, t0, w, sv) = (s(1.2), s(0.2), s(0.43), s(0.5));
    let mut rng = KeanuRng::seed_from_u64(322);
    let starts = rng
        .uniform(&[TRIALS])
        .map(|u| w.as_slice()[0] + (u - 0.5) * SW);
    let delays = rng.uniform(&[TRIALS]).map(|u| t0.as_slice()[0] + ST0 * u);
    let (times, choices) =
        sample_wiener(&a, &delays, &starts, &s(DRIFT), &sv, &[TRIALS], &mut rng)?;
    let (mut observed_times, mut observed_choices) = (Vec::new(), Vec::new());
    for (&time, &choice) in times.as_slice().iter().zip(choices.as_slice()) {
        if time <= DEADLINE {
            observed_times.push(time);
            observed_choices.push(choice);
        }
    }
    let timeouts = TRIALS - observed_times.len();
    let times = DoubleTensor::vector(observed_times);
    let choices = IntTensor::vector(observed_choices);
    let b = GraphBuilder::new();
    let (a, t0, w, sv) = (b.constant(a), b.constant(t0), b.constant(w), b.constant(sv));
    let (sw, st0) = (b.scalar(SW), b.scalar(ST0));
    let drift = b.gaussian(0.0, 2.0);
    b.set_value(drift, s(0.0));
    let response = b.wiener_full(a, t0, w, drift, sv, sw, st0, b.int_constant(choices));
    b.observe(response, times);

    // At the deadline the boundary is unknown, so add the joint survival terms.
    let boundaries = b.int_constant(IntTensor::vector(vec![0, 1]));
    let lower = b.scalar(f64::NEG_INFINITY);
    let deadline = b.scalar(DEADLINE);
    let tails = b.log_density(
        DistributionKind::Censored(Arc::new(DistributionKind::WienerFull)),
        deadline,
        &[
            a.id(),
            t0.id(),
            w.id(),
            drift.id(),
            sv.id(),
            sw.id(),
            st0.id(),
            boundaries.id(),
            lower.id(),
            deadline.id(),
        ],
    );
    let timeout = tails.log_sum_exp();
    b.factor(timeouts as f64 * timeout);
    let timeout_id = timeout.id();
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let param = Parameterisation::new(&model, &state)?;

    // One dimension lets us normalize the drift posterior directly on a grid.
    let mut grid = Vec::new();
    for i in -60..=80 {
        let v = 0.05 * f64::from(i);
        let log_posterior = param.log_prob(&[v], false, &mut state)?;
        let timeout = state
            .f64(timeout_id)
            .expect("timeout was evaluated")
            .as_slice()[0]
            .exp();
        grid.push((v, timeout, log_posterior));
    }
    let normalizer = log_sum_exp(grid.iter().map(|&(_, _, log_p)| log_p));
    let (mean, prediction) = grid
        .into_iter()
        .fold((0.0, 0.0), |(mean, risk), (v, timeout, lp)| {
            let weight = (lp - normalizer).exp();
            (mean + weight * v, risk + weight * timeout)
        });
    Ok((mean, timeouts as f64 / TRIALS as f64, prediction))
}

fn main() -> Result<()> {
    let (mean, observed, predicted) = run()?;
    println!("Posterior mean drift {mean:.3} (simulated {DRIFT})");
    println!("Timeout fraction {observed:.3}, posterior predictive {predicted:.3}");
    Ok(())
}
