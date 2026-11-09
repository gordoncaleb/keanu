//! Fit time to an event with a shared group frailty and right-censored follow-up.
//!
//! The Gaussian group effect shifts log event times for all members of a group. Exponentiation produces
//! positive times. Observations at the follow-up deadline contribute survival probabilities.
//!
//! Run with `cargo run --release --example censored_frailty`.

use keanu::graph::op::{DistributionKind, ElementBijector};
use keanu::prelude::*;
use std::sync::Arc;

const GROUPS: usize = 12;
const GROUP_SIZE: usize = 16;
const DEADLINE: f64 = 1.6;
const NOISE: f64 = 0.35;
pub const TRUE_EFFECT: f64 = -0.55;

/// Return the posterior mean treatment effect on log time and the number of right-censored observations.
pub fn run(draws: usize) -> Result<(f64, usize)> {
    let mut rng = KeanuRng::seed_from_u64(73);
    let count = GROUPS * GROUP_SIZE;
    let mut groups = Vec::with_capacity(count);
    let mut treatment = Vec::with_capacity(count);
    let mut times = Vec::with_capacity(count);
    for group in 0..GROUPS {
        let frailty = 0.4 * rng.next_gaussian();
        for subject in 0..GROUP_SIZE {
            let treated = (subject % 2) as f64;
            let time = (0.5 + frailty + TRUE_EFFECT * treated + NOISE * rng.next_gaussian()).exp();
            groups.push(group as i64);
            treatment.push(treated);
            times.push(time.min(DEADLINE));
        }
    }
    let censored_count = times.iter().filter(|&&time| time == DEADLINE).count();

    let b = GraphBuilder::new();
    let baseline = b.gaussian(0.0, 1.0);
    let effect = b.gaussian(0.0, 1.0);
    let frailty_scale = b.log_normal(0.4_f64.ln(), 0.55);
    let offsets = b.gaussian_with_shape(&[GROUPS], 0.0, 1.0);
    b.set_value(baseline, DoubleTensor::scalar(0.4));
    b.set_value(effect, DoubleTensor::scalar(-0.4));
    b.set_value(frailty_scale, DoubleTensor::scalar(0.4));
    b.set_value(offsets, DoubleTensor::zeros(&[GROUPS]));
    let groups = b.int_constant(IntTensor::vector(groups));
    let treatment = b.constant(DoubleTensor::vector(treatment));
    let mu = baseline + effect * treatment + frailty_scale * offsets.gather(0, groups);
    let kind = DistributionKind::Transformed(
        Arc::new(DistributionKind::Gaussian),
        ElementBijector::LowerBound(0.0),
    );
    let observed: DoubleVar = b.censored(
        kind,
        &[mu.id(), b.scalar(NOISE).id()],
        f64::NEG_INFINITY,
        DEADLINE,
    );
    b.observe(observed, DoubleTensor::vector(times));
    let effect = effect.id();
    let (graph, mut state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let (samples, _) = Nuts::default()
        .with_adapt_count(draws)
        .sample_with_statistics(
            &model,
            &mut state,
            &[effect],
            2 * draws,
            SamplingOptions::default().with_drop(draws),
            &mut rng,
        )?;
    Ok((
        samples.f64(effect).unwrap().mean()?.to_scalar().unwrap(),
        censored_count,
    ))
}

fn main() -> Result<()> {
    let (effect, censored) = run(500)?;
    println!("Posterior mean log-time effect {effect:.3}. Right-censored observations {censored}.");
    Ok(())
}
