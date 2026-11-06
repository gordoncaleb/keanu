//! What fraction of patients does a treatment cure, and how soon do the others relapse?
//!
//! A mixture cure model (Berkson and Gage 1952; Farewell 1982) for time-to-relapse data with right censoring. Each
//! patient is cured, and never relapses, with probability π, and otherwise relapses after a Weibull time T. A study
//! follows each patient until relapse or until their censoring time c (the end of their follow-up), so the
//! likelihood of
//!
//! - a relapse observed at t < c is (1 − π) f(t);
//! - a patient still relapse-free at c is π + (1 − π) S(c),
//!
//! with f and S the Weibull's density and survival function. Both the cure probability and the Weibull scale depend
//! on a covariate z (a standardised biomarker): π = logistic(γ₀ + γ₁ z) and T ~ Weibull(k, exp(β₀ + β₁ z)).
//!
//! The model is a heterogeneous mixture ([`GraphBuilder::mixture_of`]) of two censored components, with weights
//! (1 − π, π):
//!
//! - the uncured: a Weibull censored above at c;
//! - the cured: a uniform on [c, c + 1] censored above at c. All its mass lies at or beyond the censoring time, so its
//!   survival at c is exactly 1 and its density before c is 0, and the mixture's log density is exactly the cure
//!   model's likelihood above. Any distribution with all its mass beyond c would do; the uniform is a convenient one
//!   with a CDF, so it can be censored.
//!
//! The observed time of a censored patient is c itself, which the censored components read as "at or beyond c".
//! Data for [`PATIENTS`] patients are simulated from known values ([`TRUE_GAMMA`], [`TRUE_BETA`], [`TRUE_K`]) with
//! censoring times uniform between 2 and 8 years, so the fit can be checked against them. Run with
//! `cargo run --release --example cure_model`.

use keanu::algorithms::ChainSamples;
use keanu::algorithms::mcmc::chains::sample_chains;
use keanu::graph::op::DistributionKind;
use keanu::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The number of patients.
pub const PATIENTS: usize = 400;
/// The cure model's intercept and slope (logit scale) the data are simulated with.
pub const TRUE_GAMMA: [f64; 2] = [-0.4, 0.9];
/// The log Weibull scale's intercept and slope the data are simulated with.
pub const TRUE_BETA: [f64; 2] = [0.3, -0.5];
/// The Weibull shape the data are simulated with.
pub const TRUE_K: f64 = 1.6;
/// The seed the example's data are simulated with.
pub const DATA_SEED: u64 = 3;

/// Simulated patients: each one's covariate, censoring time, observed time (the relapse time, or the censoring
/// time if no relapse was seen) and whether a relapse was observed.
#[derive(Debug, Clone)]
pub struct Patients {
    pub z: Vec<f64>,
    pub censoring: Vec<f64>,
    pub time: Vec<f64>,
    pub relapsed: Vec<bool>,
}

/// Simulates the patients from the true parameters.
pub fn data(seed: u64) -> Patients {
    let mut rng = KeanuRng::seed_from_u64(seed);
    let mut patients = Patients {
        z: Vec::with_capacity(PATIENTS),
        censoring: Vec::with_capacity(PATIENTS),
        time: Vec::with_capacity(PATIENTS),
        relapsed: Vec::with_capacity(PATIENTS),
    };
    for _ in 0..PATIENTS {
        let z = rng.next_gaussian();
        let c = rng.next_f64_between(2.0, 8.0);
        let pi = 1.0 / (1.0 + (-(TRUE_GAMMA[0] + TRUE_GAMMA[1] * z)).exp());
        let cured = rng.next_f64() < pi;
        // a Weibull draw by inverting its survival function, exp(−(t / scale)^k)
        let scale = (TRUE_BETA[0] + TRUE_BETA[1] * z).exp();
        let relapse = scale * (-rng.next_f64_non_zero().ln()).powf(1.0 / TRUE_K);
        let relapsed = !cured && relapse < c;
        patients.z.push(z);
        patients.censoring.push(c);
        patients.time.push(if relapsed { relapse } else { c });
        patients.relapsed.push(relapsed);
    }
    patients
}

/// A fit: every chain's draws of the cure model's coefficients ([intercept, slope]), the Weibull scale's
/// coefficients and the Weibull shape.
#[derive(Debug)]
pub struct Fit {
    pub chains: ChainSamples,
    pub gamma: NodeId,
    pub beta: NodeId,
    pub k: NodeId,
    /// The number of divergent transitions after warm-up, over every chain.
    pub divergences: usize,
}

/// Fits the cure model by NUTS, with `chains` chains of `draws` draws after as many warm-up steps.
pub fn run(patients: &Patients, chains: usize, draws: usize, seed: u64) -> keanu::Result<Fit> {
    let b = GraphBuilder::new();
    let gamma = b.gaussian_with_shape(&[2], 0.0, 2.0).label("gamma");
    let beta = b.gaussian_with_shape(&[2], 0.0, 2.0).label("beta");
    let k = b.half_gaussian(3.0).label("k");
    let z = b.constant(DoubleTensor::vector(patients.z.clone()));
    let c = b.constant(DoubleTensor::vector(patients.censoring.clone()));
    let at = |i: i64| IntTensor::scalar(i);
    let pi = (gamma.gather(0, at(0)) + gamma.gather(0, at(1)) * z).sigmoid();
    let scale = (beta.gather(0, at(0)) + beta.gather(0, at(1)) * z).exp();
    // the weights of the uncured and the cured, [patients, 2]
    let weights = b.stack(-1, &[-pi + 1.0, pi]);
    let unbounded = b.scalar(f64::NEG_INFINITY);
    let uncured = [k.id(), scale.id(), unbounded.id(), c.id()];
    let cured = [c.id(), (c + 1.0).id(), unbounded.id(), c.id()];
    let censored = |kind| DistributionKind::Censored(Arc::new(kind));
    let time: DoubleVar = b.mixture_of(
        &[
            (censored(DistributionKind::Weibull), &uncured),
            (censored(DistributionKind::Uniform), &cured),
        ],
        weights,
    );
    b.observe(time, DoubleTensor::vector(patients.time.clone()));
    b.set_value(gamma, DoubleTensor::vector(vec![0.0, 0.0]));
    b.set_value(beta, DoubleTensor::vector(vec![0.0, 0.0]));
    b.set_value(k, DoubleTensor::scalar(1.0));
    let (gamma, beta, k) = (gamma.id(), beta.id(), k.id());
    let (graph, state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let nuts = Nuts::default().with_adapt_count(draws);
    let divergences = AtomicUsize::new(0);
    let chains = sample_chains(chains, seed, &state, |state, rng| {
        nuts.sample_with_statistics(
            &model,
            state,
            &[gamma, beta, k],
            2 * draws,
            SamplingOptions::default().with_drop(draws),
            rng,
        )
        .map(|(samples, statistics)| {
            divergences.fetch_add(statistics.divergent_steps.len(), Ordering::Relaxed);
            samples
        })
    })?;
    Ok(Fit {
        chains,
        gamma,
        beta,
        k,
        divergences: divergences.into_inner(),
    })
}

#[allow(dead_code)]
fn main() -> keanu::Result<()> {
    let patients = data(DATA_SEED);
    let relapses = patients.relapsed.iter().filter(|&&r| r).count();
    println!("{PATIENTS} patients, {relapses} relapses observed");
    let fit = run(&patients, 4, 1000, 1)?;
    println!("{} divergent transitions", fit.divergences);
    let report = |name: &str, node: NodeId, index: &[usize], truth: f64| -> keanu::Result<()> {
        let s = fit.chains.summarize(node, index)?;
        println!(
            "{name}: mean {:.3}, 90% interval [{:.3}, {:.3}], R-hat {:.3} (truth {truth:.3})",
            s.mean, s.q5, s.q95, s.r_hat
        );
        Ok(())
    };
    report("gamma0 (cure, intercept)", fit.gamma, &[0], TRUE_GAMMA[0])?;
    report("gamma1 (cure, slope)", fit.gamma, &[1], TRUE_GAMMA[1])?;
    report("beta0 (log scale, intercept)", fit.beta, &[0], TRUE_BETA[0])?;
    report("beta1 (log scale, slope)", fit.beta, &[1], TRUE_BETA[1])?;
    report("k (shape)", fit.k, &[], TRUE_K)?;
    Ok(())
}
