//! Where is a disease more common than expected, once chance and a known risk factor are accounted for?
//!
//! Disease mapping with the BYM2 model (Riebler et al. 2016; Morris et al. 2019) on simulated counts. Each of 100
//! areas, the cells of a 10 × 10 lattice, has an expected number of cases Eᵢ (from its population's age structure,
//! say) and a covariate xᵢ (a deprivation index, say). The observed counts are
//!
//! `yᵢ ~ Poisson(Eᵢ exp(β₀ + β₁ xᵢ + bᵢ))`,
//!
//! where the area effect b = σ(√(ρ / s) φ + √(1 − ρ) θ) mixes a spatially structured ICAR φ, which makes neighbours
//! alike, with independent noise θ. σ is the effect's typical standard deviation on the log scale and ρ the share of
//! its variance that is spatial; the scaling factor s, computed from the neighbourhood, makes them mean the same
//! whatever the map. exp(β₀ + β₁ xᵢ + bᵢ) is the area's relative risk.
//!
//! The data are simulated from known values ([`TRUE_SIGMA`], [`TRUE_RHO`], [`TRUE_BETA`]) so that the fit can be
//! checked against them. Run with `cargo run --release --example disease_mapping`.

use keanu::algorithms::ChainSamples;
use keanu::algorithms::mcmc::chains::sample_chains;
use keanu::distributions::Icar;
use keanu::prelude::*;
use std::sync::Arc;

/// The lattice's rows and columns.
pub const SIDE: usize = 10;
/// The number of areas.
pub const AREAS: usize = SIDE * SIDE;
/// The effect's total scale the counts are simulated with.
pub const TRUE_SIGMA: f64 = 0.6;
/// The spatial proportion of the effect's variance the counts are simulated with.
pub const TRUE_RHO: f64 = 0.7;
/// The intercept and the covariate's coefficient the counts are simulated with.
pub const TRUE_BETA: [f64; 2] = [-0.2, 0.3];

/// Simulated areal data: the neighbourhood, each area's expected count, covariate and observed count, and the true
/// area effects.
#[derive(Debug, Clone)]
pub struct Areas {
    pub adjacency: Arc<Adjacency>,
    pub expected: Vec<f64>,
    pub covariate: Vec<f64>,
    pub counts: Vec<i64>,
    pub effect: Vec<f64>,
}

/// Simulates the counts of the 10 × 10 lattice from the true parameters.
pub fn data(seed: u64) -> keanu::Result<Areas> {
    let adjacency = Arc::new(Adjacency::lattice(SIDE, SIDE)?);
    let mut rng = KeanuRng::seed_from_u64(seed);
    // expected counts between 5 and 40, and a standardised covariate without spatial structure (one as smooth as
    // the spatial effect would be confounded with it)
    let expected: Vec<f64> = (0..AREAS)
        .map(|_| rng.next_f64_between(5.0, 40.0))
        .collect();
    let covariate: Vec<f64> = (0..AREAS).map(|_| rng.next_gaussian()).collect();
    let phi = Icar::new(&adjacency, DoubleTensor::scalar(1.0)).sample(&[AREAS], &mut rng)?;
    let scaling = adjacency.scaling_factors()?;
    let effect: Vec<f64> = (0..AREAS)
        .map(|i| {
            TRUE_SIGMA
                * ((TRUE_RHO / scaling[i]).sqrt() * phi.as_slice()[i]
                    + (1.0 - TRUE_RHO).sqrt() * rng.next_gaussian())
        })
        .collect();
    let mut counts = Vec::with_capacity(AREAS);
    for i in 0..AREAS {
        let rate = expected[i] * (TRUE_BETA[0] + TRUE_BETA[1] * covariate[i] + effect[i]).exp();
        let draw =
            keanu::distributions::Poisson::new(DoubleTensor::scalar(rate)).sample(&[], &mut rng)?;
        counts.push(draw.as_slice()[0]);
    }
    Ok(Areas {
        adjacency,
        expected,
        covariate,
        counts,
        effect,
    })
}

/// A fit: every chain's draws, and the nodes of the parameters and of the area effects.
#[derive(Debug)]
pub struct Fit {
    pub chains: ChainSamples,
    pub beta0: NodeId,
    pub beta1: NodeId,
    pub sigma: NodeId,
    pub rho: NodeId,
    pub effect: NodeId,
}

/// Fits the BYM2 Poisson model by NUTS, with `chains` chains of `draws` draws after as many warm-up steps.
pub fn run(areas: &Areas, chains: usize, draws: usize, seed: u64) -> keanu::Result<Fit> {
    let b = GraphBuilder::new();
    let beta0 = b.gaussian(0.0, 1.0).label("beta0");
    let beta1 = b.gaussian(0.0, 1.0).label("beta1");
    let sigma = b.half_gaussian(1.0).label("sigma");
    let rho = b.beta(1.0, 1.0).label("rho");
    let bym2 = b.bym2(areas.adjacency.clone(), sigma, rho)?;
    let effect = bym2.effect.label("effect");
    let log_expected = b.constant(DoubleTensor::vector(
        areas.expected.iter().map(|e| e.ln()).collect(),
    ));
    let covariate = b.constant(DoubleTensor::vector(areas.covariate.clone()));
    let y = b.poisson_log(log_expected + beta0 + covariate * beta1 + effect);
    b.observe(y, IntTensor::vector(areas.counts.clone()));
    // start at the prior's centre: no area effect
    b.set_value(sigma, DoubleTensor::scalar(0.5));
    b.set_value(rho, DoubleTensor::scalar(0.5));
    b.set_value(beta0, DoubleTensor::scalar(0.0));
    b.set_value(beta1, DoubleTensor::scalar(0.0));
    b.set_value(bym2.spatial, DoubleTensor::zeros(&[AREAS]));
    b.set_value(bym2.unstructured, DoubleTensor::zeros(&[AREAS]));
    let (beta0, beta1, sigma, rho, effect) =
        (beta0.id(), beta1.id(), sigma.id(), rho.id(), effect.id());
    let (graph, state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let nuts = Nuts::default().with_adapt_count(draws);
    let chains = sample_chains(chains, seed, &state, |state, rng| {
        nuts.sample_with_statistics(
            &model,
            state,
            &[beta0, beta1, sigma, rho, effect],
            2 * draws,
            SamplingOptions::default().with_drop(draws),
            rng,
        )
        .map(|(samples, _)| samples)
    })?;
    Ok(Fit {
        chains,
        beta0,
        beta1,
        sigma,
        rho,
        effect,
    })
}

/// The correlation of the posterior mean area effects with the true ones.
pub fn effect_correlation(areas: &Areas, fit: &Fit) -> keanu::Result<f64> {
    let means: Vec<f64> = (0..AREAS)
        .map(|i| Ok(fit.chains.summarize(fit.effect, &[i])?.mean))
        .collect::<keanu::Result<_>>()?;
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (ma, mb) = (mean(&means), mean(&areas.effect));
    let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for (a, b) in means.iter().zip(&areas.effect) {
        ab += (a - ma) * (b - mb);
        aa += (a - ma) * (a - ma);
        bb += (b - mb) * (b - mb);
    }
    Ok(ab / (aa * bb).sqrt())
}

#[allow(dead_code)]
fn main() -> keanu::Result<()> {
    let areas = data(6)?;
    let total: i64 = areas.counts.iter().sum();
    println!("{AREAS} areas on a {SIDE} × {SIDE} lattice, {total} cases");
    let fit = run(&areas, 4, 2000, 7)?;
    for (name, node, truth) in [
        ("beta0", fit.beta0, TRUE_BETA[0]),
        ("beta1", fit.beta1, TRUE_BETA[1]),
        ("sigma", fit.sigma, TRUE_SIGMA),
        ("rho", fit.rho, TRUE_RHO),
    ] {
        let s = fit.chains.summarize(node, &[])?;
        println!(
            "{name}: mean {:.3}, 90% interval [{:.3}, {:.3}], R-hat {:.3} (truth {truth})",
            s.mean, s.q5, s.q95, s.r_hat
        );
    }
    println!(
        "correlation of the posterior mean area effects with the truth: {:.3}",
        effect_correlation(&areas, &fit)?
    );
    Ok(())
}
