//! Which of two groups does each measurement come from, and what are the groups like?
//!
//! A two-component Gaussian mixture fitted to synthetic data with NUTS. The weights have a Dirichlet prior. A
//! mixture's likelihood is unchanged when its components swap labels, so an unconstrained posterior has two copies
//! of every mode and chains settle in either one; ordering the means (the first below the second) removes the copy.
//! Keanu has no distribution over ordered vectors, so the means are built the way Stan's ordered transform builds
//! them: the first mean, then the first plus a positive gap (a half-Gaussian), stacked and cumulatively summed. NUTS
//! moves the gap through its lower-bound transform, so the ordering always holds.
//! Run with `cargo run --release --example gaussian_mixture`.

use keanu::algorithms::ChainSamples;
use keanu::algorithms::mcmc::chains::sample_chains;
use keanu::graph::op::DistributionKind;
use keanu::prelude::*;

/// The mixture the data are drawn from: 30% from N(-2, 1) and 70% from N(3, 1).
pub const TRUE_WEIGHTS: [f64; 2] = [0.3, 0.7];
pub const TRUE_MEANS: [f64; 2] = [-2.0, 3.0];
pub const TRUE_SIGMA: f64 = 1.0;

/// `count` draws from the true mixture.
pub fn data(count: usize, seed: u64) -> Vec<f64> {
    let mut rng = KeanuRng::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            let mean = if rng.next_f64() < TRUE_WEIGHTS[0] {
                TRUE_MEANS[0]
            } else {
                TRUE_MEANS[1]
            };
            mean + TRUE_SIGMA * rng.next_gaussian()
        })
        .collect()
}

/// The posterior draws of every chain, and the nodes holding the parameters.
#[derive(Debug)]
pub struct Fit {
    pub chains: ChainSamples,
    /// The means, shape [2], increasing.
    pub means: NodeId,
    /// The component scales, shape [2].
    pub sigmas: NodeId,
    /// The weights, shape [2].
    pub weights: NodeId,
}

/// Fits the mixture to `observations` with `chains` NUTS chains of `draws` draws each (after as many adaptation
/// steps).
pub fn run(observations: &[f64], chains: usize, draws: usize, seed: u64) -> keanu::Result<Fit> {
    let b = GraphBuilder::new();
    // ordered means: the first, and a positive gap to the second
    let first = b.gaussian_with_shape(&[1], 0.0, 10.0);
    let gap = b.half_gaussian_with_shape(&[1], 10.0);
    let means = b.concat(0, &[first, gap]).cumsum(0).label("means");
    let sigmas = b.half_gaussian_with_shape(&[2], 5.0).label("sigmas");
    let weights = b
        .dirichlet(DoubleTensor::vector(vec![2.0, 2.0]))
        .label("weights");
    let y: DoubleVar = b.mixture_with_shape(
        &[observations.len()],
        DistributionKind::Gaussian,
        &[means.id(), sigmas.id()],
        weights,
    );
    b.observe(y, DoubleTensor::vector(observations.to_vec()));
    // start every chain from the same plausible point; NUTS adapts from there
    b.set_value(first, DoubleTensor::vector(vec![0.0]));
    b.set_value(gap, DoubleTensor::vector(vec![1.0]));
    b.set_value(sigmas, DoubleTensor::vector(vec![1.0, 1.0]));
    b.set_value(weights, DoubleTensor::vector(vec![0.5, 0.5]));
    let (means, sigmas, weights) = (means.id(), sigmas.id(), weights.id());

    let (graph, state) = b.build()?;
    let model = Model::new(graph, &state)?;
    let nuts = Nuts::default().with_adapt_count(draws);
    let chains = sample_chains(chains, seed, &state, |state, rng| {
        let (samples, _) = nuts.sample_with_statistics(
            &model,
            state,
            &[means, sigmas, weights],
            2 * draws,
            SamplingOptions::default().with_drop(draws),
            rng,
        )?;
        Ok(samples)
    })?;
    Ok(Fit {
        chains,
        means,
        sigmas,
        weights,
    })
}

#[allow(dead_code)]
fn main() -> keanu::Result<()> {
    let observations = data(300, 7);
    let fit = run(&observations, 4, 1000, 11)?;
    println!("{}", fit.chains.summary()?);
    Ok(())
}
