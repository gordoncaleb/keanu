//! Which of two groups does each measurement come from, and what are the groups like?
//!
//! A two-component Gaussian mixture fitted to synthetic data with NUTS. The weights have a Dirichlet prior. A
//! mixture's likelihood is unchanged when its components swap labels, so an unconstrained posterior has two copies
//! of every mode and chains settle in either one; ordering the means (the first below the second) removes the copy.
//! Keanu has no distribution over ordered vectors, so the means are built the way Stan's ordered transform builds
//! them: the first mean, then the first plus a positive gap (a half-Gaussian), stacked and cumulatively summed. NUTS
//! moves the gap through its lower-bound transform, so the ordering always holds.
//!
//! The same model can be written with an explicit categorical assignment of each measurement to a group, which is
//! how the data are generated; `marginalize` sums the assignments out, leaving a model NUTS samples (with the same
//! posterior as the `Mixture` form), and `recover_marginals` gives each measurement's probability of belonging to
//! each group afterwards. [`run_marginalized`] fits that form.
//! Run with `cargo run --release --example gaussian_mixture`.

use keanu::algorithms::ChainSamples;
use keanu::algorithms::marginal::recover_marginals;
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

/// The posterior of the mixture written with explicit assignments and marginalised: the draws of every chain (of
/// the parameters, in the rewritten model), and each measurement's group probabilities.
#[derive(Debug)]
pub struct MarginalizedFit {
    pub fit: Fit,
    /// For each measurement, the posterior probability that it belongs to each group, shape [N, 2].
    pub assignments: DoubleTensor,
}

/// Fits the mixture written with an explicit categorical assignment per measurement, marginalised automatically,
/// with `chains` NUTS chains of `draws` draws each (after as many adaptation steps), and recovers the assignments.
pub fn run_marginalized(
    observations: &[f64],
    chains: usize,
    draws: usize,
    seed: u64,
) -> keanu::Result<MarginalizedFit> {
    let b = GraphBuilder::new();
    let first = b.gaussian_with_shape(&[1], 0.0, 10.0);
    let gap = b.half_gaussian_with_shape(&[1], 10.0);
    let means = b.concat(0, &[first, gap]).cumsum(0).label("means");
    let sigmas = b.half_gaussian_with_shape(&[2], 5.0).label("sigmas");
    let weights = b
        .dirichlet(DoubleTensor::vector(vec![2.0, 2.0]))
        .label("weights");
    // each measurement's group, and its measurement given the group
    let n = observations.len();
    let group = b
        .categorical(&[n], &[weights.take(&[0]), weights.take(&[1])])
        .label("group");
    let y = b.gaussian_with_shape(&[n], means.gather(0, group), sigmas.gather(0, group));
    b.observe(y, DoubleTensor::vector(observations.to_vec()));
    b.set_value(group, IntTensor::vector(vec![0; n]));
    b.set_value(first, DoubleTensor::vector(vec![0.0]));
    b.set_value(gap, DoubleTensor::vector(vec![1.0]));
    b.set_value(sigmas, DoubleTensor::vector(vec![1.0, 1.0]));
    b.set_value(weights, DoubleTensor::vector(vec![0.5, 0.5]));
    let nodes = [first.id(), gap.id(), means.id(), sigmas.id(), weights.id()];
    let group = group.id();
    let (graph, state) = b.build()?;

    // sum the groups out: the measurements' likelihood becomes a factor of the parameters alone
    let (graph, state, marginalized) = marginalize(&graph, &state, &[group])?;
    let nodes: Vec<NodeId> = nodes
        .iter()
        .map(|&id| marginalized.node(id).expect("the parameters are kept"))
        .collect();
    let model = Model::new(graph, &state)?;
    let nuts = Nuts::default().with_adapt_count(draws);
    let chains = sample_chains(chains, seed, &state, |state, rng| {
        let (samples, _) = nuts.sample_with_statistics(
            &model,
            state,
            &nodes,
            2 * draws,
            SamplingOptions::default().with_drop(draws),
            rng,
        )?;
        Ok(samples)
    })?;

    // each measurement's group probabilities given each draw, averaged over the draws of every chain
    let mut total = vec![0.0; 2 * n];
    for chain in chains.chains() {
        let groups = &recover_marginals(&model, &state, &marginalized, chain)?[0];
        for (t, p) in total.iter_mut().zip(groups.mean()?.as_slice()) {
            *t += p / chains.num_chains() as f64;
        }
    }
    let assignments = DoubleTensor::new(&[n, 2], total)?;
    Ok(MarginalizedFit {
        fit: Fit {
            chains,
            means: nodes[2],
            sigmas: nodes[3],
            weights: nodes[4],
        },
        assignments,
    })
}

#[allow(dead_code)]
fn main() -> keanu::Result<()> {
    let observations = data(300, 7);
    let fit = run(&observations, 4, 1000, 11)?;
    println!("{}", fit.chains.summary()?);

    let marginalized = run_marginalized(&observations, 4, 1000, 11)?;
    println!("with the assignments marginalised:");
    println!("{}", marginalized.fit.chains.summary()?);
    for (i, &y) in observations.iter().enumerate().take(5) {
        let p = marginalized.assignments.as_slice()[2 * i + 1];
        println!("measurement {y:.2}: in the second group with probability {p:.3}");
    }
    Ok(())
}
