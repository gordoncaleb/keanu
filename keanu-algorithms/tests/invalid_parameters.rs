//! Metropolis–Hastings on models whose latent variables can be proposed outside the valid region of a
//! distribution's parameters (non-increasing ordered logistic cutpoints, mixture weights or hidden Markov model
//! transition probabilities off the simplex). Such a state is impossible (log probability -infinity), so the
//! proposal is rejected, rather than the evaluation failing and aborting the sampler.

use keanu_algorithms::mcmc::metropolis::MetropolisHastings;
use keanu_algorithms::mcmc::proposal::GaussianProposal;
use keanu_distributions::KeanuRng;
use keanu_graph::op::DistributionKind;
use keanu_graph::{DoubleVar, GraphBuilder, Model, NodeId, State};
use keanu_tensor::IntTensor;

mod common;
use common::v;

/// Runs Metropolis–Hastings with wide Gaussian proposals on `latent` and returns its sampled values (one vector per
/// sample) and the acceptance rate.
fn run(model: &Model, state: &mut State, latent: NodeId, len: usize) -> (Vec<Vec<f64>>, f64) {
    let samples = MetropolisHastings::default()
        .proposal(GaussianProposal::new(1.0))
        .sample(
            model,
            state,
            &[latent],
            2000,
            &mut KeanuRng::seed_from_u64(7),
        )
        .unwrap();
    let values = samples.f64(latent).unwrap();
    let columns: Vec<Vec<f64>> = (0..len).map(|i| values.element(&[i]).unwrap()).collect();
    let rows: Vec<Vec<f64>> = (0..columns[0].len())
        .map(|s| columns.iter().map(|c| c[s]).collect())
        .collect();
    let distinct = rows.windows(2).filter(|w| w[0] != w[1]).count();
    let acceptance = distinct as f64 / (rows.len() - 1) as f64;
    (rows, acceptance)
}

#[test]
fn proposals_of_cutpoints_that_do_not_increase_are_rejected() {
    let b = GraphBuilder::new();
    let cutpoints = b.gaussian_with_shape(&[2], v(&[-1.0, 1.0]), 1.0);
    b.set_value(cutpoints, v(&[-1.0, 1.0]));
    let y = b.ordered_logistic(v(&[0.0; 4]), cutpoints);
    b.observe(y, IntTensor::vector(vec![0, 1, 1, 2]));
    let cutpoints = cutpoints.id();
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let (samples, acceptance) = run(&model, &mut state, cutpoints, 2);
    assert!(
        samples.iter().all(|c| c[0] < c[1]),
        "a sample has cutpoints that don't increase"
    );
    assert!(
        acceptance > 0.05 && acceptance < 0.95,
        "acceptance {acceptance}"
    );
}

#[test]
fn proposals_of_mixture_weights_off_the_simplex_are_rejected() {
    let b = GraphBuilder::new();
    let p = b.gaussian_with_shape(&[1], 0.5, 0.5);
    b.set_value(p, v(&[0.5]));
    let weights = b.concat(0, &[p, 1.0 - p]);
    let (mu, sigma) = (b.constant(v(&[-1.0, 1.0])), b.scalar(1.0));
    let x: DoubleVar = b.mixture_with_shape(
        &[3],
        DistributionKind::Gaussian,
        &[mu.id(), sigma.id()],
        weights,
    );
    b.observe(x, v(&[0.8, 1.2, -0.3]));
    let p = p.id();
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let (samples, acceptance) = run(&model, &mut state, p, 1);
    assert!(
        samples.iter().all(|p| (0.0..=1.0).contains(&p[0])),
        "a sample has weights off the simplex"
    );
    assert!(
        acceptance > 0.05 && acceptance < 0.95,
        "acceptance {acceptance}"
    );
}

#[test]
fn proposals_of_transition_probabilities_off_the_simplex_are_rejected() {
    let b = GraphBuilder::new();
    let stay = b.gaussian_with_shape(&[1], 0.8, 0.5);
    b.set_value(stay, v(&[0.8]));
    let leave = 1.0 - stay;
    let transition = b.concat(0, &[stay, leave, leave, stay]).reshape(&[2, 2]);
    let (mu, sigma) = (b.constant(v(&[-1.0, 1.0])), b.scalar(1.0));
    let x: DoubleVar = b.hidden_markov_model(
        4,
        DistributionKind::Gaussian,
        &[mu.id(), sigma.id()],
        v(&[0.5, 0.5]),
        transition,
    );
    b.observe(x, v(&[-1.2, -0.8, 0.9, 1.1]));
    let stay = stay.id();
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let (samples, acceptance) = run(&model, &mut state, stay, 1);
    assert!(
        samples.iter().all(|p| (0.0..=1.0).contains(&p[0])),
        "a sample has transition probabilities off the simplex"
    );
    assert!(
        acceptance > 0.05 && acceptance < 0.95,
        "acceptance {acceptance}"
    );
}
