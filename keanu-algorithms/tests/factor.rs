//! Factor nodes in inference: NUTS and Metropolis–Hastings sample the density a factor adds.

mod mcmc_common;

use keanu_algorithms::mcmc::SamplingOptions;
use keanu_algorithms::mcmc::metropolis::MetropolisHastings;
use keanu_algorithms::mcmc::nuts::Nuts;
use keanu_algorithms::mcmc::proposal::GaussianProposal;
use keanu_graph::{GraphBuilder, Model, NodeId, State};
use keanu_tensor::DoubleTensor;
use mcmc_common::*;

/// x with a flat prior on [-20, 20) and a factor -x²/2: a standard Gaussian (its tails beyond ±20 are negligible).
fn standard_gaussian_by_a_factor() -> (Model, State, NodeId) {
    let b = GraphBuilder::new();
    let x = b.uniform(-20.0, 20.0);
    b.factor(x * x * -0.5);
    b.set_value(x, DoubleTensor::scalar(1.0));
    let x = x.id();
    let (graph, mut state) = b.build().unwrap();
    let all: Vec<NodeId> = graph.ids().collect();
    state.lazy_eval(&graph, &all, &mut rng(0)).unwrap();
    let model = Model::new(graph, &state).unwrap();
    (model, state, x)
}

#[test]
fn nuts_samples_a_standard_gaussian_made_by_a_factor() {
    let (model, mut state, x) = standard_gaussian_by_a_factor();
    let samples = Nuts::default()
        .with_adapt_count(500)
        .sample_with_statistics(
            &model,
            &mut state,
            &[x],
            4000,
            SamplingOptions::default()
                .with_drop(500)
                .with_down_sample_interval(1),
            &mut rng(11),
        )
        .unwrap()
        .0;
    assert_moments(&series(&samples, x), 0.0, 1.0, "NUTS with a factor");
}

#[test]
fn metropolis_hastings_respects_a_factor() {
    let (model, mut state, x) = standard_gaussian_by_a_factor();
    let samples = MetropolisHastings::default()
        .proposal(GaussianProposal::new(1.5))
        .sample_with(
            &model,
            &mut state,
            &[x],
            30_000,
            SamplingOptions::default()
                .with_drop(1000)
                .with_down_sample_interval(1),
            &mut rng(12),
        )
        .unwrap();
    assert_moments(&series(&samples, x), 0.0, 1.0, "MH with a factor");
}
