//! The pointwise log likelihood: its points and names, its values against the model's log likelihood at each draw,
//! and the samples it needs.

mod mcmc_common;

use keanu_algorithms::loglik::PointwiseLogLik;
use keanu_algorithms::mcmc::chains::sample_chains;
use keanu_algorithms::mcmc::metropolis::MetropolisHastings;
use keanu_algorithms::mcmc::nuts::Nuts;
use keanu_algorithms::{AlgorithmError, NetworkSamples};
use keanu_graph::op::DistributionKind;
use keanu_graph::{DoubleVar, GraphBuilder, Model, NodeId, State, node_log_prob};
use keanu_tensor::{BoolTensor, DoubleTensor, IntTensor};
use mcmc_common::*;

/// A regression: y[i] ~ N(a + b x[i], sigma) for five points, with a scalar observation z ~ N(a, 1) too.
fn regression() -> (Model, State, Vec<NodeId>) {
    let b = GraphBuilder::new();
    let a = b.gaussian(0.0, 5.0).label("a");
    let slope = b.gaussian(0.0, 5.0).label("b");
    let sigma = b.half_gaussian(2.0).label("sigma");
    let x = b.constant(DoubleTensor::vector(vec![-1.0, -0.5, 0.0, 0.5, 1.0]));
    let y = b.gaussian(a + slope * x, sigma).label("y");
    b.observe(y, DoubleTensor::vector(vec![-1.2, -0.4, 0.3, 0.6, 1.4]));
    let z = b.gaussian(a, 1.0).label("z");
    b.observe(z, DoubleTensor::scalar(0.2));
    let latents = vec![a.id(), slope.id(), sigma.id()];
    let (graph, state) = b.build().unwrap();
    let mut state = state;
    evaluate(&graph, &mut state);
    let model = Model::new(graph, &state).unwrap();
    (model, state, latents)
}

fn evaluate(graph: &keanu_graph::Graph, state: &mut State) {
    let all: Vec<NodeId> = graph.ids().collect();
    state.lazy_eval(graph, &all, &mut rng(0)).unwrap();
}

/// Checks that the points of every draw sum to the model's log likelihood there (less `offset`, the factors).
fn assert_sums_to_log_likelihood(
    model: &Model,
    state: &State,
    chains: &[NetworkSamples],
    loglik: &PointwiseLogLik,
    factor: impl Fn(&State) -> f64,
) {
    let graph = model.graph();
    let totals = loglik.totals();
    for (c, chain) in chains.iter().enumerate() {
        assert_eq!(totals[c].len(), chain.len());
        let mut work = state.clone();
        let all: Vec<NodeId> = graph.ids().collect();
        work.lazy_eval(graph, &all, &mut rng(0)).unwrap();
        for (d, &actual) in totals[c].iter().enumerate() {
            let sample = chain.sample(d).unwrap();
            let values: Vec<_> = chain
                .nodes()
                .iter()
                .map(|&id| (id, sample.get(id).unwrap()))
                .collect();
            model.set_and_cascade(&mut work, values).unwrap();
            let expected = model.log_likelihood(&work).unwrap() - factor(&work);
            assert!(
                (actual - expected).abs() <= 1e-10 * expected.abs().max(1.0),
                "chain {c} draw {d}: {actual} vs {expected}"
            );
        }
    }
}

#[test]
fn the_points_of_each_draw_sum_to_the_log_likelihood() {
    let (model, state, latents) = regression();
    let nuts = Nuts::default().with_adapt_count(100);
    let chains = sample_chains(3, 5, &state, |state, rng| {
        nuts.sample(&model, state, &latents, 60, rng)
    })
    .unwrap();
    let loglik = PointwiseLogLik::from_chains(&model, &state, &chains).unwrap();
    assert_eq!(
        loglik.names(),
        &["y[0]", "y[1]", "y[2]", "y[3]", "y[4]", "z"]
    );
    assert_eq!(loglik.num_chains(), 3);
    assert_eq!(loglik.draws(), 60);
    assert_eq!(loglik.len(), 6);
    assert_eq!(loglik.points()[3].index, vec![3]);
    assert!(loglik.points()[5].index.is_empty());
    assert_eq!(loglik.point(2).len(), 180);
    assert_eq!(loglik.chain(1).unwrap()[7][2], loglik.point(2)[60 + 7]);
    assert_sums_to_log_likelihood(&model, &state, chains.chains(), &loglik, |_| 0.0);
}

#[test]
fn one_chain_of_samples_is_one_chain() {
    let (model, mut state, latents) = regression();
    let samples = MetropolisHastings::default()
        .sample(&model, &mut state, &latents, 200, &mut rng(3))
        .unwrap();
    let loglik = PointwiseLogLik::from_samples(&model, &state, &samples).unwrap();
    assert_eq!((loglik.num_chains(), loglik.draws()), (1, 200));
    assert_sums_to_log_likelihood(
        &model,
        &state,
        std::slice::from_ref(&samples),
        &loglik,
        |_| 0.0,
    );
}

#[test]
fn a_multivariate_gaussian_has_one_point_per_node() {
    let b = GraphBuilder::new();
    let mu = b.gaussian_with_shape(&[3], 0.0, 1.0).label("mu");
    let covariance = b.constant(DoubleTensor::eye(3).unwrap());
    let single: DoubleVar = b.multivariate_gaussian(mu, covariance).label("single");
    b.observe(single, DoubleTensor::vector(vec![0.1, 0.2, 0.3]));
    // a batch of two observed vectors, from a batched mean
    let offset =
        b.constant(DoubleTensor::new(&[2, 3], vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap());
    let batch: DoubleVar = b
        .multivariate_gaussian(mu + offset, covariance)
        .label("batch");
    b.observe(
        batch,
        DoubleTensor::new(&[2, 3], vec![0.0, 0.4, -0.2, 1.3, 0.8, 1.1]).unwrap(),
    );
    let (mu, single, batch) = (mu.id(), single.id(), batch.id());
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state);
    let model = Model::new(graph, &state).unwrap();
    let nuts = Nuts::default().with_adapt_count(50);
    let chains = sample_chains(2, 1, &state, |state, rng| {
        nuts.sample(&model, state, &[mu], 40, rng)
    })
    .unwrap();
    let loglik = PointwiseLogLik::from_chains(&model, &state, &chains).unwrap();
    assert_eq!(loglik.names(), &["single", "batch"]);
    assert_eq!(loglik.points()[0].node, single);
    assert_eq!(loglik.points()[1].node, batch);
    assert_sums_to_log_likelihood(&model, &state, chains.chains(), &loglik, |_| 0.0);
}

#[test]
fn element_wise_kinds_have_one_point_per_element() {
    // Bernoulli, Poisson, a mixture, a truncated Gaussian and an unlabelled matrix of Gaussians
    let b = GraphBuilder::new();
    let p = b.uniform(0.0, 1.0).label("p");
    let flips = b.bernoulli_with_shape(&[3], p).label("flips");
    b.observe(flips, BoolTensor::vector(vec![true, false, true]));
    let rate = b.half_gaussian(5.0).label("rate");
    let counts = b.poisson_with_shape(&[2], rate).label("counts");
    b.observe(counts, IntTensor::vector(vec![3, 6]));
    let means = b.constant(DoubleTensor::vector(vec![-2.0, 2.0]));
    let sigmas = b.constant(DoubleTensor::vector(vec![1.0, 1.0]));
    let weights = b
        .dirichlet(DoubleTensor::vector(vec![1.0, 1.0]))
        .label("weights");
    let mixed: DoubleVar = b.mixture_with_shape(
        &[2],
        DistributionKind::Gaussian,
        &[means.id(), sigmas.id()],
        weights,
    );
    b.observe(mixed, DoubleTensor::vector(vec![-1.5, 2.5]));
    let mu = b.gaussian(0.0, 1.0).label("mu");
    let truncated: DoubleVar = b.truncated(
        DistributionKind::Gaussian,
        &[mu.id(), b.scalar(1.0).id()],
        0.0,
        f64::INFINITY,
    );
    let truncated = truncated.label("truncated");
    b.observe(truncated, DoubleTensor::scalar(0.7));
    let grid = b.gaussian_with_shape(&[2, 2], mu, 1.0);
    b.observe(
        grid,
        DoubleTensor::new(&[2, 2], vec![0.1, 0.2, 0.3, 0.4]).unwrap(),
    );
    let grid = grid.id();
    let latents = vec![p.id(), rate.id(), weights.id(), mu.id()];
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state);
    let model = Model::new(graph, &state).unwrap();
    let nuts = Nuts::default().with_adapt_count(50);
    let chains = sample_chains(2, 2, &state, |state, rng| {
        nuts.sample(&model, state, &latents, 30, rng)
    })
    .unwrap();
    let loglik = PointwiseLogLik::from_chains(&model, &state, &chains).unwrap();
    let grid_name = format!("#{}", grid.index());
    let expected: Vec<String> = [
        "flips[0]",
        "flips[1]",
        "flips[2]",
        "counts[0]",
        "counts[1]",
        // the mixture is unlabelled too, so named by its index below
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    assert_eq!(&loglik.names()[..5], expected.as_slice());
    assert!(loglik.names()[5].ends_with("[0]") && loglik.names()[6].ends_with("[1]"));
    assert_eq!(loglik.names()[7], "truncated");
    assert_eq!(
        &loglik.names()[8..],
        &[
            format!("{grid_name}[0, 0]"),
            format!("{grid_name}[0, 1]"),
            format!("{grid_name}[1, 0]"),
            format!("{grid_name}[1, 1]"),
        ]
    );
    assert_sums_to_log_likelihood(&model, &state, chains.chains(), &loglik, |_| 0.0);
}

#[test]
fn factors_have_no_points() {
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 1.0);
    let y = b.gaussian_with_shape(&[2], mu, 1.0).label("y");
    b.observe(y, DoubleTensor::vector(vec![0.5, 1.5]));
    let f = b.factor(mu * mu * -0.5);
    let (mu, f) = (mu.id(), f.id());
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state);
    let model = Model::new(graph, &state).unwrap();
    let nuts = Nuts::default().with_adapt_count(50);
    let chains = sample_chains(1, 3, &state, |state, rng| {
        nuts.sample(&model, state, &[mu], 30, rng)
    })
    .unwrap();
    let loglik = PointwiseLogLik::from_chains(&model, &state, &chains).unwrap();
    assert_eq!(loglik.names(), &["y[0]", "y[1]"]);
    let graph = model.graph().clone();
    assert_sums_to_log_likelihood(&model, &state, chains.chains(), &loglik, |s| {
        node_log_prob(&graph, s, f).unwrap()
    });
}

#[test]
fn samples_missing_a_latent_the_observations_need_are_rejected() {
    let (model, mut state, latents) = regression();
    // sigma isn't recorded
    let samples = MetropolisHastings::default()
        .sample(&model, &mut state, &latents[..2], 20, &mut rng(4))
        .unwrap();
    match PointwiseLogLik::from_samples(&model, &state, &samples) {
        Err(AlgorithmError::InvalidArgument(message)) => {
            assert!(message.contains("\"sigma\""), "{message}");
            assert!(!message.contains("\"a\""), "{message}");
        }
        other => panic!("expected a missing latent error, got {other:?}"),
    }
}

#[test]
fn a_model_without_observations_or_draws_is_rejected() {
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 1.0).id();
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let samples = MetropolisHastings::default()
        .sample(&model, &mut state, &[mu], 5, &mut rng(5))
        .unwrap();
    assert!(PointwiseLogLik::from_samples(&model, &state, &samples).is_err());

    let (model, state, latents) = regression();
    let empty = NetworkSamples::new(model.graph(), &latents).unwrap();
    assert!(PointwiseLogLik::from_samples(&model, &state, &empty).is_err());
}
