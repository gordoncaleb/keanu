//! The model layer: latent / observed sets, log probabilities, constraints, lambda sections and incremental
//! log probability updates.

use keanu_distributions::KeanuRng;
use keanu_graph::{
    GraphBuilder, Model, Value, downstream_lambda_section, joint_log_prob, log_likelihood,
    upstream_lambda_section,
};
use keanu_tensor::{BoolTensor, DoubleTensor};
use std::sync::Arc;

#[test]
fn latents_observed_and_log_probs() {
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 1.0).label("mu");
    b.set_value(mu, DoubleTensor::scalar(0.4));
    let y = b.gaussian_with_shape(&[3], mu * 2.0, 1.0).label("y");
    b.observe(y, DoubleTensor::vector(vec![0.5, 1.0, 1.5]));
    let (mu, y) = (mu.id(), y.id());
    let (graph, mut state) = b.build().unwrap();
    let all: Vec<_> = graph.ids().collect();
    state
        .lazy_eval(&graph, &all, &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    let graph = Arc::new(graph);
    let model = Model::new(graph.clone(), &state).unwrap();

    assert_eq!(model.latents(), &[mu]);
    assert_eq!(model.observed(), &[y]);
    assert_eq!(model.find("mu"), Some(mu));
    assert_eq!(model.continuous_latents(), vec![mu]);
    assert!(
        (model.log_prob(&state).unwrap() - joint_log_prob(&graph, &state).unwrap()).abs() < 1e-12
    );
    assert!(
        (model.log_likelihood(&state).unwrap() - log_likelihood(&graph, &state).unwrap()).abs()
            < 1e-12
    );

    // d/dmu [N(mu; 0, 1) + sum N(y; 2 mu, 1)] = -mu + 2 * sum(y - 2 mu)
    let expected = -0.4 + 2.0 * [0.5, 1.0, 1.5].iter().map(|y| y - 0.8).sum::<f64>();
    let gradient = model.log_prob_gradient(&state).unwrap();
    assert!((gradient.get(mu).unwrap().as_slice()[0] - expected).abs() < 1e-12);
    let likelihood_gradient = model.log_likelihood_gradient(&state).unwrap();
    assert!((likelihood_gradient.get(mu).unwrap().as_slice()[0] - (expected + 0.4)).abs() < 1e-12);
}

#[test]
fn a_model_needs_a_latent_or_observed_node() {
    let b = GraphBuilder::new();
    let _ = b.scalar(1.0) + 2.0;
    let (graph, state) = b.build().unwrap();
    assert!(Model::new(graph, &state).is_err());
}

#[test]
fn a_violated_observed_constraint_makes_the_log_prob_impossible() {
    let b = GraphBuilder::new();
    let x = b.gaussian(0.0, 1.0);
    b.set_value(x, DoubleTensor::scalar(0.5));
    let positive = x.gt(0.0);
    b.observe(positive, BoolTensor::scalar(true));
    let (x, positive) = (x.id(), positive.id());
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    assert_eq!(model.observed(), &[positive]);
    assert!(model.log_prob(&state).unwrap().is_finite());

    model
        .set_and_cascade(&mut state, &[(x, Value::F64(DoubleTensor::scalar(-0.5)))])
        .unwrap();
    assert_eq!(model.log_prob(&state).unwrap(), f64::NEG_INFINITY);
}

#[test]
fn lambda_sections_stop_at_probabilistic_and_observed_nodes() {
    let b = GraphBuilder::new();
    let a = b.gaussian(0.0, 1.0);
    let doubled = a * 2.0;
    let c = b.gaussian(doubled, 1.0);
    let d = b.gaussian(c, 1.0);
    let e = b.gaussian(doubled.exp(), 1.0);
    b.observe(e, DoubleTensor::scalar(1.0));
    let (a, doubled, c, d, e) = (a.id(), doubled.id(), c.id(), d.id(), e.id());
    let (graph, state) = b.build().unwrap();

    let down = downstream_lambda_section(&graph, &state, &[a]).unwrap();
    assert_eq!(down.latent_and_observed(), &[a, c, e]);
    assert!(down.nodes().contains(&doubled));
    assert!(!down.nodes().contains(&d));

    let up = upstream_lambda_section(&graph, &state, &[d]).unwrap();
    assert_eq!(up.latent_and_observed(), &[c, d]);
}

#[test]
fn log_prob_after_a_proposal_equals_recomputing_it() {
    let b = GraphBuilder::new();
    let a = b.gaussian(0.0, 1.0);
    let s = b.half_gaussian(2.0);
    let shared = a * 2.0 + 1.0;
    let x = b.gaussian_with_shape(&[3], shared, s);
    let y = b.gaussian_with_shape(&[3], x, 0.5);
    b.observe(y, DoubleTensor::vector(vec![0.1, 0.9, 2.0]));
    let z = b.gaussian(shared.exp(), 1.0);
    b.observe(z, DoubleTensor::scalar(3.0));
    let (a, s, x) = (a.id(), s.id(), x.id());
    let (graph, mut state) = b.build().unwrap();
    let mut rng = KeanuRng::seed_from_u64(7);
    let all: Vec<_> = graph.ids().collect();
    state.lazy_eval(&graph, &all, &mut rng).unwrap();
    let model = Model::new(graph, &state).unwrap();

    let mut log_prob = model.log_prob(&state).unwrap();
    let proposals = [
        vec![(a, Value::F64(DoubleTensor::scalar(0.3)))],
        vec![
            (s, Value::F64(DoubleTensor::scalar(1.7))),
            (x, Value::F64(DoubleTensor::vector(vec![0.2, 1.0, 1.4]))),
        ],
        vec![
            (a, Value::F64(DoubleTensor::scalar(-0.8))),
            (s, Value::F64(DoubleTensor::scalar(0.9))),
        ],
    ];
    for proposal in proposals {
        log_prob = model
            .log_prob_after(&mut state, &proposal, log_prob)
            .unwrap();
        let full = model.log_prob(&state).unwrap();
        assert!(
            (log_prob - full).abs() < 1e-10,
            "incremental {log_prob} vs full {full}"
        );
    }

    // an impossible proposal, then back to a possible one
    log_prob = model
        .log_prob_after(
            &mut state,
            &[(s, Value::F64(DoubleTensor::scalar(-1.0)))],
            log_prob,
        )
        .unwrap();
    assert_eq!(log_prob, f64::NEG_INFINITY);
    log_prob = model
        .log_prob_after(
            &mut state,
            &[(s, Value::F64(DoubleTensor::scalar(1.0)))],
            log_prob,
        )
        .unwrap();
    assert!((log_prob - model.log_prob(&state).unwrap()).abs() < 1e-10);
}
