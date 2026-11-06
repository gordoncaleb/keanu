//! The Wiener first-passage time distribution as a graph node: its element-wise terms and the log-density op, a
//! mixture of it, sampling given the boundary, and the gradient through the graph. Its Hessian is in `hessian.rs`,
//! its support in `support.rs`, saving in keanu-io's round trip.

use keanu_distributions as d;
use keanu_distributions::KeanuRng;
use keanu_graph::op::DistributionKind;
use keanu_graph::{
    DoubleVar, GraphBuilder, Model, NodeId, Value, node_log_prob, node_log_prob_terms,
};
use keanu_tensor::IntTensor;

mod common;
use common::{evaluate, s, v};

const TIMES: [f64; 4] = [0.35, 0.6, 1.1, 3.0];
const BOUNDARIES: [i64; 4] = [0, 1, 1, 0];
const PARAMETERS: [f64; 5] = [1.2, 0.25, 0.45, 0.8, 0.6];

fn expected_terms(p: [f64; 5]) -> Vec<f64> {
    TIMES
        .iter()
        .zip(BOUNDARIES)
        .map(|(&x, boundary)| d::Wiener::log_prob_and_gradient(x, p, boundary).0)
        .collect()
}

/// Constants for the five f64 parameters, then the boundaries.
fn parameters(b: &GraphBuilder, p: [f64; 5]) -> Vec<NodeId> {
    let mut ids: Vec<NodeId> = p.iter().map(|&q| b.scalar(q).id()).collect();
    ids.push(b.int_constant(IntTensor::vector(BOUNDARIES.to_vec())).id());
    ids
}

#[test]
fn terms_and_the_log_density_op_match_the_distribution() {
    let b = GraphBuilder::new();
    let params = parameters(&b, PARAMETERS);
    let node = b
        .add_distribution(DistributionKind::Wiener, None, &params)
        .unwrap();
    b.observe_of(node, Value::F64(v(&TIMES))).unwrap();
    let density = b
        .log_density(DistributionKind::Wiener, b.constant(v(&TIMES)), &params)
        .id();
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state, 1);
    let expected = expected_terms(PARAMETERS);
    let terms = node_log_prob_terms(&graph, &state, node).unwrap();
    for (actual, expected) in terms.as_slice().iter().zip(&expected) {
        assert!((actual - expected).abs() < 1e-13 * expected.abs().max(1.0));
    }
    for (actual, expected) in state.f64(density).unwrap().as_slice().iter().zip(&expected) {
        assert!((actual - expected).abs() < 1e-13 * expected.abs().max(1.0));
    }
    let total: f64 = expected.iter().sum();
    assert!((node_log_prob(&graph, &state, node).unwrap() - total).abs() < 1e-12);
}

#[test]
fn a_mixture_of_two_drifts() {
    // two latent classes of trials with different drifts, the parameters sharing everything else
    let b = GraphBuilder::new();
    let shifted = [
        PARAMETERS[0],
        PARAMETERS[1],
        PARAMETERS[2],
        -1.5,
        PARAMETERS[4],
    ];
    let mut params: Vec<NodeId> = PARAMETERS
        .iter()
        .zip(shifted)
        .map(|(&p, q)| b.constant(v(&[p, q])).id())
        .collect();
    params.push(b.int_constant(IntTensor::vector(vec![1])).id());
    let y: DoubleVar<'_> = b.mixture(
        DistributionKind::Wiener,
        &params,
        b.constant(v(&[0.3, 0.7])),
    );
    b.observe(y, s(0.8));
    let y = y.id();
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state, 1);
    let first = d::Wiener::log_prob_and_gradient(0.8, PARAMETERS, 1).0;
    let second = d::Wiener::log_prob_and_gradient(0.8, shifted, 1).0;
    let expected = (0.3 * first.exp() + 0.7 * second.exp()).ln();
    let actual = node_log_prob(&graph, &state, y).unwrap();
    assert!((actual - expected).abs() < 1e-12, "{actual} vs {expected}");
}

#[test]
fn a_node_samples_times_after_t0_given_its_boundaries() {
    let b = GraphBuilder::new();
    let params = parameters(&b, PARAMETERS);
    let node = b
        .add_distribution(DistributionKind::Wiener, None, &params)
        .unwrap();
    let (graph, mut state) = b.build().unwrap();
    let mut rng = KeanuRng::seed_from_u64(4);
    state.lazy_eval(&graph, &[node], &mut rng).unwrap();
    let times = state.f64(node).unwrap();
    assert_eq!(times.shape(), &[4]);
    assert!(times.as_slice().iter().all(|&t| t > PARAMETERS[1]));
}

#[test]
fn the_gradient_through_the_graph_matches_the_distribution() {
    // latent a, t0, w, v and sv given observed times and boundaries
    let b = GraphBuilder::new();
    let latents: Vec<DoubleVar<'_>> = PARAMETERS
        .iter()
        .map(|&p| {
            let x = b.gaussian(0.0, 10.0);
            b.set_value(x, s(p));
            x
        })
        .collect();
    let ids: Vec<NodeId> = latents.iter().map(|x| x.id()).collect();
    let y = b.wiener(
        latents[0],
        latents[1],
        latents[2],
        latents[3],
        latents[4],
        IntTensor::vector(BOUNDARIES.to_vec()),
    );
    b.observe(y, v(&TIMES));
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let gradient = model.log_likelihood_gradient(&state).unwrap();
    for (j, id) in ids.iter().enumerate() {
        let expected: f64 = TIMES
            .iter()
            .zip(BOUNDARIES)
            .map(|(&x, boundary)| {
                d::Wiener::log_prob_and_gradient(x, PARAMETERS, boundary).1[j + 1]
            })
            .sum();
        let actual = gradient.get(*id).unwrap().as_slice()[0];
        assert!(
            (actual - expected).abs() < 1e-10 * expected.abs().max(1.0),
            "parameter {j}: {actual} vs {expected}"
        );
    }
}
