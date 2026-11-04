//! `Model::log_prob_and_gradient` is exactly `Model::log_prob` and `Model::log_prob_gradient` in one pass.

use keanu_distributions::KeanuRng;
use keanu_graph::{Graph, GraphBuilder, Model, NodeId, State};
use keanu_tensor::{DoubleTensor, IntTensor};
use std::sync::Arc;

/// A model with a node of most distributions, latents sampled from their priors, and the id of a uniform latent.
fn model(seed: u64) -> (Arc<Graph>, State, Model, NodeId) {
    let b = GraphBuilder::new();
    let a = b.gaussian_with_shape(&[3], 0.0, 1.0);
    let s = b.gamma(1.0, 2.0);
    let h = b.half_gaussian(s);
    let e = b.exponential(1.5);
    let bt = b.beta(2.0, 3.0);
    let l = b.laplace(a, s);
    let lg = b.logistic(0.0, s);
    let ln = b.log_normal(0.0, 0.5);
    let c = b.cauchy(a, h);
    let tri = b.triangular(0.0, 2.0, 1.0);
    let u = b.uniform(-1.0, 1.0);
    let inv = b.inverse_gamma(2.0, s);
    let hc = b.half_cauchy(e);
    let pareto = b.pareto(1.0, e + 1.0);
    let chi = b.chi_squared(3);
    let dir = b.dirichlet(DoubleTensor::vector(vec![1.0, 2.0, 3.0]));
    let _discrete = b.uniform_int(0, 5);
    let p = b.poisson(s);
    b.observe(p, IntTensor::scalar(3));
    let bin = b.binomial(bt, 10);
    b.observe(bin, IntTensor::scalar(4));
    let y = b.gaussian(
        a * s + c + l + lg + ln + tri + u + inv + hc + pareto + chi,
        e + dir,
    );
    b.observe(y, DoubleTensor::vector(vec![0.5, 1.0, 1.5]));
    let u = u.id();
    let (graph, mut state) = b.build().unwrap();
    let all: Vec<_> = graph.ids().collect();
    state
        .lazy_eval(&graph, &all, &mut KeanuRng::seed_from_u64(seed))
        .unwrap();
    let graph = Arc::new(graph);
    let model = Model::new(graph.clone(), &state).unwrap();
    (graph, state, model, u)
}

#[test]
fn log_prob_and_gradient_equal_the_separate_calls_exactly() {
    for seed in 0..20 {
        let (_, state, model, _) = model(seed);
        let log_prob = model.log_prob(&state).unwrap();
        let gradient = model.log_prob_gradient(&state).unwrap();
        let (combined_log_prob, combined_gradient) = model.log_prob_and_gradient(&state).unwrap();
        assert!(log_prob.is_finite(), "seed {seed}: {log_prob}");
        assert_eq!(
            combined_log_prob.to_bits(),
            log_prob.to_bits(),
            "seed {seed}"
        );
        let combined_gradient = combined_gradient.expect("a possible state has a gradient");
        assert_eq!(combined_gradient.len(), gradient.len());
        assert!(!gradient.is_empty());
        for (id, expected) in gradient.iter() {
            let actual = combined_gradient.get(id).unwrap();
            assert_eq!(actual.shape(), expected.shape());
            for (a, e) in actual.as_slice().iter().zip(expected.as_slice()) {
                assert_eq!(a.to_bits(), e.to_bits(), "seed {seed}, node {id:?}");
            }
        }
    }
}

#[test]
fn an_impossible_state_has_no_gradient() {
    let (graph, mut state, model, u) = model(1);
    state.set(&graph, u, DoubleTensor::scalar(5.0)).unwrap();
    assert_eq!(model.log_prob(&state).unwrap(), f64::NEG_INFINITY);
    let (log_prob, gradient) = model.log_prob_and_gradient(&state).unwrap();
    assert_eq!(log_prob, f64::NEG_INFINITY);
    assert!(gradient.is_none());
}

#[test]
fn a_contradicted_observation_is_impossible() {
    let b = GraphBuilder::new();
    let x = b.gaussian(0.0, 1.0);
    b.set_value(x, DoubleTensor::scalar(0.5));
    let positive = x.gt(0.0);
    b.observe(positive, keanu_tensor::BoolTensor::scalar(false));
    let (graph, mut state) = b.build().unwrap();
    let all: Vec<_> = graph.ids().collect();
    state
        .lazy_eval(&graph, &all, &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    let model = Model::new(graph, &state).unwrap();
    assert_eq!(model.log_prob(&state).unwrap(), f64::NEG_INFINITY);
    let (log_prob, gradient) = model.log_prob_and_gradient(&state).unwrap();
    assert_eq!(log_prob, f64::NEG_INFINITY);
    assert!(gradient.is_none());
}
