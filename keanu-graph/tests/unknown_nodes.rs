//! Node ids that aren't nodes of the graph, builder or state they are used with (out of range, from another graph,
//! or made with `NodeId::from_index`) are errors or `None`, never panics.

use keanu_distributions::KeanuRng;
use keanu_graph::{
    GraphBuilder, GraphError, Model, NodeId, Value, downstream_lambda_section, forward_partials,
    is_differentiable, log_prob, log_prob_gradient, node_log_prob, node_log_prob_at,
    node_log_prob_gradient, reverse_partials, upstream_lambda_section,
};
use keanu_tensor::DoubleTensor;

fn unknown() -> NodeId {
    NodeId::from_index(1000)
}

fn scalar(v: f64) -> Value {
    Value::F64(DoubleTensor::scalar(v))
}

#[test]
fn state_lookups_of_unknown_nodes_are_none() {
    let b = GraphBuilder::new();
    let _ = b.gaussian(0.0, 1.0);
    let (graph, state) = b.build().unwrap();
    assert!(!graph.contains(unknown()));
    assert!(graph.node(unknown()).is_none());
    assert!(graph.parents(unknown()).is_none());
    assert!(graph.children(unknown()).is_none());
    assert!(state.value(unknown()).is_none());
    assert!(state.f64(unknown()).is_none());
    assert!(!state.has_value(unknown()));
    assert!(!state.is_observed(unknown()));
}

#[test]
fn state_updates_of_unknown_nodes_are_errors() {
    let b = GraphBuilder::new();
    let x = b.gaussian(0.0, 1.0).id();
    let (graph, mut state) = b.build().unwrap();
    let mut rng = KeanuRng::seed_from_u64(1);
    let is_unknown = |r: Result<(), GraphError>| matches!(r, Err(GraphError::UnknownNode(id)) if id == unknown());

    assert!(is_unknown(state.set(&graph, unknown(), scalar(1.0))));
    assert!(is_unknown(state.set_and_cascade(
        &graph,
        unknown(),
        scalar(1.0)
    )));
    assert!(is_unknown(state.observe(&graph, unknown(), scalar(1.0))));
    assert!(is_unknown(state.unobserve(unknown())));
    assert!(is_unknown(state.clear(unknown())));
    assert!(is_unknown(state.cascade(&graph, &[unknown()])));
    assert!(is_unknown(state.eval(&graph, &[x, unknown()], &mut rng)));
    assert!(is_unknown(state.lazy_eval(&graph, &[unknown()], &mut rng)));
    assert!(matches!(
        state.sample_prior(&graph, unknown(), &mut rng),
        Err(GraphError::UnknownNode(_))
    ));
}

#[test]
fn a_state_of_another_graph_is_an_error() {
    let small = GraphBuilder::new();
    let _ = small.gaussian(0.0, 1.0);
    let (_, mut small_state) = small.build().unwrap();

    let big = GraphBuilder::new();
    let a = big.gaussian(0.0, 1.0);
    let c = (a * 2.0).exp();
    let (a, c) = (a.id(), c.id());
    let (graph, _) = big.build().unwrap();

    let mut rng = KeanuRng::seed_from_u64(1);
    assert!(small_state.eval(&graph, &[c], &mut rng).is_err());
    assert!(small_state.set_and_cascade(&graph, a, scalar(1.0)).is_err());
}

#[test]
fn model_and_section_lookups_of_unknown_nodes_are_errors() {
    let b = GraphBuilder::new();
    let x = b.gaussian(0.0, 1.0);
    let y = b.gaussian(x, 1.0);
    b.observe(y, DoubleTensor::scalar(0.5));
    let (x, y) = (x.id(), y.id());
    let (graph, mut state) = b.build().unwrap();
    state
        .lazy_eval(&graph, &[y], &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    let model = Model::new(graph.clone(), &state).unwrap();

    assert!(matches!(
        model.affected_by(&[x, unknown()]),
        Err(GraphError::UnknownNode(_))
    ));
    assert!(model.affected_by(&[x]).is_ok());
    assert!(matches!(
        model.set_and_cascade(&mut state, [(unknown(), scalar(1.0))]),
        Err(GraphError::UnknownNode(_))
    ));
    assert!(matches!(
        model.log_prob_after(&mut state, &[(unknown(), scalar(1.0))], 0.0),
        Err(GraphError::UnknownNode(_))
    ));
    assert!(downstream_lambda_section(&graph, &state, &[unknown()]).is_err());
    assert!(upstream_lambda_section(&graph, &state, &[unknown()]).is_err());
}

#[test]
fn autodiff_of_unknown_nodes_is_an_error() {
    let b = GraphBuilder::new();
    let x = b.gaussian(0.0, 1.0);
    let y = x.exp();
    let (x, y) = (x.id(), y.id());
    let (graph, mut state) = b.build().unwrap();
    state
        .lazy_eval(&graph, &[y], &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    assert!(!is_differentiable(&graph, unknown()));
    assert!(matches!(
        forward_partials(&graph, &state, unknown(), &[y]),
        Err(GraphError::UnknownNode(_))
    ));
    assert!(matches!(
        forward_partials(&graph, &state, x, &[unknown()]),
        Err(GraphError::UnknownNode(_))
    ));
    assert!(matches!(
        reverse_partials(&graph, &state, unknown(), &[x]),
        Err(GraphError::UnknownNode(_))
    ));
    assert!(matches!(
        reverse_partials(&graph, &state, y, &[x, unknown()]),
        Err(GraphError::UnknownNode(_))
    ));
}

#[test]
fn builder_lookups_of_unknown_nodes_are_none_or_errors() {
    let b = GraphBuilder::new();
    let x = b.scalar(1.0);
    assert_eq!(b.shape(x.id()), Some(vec![]));
    assert_eq!(b.shape(unknown()), None);
    assert_eq!(b.dtype(unknown()), None);
    assert_eq!(b.label_of(unknown()), None);
    assert_eq!(b.constant_value(unknown()), None);
    assert!(matches!(
        b.set_label_of(unknown(), "nope"),
        Err(GraphError::UnknownNode(_))
    ));
    assert!(b.build().is_ok());

    // a handle for an unknown node fails the build rather than panicking
    let b = GraphBuilder::new();
    let _ = b.double_var(unknown()) * 2.0;
    assert!(matches!(b.build(), Err(GraphError::UnknownNode(_))));
}

#[test]
fn log_probs_of_unknown_nodes_are_errors() {
    let b = GraphBuilder::new();
    let x = b.gaussian(0.0, 1.0);
    let y = b.gaussian(x, 1.0);
    b.observe(y, DoubleTensor::scalar(0.5));
    let (x, y) = (x.id(), y.id());
    let (graph, mut state) = b.build().unwrap();
    state
        .lazy_eval(&graph, &[x, y], &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    fn is_unknown<T>(r: Result<T, GraphError>) -> bool {
        matches!(r, Err(GraphError::UnknownNode(id)) if id == unknown())
    }

    assert!(is_unknown(node_log_prob(&graph, &state, unknown())));
    assert!(is_unknown(node_log_prob_at(
        &graph,
        &state,
        unknown(),
        &scalar(0.0)
    )));
    assert!(is_unknown(node_log_prob_gradient(
        &graph,
        &state,
        unknown()
    )));
    assert!(is_unknown(log_prob(&graph, &state, &[y, unknown()])));
    assert!(is_unknown(log_prob_gradient(
        &graph,
        &state,
        &[y, unknown()],
        &[x]
    )));
    assert!(is_unknown(log_prob_gradient(
        &graph,
        &state,
        &[y],
        &[x, unknown()]
    )));
    assert!(log_prob_gradient(&graph, &state, &[y], &[x]).is_ok());

    let model = Model::new(graph, &state).unwrap();
    let gradient = model.log_prob_gradient(&state).unwrap();
    assert!(gradient.get(unknown()).is_none());
    assert!(gradient.clone().take(unknown()).is_none());
}

#[test]
#[should_panic(expected = "node #1000 is not a node of this graph")]
fn indexing_a_graph_by_an_unknown_node_panics_like_a_slice() {
    let b = GraphBuilder::new();
    let _ = b.gaussian(0.0, 1.0);
    let (graph, _) = b.build().unwrap();
    let _ = &graph[unknown()];
}
