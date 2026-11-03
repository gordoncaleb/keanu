//! Value propagation, observation, proxies, labels and builder errors.

use keanu_distributions::KeanuRng;
use keanu_graph::{DType, Graph, GraphBuilder, GraphError, State, Value};
use keanu_tensor::{BoolTensor, DoubleTensor, IntTensor};

fn rng() -> KeanuRng {
    KeanuRng::seed_from_u64(7)
}

fn scalar(state: &State, id: keanu_graph::NodeId) -> f64 {
    state.f64(id).unwrap().as_slice()[0]
}

#[test]
fn lazy_eval_samples_missing_probabilistic_nodes_and_keeps_existing_values() {
    let b = GraphBuilder::new();
    let a = b.gaussian(0.0, 1.0);
    let doubled = a * 2.0;
    let (a, doubled) = (a.id(), doubled.id());
    let (graph, mut state) = b.build().unwrap();

    state.lazy_eval(&graph, &[doubled], &mut rng()).unwrap();
    let sampled = scalar(&state, a);
    assert_eq!(scalar(&state, doubled), 2.0 * sampled);

    // a stale deterministic value is kept by lazy_eval but recomputed by eval
    state
        .set(&graph, doubled, DoubleTensor::scalar(-99.0))
        .unwrap();
    state.lazy_eval(&graph, &[doubled], &mut rng()).unwrap();
    assert_eq!(scalar(&state, doubled), -99.0);
    state.eval(&graph, &[doubled], &mut rng()).unwrap();
    assert_eq!(scalar(&state, doubled), 2.0 * sampled);

    // eval keeps the probabilistic node's existing value rather than resampling it
    assert_eq!(scalar(&state, a), sampled);
}

#[test]
fn eval_samples_a_probabilistic_node_given_its_lazily_evaluated_parents() {
    let b = GraphBuilder::new();
    let mu = b.scalar(5.0) + 1.0;
    let x = b.gaussian(mu, 1e-9);
    let (mu, x) = (mu.id(), x.id());
    let (graph, mut state) = b.build().unwrap();
    state.eval(&graph, &[x], &mut rng()).unwrap();
    assert_eq!(scalar(&state, mu), 6.0);
    assert!((scalar(&state, x) - 6.0).abs() < 1e-6);
}

#[test]
fn cascade_recomputes_deterministic_descendants_and_stops_at_probabilistic_nodes() {
    let b = GraphBuilder::new();
    let a = b.gaussian(0.0, 1.0);
    let doubled = a * 2.0;
    let c = b.gaussian(doubled, 1.0);
    let c_plus_one = c + 1.0;
    let (a, doubled, c, c_plus_one) = (a.id(), doubled.id(), c.id(), c_plus_one.id());
    let (graph, mut state) = b.build().unwrap();
    state.lazy_eval(&graph, &[c_plus_one], &mut rng()).unwrap();
    let (c_before, c_plus_one_before) = (scalar(&state, c), scalar(&state, c_plus_one));

    state
        .set_and_cascade(&graph, a, DoubleTensor::scalar(3.0))
        .unwrap();
    assert_eq!(scalar(&state, doubled), 6.0);
    assert_eq!(
        scalar(&state, c),
        c_before,
        "the cascade must stop at the probabilistic child"
    );
    assert_eq!(scalar(&state, c_plus_one), c_plus_one_before);

    state
        .set_and_cascade(&graph, c, DoubleTensor::scalar(10.0))
        .unwrap();
    assert_eq!(scalar(&state, c_plus_one), 11.0);
}

#[test]
fn observed_nodes_keep_their_values() {
    let b = GraphBuilder::new();
    let a = b.gaussian(0.0, 1.0);
    let a_id = a.id();
    b.observe(a, DoubleTensor::scalar(0.5));
    let (graph, mut state) = b.build().unwrap();
    assert!(state.is_observed(a_id));

    state.set(&graph, a_id, DoubleTensor::scalar(2.0)).unwrap();
    assert_eq!(
        scalar(&state, a_id),
        0.5,
        "set is ignored on observed nodes"
    );
    state.clear(a_id);
    state.eval(&graph, &[a_id], &mut rng()).unwrap();
    assert_eq!(
        scalar(&state, a_id),
        0.5,
        "observed nodes are never resampled"
    );

    state.unobserve(a_id);
    state.set(&graph, a_id, DoubleTensor::scalar(2.0)).unwrap();
    assert_eq!(scalar(&state, a_id), 2.0);
}

#[test]
fn only_probabilistic_and_boolean_nodes_can_be_observed() {
    let b = GraphBuilder::new();
    let x = b.gaussian(0.0, 1.0);
    let doubled = x * 2.0;
    let positive = x.gt(0.0);
    let (x, doubled, positive) = (x.id(), doubled.id(), positive.id());
    let (graph, mut state) = b.build().unwrap();

    assert!(matches!(
        state.observe(&graph, doubled, DoubleTensor::scalar(1.0)),
        Err(GraphError::NotObservable(_))
    ));
    state
        .observe(&graph, positive, BoolTensor::scalar(true))
        .unwrap();
    assert!(matches!(
        state.observe(&graph, x, DoubleTensor::vector(vec![1.0, 2.0])),
        Err(GraphError::InvalidValue(_))
    ));
    assert!(matches!(
        state.observe(&graph, x, IntTensor::scalar(1)),
        Err(GraphError::InvalidValue(_))
    ));

    // the builder applies the same rule
    let b = GraphBuilder::new();
    let y = b.scalar(1.0) * 2.0;
    b.observe(y, DoubleTensor::scalar(2.0));
    assert!(matches!(b.build(), Err(GraphError::NotObservable(_))));
}

#[test]
fn state_clones_are_independent() {
    let b = GraphBuilder::new();
    let a = b.gaussian(0.0, 1.0).id();
    let (graph, mut first) = b.build().unwrap();
    first.set(&graph, a, DoubleTensor::scalar(1.0)).unwrap();
    let mut second = first.clone();
    second.set(&graph, a, DoubleTensor::scalar(2.0)).unwrap();
    assert_eq!(scalar(&first, a), 1.0);
    assert_eq!(scalar(&second, a), 2.0);
}

#[test]
fn proxies_connect_only_to_earlier_nodes_of_the_same_type_and_shape() {
    // connected to an earlier node: evaluates to its value
    let b = GraphBuilder::new();
    let source = b.scalar(4.0);
    let proxy = b.double_proxy(&[]);
    b.connect(proxy, source);
    let out = (proxy * 2.0).id();
    let (graph, mut state) = b.build().unwrap();
    state.lazy_eval(&graph, &[out], &mut rng()).unwrap();
    assert_eq!(scalar(&state, out), 8.0);

    // a later node would break topological order
    let b = GraphBuilder::new();
    let proxy = b.double_proxy(&[]);
    let later = b.scalar(1.0);
    b.connect(proxy, later);
    assert!(matches!(b.build(), Err(GraphError::Proxy(_))));

    // shape mismatch
    let b = GraphBuilder::new();
    let source = b.constant(DoubleTensor::vector(vec![1.0, 2.0]));
    let proxy = b.double_proxy(&[]);
    b.connect(proxy, source);
    assert!(matches!(b.build(), Err(GraphError::Proxy(_))));

    // connected twice
    let b = GraphBuilder::new();
    let (s1, s2) = (b.scalar(1.0), b.scalar(2.0));
    let proxy = b.double_proxy(&[]);
    b.connect(proxy, s1);
    b.connect(proxy, s2);
    assert!(matches!(b.build(), Err(GraphError::Proxy(_))));

    // never connected: building works but evaluating fails
    let b = GraphBuilder::new();
    let proxy = b.double_proxy(&[]);
    let out = (proxy + 1.0).id();
    let (graph, mut state) = b.build().unwrap();
    assert!(matches!(
        state.lazy_eval(&graph, &[out], &mut rng()),
        Err(GraphError::Proxy(_))
    ));
}

#[test]
fn labels_are_unique_and_can_be_found() {
    let b = GraphBuilder::new();
    let a = b.gaussian(0.0, 1.0).label("a").id();
    b.scalar(1.0).label("one");
    let (graph, _) = b.build().unwrap();
    assert_eq!(graph.find("a"), Some(a));
    assert_eq!(graph.node(a).label(), Some("a"));
    assert_eq!(graph.find("missing"), None);

    let b = GraphBuilder::new();
    b.scalar(1.0).label("x");
    b.scalar(2.0).label("x");
    assert_eq!(
        b.build().unwrap_err(),
        GraphError::DuplicateLabel("x".into())
    );
}

#[test]
fn construction_errors_are_deferred_to_build() {
    let b = GraphBuilder::new();
    let a = b.constant(DoubleTensor::vector(vec![1.0, 2.0, 3.0]));
    let c = b.constant(DoubleTensor::vector(vec![1.0, 2.0]));
    let bad = a + c;
    // building can carry on with the placeholder
    let _more = (bad * 2.0).exp().sum();
    assert!(matches!(b.error(), Some(GraphError::Tensor(_))));
    assert!(matches!(b.build(), Err(GraphError::Tensor(_))));

    let b = GraphBuilder::new();
    let m = b.constant(DoubleTensor::new(&[2, 3], vec![0.0; 6]).unwrap());
    m.cholesky();
    assert!(matches!(
        b.build(),
        Err(GraphError::InvalidNode { op: "cholesky", .. })
    ));
}

#[test]
fn assertions_fail_evaluation_when_false() {
    let b = GraphBuilder::new();
    let x = b.gaussian(0.0, 1.0);
    let checked = x.gt(0.0).assert(Some("x must be positive"));
    let (x, checked) = (x.id(), checked.id());
    let (graph, mut state) = b.build().unwrap();
    state.set(&graph, x, DoubleTensor::scalar(1.0)).unwrap();
    state.eval(&graph, &[checked], &mut rng()).unwrap();
    state.set(&graph, x, DoubleTensor::scalar(-1.0)).unwrap();
    assert_eq!(
        state.eval(&graph, &[checked], &mut rng()),
        Err(GraphError::AssertionFailed("x must be positive".into()))
    );
}

#[test]
fn lambdas_compute_from_parent_values_and_check_the_declared_type() {
    let b = GraphBuilder::new();
    let x = b.constant(DoubleTensor::vector(vec![1.0, 2.0]));
    let squared = b.lambda("square", &[x.id()], DType::F64, &[2], |v| {
        Ok(Value::F64(v[0].as_f64().unwrap().map(|x| x * x)))
    });
    let wrong = b.lambda("wrong", &[x.id()], DType::F64, &[3], |v| Ok(v[0].clone()));
    let (graph, mut state) = b.build().unwrap();
    state.eval(&graph, &[squared], &mut rng()).unwrap();
    assert_eq!(state.f64(squared).unwrap().as_slice(), &[1.0, 4.0]);
    assert!(matches!(
        state.eval(&graph, &[wrong], &mut rng()),
        Err(GraphError::Lambda(_))
    ));
}

#[test]
fn multiplexer_and_cpt_select_by_their_conditions() {
    let b = GraphBuilder::new();
    let selector = b.int_scalar(1);
    let options = [b.scalar(10.0), b.scalar(20.0), b.scalar(30.0)];
    let muxed = b.multiplexer(selector, &options).id();

    let rain = b.bool_scalar(true);
    let season = b.int_scalar(2);
    let cases = [(vec![1, 2], b.scalar(0.9)), (vec![0, 2], b.scalar(0.1))];
    let table = b.cpt(&[rain.id(), season.id()], &cases, b.scalar(0.5)).id();
    let (rain, season) = (rain.id(), season.id());
    let (graph, mut state) = b.build().unwrap();
    state.eval(&graph, &[muxed, table], &mut rng()).unwrap();
    assert_eq!(scalar(&state, muxed), 20.0);
    assert_eq!(scalar(&state, table), 0.9);

    state
        .set_and_cascade(&graph, rain, BoolTensor::scalar(false))
        .unwrap();
    assert_eq!(scalar(&state, table), 0.1);
    state
        .set_and_cascade(&graph, season, IntTensor::scalar(3))
        .unwrap();
    assert_eq!(scalar(&state, table), 0.5);
}

#[test]
fn every_distribution_can_be_sampled_through_eval() {
    let b = GraphBuilder::new();
    let ids = vec![
        b.gaussian_with_shape(&[2], 0.0, 1.0).id(),
        b.half_gaussian(1.0).id(),
        b.cauchy(0.0, 1.0).id(),
        b.half_cauchy(1.0).id(),
        b.beta(2.0, 3.0).id(),
        b.chi_squared(3).id(),
        b.dirichlet(DoubleTensor::vector(vec![1.0, 2.0, 3.0])).id(),
        b.exponential(2.0).id(),
        b.gamma(1.0, 2.0).id(),
        b.inverse_gamma(2.0, 1.0).id(),
        b.laplace(0.0, 1.0).id(),
        b.logistic(0.0, 1.0).id(),
        b.log_normal(0.0, 1.0).id(),
        b.multivariate_gaussian(DoubleTensor::vector(vec![0.0, 1.0]), DoubleTensor::eye(2))
            .id(),
        b.pareto(1.0, 2.0).id(),
        b.student_t(3).id(),
        b.triangular(0.0, 1.0, 0.5).id(),
        b.uniform_with_shape(&[3], 0.0, 1.0).id(),
        b.kde(
            keanu_distributions::Kde::new(vec![1.0, 2.0, 3.0], 0.5).unwrap(),
            &[2],
        )
        .id(),
        b.bernoulli(0.5).id(),
        b.binomial(0.5, 10).id(),
        b.categorical(&[], &[b.scalar(0.2), b.scalar(0.8)]).id(),
        b.geometric(0.5).id(),
        b.multinomial(10, DoubleTensor::vector(vec![0.2, 0.3, 0.5]))
            .id(),
        b.poisson(3.0).id(),
        b.uniform_int(-3, 4).id(),
    ];
    let (graph, mut state): (Graph, State) = b.build().unwrap();
    state.eval(&graph, &ids, &mut rng()).unwrap();
    for id in ids {
        let node = graph.node(id);
        let value = state
            .value(id)
            .unwrap_or_else(|| panic!("{id:?} has no value"));
        assert_eq!(value.shape(), node.shape(), "{id:?}");
        assert_eq!(value.dtype(), node.dtype(), "{id:?}");
    }
}

#[test]
fn graphs_can_be_shared_across_threads() {
    let b = GraphBuilder::new();
    let x = b.gaussian(0.0, 1.0);
    let y = (x * 3.0).id();
    let x = x.id();
    let (graph, state) = b.build().unwrap();
    let graph = std::sync::Arc::new(graph);
    let handles: Vec<_> = (0..4)
        .map(|i| {
            let (graph, mut state) = (graph.clone(), state.clone());
            std::thread::spawn(move || {
                state
                    .set(&graph, x, DoubleTensor::scalar(i as f64))
                    .unwrap();
                state
                    .lazy_eval(&graph, &[y], &mut KeanuRng::seed_from_u64(i))
                    .unwrap();
                state.f64(y).unwrap().as_slice()[0]
            })
        })
        .collect();
    let results: Vec<f64> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(results, vec![0.0, 3.0, 6.0, 9.0]);
}
