//! The do-operator: interventions on latent, observed and deterministic nodes against hand-written models, the
//! dropped ancestors and id mapping, and the rejected arguments.

use keanu_graph::{GraphBuilder, GraphError, Model, NodeId, Value, intervene};
use keanu_tensor::{BoolTensor, DoubleTensor, IntTensor};

/// ln N(x | μ, σ).
fn ln_normal(x: f64, mu: f64, sigma: f64) -> f64 {
    let u = (x - mu) / sigma;
    -0.5 * u * u - sigma.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
}

fn scalar(x: f64) -> Value {
    DoubleTensor::scalar(x).into()
}

#[test]
fn intervening_on_a_latent_removes_its_density_and_keeps_shared_causes() {
    // z → x, z → y, x → y: do(x = 1.5) keeps z (it also causes y) and makes x a constant
    let b = GraphBuilder::new();
    let z = b.gaussian(0.0, 1.0).label("z");
    let x = b.gaussian(z * 2.0, 1.0).label("x");
    let y = b.gaussian(x + z, 0.5).label("y");
    b.observe(y, DoubleTensor::scalar(2.0));
    b.set_value(z, DoubleTensor::scalar(0.3));
    b.set_value(x, DoubleTensor::scalar(-1.0));
    let (z, x, y) = (z.id(), x.id(), y.id());
    let (graph, state) = b.build().unwrap();

    let (new, state, intervened) = intervene(&graph, &state, &[(x, scalar(1.5))]).unwrap();
    let nx = intervened.node(x).unwrap();
    assert!(!new[nx].is_probabilistic());
    assert_eq!(state.f64(nx).unwrap().to_scalar(), Some(1.5));
    assert_eq!(intervened.intervened(), &[x]);
    assert!(intervened.is_intervened(x) && !intervened.is_intervened(z));
    assert_eq!(intervened.carried(x), None);
    assert_eq!(intervened.carried(z), intervened.node(z));
    // z * 2 and x's scale fed only x, so they are dropped (with the constant 2); z is kept
    let dropped = intervened.dropped();
    assert_eq!(dropped.len(), 3);
    assert!(dropped.iter().all(|&id| id < x && id != z));
    assert!(!intervened.is_dropped(z) && !intervened.is_dropped(y));
    for (label, id) in [("z", z), ("x", x), ("y", y)] {
        assert_eq!(new.find(label), intervened.node(id), "{label}");
    }

    let model = Model::new(new, &state).unwrap();
    let nz = intervened.node(z).unwrap();
    assert_eq!(model.latents(), &[nz]);
    let expected = ln_normal(0.3, 0.0, 1.0) + ln_normal(2.0, 1.5 + 0.3, 0.5);
    assert!((model.log_prob(&state).unwrap() - expected).abs() < 1e-12);
}

#[test]
fn intervening_on_an_observed_node_replaces_the_observation() {
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 10.0).label("mu");
    let x = b.gaussian_with_shape(&[2], mu, 1.0).label("x");
    let y = b.gaussian_with_shape(&[2], x * 3.0, 1.0).label("y");
    b.observe(x, DoubleTensor::vector(vec![1.0, 2.0]));
    b.observe(y, DoubleTensor::vector(vec![3.5, 5.0]));
    b.set_value(mu, DoubleTensor::scalar(0.5));
    let (mu, x, y) = (mu.id(), x.id(), y.id());
    let (graph, state) = b.build().unwrap();

    let value: Value = DoubleTensor::vector(vec![-1.0, 4.0]).into();
    let (new, state, intervened) = intervene(&graph, &state, &[(x, value)]).unwrap();
    let nx = intervened.node(x).unwrap();
    assert!(!state.is_observed(nx));
    assert_eq!(state.f64(nx).unwrap().as_slice(), &[-1.0, 4.0]);
    // mu fed only x, but it is a latent with no other child: dropped, as is nothing observed
    assert!(intervened.is_dropped(mu));
    assert!(state.is_observed(intervened.node(y).unwrap()));
    // the product x * 3 downstream of x is recomputed
    let model = Model::new(new, &state).unwrap();
    let expected = ln_normal(3.5, -3.0, 1.0) + ln_normal(5.0, 12.0, 1.0);
    assert!((model.log_prob(&state).unwrap() - expected).abs() < 1e-12);
}

#[test]
fn intervening_on_a_deterministic_node_cuts_it_from_its_parents() {
    let b = GraphBuilder::new();
    let a = b.gaussian(0.0, 1.0).label("a");
    let c = b.gaussian(0.0, 1.0).label("c");
    let d = (a * c).label("d");
    let e = (d + 1.0).label("e");
    let y = b.gaussian(e, 1.0).label("y");
    let w = b.gaussian(c, 1.0).label("w");
    b.observe(y, DoubleTensor::scalar(0.0));
    b.observe(w, DoubleTensor::scalar(1.0));
    for v in [a, c] {
        b.set_value(v, DoubleTensor::scalar(2.0));
    }
    let (a, c, d, e) = (a.id(), c.id(), d.id(), e.id());
    let (graph, mut state) = b.build().unwrap();
    state
        .lazy_eval(
            &graph,
            &[e],
            &mut keanu_distributions::KeanuRng::seed_from_u64(0),
        )
        .unwrap();

    let (new, state, intervened) = intervene(&graph, &state, &[(d, scalar(-0.5))]).unwrap();
    // a fed only d: dropped; c also feeds w: kept
    assert!(intervened.is_dropped(a));
    assert!(!intervened.is_dropped(c));
    let nd = intervened.node(d).unwrap();
    assert!(new[nd].parents().is_empty());
    assert_eq!(new.find("d"), Some(nd));
    // e's stale value (from d = 4) is recomputed from the intervention
    assert_eq!(
        state.f64(intervened.node(e).unwrap()).unwrap().to_scalar(),
        Some(0.5)
    );
    let model = Model::new(new, &state).unwrap();
    let expected = ln_normal(2.0, 0.0, 1.0) + ln_normal(0.0, 0.5, 1.0) + ln_normal(1.0, 2.0, 1.0);
    assert!((model.log_prob(&state).unwrap() - expected).abs() < 1e-12);
}

#[test]
fn latents_downstream_of_an_intervention_keep_their_dependence() {
    let b = GraphBuilder::new();
    let x = b.gaussian(0.0, 1.0).label("x");
    let w = b.gaussian(x, 1.0).label("w");
    let y = b.gaussian(w, 1.0).label("y");
    b.observe(y, DoubleTensor::scalar(1.0));
    b.set_value(x, DoubleTensor::scalar(0.0));
    b.set_value(w, DoubleTensor::scalar(0.7));
    let (x, w) = (x.id(), w.id());
    let (graph, state) = b.build().unwrap();

    let (new, mut state, intervened) = intervene(&graph, &state, &[(x, scalar(2.0))]).unwrap();
    let nw = intervened.node(w).unwrap();
    let model = Model::new(new, &state).unwrap();
    assert_eq!(model.latents(), &[nw]);
    assert_eq!(model.graph()[nw].parents()[0], intervened.node(x).unwrap());
    for value in [0.7, 1.9] {
        state
            .set(model.graph(), nw, DoubleTensor::scalar(value))
            .unwrap();
        let expected = ln_normal(value, 2.0, 1.0) + ln_normal(1.0, value, 1.0);
        assert!((model.log_prob(&state).unwrap() - expected).abs() < 1e-12);
    }
}

#[test]
fn several_interventions_and_discrete_values() {
    let b = GraphBuilder::new();
    let p = b.beta(2.0, 2.0).label("p");
    let coin = b.bernoulli(p).label("coin");
    let k = b.poisson(3.0).label("k");
    let y = b
        .gaussian(b.where_(coin, k.to_double(), b.scalar(-1.0)), 1.0)
        .label("y");
    b.observe(y, DoubleTensor::scalar(2.0));
    let (p, coin, k) = (p.id(), coin.id(), k.id());
    let (graph, state) = b.build().unwrap();
    let (new, state, intervened) = intervene(
        &graph,
        &state,
        &[
            (coin, BoolTensor::scalar(true).into()),
            (k, IntTensor::scalar(4).into()),
        ],
    )
    .unwrap();
    assert!(intervened.is_dropped(p));
    let model = Model::new(new, &state).unwrap();
    assert!(model.latents().is_empty());
    assert!((model.log_prob(&state).unwrap() - ln_normal(2.0, 4.0, 1.0)).abs() < 1e-12);
}

#[test]
fn rejects_unknown_nodes_wrong_values_and_bad_lists() {
    let b = GraphBuilder::new();
    let x = b.gaussian_with_shape(&[2], 0.0, 1.0).label("x");
    let x = x.id();
    let (graph, state) = b.build().unwrap();
    let vector: Value = DoubleTensor::vector(vec![1.0, 2.0]).into();

    let unknown = NodeId::from_index(7);
    assert_eq!(
        intervene(&graph, &state, &[(unknown, vector.clone())]).unwrap_err(),
        GraphError::UnknownNode(unknown)
    );
    for wrong in [scalar(1.0), IntTensor::vector(vec![1, 2]).into()] {
        let error = intervene(&graph, &state, &[(x, wrong)]).unwrap_err();
        assert!(
            matches!(&error, GraphError::InvalidValue(m) if m.contains("\"x\"")),
            "{error}"
        );
    }
    assert!(matches!(
        intervene(&graph, &state, &[]).unwrap_err(),
        GraphError::InvalidValue(_)
    ));
    assert!(matches!(
        intervene(&graph, &state, &[(x, vector.clone()), (x, vector)]).unwrap_err(),
        GraphError::InvalidValue(m) if m.contains("more than once")
    ));
}
