//! Masked observations: the rewritten model's log probability against the fully observed and the latent forms,
//! covariates read through the recombined node, and the rejected nodes and arguments.

use keanu_distributions::KeanuRng;
use keanu_graph::{
    GraphBuilder, GraphError, MaskedObservation, Model, NodeId, Parameterisation, observe_masked,
};
use keanu_tensor::{BoolTensor, DoubleTensor, IntTensor};

/// mu ~ N(0, 3), x ~ N(mu, 1) of shape [2, 3], y ~ N(2x, 0.5) of the same shape.
fn regression() -> (
    keanu_graph::Graph,
    keanu_graph::State,
    NodeId,
    NodeId,
    NodeId,
) {
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 3.0).label("mu");
    let x = b.gaussian_with_shape(&[2, 3], mu, 1.0).label("x");
    let y = b.gaussian_with_shape(&[2, 3], x * 2.0, 0.5).label("y");
    b.set_value(mu, DoubleTensor::scalar(0.4));
    let (mu, x, y) = (mu.id(), x.id(), y.id());
    let (graph, state) = b.build().unwrap();
    (graph, state, mu, x, y)
}

const X: [f64; 6] = [0.1, -0.3, 1.2, 0.8, -1.0, 0.5];
const Y: [f64; 6] = [0.4, -0.2, 2.1, 1.9, -2.2, 0.7];

fn mask(missing: &[usize]) -> BoolTensor {
    BoolTensor::new(&[2, 3], (0..6).map(|i| missing.contains(&i)).collect()).unwrap()
}

fn values(v: [f64; 6]) -> DoubleTensor {
    DoubleTensor::new(&[2, 3], v.to_vec()).unwrap()
}

/// The log probability of the original model with x and y fully observed at the given values.
fn complete_log_prob(x: [f64; 6], y: [f64; 6]) -> f64 {
    let (graph, mut state, _, xi, yi) = regression();
    state.observe(&graph, xi, values(x)).unwrap();
    state.observe(&graph, yi, values(y)).unwrap();
    let all: Vec<NodeId> = graph.ids().collect();
    state
        .lazy_eval(&graph, &all, &mut KeanuRng::seed_from_u64(0))
        .unwrap();
    Model::new(graph, &state).unwrap().log_prob(&state).unwrap()
}

#[test]
fn missing_outcomes_and_covariates_keep_the_joint_density() {
    let (graph, state, mu, x, y) = regression();
    let (x_missing, y_missing) = ([1usize, 5], [0usize, 4, 5]);
    let (new, mut state, imputed) = observe_masked(
        &graph,
        &state,
        &[
            MaskedObservation::new(y, values(Y), mask(&y_missing)),
            MaskedObservation::new(x, values(X), mask(&x_missing)),
        ],
    )
    .unwrap();
    assert_eq!(imputed.nodes()[0].original(), y);
    let (xr, yr) = (imputed.get(x).unwrap(), imputed.get(y).unwrap());
    assert_eq!(xr.missing_indices(), &x_missing);
    assert_eq!(xr.observed_indices(), &[0, 2, 3, 4]);
    for (label, id) in [
        ("x", xr.node()),
        ("y", yr.node()),
        ("x_missing", xr.missing().unwrap()),
        ("y_missing", yr.missing().unwrap()),
        ("x_observed", xr.observed().unwrap()),
        ("y_observed", yr.observed().unwrap()),
    ] {
        assert_eq!(new.find(label), Some(id), "{label}");
    }
    assert_eq!(new[xr.node()].shape(), &[2, 3]);
    assert_eq!(new[yr.missing().unwrap()].shape(), &[3]);
    assert_eq!(imputed.node(x), Some(xr.node()));

    let model = Model::new(new, &state).unwrap();
    let nmu = imputed.node(mu).unwrap();
    assert_eq!(
        model.latents(),
        &[nmu, xr.missing().unwrap(), yr.missing().unwrap()]
    );
    // at any values of the missing parts, the density is the complete model's with those values filled in
    let graph = model.graph().clone();
    for (xm, ym) in [
        ([0.3, -0.7], [1.0, 2.0, -0.5]),
        ([-1.5, 2.0], [0.0, 0.1, 3.0]),
    ] {
        state
            .set(
                &graph,
                xr.missing().unwrap(),
                DoubleTensor::vector(xm.to_vec()),
            )
            .unwrap();
        state
            .set(
                &graph,
                yr.missing().unwrap(),
                DoubleTensor::vector(ym.to_vec()),
            )
            .unwrap();
        state
            .cascade(&graph, &[xr.missing().unwrap(), yr.missing().unwrap()])
            .unwrap();
        let (mut xf, mut yf) = (X, Y);
        for (k, &i) in x_missing.iter().enumerate() {
            xf[i] = xm[k];
        }
        for (k, &i) in y_missing.iter().enumerate() {
            yf[i] = ym[k];
        }
        // the recombined nodes hold the completed values
        assert_eq!(state.f64(xr.node()).unwrap().as_slice(), &xf);
        assert_eq!(state.f64(yr.node()).unwrap().as_slice(), &yf);
        let expected = complete_log_prob(xf, yf);
        assert!((model.log_prob(&state).unwrap() - expected).abs() < 1e-10);
    }
    // gradients flow through the recombination to every latent
    let param = Parameterisation::new(&model, &state).unwrap();
    let z = param.from_state(&state).unwrap();
    let gradient = param
        .log_prob_and_gradient(&z, true, &mut state)
        .unwrap()
        .1
        .unwrap();
    assert!(gradient.iter().all(|g| g.is_finite() && *g != 0.0));
}

#[test]
fn with_nothing_missing_the_log_probability_is_unchanged() {
    let (graph, state, _, x, y) = regression();
    let (new, mut state, imputed) = observe_masked(
        &graph,
        &state,
        &[
            MaskedObservation::new(y, values(Y), mask(&[])),
            MaskedObservation::with_nan_missing(x, values(X)),
        ],
    )
    .unwrap();
    assert_eq!(new.len(), graph.len());
    let yr = imputed.get(y).unwrap();
    assert_eq!(yr.missing(), None);
    assert_eq!(yr.observed(), Some(yr.node()));
    assert!(state.is_observed(yr.node()));
    let all: Vec<NodeId> = new.ids().collect();
    state
        .lazy_eval(&new, &all, &mut KeanuRng::seed_from_u64(0))
        .unwrap();
    let model = Model::new(new, &state).unwrap();
    assert!((model.log_prob(&state).unwrap() - complete_log_prob(X, Y)).abs() < 1e-12);
}

#[test]
fn with_everything_missing_the_node_is_a_latent() {
    let (graph, state, _, _, y) = regression();
    let all_missing = BoolTensor::full(&[2, 3], true);
    let (new, mut state, imputed) = observe_masked(
        &graph,
        &state,
        &[MaskedObservation::new(y, values(Y), all_missing)],
    )
    .unwrap();
    let yr = imputed.get(y).unwrap();
    assert_eq!(yr.observed(), None);
    let missing = yr.missing().unwrap();
    assert_eq!(new[missing].shape(), &[6]);
    assert!(new.find("y_observed").is_none());
    let model = Model::new(new.clone(), &state).unwrap();
    assert!(model.latents().contains(&missing));
    assert!(model.observed().is_empty());
    // the same density as y left latent
    let (graph, mut original, _, x, y) = regression();
    original.set(&graph, y, values(Y)).unwrap();
    original.set(&graph, x, values(X)).unwrap();
    let all: Vec<NodeId> = graph.ids().collect();
    original
        .lazy_eval(&graph, &all, &mut KeanuRng::seed_from_u64(0))
        .unwrap();
    state
        .set(&new, imputed.node(x).unwrap(), values(X))
        .unwrap();
    state
        .set_and_cascade(&new, missing, DoubleTensor::vector(Y.to_vec()))
        .unwrap();
    state.cascade(&new, &[imputed.node(x).unwrap()]).unwrap();
    let expected = Model::new(graph, &original)
        .unwrap()
        .log_prob(&original)
        .unwrap();
    assert!((model.log_prob(&state).unwrap() - expected).abs() < 1e-10);
}

#[test]
fn discrete_and_broadcast_parameters_split_element_by_element() {
    // a binomial count with a per-element probability and a scalar trial count, one element missing
    let b = GraphBuilder::new();
    let p = b.constant(DoubleTensor::vector(vec![0.2, 0.5, 0.9]));
    let k = b.binomial_with_shape(&[3], p, 4).label("k");
    let k = k.id();
    let (graph, state) = b.build().unwrap();
    let missing = BoolTensor::vector(vec![false, true, false]);
    let (new, state, imputed) = observe_masked(
        &graph,
        &state,
        &[MaskedObservation::new(
            k,
            IntTensor::vector(vec![1, 0, 4]),
            missing,
        )],
    )
    .unwrap();
    let kr = imputed.get(k).unwrap();
    let model = Model::new(new, &state).unwrap();
    assert_eq!(model.latents(), &[kr.missing().unwrap()]);
    assert_eq!(
        state.i64(kr.observed().unwrap()).unwrap().as_slice(),
        &[1, 4]
    );
    let value = state.i64(kr.node()).unwrap().as_slice().to_vec();
    assert_eq!((value[0], value[2]), (1, 4));
    // ln Binomial(1 | 4, 0.2) + ln Binomial(4 | 4, 0.9) + ln Binomial(m | 4, 0.5)
    let m = value[1];
    let choose = [1.0, 4.0, 6.0, 4.0, 1.0][m as usize];
    let expected =
        (4.0 * 0.2 * 0.8f64.powi(3)).ln() + 0.9f64.powi(4).ln() + (choose * 0.5f64.powi(4)).ln();
    assert!((model.log_prob(&state).unwrap() - expected).abs() < 1e-12);
}

#[test]
fn rejects_nodes_that_cant_be_split_and_bad_arguments() {
    let b = GraphBuilder::new();
    let mu = b.constant(DoubleTensor::vector(vec![0.0, 0.0]));
    let cov = b.constant(DoubleTensor::new(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap());
    let mvn = b.multivariate_gaussian(mu, cov).label("mvn");
    let d = (mvn * 2.0).label("d");
    let g = b.gaussian_with_shape(&[2], 0.0, 1.0).label("g");
    let (mvn, d, g) = (mvn.id(), d.id(), g.id());
    let (graph, state) = b.build().unwrap();
    let pair = DoubleTensor::vector(vec![1.0, f64::NAN]);

    for (node, needle) in [(mvn, "element-wise"), (d, "deterministic")] {
        let error = observe_masked(
            &graph,
            &state,
            &[MaskedObservation::with_nan_missing(node, pair.clone())],
        )
        .unwrap_err();
        assert!(
            matches!(&error, GraphError::CannotImpute { reason, .. } if reason.contains(needle)),
            "{error}"
        );
    }
    let unknown = NodeId::from_index(99);
    assert_eq!(
        observe_masked(
            &graph,
            &state,
            &[MaskedObservation::with_nan_missing(unknown, pair.clone())]
        )
        .unwrap_err(),
        GraphError::UnknownNode(unknown)
    );
    let wrong_shape = MaskedObservation::with_nan_missing(g, DoubleTensor::vector(vec![1.0]));
    assert!(matches!(
        observe_masked(&graph, &state, &[wrong_shape]).unwrap_err(),
        GraphError::InvalidValue(_)
    ));
    let wrong_mask = MaskedObservation::new(g, pair.clone(), BoolTensor::vector(vec![true]));
    assert!(matches!(
        observe_masked(&graph, &state, &[wrong_mask]).unwrap_err(),
        GraphError::InvalidValue(_)
    ));
    let twice = MaskedObservation::with_nan_missing(g, pair);
    assert!(matches!(
        observe_masked(&graph, &state, &[twice.clone(), twice]).unwrap_err(),
        GraphError::InvalidValue(m) if m.contains("more than once")
    ));
    assert!(matches!(
        observe_masked(&graph, &state, &[]).unwrap_err(),
        GraphError::InvalidValue(_)
    ));
}
