//! Gaussian random walk, AR(p) and GARCH(1,1) nodes: their log probability and gradient equal those of the same
//! process unrolled into one Gaussian node per step, their draws have the requested shape, and malformed shapes are
//! build errors.

use keanu_distributions::KeanuRng;
use keanu_graph::{DoubleVar, GraphBuilder, GraphError, Model, NodeId, State};
use keanu_tensor::DoubleTensor;

mod common;
use common::{assert_close, v};

/// Builds a model with labelled latents (set to `values`) and returns it with its state.
fn model_of(build: impl FnOnce(&GraphBuilder)) -> (Model, State) {
    let b = GraphBuilder::new();
    build(&b);
    let (graph, mut state) = b.build().unwrap();
    let all: Vec<NodeId> = graph.ids().collect();
    state
        .lazy_eval(&graph, &all, &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    (Model::new(graph, &state).unwrap(), state)
}

/// A latent with a wide prior set to `value`, labelled.
fn latent<'b>(b: &'b GraphBuilder, label: &str, value: f64) -> DoubleVar<'b> {
    let x = b.gaussian(0.0, 3.0).label(label);
    b.set_value(x, DoubleTensor::scalar(value));
    x
}

/// Checks that two models have the same log probability and the same gradient with respect to each of `labels`.
fn same_model(a: (Model, State), b: (Model, State), labels: &[&str], what: &str) {
    let (lp_a, lp_b) = (a.0.log_prob(&a.1).unwrap(), b.0.log_prob(&b.1).unwrap());
    assert_close(
        lp_a,
        lp_b,
        1e-10 * (1.0 + lp_b.abs()),
        &format!("{what}: log prob"),
    );
    let (ga, gb) = (
        a.0.log_prob_gradient(&a.1).unwrap(),
        b.0.log_prob_gradient(&b.1).unwrap(),
    );
    for label in labels {
        let id = |m: &Model| -> NodeId { m.find(label).unwrap() };
        let (x, y) = (
            ga.get(id(&a.0)).unwrap().to_scalar().unwrap(),
            gb.get(id(&b.0)).unwrap().to_scalar().unwrap(),
        );
        assert_close(
            x,
            y,
            1e-10 * (1.0 + y.abs()),
            &format!("{what}: d/d{label}"),
        );
    }
}

const X: [f64; 6] = [0.3, 0.9, 0.4, 1.6, 2.0, 1.1];

#[test]
fn a_random_walk_node_is_its_unrolled_steps() {
    fn params(b: &GraphBuilder) -> [DoubleVar<'_>; 4] {
        [
            latent(b, "mu", 0.2),
            latent(b, "sigma", 0.7).exp(),
            latent(b, "m0", -0.1),
            latent(b, "s0", 0.4).exp(),
        ]
    }
    let node = model_of(|b| {
        let [mu, sigma, m0, s0] = params(b);
        let x = b.gaussian_random_walk(6, mu, sigma, m0, s0);
        b.observe(x, v(&X));
    });
    let unrolled = model_of(|b| {
        let [mu, sigma, m0, s0] = params(b);
        let x0 = b.gaussian(m0, s0);
        b.observe(x0, DoubleTensor::scalar(X[0]));
        for t in 1..X.len() {
            let x = b.gaussian(mu + X[t - 1], sigma);
            b.observe(x, DoubleTensor::scalar(X[t]));
        }
    });
    same_model(node, unrolled, &["mu", "sigma", "m0", "s0"], "random walk");
}

#[test]
fn an_autoregressive_node_is_its_unrolled_steps() {
    fn params(b: &GraphBuilder) -> [DoubleVar<'_>; 6] {
        [
            latent(b, "c", 0.1),
            latent(b, "r1", 0.6),
            latent(b, "r2", -0.3),
            latent(b, "sigma", -0.2).exp(),
            latent(b, "m0", 0.2),
            latent(b, "s0", 0.5).exp(),
        ]
    }
    let node = model_of(|b| {
        let [c, r1, r2, sigma, m0, s0] = params(b);
        let rho = b.stack(0, &[r1, r2]);
        let x = b.autoregressive(6, c, rho, sigma, m0, s0);
        b.observe(x, v(&X));
    });
    let unrolled = model_of(|b| {
        let [c, r1, r2, sigma, m0, s0] = params(b);
        for &x in &X[..2] {
            let initial = b.gaussian(m0, s0);
            b.observe(initial, DoubleTensor::scalar(x));
        }
        for t in 2..X.len() {
            let x = b.gaussian(c + r1 * X[t - 1] + r2 * X[t - 2], sigma);
            b.observe(x, DoubleTensor::scalar(X[t]));
        }
    });
    same_model(
        node,
        unrolled,
        &["c", "r1", "r2", "sigma", "m0", "s0"],
        "AR(2)",
    );
}

#[test]
fn a_garch_node_is_its_unrolled_steps() {
    const R: [f64; 6] = [0.3, -1.2, 0.5, 2.1, -0.4, 0.8];
    fn params(b: &GraphBuilder) -> [DoubleVar<'_>; 4] {
        [
            latent(b, "omega", -2.0).exp(),
            latent(b, "alpha", -1.0).sigmoid() * 0.5,
            latent(b, "beta", 0.5).sigmoid() * 0.5,
            latent(b, "v0", -0.1).exp(),
        ]
    }
    let node = model_of(|b| {
        let [omega, alpha, beta, v0] = params(b);
        let r = b.garch11(6, omega, alpha, beta, v0);
        b.observe(r, v(&R));
    });
    let unrolled = model_of(|b| {
        let [omega, alpha, beta, v0] = params(b);
        let mut variance = v0 * v0;
        for (t, &r) in R.iter().enumerate() {
            if t > 0 {
                variance = omega + alpha * (R[t - 1] * R[t - 1]) + beta * variance;
            }
            let x = b.gaussian(0.0, variance.sqrt());
            b.observe(x, DoubleTensor::scalar(r));
        }
    });
    same_model(
        node,
        unrolled,
        &["omega", "alpha", "beta", "v0"],
        "GARCH(1,1)",
    );
}

#[test]
fn process_nodes_sample_series_of_their_shape() {
    let b = GraphBuilder::new();
    let walk = b.gaussian_random_walk_with_shape(&[3, 20], v(&[0.0, 1.0, -1.0]), 0.5, 0.0, 1.0);
    let ar = b.autoregressive_with_shape(&[2, 10], 0.0, v(&[0.5]), 1.0, 0.0, 1.0);
    let garch = b.garch11(15, 0.1, 0.2, 0.5, 1.0);
    let ids = [walk.id(), ar.id(), garch.id()];
    let (graph, mut state) = b.build().unwrap();
    state
        .lazy_eval(&graph, &ids, &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    assert_eq!(state.f64(ids[0]).unwrap().shape(), &[3, 20]);
    assert_eq!(state.f64(ids[1]).unwrap().shape(), &[2, 10]);
    assert_eq!(state.f64(ids[2]).unwrap().shape(), &[15]);
    // the walk with drift 1 climbs, the one with drift −1 falls
    let w = state.f64(ids[0]).unwrap().as_slice();
    assert!(w[39] - w[20] > 5.0 && w[59] - w[40] < -5.0);
    let model = Model::new(graph, &state).unwrap();
    assert!(model.log_prob(&state).unwrap().is_finite());
}

#[test]
fn malformed_process_nodes_are_build_errors() {
    let invalid = |build: fn(&GraphBuilder)| {
        let b = GraphBuilder::new();
        build(&b);
        let result = b.build();
        assert!(
            matches!(result, Err(GraphError::InvalidNode { .. })),
            "expected an invalid node, got {:?}",
            result.map(|_| ())
        );
    };
    // no steps
    invalid(|b| {
        b.gaussian_random_walk(0, 0.0, 1.0, 0.0, 1.0);
    });
    // a parameter that doesn't broadcast to the batch
    invalid(|b| {
        b.gaussian_random_walk_with_shape(&[2, 5], v(&[0.0, 1.0, 2.0]), 1.0, 0.0, 1.0);
    });
    // AR coefficients without a lag axis
    invalid(|b| {
        b.autoregressive(5, 0.0, 0.5, 1.0, 0.0, 1.0);
    });
    // AR coefficients whose batch doesn't fit
    invalid(|b| {
        b.autoregressive_with_shape(
            &[2, 5],
            0.0,
            DoubleTensor::new(&[3, 1], vec![0.1, 0.2, 0.3]).unwrap(),
            1.0,
            0.0,
            1.0,
        );
    });
    // GARCH parameters outside the stationary region can't be sampled for the node's initial value
    let b = GraphBuilder::new();
    let r = b.garch11(5, 0.1, 0.6, 0.6, 1.0);
    let id = r.id();
    let (graph, mut state) = b.build().unwrap();
    assert!(
        state
            .lazy_eval(&graph, &[id], &mut KeanuRng::seed_from_u64(1))
            .is_err()
    );
    // and observed there, they are impossible
    state
        .observe(&graph, id, v(&[0.1, 0.2, 0.3, 0.4, 0.5]))
        .unwrap();
    let model = Model::new(graph, &state).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(model.log_prob(&state).unwrap(), f64::NEG_INFINITY);
}
