//! Factor nodes: terms of the joint log probability, their gradients and Hessians, impossible values, incremental
//! updates and where they can't be used.

use keanu_distributions::KeanuRng;
use keanu_graph::op::Op;
use keanu_graph::{
    GraphBuilder, GraphError, Model, NodeId, Parameterisation, Space, State, Value, joint_log_prob,
    log_likelihood, log_prior, node_log_prob,
};
use keanu_tensor::{DoubleTensor, IntTensor};

mod common;
use common::{assert_close, assert_close_rel, s, v};

fn evaluated(b: GraphBuilder) -> (Model, State) {
    let (graph, mut state) = b.build().unwrap();
    let all: Vec<NodeId> = graph.ids().collect();
    state
        .lazy_eval(&graph, &all, &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    let model = Model::new(graph, &state).unwrap();
    (model, state)
}

/// mu ~ N(0, 1), w ~ N(0, 1) of shape [3], y ~ N(mu, 1) observed, and the factor Σᵢ (sin(wᵢ) mu - (mu - 1)²)
/// (the scalar term broadcast to w's shape, so counted three times).
fn model_with_a_factor() -> (Model, State, NodeId, NodeId, NodeId, NodeId) {
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 1.0);
    let w = b.gaussian_with_shape(&[3], 0.0, 1.0);
    b.set_value(mu, s(0.4));
    b.set_value(w, v(&[0.3, -0.7, 1.1]));
    let y = b.gaussian(mu, 1.0);
    b.observe(y, s(0.9));
    let f = b.factor(w.sin() * mu - (mu - 1.0) * (mu - 1.0));
    let (mu, w, y, f) = (mu.id(), w.id(), y.id(), f.id());
    let (model, state) = evaluated(b);
    (model, state, mu, w, y, f)
}

fn gaussian(x: f64, mu: f64, sigma: f64) -> f64 {
    let z = (x - mu) / sigma;
    -0.5 * z * z - sigma.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
}

#[test]
fn a_factor_adds_its_sum_to_the_joint_log_prob_and_the_likelihood() {
    let (model, state, mu, w, y, f) = model_with_a_factor();
    let graph = model.graph();
    assert_eq!(model.factors(), &[f]);
    assert_eq!(graph.factors().collect::<Vec<_>>(), vec![f]);
    assert!(graph[f].is_factor() && !graph[f].is_probabilistic());
    assert_eq!(graph[f].shape(), &[] as &[usize]);
    let ws = [0.3, -0.7, 1.1];
    let term = ws.iter().map(|w: &f64| w.sin() * 0.4).sum::<f64>() - 3.0 * 0.36;
    assert_close(state.f64(f).unwrap().as_slice()[0], term, 1e-14, "value");
    assert_close(
        node_log_prob(graph, &state, f).unwrap(),
        term,
        1e-14,
        "term",
    );
    let prior = gaussian(0.4, 0.0, 1.0) + ws.iter().map(|&w| gaussian(w, 0.0, 1.0)).sum::<f64>();
    let likelihood = gaussian(0.9, 0.4, 1.0);
    let joint = model.log_prob(&state).unwrap();
    assert_close(joint, prior + likelihood + term, 1e-12, "joint");
    assert_close(
        joint_log_prob(graph, &state).unwrap(),
        joint,
        1e-12,
        "graph joint",
    );
    let ll = model.log_likelihood(&state).unwrap();
    assert_close(ll, likelihood + term, 1e-12, "likelihood");
    assert_close(
        log_likelihood(graph, &state).unwrap(),
        ll,
        1e-12,
        "graph likelihood",
    );
    assert_close(log_prior(graph, &state).unwrap(), prior, 1e-12, "prior");
    let (fused, _) = model.log_prob_and_gradient(&state).unwrap();
    assert_eq!(fused, joint);
    let _ = (mu, w, y);
}

#[test]
fn the_gradient_and_hessian_through_a_factor_match_finite_differences() {
    let (model, mut state, mu, w, ..) = model_with_a_factor();
    // the factor's gradient with respect to mu: Σ sin(w) - 6 (mu - 1), plus the Gaussians'
    let gradient = model.log_prob_gradient(&state).unwrap();
    let ws = [0.3, -0.7, 1.1];
    let expected =
        ws.iter().map(|w: &f64| w.sin()).sum::<f64>() - 6.0 * (0.4 - 1.0) - 0.4 + (0.9 - 0.4);
    assert_close(
        gradient.get(mu).unwrap().as_slice()[0],
        expected,
        1e-12,
        "d/dmu",
    );
    let dw = gradient.get(w).unwrap();
    for (i, &wi) in ws.iter().enumerate() {
        assert_close(dw.as_slice()[i], wi.cos() * 0.4 - wi, 1e-12, "d/dw");
    }
    for space in [Space::Constrained, Space::Unconstrained] {
        let param = Parameterisation::with_space(&model, &state, space).unwrap();
        let z = param.from_state(&state).unwrap();
        let n = z.len();
        let (_, gradient) = param.log_prob_and_gradient(&z, true, &mut state).unwrap();
        let gradient = gradient.unwrap();
        let hessian = param.hessian(&z, true, &mut state).unwrap();
        let h = 1e-6;
        for j in 0..n {
            let (mut plus, mut minus) = (z.clone(), z.clone());
            plus[j] += h;
            minus[j] -= h;
            let (lp, gp) = param
                .log_prob_and_gradient(&plus, true, &mut state)
                .unwrap();
            let (lm, gm) = param
                .log_prob_and_gradient(&minus, true, &mut state)
                .unwrap();
            assert_close_rel(gradient[j], (lp - lm) / (2.0 * h), 1e-7, "gradient");
            let (gp, gm) = (gp.unwrap(), gm.unwrap());
            for i in 0..n {
                let fd = (gp[i] - gm[i]) / (2.0 * h);
                assert_close_rel(hessian.as_slice()[i * n + j], fd, 1e-6, "Hessian");
            }
        }
    }
}

#[test]
fn a_factor_that_is_not_finite_makes_the_point_impossible() {
    for (x, what) in [(-1.0, "NaN"), (0.0, "-infinity")] {
        let b = GraphBuilder::new();
        let latent = b.gaussian(0.0, 1.0);
        b.set_value(latent, s(x));
        b.factor(latent.ln());
        let latent = latent.id();
        let (model, state) = evaluated(b);
        assert_eq!(model.log_prob(&state).unwrap(), f64::NEG_INFINITY, "{what}");
        let (log_prob, gradient) = model.log_prob_and_gradient(&state).unwrap();
        assert_eq!(log_prob, f64::NEG_INFINITY, "{what}");
        assert!(gradient.is_none(), "{what}");
        let _ = latent;
    }
    // +infinity is impossible too
    let b = GraphBuilder::new();
    let latent = b.gaussian(0.0, 1.0);
    b.set_value(latent, s(0.0));
    b.factor(-latent.ln());
    let (model, state) = evaluated(b);
    assert_eq!(model.log_prob(&state).unwrap(), f64::NEG_INFINITY);
}

#[test]
fn log_prob_after_updates_the_factors_a_proposal_reaches() {
    let (model, mut state, mu, w, ..) = model_with_a_factor();
    let before = model.log_prob(&state).unwrap();
    let proposal = vec![(mu, Value::F64(s(-0.2)))];
    assert!(
        model
            .affected_by(&[mu])
            .unwrap()
            .contains(&model.factors()[0])
    );
    let after = model.log_prob_after(&mut state, &proposal, before).unwrap();
    assert_close(after, model.log_prob(&state).unwrap(), 1e-12, "mu");
    let before = after;
    let proposal = vec![(w, Value::F64(v(&[0.0, 0.5, -1.0])))];
    let after = model.log_prob_after(&mut state, &proposal, before).unwrap();
    assert_close(after, model.log_prob(&state).unwrap(), 1e-12, "w");
}

#[test]
fn a_factor_needs_an_f64_expression() {
    let b = GraphBuilder::new();
    let k = b.int_constant(IntTensor::vector(vec![1, 2]));
    let error = b.add_node(Op::Factor, &[k.id()]).unwrap_err();
    assert!(matches!(error, GraphError::InvalidValue(_)), "{error}");
    assert!(b.add_node(Op::Factor, &[]).is_err());
}

#[test]
fn a_factor_is_not_observable() {
    let b = GraphBuilder::new();
    let x = b.gaussian(0.0, 1.0);
    let f = b.factor(x);
    b.observe(f, DoubleTensor::scalar(0.0));
    assert!(matches!(b.build(), Err(GraphError::NotObservable(_))));
}

#[test]
fn scan_bodies_cannot_hold_factors() {
    let b = GraphBuilder::new();
    let x0 = b.scalar(0.0);
    let xs = b.constant(v(&[1.0, 2.0]));
    b.scan(x0, &[xs], &[], |body, carry, xs, _| {
        body.factor(carry);
        carry + xs[0]
    });
    let error = b.build().unwrap_err();
    assert!(error.to_string().contains("factor"), "{error}");
}
