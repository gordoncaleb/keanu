//! The log-density op: its values against the distributions' log probabilities, its first and second derivatives
//! against finite differences, and the kinds and types it rejects.

use keanu_distributions::KeanuRng;
use keanu_graph::op::{DistributionKind, Op};
use keanu_graph::{
    DoubleVar, GraphBuilder, GraphError, Model, NodeId, Parameterisation, Space, State, Value,
    forward_partials, node_log_prob, reverse_partials,
};
use keanu_tensor::{DoubleTensor, IntTensor};
use std::sync::Arc;

mod common;
use common::{assert_close, assert_close_rel, s, v};

fn evaluate(graph: &keanu_graph::Graph, state: &mut State) {
    let all: Vec<NodeId> = graph.ids().collect();
    state
        .lazy_eval(graph, &all, &mut KeanuRng::seed_from_u64(1))
        .unwrap();
}

fn truncated_gaussian() -> DistributionKind {
    DistributionKind::Truncated(Arc::new(DistributionKind::Gaussian))
}

/// Each kind with a value of three elements and its parameters, some scalar and some vectors to broadcast.
fn cases() -> Vec<(DistributionKind, Value, Vec<Value>)> {
    let f = |x: DoubleTensor| Value::F64(x);
    let i = |x: Vec<i64>| Value::I64(IntTensor::vector(x));
    vec![
        (
            DistributionKind::Gaussian,
            f(v(&[0.3, -1.2, 2.0])),
            vec![f(v(&[0.1, 0.2, 0.3])), f(s(1.5))],
        ),
        (
            DistributionKind::Gamma,
            f(v(&[0.5, 2.0, 3.5])),
            vec![f(s(2.0)), f(v(&[1.5, 2.5, 0.7]))],
        ),
        (
            DistributionKind::Poisson,
            i(vec![0, 3, 7]),
            vec![f(v(&[2.5, 0.4, 6.0]))],
        ),
        (
            DistributionKind::NegativeBinomial,
            i(vec![0, 4, 9]),
            vec![f(s(3.0)), f(v(&[0.5, 2.0, 10.0]))],
        ),
        (
            truncated_gaussian(),
            f(v(&[0.2, 1.0, 2.9])),
            vec![f(s(0.5)), f(s(1.0)), f(s(0.0)), f(s(3.0))],
        ),
    ]
}

#[test]
fn the_summed_log_density_is_the_distribution_nodes_log_prob() {
    for (kind, value, parameters) in cases() {
        let name = kind.name();
        let b = GraphBuilder::new();
        let params: Vec<NodeId> = parameters
            .iter()
            .map(|p| b.add_node(Op::Constant(p.clone()), &[]).unwrap())
            .collect();
        let node = b
            .add_distribution(kind.clone(), Some(value.shape()), &params)
            .unwrap();
        b.observe_of(node, value.clone()).unwrap();
        let x = b.add_node(Op::Constant(value.clone()), &[]).unwrap();
        let mut parents = vec![x];
        parents.extend(&params);
        let density = b.add_node(Op::LogDensity(kind.clone()), &parents).unwrap();
        let (graph, mut state) = b.build().unwrap();
        evaluate(&graph, &mut state);
        assert_eq!(graph[density].shape(), &[3]);
        let elements = state.f64(density).unwrap();
        assert!(elements.as_slice().iter().all(|x| x.is_finite()), "{name}");
        assert_close_rel(
            elements.as_slice().iter().sum(),
            node_log_prob(&graph, &state, node).unwrap(),
            1e-13,
            name,
        );
    }
}

#[test]
fn a_value_outside_the_support_has_log_density_minus_infinity() {
    let b = GraphBuilder::new();
    let x = b.constant(v(&[-1.0, 2.0]));
    let density = b.log_density(
        DistributionKind::Gamma,
        x,
        &[b.scalar(1.0).id(), b.scalar(2.0).id()],
    );
    let density = density.id();
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state);
    let values = state.f64(density).unwrap().as_slice().to_vec();
    assert_eq!(values[0], f64::NEG_INFINITY);
    assert!(values[1].is_finite());
}

/// A latent with a standard Gaussian prior, set to `value`.
fn latent<'b>(b: &'b GraphBuilder, value: DoubleTensor) -> DoubleVar<'b> {
    let x = b.gaussian_with_shape(value.shape(), 0.0, 1.0);
    b.set_value(x, value);
    x
}

/// A model whose log density's value (for a continuous kind) and every parameter depend on latents, through
/// transforms that keep them valid, with the log density summed in a factor. Returns the model, its state, the
/// log density node and the latents.
fn differentiable_case(kind: &DistributionKind) -> (Model, State, NodeId, Vec<NodeId>) {
    let b = GraphBuilder::new();
    let u = latent(&b, v(&[0.3, -0.4, 0.8]));
    let p = latent(&b, s(0.2));
    let q = latent(&b, v(&[-0.3, 0.5, 0.1]));
    let (value, parameters): (NodeId, Vec<NodeId>) = match kind {
        DistributionKind::Gaussian => (u.id(), vec![(q * p).id(), (p.exp() + 0.5).id()]),
        DistributionKind::Gamma => (u.exp().id(), vec![p.exp().id(), (q.exp() + 0.5).id()]),
        DistributionKind::Poisson => (
            b.int_constant(IntTensor::vector(vec![0, 3, 7])).id(),
            vec![(q * p).exp().id()],
        ),
        DistributionKind::NegativeBinomial => (
            b.int_constant(IntTensor::vector(vec![0, 4, 9])).id(),
            vec![(p.exp() * 3.0).id(), (q.exp() + 0.2).id()],
        ),
        _ => {
            // a Gaussian truncated to [lower, upper], with the value inside the bounds
            let lower = p - 1.0;
            let upper = q.exp() + 1.0;
            let value = lower + (upper - lower) * u.sigmoid();
            (
                value.id(),
                vec![q.id(), (p.exp() + 0.5).id(), lower.id(), upper.id()],
            )
        }
    };
    let mut parents = vec![value];
    parents.extend(&parameters);
    let density = b.add_node(Op::LogDensity(kind.clone()), &parents).unwrap();
    b.factor(b.double_var(density) * b.constant(v(&[1.0, 0.7, 1.3])));
    // a count value doesn't depend on u
    let latents = match kind {
        DistributionKind::Poisson | DistributionKind::NegativeBinomial => vec![p.id(), q.id()],
        _ => vec![u.id(), p.id(), q.id()],
    };
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state);
    let model = Model::new(graph, &state).unwrap();
    (model, state, density, latents)
}

#[test]
fn forward_and_reverse_partials_match_finite_differences() {
    for (kind, ..) in cases() {
        let name = kind.name();
        let (model, mut state, density, latents) = differentiable_case(&kind);
        let graph = model.graph().clone();
        let reverse = reverse_partials(&graph, &state, density, &latents).unwrap();
        for &latent in &latents {
            let forward = forward_partials(&graph, &state, latent, &[density]).unwrap();
            let forward = forward.get(density).unwrap().as_slice().to_vec();
            let reversed = reverse.get(latent).unwrap().as_slice().to_vec();
            let base = state.f64(latent).unwrap().clone();
            let n = base.len();
            for j in 0..n {
                let h = 1e-6;
                let mut at = |delta: f64| {
                    let mut x = base.as_slice().to_vec();
                    x[j] += delta;
                    let x = DoubleTensor::new(base.shape(), x).unwrap();
                    state.set_and_cascade(&graph, latent, x).unwrap();
                    state.f64(density).unwrap().as_slice().to_vec()
                };
                let (plus, minus) = (at(h), at(-h));
                state.set_and_cascade(&graph, latent, base.clone()).unwrap();
                for i in 0..3 {
                    let fd = (plus[i] - minus[i]) / (2.0 * h);
                    // partials are shaped [of..., wrt...]: element [i, j]
                    assert_close_rel(forward[i * n + j], fd, 1e-6, &format!("{name} forward"));
                    assert_close_rel(reversed[i * n + j], fd, 1e-6, &format!("{name} reverse"));
                }
            }
        }
    }
}

#[test]
fn the_hessian_through_a_log_density_matches_finite_differences() {
    for (kind, ..) in cases() {
        let name = kind.name();
        let (model, mut state, ..) = differentiable_case(&kind);
        let param = Parameterisation::with_space(&model, &state, Space::Constrained).unwrap();
        let z = param.from_state(&state).unwrap();
        let n = z.len();
        let hessian = param
            .hessian(&z, false, &mut state)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let h = 1e-5;
        for j in 0..n {
            let (mut plus, mut minus) = (z.clone(), z.clone());
            plus[j] += h;
            minus[j] -= h;
            let (_, gp) = param
                .log_prob_and_gradient(&plus, false, &mut state)
                .unwrap();
            let (_, gm) = param
                .log_prob_and_gradient(&minus, false, &mut state)
                .unwrap();
            let (gp, gm) = (gp.unwrap(), gm.unwrap());
            for i in 0..n {
                let fd = (gp[i] - gm[i]) / (2.0 * h);
                let actual = hessian.as_slice()[i * n + j];
                assert_close(
                    actual,
                    fd,
                    2e-6 * fd.abs().max(1.0),
                    &format!("{name} H[{i}][{j}]"),
                );
            }
        }
        // and the gradient against finite differences of the log probability
        let (_, g) = param.log_prob_and_gradient(&z, false, &mut state).unwrap();
        let g = g.unwrap();
        for j in 0..n {
            let (mut plus, mut minus) = (z.clone(), z.clone());
            plus[j] += h;
            minus[j] -= h;
            let (lp, _) = param
                .log_prob_and_gradient(&plus, false, &mut state)
                .unwrap();
            let (lm, _) = param
                .log_prob_and_gradient(&minus, false, &mut state)
                .unwrap();
            assert_close_rel(
                g[j],
                (lp - lm) / (2.0 * h),
                1e-6,
                &format!("{name} gradient"),
            );
        }
    }
}

#[test]
fn kinds_without_an_elementwise_density_are_rejected() {
    let b = GraphBuilder::new();
    let x = b.constant(v(&[0.2, 0.8]));
    let mu = b.constant(v(&[0.0, 0.0]));
    let covariance = b.constant(DoubleTensor::eye(2).unwrap());
    for (kind, parents) in [
        (
            DistributionKind::MultivariateGaussian,
            vec![x.id(), mu.id(), covariance.id()],
        ),
        (DistributionKind::Dirichlet, vec![x.id(), mu.id()]),
        (DistributionKind::Categorical, vec![x.id(), mu.id()]),
        (
            DistributionKind::Mixture(Arc::new(DistributionKind::Gaussian)),
            vec![x.id(), mu.id(), mu.id(), mu.id()],
        ),
    ] {
        let name = kind.name();
        let error = b.add_node(Op::LogDensity(kind), &parents).unwrap_err();
        assert!(
            error.to_string().contains("element-wise"),
            "{name}: {error}"
        );
    }
    let flag = b.bool_constant(keanu_tensor::BoolTensor::vector(vec![true]));
    let error = b
        .add_node(
            Op::LogDensity(DistributionKind::Bernoulli),
            &[flag.id(), b.scalar(0.5).id()],
        )
        .unwrap_err();
    assert!(error.to_string().contains("element-wise"), "{error}");
}

#[test]
fn values_and_parameters_of_the_wrong_type_or_number_are_rejected() {
    let b = GraphBuilder::new();
    let x = b.constant(v(&[1.0, 2.0]));
    let rate = b.scalar(1.0);
    // a Poisson value is i64
    let error = b
        .add_node(
            Op::LogDensity(DistributionKind::Poisson),
            &[x.id(), rate.id()],
        )
        .unwrap_err();
    assert!(matches!(error, GraphError::InvalidNode { .. }), "{error}");
    // a Gaussian needs two parameters
    assert!(
        b.add_node(
            Op::LogDensity(DistributionKind::Gaussian),
            &[x.id(), rate.id()]
        )
        .is_err()
    );
    // a Binomial's n is i64
    let k = b.int_constant(IntTensor::vector(vec![1, 2]));
    assert!(
        b.add_node(
            Op::LogDensity(DistributionKind::Binomial),
            &[k.id(), rate.id(), rate.id()]
        )
        .is_err()
    );
    // shapes must broadcast
    let three = b.constant(v(&[0.0, 1.0, 2.0]));
    assert!(
        b.add_node(
            Op::LogDensity(DistributionKind::Gaussian),
            &[x.id(), three.id(), rate.id()]
        )
        .is_err()
    );
}
