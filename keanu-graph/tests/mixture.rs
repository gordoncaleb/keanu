//! Mixture distributions: agreement with a single component, normalisation, gradients against finite differences,
//! sampled moments, supports, batch shapes and invalid mixtures.

use keanu_distributions::{Gaussian, KeanuRng, Poisson};
use keanu_graph::op::DistributionKind;
use keanu_graph::{
    DoubleVar, Graph, GraphBuilder, GraphError, IntVar, NodeId, State, Support, Value,
    node_log_prob, node_log_prob_gradient, node_support,
};
use keanu_tensor::{DoubleTensor, IntTensor};
use std::sync::Arc;

const INF: f64 = f64::INFINITY;

fn s(v: f64) -> DoubleTensor {
    DoubleTensor::scalar(v)
}

fn v(values: &[f64]) -> DoubleTensor {
    DoubleTensor::vector(values.to_vec())
}

/// A graph with a mixture node of the `component` kind whose parameters and weights are constants.
struct Mixed {
    graph: Graph,
    state: State,
    node: NodeId,
    /// The component parameters, then the weights.
    parameters: Vec<NodeId>,
}

fn mixed(
    component: DistributionKind,
    parameters: &[Value],
    weights: DoubleTensor,
    shape: Option<&[usize]>,
) -> Mixed {
    let b = GraphBuilder::new();
    let mut ids: Vec<NodeId> = parameters
        .iter()
        .map(|p| {
            b.add_node(keanu_graph::op::Op::Constant(p.clone()), &[])
                .unwrap()
        })
        .collect();
    let w = b.constant(weights);
    let discrete = component.value_type() != keanu_graph::DType::F64;
    let node = match (discrete, shape) {
        (false, None) => b.mixture::<DoubleVar>(component, &ids, w).id(),
        (false, Some(shape)) => b
            .mixture_with_shape::<DoubleVar>(shape, component, &ids, w)
            .id(),
        (true, None) => b.mixture::<IntVar>(component, &ids, w).id(),
        (true, Some(shape)) => b
            .mixture_with_shape::<IntVar>(shape, component, &ids, w)
            .id(),
    };
    ids.push(w.id());
    let (graph, mut state) = b.build().unwrap();
    state
        .lazy_eval(&graph, &[node], &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    Mixed {
        graph,
        state,
        node,
        parameters: ids,
    }
}

fn f(t: DoubleTensor) -> Value {
    Value::F64(t)
}

impl Mixed {
    fn log_prob_at(&mut self, x: Value) -> f64 {
        self.state.set(&self.graph, self.node, x).unwrap();
        node_log_prob(&self.graph, &self.state, self.node).unwrap()
    }

    fn parameter(&self, i: usize) -> DoubleTensor {
        self.state
            .value(self.parameters[i])
            .unwrap()
            .as_f64()
            .unwrap()
            .clone()
    }

    fn log_prob_with(&self, parameter: usize, value: DoubleTensor) -> f64 {
        let mut state = self.state.clone();
        state
            .set(&self.graph, self.parameters[parameter], f(value))
            .unwrap();
        node_log_prob(&self.graph, &state, self.node).unwrap()
    }
}

/// A gradient part summed over the dimensions its parameter was broadcast along (a scalar's total, a trailing
/// component axis kept).
fn summed_to(part: &DoubleTensor, shape: &[usize]) -> Vec<f64> {
    let len: usize = shape.iter().product();
    if part.shape() == shape {
        return part.as_slice().to_vec();
    }
    if len == 1 {
        return vec![part.sum()];
    }
    // the parameter is [.., C] with a part of shape [.., C]: sum the leading copies
    let mut out = vec![0.0; len];
    for (i, x) in part.as_slice().iter().enumerate() {
        out[i % len] += x;
    }
    out
}

#[test]
fn a_single_component_is_that_component() {
    let x = v(&[-1.3, 0.2, 2.5]);
    let mut m = mixed(
        DistributionKind::Gaussian,
        &[f(v(&[0.5])), f(v(&[2.0]))],
        v(&[1.0]),
        Some(&[3]),
    );
    let expected = Gaussian::new(s(0.5), s(2.0)).log_prob(&x).unwrap().sum();
    let actual = m.log_prob_at(f(x.clone()));
    assert!((actual - expected).abs() < 1e-12, "{actual} vs {expected}");
    let gradient = node_log_prob_gradient(&m.graph, &m.state, m.node).unwrap();
    let g = Gaussian::new(s(0.5), s(2.0)).d_log_prob(&x).unwrap();
    let close = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(a, b)| (a - b).abs() < 1e-12);
    assert!(close(
        gradient.value.as_ref().unwrap().as_slice(),
        g.x.as_slice()
    ));
    let mu = summed_to(gradient.parameters[0].as_ref().unwrap(), &[1]);
    assert!(close(&mu, &[g.mu.sum()]));
    let sigma = summed_to(gradient.parameters[1].as_ref().unwrap(), &[1]);
    assert!(close(&sigma, &[g.sigma.sum()]));

    // and a count distribution
    let k = IntTensor::vector(vec![0, 3, 7]);
    let mut m = mixed(
        DistributionKind::Poisson,
        &[f(v(&[3.5]))],
        v(&[1.0]),
        Some(&[3]),
    );
    let expected = Poisson::new(s(3.5)).log_prob(&k).unwrap().sum();
    let actual = m.log_prob_at(Value::I64(k));
    assert!((actual - expected).abs() < 1e-12, "{actual} vs {expected}");
}

#[test]
fn a_mixture_of_identical_components_is_one_of_them() {
    let x = v(&[-0.4, 1.9]);
    let mut m = mixed(
        DistributionKind::Gaussian,
        &[f(s(1.0)), f(s(0.7))],
        v(&[0.2, 0.3, 0.5]),
        Some(&[2]),
    );
    let expected = Gaussian::new(s(1.0), s(0.7)).log_prob(&x).unwrap().sum();
    let actual = m.log_prob_at(f(x));
    assert!((actual - expected).abs() < 1e-12, "{actual} vs {expected}");
}

#[test]
fn the_log_density_is_the_log_of_the_weighted_sum() {
    let x = 0.8;
    let mut m = mixed(
        DistributionKind::Gaussian,
        &[f(v(&[-1.0, 2.0])), f(v(&[0.5, 1.5]))],
        v(&[0.25, 0.75]),
        None,
    );
    let density = |mu: f64, sigma: f64| {
        Gaussian::new(s(mu), s(sigma))
            .log_prob(&s(x))
            .unwrap()
            .as_slice()[0]
            .exp()
    };
    let expected = (0.25 * density(-1.0, 0.5) + 0.75 * density(2.0, 1.5)).ln();
    let actual = m.log_prob_at(f(s(x)));
    assert!((actual - expected).abs() < 1e-12, "{actual} vs {expected}");
    // far in a tail the logsumexp stays finite where the densities underflow
    let tail = m.log_prob_at(f(s(-60.0)));
    assert!(tail.is_finite() && tail < -800.0, "{tail}");
}

#[test]
fn mixture_densities_integrate_to_one() {
    let truncated = DistributionKind::Truncated(Arc::new(DistributionKind::Gaussian));
    let cases: Vec<(DistributionKind, Vec<Value>, DoubleTensor, f64, f64)> = vec![
        (
            DistributionKind::Gaussian,
            vec![f(v(&[-2.0, 1.0, 4.0])), f(v(&[0.5, 1.0, 2.0]))],
            v(&[0.2, 0.5, 0.3]),
            -20.0,
            25.0,
        ),
        (
            DistributionKind::Gamma,
            vec![f(v(&[1.0, 0.5])), f(v(&[2.0, 6.0]))],
            v(&[0.4, 0.6]),
            0.0,
            40.0,
        ),
        // components truncated to different ranges
        (
            truncated,
            vec![
                f(v(&[0.0, 3.0])),
                f(s(1.0)),
                f(v(&[-1.0, 2.0])),
                f(v(&[1.0, 6.0])),
            ],
            v(&[0.3, 0.7]),
            -1.0,
            6.0,
        ),
    ];
    for (component, parameters, weights, lower, upper) in cases {
        let name = format!("{component:?}");
        let mut m = mixed(component, &parameters, weights, None);
        let n = 40_000;
        let h = (upper - lower) / n as f64;
        let total: f64 = (0..n)
            .map(|i| {
                let x = lower + (i as f64 + 0.5) * h;
                m.log_prob_at(f(s(x))).exp() * h
            })
            .sum();
        assert!((total - 1.0).abs() < 1e-5, "{name}: {total}");
    }
    // a count mixture's probabilities sum to 1
    let mut m = mixed(
        DistributionKind::Poisson,
        &[f(v(&[1.5, 12.0]))],
        v(&[0.6, 0.4]),
        None,
    );
    let total: f64 = (0..100)
        .map(|k| m.log_prob_at(Value::I64(IntTensor::scalar(k))).exp())
        .sum();
    assert!((total - 1.0).abs() < 1e-12, "{total}");
}

/// Checks the gradient with respect to the value (if f64), each component parameter and the weights against
/// central differences. The weights are moved along e_i - e_j, which keeps them on the simplex, so the difference
/// gives g_i - g_j.
fn check_gradients(m: &mut Mixed, x: Value) {
    const H: f64 = 1e-6;
    let tolerance = |numeric: f64| 1e-5 * numeric.abs().max(1.0);
    m.state.set(&m.graph, m.node, x.clone()).unwrap();
    let gradient = node_log_prob_gradient(&m.graph, &m.state, m.node).unwrap();
    let weights_index = m.parameters.len() - 1;
    for (p, analytic) in gradient.parameters.iter().enumerate() {
        let at = m.parameter(p);
        let analytic = summed_to(
            analytic
                .as_ref()
                .unwrap_or_else(|| panic!("parameter {p} has no gradient")),
            at.shape(),
        );
        let moved = |i: usize, j: Option<usize>, d: f64| {
            let mut data = at.as_slice().to_vec();
            data[i] += d;
            if let Some(j) = j {
                data[j] -= d;
            }
            m.log_prob_with(p, DoubleTensor::new(at.shape(), data).unwrap())
        };
        if p == weights_index {
            let count = at.len();
            for i in 0..count {
                for j in (0..count).filter(|&j| j != i) {
                    let numeric = (moved(i, Some(j), H) - moved(i, Some(j), -H)) / (2.0 * H);
                    let expected = analytic[i] - analytic[j];
                    assert!(
                        (numeric - expected).abs() < tolerance(numeric),
                        "weights {i} - {j}: analytic {expected}, numeric {numeric}"
                    );
                }
            }
            continue;
        }
        for (i, &analytic) in analytic.iter().enumerate() {
            let numeric = (moved(i, None, H) - moved(i, None, -H)) / (2.0 * H);
            assert!(
                (numeric - analytic).abs() < tolerance(numeric),
                "parameter {p}[{i}]: analytic {analytic}, numeric {numeric}"
            );
        }
    }
    if let Value::F64(x) = &x {
        let analytic = gradient.value.expect("a value gradient");
        assert_eq!(analytic.shape(), x.shape());
        for i in 0..x.len() {
            let mut state = m.state.clone();
            let mut at = |d: f64| {
                let mut data = x.as_slice().to_vec();
                data[i] += d;
                state
                    .set(
                        &m.graph,
                        m.node,
                        f(DoubleTensor::new(x.shape(), data).unwrap()),
                    )
                    .unwrap();
                node_log_prob(&m.graph, &state, m.node).unwrap()
            };
            let numeric = (at(H) - at(-H)) / (2.0 * H);
            assert!(
                (numeric - analytic.as_slice()[i]).abs() < tolerance(numeric),
                "value[{i}]: analytic {}, numeric {numeric}",
                analytic.as_slice()[i]
            );
        }
    }
}

#[test]
fn gradients_match_finite_differences() {
    // separate parameters per component
    let mut m = mixed(
        DistributionKind::Gaussian,
        &[f(v(&[-1.0, 0.5, 2.0])), f(v(&[0.6, 1.3, 0.9]))],
        v(&[0.2, 0.3, 0.5]),
        Some(&[4]),
    );
    check_gradients(&mut m, f(v(&[-1.4, 0.1, 0.9, 3.0])));

    // a shared scale and a batch of per-row weights
    let mut m = mixed(
        DistributionKind::LogNormal,
        &[f(v(&[0.0, 1.2])), f(s(0.5))],
        DoubleTensor::new(&[3, 2], vec![0.5, 0.5, 0.1, 0.9, 0.7, 0.3]).unwrap(),
        None,
    );
    check_gradients(&mut m, f(v(&[0.8, 2.5, 4.0])));

    // truncated components
    let mut m = mixed(
        DistributionKind::Truncated(Arc::new(DistributionKind::Gaussian)),
        &[
            f(v(&[0.0, 3.0])),
            f(v(&[1.0, 0.7])),
            f(v(&[-1.0, 2.0])),
            f(v(&[1.5, 6.0])),
        ],
        v(&[0.4, 0.6]),
        Some(&[3]),
    );
    check_gradients(&mut m, f(v(&[-0.5, 1.2, 2.6])));

    // a count mixture: gradients with respect to the parameters and weights only
    let mut m = mixed(
        DistributionKind::NegativeBinomial,
        &[f(v(&[2.0, 9.0])), f(v(&[3.0, 1.5]))],
        v(&[0.35, 0.65]),
        Some(&[3]),
    );
    check_gradients(&mut m, Value::I64(IntTensor::vector(vec![0, 4, 11])));
}

#[test]
fn samples_match_the_mixture_moments() {
    let n = 100_000;
    let m = mixed(
        DistributionKind::Gaussian,
        &[f(v(&[-2.0, 3.0])), f(v(&[1.0, 0.5]))],
        v(&[0.3, 0.7]),
        Some(&[n]),
    );
    let x = m.state.value(m.node).unwrap().as_f64().unwrap().clone();
    let mean = x.sum() / n as f64;
    let variance = x.as_slice().iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    // mean Σ w μ = 1.5, variance Σ w (σ² + μ²) - 1.5² = 5.725; standard errors about 0.008 and 0.02
    assert!((mean - 1.5).abs() < 0.04, "mean {mean}");
    assert!((variance - 5.725).abs() < 0.1, "variance {variance}");
    // the share drawn from the second component
    let second = x.as_slice().iter().filter(|&&x| x > 0.5).count() as f64 / n as f64;
    assert!((second - 0.7).abs() < 0.01, "share {second}");

    // counts: mean Σ w μ = 0.6 · 1.5 + 0.4 · 12 = 5.7, variance Σ w (μ + μ²) - 5.7² = 32.16
    let m = mixed(
        DistributionKind::Poisson,
        &[f(v(&[1.5, 12.0]))],
        v(&[0.6, 0.4]),
        Some(&[n]),
    );
    let k = m.state.value(m.node).unwrap().as_i64().unwrap().clone();
    let k: Vec<f64> = k.as_slice().iter().map(|&k| k as f64).collect();
    let mean = k.iter().sum::<f64>() / n as f64;
    let variance = k.iter().map(|k| (k - mean).powi(2)).sum::<f64>() / n as f64;
    assert!((mean - 5.7).abs() < 0.08, "mean {mean}");
    assert!((variance - 32.16).abs() < 1.0, "variance {variance}");
}

#[test]
fn zero_weight_components_are_never_drawn_and_add_nothing() {
    let mut m = mixed(
        DistributionKind::Gaussian,
        &[f(v(&[0.0, 100.0])), f(s(1.0))],
        v(&[1.0, 0.0]),
        Some(&[1000]),
    );
    let x = m.state.value(m.node).unwrap().as_f64().unwrap().clone();
    assert!(x.as_slice().iter().all(|x| x.abs() < 10.0));
    let expected = Gaussian::new(s(0.0), s(1.0))
        .log_prob(&s(0.3))
        .unwrap()
        .sum();
    let mut single = mixed(
        DistributionKind::Gaussian,
        &[f(v(&[0.0, 100.0])), f(s(1.0))],
        v(&[1.0, 0.0]),
        None,
    );
    assert!((single.log_prob_at(f(s(0.3))) - expected).abs() < 1e-12);
    // the gradient with respect to the unused weight is still its density ratio, finite
    let gradient = node_log_prob_gradient(&single.graph, &single.state, single.node).unwrap();
    assert!(
        gradient.parameters[2]
            .as_ref()
            .unwrap()
            .as_slice()
            .iter()
            .all(|g| g.is_finite())
    );
    let _ = m.log_prob_at(f(DoubleTensor::full(&[1000], 0.0)));
}

#[test]
fn supports_hold_every_component() {
    let support = |component, parameters: &[Value], weights| {
        let m = mixed(component, parameters, weights, None);
        node_support(&m.graph, &m.state, m.node).unwrap()
    };
    let two = v(&[0.5, 0.5]);
    assert_eq!(
        support(
            DistributionKind::Gaussian,
            &[f(v(&[0.0, 1.0])), f(s(1.0))],
            two.clone()
        ),
        Support::Real
    );
    assert_eq!(
        support(
            DistributionKind::Gamma,
            &[f(v(&[1.0, 2.0])), f(s(2.0))],
            two.clone()
        ),
        Support::greater_than(0.0)
    );
    assert_eq!(
        support(
            DistributionKind::Uniform,
            &[f(v(&[0.0, 2.0])), f(v(&[1.0, 5.0]))],
            two.clone()
        ),
        Support::interval(0.0, 5.0)
    );
    assert_eq!(
        support(
            DistributionKind::Pareto,
            &[f(v(&[3.0, 1.0])), f(s(2.0))],
            two.clone()
        ),
        Support::greater_than(1.0)
    );
    // truncated components: one bounded below at 0, one at -1, neither above
    assert_eq!(
        support(
            DistributionKind::Truncated(Arc::new(DistributionKind::Gaussian)),
            &[f(s(0.0)), f(s(1.0)), f(v(&[0.0, -1.0])), f(s(INF))],
            two.clone()
        ),
        Support::greater_than(-1.0)
    );
    assert_eq!(
        support(
            DistributionKind::VonMises,
            &[f(v(&[0.0, 2.0])), f(s(3.0))],
            two.clone()
        ),
        Support::Circular
    );
    assert_eq!(
        support(DistributionKind::Poisson, &[f(v(&[1.0, 5.0]))], two),
        Support::Discrete
    );
}

#[test]
fn batch_shapes_broadcast_without_the_component_axis() {
    // two rows of two-component parameters
    let means = DoubleTensor::new(&[2, 2], vec![-1.0, 1.0, 10.0, 20.0]).unwrap();
    let mut m = mixed(
        DistributionKind::Gaussian,
        &[f(means), f(s(1.0))],
        v(&[0.5, 0.5]),
        None,
    );
    assert_eq!(m.graph[m.node].shape(), &[2]);
    let density = |x: f64, mu: f64| {
        Gaussian::new(s(mu), s(1.0))
            .log_prob(&s(x))
            .unwrap()
            .sum()
            .exp()
    };
    let expected = (0.5 * density(0.0, -1.0) + 0.5 * density(0.0, 1.0)).ln()
        + (0.5 * density(15.0, 10.0) + 0.5 * density(15.0, 20.0)).ln();
    let actual = m.log_prob_at(f(v(&[0.0, 15.0])));
    assert!((actual - expected).abs() < 1e-12, "{actual} vs {expected}");
    // samples follow each row's components
    let m = mixed(
        DistributionKind::Gaussian,
        &[
            f(DoubleTensor::new(&[2, 2], vec![-1.0, 1.0, 10.0, 20.0]).unwrap()),
            f(s(0.01)),
        ],
        v(&[0.5, 0.5]),
        Some(&[500, 2]),
    );
    let x = m.state.value(m.node).unwrap().as_f64().unwrap().clone();
    for row in x.as_slice().chunks(2) {
        assert!((row[0].abs() - 1.0).abs() < 0.1, "{row:?}");
        assert!(
            (row[1] - 10.0).abs() < 0.1 || (row[1] - 20.0).abs() < 0.1,
            "{row:?}"
        );
    }
}

#[test]
fn invalid_mixtures_are_errors() {
    let build_error = |build: &dyn Fn(&GraphBuilder)| {
        let b = GraphBuilder::new();
        build(&b);
        b.build().unwrap_err()
    };
    let invalid_node = |e: GraphError, what: &str| {
        assert!(
            matches!(&e, GraphError::InvalidNode { op: "mixture", message } if message.contains(what)),
            "{e}"
        );
    };
    // a component without an element-wise density
    invalid_node(
        build_error(&|b| {
            let c = b.constant(v(&[1.0, 1.0]));
            let _: DoubleVar = b.mixture(DistributionKind::Dirichlet, &[c.id()], v(&[0.5, 0.5]));
        }),
        "cannot be mixed",
    );
    invalid_node(
        build_error(&|b| {
            let (mu, sigma, w) = (b.scalar(0.0), b.scalar(1.0), b.constant(v(&[1.0])));
            let inner = DistributionKind::Mixture(Arc::new(DistributionKind::Gaussian));
            let _: DoubleVar = b.mixture(inner, &[mu.id(), sigma.id(), w.id()], v(&[1.0]));
        }),
        "cannot be mixed",
    );
    // a component parameter whose last axis doesn't match the weights
    invalid_node(
        build_error(&|b| {
            let (mu, sigma) = (b.constant(v(&[0.0, 1.0, 2.0])), b.scalar(1.0));
            let _: DoubleVar = b.mixture(
                DistributionKind::Gaussian,
                &[mu.id(), sigma.id()],
                v(&[0.5, 0.5]),
            );
        }),
        "component axis",
    );
    // weights without a component axis
    invalid_node(
        build_error(&|b| {
            let (mu, sigma) = (b.scalar(0.0), b.scalar(1.0));
            let _: DoubleVar = b.mixture(DistributionKind::Gaussian, &[mu.id(), sigma.id()], 1.0);
        }),
        "component axis",
    );
    // the wrong handle type
    invalid_node(
        build_error(&|b| {
            let mu = b.constant(v(&[1.0, 2.0]));
            let _: DoubleVar = b.mixture(DistributionKind::Poisson, &[mu.id()], v(&[0.5, 0.5]));
        }),
        "I64",
    );
    // the wrong number of component parameters
    let e = build_error(&|b| {
        let mu = b.scalar(0.0);
        let _: DoubleVar = b.mixture(DistributionKind::Gaussian, &[mu.id()], v(&[1.0]));
    });
    assert!(matches!(e, GraphError::InvalidNode { .. }), "{e}");
    // a mixture can't be truncated
    let e = build_error(&|b| {
        let (mu, sigma, w) = (b.scalar(0.0), b.scalar(1.0), b.constant(v(&[1.0])));
        let inner = DistributionKind::Mixture(Arc::new(DistributionKind::Gaussian));
        let _: DoubleVar = b.truncated(inner, &[mu.id(), sigma.id(), w.id()], 0.0, 1.0);
    });
    assert!(e.to_string().contains("no CDF"), "{e}");

    // weights off the simplex are an error when evaluated
    for weights in [v(&[0.5, 0.6]), v(&[1.2, -0.2]), v(&[f64::NAN, 1.0])] {
        let b = GraphBuilder::new();
        let (mu, sigma) = (b.constant(v(&[0.0, 1.0])), b.scalar(1.0));
        let w = b.constant(weights);
        let x: DoubleVar = b.mixture(DistributionKind::Gaussian, &[mu.id(), sigma.id()], w);
        let x = x.id();
        let (graph, mut state) = b.build().unwrap();
        assert!(
            state
                .lazy_eval(&graph, &[x], &mut KeanuRng::seed_from_u64(1))
                .is_err()
        );
        state.set(&graph, x, f(s(0.0))).unwrap();
        let e = node_log_prob(&graph, &state, x).unwrap_err();
        assert!(e.to_string().contains("weights"), "{e}");
    }
}

#[test]
fn parameter_names_list_the_component_then_the_weights() {
    let kind = DistributionKind::Mixture(Arc::new(DistributionKind::Gaussian));
    assert_eq!(
        kind.parameter_names().unwrap().as_ref(),
        &["mu", "sigma", "weights"]
    );
    let nested = DistributionKind::Mixture(Arc::new(DistributionKind::Truncated(Arc::new(
        DistributionKind::Poisson,
    ))));
    assert_eq!(
        nested.parameter_names().unwrap().as_ref(),
        &["mu", "lower", "upper", "weights"]
    );
}
