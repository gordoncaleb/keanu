//! Hidden Markov model and linear-Gaussian state-space nodes: brute-force agreement, reduction to i.i.d.
//! emissions, gradients against finite differences, sampled moments, supports, time-varying parameters and invalid
//! models.

use keanu_distributions::{Gaussian, KeanuRng, LinearGaussianStateSpace, Poisson};
use keanu_graph::op::{DistributionKind, Op};
use keanu_graph::{
    DoubleVar, Graph, GraphBuilder, GraphError, IntVar, NodeId, State, Support, Value,
    node_log_prob, node_log_prob_gradient, node_support,
};
use keanu_tensor::{DoubleTensor, IntTensor};
use std::sync::Arc;

fn s(v: f64) -> DoubleTensor {
    DoubleTensor::scalar(v)
}

fn v(values: &[f64]) -> DoubleTensor {
    DoubleTensor::vector(values.to_vec())
}

fn m(rows: usize, cols: usize, values: &[f64]) -> DoubleTensor {
    DoubleTensor::new(&[rows, cols], values.to_vec()).unwrap()
}

fn f(t: DoubleTensor) -> Value {
    Value::F64(t)
}

const INITIAL: [f64; 3] = [0.5, 0.3, 0.2];
const TRANSITION: [f64; 9] = [0.7, 0.2, 0.1, 0.25, 0.5, 0.25, 0.1, 0.3, 0.6];

/// A graph with a hidden Markov model node whose parameters are constants.
struct Hmm {
    graph: Graph,
    state: State,
    node: NodeId,
    /// The emission parameters, then the initial and transition probabilities.
    parameters: Vec<NodeId>,
}

fn hmm(
    emission: DistributionKind,
    parameters: &[Value],
    initial: DoubleTensor,
    transition: DoubleTensor,
    shape: &[usize],
) -> Hmm {
    let b = GraphBuilder::new();
    let mut ids: Vec<NodeId> = parameters
        .iter()
        .map(|p| b.add_node(Op::Constant(p.clone()), &[]).unwrap())
        .collect();
    let (initial, transition) = (b.constant(initial), b.constant(transition));
    let node = if emission.value_type() == keanu_graph::DType::F64 {
        b.hidden_markov_model_with_shape::<DoubleVar>(shape, emission, &ids, initial, transition)
            .id()
    } else {
        b.hidden_markov_model_with_shape::<IntVar>(shape, emission, &ids, initial, transition)
            .id()
    };
    ids.extend([initial.id(), transition.id()]);
    let (graph, mut state) = b.build().unwrap();
    state
        .lazy_eval(&graph, &[node], &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    Hmm {
        graph,
        state,
        node,
        parameters: ids,
    }
}

impl Hmm {
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

/// A gradient part summed over the leading copies of a parameter of `shape` it was broadcast along.
fn summed_to(part: &DoubleTensor, shape: &[usize]) -> Vec<f64> {
    let len: usize = shape.iter().product();
    if part.shape() == shape {
        return part.as_slice().to_vec();
    }
    if len == 1 {
        return vec![part.sum()];
    }
    let mut out = vec![0.0; len];
    for (i, x) in part.as_slice().iter().enumerate() {
        out[i % len] += x;
    }
    out
}

fn close(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() <= tolerance * (1.0 + b.abs())
}

#[test]
fn the_hmm_log_likelihood_sums_every_state_path() {
    let (mu, sigma) = ([-1.0, 0.5, 2.0], [0.7, 1.0, 0.4]);
    let x = [-0.8, 1.9, 0.3, 2.2];
    let mut h = hmm(
        DistributionKind::Gaussian,
        &[f(v(&mu)), f(v(&sigma))],
        v(&INITIAL),
        m(3, 3, &TRANSITION),
        &[4],
    );
    let emission = |t: usize, j: usize| {
        Gaussian::new(s(mu[j]), s(sigma[j]))
            .log_prob(&s(x[t]))
            .unwrap()
            .sum()
            .exp()
    };
    let mut total = 0.0;
    for code in 0..81usize {
        let path: Vec<usize> = (0..4).map(|t| code / 3usize.pow(t as u32) % 3).collect();
        let mut p = INITIAL[path[0]] * emission(0, path[0]);
        for t in 1..4 {
            p *= TRANSITION[path[t - 1] * 3 + path[t]] * emission(t, path[t]);
        }
        total += p;
    }
    let log_prob = h.log_prob_at(f(v(&x)));
    assert!(
        close(log_prob, total.ln(), 1e-12),
        "{log_prob} vs {}",
        total.ln()
    );
}

#[test]
fn a_one_state_hmm_is_iid_emissions() {
    let counts = IntTensor::new(&[5, 2], vec![0, 3, 2, 1, 4, 4, 1, 0, 2, 5]).unwrap();
    let mut h = hmm(
        DistributionKind::Poisson,
        &[f(v(&[2.5]))],
        v(&[1.0]),
        m(1, 1, &[1.0]),
        &[5, 2],
    );
    let expected = Poisson::new(s(2.5)).log_prob(&counts).unwrap().sum();
    let actual = h.log_prob_at(Value::I64(counts));
    assert!(close(actual, expected, 1e-12), "{actual} vs {expected}");
    // and its gradient is the emission's
    let gradient = node_log_prob_gradient(&h.graph, &h.state, h.node).unwrap();
    assert!(gradient.value.is_none());
    let rate = summed_to(gradient.parameters[0].as_ref().unwrap(), &[1]);
    let d = Poisson::new(s(2.5))
        .d_log_prob(&IntTensor::new(&[5, 2], vec![0, 3, 2, 1, 4, 4, 1, 0, 2, 5]).unwrap())
        .unwrap();
    assert!(close(rate[0], d.mu.sum(), 1e-12));
}

#[test]
fn hmm_gradients_match_finite_differences() {
    // per-state means, a shared scale and time-varying offsets: mu has a time axis as well as the state axis
    let x = v(&[-0.8, 1.9, 0.3, 2.2, -1.5]);
    let mu = m(
        5,
        3,
        &[
            -1.0, 0.5, 2.0, -0.9, 0.4, 2.1, -1.2, 0.6, 1.8, -1.0, 0.5, 2.3, -0.7, 0.2, 2.0,
        ],
    );
    let mut h = hmm(
        DistributionKind::Gaussian,
        &[f(mu), f(s(0.8))],
        v(&INITIAL),
        m(3, 3, &TRANSITION),
        &[5],
    );
    let _ = h.log_prob_at(f(x.clone()));
    let gradient = node_log_prob_gradient(&h.graph, &h.state, h.node).unwrap();
    let step = 1e-6;
    for (i, part) in gradient.parameters.iter().enumerate() {
        let base = h.parameter(i);
        let analytic = summed_to(part.as_ref().unwrap(), base.shape());
        for (k, &a) in analytic.iter().enumerate() {
            let at = |delta: f64| {
                let mut values = base.as_slice().to_vec();
                values[k] += delta;
                h.log_prob_with(i, DoubleTensor::new(base.shape(), values).unwrap())
            };
            let numeric = (at(step) - at(-step)) / (2.0 * step);
            assert!(
                close(a, numeric, 1e-5),
                "parameter {i}[{k}]: {a} vs {numeric}"
            );
        }
    }
    let dx = gradient.value.unwrap();
    for k in 0..5 {
        let at = |delta: f64| {
            let mut values = x.as_slice().to_vec();
            values[k] += delta;
            let mut state = h.state.clone();
            state.set(&h.graph, h.node, f(v(&values))).unwrap();
            node_log_prob(&h.graph, &state, h.node).unwrap()
        };
        let numeric = (at(step) - at(-step)) / (2.0 * step);
        assert!(close(dx.as_slice()[k], numeric, 1e-5), "x[{k}]");
    }
}

#[test]
fn hmm_samples_reproduce_the_stationary_moments() {
    // stationary distribution (0.75, 0.25), second eigenvalue 0.6
    let n = 40_000;
    let h = hmm(
        DistributionKind::Gaussian,
        &[f(v(&[-1.0, 2.0])), f(s(0.5))],
        v(&[0.75, 0.25]),
        m(2, 2, &[0.9, 0.1, 0.3, 0.7]),
        &[n],
    );
    let x = h.state.value(h.node).unwrap().as_f64().unwrap().clone();
    let x = x.as_slice();
    let mean = x.iter().sum::<f64>() / n as f64;
    let variance = x.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let lag1 = x
        .windows(2)
        .map(|w| (w[0] - mean) * (w[1] - mean))
        .sum::<f64>()
        / n as f64;
    // mean Σ π μ = -0.25; variance σ² + Var(μ) = 0.25 + 0.1875 · 9 = 1.9375; lag-1 covariance
    // 0.6 · Var(μ) = 1.0125. The state path is autocorrelated, inflating the mean's variance by about
    // (1 + 0.6) / (1 - 0.6) = 4: its standard error is about sqrt(4 · 1.9 / 40000) ≈ 0.014
    assert!((mean + 0.25).abs() < 0.06, "mean {mean}");
    assert!((variance - 1.9375).abs() < 0.1, "variance {variance}");
    assert!((lag1 - 1.0125).abs() < 0.1, "lag-1 covariance {lag1}");
}

#[test]
fn several_observations_per_step_share_the_state() {
    // two observations per step from the same state: with distinct, well separated means they always agree on it
    let h = hmm(
        DistributionKind::Gaussian,
        &[f(v(&[-10.0, 10.0])), f(s(0.1))],
        v(&[0.5, 0.5]),
        m(2, 2, &[0.5, 0.5, 0.5, 0.5]),
        &[200, 2],
    );
    let x = h.state.value(h.node).unwrap().as_f64().unwrap().clone();
    for pair in x.as_slice().chunks(2) {
        assert_eq!(pair[0] > 0.0, pair[1] > 0.0, "{pair:?}");
    }
}

#[test]
fn hmm_supports_hold_every_state() {
    let support = |emission, parameters: &[Value]| {
        let h = hmm(
            emission,
            parameters,
            v(&[0.5, 0.5]),
            m(2, 2, &[0.5; 4]),
            &[3],
        );
        node_support(&h.graph, &h.state, h.node).unwrap()
    };
    assert_eq!(
        support(DistributionKind::Gaussian, &[f(v(&[0.0, 1.0])), f(s(1.0))]),
        Support::Real
    );
    assert_eq!(
        support(DistributionKind::Exponential, &[f(v(&[1.0, 2.0]))]),
        Support::greater_than(0.0)
    );
    assert_eq!(
        support(
            DistributionKind::Uniform,
            &[f(v(&[0.0, -1.0])), f(v(&[1.0, 3.0]))]
        ),
        Support::interval(-1.0, 3.0)
    );
    assert_eq!(
        support(DistributionKind::Poisson, &[f(v(&[1.0, 2.0]))]),
        Support::Discrete
    );
}

#[test]
fn invalid_hidden_markov_models_are_errors() {
    let build_error = |build: &dyn Fn(&GraphBuilder)| {
        let b = GraphBuilder::new();
        build(&b);
        b.build().unwrap_err()
    };
    let invalid_node = |e: GraphError, what: &str| {
        assert!(
            matches!(&e, GraphError::InvalidNode { op: "hidden_markov_model", message } if message.contains(what)),
            "{e}"
        );
    };
    let gaussian = |b: &GraphBuilder, mu: DoubleTensor| [b.constant(mu).id(), b.scalar(1.0).id()];
    // an emission without an element-wise density
    invalid_node(
        build_error(&|b| {
            let c = b.constant(v(&[1.0, 1.0]));
            let _: DoubleVar = b.hidden_markov_model(
                3,
                DistributionKind::Dirichlet,
                &[c.id()],
                v(&[0.5, 0.5]),
                m(2, 2, &[0.5; 4]),
            );
        }),
        "emission",
    );
    // an emission parameter whose state axis doesn't match
    invalid_node(
        build_error(&|b| {
            let _: DoubleVar = b.hidden_markov_model(
                3,
                DistributionKind::Gaussian,
                &gaussian(b, v(&[0.0, 1.0, 2.0])),
                v(&[0.5, 0.5]),
                m(2, 2, &[0.5; 4]),
            );
        }),
        "state axis",
    );
    // a non-square or mismatched transition matrix
    for transition in [m(2, 3, &[0.5; 6]), m(3, 3, &[1.0 / 3.0; 9]), v(&[0.5, 0.5])] {
        invalid_node(
            build_error(&|b| {
                let _: DoubleVar = b.hidden_markov_model(
                    3,
                    DistributionKind::Gaussian,
                    &gaussian(b, v(&[0.0, 1.0])),
                    v(&[0.5, 0.5]),
                    transition.clone(),
                );
            }),
            "transition",
        );
    }
    // initial probabilities that aren't a vector
    invalid_node(
        build_error(&|b| {
            let _: DoubleVar = b.hidden_markov_model(
                3,
                DistributionKind::Gaussian,
                &gaussian(b, v(&[0.0, 1.0])),
                m(1, 2, &[0.5, 0.5]),
                m(2, 2, &[0.5; 4]),
            );
        }),
        "initial",
    );
    // no time steps, or emission parameters with more batch axes than the observations
    invalid_node(
        build_error(&|b| {
            let _: DoubleVar = b.hidden_markov_model(
                0,
                DistributionKind::Gaussian,
                &gaussian(b, v(&[0.0, 1.0])),
                v(&[0.5, 0.5]),
                m(2, 2, &[0.5; 4]),
            );
        }),
        "T > 0",
    );
    invalid_node(
        build_error(&|b| {
            let _: DoubleVar = b.hidden_markov_model(
                3,
                DistributionKind::Gaussian,
                &gaussian(b, m(4, 2, &[0.0; 8])),
                v(&[0.5, 0.5]),
                m(2, 2, &[0.5; 4]),
            );
        }),
        "does not broadcast",
    );
    // the wrong handle type
    invalid_node(
        build_error(&|b| {
            let rate = b.constant(v(&[1.0, 2.0]));
            let _: DoubleVar = b.hidden_markov_model(
                3,
                DistributionKind::Poisson,
                &[rate.id()],
                v(&[0.5, 0.5]),
                m(2, 2, &[0.5; 4]),
            );
        }),
        "I64",
    );
    // probabilities off the simplex are an error when evaluated, not a panic
    for (initial, transition) in [
        (v(&[0.5, 0.6]), m(2, 2, &[0.5; 4])),
        (v(&[0.5, 0.5]), m(2, 2, &[0.5, 0.5, 0.9, 0.3])),
        (v(&[f64::NAN, 1.0]), m(2, 2, &[0.5; 4])),
    ] {
        let b = GraphBuilder::new();
        let ids = gaussian(&b, v(&[0.0, 1.0]));
        let (i, t) = (b.constant(initial), b.constant(transition));
        let x: DoubleVar = b.hidden_markov_model(3, DistributionKind::Gaussian, &ids, i, t);
        let x = x.id();
        let (graph, mut state) = b.build().unwrap();
        assert!(
            state
                .lazy_eval(&graph, &[x], &mut KeanuRng::seed_from_u64(1))
                .is_err()
        );
        state.set(&graph, x, f(v(&[0.0, 0.1, 0.2]))).unwrap();
        assert!(node_log_prob(&graph, &state, x).is_err());
        assert!(node_log_prob_gradient(&graph, &state, x).is_err());
    }
}

#[test]
fn a_nan_emission_term_makes_the_hmm_log_prob_nan() {
    // the observation's log density is NaN under every state: the forward pass must not drop the NaN terms (as a
    // log-sum-exp whose max skips NaN does, returning -inf), and agrees with HiddenMarkovChain
    let initial = v(&[0.5, 0.5]);
    let transition = m(2, 2, &[0.9, 0.1, 0.2, 0.8]);
    let mut h = hmm(
        DistributionKind::Gaussian,
        &[f(v(&[-1.0, 1.0])), f(s(1.0))],
        initial.clone(),
        transition.clone(),
        &[2],
    );
    let log_prob = h.log_prob_at(f(v(&[0.3, f64::NAN])));
    assert!(log_prob.is_nan(), "{log_prob}");
    let emissions = DoubleTensor::new(&[2, 2], vec![-1.0, -2.0, f64::NAN, f64::NAN]).unwrap();
    let chain = keanu_distributions::HiddenMarkovChain::new(&initial, &transition);
    assert!(chain.log_likelihood(&emissions).unwrap().is_nan());
}

#[test]
fn hmm_parameter_names_list_the_emission_then_the_chain() {
    let kind = DistributionKind::HiddenMarkovModel(Arc::new(DistributionKind::Gaussian));
    assert_eq!(
        kind.parameter_names().unwrap().as_ref(),
        &["mu", "sigma", "initial", "transition"]
    );
    assert_eq!(
        DistributionKind::LinearGaussianStateSpace
            .parameter_names()
            .unwrap()
            .as_ref(),
        &[
            "initial_mean",
            "initial_covariance",
            "transition_matrix",
            "transition_covariance",
            "observation_matrix",
            "observation_covariance"
        ]
    );
}

/// The parameters of a two-dimensional state observed in one dimension.
fn state_space_parameters() -> [DoubleTensor; 6] {
    [
        v(&[0.0, 1.0]),
        m(2, 2, &[1.0, 0.3, 0.3, 2.0]),
        m(2, 2, &[0.9, 0.2, 0.0, 0.7]),
        m(2, 2, &[0.2, 0.0, 0.0, 0.1]),
        m(1, 2, &[1.0, 0.5]),
        m(1, 1, &[0.3]),
    ]
}

#[test]
fn a_state_space_node_has_the_kalman_log_likelihood_and_gradient() {
    let p = state_space_parameters();
    let b = GraphBuilder::new();
    let ids: Vec<DoubleVar> = p.iter().map(|t| b.constant(t.clone())).collect();
    let y = b.linear_gaussian_state_space(6, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5]);
    assert_eq!(y.shape(), vec![6, 1]);
    let y = y.id();
    let (graph, mut state) = b.build().unwrap();
    // a draw has the observations' shape
    state
        .lazy_eval(&graph, &[y], &mut KeanuRng::seed_from_u64(2))
        .unwrap();
    assert_eq!(state.value(y).unwrap().shape(), &[6, 1]);
    let observations = m(6, 1, &[0.4, 1.3, 0.2, -0.6, 0.9, 1.1]);
    state.set(&graph, y, f(observations.clone())).unwrap();
    let [a, b2, c, d, e, g] = p.clone();
    let model = LinearGaussianStateSpace::new(a, b2, c, d, e, g);
    let expected = model.log_prob(&observations).unwrap();
    assert_eq!(node_log_prob(&graph, &state, y).unwrap(), expected);
    let gradient = node_log_prob_gradient(&graph, &state, y).unwrap();
    let d = model.d_log_prob(&observations).unwrap();
    assert_eq!(gradient.value.as_ref(), Some(&d.x));
    let parts: Vec<&DoubleTensor> = gradient
        .parameters
        .iter()
        .map(|g| g.as_ref().unwrap())
        .collect();
    assert_eq!(parts[1], &d.initial_covariance);
    assert_eq!(parts[4], &d.observation_matrix);
    assert_eq!(node_support(&graph, &state, y).unwrap(), Support::Real);
}

#[test]
fn invalid_state_space_models_are_errors() {
    let p = state_space_parameters();
    let build = |index: usize, t: DoubleTensor, steps: usize| {
        let b = GraphBuilder::new();
        let mut p = p.clone();
        p[index] = t;
        let ids: Vec<DoubleVar> = p.iter().map(|t| b.constant(t.clone())).collect();
        let _ =
            b.linear_gaussian_state_space(steps, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5]);
        b.build().map(|_| ())
    };
    assert!(build(0, v(&[0.0, 1.0]), 3).is_ok());
    // non-square or mismatched matrices
    for (index, t) in [
        (0, m(2, 1, &[0.0, 1.0])),
        (1, m(2, 3, &[0.0; 6])),
        (2, m(3, 3, &[0.0; 9])),
        (3, v(&[0.1, 0.1])),
        (4, m(1, 3, &[1.0; 3])),
        (5, m(2, 2, &[1.0; 4])),
    ] {
        let e = build(index, t, 3).unwrap_err();
        assert!(
            matches!(
                &e,
                GraphError::InvalidNode {
                    op: "linear_gaussian_state_space",
                    ..
                }
            ),
            "{index}: {e}"
        );
    }
    // a saved shape with the wrong observation dimension
    let b = GraphBuilder::new();
    let ids: Vec<NodeId> = p.iter().map(|t| b.constant(t.clone()).id()).collect();
    assert!(
        b.add_distribution(
            DistributionKind::LinearGaussianStateSpace,
            Some(&[4, 2]),
            &ids
        )
        .is_err()
    );
    assert!(
        b.add_distribution(DistributionKind::LinearGaussianStateSpace, None, &ids)
            .is_err()
    );
    assert!(
        b.add_distribution(
            DistributionKind::LinearGaussianStateSpace,
            Some(&[4, 1]),
            &ids
        )
        .is_ok()
    );
}
