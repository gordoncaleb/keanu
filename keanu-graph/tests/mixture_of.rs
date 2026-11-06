//! Heterogeneous mixtures and transformed distributions: agreement with hand-written densities (values,
//! gradients and Hessians), equality with the distributions they reproduce, sampling, supports, the log-density
//! op, pointwise terms and invalid nodes.

use keanu_distributions::{KeanuRng, LogNormal};
use keanu_graph::op::{DistributionKind as K, ElementBijector, Op};
use keanu_graph::{
    DoubleVar, GraphBuilder, GraphError, IntVar, Model, NodeId, Parameterisation, Space, State,
    Support, Value, node_log_prob, node_log_prob_gradient, node_log_prob_terms, node_support,
};
use keanu_tensor::special::ln_gamma;
use keanu_tensor::{DoubleTensor, IntTensor};
use std::f64::consts::PI;
use std::sync::Arc;

mod common;
use common::{assert_all_close, assert_close, s, v};

fn gamma_log_pdf(x: f64, theta: f64, k: f64) -> f64 {
    (k - 1.0) * x.ln() - x / theta - k * theta.ln() - ln_gamma(k)
}

fn log_normal_log_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    let z = (x.ln() - mu) / sigma;
    -x.ln() - sigma.ln() - 0.5 * (2.0 * PI).ln() - 0.5 * z * z
}

fn log_sum_exp(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// The hand-written log density of a mixture of a gamma (θ, k) and a log-normal (μ, σ) with weights (w, 1 − w).
fn hand_mixture(x: f64, p: &[f64; 5]) -> f64 {
    let [theta, k, mu, sigma, w] = *p;
    log_sum_exp(
        w.ln() + gamma_log_pdf(x, theta, k),
        (1.0 - w).ln() + log_normal_log_pdf(x, mu, sigma),
    )
}

const DATA: [f64; 4] = [0.3, 1.1, 2.4, 6.0];

/// A node of a gamma and log-normal mixture with constant parameters, observed at [`DATA`].
fn constant_mixture(p: &[f64; 5]) -> (keanu_graph::Graph, State, NodeId, Vec<NodeId>) {
    let b = GraphBuilder::new();
    let ids: Vec<NodeId> = p[..4].iter().map(|&x| b.scalar(x).id()).collect();
    let weights = b.constant(v(&[p[4], 1.0 - p[4]]));
    let components = [(K::Gamma, &ids[..2]), (K::LogNormal, &ids[2..])];
    let y: DoubleVar = b.mixture_of_with_shape(&[4], &components, weights);
    b.observe(y, v(&DATA));
    let node = y.id();
    let mut parameters = vec![weights.id()];
    parameters.extend(ids);
    let (graph, state) = b.build().unwrap();
    (graph, state, node, parameters)
}

#[test]
fn a_gamma_and_log_normal_mixture_matches_the_hand_written_logsumexp() {
    let p = [1.3, 2.2, 0.4, 0.7, 0.35];
    let (graph, state, node, _) = constant_mixture(&p);
    let expected: Vec<f64> = DATA.iter().map(|&x| hand_mixture(x, &p)).collect();
    let terms = node_log_prob_terms(&graph, &state, node).unwrap();
    assert_eq!(terms.shape(), &[4]);
    assert_all_close(terms.as_slice(), &expected, 1e-12, "terms");
    let total = node_log_prob(&graph, &state, node).unwrap();
    assert_close(total, expected.iter().sum(), 1e-12, "log prob");

    // the gradient with respect to the value, each parameter and the weights, against central differences
    let gradient = node_log_prob_gradient(&graph, &state, node).unwrap();
    let h = 1e-6;
    let dx = gradient.value.unwrap();
    for (i, &x) in DATA.iter().enumerate() {
        let fd = (hand_mixture(x + h, &p) - hand_mixture(x - h, &p)) / (2.0 * h);
        assert_close(dx.as_slice()[i], fd, 1e-6, "d/dx");
    }
    let total_with = |j: usize, delta: f64| -> f64 {
        let mut q = p;
        q[j] += delta;
        DATA.iter().map(|&x| hand_mixture(x, &q)).sum()
    };
    // parents: the weights, then θ, k, μ, σ
    for (j, part) in gradient.parameters[1..].iter().enumerate() {
        let fd = (total_with(j, h) - total_with(j, -h)) / (2.0 * h);
        assert_close(part.as_ref().unwrap().sum(), fd, 1e-6, "d/dparameter");
    }
    // with respect to each weight separately: p_c(x) / p(x)
    let dw = gradient.parameters[0].as_ref().unwrap();
    assert_eq!(dw.shape(), &[4, 2]);
    for (i, &x) in DATA.iter().enumerate() {
        let total = hand_mixture(x, &p);
        let gamma = (gamma_log_pdf(x, p[0], p[1]) - total).exp();
        let log_normal = (log_normal_log_pdf(x, p[2], p[3]) - total).exp();
        assert_close(dw.as_slice()[2 * i], gamma, 1e-12, "d/dw1");
        assert_close(dw.as_slice()[2 * i + 1], log_normal, 1e-12, "d/dw2");
    }
}

#[test]
fn a_gamma_and_log_normal_mixtures_hessian_matches_the_hand_written_one() {
    // latents with standard normal priors: θ, k, μ, σ, the weight w (constrained space), and a latent mixture value
    let p = [1.3, 2.2, 0.4, 0.7, 0.35];
    let x0 = 1.7;
    let b = GraphBuilder::new();
    let latent = |value: f64| {
        let z = b.gaussian(0.0, 1.0);
        b.set_value(z, DoubleTensor::scalar(value));
        z
    };
    let latents: Vec<DoubleVar> = p.iter().map(|&x| latent(x)).collect();
    let w = latents[4];
    let weights = b.stack(-1, &[w, -w + 1.0]);
    let ids: Vec<NodeId> = latents[..4].iter().map(|l| l.id()).collect();
    let components = [(K::Gamma, &ids[..2]), (K::LogNormal, &ids[2..])];
    let x: DoubleVar = b.mixture_of(&components, weights);
    b.set_value(x, DoubleTensor::scalar(x0));
    let y: DoubleVar = b.mixture_of_with_shape(&[4], &components, weights);
    b.observe(y, v(&DATA));
    let latent_ids: Vec<NodeId> = latents.iter().map(|l| l.id()).chain([x.id()]).collect();
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let param = Parameterisation::with_space(&model, &state, Space::Constrained).unwrap();
    // each latent's coordinate
    let order: Vec<usize> = latent_ids
        .iter()
        .map(|&id| param.latents().iter().position(|&l| l == id).unwrap())
        .collect();
    let z = param.from_state(&state).unwrap();
    // the hand-written joint log density of (θ, k, μ, σ, w, x)
    let joint = |u: &[f64]| -> f64 {
        let q = [u[0], u[1], u[2], u[3], u[4]];
        let prior: f64 = q.iter().map(|t| -0.5 * t * t).sum();
        prior + hand_mixture(u[5], &q) + DATA.iter().map(|&d| hand_mixture(d, &q)).sum::<f64>()
    };
    let at = |z: &[f64]| -> Vec<f64> { order.iter().map(|&i| z[i]).collect() };
    let hessian = param.hessian(&z, false, &mut state).unwrap();
    let n = z.len();
    let h = 1e-4;
    let u0 = at(&z);
    for a in 0..6 {
        for c in 0..6 {
            let shifted = |da: f64, dc: f64| {
                let mut u = u0.clone();
                u[a] += da;
                u[c] += dc;
                joint(&u)
            };
            let fd =
                (shifted(h, h) - shifted(h, -h) - shifted(-h, h) + shifted(-h, -h)) / (4.0 * h * h);
            let exact = hessian.as_slice()[order[a] * n + order[c]];
            assert!(
                (exact - fd).abs() <= 1e-4 * fd.abs().max(1.0),
                "H[{a}][{c}] = {exact}, the hand-written density's second differences give {fd}"
            );
        }
    }
}

#[test]
fn a_transformed_gaussian_with_a_lower_bound_is_a_shifted_log_normal() {
    let (mu, sigma, shift) = (0.3, 0.6, 1.5);
    let b = GraphBuilder::new();
    let (m, sd) = (b.scalar(mu), b.scalar(sigma));
    let x = b.transformed_with_shape(
        &[4],
        K::Gaussian,
        &[m.id(), sd.id()],
        ElementBijector::LowerBound(shift),
    );
    let values = [1.6, 2.0, 3.3, 9.0];
    b.observe(x, v(&values));
    let x = x.id();
    let (graph, state) = b.build().unwrap();
    let shifted: Vec<f64> = values.iter().map(|x| x - shift).collect();
    let (mu_t, sigma_t) = (s(mu), s(sigma));
    let log_normal = LogNormal::new(&mu_t, &sigma_t);
    let expected = log_normal.log_prob(&v(&shifted)).unwrap();
    let terms = node_log_prob_terms(&graph, &state, x).unwrap();
    assert_all_close(terms.as_slice(), expected.as_slice(), 1e-12, "terms");
    let g = log_normal.d_log_prob(&v(&shifted)).unwrap();
    let gradient = node_log_prob_gradient(&graph, &state, x).unwrap();
    assert_all_close(
        gradient.value.unwrap().as_slice(),
        g.x.as_slice(),
        1e-12,
        "d/dx",
    );
    assert_close(
        gradient.parameters[0].as_ref().unwrap().sum(),
        g.mu.sum(),
        1e-12,
        "d/dmu",
    );
    assert_close(
        gradient.parameters[1].as_ref().unwrap().sum(),
        g.sigma.sum(),
        1e-12,
        "d/dsigma",
    );
    assert_eq!(
        node_support(&graph, &state, x).unwrap(),
        Support::greater_than(shift)
    );
    // values at or below the shift are impossible
    let b = GraphBuilder::new();
    let (m, sd) = (b.scalar(mu), b.scalar(sigma));
    let x = b.transformed_with_shape(
        &[4],
        K::Gaussian,
        &[m.id(), sd.id()],
        ElementBijector::LowerBound(shift),
    );
    b.observe(x, v(&[1.5, 2.0, 0.0, 3.0]));
    let x = x.id();
    let (graph, state) = b.build().unwrap();
    let terms = node_log_prob_terms(&graph, &state, x).unwrap();
    assert_eq!(terms.as_slice()[0], f64::NEG_INFINITY);
    assert_eq!(terms.as_slice()[2], f64::NEG_INFINITY);
    assert!(terms.as_slice()[1].is_finite());
}

/// The draws of a node from its prior.
fn draws(build: impl FnOnce(&GraphBuilder) -> NodeId) -> Vec<f64> {
    let b = GraphBuilder::new();
    let id = build(&b);
    let (graph, mut state) = b.build().unwrap();
    let mut rng = KeanuRng::seed_from_u64(7);
    state.lazy_eval(&graph, &[id], &mut rng).unwrap();
    state
        .value(id)
        .unwrap()
        .as_f64()
        .unwrap()
        .as_slice()
        .to_vec()
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

#[test]
fn transformed_draws_are_the_forward_map_of_base_draws() {
    let n = 40_000;
    // exp of N(0.3, 0.5) plus 1: mean 1 + exp(0.3 + 0.125)
    let x = draws(|b| {
        b.transformed_with_shape(
            &[n],
            K::Gaussian,
            &[b.scalar(0.3).id(), b.scalar(0.5).id()],
            ElementBijector::LowerBound(1.0),
        )
        .id()
    });
    assert!(x.iter().all(|&x| x > 1.0));
    assert_close(mean(&x), 1.0 + (0.425f64).exp(), 0.02, "lower bound mean");
    // an affine map of a gamma: 2 − 3y has mean 2 − 3θk
    let x = draws(|b| {
        b.transformed_with_shape(
            &[n],
            K::Gamma,
            &[b.scalar(0.5).id(), b.scalar(2.0).id()],
            ElementBijector::Affine {
                loc: 2.0,
                scale: -3.0,
            },
        )
        .id()
    });
    assert!(x.iter().all(|&x| x < 2.0));
    assert_close(mean(&x), -1.0, 0.03, "affine mean");
    // an interval: logit-normal draws in (−1, 3)
    let x = draws(|b| {
        b.transformed_with_shape(
            &[n],
            K::Gaussian,
            &[b.scalar(0.0).id(), b.scalar(1.0).id()],
            ElementBijector::Interval(-1.0, 3.0),
        )
        .id()
    });
    assert!(x.iter().all(|&x| x > -1.0 && x < 3.0));
    assert_close(mean(&x), 1.0, 0.03, "interval mean (symmetric)");
}

#[test]
fn transformed_supports_are_the_images_of_the_base_supports() {
    let b = GraphBuilder::new();
    let (mu, sigma) = (b.scalar(0.0).id(), b.scalar(1.0).id());
    let (theta, k) = (b.scalar(1.0).id(), b.scalar(2.0).id());
    let upper = b.transformed(K::Gaussian, &[mu, sigma], ElementBijector::UpperBound(4.0));
    let interval = b.transformed(
        K::Gaussian,
        &[mu, sigma],
        ElementBijector::Interval(1.0, 2.0),
    );
    let affine = b.transformed(
        K::Gamma,
        &[theta, k],
        ElementBijector::Affine {
            loc: 1.0,
            scale: -2.0,
        },
    );
    let positive = b.transformed(K::Gamma, &[theta, k], ElementBijector::LowerBound(-1.0));
    let [upper, interval, affine, positive] = [upper, interval, affine, positive].map(|x| x.id());
    let (graph, mut state) = b.build().unwrap();
    state
        .lazy_eval(
            &graph,
            &[upper, interval, affine, positive],
            &mut KeanuRng::seed_from_u64(1),
        )
        .unwrap();
    let support = |id: NodeId| node_support(&graph, &state, id).unwrap();
    assert_eq!(support(upper), Support::less_than(4.0));
    assert_eq!(support(interval), Support::interval(1.0, 2.0));
    assert_eq!(support(affine), Support::less_than(1.0));
    // the gamma's (0, ∞) maps to (−1 + exp(0), ∞) = (0, ∞)
    assert_eq!(support(positive), Support::greater_than(0.0));
}

#[test]
fn mixture_draws_follow_the_weights_and_the_support_is_the_hull() {
    let n = 40_000;
    let b = GraphBuilder::new();
    let gamma = [b.scalar(0.5).id(), b.scalar(2.0).id()];
    let uniform = [b.scalar(-2.0).id(), b.scalar(-1.0).id()];
    let weights = b.constant(v(&[0.25, 0.75]));
    let components = [(K::Gamma, &gamma[..]), (K::Uniform, &uniform[..])];
    let y = b
        .mixture_of_with_shape::<DoubleVar>(&[n], &components, weights)
        .id();
    let (graph, mut state) = b.build().unwrap();
    state
        .lazy_eval(&graph, &[y], &mut KeanuRng::seed_from_u64(3))
        .unwrap();
    let x = state
        .value(y)
        .unwrap()
        .as_f64()
        .unwrap()
        .as_slice()
        .to_vec();
    let positive = x.iter().filter(|&&x| x > 0.0).count() as f64 / n as f64;
    assert_close(positive, 0.25, 0.01, "the gamma's share");
    // 0.25 · 1 + 0.75 · (−1.5)
    assert_close(mean(&x), 0.25 - 1.125, 0.02, "mean");
    assert_eq!(
        node_support(&graph, &state, y).unwrap(),
        Support::greater_than(-2.0)
    );
    // between the components the density is zero
    state
        .set(&graph, y, Value::F64(DoubleTensor::full(&[n], -0.5)))
        .unwrap();
    assert_eq!(node_log_prob(&graph, &state, y).unwrap(), f64::NEG_INFINITY);
}

#[test]
fn a_mixture_of_counts_matches_its_components() {
    let b = GraphBuilder::new();
    let rate = [b.scalar(2.0).id()];
    let negative_binomial = [b.scalar(6.0).id(), b.scalar(0.5).id()];
    let weights = b.constant(v(&[0.6, 0.4]));
    let components = [
        (K::Poisson, &rate[..]),
        (K::NegativeBinomial, &negative_binomial[..]),
    ];
    let y: IntVar = b.mixture_of_with_shape(&[3], &components, weights);
    let poisson: IntVar = b.poisson_with_shape(&[3], 2.0);
    let nb: IntVar = b.negative_binomial_with_shape(&[3], 6.0, 0.5);
    let counts = IntTensor::vector(vec![0, 3, 11]);
    for node in [y, poisson, nb] {
        b.observe(node, counts.clone());
    }
    let [y, poisson, nb] = [y, poisson, nb].map(|x| x.id());
    let (graph, state) = b.build().unwrap();
    let terms = |id: NodeId| node_log_prob_terms(&graph, &state, id).unwrap();
    let (p, q) = (terms(poisson), terms(nb));
    let expected: Vec<f64> = p
        .as_slice()
        .iter()
        .zip(q.as_slice())
        .map(|(p, q)| log_sum_exp(0.6f64.ln() + p, 0.4f64.ln() + q))
        .collect();
    assert_all_close(terms(y).as_slice(), &expected, 1e-12, "terms");
    assert_eq!(node_support(&graph, &state, y).unwrap(), Support::Discrete);
}

#[test]
fn a_heterogeneous_mixtures_log_density_is_its_terms() {
    let p = [1.3, 2.2, 0.4, 0.7, 0.35];
    let b = GraphBuilder::new();
    let ids: Vec<NodeId> = p[..4].iter().map(|&x| b.scalar(x).id()).collect();
    let weights = b.constant(v(&[p[4], 1.0 - p[4]]));
    let x = b.constant(v(&DATA));
    let mut parameters = vec![weights.id()];
    parameters.extend(&ids);
    let kind = K::MixtureOf(Arc::from(vec![K::Gamma, K::LogNormal]));
    let density = b.log_density(kind, x, &parameters).id();
    let (graph, mut state) = b.build().unwrap();
    state
        .lazy_eval(&graph, &[density], &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    let expected: Vec<f64> = DATA.iter().map(|&x| hand_mixture(x, &p)).collect();
    assert_all_close(
        state.f64(density).unwrap().as_slice(),
        &expected,
        1e-12,
        "log density",
    );
}

#[test]
fn a_mixture_with_a_cured_component_has_the_cure_model_likelihood() {
    // a censored Weibull and a censored uniform on [c, c + 1] (all its mass at or beyond the censoring time c):
    // an event at t < c has log((1 − π) f(t)), a time censored at c has log(π + (1 − π) S(c))
    let (alpha, beta, pi) = (1.4, 2.0, 0.3);
    let c = [2.0, 1.5, 3.0];
    let times = [0.8, 1.5, 3.0];
    let b = GraphBuilder::new();
    let censoring = b.constant(v(&c));
    let weibull = [
        b.scalar(alpha).id(),
        b.scalar(beta).id(),
        b.scalar(f64::NEG_INFINITY).id(),
        censoring.id(),
    ];
    let cured = [
        censoring.id(),
        (censoring + 1.0).id(),
        b.scalar(f64::NEG_INFINITY).id(),
        censoring.id(),
    ];
    let weights = b.constant(v(&[1.0 - pi, pi]));
    let components = [
        (K::Censored(Arc::new(K::Weibull)), &weibull[..]),
        (K::Censored(Arc::new(K::Uniform)), &cured[..]),
    ];
    let y: DoubleVar = b.mixture_of(&components, weights);
    b.observe(y, v(&times));
    let reference: DoubleVar = b.weibull_with_shape(&[3], alpha, beta);
    b.observe(reference, v(&times));
    let (y, reference) = (y.id(), reference.id());
    let (graph, mut state) = b.build().unwrap();
    let all: Vec<NodeId> = graph.ids().collect();
    state
        .lazy_eval(&graph, &all, &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    let terms = node_log_prob_terms(&graph, &state, y).unwrap();
    // the Weibull's density and survival function, from the graph's own Weibull and its CDF in closed form
    let density = node_log_prob_terms(&graph, &state, reference).unwrap();
    let (alpha_t, beta_t) = (s(alpha), s(beta));
    let weibull = keanu_distributions::Weibull::new(&alpha_t, &beta_t);
    let survival = |t: f64| weibull.log_ccdf(&s(t)).unwrap().as_slice()[0];
    assert_close(
        terms.as_slice()[0],
        (1.0 - pi).ln() + density.as_slice()[0],
        1e-12,
        "an event",
    );
    for (&term, &t) in terms.as_slice().iter().zip(&times).skip(1) {
        let expected = log_sum_exp(pi.ln(), (1.0 - pi).ln() + survival(t));
        assert_close(term, expected, 1e-12, "a censored time");
    }
}

fn build_error(build: impl FnOnce(&GraphBuilder)) -> String {
    let b = GraphBuilder::new();
    build(&b);
    match b.build() {
        Err(GraphError::InvalidNode { message, .. }) => message,
        other => panic!("expected an invalid node, got {other:?}"),
    }
}

#[test]
fn invalid_heterogeneous_mixtures_and_transformations_are_refused() {
    // components of different value types
    let message = build_error(|b| {
        let rate = [b.scalar(1.0).id()];
        let gaussian = [b.scalar(0.0).id(), b.scalar(1.0).id()];
        let weights = b.constant(v(&[0.5, 0.5]));
        let _: DoubleVar = b.mixture_of(
            &[(K::Gaussian, &gaussian[..]), (K::Poisson, &rate[..])],
            weights,
        );
    });
    assert!(message.contains("one type"), "{message}");
    // a component that isn't element-wise
    let message = build_error(|b| {
        let concentration = [b.constant(v(&[1.0, 1.0])).id()];
        let gaussian = [b.scalar(0.0).id(), b.scalar(1.0).id()];
        let weights = b.constant(v(&[0.5, 0.5]));
        let _: DoubleVar = b.mixture_of(
            &[
                (K::Dirichlet, &concentration[..]),
                (K::Gaussian, &gaussian[..]),
            ],
            weights,
        );
    });
    assert!(message.contains("cannot be mixed"), "{message}");
    // weights whose last axis isn't the number of components
    let message = build_error(|b| {
        let gaussian = [b.scalar(0.0).id(), b.scalar(1.0).id()];
        let weights = b.constant(v(&[0.2, 0.3, 0.5]));
        let _: DoubleVar = b.mixture_of(
            &[(K::Gaussian, &gaussian[..]), (K::Gaussian, &gaussian[..])],
            weights,
        );
    });
    assert!(message.contains("2 components"), "{message}");
    // a censored base, and an empty interval
    let message = build_error(|b| {
        let p = [
            b.scalar(0.0).id(),
            b.scalar(1.0).id(),
            b.scalar(-1.0).id(),
            b.scalar(1.0).id(),
        ];
        b.transformed(
            K::Censored(Arc::new(K::Gaussian)),
            &p,
            ElementBijector::LowerBound(0.0),
        );
    });
    assert!(message.contains("cannot be transformed"), "{message}");
    let message = build_error(|b| {
        let p = [b.scalar(0.0).id(), b.scalar(1.0).id()];
        b.transformed(K::Gaussian, &p, ElementBijector::Interval(1.0, 1.0));
    });
    assert!(message.contains("bijector"), "{message}");
    // the handle type must match
    let b = GraphBuilder::new();
    let gaussian = [b.scalar(0.0).id(), b.scalar(1.0).id()];
    let weights = b.constant(v(&[1.0]));
    let _: IntVar = b.mixture_of(&[(K::Gaussian, &gaussian[..])], weights);
    assert!(b.build().is_err());
}

#[test]
fn the_kinds_report_their_names_and_parameters() {
    let kind = K::MixtureOf(Arc::from(vec![K::Gamma, K::Poisson]));
    assert_eq!(kind.name(), "mixture_of");
    assert_eq!(
        kind.parameter_names().unwrap().as_ref(),
        ["weights", "theta", "k", "mu"]
    );
    let kind = K::Transformed(Arc::new(K::Gaussian), ElementBijector::LowerBound(0.0));
    assert_eq!(kind.name(), "transformed");
    assert_eq!(kind.parameter_names().unwrap().as_ref(), ["mu", "sigma"]);
    assert!(matches!(
        Op::LogDensity(kind.clone()),
        Op::LogDensity(K::Transformed(..))
    ));
}
