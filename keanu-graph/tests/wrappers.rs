//! Truncated and censored distributions: densities, supports, sampling, gradients against finite differences and
//! invalid bounds.

use keanu_distributions::{Gamma, Gaussian, KeanuRng, Poisson};
use keanu_graph::op::DistributionKind;
use keanu_graph::{
    DoubleVar, Graph, GraphBuilder, GraphError, IntVar, NodeId, State, Support, Value,
    node_log_prob, node_log_prob_gradient, node_log_prob_terms, node_support,
};
use keanu_tensor::{DoubleTensor, IntTensor};

mod common;
use common::{evaluate, s, v};

const INF: f64 = f64::INFINITY;

/// A graph with a truncated (or censored) Gaussian(mu, sigma) on [lower, upper], whose parameters and bounds are
/// constants; returns the node and the parameter nodes.
struct Wrapped {
    graph: Graph,
    state: State,
    node: NodeId,
    parameters: Vec<NodeId>,
}

fn wrapped(
    censored: bool,
    kind: DistributionKind,
    parameters: &[DoubleTensor],
    lower: DoubleTensor,
    upper: DoubleTensor,
) -> Wrapped {
    let b = GraphBuilder::new();
    let mut ids: Vec<NodeId> = parameters
        .iter()
        .map(|p| b.constant(p.clone()).id())
        .collect();
    let (l, u) = (b.constant(lower), b.constant(upper));
    let x: DoubleVar = if censored {
        b.censored(kind, &ids, l, u)
    } else {
        b.truncated(kind, &ids, l, u)
    };
    ids.extend([l.id(), u.id()]);
    let node = x.id();
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state, 1);
    Wrapped {
        graph,
        state,
        node,
        parameters: ids,
    }
}

impl Wrapped {
    fn log_prob_at(&mut self, x: DoubleTensor) -> f64 {
        self.state
            .set(&self.graph, self.node, Value::F64(x))
            .unwrap();
        node_log_prob(&self.graph, &self.state, self.node).unwrap()
    }
}

#[test]
fn truncated_gaussian_density_is_twice_the_gaussian_above_zero() {
    let mut w = wrapped(
        false,
        DistributionKind::Gaussian,
        &[s(0.0), s(1.0)],
        s(0.0),
        s(INF),
    );
    let normal = Gaussian::new(s(0.0), s(1.0));
    for x in [0.0, 0.3, 1.7, 4.0] {
        let expected = normal.log_prob(&s(x)).unwrap().as_slice()[0] + std::f64::consts::LN_2;
        let actual = w.log_prob_at(s(x));
        assert!(
            (actual - expected).abs() < 1e-12,
            "{x}: {actual} vs {expected}"
        );
    }
    assert_eq!(w.log_prob_at(s(-0.1)), f64::NEG_INFINITY);
    assert_eq!(
        node_support(&w.graph, &w.state, w.node).unwrap(),
        Support::greater_than(0.0)
    );
}

#[test]
fn supports_intersect_the_base_support_with_the_bounds() {
    let support = |kind, parameters: &[DoubleTensor], lower, upper| {
        let w = wrapped(false, kind, parameters, s(lower), s(upper));
        node_support(&w.graph, &w.state, w.node).unwrap()
    };
    let gaussian = [s(0.0), s(1.0)];
    assert_eq!(
        support(DistributionKind::Gaussian, &gaussian, -1.0, 2.0),
        Support::interval(-1.0, 2.0)
    );
    assert_eq!(
        support(DistributionKind::Gaussian, &gaussian, -INF, 2.0),
        Support::less_than(2.0)
    );
    assert_eq!(
        support(DistributionKind::Gaussian, &gaussian, -INF, INF),
        Support::Real
    );
    // a gamma truncated above has the interval (0, upper); below 0 the bound adds nothing
    let gamma = [s(1.0), s(2.0)];
    assert_eq!(
        support(DistributionKind::Gamma, &gamma, -INF, 5.0),
        Support::interval(0.0, 5.0)
    );
    assert_eq!(
        support(DistributionKind::Gamma, &gamma, -3.0, INF),
        Support::greater_than(0.0)
    );
    assert_eq!(
        support(DistributionKind::Uniform, &[s(0.0), s(10.0)], 2.0, 20.0),
        Support::interval(2.0, 10.0)
    );
}

#[test]
fn truncated_densities_integrate_to_one() {
    let cases: Vec<(DistributionKind, Vec<DoubleTensor>, f64, f64)> = vec![
        (DistributionKind::Gaussian, vec![s(0.5), s(2.0)], -1.0, 3.0),
        (DistributionKind::Gamma, vec![s(1.5), s(2.0)], 0.5, 6.0),
        (DistributionKind::Logistic, vec![s(0.0), s(1.0)], 8.0, 12.0),
        (DistributionKind::Cauchy, vec![s(0.0), s(1.0)], -30.0, -10.0),
        (DistributionKind::Weibull, vec![s(1.5), s(2.0)], 0.5, 3.0),
    ];
    for (kind, parameters, lower, upper) in cases {
        let name = format!("{kind:?}");
        let mut w = wrapped(false, kind, &parameters, s(lower), s(upper));
        let n = 20_000;
        let h = (upper - lower) / n as f64;
        let mut total = 0.0;
        for i in 0..n {
            let x = lower + (i as f64 + 0.5) * h;
            total += w.log_prob_at(s(x)).exp() * h;
        }
        assert!((total - 1.0).abs() < 1e-6, "{name}: {total}");
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

#[test]
fn truncated_samples_lie_within_the_bounds_and_match_the_truncated_mean() {
    // N(0, 1) truncated to [a, b] has mean (φ(a) - φ(b)) / (Φ(b) - Φ(a))
    let phi = |x: f64| (-x * x / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let cdf = |x: f64| keanu_tensor::special::ndtr(x);
    let n = 20_000;
    for (a, b) in [(0.5, 2.0), (-INF, -1.0), (3.0, INF), (8.0, 9.0)] {
        let w = {
            let builder = GraphBuilder::new();
            let (mu, sigma) = (builder.scalar(0.0), builder.scalar(1.0));
            let x: DoubleVar = builder.truncated(
                DistributionKind::Gaussian,
                &[mu.id(), sigma.id()],
                DoubleTensor::full(&[n], a),
                b,
            );
            let id = x.id();
            let (graph, mut state) = builder.build().unwrap();
            evaluate(&graph, &mut state, 7);
            state
                .value(id)
                .unwrap()
                .as_f64()
                .unwrap()
                .as_slice()
                .to_vec()
        };
        assert!(w.iter().all(|&x| x >= a && x <= b), "[{a}, {b}]");
        let (pa, pb) = (
            if a.is_finite() { phi(a) } else { 0.0 },
            if b.is_finite() { phi(b) } else { 0.0 },
        );
        // the survival form keeps the precision in the upper tail
        let z = cdf(-a) - cdf(-b);
        let expected = (pa - pb) / z;
        let variance = 1.0 + (a.max(-1e300) * pa - b.min(1e300) * pb) / z - expected * expected;
        let se = (variance / n as f64).sqrt();
        let m = mean(&w);
        assert!(
            (m - expected).abs() < 5.0 * se,
            "[{a}, {b}]: mean {m} vs {expected} (se {se})"
        );
    }
}

/// `n` draws of `kind` with constant `parameters` truncated to [a, b].
fn truncated_draws(
    kind: DistributionKind,
    parameters: &[f64],
    a: f64,
    b: f64,
    n: usize,
) -> Vec<f64> {
    let builder = GraphBuilder::new();
    let ids: Vec<NodeId> = parameters.iter().map(|&p| builder.scalar(p).id()).collect();
    let x: DoubleVar = builder.truncated(kind, &ids, DoubleTensor::full(&[n], a), b);
    let id = x.id();
    let (graph, mut state) = builder.build().unwrap();
    evaluate(&graph, &mut state, 9);
    state
        .value(id)
        .unwrap()
        .as_f64()
        .unwrap()
        .as_slice()
        .to_vec()
}

#[test]
fn truncated_samples_in_far_tails_are_finite_and_inside_the_bounds() {
    let n = 4000;
    // N(0, 1) beyond l has mean φ(l)/Q(l) = l + 1/l − 2/l³ + 10/l⁵ − … and a variance of about 1/l²; the tails
    // probabilities underflow to 0 as plain numbers
    for (a, b, sign) in [
        (38.5, INF, 1.0),
        (40.0, INF, 1.0),
        (40.0, 41.0, 1.0),
        (-INF, -40.0, -1.0),
    ] {
        let w = truncated_draws(DistributionKind::Gaussian, &[0.0, 1.0], a, b, n);
        assert!(
            w.iter().all(|&x| x.is_finite() && x > a && x < b),
            "[{a}, {b}]: {:?}",
            w.iter().find(|&&x| !(x.is_finite() && x > a && x < b))
        );
        let l: f64 = if sign > 0.0 { a } else { -b };
        let expected = sign * (l + 1.0 / l - 2.0 / l.powi(3) + 10.0 / l.powi(5));
        let se = (1.0 / (l * l) / n as f64).sqrt();
        let m = mean(&w);
        assert!(
            (m - expected).abs() < 5.0 * se,
            "[{a}, {b}]: mean {m} vs {expected} (se {se})"
        );
        // the draws spread out rather than collapsing onto one value
        assert!(w.iter().any(|&x| (x - w[0]).abs() > 1e-3), "[{a}, {b}]");
    }
    // Exponential(1) beyond 800 is 800 plus an Exponential(1), by memorylessness
    let w = truncated_draws(DistributionKind::Exponential, &[1.0], 800.0, INF, n);
    assert!(
        w.iter().all(|&x| x.is_finite() && x > 800.0),
        "{:?}",
        &w[..5]
    );
    let m = mean(&w);
    let se = (1.0 / n as f64).sqrt();
    assert!((m - 801.0).abs() < 5.0 * se, "mean {m}");
}

#[test]
fn truncation_without_an_inverse_cdf_samples_by_rejection() {
    // Gamma(theta 1, k 2) truncated to [1, 3]
    let n = 20_000;
    let b = GraphBuilder::new();
    let (theta, k) = (b.scalar(1.0), b.scalar(2.0));
    let x: DoubleVar = b.truncated(
        DistributionKind::Gamma,
        &[theta.id(), k.id()],
        DoubleTensor::full(&[n], 1.0),
        3.0,
    );
    let id = x.id();
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state, 5);
    let samples = state
        .value(id)
        .unwrap()
        .as_f64()
        .unwrap()
        .as_slice()
        .to_vec();
    assert!(samples.iter().all(|&x| (1.0..=3.0).contains(&x)));
    // the truncated mean, by quadrature of x times the density
    let gamma = Gamma::new(s(1.0), s(2.0));
    let (mut z, mut first) = (0.0, 0.0);
    let steps = 20_000;
    let h = 2.0 / steps as f64;
    for i in 0..steps {
        let x = 1.0 + (i as f64 + 0.5) * h;
        let p = gamma.log_prob(&s(x)).unwrap().as_slice()[0].exp() * h;
        z += p;
        first += x * p;
    }
    let expected = first / z;
    let m = mean(&samples);
    assert!((m - expected).abs() < 0.01, "mean {m} vs {expected}");

    // bounds holding almost none of the probability: an error, not a hang
    let b = GraphBuilder::new();
    let (theta, k) = (b.scalar(1.0), b.scalar(2.0));
    let x: DoubleVar = b.truncated(DistributionKind::Gamma, &[theta.id(), k.id()], 60.0, 61.0);
    let id = x.id();
    let (graph, mut state) = b.build().unwrap();
    let error = state
        .eval(&graph, &[id], &mut KeanuRng::seed_from_u64(1))
        .unwrap_err();
    assert!(matches!(error, GraphError::Distribution(_)), "{error}");
}

#[test]
fn zero_truncated_poisson() {
    let mu = 1.3;
    let b = GraphBuilder::new();
    let m = b.scalar(mu);
    let k: IntVar = b.truncated(DistributionKind::Poisson, &[m.id()], 1.0, INF);
    let id = k.id();
    let (graph, mut state) = b.build().unwrap();
    let poisson = Poisson::new(s(mu));
    let normaliser = (1.0 - (-mu).exp()).ln();
    for x in [1, 2, 5] {
        state
            .set(&graph, id, Value::I64(IntTensor::scalar(x)))
            .unwrap();
        let expected = poisson.log_prob(&IntTensor::scalar(x)).unwrap().as_slice()[0] - normaliser;
        let actual = node_log_prob(&graph, &state, id).unwrap();
        assert!(
            (actual - expected).abs() < 1e-12,
            "{x}: {actual} vs {expected}"
        );
    }
    state
        .set(&graph, id, Value::I64(IntTensor::scalar(0)))
        .unwrap();
    assert_eq!(
        node_log_prob(&graph, &state, id).unwrap(),
        f64::NEG_INFINITY
    );
    assert_eq!(node_support(&graph, &state, id).unwrap(), Support::Discrete);
    // samples by rejection, all at least 1, with mean mu / (1 - e^-mu)
    let n = 20_000;
    let b = GraphBuilder::new();
    let m = b.scalar(mu);
    let k: IntVar = b.truncated(
        DistributionKind::Poisson,
        &[m.id()],
        DoubleTensor::full(&[n], 1.0),
        INF,
    );
    let id = k.id();
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state, 9);
    let samples: Vec<f64> = state
        .value(id)
        .unwrap()
        .as_i64()
        .unwrap()
        .as_slice()
        .iter()
        .map(|&k| k as f64)
        .collect();
    assert!(samples.iter().all(|&k| k >= 1.0));
    let expected = mu / (1.0 - (-mu).exp());
    let variance = expected * (1.0 + mu - expected);
    let m = mean(&samples);
    assert!(
        (m - expected).abs() < 5.0 * (variance / n as f64).sqrt(),
        "{m} vs {expected}"
    );
}

#[test]
fn censored_likelihood_at_the_bounds_is_the_tail_probability() {
    let (mu, sigma, lower, upper) = (0.5, 1.5, -1.0, 2.0);
    let mut w = wrapped(
        true,
        DistributionKind::Gaussian,
        &[s(mu), s(sigma)],
        s(lower),
        s(upper),
    );
    let normal = Gaussian::new(s(mu), s(sigma));
    let at = |t: keanu_distributions::Result<DoubleTensor>| t.unwrap().as_slice()[0];
    assert!((w.log_prob_at(s(upper)) - at(normal.log_ccdf(&s(upper)))).abs() < 1e-14);
    assert!((w.log_prob_at(s(lower)) - at(normal.log_cdf(&s(lower)))).abs() < 1e-14);
    assert!((w.log_prob_at(s(0.3)) - at(normal.log_prob(&s(0.3)))).abs() < 1e-14);
    assert_eq!(w.log_prob_at(s(2.5)), f64::NEG_INFINITY);
    assert_eq!(w.log_prob_at(s(-1.5)), f64::NEG_INFINITY);

    // a discrete base: P(X >= upper) = P(X > upper - 1)
    let b = GraphBuilder::new();
    let m = b.scalar(3.0);
    let k: IntVar = b.censored(DistributionKind::Poisson, &[m.id()], 1.0, 5.0);
    let id = k.id();
    let (graph, mut state) = b.build().unwrap();
    let poisson = Poisson::new(s(3.0));
    state
        .set(&graph, id, Value::I64(IntTensor::scalar(5)))
        .unwrap();
    let expected = at(poisson.log_ccdf(&IntTensor::scalar(4)));
    assert!((node_log_prob(&graph, &state, id).unwrap() - expected).abs() < 1e-14);
    state
        .set(&graph, id, Value::I64(IntTensor::scalar(1)))
        .unwrap();
    let expected = at(poisson.log_cdf(&IntTensor::scalar(1)));
    assert!((node_log_prob(&graph, &state, id).unwrap() - expected).abs() < 1e-14);
}

#[test]
fn bounded_poisson_binomial_broadcasts_bounds_over_trial_batches() {
    let b = GraphBuilder::new();
    let p = b.constant(
        DoubleTensor::new(&[2, 1, 4], vec![0.5, 0.5, 0.5, 0.5, 0.0, 1.0, 0.5, 0.5]).unwrap(),
    );
    let lower = v(&[1.0, 1.0, -INF]);
    let upper = v(&[4.0, 3.0, 3.0]);
    let censored: IntVar<'_> = b.censored(
        DistributionKind::PoissonBinomial,
        &[p.id()],
        lower.clone(),
        upper.clone(),
    );
    let truncated: IntVar<'_> =
        b.truncated(DistributionKind::PoissonBinomial, &[p.id()], lower, upper);
    let x = IntTensor::new(&[2, 3], vec![1, 2, 3, 1, 2, 3]).unwrap();
    b.observe(censored, x.clone());
    b.observe(truncated, x);
    let (censored, truncated) = (censored.id(), truncated.id());
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state, 91);
    for (node, expected) in [
        (
            censored,
            [5.0 / 16.0, 6.0 / 16.0, 5.0 / 16.0, 0.25, 0.5, 0.25],
        ),
        (
            truncated,
            [4.0 / 15.0, 6.0 / 14.0, 4.0 / 15.0, 0.25, 0.5, 0.25],
        ),
    ] {
        let terms = node_log_prob_terms(&graph, &state, node).unwrap();
        assert_eq!(terms.shape(), &[2, 3]);
        for (&term, probability) in terms.as_slice().iter().zip(expected) {
            assert!((term.exp() - probability).abs() < 1e-14);
        }
        assert_eq!(node_log_prob(&graph, &state, node).unwrap(), terms.sum());
    }
}

#[test]
fn censored_samples_are_clipped_to_the_bounds() {
    let n = 10_000;
    let b = GraphBuilder::new();
    let (mu, sigma) = (b.scalar(0.0), b.scalar(1.0));
    let x: DoubleVar = b.censored(
        DistributionKind::Gaussian,
        &[mu.id(), sigma.id()],
        DoubleTensor::full(&[n], -0.5),
        1.0,
    );
    let id = x.id();
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state, 3);
    let samples = state
        .value(id)
        .unwrap()
        .as_f64()
        .unwrap()
        .as_slice()
        .to_vec();
    assert!(samples.iter().all(|&x| (-0.5..=1.0).contains(&x)));
    let at_upper = samples.iter().filter(|&&x| x == 1.0).count() as f64 / n as f64;
    let expected = keanu_tensor::special::ndtr(-1.0);
    assert!(
        (at_upper - expected).abs() < 0.015,
        "{at_upper} vs {expected}"
    );
}

/// Checks a node's log prob gradient with respect to each of its f64 parents (and its value, if `value`) against
/// central differences. For a censored node (`value` false), a value element at a bound moves with the bound: it
/// stands for "at or beyond the bound", and the gradient with respect to the bound is taken with that held.
fn check_gradients(w: &mut Wrapped, x: DoubleTensor, value: bool) {
    const H: f64 = 1e-6;
    w.state
        .set(&w.graph, w.node, Value::F64(x.clone()))
        .unwrap();
    let gradient = node_log_prob_gradient(&w.graph, &w.state, w.node).unwrap();
    let mut wrt: Vec<(String, NodeId, DoubleTensor, Option<DoubleTensor>)> = w
        .parameters
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            (
                format!("parameter {i}"),
                p,
                w.state.value(p).unwrap().as_f64().unwrap().clone(),
                gradient.parameters[i].clone(),
            )
        })
        .collect();
    if value {
        wrt.push(("value".into(), w.node, x.clone(), gradient.value.clone()));
    }
    for (name, id, at, analytic) in wrt {
        let analytic = analytic.unwrap_or_else(|| panic!("{name} has no gradient"));
        if at.as_slice().iter().all(|b| b.is_infinite()) {
            // perturbing an infinite bound is meaningless; the log prob doesn't depend on it
            assert_eq!(analytic.sum(), 0.0, "{name}");
            continue;
        }
        // element-wise gradients are summed over broadcast dimensions
        let analytic: Vec<f64> = if analytic.len() != at.len() && at.len() == 1 {
            vec![analytic.sum()]
        } else {
            analytic.as_slice().to_vec()
        };
        for i in 0..at.len() {
            let perturbed = |d: f64| {
                let mut data = at.as_slice().to_vec();
                data[i] += d;
                let mut state = w.state.clone();
                if !value && id != w.node {
                    // the value elements at this bound element (every element for a scalar bound)
                    let mut moved = x.as_slice().to_vec();
                    for (k, m) in moved.iter_mut().enumerate() {
                        if (at.len() == 1 || k == i) && *m == at.as_slice()[i] {
                            *m += d;
                        }
                    }
                    state
                        .set(
                            &w.graph,
                            w.node,
                            Value::F64(DoubleTensor::new(x.shape(), moved).unwrap()),
                        )
                        .unwrap();
                }
                state
                    .set(
                        &w.graph,
                        id,
                        Value::F64(DoubleTensor::new(at.shape(), data).unwrap()),
                    )
                    .unwrap();
                node_log_prob(&w.graph, &state, w.node).unwrap()
            };
            let numeric = (perturbed(H) - perturbed(-H)) / (2.0 * H);
            assert!(
                (numeric - analytic[i]).abs() < 1e-5 * numeric.abs().max(1.0),
                "{name}[{i}]: analytic {}, numeric {numeric}",
                analytic[i]
            );
        }
    }
}

#[test]
fn truncated_gradients_match_finite_differences() {
    let x = v(&[-0.5, 0.4, 2.2]);
    for (lower, upper) in [(-1.0, 3.0), (-1.0, INF), (-INF, 2.5), (1.5, 4.0)] {
        let x = if lower > 0.0 {
            v(&[1.6, 2.0, 3.9])
        } else {
            x.clone()
        };
        let mut w = wrapped(
            false,
            DistributionKind::Gaussian,
            &[s(0.3), s(1.2)],
            DoubleTensor::full(&[3], lower),
            s(upper),
        );
        check_gradients(&mut w, x, true);
    }
    // shape-parameter gradients through the incomplete gamma and beta functions
    let mut w = wrapped(
        false,
        DistributionKind::Gamma,
        &[s(1.3), s(2.4)],
        DoubleTensor::full(&[3], 0.5),
        s(4.0),
    );
    check_gradients(&mut w, v(&[0.7, 2.0, 3.5]), true);
    let mut w = wrapped(
        false,
        DistributionKind::Beta,
        &[s(2.0), s(3.5)],
        DoubleTensor::full(&[3], 0.1),
        s(0.6),
    );
    check_gradients(&mut w, v(&[0.2, 0.35, 0.55]), true);
    // the upper tail, where the normaliser comes from the survival function
    let mut w = wrapped(
        false,
        DistributionKind::Gaussian,
        &[s(0.0), s(1.0)],
        DoubleTensor::full(&[3], 3.0),
        s(5.0),
    );
    check_gradients(&mut w, v(&[3.1, 3.5, 4.5]), true);
}

#[test]
fn censored_gradients_match_finite_differences() {
    // one element at each bound and one inside
    let mut w = wrapped(
        true,
        DistributionKind::Gaussian,
        &[s(0.3), s(1.2)],
        DoubleTensor::full(&[3], -1.0),
        s(2.0),
    );
    check_gradients(&mut w, v(&[-1.0, 0.5, 2.0]), false);
    let mut w = wrapped(
        true,
        DistributionKind::Gamma,
        &[s(1.3), s(2.4)],
        DoubleTensor::full(&[3], 0.5),
        s(4.0),
    );
    check_gradients(&mut w, v(&[0.5, 2.0, 4.0]), false);
}

#[test]
fn invalid_bounds_and_bases_are_errors() {
    // constant bounds are checked when building
    let b = GraphBuilder::new();
    let (mu, sigma) = (b.scalar(0.0), b.scalar(1.0));
    let _: DoubleVar = b.truncated(DistributionKind::Gaussian, &[mu.id(), sigma.id()], 1.0, 1.0);
    assert!(matches!(
        b.build(),
        Err(GraphError::InvalidNode {
            op: "truncated",
            ..
        })
    ));
    let b = GraphBuilder::new();
    let (mu, sigma) = (b.scalar(0.0), b.scalar(1.0));
    let _: DoubleVar = b.censored(
        DistributionKind::Gaussian,
        &[mu.id(), sigma.id()],
        2.0,
        -1.0,
    );
    assert!(matches!(
        b.build(),
        Err(GraphError::InvalidNode { op: "censored", .. })
    ));

    // bounds that are not constants and cross make the state impossible (as a uniform's do), so samplers reject
    // it; the support and sampling are still errors
    for censored in [false, true] {
        let b = GraphBuilder::new();
        let (mu, sigma) = (b.scalar(0.0), b.scalar(1.0));
        let upper = b.gaussian(0.0, 1.0);
        let params = [mu.id(), sigma.id()];
        let x: DoubleVar = if censored {
            b.censored(DistributionKind::Gaussian, &params, 0.5, upper)
        } else {
            b.truncated(DistributionKind::Gaussian, &params, 0.5, upper)
        };
        let (x, upper) = (x.id(), upper.id());
        let (graph, mut state) = b.build().unwrap();
        for (bound, value) in [(0.0, 0.0), (0.5, 0.5), (f64::NAN, 0.7)] {
            state.set(&graph, upper, Value::F64(s(bound))).unwrap();
            state.set(&graph, x, Value::F64(s(value))).unwrap();
            assert_eq!(
                node_log_prob(&graph, &state, x).unwrap(),
                f64::NEG_INFINITY,
                "censored {censored}, upper {bound}"
            );
            assert!(node_log_prob_gradient(&graph, &state, x).is_ok());
            assert!(matches!(
                node_support(&graph, &state, x),
                Err(GraphError::Distribution(_))
            ));
        }
        // drawing a value between crossed bounds is still an error
        state.set(&graph, upper, Value::F64(s(0.2))).unwrap();
        state.clear(x).unwrap();
        assert!(matches!(
            state.lazy_eval(&graph, &[x], &mut KeanuRng::seed_from_u64(1)),
            Err(GraphError::Distribution(_))
        ));
    }

    // a base without a CDF, wrong parameters, a wrapper of a wrapper, the wrong handle type
    let b = GraphBuilder::new();
    let c = b.scalar(1.0);
    let _: DoubleVar = b.truncated(DistributionKind::HalfCauchy, &[c.id()], 0.0, 1.0);
    assert!(matches!(b.build(), Err(GraphError::InvalidNode { .. })));
    let b = GraphBuilder::new();
    let c = b.scalar(1.0);
    let _: DoubleVar = b.truncated(DistributionKind::Gaussian, &[c.id()], 0.0, 1.0);
    assert!(b.build().is_err());
    let b = GraphBuilder::new();
    let (mu, sigma) = (b.scalar(0.0), b.scalar(1.0));
    let inner = DistributionKind::Truncated(std::sync::Arc::new(DistributionKind::Gaussian));
    let (l, u) = (b.scalar(-1.0), b.scalar(1.0));
    let _: DoubleVar = b.truncated(inner, &[mu.id(), sigma.id(), l.id(), u.id()], -0.5, 0.5);
    assert!(b.build().is_err());
    let b = GraphBuilder::new();
    let m = b.scalar(2.0);
    let _: DoubleVar = b.truncated(DistributionKind::Poisson, &[m.id()], 1.0, 4.0);
    assert!(b.build().is_err());
    // a discrete base needs whole-number bounds
    let b = GraphBuilder::new();
    let m = b.scalar(2.0);
    let k: IntVar = b.truncated(DistributionKind::Poisson, &[m.id()], 0.5, 4.0);
    let k = k.id();
    let (graph, mut state) = b.build().unwrap();
    state
        .set(&graph, k, Value::I64(IntTensor::scalar(2)))
        .unwrap();
    assert!(node_log_prob(&graph, &state, k).is_err());
}
