//! Zero-inflated and hurdle kinds and the count and logit kinds as graph nodes: masses that sum to 1 and match closed
//! forms, log densities and node log probabilities that agree, gradients against finite differences, draws, the
//! inner kinds accepted, finite supports for Gibbs steps, and marginalisation of logit latents.

use keanu_distributions::KeanuRng;
use keanu_graph::op::{DistributionKind as K, Op};
use keanu_graph::{
    DoubleVar, GraphBuilder, GraphError, IntVar, Model, NodeId, Parameterisation, Space, State,
    Value, marginalize, node_log_prob, node_log_prob_terms,
};
use keanu_tensor::{BoolTensor, IntTensor};
use std::sync::Arc;

mod common;
use common::{assert_close, assert_close_rel, s, v};

fn evaluate(graph: &keanu_graph::Graph, state: &mut State) {
    let all: Vec<NodeId> = graph.ids().collect();
    state
        .lazy_eval(graph, &all, &mut KeanuRng::seed_from_u64(1))
        .unwrap();
}

fn zero_inflated(inner: K) -> K {
    K::ZeroInflated(Arc::new(inner))
}

fn hurdle(inner: K) -> K {
    K::Hurdle(Arc::new(inner))
}

/// The element-wise log density of `kind` at `value` with constant `parameters`.
fn log_density(kind: &K, value: Value, parameters: &[Value]) -> Vec<f64> {
    let b = GraphBuilder::new();
    let mut parents = vec![b.add_node(Op::Constant(value), &[]).unwrap()];
    for p in parameters {
        parents.push(b.add_node(Op::Constant(p.clone()), &[]).unwrap());
    }
    let density = b.add_node(Op::LogDensity(kind.clone()), &parents).unwrap();
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state);
    state.f64(density).unwrap().as_slice().to_vec()
}

fn counts(n: i64) -> Value {
    Value::I64(IntTensor::vector((0..=n).collect()))
}

fn f(x: f64) -> Value {
    Value::F64(s(x))
}

fn int(k: i64) -> Value {
    Value::I64(IntTensor::scalar(k))
}

fn poisson(k: i64, mu: f64) -> f64 {
    let ln_k: f64 = (1..=k).map(|i| (i as f64).ln()).sum();
    k as f64 * mu.ln() - mu - ln_k
}

#[test]
fn zero_inflated_and_hurdle_masses_match_their_closed_forms_and_sum_to_one() {
    let (mu, psi) = (2.3, 0.35);
    let zip = log_density(&zero_inflated(K::Poisson), counts(60), &[f(mu), f(psi)]);
    assert_close_rel(
        zip[0],
        (psi + (1.0 - psi) * (-mu).exp()).ln(),
        1e-14,
        "P(0)",
    );
    for k in 1..=5 {
        let expected = (1.0 - psi).ln() + poisson(k, mu);
        assert_close_rel(zip[k as usize], expected, 1e-13, "P(k)");
    }
    let hp = log_density(&hurdle(K::Poisson), counts(60), &[f(mu), f(psi)]);
    assert_close_rel(hp[0], psi.ln(), 1e-14, "hurdle P(0)");
    for k in 1..=5 {
        let expected = (1.0 - psi).ln() + poisson(k, mu) - (1.0 - (-mu).exp()).ln();
        assert_close_rel(hp[k as usize], expected, 1e-13, "hurdle P(k)");
    }
    let bb = [int(9), f(1.4), f(0.6)];
    let with_psi = |psi: f64| {
        let mut p = bb.to_vec();
        p.push(f(psi));
        p
    };
    for (kind, value, parameters) in [
        (zero_inflated(K::Poisson), counts(60), vec![f(mu), f(psi)]),
        (hurdle(K::Poisson), counts(60), vec![f(mu), f(psi)]),
        (
            zero_inflated(K::NegativeBinomial),
            counts(400),
            vec![f(3.0), f(0.8), f(0.2)],
        ),
        (
            hurdle(K::NegativeBinomial),
            counts(400),
            vec![f(3.0), f(0.8), f(0.7)],
        ),
        (zero_inflated(K::BetaBinomial), counts(9), with_psi(0.4)),
        (hurdle(K::BetaBinomial), counts(9), with_psi(0.1)),
        (hurdle(K::PoissonLog), counts(60), vec![f(0.5), f(0.25)]),
        (
            zero_inflated(K::BinomialLogit),
            counts(6),
            vec![f(-0.4), int(6), f(0.5)],
        ),
        // ψ at its ends
        (zero_inflated(K::Poisson), counts(60), vec![f(mu), f(0.0)]),
        (zero_inflated(K::Poisson), counts(60), vec![f(mu), f(1.0)]),
        (hurdle(K::Poisson), counts(60), vec![f(mu), f(1.0)]),
    ] {
        let name = format!("{kind:?}");
        let total: f64 = log_density(&kind, value, &parameters)
            .iter()
            .map(|l| l.exp())
            .sum();
        assert_close(total, 1.0, 1e-12, &name);
    }
    // a hurdle has no negative values, and a ψ outside [0, 1] has no values at all
    let negative = log_density(
        &hurdle(K::Poisson),
        Value::I64(IntTensor::scalar(-1)),
        &[f(mu), f(psi)],
    );
    assert_eq!(negative, [f64::NEG_INFINITY]);
    for bad in [-0.1, 1.5, f64::NAN] {
        let values = log_density(&zero_inflated(K::Poisson), counts(3), &[f(mu), f(bad)]);
        assert!(values.iter().all(|&l| l == f64::NEG_INFINITY), "{bad}");
    }
}

#[test]
fn a_continuous_hurdle_has_a_point_mass_at_zero() {
    // gamma with scale θ = 2, shape k = 3: ln p(x) = 2 ln x − ln Γ(3) − x/2 − 3 ln 2
    let gamma = |x: f64| 2.0 * x.ln() - 2f64.ln() - x / 2.0 - 3.0 * 2f64.ln();
    let psi = 0.3;
    let values = log_density(
        &hurdle(K::Gamma),
        Value::F64(v(&[0.0, 1.5, 4.0, -1.0])),
        &[f(2.0), f(3.0), f(psi)],
    );
    assert_close_rel(values[0], psi.ln(), 1e-14, "zero");
    assert_close_rel(values[1], (1.0 - psi).ln() + gamma(1.5), 1e-13, "x = 1.5");
    assert_close_rel(values[2], (1.0 - psi).ln() + gamma(4.0), 1e-13, "x = 4");
    assert_eq!(values[3], f64::NEG_INFINITY);
}

#[test]
fn log_densities_and_node_log_probabilities_agree() {
    let i = |x: Vec<i64>| Value::I64(IntTensor::vector(x));
    for (kind, value, parameters) in [
        (
            K::BetaBinomial,
            i(vec![0, 3, 8]),
            vec![int(8), f(1.5), f(2.5)],
        ),
        (K::BinomialLogit, i(vec![0, 3, 8]), vec![f(0.4), int(8)]),
        (
            K::PoissonLog,
            i(vec![0, 3, 8]),
            vec![Value::F64(v(&[0.1, 1.0, 2.0]))],
        ),
        (
            K::BernoulliLogit,
            Value::Bool(BoolTensor::vector(vec![true, false, true])),
            vec![Value::F64(v(&[-1.0, 0.5, 3.0]))],
        ),
        (
            zero_inflated(K::Poisson),
            i(vec![0, 0, 4]),
            vec![f(1.5), f(0.3)],
        ),
        (
            hurdle(K::NegativeBinomial),
            i(vec![0, 2, 4]),
            vec![f(1.5), f(2.0), f(0.3)],
        ),
        (
            hurdle(K::LogNormal),
            Value::F64(v(&[0.0, 0.5, 2.0])),
            vec![f(0.1), f(0.8), f(0.4)],
        ),
    ] {
        let name = format!("{kind:?}");
        let b = GraphBuilder::new();
        let params: Vec<NodeId> = parameters
            .iter()
            .map(|p| b.add_node(Op::Constant(p.clone()), &[]).unwrap())
            .collect();
        let node = b
            .add_distribution(kind.clone(), Some(&[3]), &params)
            .unwrap();
        b.observe_of(node, value.clone()).unwrap();
        let (graph, mut state) = b.build().unwrap();
        evaluate(&graph, &mut state);
        let expected = log_density(&kind, value, &parameters);
        let terms = node_log_prob_terms(&graph, &state, node).unwrap();
        assert_eq!(terms.shape(), &[3], "{name}");
        for (t, e) in terms.as_slice().iter().zip(&expected) {
            assert_close_rel(*t, *e, 1e-14, &name);
        }
        assert_close_rel(
            node_log_prob(&graph, &state, node).unwrap(),
            expected.iter().sum(),
            1e-13,
            &name,
        );
    }
}

/// A latent with a standard Gaussian prior set to `value`.
fn latent(b: &GraphBuilder, value: f64) -> DoubleVar<'_> {
    let x = b.gaussian(0.0, 1.0);
    b.set_value(x, s(value));
    x
}

/// Checks the joint log probability's gradient against central differences, in constrained space.
fn check_gradient(name: &str, build: impl FnOnce(&GraphBuilder)) {
    let b = GraphBuilder::new();
    build(&b);
    let (graph, mut state) = b.build().unwrap();
    evaluate(&graph, &mut state);
    let model = Model::new(graph, &state).unwrap();
    let param = Parameterisation::with_space(&model, &state, Space::Constrained).unwrap();
    let z = param.from_state(&state).unwrap();
    let (_, gradient) = param.log_prob_and_gradient(&z, false, &mut state).unwrap();
    let gradient = gradient.expect("a possible point");
    for j in 0..z.len() {
        let h = 1e-6;
        let mut at = |delta: f64| {
            let mut x = z.clone();
            x[j] += delta;
            param
                .log_prob_and_gradient(&x, false, &mut state)
                .unwrap()
                .0
        };
        let fd = (at(h) - at(-h)) / (2.0 * h);
        assert_close(gradient[j], fd, 1e-6, &format!("{name}: coordinate {j}"));
    }
}

#[test]
fn gradients_match_finite_differences() {
    check_gradient("zero-inflated Poisson", |b| {
        let rate = latent(b, 0.4).exp();
        let psi = latent(b, -0.6).sigmoid();
        let y: IntVar = b.zero_inflated_with_shape(&[4], K::Poisson, &[rate.id()], psi);
        b.observe(y, IntTensor::vector(vec![0, 0, 2, 5]));
    });
    check_gradient("zero-inflated beta-binomial", |b| {
        let (alpha, beta) = (latent(b, 0.2).exp(), latent(b, -0.3).exp());
        let n = b.int_scalar(7);
        let psi = latent(b, 0.1).sigmoid();
        let y: IntVar = b.zero_inflated_with_shape(
            &[3],
            K::BetaBinomial,
            &[n.id(), alpha.id(), beta.id()],
            psi,
        );
        b.observe(y, IntTensor::vector(vec![0, 3, 7]));
    });
    check_gradient("hurdle negative binomial", |b| {
        let (mu, alpha) = (latent(b, 0.9).exp(), latent(b, 0.2).exp());
        let psi = latent(b, 0.4).sigmoid();
        let y: IntVar = b.hurdle_with_shape(&[3], K::NegativeBinomial, &[mu.id(), alpha.id()], psi);
        b.observe(y, IntTensor::vector(vec![0, 1, 6]));
    });
    check_gradient("hurdle gamma", |b| {
        let (theta, k) = (latent(b, 0.3).exp(), latent(b, 0.5).exp());
        let psi = latent(b, -0.2).sigmoid();
        let y: DoubleVar = b.hurdle_with_shape(&[3], K::Gamma, &[theta.id(), k.id()], psi);
        b.observe(y, v(&[0.0, 0.8, 3.1]));
    });
    check_gradient("logit and log forms", |b| {
        let eta = latent(b, 0.7);
        let y = b.bernoulli_logit_with_shape(&[2], eta);
        b.observe(y, BoolTensor::vector(vec![true, false]));
        let z = b.binomial_logit_with_shape(&[2], eta * 2.0, 6);
        b.observe(z, IntTensor::vector(vec![1, 5]));
        let w = b.poisson_log_with_shape(&[2], eta - 0.5);
        b.observe(w, IntTensor::vector(vec![0, 3]));
        let logits = [eta, latent(b, -0.4), b.scalar(0.0)];
        let c = b.categorical_logit(&[3], &logits);
        b.observe(c, IntTensor::vector(vec![0, 1, 2]));
        let bb = b.beta_binomial_with_shape(&[2], 5, eta.exp(), 1.5);
        b.observe(bb, IntTensor::vector(vec![0, 4]));
    });
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

#[test]
fn draws_have_the_right_zeros_and_means() {
    let draws = 40_000;
    let sample = |kind: K, parameters: &[Value]| -> Vec<f64> {
        let b = GraphBuilder::new();
        let params: Vec<NodeId> = parameters
            .iter()
            .map(|p| b.add_node(Op::Constant(p.clone()), &[]).unwrap())
            .collect();
        let node = b.add_distribution(kind, Some(&[draws]), &params).unwrap();
        let (graph, mut state) = b.build().unwrap();
        state
            .lazy_eval(&graph, &[node], &mut KeanuRng::seed_from_u64(7))
            .unwrap();
        match state.value(node).unwrap() {
            Value::I64(t) => t.as_slice().iter().map(|&k| k as f64).collect(),
            Value::F64(t) => t.as_slice().to_vec(),
            other => panic!("unexpected value {other:?}"),
        }
    };
    let zeros = |x: &[f64]| x.iter().filter(|&&x| x == 0.0).count() as f64 / x.len() as f64;
    let (mu, psi) = (2.5, 0.3);
    let zip = sample(zero_inflated(K::Poisson), &[f(mu), f(psi)]);
    assert_close(
        zeros(&zip),
        psi + (1.0 - psi) * (-mu).exp(),
        0.01,
        "zero-inflated zeros",
    );
    assert_close(mean(&zip), (1.0 - psi) * mu, 0.02, "zero-inflated mean");
    let hp = sample(hurdle(K::Poisson), &[f(mu), f(psi)]);
    assert_close(zeros(&hp), psi, 0.01, "hurdle zeros");
    let positive: Vec<f64> = hp.iter().copied().filter(|&x| x > 0.0).collect();
    assert_close(
        mean(&positive),
        mu / (1.0 - (-mu).exp()),
        0.02,
        "hurdle positive mean",
    );
    let hg = sample(hurdle(K::Gamma), &[f(2.0), f(3.0), f(psi)]);
    assert_close(zeros(&hg), psi, 0.01, "hurdle gamma zeros");
    let positive: Vec<f64> = hg.iter().copied().filter(|&x| x > 0.0).collect();
    assert_close(mean(&positive), 6.0, 0.02, "hurdle gamma positive mean");
    let mut rng = KeanuRng::seed_from_u64(1);
    let b = GraphBuilder::new();
    let rate = b.scalar(1.0);
    let y: IntVar = b.zero_inflated(K::Poisson, &[rate.id()], 1.2);
    let y = y.id();
    let (graph, mut state) = b.build().unwrap();
    assert!(state.lazy_eval(&graph, &[y], &mut rng).is_err());
}

/// A hurdle over a count that is almost always 0 still draws its positive values (by inversion of the inner
/// kind's mass above 0, not rejection): with a Poisson rate μ the positive part is 1 with probability
/// μ / (e^μ − 1) ≈ 1 − μ/2.
#[test]
fn hurdle_counts_with_little_mass_above_zero_can_be_sampled() {
    let draws = 20_000;
    let (mu, psi) = (0.002, 0.5);
    let sample = |inner: K, parameters: Vec<Value>, size: usize| -> Vec<i64> {
        let b = GraphBuilder::new();
        let params: Vec<NodeId> = parameters
            .into_iter()
            .map(|p| b.add_node(Op::Constant(p), &[]).unwrap())
            .collect();
        let node = b
            .add_distribution(hurdle(inner), Some(&[size]), &params)
            .unwrap();
        let (graph, mut state) = b.build().unwrap();
        state
            .lazy_eval(&graph, &[node], &mut KeanuRng::seed_from_u64(3))
            .unwrap_or_else(|e| panic!("sampling failed: {e}"));
        state.i64(node).unwrap().as_slice().to_vec()
    };
    // the reviewer's case
    assert_eq!(sample(K::Poisson, vec![f(mu), f(psi)], 20).len(), 20);
    let x = sample(K::Poisson, vec![f(mu), f(psi)], draws);
    let zeros = x.iter().filter(|&&k| k == 0).count() as f64 / draws as f64;
    assert_close(zeros, psi, 0.015, "hurdle zeros");
    let positive: Vec<i64> = x.iter().copied().filter(|&k| k > 0).collect();
    let ones = positive.iter().filter(|&&k| k == 1).count() as f64 / positive.len() as f64;
    assert_close(ones, mu / mu.exp_m1(), 0.003, "positive draws that are 1");
    // a tiny rate, for which a draw from the Poisson is almost never positive, and its log form
    let x = sample(K::Poisson, vec![f(1e-12), f(0.0)], 50);
    assert!(x.iter().all(|&k| k == 1), "{x:?}");
    let x = sample(K::PoissonLog, vec![f(-30.0), f(0.0)], 50);
    assert!(x.iter().all(|&k| k == 1), "{x:?}");
    // a binomial with a small probability: the positive part's mean is np / (1 − (1 − p)ⁿ)
    let (n, p) = (4, 1e-3);
    let x = sample(K::Binomial, vec![f(p), int(n), f(0.4)], draws);
    let positive: Vec<f64> = x.iter().filter(|&&k| k > 0).map(|&k| k as f64).collect();
    let expected = n as f64 * p / (1.0 - (1.0 - p).powi(n as i32));
    assert_close(mean(&positive), expected, 0.01, "binomial positive mean");
    // large rates still draw (by rejection, which accepts almost every draw)
    let x = sample(K::Poisson, vec![f(400.0), f(0.2)], draws);
    let positive: Vec<f64> = x.iter().filter(|&&k| k > 0).map(|&k| k as f64).collect();
    assert_close(mean(&positive), 400.0, 1.0, "large-rate positive mean");
}

#[test]
fn inner_kinds_must_be_counts_or_positive_and_continuous() {
    let build = |kind: K, parameters: &[f64]| {
        let b = GraphBuilder::new();
        let ids: Vec<NodeId> = parameters.iter().map(|&p| b.scalar(p).id()).collect();
        b.add_distribution(kind, None, &ids).map(|_| ())
    };
    let invalid = |r: Result<(), GraphError>, what: &str| match r {
        Err(GraphError::InvalidNode { message, .. }) => {
            assert!(message.contains("inner"), "{what}: {message}")
        }
        other => panic!("{what}: expected an invalid node, got {other:?}"),
    };
    invalid(
        build(zero_inflated(K::Gaussian), &[0.0, 1.0, 0.5]),
        "zero-inflated Gaussian",
    );
    invalid(
        build(zero_inflated(K::Gamma), &[1.0, 1.0, 0.5]),
        "zero-inflated gamma",
    );
    invalid(
        build(zero_inflated(K::Bernoulli), &[0.5, 0.5]),
        "zero-inflated Bernoulli",
    );
    invalid(
        build(hurdle(K::Gaussian), &[0.0, 1.0, 0.5]),
        "hurdle Gaussian",
    );
    invalid(
        build(hurdle(K::Categorical), &[0.5, 0.5, 0.5]),
        "hurdle categorical",
    );
    build(hurdle(K::Gamma), &[1.0, 1.0, 0.5]).unwrap();
    build(hurdle(K::Poisson), &[1.0, 0.5]).unwrap();
    build(
        zero_inflated(K::Truncated(Arc::new(K::Poisson))),
        &[1.0, 0.0, 9.0, 0.5],
    )
    .unwrap();
    // they have no CDF, so they can't be truncated, but they can be mixed
    let b = GraphBuilder::new();
    let ids = [b.scalar(1.0).id(), b.scalar(0.5).id()];
    let _: IntVar = b.truncated(zero_inflated(K::Poisson), &ids, 0.0, 5.0);
    assert!(b.build().is_err());
    // the handle must match the inner kind's type
    let b = GraphBuilder::new();
    let _: DoubleVar = b.zero_inflated(K::Poisson, &[b.scalar(1.0).id()], 0.5);
    assert!(b.build().is_err());
}

#[test]
#[allow(clippy::single_range_in_vec_init)]
fn finite_supports_include_zero() {
    let p = Value::F64(s(0.3));
    let n = Value::I64(IntTensor::vector(vec![2, 5]));
    assert_eq!(
        K::BetaBinomial.finite_support(&[&n, &p, &p], &[2]).unwrap(),
        Some(vec![0..=2, 0..=5])
    );
    assert_eq!(
        K::BinomialLogit.finite_support(&[&p, &n], &[2]).unwrap(),
        Some(vec![0..=2, 0..=5])
    );
    assert_eq!(
        K::BernoulliLogit.finite_support(&[&p], &[]).unwrap(),
        Some(vec![0..=1])
    );
    assert_eq!(
        K::CategoricalLogit
            .finite_support(&[&p, &p, &p], &[])
            .unwrap(),
        Some(vec![0..=2])
    );
    let zib = zero_inflated(K::Binomial);
    assert_eq!(
        zib.finite_support(&[&p, &n, &p], &[2]).unwrap(),
        Some(vec![0..=2, 0..=5])
    );
    // a truncated count above zero gains 0
    let truncated = zero_inflated(K::Truncated(Arc::new(K::Poisson)));
    let (lower, upper) = (Value::F64(s(2.0)), Value::F64(s(4.0)));
    assert_eq!(
        truncated
            .finite_support(&[&p, &lower, &upper, &p], &[])
            .unwrap(),
        Some(vec![0..=4])
    );
    assert!(zib.has_finite_support() && hurdle(K::BetaBinomial).has_finite_support());
    for kind in [
        zero_inflated(K::Poisson),
        hurdle(K::NegativeBinomial),
        K::PoissonLog,
    ] {
        assert!(!kind.has_finite_support());
    }
    assert_eq!(
        zero_inflated(K::Poisson)
            .finite_support(&[&p, &p], &[])
            .unwrap(),
        None
    );
    assert_eq!(
        hurdle(K::Gamma).finite_support(&[&p, &p, &p], &[]).unwrap(),
        None
    );
}

fn log_sum_exp(xs: &[f64]) -> f64 {
    let max = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    max + xs.iter().map(|x| (x - max).exp()).sum::<f64>().ln()
}

#[test]
fn logit_latents_marginalise_to_their_enumerated_sums() {
    // a vector of Bernoulli-logit indicators, each choosing the mean of its observation
    let data = [0.3, 2.4, -0.5];
    let b = GraphBuilder::new();
    let eta = b.gaussian(0.0, 1.0);
    b.set_value(eta, s(0.6));
    let z = b.bernoulli_logit_with_shape(&[3], eta);
    b.set_value(z, BoolTensor::vector(vec![false; 3]));
    let means = b.where_(
        z,
        b.constant(v(&[2.0, 2.0, 2.0])),
        b.constant(v(&[0.0, 0.0, 0.0])),
    );
    let y = b.gaussian(means, 1.0);
    b.observe(y, v(&data));
    // and a categorical-logit assignment among three means
    let logits = [b.scalar(0.2), b.scalar(-0.4), eta];
    let c = b.categorical_logit(&[2], &logits);
    b.set_value(c, IntTensor::vector(vec![0, 0]));
    let centres = b.constant(v(&[-1.0, 0.5, 3.0]));
    let w = b.gaussian(centres.gather(0, c), 0.8);
    b.observe(w, v(&[0.1, 2.7]));
    let (z, c) = (z.id(), c.id());
    let (graph, state) = b.build().unwrap();
    let (graph, state, marginalized) = marginalize(&graph, &state, &[z, c]).unwrap();
    let model = Model::new(graph, &state).unwrap();

    let gaussian = |x: f64, m: f64, sd: f64| {
        -0.5 * ((x - m) / sd).powi(2) - sd.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
    };
    let log_sigmoid = |t: f64| -(1.0 + (-t).exp()).ln();
    let mut expected = gaussian(0.6, 0.0, 1.0);
    for &x in &data {
        expected += log_sum_exp(&[
            log_sigmoid(0.6) + gaussian(x, 2.0, 1.0),
            log_sigmoid(-0.6) + gaussian(x, 0.0, 1.0),
        ]);
    }
    let l = [0.2, -0.4, 0.6];
    let total = log_sum_exp(&l);
    for x in [0.1, 2.7] {
        let terms: Vec<f64> = [-1.0, 0.5, 3.0]
            .iter()
            .zip(&l)
            .map(|(&m, &lk)| lk - total + gaussian(x, m, 0.8))
            .collect();
        expected += log_sum_exp(&terms);
    }
    assert_close_rel(model.log_prob(&state).unwrap(), expected, 1e-12, "log prob");
    assert_eq!(marginalized.latents().len(), 2);
    let assignment = marginalized
        .latent(c)
        .unwrap()
        .probabilities(&state)
        .unwrap();
    assert_eq!(assignment.shape(), &[2, 3]);
    model.log_prob_gradient(&state).unwrap();
}
