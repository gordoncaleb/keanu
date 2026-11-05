//! Zero-inflated, hurdle, beta-binomial and logit kinds under inference: NUTS recovers a zero-inflated Poisson's ψ
//! and λ and a hurdle gamma's parameters from data with zeros, and Gibbs steps sample beta-binomial, Bernoulli-logit
//! and zero-inflated latents exactly.

mod mcmc_common;

use keanu_algorithms::mcmc::SamplingOptions;
use keanu_algorithms::mcmc::chains::sample_chains;
use keanu_algorithms::mcmc::gibbs::Gibbs;
use keanu_algorithms::mcmc::nuts::Nuts;
use keanu_algorithms::mcmc::select::Sampler;
use keanu_algorithms::{ChainSamples, NetworkSamples};
use keanu_distributions::KeanuRng;
use keanu_graph::op::DistributionKind as K;
use keanu_graph::{DoubleVar, GraphBuilder, IntVar, Model, NodeId, State, Value};
use keanu_tensor::{BoolTensor, DoubleTensor, IntTensor};
use mcmc_common::*;

/// Asserts that `truth` is inside the 90% posterior interval of `id` and the chains converged.
fn assert_recovers(chains: &ChainSamples, id: NodeId, truth: f64, label: &str) {
    let summary = chains.summarize(id, &[]).unwrap();
    assert!(summary.r_hat < 1.02, "{label}: R-hat {}", summary.r_hat);
    assert!(
        summary.q5 < truth && truth < summary.q95,
        "{label}: {truth} outside the 90% interval [{}, {}] (mean {})",
        summary.q5,
        summary.q95,
        summary.mean
    );
}

/// Draws a sample of `shape` from a node of `kind` with constant parameters.
fn simulate(kind: K, parameters: &[f64], shape: &[usize], seed: u64) -> Value {
    let b = GraphBuilder::new();
    let ids: Vec<NodeId> = parameters.iter().map(|&p| b.scalar(p).id()).collect();
    let node = b.add_distribution(kind, Some(shape), &ids).unwrap();
    let (graph, mut state) = b.build().unwrap();
    state
        .lazy_eval(&graph, &[node], &mut KeanuRng::seed_from_u64(seed))
        .unwrap();
    state.value(node).unwrap().clone()
}

fn nuts_chains(model: &Model, state: &State, nodes: &[NodeId], seed: u64) -> ChainSamples {
    let nuts = Nuts::default().with_adapt_count(400);
    let options = SamplingOptions::default().with_drop(400);
    sample_chains(4, seed, state, |state, rng| {
        nuts.sample_with_statistics(model, state, nodes, 900, options, rng)
            .map(|(samples, _)| samples)
    })
    .unwrap()
}

#[test]
fn nuts_recovers_a_zero_inflated_poissons_psi_and_rate() {
    let (rate, psi) = (3.2, 0.35);
    let data = simulate(
        K::ZeroInflated(std::sync::Arc::new(K::Poisson)),
        &[rate, psi],
        &[400],
        3,
    );
    let b = GraphBuilder::new();
    let lambda = b.gamma(5.0, 1.0).label("lambda");
    let structural = b.beta(1.0, 1.0).label("psi");
    let y: IntVar = b.zero_inflated_with_shape(&[400], K::Poisson, &[lambda.id()], structural);
    b.observe(y, data);
    let (lambda, structural) = (lambda.id(), structural.id());
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let chains = nuts_chains(&model, &state, &[lambda, structural], 11);
    assert_recovers(&chains, lambda, rate, "lambda");
    assert_recovers(&chains, structural, psi, "psi");
}

#[test]
fn nuts_recovers_a_hurdle_gamma_on_continuous_data_with_zeros() {
    let (theta, shape, psi) = (1.5, 2.5, 0.25);
    let data = simulate(
        K::Hurdle(std::sync::Arc::new(K::Gamma)),
        &[theta, shape, psi],
        &[400],
        7,
    );
    let zeros = data
        .as_f64()
        .unwrap()
        .as_slice()
        .iter()
        .filter(|&&x| x == 0.0)
        .count();
    assert!(zeros > 60 && zeros < 140, "{zeros} zeros");
    let b = GraphBuilder::new();
    let scale = b.half_gaussian(3.0).label("theta");
    let k = b.half_gaussian(5.0).label("k");
    let dry = b.beta(1.0, 1.0).label("psi");
    let y: DoubleVar = b.hurdle_with_shape(&[400], K::Gamma, &[scale.id(), k.id()], dry);
    b.observe(y, data);
    let (scale, k, dry) = (scale.id(), k.id(), dry.id());
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let chains = nuts_chains(&model, &state, &[scale, k, dry], 13);
    assert_recovers(&chains, scale, theta, "theta");
    assert_recovers(&chains, k, shape, "k");
    assert_recovers(&chains, dry, psi, "psi");
}

/// The posterior probability of each of `values` of `id`, by evaluating the joint log probability at each.
fn exact_posterior(model: &Model, state: &State, id: NodeId, values: &[Value]) -> Vec<f64> {
    let log_probs: Vec<f64> = values
        .iter()
        .map(|value| {
            let mut at = state.clone();
            model
                .set_and_cascade(&mut at, [(id, value.clone())])
                .unwrap();
            model.log_prob(&at).unwrap()
        })
        .collect();
    let max = log_probs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = log_probs.iter().map(|lp| (lp - max).exp()).collect();
    let total: f64 = weights.iter().sum();
    weights.iter().map(|w| w / total).collect()
}

fn frequencies(samples: &NetworkSamples, id: NodeId, values: &[i64]) -> Vec<f64> {
    let draws: Vec<i64> = match samples.i64(id) {
        Some(series) => series.as_slice().iter().map(|t| t.as_slice()[0]).collect(),
        None => samples
            .bool(id)
            .unwrap()
            .as_slice()
            .iter()
            .map(|t| i64::from(t.as_slice()[0]))
            .collect(),
    };
    values
        .iter()
        .map(|v| draws.iter().filter(|&&d| d == *v).count() as f64 / draws.len() as f64)
        .collect()
}

#[test]
fn gibbs_samples_count_and_logit_latents_exactly() {
    // a beta-binomial count, a zero-inflated binomial and a Bernoulli-logit indicator, each with an observed
    // Gaussian child
    let b = GraphBuilder::new();
    let count = b.beta_binomial(6, 1.5, 2.5);
    let y = b.gaussian(count.to_double() * 0.5, 0.8);
    b.observe(y, DoubleTensor::scalar(1.4));
    let inflated: IntVar = b.zero_inflated(
        K::Binomial,
        &[b.scalar(0.6).id(), b.int_scalar(4).id()],
        0.3,
    );
    let w = b.gaussian(inflated.to_double(), 1.0);
    b.observe(w, DoubleTensor::scalar(0.4));
    let flag = b.bernoulli_logit(-0.4);
    let u = b.gaussian(b.where_(flag, b.scalar(1.5), b.scalar(-0.5)), 1.0);
    b.observe(u, DoubleTensor::scalar(0.9));
    let (count, inflated, flag) = (count.id(), inflated.id(), flag.id());
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    assert!(matches!(
        Sampler::for_model(&model, &state),
        Sampler::Gibbs(_)
    ));
    let samples = Gibbs::new(&[count, inflated, flag])
        .sample(
            &model,
            &mut state,
            &[count, inflated, flag],
            20_000,
            &mut rng(4),
        )
        .unwrap();
    let int = |k: i64| Value::I64(IntTensor::scalar(k));
    for (id, values, label) in [
        (count, (0..=6).collect::<Vec<_>>(), "beta-binomial"),
        (inflated, (0..=4).collect(), "zero-inflated binomial"),
    ] {
        let exact = exact_posterior(
            &model,
            &state,
            id,
            &values.iter().map(|&k| int(k)).collect::<Vec<_>>(),
        );
        for (f, p) in frequencies(&samples, id, &values).iter().zip(&exact) {
            assert!((f - p).abs() < 0.015, "{label}: frequency {f}, exact {p}");
        }
    }
    let exact = exact_posterior(
        &model,
        &state,
        flag,
        &[
            BoolTensor::scalar(false).into(),
            BoolTensor::scalar(true).into(),
        ],
    );
    let f = frequencies(&samples, flag, &[0, 1]);
    assert!(
        (f[1] - exact[1]).abs() < 0.015,
        "Bernoulli logit: {f:?} vs {exact:?}"
    );
}
