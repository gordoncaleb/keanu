//! Helpers shared by the MCMC tests (`mcmc_*.rs`).

#![allow(dead_code)]

use keanu_algorithms::NetworkSamples;
use keanu_algorithms::statistics::effective_sample_size;
use keanu_distributions::KeanuRng;
use keanu_graph::{GraphBuilder, Model, NodeId, State};
use keanu_tensor::{DoubleTensor, IntTensor};

pub fn rng(seed: u64) -> KeanuRng {
    KeanuRng::seed_from_u64(seed)
}

pub fn series(samples: &NetworkSamples, id: NodeId) -> Vec<f64> {
    samples.f64(id).unwrap().element(&[]).unwrap()
}

pub fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

pub fn variance(xs: &[f64]) -> f64 {
    let m = mean(xs);
    xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (xs.len() as f64 - 1.0)
}

/// Checks a chain's mean and variance against the true values, within Monte Carlo error using the chain's
/// effective sample size.
pub fn assert_moments(xs: &[f64], true_mean: f64, true_variance: f64, label: &str) {
    let ess = effective_sample_size(xs).max(10.0);
    let (m, v) = (mean(xs), variance(xs));
    let mean_tolerance = 5.0 * (true_variance / ess).sqrt();
    let variance_tolerance = 6.0 * true_variance * (2.0 / ess).sqrt();
    assert!(
        (m - true_mean).abs() < mean_tolerance,
        "{label}: mean {m} vs {true_mean} (tolerance {mean_tolerance}, ess {ess})"
    );
    assert!(
        (v - true_variance).abs() < variance_tolerance,
        "{label}: variance {v} vs {true_variance} (tolerance {variance_tolerance}, ess {ess})"
    );
}

/// mu ~ N(0, 2²), x_i ~ N(mu, 1) observed. The posterior of mu is Gaussian.
pub fn gaussian_model() -> (Model, State, NodeId, f64, f64) {
    let data = [1.2, 0.7, 2.1, 1.5, 0.9];
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 2.0);
    let x = b.gaussian_with_shape(&[data.len()], mu, 1.0);
    b.observe(x, DoubleTensor::vector(data.to_vec()));
    b.set_value(mu, DoubleTensor::scalar(0.0));
    let mu = mu.id();
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let precision = 1.0 / 4.0 + data.len() as f64;
    let posterior_mean = data.iter().sum::<f64>() / precision;
    (model, state, mu, posterior_mean, 1.0 / precision)
}

/// p ~ Beta(2, 3), k_i ~ Binomial(10, p) observed: the posterior is Beta(2 + Σk, 3 + Σ(10 - k)).
pub fn beta_binomial_model() -> (Model, State, NodeId, f64, f64) {
    let counts = [3i64, 5, 2];
    let b = GraphBuilder::new();
    let p = b.beta(2.0, 3.0);
    let k = b.binomial_with_shape(&[counts.len()], p, 10);
    b.observe(k, IntTensor::vector(counts.to_vec()));
    b.set_value(p, DoubleTensor::scalar(0.5));
    let p = p.id();
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let alpha = 2.0 + counts.iter().sum::<i64>() as f64;
    let beta = 3.0 + counts.iter().map(|k| 10 - k).sum::<i64>() as f64;
    let m = alpha / (alpha + beta);
    let v = alpha * beta / ((alpha + beta).powi(2) * (alpha + beta + 1.0));
    (model, state, p, m, v)
}
