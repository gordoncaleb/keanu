//! A log density of +∞ (a Beta with a shape below 1 at a bound of its support, reached when the interval
//! transform rounds to the bound) is an impossible point, not an infinitely likely one: samplers keep mixing and
//! find the exact moments.

mod mcmc_common;

use keanu_algorithms::mcmc::SamplingOptions;
use keanu_algorithms::mcmc::metropolis::MetropolisHastings;
use keanu_algorithms::mcmc::nuts::Nuts;
use keanu_graph::{GraphBuilder, Model, NodeId, State};
use keanu_tensor::{DoubleTensor, IntTensor};
use mcmc_common::*;

/// p ~ Beta(0.5, 0.5), with `n` successes in `n` Bernoulli trials observed when `n > 0`: the posterior is
/// Beta(0.5 + n, 0.5). Returns the model, its state, p, and the exact mean and variance.
fn beta_model(n: i64) -> (Model, State, NodeId, f64, f64) {
    let b = GraphBuilder::new();
    let p = b.beta(0.5, 0.5);
    if n > 0 {
        let k = b.binomial(p, n);
        b.observe(k, IntTensor::scalar(n));
    }
    b.set_value(p, DoubleTensor::scalar(0.5));
    let p = p.id();
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let (a, c) = (0.5 + n as f64, 0.5);
    let mean = a / (a + c);
    let variance = a * c / ((a + c).powi(2) * (a + c + 1.0));
    (model, state, p, mean, variance)
}

fn options() -> SamplingOptions {
    SamplingOptions::default()
        .with_drop(1000)
        .with_down_sample_interval(1)
}

#[test]
fn nuts_treats_an_infinite_density_at_a_bound_as_impossible() {
    for (n, seed) in [(0, 11), (50, 12)] {
        let (model, mut state, p, m, v) = beta_model(n);
        let samples = Nuts::default()
            .with_adapt_count(1000)
            .sample_with_statistics(&model, &mut state, &[p], 11_000, options(), &mut rng(seed))
            .unwrap()
            .0;
        let xs = series(&samples, p);
        assert!(
            xs.iter().all(|&x| x > 0.0 && x < 1.0),
            "n = {n}: a sample reached a bound"
        );
        assert_moments(&xs, m, v, &format!("NUTS Beta(0.5 + {n}, 0.5)"));
    }
}

#[test]
fn metropolis_hastings_treats_an_infinite_density_at_a_bound_as_impossible() {
    for (n, seed) in [(0, 13), (50, 14)] {
        let (model, mut state, p, m, v) = beta_model(n);
        let samples = MetropolisHastings::default()
            .sample_with(&model, &mut state, &[p], 60_000, options(), &mut rng(seed))
            .unwrap();
        let xs = series(&samples, p);
        assert!(
            xs.iter().all(|&x| x > 0.0 && x < 1.0),
            "n = {n}: a sample reached a bound"
        );
        assert_moments(&xs, m, v, &format!("MH Beta(0.5 + {n}, 0.5)"));
    }
}
