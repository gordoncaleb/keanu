//! The discrete and multivariate families of the distribution tail under inference: NUTS recovers a covariance
//! matrix with an inverse Wishart prior (through the covariance transform) and a von Mises–Fisher mean direction
//! (through the unit-vector transform), and Gibbs steps sample hypergeometric and ordered probit latents exactly.

mod mcmc_common;

use keanu_algorithms::ChainSamples;
use keanu_algorithms::mcmc::SamplingOptions;
use keanu_algorithms::mcmc::chains::sample_chains;
use keanu_algorithms::mcmc::gibbs::Gibbs;
use keanu_algorithms::mcmc::nuts::Nuts;
use keanu_algorithms::mcmc::select::Sampler;
use keanu_distributions as d;
use keanu_graph::{GraphBuilder, Model, NodeId, State, Value};
use keanu_tensor::{DoubleTensor, IntTensor};
use mcmc_common::*;

fn nuts_chains(model: &Model, state: &State, nodes: &[NodeId], seed: u64) -> ChainSamples {
    let nuts = Nuts::default().with_adapt_count(300);
    let options = SamplingOptions::default().with_drop(300);
    sample_chains(2, seed, state, |state, rng| {
        nuts.sample_with_statistics(model, state, nodes, 600, options, rng)
            .map(|(samples, _)| samples)
    })
    .unwrap()
}

#[test]
fn nuts_recovers_a_covariance_matrix_with_an_inverse_wishart_prior() {
    let truth = DoubleTensor::new(&[2, 2], vec![1.0, 0.6, 0.6, 2.0]).unwrap();
    let n = 300;
    let mean = DoubleTensor::zeros(&[n, 2]);
    let data = d::MultivariateGaussian::new(&mean, &truth)
        .sample(&[n, 2], &mut rng(5))
        .unwrap();
    let b = GraphBuilder::new();
    let covariance = b.inverse_wishart(4.0, b.constant(DoubleTensor::eye(2).unwrap()));
    let y = b.multivariate_gaussian(b.constant(mean), covariance);
    b.observe(y, data);
    let covariance = covariance.id();
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let chains = nuts_chains(&model, &state, &[covariance], 17);
    for (index, true_value) in [([0, 0], 1.0), ([0, 1], 0.6), ([1, 1], 2.0)] {
        let summary = chains.summarize(covariance, &index).unwrap();
        assert!(summary.r_hat < 1.05, "{index:?}: R-hat {}", summary.r_hat);
        // within the posterior's 99% range (the 90% interval widened by half again)
        let half = (summary.q95 - summary.q5) * 0.75;
        assert!(
            (summary.mean - true_value).abs() < half,
            "{index:?}: posterior mean {} (interval [{}, {}]) for {true_value}",
            summary.mean,
            summary.q5,
            summary.q95
        );
    }
    // every draw is symmetric positive definite
    let draws = chains.element(covariance, &[0, 1]).unwrap();
    let mirror = chains.element(covariance, &[1, 0]).unwrap();
    assert_eq!(draws, mirror);
}

#[test]
fn nuts_recovers_a_von_mises_fisher_mean_direction() {
    let truth = DoubleTensor::vector(vec![0.0, 0.6, 0.8]);
    let data = d::VonMisesFisher::new(&truth, DoubleTensor::scalar(8.0))
        .sample(&[200, 3], &mut rng(6))
        .unwrap();
    let b = GraphBuilder::new();
    // a uniform prior on the direction (κ = 0)
    let direction = b.von_mises_fisher(b.constant(DoubleTensor::vector(vec![1.0, 0.0, 0.0])), 0.0);
    let y = b.von_mises_fisher_with_shape(&[200, 3], direction, 8.0);
    b.observe(y, data);
    let direction = direction.id();
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let chains = nuts_chains(&model, &state, &[direction], 19);
    for (i, true_value) in truth.as_slice().iter().enumerate() {
        let summary = chains.summarize(direction, &[i]).unwrap();
        assert!(summary.r_hat < 1.05, "{i}: R-hat {}", summary.r_hat);
        assert!(
            (summary.mean - true_value).abs() < 0.05,
            "{i}: {}",
            summary.mean
        );
    }
}

/// The posterior probability of each of `values` of `id`, by evaluating the joint log probability at each.
fn exact_posterior(model: &Model, state: &State, id: NodeId, values: &[i64]) -> Vec<f64> {
    let log_probs: Vec<f64> = values
        .iter()
        .map(|&value| {
            let mut at = state.clone();
            model
                .set_and_cascade(&mut at, [(id, Value::I64(IntTensor::scalar(value)))])
                .unwrap();
            model.log_prob(&at).unwrap()
        })
        .collect();
    let max = log_probs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = log_probs.iter().map(|lp| (lp - max).exp()).collect();
    let total: f64 = weights.iter().sum();
    weights.iter().map(|w| w / total).collect()
}

#[test]
fn gibbs_samples_hypergeometric_and_ordered_probit_latents_exactly() {
    let b = GraphBuilder::new();
    let count = b.hyper_geometric(20, 7, 6);
    let y = b.gaussian(count.to_double() * 0.5, 0.8);
    b.observe(y, DoubleTensor::scalar(1.4));
    let cutpoints = b.constant(DoubleTensor::vector(vec![-0.5, 0.4, 1.2]));
    let category = b.ordered_probit(0.3, cutpoints);
    let z = b.gaussian(category.to_double(), 1.0);
    b.observe(z, DoubleTensor::scalar(2.3));
    let (count, category) = (count.id(), category.id());
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    assert!(matches!(
        Sampler::for_model(&model, &state),
        Sampler::Gibbs(_)
    ));
    let samples = Gibbs::new(&[count, category])
        .sample(&model, &mut state, &[count, category], 20_000, &mut rng(8))
        .unwrap();
    for (id, values, label) in [
        (count, (0..=6).collect::<Vec<_>>(), "hypergeometric"),
        (category, (0..=3).collect(), "ordered probit"),
    ] {
        let exact = exact_posterior(&model, &state, id, &values);
        let draws: Vec<i64> = samples
            .i64(id)
            .unwrap()
            .as_slice()
            .iter()
            .map(|t| t.as_slice()[0])
            .collect();
        for (v, p) in values.iter().zip(&exact) {
            let f = draws.iter().filter(|&&d| d == *v).count() as f64 / draws.len() as f64;
            assert!(
                (f - p).abs() < 0.015,
                "{label} {v}: frequency {f}, exact {p}"
            );
        }
    }
}
