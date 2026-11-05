//! Every sampler on the same analytic posteriors, sampler selection, parallel chains, drop and down-sampling, and unknown nodes.

mod mcmc_common;

use keanu_algorithms::mcmc::chains::run_chains;
use keanu_algorithms::mcmc::metropolis::{MetropolisHastings, VariableSelector};
use keanu_algorithms::mcmc::nuts::Nuts;
use keanu_algorithms::mcmc::proposal::{GaussianProposal, ProposalDistribution};
use keanu_algorithms::mcmc::select::{Sampler, is_differentiable_wrt_latents};
use keanu_algorithms::mcmc::{SamplingOptions, generate};
use keanu_algorithms::statistics::effective_sample_size;
use keanu_algorithms::{AlgorithmError, NetworkSamples};
use keanu_graph::{GraphBuilder, Model, NodeId};
use keanu_tensor::{DoubleTensor, IntTensor};
use mcmc_common::*;

#[test]
fn samplers_find_the_beta_binomial_posterior() {
    let (model, state, p, m, v) = beta_binomial_model();
    let options = SamplingOptions::default()
        .with_drop(1000)
        .with_down_sample_interval(1);
    let mh = MetropolisHastings::default()
        .sample_with(
            &model,
            &mut state.clone(),
            &[p],
            30_000,
            options,
            &mut rng(5),
        )
        .unwrap();
    assert_moments(&series(&mh, p), m, v, "Beta-Binomial MH");
    let nuts = Nuts::default()
        .with_adapt_count(500)
        .sample_with_statistics(&model, &mut state.clone(), &[p], 5000, options, &mut rng(6))
        .unwrap()
        .0;
    assert_moments(&series(&nuts, p), m, v, "Beta-Binomial NUTS");
}

/// rate ~ Gamma(theta = 1.5, k = 2), x_i ~ Poisson(rate) observed: the posterior is
/// Gamma(k + Σx, theta / (1 + n theta)).
#[test]
fn samplers_find_the_gamma_poisson_posterior() {
    let counts = [1i64, 4, 2, 0, 3];
    let b = GraphBuilder::new();
    let rate = b.gamma(1.5, 2.0);
    let x = b.poisson_with_shape(&[counts.len()], rate);
    b.observe(x, IntTensor::vector(counts.to_vec()));
    b.set_value(rate, DoubleTensor::scalar(1.0));
    let rate = rate.id();
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let k = 2.0 + counts.iter().sum::<i64>() as f64;
    let theta = 1.5 / (1.0 + counts.len() as f64 * 1.5);
    let (m, v) = (k * theta, k * theta * theta);

    let options = SamplingOptions::default()
        .with_drop(1000)
        .with_down_sample_interval(1);
    let mh = MetropolisHastings::default()
        .sample_with(
            &model,
            &mut state.clone(),
            &[rate],
            30_000,
            options,
            &mut rng(7),
        )
        .unwrap();
    assert_moments(&series(&mh, rate), m, v, "Gamma-Poisson MH");
    let nuts = Nuts::default()
        .with_adapt_count(500)
        .sample_with_statistics(
            &model,
            &mut state.clone(),
            &[rate],
            5000,
            options,
            &mut rng(8),
        )
        .unwrap()
        .0;
    assert_moments(&series(&nuts, rate), m, v, "Gamma-Poisson NUTS");
}

/// x ~ N(0, 1), y ~ N(ρx, 1 - ρ²): a correlated 2D Gaussian with unit variances and correlation ρ.
#[test]
fn samplers_find_a_correlated_two_dimensional_gaussian() {
    let rho = 0.8;
    let b = GraphBuilder::new();
    let x = b.gaussian(0.0, 1.0);
    let y = b.gaussian(x * rho, (1.0f64 - rho * rho).sqrt());
    b.set_value(x, DoubleTensor::scalar(0.0));
    b.set_value(y, DoubleTensor::scalar(0.0));
    let (x, y) = (x.id(), y.id());
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();

    let check = |samples: &NetworkSamples, label: &str| {
        let (xs, ys) = (series(samples, x), series(samples, y));
        assert_moments(&xs, 0.0, 1.0, label);
        assert_moments(&ys, 0.0, 1.0, label);
        let (mx, my) = (mean(&xs), mean(&ys));
        let cov = xs
            .iter()
            .zip(&ys)
            .map(|(a, b)| (a - mx) * (b - my))
            .sum::<f64>()
            / xs.len() as f64;
        let ess = effective_sample_size(&xs).min(effective_sample_size(&ys));
        assert!(
            (cov - rho).abs() < 5.0 / ess.sqrt(),
            "{label}: covariance {cov} (ess {ess})"
        );
    };
    let options = SamplingOptions::default()
        .with_drop(1000)
        .with_down_sample_interval(1);
    let mh = MetropolisHastings::default()
        .proposal(GaussianProposal::new(0.7))
        .selector(VariableSelector::Full)
        .sample_with(
            &model,
            &mut state.clone(),
            &[x, y],
            60_000,
            options,
            &mut rng(9),
        )
        .unwrap();
    check(&mh, "2D MH");
    let nuts = Nuts::default()
        .with_adapt_count(500)
        .sample_with_statistics(
            &model,
            &mut state.clone(),
            &[x, y],
            6000,
            options,
            &mut rng(10),
        )
        .unwrap()
        .0;
    check(&nuts, "2D NUTS");
}

#[test]
fn sampler_selection_uses_nuts_for_differentiable_models() {
    let (model, state, ..) = gaussian_model();
    assert!(is_differentiable_wrt_latents(&model, &state));
    assert!(matches!(
        Sampler::for_model(&model, &state),
        Sampler::Nuts(_)
    ));

    // a latent behind a floor is not differentiable
    let b = GraphBuilder::new();
    let a = b.gaussian(0.0, 1.0);
    let x = b.gaussian(a.floor(), 1.0);
    b.observe(x, DoubleTensor::scalar(1.0));
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    assert!(!is_differentiable_wrt_latents(&model, &state));

    // but a floor of constants and observations is fine
    let b = GraphBuilder::new();
    let a = b.gaussian(0.0, 1.0);
    let c = b.constant(DoubleTensor::scalar(2.7)).floor();
    let x = b.gaussian(a + c, 1.0);
    b.observe(x, DoubleTensor::scalar(1.0));
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    assert!(is_differentiable_wrt_latents(&model, &state));
}

#[test]
fn chains_are_reproducible_and_independent() {
    let (model, state, mu, ..) = gaussian_model();
    let nuts = Nuts::default().with_adapt_count(50);
    let run = || {
        run_chains(4, 99, &state, |state, rng| {
            Ok(series(&nuts.sample(&model, state, &[mu], 100, rng)?, mu))
        })
        .unwrap()
    };
    let first = run();
    assert_eq!(first, run(), "the same seed gives the same chains");
    for i in 0..first.len() {
        for j in i + 1..first.len() {
            assert_ne!(first[i], first[j], "chains {i} and {j} are identical");
        }
    }

    // the same seed gives the same single chain, a different seed a different one
    let mh = MetropolisHastings::default();
    let a = mh
        .sample(&model, &mut state.clone(), &[mu], 500, &mut rng(1))
        .unwrap();
    let b = mh
        .sample(&model, &mut state.clone(), &[mu], 500, &mut rng(1))
        .unwrap();
    let c = mh
        .sample(&model, &mut state.clone(), &[mu], 500, &mut rng(2))
        .unwrap();
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn drop_and_down_sampling() {
    let (model, mut state, mu, ..) = gaussian_model();
    let mut r = rng(19);
    let mh = MetropolisHastings::default();
    let mut sampler = mh.sampler(&model, &mut state, &[mu], &mut r).unwrap();
    let samples = generate(
        &mut sampler,
        &mut state,
        100,
        SamplingOptions::default()
            .with_drop(10)
            .with_down_sample_interval(3),
        &mut r,
    )
    .unwrap();
    assert_eq!(samples.len(), 30);
    assert!(sampler.log_prob().is_finite());
    assert!(
        generate(
            &mut sampler,
            &mut state,
            10,
            SamplingOptions::default()
                .with_drop(10)
                .with_down_sample_interval(1),
            &mut r
        )
        .is_err()
    );
}

#[test]
fn unknown_nodes_are_errors_not_panics() {
    use keanu_graph::GraphError;
    let unknown = NodeId::from_index(1000);
    let is_unknown = |e: AlgorithmError| matches!(e, AlgorithmError::Graph(GraphError::UnknownNode(id)) if id == unknown);

    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 1.0);
    let x = b.gaussian(mu, 1.0);
    b.observe(x, DoubleTensor::scalar(0.5));
    let mu = mu.id();
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let mut rng = rng(1);

    let mh = MetropolisHastings::default();
    assert!(
        mh.sampler(&model, &mut state, &[mu, unknown], &mut rng)
            .is_err_and(is_unknown)
    );
    assert!(
        mh.sample(&model, &mut state, &[unknown], 10, &mut rng)
            .is_err_and(is_unknown)
    );
    let nuts = Nuts::default();
    assert!(
        nuts.sampler(&model, &mut state, &[mu, unknown], &mut rng)
            .is_err_and(is_unknown)
    );
    assert!(
        GaussianProposal::new(1.0)
            .propose(&model, &state, &[unknown], &mut rng)
            .is_err_and(is_unknown)
    );
}
