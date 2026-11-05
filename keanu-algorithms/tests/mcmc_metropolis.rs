//! Metropolis–Hastings against analytic and discrete posteriors, its proposals and rejection strategies, plus regression tests for bugs found in the Java implementation.

mod mcmc_common;

use keanu_algorithms::mcmc::metropolis::{MetropolisHastings, RejectionStrategy, VariableSelector};
use keanu_algorithms::mcmc::proposal::{AcceptanceRateTracker, GaussianProposal, PriorProposal};
use keanu_algorithms::mcmc::select::{Sampler, is_differentiable_wrt_latents};
use keanu_algorithms::mcmc::{SamplingAlgorithm, SamplingOptions, generate};
use keanu_graph::{GraphBuilder, Model};
use keanu_tensor::{DoubleTensor, IntTensor};
use mcmc_common::*;

#[test]
fn metropolis_hastings_with_prior_proposals_finds_the_gaussian_posterior() {
    let (model, mut state, mu, m, v) = gaussian_model();
    let samples = MetropolisHastings::default()
        .sample_with(
            &model,
            &mut state,
            &[mu],
            30_000,
            SamplingOptions::default()
                .with_drop(1000)
                .with_down_sample_interval(1),
            &mut rng(1),
        )
        .unwrap();
    assert_eq!(samples.len(), 29_000);
    assert_moments(&series(&samples, mu), m, v, "MH prior proposal");
}

#[test]
fn metropolis_hastings_with_gaussian_proposals_finds_the_gaussian_posterior() {
    let (model, mut state, mu, m, v) = gaussian_model();
    for rejection in [
        RejectionStrategy::RollbackToCachedValues,
        RejectionStrategy::RollbackAndCascade,
    ] {
        let samples = MetropolisHastings::default()
            .proposal(GaussianProposal::new(0.5))
            .selector(VariableSelector::Full)
            .rejection(rejection)
            .sample_with(
                &model,
                &mut state.clone(),
                &[mu],
                30_000,
                SamplingOptions::default()
                    .with_drop(1000)
                    .with_down_sample_interval(1),
                &mut rng(2),
            )
            .unwrap();
        assert_moments(&series(&samples, mu), m, v, "MH Gaussian proposal");
    }
    let _ = &mut state;
}

#[test]
fn rejection_strategies_give_identical_chains() {
    let (model, state, mu, _, _) = gaussian_model();
    let run = |rejection| {
        MetropolisHastings::default()
            .proposal(GaussianProposal::new(1.0))
            .rejection(rejection)
            .sample(&model, &mut state.clone(), &[mu], 2000, &mut rng(3))
            .unwrap()
    };
    assert_eq!(
        run(RejectionStrategy::RollbackToCachedValues),
        run(RejectionStrategy::RollbackAndCascade)
    );
}

/// Regression: proposing dependent variables together from their priors needs q(x' | x) evaluated with the
/// parents from before the proposal. a ~ N(0, 1), b ~ N(a, 1) has Var(b) = 2 and Cov(a, b) = 1; Java's original
/// correction gave Cov ≈ 0.58.
#[test]
fn joint_prior_proposals_of_dependent_variables_target_the_right_distribution() {
    let builder = GraphBuilder::new();
    let a = builder.gaussian(0.0, 1.0);
    let b = builder.gaussian(a, 1.0);
    builder.set_value(a, DoubleTensor::scalar(0.3));
    builder.set_value(b, DoubleTensor::scalar(-0.2));
    let (a, b) = (a.id(), b.id());
    let (graph, mut state) = builder.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let samples = MetropolisHastings::default()
        .proposal(PriorProposal)
        .selector(VariableSelector::Full)
        .sample(&model, &mut state, &[a, b], 60_000, &mut rng(11))
        .unwrap();
    let (xs, ys) = (series(&samples, a), series(&samples, b));
    assert_moments(&xs, 0.0, 1.0, "a");
    assert_moments(&ys, 0.0, 2.0, "b");
    let (mx, my) = (mean(&xs), mean(&ys));
    let cov = xs
        .iter()
        .zip(&ys)
        .map(|(p, q)| (p - mx) * (q - my))
        .sum::<f64>()
        / xs.len() as f64;
    assert!((cov - 1.0).abs() < 0.1, "covariance {cov}");
}

/// A discrete model where the exact posterior is easy: k ~ UniformInt[0, 4), x ~ Poisson(k + 1) observed.
/// Detailed balance means the chain's state frequencies match the posterior.
#[test]
fn metropolis_hastings_samples_a_discrete_posterior_exactly() {
    let observed = 3i64;
    let b = GraphBuilder::new();
    let k = b.uniform_int(0, 4);
    let x = b.poisson(k.to_double() + 1.0);
    b.observe(x, IntTensor::scalar(observed));
    b.set_value(k, IntTensor::scalar(0));
    let k = k.id();
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();

    assert!(!is_differentiable_wrt_latents(&model, &state));
    let sampler = Sampler::for_model(&model, &state);
    assert!(matches!(sampler, Sampler::MetropolisHastings(_)));
    let samples = sampler
        .sample(
            &model,
            &mut state,
            &[k],
            40_000,
            SamplingOptions::default(),
            &mut rng(12),
        )
        .unwrap();

    let likelihood = |mu: f64| (-mu + observed as f64 * mu.ln()).exp();
    let weights: Vec<f64> = (0..4).map(|k| likelihood(k as f64 + 1.0)).collect();
    let total: f64 = weights.iter().sum();
    let draws: Vec<i64> = samples
        .i64(k)
        .unwrap()
        .as_slice()
        .iter()
        .map(|t| t.as_slice()[0])
        .collect();
    for (value, w) in weights.iter().enumerate() {
        let frequency =
            draws.iter().filter(|&&d| d == value as i64).count() as f64 / draws.len() as f64;
        assert!(
            (frequency - w / total).abs() < 0.02,
            "P(k = {value}): {frequency} vs {}",
            w / total
        );
    }
}

#[test]
fn acceptance_rates_are_tracked() {
    let (model, mut state, mu, ..) = gaussian_model();
    let mut tracker = AcceptanceRateTracker::new();
    {
        let mh = MetropolisHastings::default().proposal(GaussianProposal::new(0.5));
        let mut r = rng(13);
        let mut sampler = mh
            .sampler(&model, &mut state, &[mu], &mut r)
            .unwrap()
            .with_listener(&mut tracker);
        generate(
            &mut sampler,
            &mut state,
            2000,
            SamplingOptions::default(),
            &mut r,
        )
        .unwrap();
    }
    let rate = tracker.acceptance_rate(mu).unwrap();
    assert!(rate > 0.2 && rate < 0.95, "acceptance rate {rate}");
}

/// Regression: when evaluating a proposal failed part way (here the Cholesky factor of a negative 1×1 matrix,
/// computed while cascading the proposed value), Metropolis–Hastings returned the error with the proposal still
/// applied. It now rolls the state back first.
#[test]
fn metropolis_hastings_rolls_back_a_proposal_that_fails() {
    for rejection in [
        RejectionStrategy::RollbackToCachedValues,
        RejectionStrategy::RollbackAndCascade,
    ] {
        let b = GraphBuilder::new();
        let x = b.gaussian_with_shape(&[1, 1], 1.0, 0.5);
        b.set_value(x, DoubleTensor::new(&[1, 1], vec![1.0]).unwrap());
        let factor = x.cholesky();
        let y = b.gaussian_with_shape(&[1, 1], factor, 1.0);
        b.observe(y, DoubleTensor::new(&[1, 1], vec![1.0]).unwrap());
        let (x, factor) = (x.id(), factor.id());
        let (graph, mut state) = b.build().unwrap();
        let model = Model::new(graph, &state).unwrap();

        let mh = MetropolisHastings::new()
            .proposal(GaussianProposal::new(2.0))
            .rejection(rejection);
        let mut r = rng(21);
        let mut sampler = mh.sampler(&model, &mut state, &[x], &mut r).unwrap();
        let mut failed = false;
        for _ in 0..100 {
            let before = (state.clone(), sampler.log_prob());
            if sampler.step(&mut state, &mut r).is_err() {
                assert_eq!(state.value(x), before.0.value(x), "{rejection:?}");
                assert_eq!(state.value(factor), before.0.value(factor), "{rejection:?}");
                assert_eq!(sampler.log_prob(), before.1);
                failed = true;
                break;
            }
        }
        assert!(failed, "{rejection:?}: no proposal went negative");
    }
}

/// A latent truncation bound that crosses the other bound makes the state impossible (log probability −∞), as a
/// uniform's bounds do, so Metropolis–Hastings rejects the proposal instead of failing. lower ~ N(0, 1) and
/// x ~ N(0, 1) truncated to [lower, 1] observed at -0.5: the posterior of lower is N(0, 1) restricted to
/// lower <= -0.5 and reweighted by 1 / P(lower <= X <= 1), and proposals of step 2 often land above 1.
#[test]
fn metropolis_hastings_rejects_a_latent_truncation_bound_that_crosses_the_other() {
    use keanu_graph::op::DistributionKind;
    let b = GraphBuilder::new();
    let lower = b.gaussian(0.0, 1.0);
    b.set_value(lower, DoubleTensor::scalar(-1.0));
    let (mu, sigma) = (b.scalar(0.0), b.scalar(1.0));
    let x: keanu_graph::DoubleVar<'_> = b.truncated(
        DistributionKind::Gaussian,
        &[mu.id(), sigma.id()],
        lower,
        1.0,
    );
    b.observe(x, DoubleTensor::scalar(-0.5));
    let lower = lower.id();
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    for rejection in [
        RejectionStrategy::RollbackToCachedValues,
        RejectionStrategy::RollbackAndCascade,
    ] {
        let samples = MetropolisHastings::default()
            .proposal(GaussianProposal::new(2.0))
            .rejection(rejection)
            .sample(&model, &mut state.clone(), &[lower], 2000, &mut rng(23))
            .unwrap_or_else(|e| panic!("{rejection:?}: {e}"));
        let draws = series(&samples, lower);
        assert!(draws.iter().all(|&l| l <= -0.5), "{rejection:?}");
    }
}
