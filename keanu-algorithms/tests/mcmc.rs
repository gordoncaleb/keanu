//! Metropolis–Hastings, NUTS and simulated annealing against analytic posteriors, plus regression tests for bugs
//! found in the Java implementation.

use keanu_algorithms::mcmc::annealing::{AnnealingSchedule, SimulatedAnnealing};
use keanu_algorithms::mcmc::chains::run_chains;
use keanu_algorithms::mcmc::metropolis::{MetropolisHastings, RejectionStrategy, VariableSelector};
use keanu_algorithms::mcmc::nuts::{Nuts, QuadraticPotential};
use keanu_algorithms::mcmc::proposal::{
    AcceptanceRateTracker, GaussianProposal, PriorProposal, Proposal, ProposalDistribution,
};
use keanu_algorithms::mcmc::select::{Sampler, is_differentiable_wrt_latents};
use keanu_algorithms::mcmc::{SamplingAlgorithm, SamplingOptions, generate};
use keanu_algorithms::statistics::effective_sample_size;
use keanu_algorithms::{AlgorithmError, NetworkSamples, diagnostics};
use keanu_distributions::KeanuRng;
use keanu_graph::{GraphBuilder, Model, NodeId, Space, State};
use keanu_tensor::{DoubleTensor, IntTensor};
use std::sync::{Arc, Mutex};

fn rng(seed: u64) -> KeanuRng {
    KeanuRng::seed_from_u64(seed)
}

fn series(samples: &NetworkSamples, id: NodeId) -> Vec<f64> {
    samples.f64(id).unwrap().element(&[]).unwrap()
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn variance(xs: &[f64]) -> f64 {
    let m = mean(xs);
    xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (xs.len() as f64 - 1.0)
}

/// Checks a chain's mean and variance against the true values, within Monte Carlo error using the chain's
/// effective sample size.
fn assert_moments(xs: &[f64], true_mean: f64, true_variance: f64, label: &str) {
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
fn gaussian_model() -> (Model, State, NodeId, f64, f64) {
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

#[test]
fn nuts_finds_the_gaussian_posterior() {
    let (model, mut state, mu, m, v) = gaussian_model();
    let nuts = Nuts::default().with_adapt_count(500);
    let samples = nuts
        .sample_with_statistics(
            &model,
            &mut state,
            &[mu],
            5000,
            SamplingOptions::default()
                .with_drop(500)
                .with_down_sample_interval(1),
            &mut rng(4),
        )
        .unwrap()
        .0;
    assert_moments(&series(&samples, mu), m, v, "NUTS");
}

/// p ~ Beta(2, 3), k_i ~ Binomial(10, p) observed: the posterior is Beta(2 + Σk, 3 + Σ(10 - k)).
fn beta_binomial_model() -> (Model, State, NodeId, f64, f64) {
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

/// A latent N(0, 1) truncated to [a, b] (either bound may be infinite): NUTS stays inside the bounds through the
/// truncation's support transform and finds the truncated mean and variance.
#[test]
fn nuts_keeps_a_truncated_latent_inside_its_bounds() {
    use keanu_graph::DoubleVar;
    use keanu_graph::op::DistributionKind;
    let phi = |x: f64| {
        if x.is_finite() {
            (-x * x / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt()
        } else {
            0.0
        }
    };
    let x_phi = |x: f64| if x.is_finite() { x * phi(x) } else { 0.0 };
    let cdf = keanu_tensor::special::ndtr;
    for (a, b, seed) in [(0.5, 2.0, 21), (-f64::INFINITY, -1.0, 22)] {
        let builder = GraphBuilder::new();
        let (mu, sigma) = (builder.scalar(0.0), builder.scalar(1.0));
        let x: DoubleVar =
            builder.truncated(DistributionKind::Gaussian, &[mu.id(), sigma.id()], a, b);
        let start = if b.is_finite() && a.is_finite() {
            (a + b) / 2.0
        } else {
            b - 1.0
        };
        builder.set_value(x, DoubleTensor::scalar(start));
        let x = x.id();
        let (graph, mut state) = builder.build().unwrap();
        let model = Model::new(graph, &state).unwrap();
        let samples = Nuts::default()
            .with_adapt_count(500)
            .sample_with_statistics(
                &model,
                &mut state,
                &[x],
                4000,
                SamplingOptions::default()
                    .with_drop(500)
                    .with_down_sample_interval(1),
                &mut rng(seed),
            )
            .unwrap()
            .0;
        let xs = series(&samples, x);
        assert!(
            xs.iter().all(|&x| x > a && x < b),
            "[{a}, {b}]: a sample left the bounds"
        );
        let z = cdf(b) - cdf(a);
        let m = (phi(a) - phi(b)) / z;
        let v = 1.0 + (x_phi(a) - x_phi(b)) / z - m * m;
        assert_moments(&xs, m, v, &format!("truncated [{a}, {b}]"));
    }
}

/// A vector latent N(0, 1) truncated element-wise to [0, ∞) and (-∞, 1]: the elements have different bound kinds,
/// and NUTS still samples it in unconstrained space and finds each element's truncated moments.
#[test]
fn nuts_samples_a_truncated_vector_latent_with_mixed_bounds() {
    use keanu_graph::DoubleVar;
    use keanu_graph::op::DistributionKind;
    let phi = |x: f64| {
        if x.is_finite() {
            (-x * x / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt()
        } else {
            0.0
        }
    };
    let x_phi = |x: f64| if x.is_finite() { x * phi(x) } else { 0.0 };
    let cdf = keanu_tensor::special::ndtr;
    let (lower, upper) = ([0.0, f64::NEG_INFINITY], [f64::INFINITY, 1.0]);
    let builder = GraphBuilder::new();
    let (mu, sigma) = (builder.scalar(0.0), builder.scalar(1.0));
    let x: DoubleVar = builder.truncated(
        DistributionKind::Gaussian,
        &[mu.id(), sigma.id()],
        DoubleTensor::vector(lower.to_vec()),
        DoubleTensor::vector(upper.to_vec()),
    );
    builder.set_value(x, DoubleTensor::vector(vec![0.5, 0.0]));
    let x = x.id();
    let (graph, mut state) = builder.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let samples = Nuts::default()
        .with_adapt_count(500)
        .sample_with_statistics(
            &model,
            &mut state,
            &[x],
            4000,
            SamplingOptions::default()
                .with_drop(500)
                .with_down_sample_interval(1),
            &mut rng(23),
        )
        .unwrap()
        .0;
    let chain = samples.f64(x).unwrap();
    for i in 0..2 {
        let (a, b) = (lower[i], upper[i]);
        let xs = chain.element(&[i]).unwrap();
        assert!(xs.iter().all(|&x| x > a && x < b), "[{a}, {b}]");
        let z = cdf(b) - cdf(a);
        let m = (phi(a) - phi(b)) / z;
        let v = 1.0 + (x_phi(a) - x_phi(b)) / z - m * m;
        assert_moments(&xs, m, v, &format!("element {i} truncated to [{a}, {b}]"));
    }
}

/// A von Mises latent whose mean direction is near the ±π seam: NUTS moves through its circular transform across
/// the seam and recovers the circular mean and the mean resultant length I₁(κ)/I₀(κ).
#[test]
fn nuts_recovers_the_circular_mean_of_a_von_mises_latent() {
    let (mu, kappa) = (std::f64::consts::PI - 0.1, 10.0);
    let b = GraphBuilder::new();
    let theta = b.von_mises(mu, kappa);
    b.set_value(theta, DoubleTensor::scalar(0.0));
    let theta = theta.id();
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let samples = Nuts::default()
        .with_adapt_count(500)
        .sample_with_statistics(
            &model,
            &mut state,
            &[theta],
            4000,
            SamplingOptions::default()
                .with_drop(500)
                .with_down_sample_interval(1),
            &mut rng(23),
        )
        .unwrap()
        .0;
    let xs = series(&samples, theta);
    let cos: Vec<f64> = xs.iter().map(|x| x.cos()).collect();
    let sin: Vec<f64> = xs.iter().map(|x| x.sin()).collect();
    let circular_mean = mean(&sin).atan2(mean(&cos));
    // the angular distance, so a mean on the other side of the seam counts as close
    let error = (circular_mean - mu)
        .sin()
        .atan2((circular_mean - mu).cos())
        .abs();
    let deviations: Vec<f64> = xs.iter().map(|x| (x - mu).sin()).collect();
    let ess = effective_sample_size(&deviations).max(10.0);
    // the circular standard error is about sqrt(E[sin²(x - mu)] / ess) / R
    let resultant = (keanu_tensor::special::log_bessel_i1(kappa)
        - keanu_tensor::special::log_bessel_i0(kappa))
    .exp();
    let tolerance = 5.0 * (variance(&deviations) / ess).sqrt() / resultant;
    assert!(
        error < tolerance,
        "circular mean {circular_mean} vs {mu} (tolerance {tolerance})"
    );
    let r = mean(&cos).hypot(mean(&sin));
    assert!((r - resultant).abs() < 0.02, "resultant {r} vs {resultant}");
    // some samples fall on each side of the seam
    let (lo, hi) = xs
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| {
            (a.min(x), b.max(x))
        });
    assert!(lo < 0.0 && hi > 0.0, "{lo} {hi}");
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

/// Regression: a NUTS configuration (in Java, its adaptive potential) could not be reused for a second run.
#[test]
fn nuts_can_be_reused() {
    let (model, state, mu, ..) = gaussian_model();
    let nuts = Nuts::default().with_adapt_count(100);
    let mut r = rng(14);
    nuts.sample(&model, &mut state.clone(), &[mu], 200, &mut r)
        .unwrap();
    let second = nuts
        .sample(&model, &mut state.clone(), &[mu], 200, &mut r)
        .unwrap();
    let values = series(&second, mu);
    assert!(values.iter().all(|v| v.is_finite()));
    let mut distinct = values.clone();
    distinct.sort_by(f64::total_cmp);
    distinct.dedup();
    assert!(
        distinct.len() > 50,
        "only {} distinct values",
        distinct.len()
    );
}

/// Regression: the first sample after adaptation must already use the averaged step size, which then stays fixed.
#[test]
fn nuts_switches_to_the_averaged_step_size_when_adaptation_ends() {
    let (model, mut state, mu, ..) = gaussian_model();
    let adapt_count = 50;
    let nuts = Nuts::default()
        .with_adapt_count(adapt_count)
        .with_save_statistics(true);
    let (_, statistics) = nuts
        .sample_with_statistics(
            &model,
            &mut state,
            &[mu],
            80,
            SamplingOptions::default(),
            &mut rng(15),
        )
        .unwrap();
    let after = &statistics.step_size[adapt_count..];
    assert!(
        after.iter().all(|&s| s == after[0]),
        "step sizes after adaptation: {after:?}"
    );
    assert_ne!(statistics.step_size[adapt_count - 1], after[0]);
    assert_eq!(statistics.tree_size.len(), 80);
    assert_eq!(statistics.log_prob.len(), 80);
    assert!(
        statistics
            .mean_tree_accept
            .iter()
            .all(|a| (0.0..=1.0).contains(a))
    );
    // as documented: each entry is the previous tree's acceptance (none before the first), frozen once adaptation
    // ends at the last adapted tree's (step adapt_count - 1, recorded at step adapt_count)
    let accept = &statistics.mean_tree_accept;
    assert_eq!(accept[0], 0.0);
    assert_ne!(accept[1], 0.0);
    assert!(
        accept[adapt_count..]
            .iter()
            .all(|&a| a == accept[adapt_count]),
        "{:?}",
        &accept[adapt_count..]
    );
    assert_ne!(accept[adapt_count - 1], accept[adapt_count]);
}

#[test]
fn nuts_records_divergent_steps_after_adaptation() {
    let (model, mut state, mu, ..) = gaussian_model();
    let nuts = Nuts::default()
        .with_adapt_count(0)
        .with_adapt_step_size(false)
        .with_initial_step_size(5.0)
        .with_max_energy_change(1e-3);
    let (_, statistics) = nuts
        .sample_with_statistics(
            &model,
            &mut state,
            &[mu],
            50,
            SamplingOptions::default(),
            &mut rng(16),
        )
        .unwrap();
    assert!(!statistics.divergent_steps.is_empty());
}

#[test]
fn nuts_rejects_invalid_settings_and_discrete_latents() {
    let (model, mut state, mu, ..) = gaussian_model();
    let bad = Nuts::default().with_max_tree_height(0);
    assert!(matches!(
        bad.sampler(&model, &mut state, &[mu], &mut rng(1)),
        Err(AlgorithmError::InvalidSetting(_))
    ));

    let b = GraphBuilder::new();
    let k = b.uniform_int(0, 4);
    let x = b.poisson(k.to_double() + 1.0);
    b.observe(x, IntTensor::scalar(2));
    let k = k.id();
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    assert!(matches!(
        Nuts::default().sampler(&model, &mut state, &[k], &mut rng(1)),
        Err(AlgorithmError::DiscreteLatents(_))
    ));
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
fn simulated_annealing_finds_the_map() {
    let (model, mut state, mu, m, _) = gaussian_model();
    let annealing =
        SimulatedAnnealing::new(MetropolisHastings::default().proposal(GaussianProposal::new(0.2)));
    let estimate = annealing
        .max_a_posteriori(&model, &mut state, 5000, &mut rng(17))
        .unwrap();
    let value = estimate
        .values
        .iter()
        .find(|(id, _)| *id == mu)
        .unwrap()
        .1
        .as_f64()
        .unwrap()
        .as_slice()[0];
    assert!((value - m).abs() < 0.05, "MAP {value} vs {m}");
    assert!(estimate.log_prob >= model.log_prob(&state).unwrap());
    assert!(AnnealingSchedule::exponential(10, 2.0, 0.01).temperature(10) - 0.01 < 1e-12);
}

/// A proposal that records how many nodes each proposal was for.
#[derive(Debug, Default)]
struct CountingProposal {
    sizes: Mutex<Vec<usize>>,
}

impl ProposalDistribution for CountingProposal {
    fn propose(
        &self,
        model: &Model,
        state: &State,
        nodes: &[NodeId],
        rng: &mut KeanuRng,
    ) -> keanu_algorithms::Result<Proposal> {
        self.sizes.lock().unwrap().push(nodes.len());
        GaussianProposal::new(0.1).propose(model, state, nodes, rng)
    }
    fn log_prob_from_given_to(
        &self,
        _: &Model,
        _: &State,
        _: &Proposal,
    ) -> keanu_algorithms::Result<f64> {
        Ok(0.0)
    }
    fn log_prob_to_given_from(
        &self,
        _: &Model,
        _: &State,
        _: &Proposal,
    ) -> keanu_algorithms::Result<f64> {
        Ok(0.0)
    }
}

/// Regression: Java's simulated annealing ignored the configured variable selector.
#[test]
fn simulated_annealing_honours_the_variable_selector() {
    let b = GraphBuilder::new();
    let x = b.gaussian(0.0, 1.0);
    let y = b.gaussian(0.0, 1.0);
    b.set_value(x, DoubleTensor::scalar(0.5));
    b.set_value(y, DoubleTensor::scalar(0.5));
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();

    let proposal = Arc::new(CountingProposal::default());
    let mh = MetropolisHastings::default()
        .shared_proposal(proposal.clone())
        .selector(VariableSelector::Full);
    SimulatedAnnealing::new(mh)
        .max_a_posteriori(&model, &mut state, 20, &mut rng(18))
        .unwrap();
    assert!(proposal.sizes.lock().unwrap().iter().all(|&n| n == 2));
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

/// Regression: a zero initial weight or a non-positive initial variance gave a zero mass matrix entry, so
/// momentum / mass was NaN and the chain never moved.
#[test]
fn nuts_rejects_a_degenerate_initial_potential() {
    let (model, mut state, mu, ..) = gaussian_model();
    for potential in [
        QuadraticPotential::default().with_initial_weight(0.0),
        QuadraticPotential::default().with_initial_weight(f64::NAN),
        QuadraticPotential::default().with_initial_weight(f64::INFINITY),
        QuadraticPotential::default().with_initial_variance(0.0),
        QuadraticPotential::default().with_initial_variance(-1.0),
        QuadraticPotential::default().with_initial_variance(f64::NAN),
    ] {
        let nuts = Nuts::default().with_potential(potential);
        assert!(
            matches!(
                nuts.sampler(&model, &mut state, &[mu], &mut rng(1)),
                Err(AlgorithmError::InvalidSetting(_))
            ),
            "{potential:?}"
        );
    }
}

/// Regression: a schedule over zero iterations gave NaN temperatures, and non-positive temperatures weren't
/// rejected.
#[test]
fn simulated_annealing_rejects_invalid_schedules() {
    let (model, mut state, ..) = gaussian_model();
    let annealing = SimulatedAnnealing::default();
    for schedule in [
        AnnealingSchedule::exponential(0, 2.0, 0.01),
        AnnealingSchedule::exponential(10, 0.0, 0.01),
        AnnealingSchedule::exponential(10, 2.0, -1.0),
        AnnealingSchedule::exponential(10, f64::NAN, 0.01),
        AnnealingSchedule::exponential(10, 2.0, f64::INFINITY),
    ] {
        assert!(
            matches!(
                annealing.max_a_posteriori_with(&model, &mut state, 5, schedule, &mut rng(1)),
                Err(AlgorithmError::InvalidSetting(_))
            ),
            "{schedule:?}"
        );
    }
    // no iterations with the default schedule just returns the start
    assert!(
        annealing
            .max_a_posteriori(&model, &mut state, 0, &mut rng(1))
            .is_ok()
    );
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

/// mu ~ N(0, 3²) (two elements), sigma ~ Gamma(theta = 1, k = 2), x ~ N(mu, sigma) observed: a model with a
/// vector latent and a positive one, for checking chains step for step.
fn mixed_support_model() -> (Model, State, NodeId, NodeId) {
    let b = GraphBuilder::new();
    let mu = b.gaussian_with_shape(&[2], 0.0, 3.0);
    let sigma = b.gamma(1.0, 2.0);
    let x = b.gaussian_with_shape(&[2], mu, sigma);
    b.observe(x, DoubleTensor::vector(vec![0.8, -1.3]));
    b.set_value(mu, DoubleTensor::vector(vec![0.1, 0.2]));
    b.set_value(sigma, DoubleTensor::scalar(1.5));
    let (mu, sigma) = (mu.id(), sigma.id());
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    (model, state, mu, sigma)
}

/// `Space::Constrained` is NUTS as it was before it moved to unconstrained space: these sums, last draws and log
/// probabilities were recorded from that implementation (seed 42, 40 adaptation steps, 80 samples) and must be
/// reproduced exactly.
#[test]
fn nuts_in_constrained_space_reproduces_the_pre_transform_chain() {
    let (model, mut state, mu, sigma) = mixed_support_model();
    let samples = Nuts::default()
        .with_adapt_count(40)
        .with_space(Space::Constrained)
        .sample(&model, &mut state, &[mu, sigma], 80, &mut rng(42))
        .unwrap();
    let mus: Vec<f64> = samples
        .f64(mu)
        .unwrap()
        .as_slice()
        .iter()
        .flat_map(|t| t.as_slice().to_vec())
        .collect();
    let sigmas = series(&samples, sigma);
    let squares =
        sigmas.iter().map(|s| s * s).sum::<f64>() + mus.iter().map(|m| m * m).sum::<f64>();
    assert_eq!(mus.iter().sum::<f64>(), -22.039102939226353);
    assert_eq!(sigmas.iter().sum::<f64>(), 181.2339512038264);
    assert_eq!(squares, 1190.4332070957182);
    assert_eq!(samples.log_probs().iter().sum::<f64>(), -793.1055813581459);
    assert_eq!(
        &mus[mus.len() - 2..],
        &[0.42542978748156246, -2.1173053684998595]
    );
    assert_eq!(*sigmas.last().unwrap(), 1.4752970588003893);
}

/// The posterior mean and variance of sigma ~ HalfGaussian(1) given y_i ~ N(0, sigma), by quadrature of
/// p(σ | y) ∝ exp(−σ²/2) σ⁻ⁿ exp(−Σy² / 2σ²) over σ > 0.
fn half_gaussian_scale_posterior(data: &[f64]) -> (f64, f64) {
    let n = data.len() as f64;
    let s: f64 = data.iter().map(|y| y * y).sum();
    let log_density =
        |sigma: f64| -sigma * sigma / 2.0 - n * sigma.ln() - s / (2.0 * sigma * sigma);
    // the trapezoidal rule on (0, 12]: the density vanishes faster than any power at both ends
    let steps = 200_000;
    let h = 12.0 / steps as f64;
    let (mut z, mut m1, mut m2) = (0.0, 0.0, 0.0);
    for i in 1..=steps {
        let sigma = i as f64 * h;
        let w = if i == steps { 0.5 } else { 1.0 } * log_density(sigma).exp();
        z += w;
        m1 += w * sigma;
        m2 += w * sigma * sigma;
    }
    let mean = m1 / z;
    (mean, m2 / z - mean * mean)
}

/// AE1: sampled in unconstrained space, a positive scale never leaves its support, no trajectory diverges, and the
/// draws match the posterior computed by quadrature.
#[test]
fn nuts_samples_a_half_gaussian_scale_inside_its_support() {
    let data = [0.4, -1.1, 0.7, 1.6, -0.3];
    let b = GraphBuilder::new();
    let sigma = b.half_gaussian(1.0);
    let y = b.gaussian_with_shape(&[data.len()], 0.0, sigma);
    b.observe(y, DoubleTensor::vector(data.to_vec()));
    b.set_value(sigma, DoubleTensor::scalar(1.0));
    let sigma = sigma.id();
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();

    let (samples, statistics) = Nuts::default()
        .with_adapt_count(500)
        .sample_with_statistics(
            &model,
            &mut state,
            &[sigma],
            5000,
            SamplingOptions::default()
                .with_drop(500)
                .with_down_sample_interval(1),
            &mut rng(11),
        )
        .unwrap();
    let draws = series(&samples, sigma);
    assert!(draws.iter().all(|&s| s > 0.0));
    assert!(
        statistics.divergent_steps.is_empty(),
        "{:?}",
        statistics.divergent_steps
    );
    // the recorded log probabilities are the model's at the constrained values, without the Jacobian
    let mut at = state.clone();
    for sample in samples.iter().take(3) {
        at.set_and_cascade(model.graph(), sigma, sample.f64(sigma).unwrap().clone())
            .unwrap();
        let expected = model.log_prob(&at).unwrap();
        assert!(
            (sample.log_prob() - expected).abs() < 1e-9,
            "{} vs {expected}",
            sample.log_prob()
        );
    }
    let (m, v) = half_gaussian_scale_posterior(&data);
    assert_moments(&draws, m, v, "HalfGaussian scale NUTS");
}

/// p ~ Dirichlet(1, 2, 3), k ~ Multinomial(20, p) observed: the posterior is Dirichlet(α + k). In constrained
/// space every move would leave the simplex; in unconstrained space every draw is on it.
#[test]
fn nuts_finds_the_dirichlet_multinomial_posterior() {
    let alpha = [1.0, 2.0, 3.0];
    let counts = [4i64, 7, 9];
    let b = GraphBuilder::new();
    let p = b.dirichlet(DoubleTensor::vector(alpha.to_vec()));
    let k = b.multinomial(counts.iter().sum::<i64>(), p);
    b.observe(k, IntTensor::vector(counts.to_vec()));
    b.set_value(p, DoubleTensor::vector(vec![1.0 / 3.0; 3]));
    let p = p.id();
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();

    let samples = Nuts::default()
        .with_adapt_count(500)
        .sample_with_statistics(
            &model,
            &mut state,
            &[p],
            4000,
            SamplingOptions::default()
                .with_drop(500)
                .with_down_sample_interval(1),
            &mut rng(12),
        )
        .unwrap()
        .0;
    let chain = samples.f64(p).unwrap();
    for draw in chain.as_slice() {
        assert!(draw.as_slice().iter().all(|&x| x > 0.0));
        assert!((draw.sum() - 1.0).abs() < 1e-12);
    }
    let posterior: Vec<f64> = alpha
        .iter()
        .zip(&counts)
        .map(|(a, &k)| a + k as f64)
        .collect();
    let total: f64 = posterior.iter().sum();
    for (i, a) in posterior.iter().enumerate() {
        let m = a / total;
        let v = a * (total - a) / (total * total * (total + 1.0));
        assert_moments(
            &chain.element(&[i]).unwrap(),
            m,
            v,
            &format!("Dirichlet p[{i}]"),
        );
    }
}

/// A start outside the support has no unconstrained point; in either space it is an impossible start.
#[test]
fn nuts_rejects_a_start_outside_the_support() {
    let b = GraphBuilder::new();
    let a = b.uniform(0.0, 1.0);
    b.set_value(a, DoubleTensor::scalar(5.0));
    let a = a.id();
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    for space in [Space::Unconstrained, Space::Constrained] {
        assert_eq!(
            Nuts::default()
                .with_space(space)
                .sampler(&model, &mut state.clone(), &[a], &mut rng(1))
                .unwrap_err(),
            AlgorithmError::ImpossibleStart,
            "{space:?}"
        );
    }
}

/// A bound that moves with another latent has no fixed transform: the error says so and suggests the opt-out,
/// which still samples.
#[test]
fn nuts_reports_a_support_that_depends_on_a_latent() {
    let b = GraphBuilder::new();
    let hi = b.uniform(1.0, 2.0);
    let x = b.uniform(0.0, hi);
    b.set_value(hi, DoubleTensor::scalar(1.5));
    b.set_value(x, DoubleTensor::scalar(0.5));
    let x = x.id();
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let error = Nuts::default()
        .sampler(&model, &mut state.clone(), &[x], &mut rng(1))
        .unwrap_err();
    assert!(error.to_string().contains("Space::Constrained"), "{error}");
    Nuts::default()
        .with_adapt_count(20)
        .with_space(Space::Constrained)
        .sample(&model, &mut state.clone(), &[x], 20, &mut rng(1))
        .unwrap();
}

/// Latent draws of a two-component Gaussian mixture whose weights are a latent Dirichlet, each observed with noise:
/// the weights don't move the mixture's support, so NUTS runs in unconstrained space and every draw of the weights
/// is on the simplex.
#[test]
fn nuts_samples_latent_mixture_weights() {
    use keanu_graph::DoubleVar;
    use keanu_graph::op::DistributionKind;
    let b = GraphBuilder::new();
    let w = b.dirichlet(DoubleTensor::vector(vec![1.0, 1.0]));
    let mu = b.constant(DoubleTensor::vector(vec![-2.0, 2.0]));
    let sigma = b.scalar(1.0);
    let data = [1.5, 2.5, -1.8, 2.2];
    let x: DoubleVar = b.mixture_with_shape(
        &[data.len()],
        DistributionKind::Gaussian,
        &[mu.id(), sigma.id()],
        w,
    );
    let y = b.gaussian(x, 0.3);
    b.observe(y, DoubleTensor::vector(data.to_vec()));
    b.set_value(w, DoubleTensor::vector(vec![0.5, 0.5]));
    b.set_value(x, DoubleTensor::vector(data.to_vec()));
    let w = w.id();
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let samples = Nuts::default()
        .with_adapt_count(100)
        .sample(&model, &mut state, &[w], 400, &mut rng(31))
        .unwrap();
    let chain = samples.f64(w).unwrap();
    for draw in chain.as_slice() {
        assert!(draw.as_slice().iter().all(|&x| x > 0.0));
        assert!((draw.sum() - 1.0).abs() < 1e-12);
    }
    // three of the four observations favour the second component
    let second = chain.element(&[1]).unwrap();
    assert!(mean(&second) > 0.5, "{}", mean(&second));
}

const SCHOOL_EFFECTS: [f64; 8] = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0];
const SCHOOL_SIGMAS: [f64; 8] = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0];

/// Eight schools (Rubin 1981): mu ~ N(0, 5), tau ~ HalfCauchy(5), theta ~ N(mu, tau), y ~ N(theta, sigma). The
/// centred form samples theta directly, giving the funnel between theta and tau; the non-centred form samples
/// eta ~ N(0, 1) and sets theta = mu + tau·eta. Returns the model, its state and the latents.
fn eight_schools(centred: bool) -> (Model, State, Vec<NodeId>) {
    let schools = SCHOOL_EFFECTS.len();
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 5.0);
    let tau = b.half_cauchy(5.0);
    b.set_value(mu, DoubleTensor::scalar(0.0));
    b.set_value(tau, DoubleTensor::scalar(1.0));
    let (theta, latents) = if centred {
        let theta = b.gaussian_with_shape(&[schools], mu, tau);
        b.set_value(theta, DoubleTensor::vector(vec![0.0; schools]));
        (theta, vec![mu.id(), tau.id(), theta.id()])
    } else {
        let eta = b.gaussian_with_shape(&[schools], 0.0, 1.0);
        b.set_value(eta, DoubleTensor::vector(vec![0.0; schools]));
        (mu + tau * eta, vec![mu.id(), tau.id(), eta.id()])
    };
    let sigma = b.constant(DoubleTensor::vector(SCHOOL_SIGMAS.to_vec()));
    let y = b.gaussian_with_shape(&[schools], theta, sigma);
    b.observe(y, DoubleTensor::vector(SCHOOL_EFFECTS.to_vec()));
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    (model, state, latents)
}

/// The divergences of four NUTS chains of eight schools after adaptation, and each chain's draw count and
/// energy series length.
fn eight_schools_divergences(centred: bool) -> (usize, Vec<(usize, usize)>) {
    let (model, state, latents) = eight_schools(centred);
    let (adapt, draws) = (500, 500);
    let nuts = Nuts::default()
        .with_adapt_count(adapt)
        .with_target_acceptance_prob(0.9);
    let chains = run_chains(4, 8, &state, |state, rng| {
        nuts.sample_with_statistics(
            &model,
            state,
            &latents,
            adapt + draws,
            SamplingOptions::default().with_drop(adapt),
            rng,
        )
    })
    .unwrap();
    let divergences = chains.iter().map(|(_, s)| s.divergence_count()).sum();
    let lengths = chains
        .iter()
        .map(|(samples, s)| (samples.len(), s.energy[adapt..].len()))
        .collect();
    (divergences, lengths)
}

#[test]
fn centred_eight_schools_diverges_and_non_centred_does_not() {
    let (centred, _) = eight_schools_divergences(true);
    assert!(centred > 0, "centred eight schools reported no divergences");
    let (non_centred, lengths) = eight_schools_divergences(false);
    assert_eq!(non_centred, 0, "non-centred eight schools diverged");
    for (draws, energies) in lengths {
        assert_eq!(draws, energies);
    }
}

#[test]
fn nuts_records_an_energy_per_step_and_flags_divergences() {
    let (model, mut state, mu, ..) = gaussian_model();
    let nuts = Nuts::default()
        .with_adapt_count(0)
        .with_adapt_step_size(false)
        .with_initial_step_size(5.0)
        .with_max_energy_change(1e-3);
    let (samples, statistics) = nuts
        .sample_with_statistics(
            &model,
            &mut state,
            &[mu],
            50,
            SamplingOptions::default(),
            &mut rng(16),
        )
        .unwrap();
    assert_eq!(statistics.energy.len(), samples.len());
    assert!(statistics.energy.iter().all(|e| e.is_finite()));
    let flags = statistics.divergent_flags();
    assert_eq!(flags.len(), samples.len());
    assert_eq!(
        flags.iter().filter(|&&d| d).count(),
        statistics.divergence_count()
    );
    assert!(statistics.divergence_count() > 0);
    for &step in &statistics.divergent_steps {
        assert!(flags[step]);
    }
}

#[test]
fn e_bfmi_of_a_standard_gaussian_is_healthy() {
    let b = GraphBuilder::new();
    let x = b.gaussian_with_shape(&[10], 0.0, 1.0);
    b.set_value(x, DoubleTensor::vector(vec![0.0; 10]));
    let x = x.id();
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let (adapt, draws) = (500, 2000);
    let (samples, statistics) = Nuts::default()
        .with_adapt_count(adapt)
        .sample_with_statistics(
            &model,
            &mut state,
            &[x],
            adapt + draws,
            SamplingOptions::default().with_drop(adapt),
            &mut rng(21),
        )
        .unwrap();
    assert_eq!(statistics.energy.len(), adapt + samples.len());
    let e_bfmi = diagnostics::e_bfmi(&statistics.energy[adapt..]);
    assert!(e_bfmi > diagnostics::E_BFMI_THRESHOLD, "E-BFMI {e_bfmi}");
}

#[test]
fn e_bfmi_is_nan_for_too_few_or_degenerate_energies() {
    assert!(diagnostics::e_bfmi(&[]).is_nan());
    assert!(diagnostics::e_bfmi(&[1.0]).is_nan());
    assert!(diagnostics::e_bfmi(&[2.0, 2.0, 2.0]).is_nan());
    assert!(diagnostics::e_bfmi(&[1.0, f64::NAN, 2.0]).is_nan());
    assert!(diagnostics::e_bfmi(&[1.0, f64::INFINITY, 2.0]).is_nan());
    // alternating energies: three successive differences of ±2 over four unit deviations, 12 / 4
    let e = diagnostics::e_bfmi(&[1.0, -1.0, 1.0, -1.0]);
    assert!((e - 3.0).abs() < 1e-12, "{e}");
    // a slow drift: successive differences are small next to the spread
    let drift: Vec<f64> = (0..100).map(f64::from).collect();
    assert!(diagnostics::e_bfmi(&drift) < 0.01);
}

/// The AR(1) model x_t = φ x_{t-1} + ε_t, φ ~ N(0.5, 0.3²), ε ~ N(0, 1), y_t ~ N(x_t, 0.5) observed, with x built by
/// a scan or unrolled. Returns the model, its state and (φ, ε).
fn ar1_model(scan: bool) -> (Model, State, NodeId, NodeId) {
    let data = [0.4, 0.9, 0.2, -0.5, -0.1];
    let b = GraphBuilder::new();
    let phi = b.gaussian(0.5, 0.3);
    let eps = b.gaussian_with_shape(&[data.len()], 0.0, 1.0);
    let x0 = b.scalar(0.2);
    let x = if scan {
        b.scan(x0, &[eps], &[phi], |_, x, e, c| c[0] * x + e[0])
    } else {
        let mut x = x0;
        let items: Vec<_> = (0..data.len())
            .map(|t| {
                x = phi * x + eps.take(&[t]);
                x
            })
            .collect();
        b.stack(0, &items)
    };
    let y = b.gaussian(x, 0.5);
    b.observe(y, DoubleTensor::vector(data.to_vec()));
    b.set_value(phi, DoubleTensor::scalar(0.5));
    b.set_value(eps, DoubleTensor::vector(vec![0.0; data.len()]));
    let (phi, eps) = (phi.id(), eps.id());
    let (graph, state) = b.build().unwrap();
    (Model::new(graph, &state).unwrap(), state, phi, eps)
}

#[test]
fn nuts_on_a_scan_model_matches_the_unrolled_model() {
    let run = |scan: bool, seed: u64| {
        let (model, mut state, phi, eps) = ar1_model(scan);
        let samples = Nuts::default()
            .with_adapt_count(300)
            .sample_with_statistics(
                &model,
                &mut state,
                &[phi, eps],
                3000,
                SamplingOptions::default()
                    .with_drop(300)
                    .with_down_sample_interval(1),
                &mut rng(seed),
            )
            .unwrap()
            .0;
        let eps0 = samples.f64(eps).unwrap().element(&[0]).unwrap();
        (series(&samples, phi), eps0)
    };
    // the scan's log probability and gradient are bit for bit the unrolled model's, so with the same seed the
    // chains are identical
    let (phi_scan, eps_scan) = run(true, 21);
    let (phi_unrolled, eps_unrolled) = run(false, 21);
    assert_eq!(phi_scan, phi_unrolled);
    assert_eq!(eps_scan, eps_unrolled);
    // and an independent unrolled chain agrees on the posterior means within Monte Carlo error
    let (phi_other, eps_other) = run(false, 22);
    for (label, a, b) in [
        ("phi", &phi_scan, &phi_other),
        ("eps[0]", &eps_scan, &eps_other),
    ] {
        let (ess_a, ess_b) = (effective_sample_size(a), effective_sample_size(b));
        let v = variance(b);
        let tolerance = 5.0 * (v / ess_a + v / ess_b).sqrt();
        assert!(
            (mean(a) - mean(b)).abs() < tolerance,
            "{label}: scan mean {} vs unrolled {} (tolerance {tolerance})",
            mean(a),
            mean(b)
        );
    }
}
