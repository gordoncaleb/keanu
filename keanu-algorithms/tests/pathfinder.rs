//! Pathfinder against known posteriors: a correlated Gaussian (single and multi-path), eight schools against NUTS,
//! importance-sampling diagnostics, and its draws as starting points for NUTS chains.

use keanu_algorithms::AlgorithmError;
use keanu_algorithms::mcmc::SamplingOptions;
use keanu_algorithms::mcmc::chains::{run_chains, sample_chains_from};
use keanu_algorithms::mcmc::nuts::Nuts;
use keanu_algorithms::vi::Pathfinder;
use keanu_distributions::KeanuRng;
use keanu_graph::{GraphBuilder, Model, NodeId, State};
use keanu_tensor::DoubleTensor;

fn rng(seed: u64) -> KeanuRng {
    KeanuRng::seed_from_u64(seed)
}

/// x ~ MVN(m, Σ) with Σ = diag(s) R diag(s) for an AR(1)-like correlation R_ij = ρ^|i − j|.
fn correlated_gaussian(n: usize, rho: f64) -> (Model, State, NodeId, Vec<f64>, Vec<f64>) {
    let scale: Vec<f64> = (0..n).map(|i| 0.5 + 0.25 * i as f64).collect();
    let mean: Vec<f64> = (0..n).map(|i| i as f64 - 1.0).collect();
    let covariance: Vec<f64> = (0..n * n)
        .map(|k| {
            let (i, j) = (k / n, k % n);
            rho.powi((i as i32 - j as i32).abs()) * scale[i] * scale[j]
        })
        .collect();
    let b = GraphBuilder::new();
    let x = b.multivariate_gaussian(
        DoubleTensor::vector(mean.clone()),
        DoubleTensor::new(&[n, n], covariance.clone()).unwrap(),
    );
    b.set_value(x, DoubleTensor::vector(vec![3.0; n]));
    let x = x.id();
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    (model, state, x, mean, covariance)
}

fn moments(draws: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    let n = draws[0].len();
    let count = draws.len() as f64;
    let mean: Vec<f64> = (0..n)
        .map(|i| draws.iter().map(|d| d[i]).sum::<f64>() / count)
        .collect();
    let covariance = (0..n * n)
        .map(|k| {
            let (i, j) = (k / n, k % n);
            draws
                .iter()
                .map(|d| (d[i] - mean[i]) * (d[j] - mean[j]))
                .sum::<f64>()
                / (count - 1.0)
        })
        .collect();
    (mean, covariance)
}

/// The largest errors of the draws' moments: of a mean, in posterior standard deviations; of a variance, as the
/// absolute log of its ratio to the true one; and of a correlation.
fn moment_errors(draws: &[Vec<f64>], mean: &[f64], covariance: &[f64]) -> (f64, f64, f64) {
    let n = mean.len();
    let (m, c) = moments(draws);
    let (mut mean_error, mut variance_error, mut correlation_error) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let variance = covariance[i * n + i];
        mean_error = mean_error.max((m[i] - mean[i]).abs() / variance.sqrt());
        variance_error = variance_error.max((c[i * n + i] / variance).ln().abs());
        for j in 0..n {
            let sample = c[i * n + j] / (c[i * n + i] * c[j * n + j]).sqrt();
            let truth = covariance[i * n + j] / (variance * covariance[j * n + j]).sqrt();
            correlation_error = correlation_error.max((sample - truth).abs());
        }
    }
    (mean_error, variance_error, correlation_error)
}

/// A single path on a 5-D correlated Gaussian: the path's records, the draws' log densities and samples, and the
/// draws' moments. The best approximation along an L-BFGS path is not the posterior even for a Gaussian (inexact
/// line searches lose BFGS's exactness on quadratics, and the 5-draw ELBO estimates choose among noisy candidates),
/// so its moments are only roughly right: over seeds 1 to 6 the largest errors were 0.31 sd in a mean, a factor of
/// 1.7 in a variance and 0.56 in a correlation. The draws do have the approximation's own moments.
#[test]
fn single_path_pathfinder_approximates_a_correlated_gaussian() {
    let (model, mut state, x, mean, covariance) = correlated_gaussian(5, 0.8);
    let result = Pathfinder::default()
        .single_path(&model, &mut state, &mut rng(1))
        .unwrap();
    assert_eq!(result.paths().len(), 1);
    assert_eq!(result.failed_paths(), 0);
    assert_eq!(result.draws().len(), 1000);
    let path = &result.paths()[0];
    assert!(path.converged());
    assert!(path.iterations() > 0 && path.elbo().is_finite());
    assert_eq!(path.elbo_trace().len(), path.iterations());
    assert!(path.best_iteration() >= 1 && path.best_iteration() <= path.iterations());
    assert_eq!(path.elbo_trace()[path.best_iteration() - 1], path.elbo());
    assert!(result.evaluations() >= path.evaluations());

    // the draws have the chosen approximation's mean and covariance
    let approximation = path.covariance();
    let (draws_error, ..) = moment_errors(result.draws(), path.mean(), approximation.as_slice());
    assert!(draws_error < 0.15, "{draws_error}");
    // and roughly the posterior's
    let (mean_error, variance_error, correlation_error) =
        moment_errors(result.draws(), &mean, &covariance);
    assert!(mean_error < 0.5, "mean error {mean_error} sd");
    assert!(variance_error < 0.8, "log variance ratio {variance_error}");
    assert!(
        correlation_error < 0.7,
        "correlation error {correlation_error}"
    );
    assert!(result.pareto_k().is_finite(), "k̂ {}", result.pareto_k());

    // the log densities: the model's (an identity transform, so no Jacobian) and the approximation's
    for (z, lp) in result.draws().iter().zip(result.log_density()).take(5) {
        let mut at = state.clone();
        result.parameterisation().to_state(z, &mut at).unwrap();
        assert!((model.log_prob(&at).unwrap() - lp).abs() < 1e-9);
    }
    assert!(
        result
            .approximation_log_density()
            .iter()
            .all(|q| q.is_finite())
    );
    // the draws as samples, with the model's log probability
    let samples = result.sample(&mut state, &[x]).unwrap();
    assert_eq!(samples.len(), 1000);
    assert_eq!(samples.log_probs(), result.log_density());
}

/// Multi-path Pathfinder on the same Gaussian: the importance resampling of the four paths' pooled draws corrects
/// the single paths' errors. Over seeds 1 to 6 the largest errors were 0.10 sd in a mean, 11% in a variance and 0.09
/// in a correlation.
#[test]
fn multi_path_pathfinder_recovers_a_correlated_gaussian() {
    let (model, mut state, _, mean, covariance) = correlated_gaussian(5, 0.8);
    let result = Pathfinder::default()
        .multi_path(&model, &mut state, &mut rng(1))
        .unwrap();
    assert_eq!(result.paths().len(), 4);
    assert_eq!(result.failed_paths(), 0);
    assert_eq!(result.draws().len(), 1000);
    assert!(result.pareto_k().is_finite(), "k̂ {}", result.pareto_k());
    let (mean_error, variance_error, correlation_error) =
        moment_errors(result.draws(), &mean, &covariance);
    assert!(mean_error < 0.2, "mean error {mean_error} sd");
    assert!(variance_error < 0.25, "log variance ratio {variance_error}");
    assert!(
        correlation_error < 0.15,
        "correlation error {correlation_error}"
    );
}

/// On a 20-D Gaussian (more coordinates than the 2 × 6 columns of the low-rank part, so draws go through its thin
/// QR factorisation) a single path's mean is accurate (within 0.19 sd over 12 seeds), though its variances are not.
#[test]
fn single_path_pathfinder_finds_the_mean_of_a_larger_gaussian() {
    let (model, mut state, _, mean, covariance) = correlated_gaussian(20, 0.6);
    let result = Pathfinder::default()
        .single_path(&model, &mut state, &mut rng(2))
        .unwrap();
    let (mean_error, ..) = moment_errors(result.draws(), &mean, &covariance);
    assert!(mean_error < 0.25, "mean error {mean_error} sd");
    assert!(result.pareto_k().is_finite());
}

/// The non-centred eight schools model.
fn eight_schools() -> (Model, State, [NodeId; 3]) {
    let effects = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0];
    let errors = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0];
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 5.0);
    let tau = b.half_cauchy(5.0);
    let eta = b.gaussian_with_shape(&[8], 0.0, 1.0);
    b.set_value(mu, DoubleTensor::scalar(0.0));
    b.set_value(tau, DoubleTensor::scalar(1.0));
    b.set_value(eta, DoubleTensor::vector(vec![0.0; 8]));
    let theta = mu + tau * eta;
    let sigma = b.constant(DoubleTensor::vector(errors.to_vec()));
    let y = b.gaussian_with_shape(&[8], theta, sigma);
    b.observe(y, DoubleTensor::vector(effects.to_vec()));
    let ids = [mu.id(), tau.id(), theta.id()];
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    (model, state, ids)
}

/// The mean of a node's element over a set of samples.
fn element_mean(samples: &keanu_algorithms::NetworkSamples, id: NodeId, index: &[usize]) -> f64 {
    let values = samples.f64(id).unwrap().element(index).unwrap();
    values.iter().sum::<f64>() / values.len() as f64
}

/// Multi-path Pathfinder on (non-centred) eight schools: the resampled draws' means of mu, tau and every theta are
/// within 1.2 posterior standard deviations of NUTS's (the largest difference over seeds 1 to 6 was 0.99), and the
/// importance resampling brings tau's mean much closer than the unweighted pooled draws. Eight schools is hard for
/// Pathfinder: the mode of its unconstrained density (with the Jacobian, at log tau ≈ 3.4, where L-BFGS converges)
/// is far from the posterior's bulk (tau's posterior mean is 3.6), and k̂ (0.5 to 1.1) says the approximation is
/// poor.
#[test]
fn multi_path_pathfinder_on_eight_schools_is_near_nuts() {
    let (model, mut state, [mu, tau, theta]) = eight_schools();
    let nodes = [mu, tau, theta];
    let nuts = Nuts::default()
        .with_adapt_count(1000)
        .with_target_acceptance_prob(0.95);
    let chains = run_chains(4, 3, &state, |state, rng| {
        nuts.sample_with_statistics(
            &model,
            state,
            &nodes,
            3000,
            SamplingOptions::default().with_drop(1000),
            rng,
        )
        .map(|(samples, _)| samples)
    })
    .unwrap();
    let reference = |id: NodeId, index: &[usize]| {
        let all: Vec<f64> = chains
            .iter()
            .flat_map(|c| c.f64(id).unwrap().element(index).unwrap())
            .collect();
        let m = all.iter().sum::<f64>() / all.len() as f64;
        let sd = (all.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / all.len() as f64).sqrt();
        (m, sd)
    };

    let result = Pathfinder::default()
        .multi_path(&model, &mut state, &mut rng(1))
        .unwrap();
    assert!(result.pareto_k().is_finite());
    assert_eq!(
        result.is_reliable(),
        result.pareto_k() <= keanu_algorithms::psis::khat_threshold(4000)
    );
    let samples = result.sample(&mut state, &nodes).unwrap();
    let mut checks = vec![(mu, vec![]), (tau, vec![])];
    checks.extend((0..8).map(|i| (theta, vec![i])));
    for (id, index) in checks {
        let (m, sd) = reference(id, &index);
        let estimate = element_mean(&samples, id, &index);
        assert!(
            (estimate - m).abs() < 1.2 * sd,
            "{id:?}{index:?}: Pathfinder {estimate} vs NUTS {m} (sd {sd})"
        );
    }

    // the same paths without resampling (the same seed gives the same paths): tau's mean is far off
    let pooled = Pathfinder::default()
        .with_resample(false)
        .multi_path(&model, &mut state, &mut rng(1))
        .unwrap();
    assert_eq!(pooled.draws().len(), 4000);
    assert_eq!(pooled.pareto_k(), result.pareto_k());
    let pooled_samples = pooled.sample(&mut state, &nodes).unwrap();
    let (m, _) = reference(tau, &[]);
    let resampled_error = (element_mean(&samples, tau, &[]) - m).abs();
    let pooled_error = (element_mean(&pooled_samples, tau, &[]) - m).abs();
    assert!(
        resampled_error < pooled_error,
        "tau: resampled {resampled_error} vs pooled {pooled_error}"
    );
}

#[test]
fn pathfinder_rejects_invalid_settings_and_discrete_latents() {
    let (model, mut state, ..) = correlated_gaussian(2, 0.5);
    for pathfinder in [
        Pathfinder::default().with_draws(0),
        Pathfinder::default().with_elbo_draws(0),
        Pathfinder::default().with_paths(0),
    ] {
        assert!(matches!(
            pathfinder.multi_path(&model, &mut state, &mut rng(1)),
            Err(AlgorithmError::InvalidSetting(_))
        ));
    }
    let b = GraphBuilder::new();
    let p = b.beta(1.0, 1.0);
    b.bernoulli(p);
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    assert!(matches!(
        Pathfinder::default().single_path(&model, &mut state, &mut rng(1)),
        Err(AlgorithmError::DiscreteLatents(_))
    ));
}

/// The starting states hold distinct draws, and NUTS chains run from them.
#[test]
fn pathfinder_draws_start_nuts_chains() {
    let (model, mut state, [mu, tau, theta]) = eight_schools();
    let result = Pathfinder::default()
        .multi_path(&model, &mut state, &mut rng(5))
        .unwrap();
    let starts = result.initial_states(&state, 4, &mut rng(6)).unwrap();
    assert_eq!(starts.len(), 4);
    let values: Vec<f64> = starts
        .iter()
        .map(|s| s.f64(tau).unwrap().to_scalar().unwrap())
        .collect();
    for i in 0..4 {
        for j in 0..i {
            assert_ne!(values[i], values[j]);
        }
    }
    let nuts = Nuts::default().with_adapt_count(200);
    let chains = sample_chains_from(&starts, 7, |state, rng| {
        // each chain starts from its own draw
        nuts.sample(&model, state, &[mu, tau, theta], 400, rng)
    })
    .unwrap();
    assert_eq!(chains.num_chains(), 4);
    assert!(result.initial_states(&state, 0, &mut rng(1)).is_err());
    assert!(result.initial_states(&state, 1001, &mut rng(1)).is_err());
}
