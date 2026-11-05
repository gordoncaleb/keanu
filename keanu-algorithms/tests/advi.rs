//! ADVI against analytic Gaussian posteriors and NUTS, plus its settings, errors and reproducibility.

use keanu_algorithms::mcmc::SamplingOptions;
use keanu_algorithms::mcmc::chains::run_chains;
use keanu_algorithms::mcmc::nuts::Nuts;
use keanu_algorithms::vi::{Advi, AdviResult, VariationalFamily};
use keanu_algorithms::{AlgorithmError, NetworkSamples};
use keanu_distributions::KeanuRng;
use keanu_graph::{GraphBuilder, Model, NodeId, State};
use keanu_tensor::DoubleTensor;

fn rng(seed: u64) -> KeanuRng {
    KeanuRng::seed_from_u64(seed)
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn variance(xs: &[f64]) -> f64 {
    let m = mean(xs);
    xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (xs.len() as f64 - 1.0)
}

fn series(samples: &NetworkSamples, id: NodeId, index: &[usize]) -> Vec<f64> {
    samples.f64(id).unwrap().element(index).unwrap()
}

const PRIOR_SD: f64 = 10.0;
const OBS_SD: [f64; 3] = [1.0, 2.0, 0.5];
const OBS: [f64; 3] = [3.0, -1.0, 2.0];

/// mu ~ N(0, 10) with three independent elements, each observed once through y ~ N(mu, OBS_SD): an independent
/// Gaussian posterior. Returns the model, its state, mu, and the posterior means and variances.
fn independent_gaussian() -> (Model, State, NodeId, Vec<f64>, Vec<f64>) {
    let b = GraphBuilder::new();
    let mu = b.gaussian_with_shape(&[3], 0.0, PRIOR_SD);
    b.set_value(mu, DoubleTensor::vector(vec![0.0; 3]));
    let sd = b.constant(DoubleTensor::vector(OBS_SD.to_vec()));
    let y = b.gaussian_with_shape(&[3], mu, sd);
    b.observe(y, DoubleTensor::vector(OBS.to_vec()));
    let mu = mu.id();
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let precision: Vec<f64> = OBS_SD
        .iter()
        .map(|s| 1.0 / (PRIOR_SD * PRIOR_SD) + 1.0 / (s * s))
        .collect();
    let means = (0..3)
        .map(|i| OBS[i] / (OBS_SD[i] * OBS_SD[i]) / precision[i])
        .collect();
    let variances = precision.iter().map(|p| 1.0 / p).collect();
    (model, state, mu, means, variances)
}

/// x1 ~ N(0, 1), x2 ~ N(x1, 0.5), y ~ N(x1 + x2, 1) observed at 2: a correlated bivariate Gaussian posterior.
/// Returns the model, its state, the latents, and the posterior mean, covariance and precision.
#[allow(clippy::type_complexity)]
fn correlated_gaussian() -> (
    Model,
    State,
    [NodeId; 2],
    [f64; 2],
    [[f64; 2]; 2],
    [[f64; 2]; 2],
) {
    let b = GraphBuilder::new();
    let x1 = b.gaussian(0.0, 1.0);
    let x2 = b.gaussian(x1, 0.5);
    b.set_value(x1, DoubleTensor::scalar(0.0));
    b.set_value(x2, DoubleTensor::scalar(0.0));
    let y = b.gaussian(x1 + x2, 1.0);
    b.observe(y, DoubleTensor::scalar(2.0));
    let ids = [x1.id(), x2.id()];
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    // the prior precision of (x1, x2): x1²/1 + (x2 - x1)²/0.25, plus the likelihood's (x1 + x2)²/1
    let precision = [[1.0 + 4.0 + 1.0, -4.0 + 1.0], [-4.0 + 1.0, 4.0 + 1.0]];
    let det = precision[0][0] * precision[1][1] - precision[0][1] * precision[1][0];
    let covariance = [
        [precision[1][1] / det, -precision[0][1] / det],
        [-precision[1][0] / det, precision[0][0] / det],
    ];
    // the mean solves precision · m = (2, 2) (the likelihood's y / 1 times its design (1, 1))
    let mean = [
        covariance[0][0] * 2.0 + covariance[0][1] * 2.0,
        covariance[1][0] * 2.0 + covariance[1][1] * 2.0,
    ];
    (model, state, ids, mean, covariance, precision)
}

fn covariance_entry(result: &AdviResult<'_>, i: usize, j: usize) -> f64 {
    let d = result.mean().len();
    result.covariance().as_slice()[i * d + j]
}

#[test]
fn mean_field_recovers_an_independent_gaussian_posterior() {
    let (model, mut state, mu, means, variances) = independent_gaussian();
    let result = Advi::mean_field()
        .with_tol_rel_obj(0.001)
        .with_grad_samples(10)
        .fit(&model, &mut state, &mut rng(1))
        .unwrap();
    let range = result.parameterisation().range(mu).unwrap();
    for i in 0..3 {
        let m = result.mean()[range.start + i];
        let v = covariance_entry(&result, range.start + i, range.start + i);
        let sd = variances[i].sqrt();
        assert!(
            (m - means[i]).abs() < 0.1 * sd,
            "mean {i}: {m} vs {}",
            means[i]
        );
        assert!(
            (v / variances[i] - 1.0).abs() < 0.2,
            "variance {i}: {v} vs {}",
            variances[i]
        );
        // mean field: no covariance between elements
        let other = (i + 1) % 3;
        assert_eq!(
            covariance_entry(&result, range.start + i, range.start + other),
            0.0
        );
    }

    // draws follow the approximation
    let samples = result.sample(&mut state, &[mu], 4000, &mut rng(2)).unwrap();
    assert_eq!(samples.len(), 4000);
    for i in 0..3 {
        let xs = series(&samples, mu, &[i]);
        let sd = variances[i].sqrt();
        assert!((mean(&xs) - means[i]).abs() < 0.15 * sd, "draw mean {i}");
        assert!(
            (variance(&xs) / variances[i] - 1.0).abs() < 0.25,
            "draw variance {i}"
        );
    }
    // log probs are the model's at each draw, as other samplers record them
    let first = samples.sample(0).unwrap();
    let mut probe = state.clone();
    probe
        .set(model.graph(), mu, first.f64(mu).unwrap().clone())
        .unwrap();
    assert!((model.log_prob(&probe).unwrap() - first.log_prob()).abs() < 1e-9);
}

#[test]
fn full_rank_recovers_correlation_and_mean_field_underestimates_variance() {
    let (model, state, [x1, x2], post_mean, post_cov, precision) = correlated_gaussian();
    let correlation = post_cov[0][1] / (post_cov[0][0] * post_cov[1][1]).sqrt();
    assert!(correlation.abs() > 0.5, "the test posterior is correlated");

    let full = Advi::full_rank()
        .with_tol_rel_obj(0.001)
        .with_grad_samples(10)
        .fit(&model, &mut state.clone(), &mut rng(3))
        .unwrap();
    assert_eq!(full.family(), VariationalFamily::FullRank);
    let (i1, i2) = (
        full.parameterisation().range(x1).unwrap().start,
        full.parameterisation().range(x2).unwrap().start,
    );
    for (i, m) in [(i1, post_mean[0]), (i2, post_mean[1])] {
        assert!(
            (full.mean()[i] - m).abs() < 0.1,
            "{} vs {m}",
            full.mean()[i]
        );
    }
    let c = |r: &AdviResult<'_>, a: usize, b: usize| covariance_entry(r, a, b);
    let fr_corr = c(&full, i1, i2) / (c(&full, i1, i1) * c(&full, i2, i2)).sqrt();
    assert!(
        (fr_corr - correlation).abs() < 0.1,
        "full rank correlation {fr_corr} vs {correlation}"
    );
    assert!((c(&full, i1, i1) / post_cov[0][0] - 1.0).abs() < 0.25);
    assert!((c(&full, i2, i2) / post_cov[1][1] - 1.0).abs() < 0.25);

    let field = Advi::mean_field()
        .with_tol_rel_obj(0.001)
        .with_grad_samples(10)
        .fit(&model, &mut state.clone(), &mut rng(3))
        .unwrap();
    // mean field's marginal variances are the reciprocals of the precision's diagonal, below the true variances
    for (k, i) in [i1, i2].into_iter().enumerate() {
        let v = c(&field, i, i);
        assert!(v < 0.8 * post_cov[k][k], "mean field variance {v}");
        assert!(
            (v * precision[k][k] - 1.0).abs() < 0.25,
            "{v} vs 1/{}",
            precision[k][k]
        );
        assert!((field.mean()[i] - post_mean[k]).abs() < 0.15);
    }
}

const SCHOOL_EFFECTS: [f64; 8] = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0];
const SCHOOL_SIGMAS: [f64; 8] = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0];

/// Non-centred eight schools (as in `tests/mcmc_nuts.rs`): mu ~ N(0, 5), tau ~ HalfCauchy(5), eta ~ N(0, 1),
/// theta = mu + tau·eta, y ~ N(theta, sigma).
fn eight_schools() -> (Model, State, Vec<NodeId>) {
    let schools = SCHOOL_EFFECTS.len();
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 5.0);
    let tau = b.half_cauchy(5.0);
    b.set_value(mu, DoubleTensor::scalar(0.0));
    b.set_value(tau, DoubleTensor::scalar(1.0));
    let eta = b.gaussian_with_shape(&[schools], 0.0, 1.0);
    b.set_value(eta, DoubleTensor::vector(vec![0.0; schools]));
    let latents = vec![mu.id(), tau.id(), eta.id()];
    let theta = mu + tau * eta;
    let sigma = b.constant(DoubleTensor::vector(SCHOOL_SIGMAS.to_vec()));
    let y = b.gaussian_with_shape(&[schools], theta, sigma);
    b.observe(y, DoubleTensor::vector(SCHOOL_EFFECTS.to_vec()));
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    (model, state, latents)
}

#[test]
fn eight_schools_means_are_within_a_posterior_sd_of_nuts() {
    let (model, state, latents) = eight_schools();
    let (adapt, draws) = (500, 1000);
    let nuts = Nuts::default()
        .with_adapt_count(adapt)
        .with_target_acceptance_prob(0.9);
    let chains = run_chains(2, 11, &state, |state, rng| {
        nuts.sample_with_statistics(
            &model,
            state,
            &latents,
            adapt + draws,
            SamplingOptions::default().with_drop(adapt),
            rng,
        )
        .map(|(samples, _)| samples)
    })
    .unwrap();

    for family in [VariationalFamily::MeanField, VariationalFamily::FullRank] {
        let mut s = state.clone();
        let result = Advi::default()
            .with_family(family)
            .with_tol_rel_obj(0.001)
            .with_grad_samples(10)
            .fit(&model, &mut s, &mut rng(5))
            .unwrap();
        let samples = result.sample(&mut s, &latents, 4000, &mut rng(6)).unwrap();
        let elements: Vec<(NodeId, Vec<usize>)> = vec![(latents[0], vec![]), (latents[1], vec![])]
            .into_iter()
            .chain((0..8).map(|i| (latents[2], vec![i])))
            .collect();
        for (id, index) in elements {
            let nuts_draws: Vec<f64> = chains.iter().flat_map(|c| series(c, id, &index)).collect();
            let (nuts_mean, nuts_sd) = (mean(&nuts_draws), variance(&nuts_draws).sqrt());
            let advi_mean = mean(&series(&samples, id, &index));
            assert!(
                (advi_mean - nuts_mean).abs() < nuts_sd,
                "{family:?} {id:?}{index:?}: ADVI {advi_mean} vs NUTS {nuts_mean} ± {nuts_sd}"
            );
        }
    }
}

#[test]
fn elbo_increases_and_converges_before_the_cap() {
    let (model, mut state, ..) = independent_gaussian();
    let advi = Advi::mean_field();
    let result = advi.fit(&model, &mut state, &mut rng(7)).unwrap();
    let trace = result.elbo_trace();
    assert!(trace.len() >= 2);
    assert!(trace.iter().all(|e| e.is_finite()));
    // the first entry is the initial approximation's; later ones are better on average
    let later = mean(&trace[1..]);
    assert!(
        later > trace[0],
        "ELBO {later} after vs {} at the start",
        trace[0]
    );
    assert!(result.converged());
    assert!(result.iterations() < advi.max_iterations);
    assert_eq!(result.iterations() % advi.eval_elbo, 0);
    assert_eq!(trace.len(), 1 + result.iterations() / advi.eval_elbo);
    assert!(result.eta() > 0.0);
    // the state is left at the approximation's mean
    let param = result.parameterisation();
    let z = param.from_state(&state).unwrap();
    for (a, b) in z.iter().zip(result.mean()) {
        assert!((a - b).abs() < 1e-9);
    }
}

#[test]
fn a_capped_run_reports_not_converged() {
    let (model, mut state, ..) = independent_gaussian();
    let result = Advi::mean_field()
        .with_max_iterations(150)
        .with_tol_rel_obj(1e-12)
        .fit(&model, &mut state, &mut rng(7))
        .unwrap();
    assert!(!result.converged());
    assert_eq!(result.iterations(), 150);
    assert_eq!(result.elbo_trace().len(), 2);
}

#[test]
fn fixed_eta_skips_adaptation() {
    let (model, mut state, mu, means, _) = independent_gaussian();
    let result = Advi::mean_field()
        .with_adapt(false)
        .with_eta(0.5)
        .fit(&model, &mut state, &mut rng(8))
        .unwrap();
    assert_eq!(result.eta(), 0.5);
    let start = result.parameterisation().range(mu).unwrap().start;
    assert!((result.mean()[start] - means[0]).abs() < 0.5);
}

#[test]
fn results_are_reproducible_for_a_fixed_seed() {
    let (model, state, mu, ..) = independent_gaussian();
    let run = |seed| {
        let mut s = state.clone();
        let result = Advi::full_rank()
            .fit(&model, &mut s, &mut rng(seed))
            .unwrap();
        let samples = result.sample(&mut s, &[mu], 10, &mut rng(seed)).unwrap();
        (
            result.mean().to_vec(),
            result.covariance(),
            result.elbo_trace().to_vec(),
            samples,
        )
    };
    assert_eq!(run(9), run(9));
    assert_ne!(run(9).0, run(10).0);
}

#[test]
fn constrained_latents_give_draws_inside_their_supports() {
    let b = GraphBuilder::new();
    let sigma = b.half_gaussian(2.0);
    let rate = b.gamma(1.0, 2.0);
    b.set_value(sigma, DoubleTensor::scalar(1.0));
    b.set_value(rate, DoubleTensor::scalar(1.0));
    let y = b.gaussian_with_shape(&[4], 0.0, sigma);
    b.observe(y, DoubleTensor::vector(vec![0.3, -1.2, 0.8, 2.0]));
    let z = b.exponential_with_shape(&[3], rate);
    b.observe(z, DoubleTensor::vector(vec![0.5, 1.5, 0.2]));
    let (sigma, rate) = (sigma.id(), rate.id());
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    for family in [VariationalFamily::MeanField, VariationalFamily::FullRank] {
        let result = Advi::default()
            .with_family(family)
            .fit(&model, &mut state, &mut rng(12))
            .unwrap();
        assert!(state.f64(sigma).unwrap().as_slice()[0] > 0.0);
        let samples = result
            .sample(&mut state, &[sigma, rate], 1000, &mut rng(13))
            .unwrap();
        for id in [sigma, rate] {
            let xs = series(&samples, id, &[]);
            assert!(xs.iter().all(|&x| x > 0.0 && x.is_finite()));
        }
        assert!(samples.log_probs().iter().all(|lp| lp.is_finite()));
    }
}

#[test]
fn discrete_latents_are_rejected() {
    let b = GraphBuilder::new();
    let mu = b.gaussian(2.0, 1.0);
    let k = b.poisson(3.0);
    b.set_value(mu, DoubleTensor::scalar(2.0));
    let y = b.gaussian(mu, 1.0);
    b.observe(y, DoubleTensor::scalar(1.0));
    let k = k.label("k");
    let _ = k;
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let error = Advi::default()
        .fit(&model, &mut state, &mut rng(1))
        .unwrap_err();
    assert_eq!(error, AlgorithmError::DiscreteLatents(vec!["k".into()]));
}

#[test]
fn a_model_without_latents_is_rejected() {
    let b = GraphBuilder::new();
    let y = b.gaussian(0.0, 1.0);
    b.observe(y, DoubleTensor::scalar(1.0));
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let error = Advi::default()
        .fit(&model, &mut state, &mut rng(1))
        .unwrap_err();
    assert_eq!(error, AlgorithmError::NoLatents);
}

#[test]
fn a_non_finite_initial_elbo_is_an_error() {
    // y ~ N(x, 1e-300) observed far from where any draw of x lands: every log density underflows to -inf
    let b = GraphBuilder::new();
    let x = b.gaussian(0.0, 1.0);
    b.set_value(x, DoubleTensor::scalar(0.0));
    let y = b.gaussian(x, 1e-300);
    b.observe(y, DoubleTensor::scalar(1e10));
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let error = Advi::default()
        .fit(&model, &mut state, &mut rng(1))
        .unwrap_err();
    assert!(matches!(error, AlgorithmError::NonFiniteElbo(_)), "{error}");
}

#[test]
fn invalid_settings_are_rejected() {
    let (model, state, mu, ..) = independent_gaussian();
    let invalid = [
        Advi::default().with_grad_samples(0),
        Advi::default().with_elbo_samples(0),
        Advi::default().with_eval_elbo(0),
        Advi::default().with_tol_rel_obj(0.0),
        Advi::default().with_tol_rel_obj(f64::NAN),
        Advi::default().with_max_iterations(0),
        Advi::default().with_adapt_iterations(0),
        Advi::default().with_adapt(false).with_eta(0.0),
        Advi::default().with_adapt(false).with_eta(f64::INFINITY),
    ];
    for advi in invalid {
        let error = advi
            .fit(&model, &mut state.clone(), &mut rng(1))
            .unwrap_err();
        assert!(
            matches!(error, AlgorithmError::InvalidSetting(_)),
            "{advi:?}: {error}"
        );
    }
    // sampling needs nodes
    let mut s = state.clone();
    let result = Advi::default()
        .with_max_iterations(100)
        .fit(&model, &mut s, &mut rng(1))
        .unwrap();
    assert!(matches!(
        result.sample(&mut s, &[], 1, &mut rng(1)),
        Err(AlgorithmError::InvalidArgument(_))
    ));
    let _ = mu;
}
