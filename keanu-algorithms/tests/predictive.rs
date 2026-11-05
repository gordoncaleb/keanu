use keanu_algorithms::{AlgorithmError, NetworkSamples, PredictiveSampler};
use keanu_distributions::KeanuRng;
use keanu_graph::{GraphBuilder, GraphError, Model, NodeId, State};
use keanu_tensor::{DoubleTensor, IntTensor};

fn mean_and_variance(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, variance)
}

fn f64_draws(samples: &NetworkSamples, id: NodeId) -> Vec<f64> {
    samples
        .f64(id)
        .unwrap()
        .as_slice()
        .iter()
        .map(|t| t.to_scalar().unwrap())
        .collect()
}

/// mu ~ N(0, 2), y_i ~ N(mu, 1) for three observations.
struct Conjugate {
    model: Model,
    state: State,
    mu: NodeId,
    ys: [NodeId; 3],
    posterior_mean: f64,
    posterior_variance: f64,
}

fn conjugate() -> Conjugate {
    let data = [1.2, 2.5, 1.9];
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 2.0);
    let ys = data.map(|y| {
        let node = b.gaussian(mu, 1.0);
        b.observe(node, DoubleTensor::scalar(y));
        node.id()
    });
    let mu = mu.id();
    let (graph, mut state) = b.build().unwrap();
    let all: Vec<_> = graph.ids().collect();
    state
        .lazy_eval(&graph, &all, &mut KeanuRng::seed_from_u64(0))
        .unwrap();
    let model = Model::new(graph, &state).unwrap();
    // posterior precision 1/2^2 + 3/1^2, posterior mean (sum y / 1^2) / precision
    let precision = 0.25 + 3.0;
    Conjugate {
        model,
        state,
        mu,
        ys,
        posterior_mean: data.iter().sum::<f64>() / precision,
        posterior_variance: 1.0 / precision,
    }
}

/// Exact posterior draws of mu, recorded as a sampler would.
fn exact_posterior(c: &Conjugate, count: usize, rng: &mut KeanuRng) -> NetworkSamples {
    let graph = c.model.graph();
    let mut state = c.state.clone();
    let mut samples = NetworkSamples::new(graph, &[c.mu]).unwrap();
    for _ in 0..count {
        let z = rng.next_gaussian();
        let mu = c.posterior_mean + c.posterior_variance.sqrt() * z;
        state.set(graph, c.mu, DoubleTensor::scalar(mu)).unwrap();
        samples.record(&state, 0.0).unwrap();
    }
    samples
}

#[test]
fn posterior_predictive_of_a_conjugate_gaussian_matches_the_analytic_predictive() {
    let c = conjugate();
    let mut rng = KeanuRng::seed_from_u64(11);
    let posterior = exact_posterior(&c, 40_000, &mut rng);
    let predictive = PredictiveSampler::new()
        .posterior(&c.model, &c.state, &posterior, &c.ys, &mut rng)
        .unwrap();
    assert_eq!(predictive.len(), 40_000);
    assert_eq!(predictive.nodes(), &c.ys);

    // y_new | data ~ N(posterior mean, 1 + posterior variance)
    let expected_variance = 1.0 + c.posterior_variance;
    for &y in &c.ys {
        let (mean, variance) = mean_and_variance(&f64_draws(&predictive, y));
        // standard errors: sqrt(1.31 / 40000) = 0.006 for the mean, 1.31 * sqrt(2 / 40000) = 0.009 for the
        // variance; tolerances are about 5 standard errors
        assert!((mean - c.posterior_mean).abs() < 0.03, "{mean}");
        assert!((variance - expected_variance).abs() < 0.05, "{variance}");
    }
    // the observations are drawn independently given mu: their covariance is the posterior variance of mu
    let (a, b) = (
        f64_draws(&predictive, c.ys[0]),
        f64_draws(&predictive, c.ys[1]),
    );
    let (ma, mb) = (mean_and_variance(&a).0, mean_and_variance(&b).0);
    let covariance = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - ma) * (y - mb))
        .sum::<f64>()
        / (a.len() as f64 - 1.0);
    assert!(
        (covariance - c.posterior_variance).abs() < 0.03,
        "{covariance}"
    );
}

#[test]
fn posterior_predictive_recomputes_deterministic_nodes_from_each_draw() {
    // y ~ N(2 * mu + 1, 0.001): each predictive draw is close to 2 * mu + 1 for its own posterior draw
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 1.0);
    let shifted = mu * 2.0 + 1.0;
    let y = b.gaussian(shifted, 0.001);
    b.observe(y, DoubleTensor::scalar(5.0));
    let (mu, shifted, y) = (mu.id(), shifted.id(), y.id());
    let (graph, mut state) = b.build().unwrap();
    let mut rng = KeanuRng::seed_from_u64(3);
    let all: Vec<_> = graph.ids().collect();
    state.lazy_eval(&graph, &all, &mut rng).unwrap();
    let model = Model::new(graph, &state).unwrap();

    // samples that also record the deterministic node, with a stale value in the input state
    let mut posterior = NetworkSamples::new(model.graph(), &[mu, shifted]).unwrap();
    let mut source = state.clone();
    for i in 0..20 {
        source
            .set_and_cascade(model.graph(), mu, DoubleTensor::scalar(i as f64))
            .unwrap();
        posterior.record(&source, 0.0).unwrap();
    }
    let predictive = PredictiveSampler::new()
        .posterior(&model, &state, &posterior, &[y], &mut rng)
        .unwrap();
    for (i, draw) in f64_draws(&predictive, y).into_iter().enumerate() {
        assert!((draw - (2.0 * i as f64 + 1.0)).abs() < 0.01, "{i}: {draw}");
    }
}

#[test]
fn posterior_predictive_of_an_observation_downstream_of_another() {
    // y2 ~ N(y1, 1) with y1 a target too: y2 follows the predictive draw of y1, not y1's observed value
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 1.0);
    let y1 = b.gaussian(mu, 0.001);
    let y2 = b.gaussian(y1 + 0.0, 0.001);
    b.observe(y1, DoubleTensor::scalar(100.0));
    b.observe(y2, DoubleTensor::scalar(100.0));
    let (mu, y1, y2) = (mu.id(), y1.id(), y2.id());
    let (graph, mut state) = b.build().unwrap();
    let mut rng = KeanuRng::seed_from_u64(4);
    let all: Vec<_> = graph.ids().collect();
    state.lazy_eval(&graph, &all, &mut rng).unwrap();
    let model = Model::new(graph, &state).unwrap();
    let mut posterior = NetworkSamples::new(model.graph(), &[mu]).unwrap();
    let mut source = state.clone();
    source
        .set(model.graph(), mu, DoubleTensor::scalar(-3.0))
        .unwrap();
    posterior.record(&source, 0.0).unwrap();

    let predictive = PredictiveSampler::new()
        .posterior(&model, &state, &posterior, &[y1, y2], &mut rng)
        .unwrap();
    assert!((f64_draws(&predictive, y1)[0] + 3.0).abs() < 0.01);
    assert!((f64_draws(&predictive, y2)[0] + 3.0).abs() < 0.01);

    // with only y2 as the target, y1 stays at its observed value
    let predictive = PredictiveSampler::new()
        .posterior(&model, &state, &posterior, &[y2], &mut rng)
        .unwrap();
    assert!((f64_draws(&predictive, y2)[0] - 100.0).abs() < 0.01);
    // and mu is irrelevant to it, so the posterior samples needn't hold mu
    let mut no_latents = NetworkSamples::new(model.graph(), &[]).unwrap();
    no_latents.record(&state, 0.0).unwrap();
    let predictive = PredictiveSampler::new()
        .posterior(&model, &state, &no_latents, &[y2], &mut rng)
        .unwrap();
    assert!((f64_draws(&predictive, y2)[0] - 100.0).abs() < 0.01);
}

#[test]
fn posterior_predictive_thins_to_the_requested_number_of_draws() {
    let c = conjugate();
    let mut rng = KeanuRng::seed_from_u64(5);
    let posterior = exact_posterior(&c, 1000, &mut rng);
    let thinned = PredictiveSampler::new()
        .with_draws(100)
        .posterior(&c.model, &c.state, &posterior, &c.ys[..1], &mut rng)
        .unwrap();
    assert_eq!(thinned.len(), 100);

    // every draw of a nearly noiseless observation tracks an evenly spaced posterior draw
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 1.0);
    let y = b.gaussian(mu, 1e-6);
    b.observe(y, DoubleTensor::scalar(0.0));
    let (mu, y) = (mu.id(), y.id());
    let (graph, mut state) = b.build().unwrap();
    let all: Vec<_> = graph.ids().collect();
    state.lazy_eval(&graph, &all, &mut rng).unwrap();
    let model = Model::new(graph, &state).unwrap();
    let mut posterior = NetworkSamples::new(model.graph(), &[mu]).unwrap();
    let mut source = state.clone();
    for i in 0..10 {
        source
            .set(model.graph(), mu, DoubleTensor::scalar(i as f64))
            .unwrap();
        posterior.record(&source, 0.0).unwrap();
    }
    let thinned = PredictiveSampler::new()
        .with_draws(4)
        .posterior(&model, &state, &posterior, &[y], &mut rng)
        .unwrap();
    let rounded: Vec<f64> = f64_draws(&thinned, y).into_iter().map(f64::round).collect();
    // indices 0, 10/4, 20/4, 30/4 rounded down
    assert_eq!(rounded, [0.0, 2.0, 5.0, 7.0]);

    for draws in [0, 11] {
        let error = PredictiveSampler::new()
            .with_draws(draws)
            .posterior(&model, &state, &posterior, &[y], &mut rng)
            .unwrap_err();
        assert!(
            matches!(error, AlgorithmError::InvalidSetting(_)),
            "{error}"
        );
    }
    // as many draws as there are samples keeps them all
    let all_draws = PredictiveSampler::new()
        .with_draws(10)
        .posterior(&model, &state, &posterior, &[y], &mut rng)
        .unwrap();
    assert_eq!(all_draws.len(), 10);
}

/// lambda ~ Gamma(scale 2, shape 3), y ~ Poisson(lambda), y observed.
fn gamma_poisson() -> (Model, State, NodeId, NodeId) {
    let b = GraphBuilder::new();
    let lambda = b.gamma(2.0, 3.0);
    let y = b.poisson(lambda);
    b.observe(y, IntTensor::scalar(4));
    let (lambda, y) = (lambda.id(), y.id());
    let (graph, mut state) = b.build().unwrap();
    let all: Vec<_> = graph.ids().collect();
    state
        .lazy_eval(&graph, &all, &mut KeanuRng::seed_from_u64(0))
        .unwrap();
    let model = Model::new(graph, &state).unwrap();
    (model, state, lambda, y)
}

#[test]
fn prior_predictive_of_a_gamma_poisson_matches_the_negative_binomial() {
    let (model, state, _lambda, y) = gamma_poisson();
    let mut rng = KeanuRng::seed_from_u64(21);
    let predictive = PredictiveSampler::new()
        .prior(&model, &state, &[y], 40_000, &mut rng)
        .unwrap();
    assert_eq!(predictive.len(), 40_000);
    let draws: Vec<f64> = predictive
        .i64(y)
        .unwrap()
        .as_slice()
        .iter()
        .map(|t| t.to_scalar().unwrap() as f64)
        .collect();
    let (mean, variance) = mean_and_variance(&draws);
    // negative binomial with r = k = 3 and mean k * theta = 6: variance k theta + k theta^2 = 18.
    // standard errors: sqrt(18 / 40000) = 0.02 for the mean; about 18 * sqrt((2 + 2.4) / 40000) = 0.19 for the
    // variance (excess kurtosis 6 / r + 1 / (r p (1 - p))... about 2.4); tolerances are about 4 standard errors
    assert!((mean - 6.0).abs() < 0.1, "{mean}");
    assert!((variance - 18.0).abs() < 0.8, "{variance}");
    // the input state is untouched
    assert_eq!(state.i64(y).unwrap().to_scalar().unwrap(), 4);
    assert!(state.is_observed(y));
}

#[test]
fn prior_predictive_ignores_observations_that_are_not_upstream_of_the_targets() {
    // two observations of the same latent: only y1 is a target, and y2 is irrelevant to it
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 1.0);
    let y1 = b.gaussian(mu, 1.0);
    let y2 = b.gaussian(mu, 1.0);
    b.observe(y1, DoubleTensor::scalar(10.0));
    b.observe(y2, DoubleTensor::scalar(10.0));
    let (y1, y2) = (y1.id(), y2.id());
    let (graph, mut state) = b.build().unwrap();
    let mut rng = KeanuRng::seed_from_u64(6);
    let all: Vec<_> = graph.ids().collect();
    state.lazy_eval(&graph, &all, &mut rng).unwrap();
    let model = Model::new(graph, &state).unwrap();
    let predictive = PredictiveSampler::new()
        .prior(&model, &state, &[y1], 20_000, &mut rng)
        .unwrap();
    let (mean, variance) = mean_and_variance(&f64_draws(&predictive, y1));
    // y1 ~ N(0, 2) a priori: standard errors 0.01 and 0.02
    assert!(mean.abs() < 0.05, "{mean}");
    assert!((variance - 2.0).abs() < 0.1, "{variance}");
    assert_eq!(state.f64(y2).unwrap().to_scalar().unwrap(), 10.0);
}

#[test]
fn prior_predictive_rejects_an_observation_upstream_of_a_target_that_depends_on_a_latent() {
    // y2 ~ N(y1, 1) with only y2 a target: a prior draw of y2 would need a prior draw of y1, which is observed
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 1.0);
    let y1 = b.gaussian(mu, 1.0);
    let y2 = b.gaussian(y1, 1.0);
    b.observe(y1, DoubleTensor::scalar(1.0));
    b.observe(y2, DoubleTensor::scalar(1.0));
    let (y1, y2) = (y1.id(), y2.id());
    let (graph, mut state) = b.build().unwrap();
    let mut rng = KeanuRng::seed_from_u64(7);
    let all: Vec<_> = graph.ids().collect();
    state.lazy_eval(&graph, &all, &mut rng).unwrap();
    let model = Model::new(graph, &state).unwrap();
    let error = PredictiveSampler::new()
        .prior(&model, &state, &[y2], 10, &mut rng)
        .unwrap_err();
    assert!(
        matches!(error, AlgorithmError::InvalidArgument(_)),
        "{error}"
    );
    // with both as targets, it works
    let predictive = PredictiveSampler::new()
        .prior(&model, &state, &[y1, y2], 10, &mut rng)
        .unwrap();
    assert_eq!(predictive.len(), 10);
}

#[test]
fn prior_predictive_keeps_observations_without_random_ancestors() {
    // x is observed data with no random parents: it stays fixed, as in the forward sampler
    let b = GraphBuilder::new();
    let x = b.gaussian(0.0, 1.0);
    b.observe(x, DoubleTensor::scalar(50.0));
    let y = b.gaussian(x, 0.001);
    b.observe(y, DoubleTensor::scalar(0.0));
    let y = y.id();
    let (graph, mut state) = b.build().unwrap();
    let mut rng = KeanuRng::seed_from_u64(8);
    let all: Vec<_> = graph.ids().collect();
    state.lazy_eval(&graph, &all, &mut rng).unwrap();
    let model = Model::new(graph, &state).unwrap();
    let predictive = PredictiveSampler::new()
        .prior(&model, &state, &[y], 5, &mut rng)
        .unwrap();
    assert!(
        f64_draws(&predictive, y)
            .iter()
            .all(|d| (d - 50.0).abs() < 0.01)
    );
}

#[test]
fn a_latent_or_deterministic_target_is_an_error() {
    let c = conjugate();
    let mut rng = KeanuRng::seed_from_u64(9);
    let posterior = exact_posterior(&c, 5, &mut rng);
    let sampler = PredictiveSampler::new();
    let error = sampler
        .posterior(&c.model, &c.state, &posterior, &[c.mu], &mut rng)
        .unwrap_err();
    assert!(
        matches!(error, AlgorithmError::InvalidArgument(_)),
        "{error}"
    );
    let error = sampler
        .prior(&c.model, &c.state, &[c.ys[0], c.mu], 5, &mut rng)
        .unwrap_err();
    assert!(
        matches!(error, AlgorithmError::InvalidArgument(_)),
        "{error}"
    );
    let error = sampler
        .prior(&c.model, &c.state, &[], 5, &mut rng)
        .unwrap_err();
    assert!(
        matches!(error, AlgorithmError::InvalidArgument(_)),
        "{error}"
    );

    // an observed boolean deterministic node has no distribution to sample
    let b = GraphBuilder::new();
    let a = b.gaussian(0.0, 1.0);
    let positive = a.gt(0.0);
    b.observe(positive, keanu_tensor::BoolTensor::scalar(true));
    let positive = positive.id();
    let (graph, mut state) = b.build().unwrap();
    let all: Vec<_> = graph.ids().collect();
    state.lazy_eval(&graph, &all, &mut rng).unwrap();
    let model = Model::new(graph, &state).unwrap();
    let error = sampler
        .prior(&model, &state, &[positive], 5, &mut rng)
        .unwrap_err();
    assert!(
        matches!(error, AlgorithmError::InvalidArgument(_)),
        "{error}"
    );
}

#[test]
fn an_unknown_target_is_an_error_not_a_panic() {
    let other = GraphBuilder::new();
    for _ in 0..10 {
        let _ = other.gaussian(0.0, 1.0);
    }
    let foreign = other.gaussian(0.0, 1.0).id();

    let c = conjugate();
    let mut rng = KeanuRng::seed_from_u64(10);
    let posterior = exact_posterior(&c, 5, &mut rng);
    let sampler = PredictiveSampler::new();
    let error = sampler
        .posterior(&c.model, &c.state, &posterior, &[foreign], &mut rng)
        .unwrap_err();
    assert_eq!(
        error,
        AlgorithmError::Graph(GraphError::UnknownNode(foreign))
    );
    let error = sampler
        .prior(&c.model, &c.state, &[foreign], 5, &mut rng)
        .unwrap_err();
    assert_eq!(
        error,
        AlgorithmError::Graph(GraphError::UnknownNode(foreign))
    );
}

#[test]
fn posterior_samples_missing_a_latent_the_targets_depend_on_are_an_error() {
    // y ~ N(mu, sigma) with samples of mu only: sigma's posterior is unknown
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 1.0);
    let sigma = b.gamma(1.0, 2.0);
    let y = b.gaussian(mu, sigma);
    b.observe(y, DoubleTensor::scalar(1.0));
    let (mu, y) = (mu.id(), y.id());
    let (graph, mut state) = b.build().unwrap();
    let mut rng = KeanuRng::seed_from_u64(12);
    let all: Vec<_> = graph.ids().collect();
    state.lazy_eval(&graph, &all, &mut rng).unwrap();
    let model = Model::new(graph, &state).unwrap();
    let mut posterior = NetworkSamples::new(model.graph(), &[mu]).unwrap();
    posterior.record(&state, 0.0).unwrap();
    let error = PredictiveSampler::new()
        .posterior(&model, &state, &posterior, &[y], &mut rng)
        .unwrap_err();
    assert!(
        matches!(error, AlgorithmError::InvalidArgument(_)),
        "{error}"
    );
}

#[test]
fn the_input_state_is_unchanged() {
    let c = conjugate();
    let before = c.state.clone();
    let mut rng = KeanuRng::seed_from_u64(13);
    let posterior = exact_posterior(&c, 50, &mut rng);
    let sampler = PredictiveSampler::new();
    sampler
        .posterior(&c.model, &c.state, &posterior, &c.ys, &mut rng)
        .unwrap();
    sampler
        .prior(&c.model, &c.state, &c.ys, 50, &mut rng)
        .unwrap();
    assert_eq!(c.state, before);
    for (&y, expected) in c.ys.iter().zip([1.2, 2.5, 1.9]) {
        assert!(c.state.is_observed(y));
        assert_eq!(c.state.f64(y).unwrap().to_scalar().unwrap(), expected);
    }
}
