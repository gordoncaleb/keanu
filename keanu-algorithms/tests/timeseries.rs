//! Smoothing, decoding and forecasting over posterior draws: each draw's results are those of the distributions'
//! own methods at the draw's parameters, deterministic parents are recomputed per draw, and missing latents and
//! the wrong kind of node are errors.

use keanu_algorithms::timeseries::{HiddenMarkovPosterior, StateSpacePosterior};
use keanu_algorithms::{AlgorithmError, ChainSamples, NetworkSamples};
use keanu_distributions::{
    FutureStep, Gaussian, HiddenMarkovChain, KeanuRng, LinearGaussianStateSpace,
};
use keanu_graph::op::DistributionKind;
use keanu_graph::{DoubleVar, GraphBuilder, Model, NodeId, State};
use keanu_tensor::DoubleTensor;

fn m(rows: usize, cols: usize, values: &[f64]) -> DoubleTensor {
    DoubleTensor::new(&[rows, cols], values.to_vec()).unwrap()
}

const Y: [f64; 5] = [0.3, 0.9, 1.4, 1.1, 2.0];

/// A local level model whose level variance is the square of a latent scale, and whose observation variance is a
/// second latent.
fn local_level() -> (Model, State, NodeId, NodeId, NodeId) {
    let b = GraphBuilder::new();
    let one = || m(1, 1, &[1.0]);
    let scale = b.half_gaussian(1.0);
    let noise = b.half_gaussian(1.0);
    let y = b.linear_gaussian_state_space(
        5,
        DoubleTensor::vector(vec![0.0]),
        m(1, 1, &[10.0]),
        one(),
        (scale * scale).reshape(&[1, 1]),
        one(),
        noise.reshape(&[1, 1]),
    );
    b.observe(y, m(5, 1, &Y));
    b.set_value(scale, DoubleTensor::scalar(1.0));
    b.set_value(noise, DoubleTensor::scalar(0.5));
    let (scale, noise, y) = (scale.id(), noise.id(), y.id());
    let (graph, state) = b.build().unwrap();
    (Model::new(graph, &state).unwrap(), state, scale, noise, y)
}

fn level_model(scale: f64, noise: f64) -> LinearGaussianStateSpace<'static> {
    LinearGaussianStateSpace::new(
        DoubleTensor::vector(vec![0.0]),
        m(1, 1, &[10.0]),
        m(1, 1, &[1.0]),
        m(1, 1, &[scale * scale]),
        m(1, 1, &[1.0]),
        m(1, 1, &[noise]),
    )
}

const DRAWS: [(f64, f64); 3] = [(0.5, 0.2), (1.0, 0.5), (0.2, 1.3)];

fn level_samples(model: &Model, state: &mut State, nodes: &[NodeId]) -> NetworkSamples {
    let mut samples = NetworkSamples::new(model.graph(), nodes).unwrap();
    for (scale, noise) in DRAWS {
        state
            .set_and_cascade(model.graph(), nodes[0], DoubleTensor::scalar(scale))
            .unwrap();
        state
            .set_and_cascade(model.graph(), nodes[1], DoubleTensor::scalar(noise))
            .unwrap();
        samples
            .record(state, model.log_prob(state).unwrap())
            .unwrap();
    }
    samples
}

#[test]
fn state_space_posteriors_run_each_draws_model() {
    let (model, mut state, scale, noise, y) = local_level();
    let samples = level_samples(&model, &mut state, &[scale, noise]);
    let posterior = StateSpacePosterior::new(&model, &state, y, &samples).unwrap();
    assert_eq!(posterior.len(), 3);
    let x = m(5, 1, &Y);
    let smoothed = posterior.smooth().unwrap();
    let forecasts = posterior.forecast(4).unwrap();
    for (i, (s, n)) in DRAWS.into_iter().enumerate() {
        let direct = level_model(s, n);
        assert_eq!(posterior.model(i).unwrap(), direct);
        assert_eq!(posterior.observations(i).unwrap(), &x);
        assert_eq!(smoothed[i], direct.smooth(&x).unwrap());
        assert_eq!(forecasts[i], direct.forecast(&x, 4).unwrap());
    }
    // the posterior mean of the states averages the smoothed means
    let mean = posterior.mean_states().unwrap();
    for t in 0..5 {
        let expected = smoothed.iter().map(|s| s.means.as_slice()[t]).sum::<f64>() / 3.0;
        assert!((mean.as_slice()[t] - expected).abs() < 1e-14);
    }
    // future matrices given per draw: the observation noise of the draw, doubled
    let doubled = posterior
        .forecast_with(|_, model| {
            Ok(vec![
                FutureStep::new(
                    model.transition_matrix.clone().into_owned(),
                    model.transition_covariance.clone().into_owned(),
                    model.observation_matrix.clone().into_owned(),
                    &*model.observation_covariance * 2.0,
                );
                2
            ])
        })
        .unwrap();
    for (i, (_, n)) in DRAWS.into_iter().enumerate() {
        let (a, b) = (
            doubled[i].observation_covariances.as_slice(),
            forecasts[i].observation_covariances.as_slice(),
        );
        assert!((a[0] - b[0] - n).abs() < 1e-12);
    }
    // draws of states and of the future, one per posterior draw
    let mut rng = KeanuRng::seed_from_u64(1);
    let states = posterior.sample_states(&mut rng).unwrap();
    assert_eq!(states.len(), 3);
    assert_eq!(states[0].shape(), &[5, 1]);
    let paths = posterior.sample_forecast(6, &mut rng).unwrap();
    assert_eq!(paths[2].observations.shape(), &[6, 1]);
    let with = posterior
        .sample_forecast_with(
            |_, _| Err(AlgorithmError::InvalidArgument("no".into())),
            &mut rng,
        )
        .unwrap_err();
    assert!(matches!(with, AlgorithmError::InvalidArgument(_)));
    // several chains are concatenated in order
    let chains = ChainSamples::new(vec![samples.clone(), samples.clone()]).unwrap();
    let both = StateSpacePosterior::from_chains(&model, &state, y, &chains).unwrap();
    assert_eq!(both.len(), 6);
    assert_eq!(both.model(4), posterior.model(1));
}

#[test]
fn every_latent_the_parameters_need_must_be_sampled() {
    let (model, mut state, scale, noise, y) = local_level();
    let samples = level_samples(&model, &mut state, &[scale, noise]);
    // a recording of the scale only
    let mut partial = NetworkSamples::new(model.graph(), &[scale]).unwrap();
    partial.record(&state, 0.0).unwrap();
    let error = StateSpacePosterior::new(&model, &state, y, &partial).unwrap_err();
    assert!(
        matches!(&error, AlgorithmError::InvalidArgument(m) if m.contains("without samples")),
        "{error}"
    );
    // not a state space node
    let error = StateSpacePosterior::new(&model, &state, scale, &samples).unwrap_err();
    assert!(
        matches!(error, AlgorithmError::InvalidArgument(_)),
        "{error}"
    );
    let error = HiddenMarkovPosterior::new(&model, &state, y, &samples).unwrap_err();
    assert!(
        matches!(error, AlgorithmError::InvalidArgument(_)),
        "{error}"
    );
}

const OBSERVATIONS: [f64; 6] = [-1.2, -0.8, 2.3, 1.9, -1.1, 2.4];

#[test]
fn hidden_markov_posteriors_decode_each_draw() {
    // two Gaussian states with latent means (one a deterministic function of a latent gap)
    let b = GraphBuilder::new();
    let low = b.gaussian_with_shape(&[1], 0.0, 5.0);
    let gap = b.half_gaussian_with_shape(&[1], 5.0);
    let means = b.concat(0, &[low, gap]).cumsum(0);
    let stay = b.uniform(0.0, 1.0);
    let leave = b.scalar(1.0) - stay;
    let transition = b.stack(0, &[b.stack(0, &[stay, leave]), b.stack(0, &[leave, stay])]);
    let y: DoubleVar = b.hidden_markov_model(
        OBSERVATIONS.len(),
        DistributionKind::Gaussian,
        &[means.id(), b.scalar(0.7).id()],
        DoubleTensor::vector(vec![0.5, 0.5]),
        transition,
    );
    b.observe(y, DoubleTensor::vector(OBSERVATIONS.to_vec()));
    b.set_value(low, DoubleTensor::vector(vec![-1.0]));
    b.set_value(gap, DoubleTensor::vector(vec![3.0]));
    b.set_value(stay, DoubleTensor::scalar(0.8));
    let (low, gap, stay, y) = (low.id(), gap.id(), stay.id(), y.id());
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let draws = [(-1.0, 3.0, 0.8), (-0.5, 2.0, 0.6)];
    let mut samples = NetworkSamples::new(model.graph(), &[low, gap, stay]).unwrap();
    for (l, g, s) in draws {
        state
            .set_and_cascade(model.graph(), low, DoubleTensor::vector(vec![l]))
            .unwrap();
        state
            .set_and_cascade(model.graph(), gap, DoubleTensor::vector(vec![g]))
            .unwrap();
        state
            .set_and_cascade(model.graph(), stay, DoubleTensor::scalar(s))
            .unwrap();
        samples
            .record(&state, model.log_prob(&state).unwrap())
            .unwrap();
    }
    let posterior = HiddenMarkovPosterior::new(&model, &state, y, &samples).unwrap();
    assert_eq!(posterior.len(), 2);
    let marginals = posterior.posterior_marginals().unwrap();
    let paths = posterior.viterbi().unwrap();
    for (i, (l, g, s)) in draws.into_iter().enumerate() {
        let chain = HiddenMarkovChain::new(
            DoubleTensor::vector(vec![0.5, 0.5]),
            m(2, 2, &[s, 1.0 - s, 1.0 - s, s]),
        );
        let mut e = Vec::new();
        for &x in &OBSERVATIONS {
            for mean in [l, l + g] {
                e.push(
                    Gaussian::new(DoubleTensor::scalar(mean), DoubleTensor::scalar(0.7))
                        .log_prob(&DoubleTensor::scalar(x))
                        .unwrap()
                        .sum(),
                );
            }
        }
        let e = m(6, 2, &e);
        let (stored_chain, stored_e) = posterior.draw(i).unwrap();
        assert_eq!(stored_chain, &chain);
        for (a, b) in stored_e.as_slice().iter().zip(e.as_slice()) {
            assert!((a - b).abs() < 1e-12);
        }
        let expected = chain.posterior_marginals(&e).unwrap();
        for (a, b) in marginals[i].as_slice().iter().zip(expected.as_slice()) {
            assert!((a - b).abs() < 1e-12);
        }
        assert_eq!(paths[i].0, chain.viterbi(&e).unwrap().0);
    }
    // the clearly separated observations decode to the obvious path
    assert_eq!(paths[0].0, vec![0, 0, 1, 1, 0, 1]);
    let mean = posterior.mean_marginals().unwrap();
    assert_eq!(mean.shape(), &[6, 2]);
    let sampled = posterior
        .sample_states(&mut KeanuRng::seed_from_u64(3))
        .unwrap();
    assert_eq!(sampled.len(), 2);
    assert_eq!(sampled[0].len(), 6);
}

/// The posterior mean and standard deviation of a scalar node.
fn mean_sd(samples: &NetworkSamples, id: NodeId) -> (f64, f64) {
    let draws = samples.f64(id).unwrap();
    let values: Vec<f64> = draws.as_slice().iter().map(|t| t.as_slice()[0]).collect();
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    (mean, var.sqrt())
}

#[test]
fn nuts_recovers_autoregressive_and_garch_parameters() {
    use keanu_algorithms::mcmc::nuts::Nuts;
    use keanu_distributions::{Autoregressive, Garch11};
    let s = DoubleTensor::scalar;
    let mut rng = KeanuRng::seed_from_u64(4);

    // AR(1): rho = 0.7, sigma = 0.5, 400 steps
    let data = Autoregressive::new(
        s(0.0),
        DoubleTensor::vector(vec![0.7]),
        s(0.5),
        s(0.0),
        s(1.0),
    )
    .sample(&[400], &mut rng)
    .unwrap();
    let b = GraphBuilder::new();
    let rho = b.uniform_with_shape(&[1], -1.0, 1.0);
    let sigma = b.half_gaussian(2.0);
    let x = b.autoregressive(400, 0.0, rho, sigma, 0.0, 1.0);
    b.observe(x, data);
    b.set_value(rho, DoubleTensor::vector(vec![0.0]));
    b.set_value(sigma, s(1.0));
    let (rho, sigma) = (rho.id(), sigma.id());
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let samples = Nuts::default()
        .sample(&model, &mut state, &[rho, sigma], 800, &mut rng)
        .unwrap()
        .drop(300)
        .unwrap();
    for (id, truth, what) in [(rho, 0.7, "rho"), (sigma, 0.5, "sigma")] {
        let (mean, sd) = mean_sd(&samples, id);
        assert!((mean - truth).abs() < 4.0 * sd, "{what}: {mean} ± {sd}");
    }

    // GARCH(1,1): omega = 0.1, alpha = 0.15, beta = 0.7, 1000 returns, (alpha, beta, rest) ~ Dirichlet(1, 1, 1)
    let returns = Garch11::new(s(0.1), s(0.15), s(0.7), s(1.0))
        .sample(&[1000], &mut rng)
        .unwrap();
    let b = GraphBuilder::new();
    let weights = b.dirichlet(DoubleTensor::vector(vec![1.0, 1.0, 1.0]));
    let omega = b.half_gaussian(1.0);
    let r = b.garch11(1000, omega, weights.slice("0"), weights.slice("1"), 1.0);
    b.observe(r, returns);
    b.set_value(weights, DoubleTensor::vector(vec![0.2, 0.5, 0.3]));
    b.set_value(omega, s(0.3));
    let (weights, omega) = (weights.id(), omega.id());
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let samples = Nuts::default()
        .sample(&model, &mut state, &[weights, omega], 800, &mut rng)
        .unwrap()
        .drop(300)
        .unwrap();
    let draws = samples.f64(weights).unwrap();
    let persistence: Vec<f64> = draws
        .as_slice()
        .iter()
        .map(|w| w.as_slice()[0] + w.as_slice()[1])
        .collect();
    let n = persistence.len() as f64;
    let mean = persistence.iter().sum::<f64>() / n;
    let sd = (persistence.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / n).sqrt();
    assert!(
        (mean - 0.85).abs() < 4.0 * sd,
        "alpha + beta: {mean} ± {sd}"
    );
    let (mean, sd) = mean_sd(&samples, omega);
    assert!((mean - 0.1).abs() < 4.0 * sd, "omega: {mean} ± {sd}");
}
