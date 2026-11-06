//! SMC-ABC on simulator nodes: a Gaussian simulator whose ABC posterior and evidence have closed forms, approaching
//! the exact posterior as ε shrinks; the Laplace kernel; summary statistics, built in and custom; ABC mixed with
//! exact likelihood terms; prior predictive draws of a simulator; and the errors: ABC's settings and models, and
//! every other algorithm rejecting a model with a simulator node.

use keanu_algorithms::abc::{Abc, AbcKernel, SummaryStatistics};
use keanu_algorithms::forward::ForwardSampler;
use keanu_algorithms::laplace::Laplace;
use keanu_algorithms::loglik::PointwiseLogLik;
use keanu_algorithms::mcmc::SamplingOptions;
use keanu_algorithms::mcmc::annealing::SimulatedAnnealing;
use keanu_algorithms::mcmc::compound::Compound;
use keanu_algorithms::mcmc::gibbs::{DiscreteMetropolis, Gibbs};
use keanu_algorithms::mcmc::metropolis::MetropolisHastings;
use keanu_algorithms::mcmc::nuts::Nuts;
use keanu_algorithms::mcmc::select::Sampler;
use keanu_algorithms::optimizer::GradientOptimizer;
use keanu_algorithms::optimizer::bobyqa::NonGradientOptimizer;
use keanu_algorithms::particle_filter::ParticleFilter;
use keanu_algorithms::predictive::PredictiveSampler;
use keanu_algorithms::smc::{Smc, SmcResult};
use keanu_algorithms::vi::{Advi, Pathfinder};
use keanu_algorithms::{AlgorithmError, NetworkSamples};
use keanu_distributions::KeanuRng;
use keanu_graph::{DoubleVar, GraphBuilder, GraphError, Model, NodeId, State, Value};
use keanu_tensor::DoubleTensor;
use std::f64::consts::PI;

const N: usize = 20;
/// The observed sample mean.
const Y_BAR: f64 = 0.9;

/// N draws from N(mu, 1).
fn gaussian_draws(parameters: &[&Value], rng: &mut KeanuRng) -> Result<Value, String> {
    let mu = parameters[0].as_f64().ok_or("mu is f64")?.as_slice()[0];
    Ok(Value::F64(DoubleTensor::vector(
        (0..N).map(|_| mu + rng.next_gaussian()).collect(),
    )))
}

/// Observations with mean `Y_BAR`.
fn observations() -> DoubleTensor {
    DoubleTensor::vector(
        (0..N)
            .map(|i| Y_BAR + if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect(),
    )
}

/// mu ~ N(0, 1), and a simulator of N points from N(mu, 1) observed at `observations()`.
fn gaussian_simulator() -> (Model, State, NodeId) {
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 1.0).label("mu");
    let data: DoubleVar = b.simulator(&[mu.id()], &[N], gaussian_draws);
    b.observe(data.label("data"), observations());
    let mu = mu.id();
    let (graph, state) = b.build().unwrap();
    (Model::new(graph, &state).unwrap(), state, mu)
}

fn pooled(result: &SmcResult, id: NodeId) -> Vec<f64> {
    result
        .samples
        .chains()
        .iter()
        .flat_map(|c| {
            c.f64(id)
                .unwrap()
                .as_slice()
                .iter()
                .map(|t| t.as_slice()[0])
                .collect::<Vec<_>>()
        })
        .collect()
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn variance(xs: &[f64]) -> f64 {
    let m = mean(xs);
    xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / xs.len() as f64
}

fn smc() -> Smc {
    Smc::default().with_particles(1000).with_chains(4)
}

/// With the mean as statistic and the Gaussian kernel, the simulated mean is N(mu, 1/N), so the expected kernel is
/// ε √(2π) N(ȳ; mu, 1/N + ε²): the ABC posterior is the conjugate posterior with observation variance 1/N + ε², and
/// the ABC evidence is ε √(2π) N(ȳ; 0, 1 + 1/N + ε²). Returns the posterior mean, variance and log evidence.
fn abc_closed_form(epsilon: f64) -> (f64, f64, f64) {
    let v = 1.0 / N as f64 + epsilon * epsilon;
    let precision = 1.0 + 1.0 / v;
    let total = 1.0 + v;
    let log_evidence = (epsilon * (2.0 * PI).sqrt()).ln()
        - 0.5 * (2.0 * PI * total).ln()
        - 0.5 * Y_BAR * Y_BAR / total;
    (Y_BAR / v / precision, 1.0 / precision, log_evidence)
}

#[test]
fn abc_approaches_the_exact_posterior_as_epsilon_shrinks() {
    let (model, state, mu) = gaussian_simulator();
    // the exact posterior given the (sufficient) mean
    let exact_mean = Y_BAR * N as f64 / (N as f64 + 1.0);
    let exact_variance = 1.0 / (N as f64 + 1.0);
    let mut errors = Vec::new();
    for (seed, epsilon) in [(1, 1.0), (2, 0.3), (3, 0.05)] {
        let abc = Abc::new(epsilon).with_statistics(SummaryStatistics::Mean);
        let result = smc().sample_abc(&model, &state, &[mu], &abc, seed).unwrap();
        let xs = pooled(&result, mu);
        let (m, v, log_evidence) = abc_closed_form(epsilon);
        assert!(
            (mean(&xs) - m).abs() < 0.04 + 0.1 * v.sqrt(),
            "ε = {epsilon}: mean {}, expected {m}",
            mean(&xs)
        );
        assert!(
            (variance(&xs) / v - 1.0).abs() < 0.15,
            "ε = {epsilon}: variance {}, expected {v}",
            variance(&xs)
        );
        let lml = result.combined_log_marginal_likelihood();
        assert!(
            (lml - log_evidence).abs() < 0.15,
            "ε = {epsilon}: log evidence {lml}, expected {log_evidence}"
        );
        errors.push((mean(&xs) - exact_mean).abs() + (variance(&xs) - exact_variance).abs());
    }
    assert!(errors[0] > errors[1] && errors[1] > errors[2], "{errors:?}");
    assert!(errors[2] < 0.05, "{errors:?}");
}

#[test]
fn the_laplace_kernel_concentrates_near_the_exact_posterior() {
    let (model, state, mu) = gaussian_simulator();
    let abc = Abc::new(0.05)
        .with_kernel(AbcKernel::Laplace)
        .with_statistics(SummaryStatistics::Mean);
    let result = smc().sample_abc(&model, &state, &[mu], &abc, 4).unwrap();
    let xs = pooled(&result, mu);
    let exact_mean = Y_BAR * N as f64 / (N as f64 + 1.0);
    let exact_variance = 1.0 / (N as f64 + 1.0);
    assert!((mean(&xs) - exact_mean).abs() < 0.05, "{}", mean(&xs));
    // the kernel's variance 2ε² widens the posterior a little
    assert!(
        variance(&xs) > 0.9 * exact_variance && variance(&xs) < 1.3 * exact_variance,
        "{}",
        variance(&xs)
    );
}

#[test]
fn a_custom_statistic_matches_the_built_in_one() {
    let (model, state, mu) = gaussian_simulator();
    let builtin = Abc::new(0.2).with_statistics(SummaryStatistics::Mean);
    let custom = Abc::new(0.2).with_statistics(SummaryStatistics::custom(|value| {
        let x = value.as_f64().ok_or("f64 data")?.as_slice();
        Ok(vec![x.iter().sum::<f64>() / x.len() as f64])
    }));
    let smc = Smc::default().with_particles(200).with_chains(2);
    let a = smc.sample_abc(&model, &state, &[mu], &builtin, 5).unwrap();
    let b = smc.sample_abc(&model, &state, &[mu], &custom, 5).unwrap();
    assert_eq!(pooled(&a, mu), pooled(&b, mu));
    assert_eq!(a.log_marginal_likelihood, b.log_marginal_likelihood);
}

#[test]
fn built_in_statistics() {
    let value = Value::F64(DoubleTensor::vector(vec![3.0, 1.0, 4.0, 2.0]));
    let compute = |s: SummaryStatistics| s.compute(&value).unwrap();
    assert_eq!(
        compute(SummaryStatistics::Identity),
        vec![3.0, 1.0, 4.0, 2.0]
    );
    assert_eq!(compute(SummaryStatistics::Sort), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(compute(SummaryStatistics::Mean), vec![2.5]);
    assert_eq!(compute(SummaryStatistics::Median), vec![2.5]);
    assert_eq!(compute(SummaryStatistics::Variance), vec![1.25]);
    let counts = Value::I64(keanu_tensor::IntTensor::vector(vec![5, 1, 3]));
    assert_eq!(
        SummaryStatistics::Median.compute(&counts).unwrap(),
        vec![3.0]
    );
}

#[test]
fn the_variance_statistic_infers_a_scale() {
    // sigma ~ U(0.1, 5); 50 points from N(0, sigma) observed with variance 4: the posterior is near 2
    let b = GraphBuilder::new();
    let sigma = b.uniform(0.1, 5.0);
    let data: DoubleVar = b.simulator(&[sigma.id()], &[50], |p: &[&Value], rng: &mut KeanuRng| {
        let sigma = p[0].as_f64().ok_or("f64")?.as_slice()[0];
        Ok(Value::F64(DoubleTensor::vector(
            (0..50).map(|_| sigma * rng.next_gaussian()).collect(),
        )))
    });
    b.observe(
        data,
        DoubleTensor::vector(
            (0..50)
                .map(|i| if i % 2 == 0 { 2.0 } else { -2.0 })
                .collect(),
        ),
    );
    let sigma = sigma.id();
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let abc = Abc::new(0.2).with_statistics(SummaryStatistics::Variance);
    let result = Smc::default()
        .with_particles(500)
        .with_chains(2)
        .sample_abc(&model, &state, &[sigma], &abc, 6)
        .unwrap();
    let xs = pooled(&result, sigma);
    assert!((mean(&xs) - 2.0).abs() < 0.3, "{}", mean(&xs));
}

#[test]
fn abc_adds_the_exact_likelihood_of_other_observations() {
    // as gaussian_simulator, plus z ~ N(mu, 1) observed at 2: the posterior multiplies in N(2; mu, 1)
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 1.0);
    let data: DoubleVar = b.simulator(&[mu.id()], &[N], gaussian_draws);
    b.observe(data, observations());
    let z = b.gaussian(mu, 1.0);
    b.observe(z, DoubleTensor::scalar(2.0));
    let mu = mu.id();
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let epsilon = 0.1;
    let abc = Abc::new(epsilon).with_statistics(SummaryStatistics::Mean);
    let result = smc().sample_abc(&model, &state, &[mu], &abc, 7).unwrap();
    let xs = pooled(&result, mu);
    let v = 1.0 / N as f64 + epsilon * epsilon;
    let precision = 1.0 + 1.0 / v + 1.0;
    let expected = (Y_BAR / v + 2.0) / precision;
    assert!(
        (mean(&xs) - expected).abs() < 0.04,
        "{} vs {expected}",
        mean(&xs)
    );
    assert!(
        (variance(&xs) * precision - 1.0).abs() < 0.15,
        "{}",
        variance(&xs)
    );
}

#[test]
fn simulators_can_be_sampled_forward_and_predictively() {
    let (model, state, mu) = gaussian_simulator();
    let data = model.find("data").unwrap();
    let mut rng = KeanuRng::seed_from_u64(8);
    let draws = PredictiveSampler::new()
        .prior(&model, &state, &[data], 2000, &mut rng)
        .unwrap();
    // the prior predictive of each point is N(0, 2)
    let firsts: Vec<f64> = draws
        .f64(data)
        .unwrap()
        .as_slice()
        .iter()
        .map(|t| t.as_slice()[0])
        .collect();
    assert!(mean(&firsts).abs() < 0.1 && (variance(&firsts) - 2.0).abs() < 0.2);

    // an unobserved simulator is a latent that forward sampling draws
    let b = GraphBuilder::new();
    let m = b.gaussian(0.0, 1.0);
    let x: DoubleVar = b.simulator(&[m.id()], &[N], gaussian_draws);
    let x = x.id();
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let samples = ForwardSampler::new()
        .sample(&model, &mut state, &[x], 10, &mut rng)
        .unwrap();
    assert_eq!(samples.len(), 10);
    let _ = mu;
}

#[test]
fn abc_settings_and_models_are_checked() {
    let (model, state, mu) = gaussian_simulator();
    let smc = Smc::default().with_particles(50).with_chains(1);
    for epsilon in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let error = smc
            .sample_abc(&model, &state, &[mu], &Abc::new(epsilon), 1)
            .unwrap_err();
        assert!(
            matches!(error, AlgorithmError::InvalidSetting(_)),
            "{error}"
        );
    }
    // two scales for one statistic
    let abc = Abc::new(1.0)
        .with_epsilons(vec![1.0, 2.0])
        .with_statistics(SummaryStatistics::Mean);
    let error = smc.sample_abc(&model, &state, &[mu], &abc, 1).unwrap_err();
    assert!(
        matches!(error, AlgorithmError::InvalidSetting(_)),
        "{error}"
    );
    // a statistic that changes size
    let abc = Abc::new(1.0).with_statistics(SummaryStatistics::custom(|value| {
        Ok(vec![
            0.0;
            if value.as_f64().unwrap().as_slice()[0] == Y_BAR + 0.5 {
                1
            } else {
                2
            }
        ])
    }));
    let error = smc.sample_abc(&model, &state, &[mu], &abc, 1).unwrap_err();
    assert!(error.to_string().contains("summary statistics"), "{error}");
    // a failing statistic
    let abc = Abc::new(1.0).with_statistics(SummaryStatistics::custom(|_| Err("no".into())));
    let error = smc.sample_abc(&model, &state, &[mu], &abc, 1).unwrap_err();
    assert!(error.to_string().contains("no"), "{error}");

    // no simulator
    let b = GraphBuilder::new();
    let m = b.gaussian(0.0, 1.0);
    let y = b.gaussian(m, 1.0);
    b.observe(y, DoubleTensor::scalar(0.0));
    let m = m.id();
    let (graph, plain) = b.build().unwrap();
    let plain_model = Model::new(graph, &plain).unwrap();
    let error = smc
        .sample_abc(&plain_model, &plain, &[m], &Abc::new(1.0), 1)
        .unwrap_err();
    assert!(error.to_string().contains("simulator"), "{error}");

    // an unobserved simulator
    let b = GraphBuilder::new();
    let m = b.gaussian(0.0, 1.0);
    let _: DoubleVar = b.simulator(&[m.id()], &[N], gaussian_draws);
    let m = m.id();
    let (graph, latent) = b.build().unwrap();
    let latent_model = Model::new(graph, &latent).unwrap();
    let error = smc
        .sample_abc(&latent_model, &latent, &[m], &Abc::new(1.0), 1)
        .unwrap_err();
    assert!(error.to_string().contains("not observed"), "{error}");

    // a simulator returning the wrong shape
    let b = GraphBuilder::new();
    let m = b.gaussian(0.0, 1.0);
    let x: DoubleVar = b.simulator(&[m.id()], &[3], |_: &[&Value], _: &mut KeanuRng| {
        Ok(Value::F64(DoubleTensor::vector(vec![0.0; 2])))
    });
    b.observe(x, DoubleTensor::vector(vec![0.0; 3]));
    let m = m.id();
    let (graph, wrong) = b.build().unwrap();
    let wrong_model = Model::new(graph, &wrong).unwrap();
    let error = smc
        .sample_abc(&wrong_model, &wrong, &[m], &Abc::new(1.0), 1)
        .unwrap_err();
    assert!(
        matches!(error, AlgorithmError::Graph(GraphError::Simulator(_))),
        "{error}"
    );
}

/// The error of every algorithm but ABC on a model with a simulator node.
fn no_likelihood<T: std::fmt::Debug>(what: &str, result: Result<T, AlgorithmError>) {
    match result {
        Err(AlgorithmError::Graph(GraphError::NoLikelihood(node))) => {
            assert_eq!(node, "\"data\"", "{what}");
        }
        other => panic!("{what}: expected NoLikelihood, got {other:?}"),
    }
}

#[test]
fn other_algorithms_reject_simulator_nodes() {
    let (model, state, mu) = gaussian_simulator();
    let data = model.find("data").unwrap();
    let mut rng = KeanuRng::seed_from_u64(9);
    let mut evaluated = state.clone();
    evaluated.lazy_eval(model.graph(), &[mu], &mut rng).unwrap();

    // the model's own log probability names the node
    let error = model.log_prob(&evaluated).unwrap_err();
    assert!(
        error.to_string().contains("\"data\" is a simulator node"),
        "{error}"
    );
    assert_eq!(model.simulators(), &[data]);
    assert!(model.check_likelihood().is_err());

    let mut s = state.clone();
    no_likelihood(
        "NUTS",
        Nuts::default().sample(&model, &mut s, &[mu], 10, &mut rng),
    );
    no_likelihood(
        "NUTS block",
        Nuts::default()
            .block_sampler(&model, &mut s, &[mu], &[mu], &mut rng)
            .map(|_| ()),
    );
    no_likelihood(
        "Metropolis-Hastings",
        MetropolisHastings::default().sample(&model, &mut s, &[mu], 10, &mut rng),
    );
    no_likelihood(
        "sampler for the model",
        Sampler::for_model(&model, &s).sample(
            &model,
            &mut s,
            &[mu],
            10,
            SamplingOptions::default(),
            &mut rng,
        ),
    );
    no_likelihood(
        "compound",
        Compound::for_model(&model, &s).sample(&model, &mut s, &[mu], 10, &mut rng),
    );
    no_likelihood(
        "gradient optimizer",
        GradientOptimizer::new(&model).max_a_posteriori(&mut s),
    );
    no_likelihood(
        "non-gradient optimizer",
        NonGradientOptimizer::new(&model).max_a_posteriori(&mut s),
    );
    no_likelihood(
        "Laplace",
        Laplace::default().fit(&model, &mut s).map(|_| ()),
    );
    no_likelihood(
        "ADVI",
        Advi::default().fit(&model, &mut s, &mut rng).map(|_| ()),
    );
    no_likelihood(
        "Pathfinder",
        Pathfinder::default()
            .single_path(&model, &mut s, &mut rng)
            .map(|_| ()),
    );
    no_likelihood(
        "simulated annealing",
        SimulatedAnnealing::new(MetropolisHastings::default())
            .max_a_posteriori(&model, &mut s, 10, &mut rng)
            .map(|_| ()),
    );
    no_likelihood(
        "particle filter",
        ParticleFilter::default().run(&model, &mut s, &mut rng),
    );
    no_likelihood(
        "SMC",
        Smc::default()
            .with_particles(50)
            .sample(&model, &state, &[mu], 1),
    );
    let mut draws = NetworkSamples::new(model.graph(), &[mu]).unwrap();
    draws.record(&evaluated, 0.0).unwrap();
    no_likelihood(
        "pointwise log likelihood",
        PointwiseLogLik::from_samples(&model, &state, &draws),
    );
}

#[test]
fn discrete_samplers_reject_simulator_nodes() {
    // a discrete latent picks the simulator's mean
    let b = GraphBuilder::new();
    let k = b.bernoulli(0.5);
    let mu = b.where_(k, b.scalar(1.0), b.scalar(-1.0));
    let data: DoubleVar = b.simulator(&[mu.id()], &[N], gaussian_draws);
    b.observe(data.label("data"), observations());
    let counts = b.poisson(2.0);
    let (k, counts) = (k.id(), counts.id());
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let mut rng = KeanuRng::seed_from_u64(10);
    let mut s = state.clone();
    no_likelihood(
        "Gibbs",
        Gibbs::new(&[k]).sample(&model, &mut s, &[k], 10, &mut rng),
    );
    no_likelihood(
        "discrete Metropolis",
        DiscreteMetropolis::new(&[counts]).sample(&model, &mut s, &[counts], 10, &mut rng),
    );
}
