//! Gibbs sampling and discrete random-walk Metropolis against exact posteriors, computed by enumerating the model's
//! log probability or in closed form.

mod mcmc_common;

use keanu_algorithms::AlgorithmError;
use keanu_algorithms::NetworkSamples;
use keanu_algorithms::mcmc::SamplingOptions;
use keanu_algorithms::mcmc::gibbs::{DiscreteMetropolis, Gibbs};
use keanu_algorithms::statistics::effective_sample_size;
use keanu_graph::{GraphBuilder, Model, NodeId, State, Value};
use keanu_tensor::{BoolTensor, DoubleTensor, IntTensor};
use mcmc_common::*;

/// The posterior probability of each of `values` of `id`, by evaluating the joint log probability at each.
fn exact_posterior(model: &Model, state: &State, id: NodeId, values: &[Value]) -> Vec<f64> {
    let log_probs: Vec<f64> = values
        .iter()
        .map(|value| {
            let mut at = state.clone();
            model
                .set_and_cascade(&mut at, [(id, value.clone())])
                .unwrap();
            model.log_prob(&at).unwrap()
        })
        .collect();
    normalise(&log_probs)
}

fn normalise(log_probs: &[f64]) -> Vec<f64> {
    let max = log_probs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = log_probs.iter().map(|lp| (lp - max).exp()).collect();
    let total: f64 = weights.iter().sum();
    weights.iter().map(|w| w / total).collect()
}

/// Checks the frequency of each value in `draws` against its exact probability, within five Monte Carlo standard
/// errors from the indicator series' effective sample size.
fn assert_frequencies(draws: &[i64], values: &[i64], exact: &[f64], label: &str) {
    for (&value, &p) in values.iter().zip(exact) {
        let indicator: Vec<f64> = draws.iter().map(|&d| f64::from(d == value)).collect();
        let frequency = mean(&indicator);
        let ess = if frequency > 0.0 && frequency < 1.0 {
            effective_sample_size(&indicator).max(10.0)
        } else {
            draws.len() as f64
        };
        let tolerance = 5.0 * (p * (1.0 - p) / ess).sqrt() + 0.002;
        assert!(
            (frequency - p).abs() < tolerance,
            "{label}: value {value} has frequency {frequency}, expected {p} (tolerance {tolerance}, ess {ess})"
        );
    }
}

fn ints(samples: &NetworkSamples, id: NodeId, element: usize) -> Vec<i64> {
    samples
        .i64(id)
        .unwrap()
        .as_slice()
        .iter()
        .map(|t| t.as_slice()[element])
        .collect()
}

fn bools(samples: &NetworkSamples, id: NodeId, element: usize) -> Vec<i64> {
    samples
        .bool(id)
        .unwrap()
        .as_slice()
        .iter()
        .map(|t| i64::from(t.as_slice()[element]))
        .collect()
}

#[test]
fn gibbs_samples_a_bernoulli_latent_exactly() {
    let b = GraphBuilder::new();
    let z = b.bernoulli(0.4);
    let mean = b.where_(z, b.scalar(1.0), b.scalar(-1.0));
    let y = b.gaussian_with_shape(&[3], mean, 1.5);
    b.observe(y, DoubleTensor::vector(vec![0.3, -0.4, 0.2]));
    let z = z.id();
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let exact = exact_posterior(
        &model,
        &state,
        z,
        &[
            BoolTensor::scalar(false).into(),
            BoolTensor::scalar(true).into(),
        ],
    );
    let samples = Gibbs::new(&[z])
        .sample(&model, &mut state, &[z], 5000, &mut rng(1))
        .unwrap();
    assert_frequencies(&bools(&samples, z, 0), &[0, 1], &exact, "Bernoulli");
    // the recorded log probabilities are the joint ones
    for sample in samples.iter().take(10) {
        let mut at = state.clone();
        model
            .set_and_cascade(&mut at, [(z, sample.get(z).unwrap())])
            .unwrap();
        assert!((sample.log_prob() - model.log_prob(&at).unwrap()).abs() < 1e-9);
    }
}

#[test]
fn gibbs_updates_each_element_of_a_categorical_vector() {
    // z_i ~ Categorical(0.2, 0.5, 0.3), y_i ~ N(2 z_i - 2, 1): each element's posterior depends only on its own y
    let data = [-1.7, 0.4, 2.5];
    let b = GraphBuilder::new();
    let probabilities = [b.scalar(0.2), b.scalar(0.5), b.scalar(0.3)];
    let z = b.categorical(&[3], &probabilities);
    let y = b.gaussian_with_shape(&[3], z.to_double() * 2.0 - 2.0, 1.0);
    b.observe(y, DoubleTensor::vector(data.to_vec()));
    b.set_value(z, IntTensor::vector(vec![0, 0, 0]));
    let z = z.id();
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let samples = Gibbs::new(&[z])
        .sample(&model, &mut state, &[z], 6000, &mut rng(2))
        .unwrap();
    let draws: Vec<Vec<i64>> = (0..3).map(|i| ints(&samples, z, i)).collect();
    for (i, &y) in data.iter().enumerate() {
        let log_probs: Vec<f64> = [0.2f64, 0.5, 0.3]
            .iter()
            .enumerate()
            .map(|(k, p)| p.ln() - 0.5 * (y - (2.0 * k as f64 - 2.0)).powi(2))
            .collect();
        assert_frequencies(
            &draws[i],
            &[0, 1, 2],
            &normalise(&log_probs),
            &format!("categorical element {i}"),
        );
    }
    // the elements are independent: the joint frequency of a pair is the product of the marginals
    let both = draws[0]
        .iter()
        .zip(&draws[1])
        .filter(|&(&a, &b)| a == 0 && b == 1)
        .count() as f64
        / draws[0].len() as f64;
    let p0 = draws[0].iter().filter(|&&a| a == 0).count() as f64 / draws[0].len() as f64;
    let p1 = draws[1].iter().filter(|&&b| b == 1).count() as f64 / draws[1].len() as f64;
    assert!((both - p0 * p1).abs() < 0.03, "{both} vs {}", p0 * p1);
}

#[test]
fn gibbs_finds_a_change_point_by_enumeration() {
    // y_t ~ N(t < tau ? 1 : 3, 1) for t in 0..10, tau ~ UniformInt[0, 11)
    let data = [0.8, 1.6, 0.4, 1.2, 2.1, 3.3, 2.7, 3.4, 2.9, 3.1];
    let b = GraphBuilder::new();
    let tau = b.uniform_int(0, 11);
    let t = b.int_constant(IntTensor::vector((0..10).collect()));
    let mean = b.where_(tau.gt(t), b.scalar(1.0), b.scalar(3.0));
    let y = b.gaussian_with_shape(&[10], mean, 1.0);
    b.observe(y, DoubleTensor::vector(data.to_vec()));
    let tau = tau.id();
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let values: Vec<i64> = (0..11).collect();
    let exact = exact_posterior(
        &model,
        &state,
        tau,
        &values
            .iter()
            .map(|&v| IntTensor::scalar(v).into())
            .collect::<Vec<_>>(),
    );
    let samples = Gibbs::new(&[tau])
        .sample(&model, &mut state, &[tau], 5000, &mut rng(3))
        .unwrap();
    assert_frequencies(&ints(&samples, tau, 0), &values, &exact, "change point");
}

#[test]
fn gibbs_follows_supports_that_depend_on_other_latents() {
    // n ~ UniformInt[1, 6), k ~ Binomial(0.4, n), y ~ N(k, 1) observed at 3.2; the binomial's support moves with n,
    // and ordered logistic z with y2 ~ N(z, 0.7) observed at 2.4
    let b = GraphBuilder::new();
    let n = b.uniform_int(1, 6);
    let k = b.binomial(0.4, n);
    let y = b.gaussian(k.to_double(), 1.0);
    b.observe(y, DoubleTensor::scalar(3.2));
    let cutpoints = b.constant(DoubleTensor::vector(vec![-1.0, 0.5, 1.5]));
    let z = b.ordered_logistic(0.2, cutpoints);
    let y2 = b.gaussian(z.to_double(), 0.7);
    b.observe(y2, DoubleTensor::scalar(2.4));
    b.set_value(n, IntTensor::scalar(5));
    b.set_value(k, IntTensor::scalar(1));
    b.set_value(z, IntTensor::scalar(0));
    let (n, k, z) = (n.id(), k.id(), z.id());
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    model
        .probe_for_non_zero_probability(&mut state, 1, &mut rng(0))
        .unwrap();

    // the exact joint of (n, k) by enumeration; z is independent of them
    let mut joint = Vec::new();
    for nv in 1..6i64 {
        for kv in 0..=nv {
            let mut at = state.clone();
            model
                .set_and_cascade(
                    &mut at,
                    [
                        (n, IntTensor::scalar(nv).into()),
                        (k, IntTensor::scalar(kv).into()),
                    ],
                )
                .unwrap();
            joint.push((nv, kv, model.log_prob(&at).unwrap()));
        }
    }
    let p = normalise(&joint.iter().map(|j| j.2).collect::<Vec<_>>());
    let exact_n: Vec<f64> = (1..6)
        .map(|nv| {
            joint
                .iter()
                .zip(&p)
                .filter(|(j, _)| j.0 == nv)
                .map(|(_, p)| p)
                .sum()
        })
        .collect();
    let exact_k: Vec<f64> = (0..6)
        .map(|kv| {
            joint
                .iter()
                .zip(&p)
                .filter(|(j, _)| j.1 == kv)
                .map(|(_, p)| p)
                .sum()
        })
        .collect();
    let exact_z = exact_posterior(
        &model,
        &state,
        z,
        &(0..4)
            .map(|v| IntTensor::scalar(v).into())
            .collect::<Vec<_>>(),
    );

    let samples = Gibbs::new(&[n, k, z])
        .sample_with(
            &model,
            &mut state,
            &[n, k, z],
            12_000,
            SamplingOptions::default().with_drop(100),
            &mut rng(4),
        )
        .unwrap();
    assert_frequencies(
        &ints(&samples, n, 0),
        &[1, 2, 3, 4, 5],
        &exact_n,
        "binomial size",
    );
    assert_frequencies(
        &ints(&samples, k, 0),
        &[0, 1, 2, 3, 4, 5],
        &exact_k,
        "binomial count",
    );
    assert_frequencies(
        &ints(&samples, z, 0),
        &[0, 1, 2, 3],
        &exact_z,
        "ordered logistic",
    );
}

#[test]
fn discrete_metropolis_samples_an_infinite_support() {
    // k ~ Poisson(10), y ~ Binomial(0.4, k) observed at 3: by Poisson thinning, k - 3 ~ Poisson(6)
    let b = GraphBuilder::new();
    let k = b.poisson(10.0);
    let y = b.binomial(0.4, k);
    b.observe(y, IntTensor::scalar(3));
    b.set_value(k, IntTensor::scalar(5));
    let k = k.id();
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let mut sampler = DiscreteMetropolis::new(&[k])
        .sampler(&model, &mut state, &[k], &mut rng(5))
        .unwrap();
    let samples = keanu_algorithms::mcmc::generate(
        &mut sampler,
        &mut state,
        20_000,
        SamplingOptions::default().with_drop(500),
        &mut rng(6),
    )
    .unwrap();
    let rate = sampler.acceptance_rate();
    assert!(rate > 0.3 && rate < 0.95, "acceptance rate {rate}");
    let draws: Vec<f64> = ints(&samples, k, 0).iter().map(|&d| d as f64).collect();
    assert_moments(&draws, 9.0, 6.0, "thinned Poisson");
    let values: Vec<i64> = (3..12).collect();
    let exact: Vec<f64> = values
        .iter()
        .map(|&v| {
            let j = (v - 3) as i32;
            (-6.0f64).exp() * 6f64.powi(j) / (1..=j).map(f64::from).product::<f64>()
        })
        .collect();
    assert_frequencies(&ints(&samples, k, 0), &values, &exact, "thinned Poisson");
}

#[test]
fn discrete_steps_reject_latents_they_cannot_update() {
    let b = GraphBuilder::new();
    let count = b.poisson(3.0);
    let large = b.binomial(0.5, 5000);
    let flag = b.bernoulli(0.5);
    let x = b.gaussian(0.0, 1.0);
    let y = b.gaussian(
        x + count.to_double() + large.to_double() + flag.to_double_mask(),
        1.0,
    );
    b.observe(y, DoubleTensor::scalar(1.0));
    let (count, large, flag, x) = (count.id(), large.id(), flag.id(), x.id());
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let message = |result: Result<_, AlgorithmError>| match result {
        Err(AlgorithmError::InvalidArgument(message)) => message,
        Err(other) => panic!("unexpected error {other:?}"),
        Ok(_) => panic!("expected an error"),
    };
    let gibbs = |latents: &[NodeId]| {
        Gibbs::new(latents)
            .sampler(&model, &mut state.clone(), &[x], &mut rng(1))
            .map(|_| ())
    };
    assert!(message(gibbs(&[count])).contains("DiscreteMetropolis"));
    assert!(message(gibbs(&[large])).contains("maximum support size"));
    assert!(message(gibbs(&[x])).contains("continuous"));
    assert!(message(gibbs(&[flag, flag])).contains("twice"));
    assert!(message(gibbs(&[])).contains("at least one"));
    gibbs(&[flag]).unwrap();
    // a larger maximum takes the binomial
    Gibbs::new(&[large])
        .with_max_support_size(10_000)
        .sampler(&model, &mut state.clone(), &[x], &mut rng(1))
        .unwrap();

    let walk = |latents: &[NodeId]| {
        DiscreteMetropolis::new(latents)
            .sampler(&model, &mut state.clone(), &[x], &mut rng(1))
            .map(|_| ())
    };
    assert!(message(walk(&[flag])).contains("Gibbs"));
    assert!(message(walk(&[x])).contains("continuous"));
    walk(&[count, large]).unwrap();
    assert!(matches!(
        DiscreteMetropolis::new(&[count]).with_max_step(0).sampler(
            &model,
            &mut state.clone(),
            &[x],
            &mut rng(1)
        ),
        Err(AlgorithmError::InvalidSetting(_))
    ));
}
