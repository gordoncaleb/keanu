//! The compound sampler (NUTS for continuous latents with Gibbs or discrete random-walk steps for discrete ones)
//! against exact or independently sampled posteriors, and the default sampler choice.

mod mcmc_common;

use keanu_algorithms::mcmc::SamplingOptions;
use keanu_algorithms::mcmc::chains::sample_chains;
use keanu_algorithms::mcmc::compound::{Compound, CompoundStep};
use keanu_algorithms::mcmc::metropolis::MetropolisHastings;
use keanu_algorithms::mcmc::nuts::{Metric, Nuts, QuadraticPotential};
use keanu_algorithms::mcmc::select::{Sampler, is_differentiable_wrt};
use keanu_algorithms::{AlgorithmError, ChainSamples};
use keanu_graph::op::DistributionKind;
use keanu_graph::{DoubleVar, GraphBuilder, Model, NodeId, State};
use keanu_tensor::{DoubleTensor, IntTensor};
use mcmc_common::*;

/// Draws from 0.4 N(-2, 1) + 0.6 N(3, 1).
fn mixture_data(count: usize, seed: u64) -> Vec<f64> {
    let mut r = rng(seed);
    (0..count)
        .map(|_| {
            let mean = if r.next_f64() < 0.4 { -2.0 } else { 3.0 };
            mean + r.next_gaussian()
        })
        .collect()
}

/// A two-component mixture with unit scales, ordered means (the first, plus a positive gap) and Dirichlet(2, 2)
/// weights, written either with `Mixture` (marginalised) or with an explicit Categorical assignment per point.
/// Returns the model, the state and the means node.
fn mixture_model(data: &[f64], explicit: bool) -> (Model, State, NodeId) {
    let b = GraphBuilder::new();
    let first = b.gaussian_with_shape(&[1], 0.0, 10.0);
    let gap = b.half_gaussian_with_shape(&[1], 10.0);
    let means = b.concat(0, &[first, gap]).cumsum(0);
    let weights = b.dirichlet(DoubleTensor::vector(vec![2.0, 2.0]));
    let n = data.len();
    let y: DoubleVar = if explicit {
        let z = b.categorical(&[n], &[weights.take(&[0]), weights.take(&[1])]);
        b.set_value(z, IntTensor::vector(vec![0; n]));
        b.gaussian_with_shape(&[n], means.gather(0, z), 1.0)
    } else {
        let sigmas = b.constant(DoubleTensor::vector(vec![1.0, 1.0]));
        b.mixture_with_shape(
            &[n],
            DistributionKind::Gaussian,
            &[means.id(), sigmas.id()],
            weights,
        )
    };
    b.observe(y, DoubleTensor::vector(data.to_vec()));
    b.set_value(first, DoubleTensor::vector(vec![0.0]));
    b.set_value(gap, DoubleTensor::vector(vec![1.0]));
    b.set_value(weights, DoubleTensor::vector(vec![0.5, 0.5]));
    let means = means.id();
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    (model, state, means)
}

/// The posterior mean of an element over all chains, and its Monte Carlo standard error.
fn mean_and_mcse(chains: &ChainSamples, id: NodeId, index: &[usize]) -> (f64, f64) {
    let summary = chains.summarize(id, index).unwrap();
    (summary.mean, summary.mcse_mean)
}

/// AE1, the sampler half: the compound sampler on the model with explicit assignments recovers the component means
/// NUTS finds on the marginalised model.
#[test]
fn compound_sampling_of_explicit_assignments_matches_the_marginalised_mixture() {
    let data = mixture_data(60, 3);
    let (explicit, explicit_state, explicit_means) = mixture_model(&data, true);
    let (marginal, marginal_state, marginal_means) = mixture_model(&data, false);

    let sampler = Sampler::for_model(&explicit, &explicit_state);
    let Sampler::Compound(compound) = sampler else {
        panic!("a mixed model defaults to the compound sampler, not {sampler:?}");
    };
    assert!(matches!(
        compound.steps(),
        [CompoundStep::Gibbs(_), CompoundStep::Nuts { .. }]
    ));
    let compound = compound.with_nuts_settings(Nuts::default().with_adapt_count(300));
    let options = SamplingOptions::default().with_drop(300);
    let explicit_chains = sample_chains(4, 5, &explicit_state, |state, rng| {
        compound
            .sample_with_statistics(&explicit, state, &[explicit_means], 1300, options, rng)
            .map(|(samples, _)| samples)
    })
    .unwrap();
    let nuts = Nuts::default().with_adapt_count(300);
    let marginal_chains = sample_chains(4, 6, &marginal_state, |state, rng| {
        nuts.sample_with_statistics(&marginal, state, &[marginal_means], 1300, options, rng)
            .map(|(samples, _)| samples)
    })
    .unwrap();
    for c in 0..2 {
        let rhat = explicit_chains.rhat(explicit_means, &[c]).unwrap();
        assert!(rhat < 1.01, "component {c}: R-hat {rhat}");
        let (a, a_error) = mean_and_mcse(&explicit_chains, explicit_means, &[c]);
        let (b, b_error) = mean_and_mcse(&marginal_chains, marginal_means, &[c]);
        let tolerance = 4.0 * (a_error * a_error + b_error * b_error).sqrt();
        assert!(
            (a - b).abs() < tolerance,
            "component {c}: explicit mean {a} vs marginalised {b} (tolerance {tolerance})"
        );
    }
}

/// A spike-and-slab regression: β_j = γ_j b_j with γ_j ~ Bernoulli(0.5) and b_j ~ N(0, 2²); the posterior inclusion
/// probabilities pick out the coefficients the data were simulated with.
#[test]
fn spike_and_slab_regression_recovers_the_inclusion_pattern() {
    let truth = [2.0, 0.0, -1.5, 0.0, 0.0];
    let (n, p) = (60, truth.len());
    let mut r = rng(7);
    let x: Vec<f64> = (0..n * p).map(|_| r.next_gaussian()).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| (0..p).map(|j| x[i * p + j] * truth[j]).sum::<f64>() + 0.5 * r.next_gaussian())
        .collect();

    let b = GraphBuilder::new();
    let included = b.bernoulli_with_shape(&[p], 0.5);
    let slab = b.gaussian_with_shape(&[p], 0.0, 2.0);
    let beta = slab * included.to_double_mask();
    let design = b.constant(DoubleTensor::new(&[n, p], x).unwrap());
    let linear = design
        .matmul(beta.reshape(&[p as isize, 1]))
        .reshape(&[n as isize]);
    let observed = b.gaussian_with_shape(&[n], linear, 0.5);
    b.observe(observed, DoubleTensor::vector(y));
    b.set_value(included, keanu_tensor::BoolTensor::full(&[p], true));
    b.set_value(slab, DoubleTensor::full(&[p], 0.0));
    let (included, slab) = (included.id(), slab.id());
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();

    let compound = Compound::for_model(&model, &state)
        .with_nuts_settings(Nuts::default().with_adapt_count(300));
    let chains = sample_chains(2, 8, &state, |state, rng| {
        compound
            .sample_with_statistics(
                &model,
                state,
                &[included, slab],
                1800,
                SamplingOptions::default().with_drop(300),
                rng,
            )
            .map(|(samples, _)| samples)
    })
    .unwrap();
    for (j, &coefficient) in truth.iter().enumerate() {
        let inclusion: Vec<f64> = chains.element(included, &[j]).unwrap().concat();
        let probability = mean(&inclusion);
        if coefficient != 0.0 {
            assert!(
                probability > 0.95,
                "coefficient {j}: inclusion {probability}"
            );
            let slabs = chains.element(slab, &[j]).unwrap().concat();
            assert!(
                (mean(&slabs) - coefficient).abs() < 0.3,
                "coefficient {j}: {} vs {coefficient}",
                mean(&slabs)
            );
        } else {
            assert!(
                probability < 0.5,
                "coefficient {j}: inclusion {probability}"
            );
        }
    }
}

/// λ ~ Gamma(scale 2, shape 3), y_i ~ Poisson(λ) observed and a new count ~ Poisson(λ): the new count's posterior
/// predictive distribution is negative binomial, sampled with NUTS for λ and the discrete random walk for the count.
#[test]
fn compound_sampling_gives_the_gamma_poisson_posterior_predictive() {
    let counts = [4i64, 2, 5, 3, 6];
    let b = GraphBuilder::new();
    let rate = b.gamma(2.0, 3.0);
    let y = b.poisson_with_shape(&[counts.len()], rate);
    b.observe(y, IntTensor::vector(counts.to_vec()));
    let new = b.poisson(rate);
    b.set_value(rate, DoubleTensor::scalar(3.0));
    b.set_value(new, IntTensor::scalar(3));
    let (rate, new) = (rate.id(), new.id());
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();

    let compound = Compound::for_model(&model, &state)
        .with_nuts_settings(Nuts::default().with_adapt_count(300));
    assert!(matches!(
        compound.steps(),
        [
            CompoundStep::DiscreteMetropolis(_),
            CompoundStep::Nuts { .. }
        ]
    ));
    let (samples, statistics) = compound
        .sample_with_statistics(
            &model,
            &mut state,
            &[rate, new],
            10_300,
            SamplingOptions::default().with_drop(300),
            &mut rng(9),
        )
        .unwrap();
    assert_eq!(statistics.nuts.len(), 1);
    assert_eq!(statistics.nuts[0].energy.len(), 10_300);
    assert_eq!(statistics.nuts[0].divergence_count(), 0);
    assert_eq!(statistics.discrete_acceptance_rate.len(), 1);

    // the posterior of λ is Gamma(shape 3 + Σy, rate 1/2 + n)
    let shape = 3.0 + counts.iter().sum::<i64>() as f64;
    let rate_parameter = 0.5 + counts.len() as f64;
    let lambda_mean = shape / rate_parameter;
    let lambda_variance = shape / (rate_parameter * rate_parameter);
    assert_moments(&series(&samples, rate), lambda_mean, lambda_variance, "λ");
    let draws: Vec<f64> = samples
        .i64(new)
        .unwrap()
        .as_slice()
        .iter()
        .map(|t| t.to_scalar().unwrap() as f64)
        .collect();
    assert_moments(
        &draws,
        lambda_mean,
        lambda_mean + lambda_variance,
        "predictive count",
    );
}

#[test]
fn the_default_sampler_depends_on_the_kinds_of_latents() {
    // all continuous and differentiable: NUTS, as before
    let (model, state, ..) = gaussian_model();
    assert!(matches!(
        Sampler::for_model(&model, &state),
        Sampler::Nuts(_)
    ));

    // all discrete with finite supports: Gibbs over all of them
    let b = GraphBuilder::new();
    let k = b.uniform_int(0, 4);
    let flag = b.bernoulli(0.3);
    let x = b.poisson(k.to_double() + flag.to_double_mask() + 1.0);
    b.observe(x, IntTensor::scalar(3));
    let (k, flag) = (k.id(), flag.id());
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    match Sampler::for_model(&model, &state) {
        Sampler::Gibbs(gibbs) => assert_eq!(gibbs.latents, [k, flag]),
        other => panic!("expected Gibbs, got {other:?}"),
    }

    // a count with infinitely many values gets the random walk; a multinomial and a continuous latent behind a
    // floor get Metropolis–Hastings
    let b = GraphBuilder::new();
    let count = b.poisson(3.0);
    let draws: keanu_graph::IntVar = b.multinomial(4, DoubleTensor::vector(vec![0.5, 0.5]));
    let a = b.gaussian(0.0, 1.0);
    let x = b.gaussian(count.to_double() + draws.to_double().sum() + a.floor(), 1.0);
    b.observe(x, DoubleTensor::scalar(1.0));
    let (count, draws, a) = (count.id(), draws.id(), a.id());
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    assert!(!is_differentiable_wrt(&model, &state, &[a]));
    let Sampler::Compound(compound) = Sampler::for_model(&model, &state) else {
        panic!("expected a compound sampler");
    };
    match compound.steps() {
        [
            CompoundStep::DiscreteMetropolis(walk),
            CompoundStep::MetropolisHastings { latents, .. },
        ] => {
            assert_eq!(walk.latents, [count]);
            assert_eq!(latents, &[draws, a]);
        }
        other => panic!("unexpected steps {other:?}"),
    }
}

#[test]
fn differentiability_ignores_operations_of_other_latents() {
    // the comparison of a discrete change point with the time isn't differentiable, but only the rates matter
    let b = GraphBuilder::new();
    let switch = b.uniform_int(0, 5);
    let early = b.exponential(1.0);
    let late = b.exponential(1.0);
    let t = b.int_constant(IntTensor::vector(vec![0, 1, 2, 3]));
    let y = b.poisson(b.where_(switch.gt(t), early, late));
    b.observe(y, IntTensor::vector(vec![3, 2, 0, 1]));
    let (early, late) = (early.id(), late.id());
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    assert!(is_differentiable_wrt(&model, &state, &[early, late]));
    assert!(is_differentiable_wrt(&model, &state, &[early]));
    assert!(!is_differentiable_wrt(&model, &state, &[]));
}

#[test]
fn compound_steps_must_cover_the_latents_once() {
    let b = GraphBuilder::new();
    let k = b.uniform_int(0, 3);
    let mu = b.gaussian(0.0, 1.0);
    let y = b.gaussian(mu + k.to_double(), 1.0);
    b.observe(y, DoubleTensor::scalar(1.0));
    let (k, mu) = (k.id(), mu.id());
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let error =
        |compound: Compound| match compound.sampler(&model, &mut state.clone(), &[mu], &mut rng(1))
        {
            Err(AlgorithmError::InvalidArgument(message)) => message,
            Err(other) => panic!("unexpected error {other:?}"),
            Ok(_) => panic!("expected an error"),
        };
    assert!(error(Compound::new()).contains("at least one step"));
    assert!(error(Compound::new().gibbs(&[k])).contains("no step updates"));
    assert!(
        error(
            Compound::new()
                .gibbs(&[k])
                .nuts(&[mu])
                .metropolis_hastings(MetropolisHastings::default(), &[mu])
        )
        .contains("both")
    );
    Compound::new()
        .gibbs(&[k])
        .metropolis_hastings(MetropolisHastings::default(), &[mu])
        .sampler(&model, &mut state.clone(), &[mu], &mut rng(1))
        .unwrap();
}

/// A dense-metric NUTS block inside a compound sampler: a, b ~ N(0, 10²) and y_i ~ N(a + b x_i, 1) with x far from
/// centred (so a and b are strongly correlated), and k ~ UniformInt[0, 3) with w_j ~ N(k, 0.7), sampled by Gibbs.
/// The posterior factorises: (a, b) is Gaussian, k is discrete, both exact.
#[test]
fn a_dense_nuts_block_samples_correlated_latents_in_a_compound_sampler() {
    let xs = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let ys = [2.9, 3.6, 4.2, 4.4, 5.3, 5.6];
    let ws = [1.2, 0.8, 1.5];
    let b = GraphBuilder::new();
    let a = b.gaussian(0.0, 10.0);
    let slope = b.gaussian(0.0, 10.0);
    let x = b.constant(DoubleTensor::vector(xs.to_vec()));
    let y = b.gaussian(a + slope * x, 1.0);
    b.observe(y, DoubleTensor::vector(ys.to_vec()));
    let k = b.uniform_int(0, 3);
    let w = b.gaussian_with_shape(&[ws.len()], k.to_double(), 0.7);
    b.observe(w, DoubleTensor::vector(ws.to_vec()));
    b.set_value(a, DoubleTensor::scalar(0.0));
    b.set_value(slope, DoubleTensor::scalar(0.0));
    b.set_value(k, IntTensor::scalar(0));
    let (a, slope, k) = (a.id(), slope.id(), k.id());
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();

    let nuts = Nuts::default()
        .with_adapt_count(500)
        .with_potential(QuadraticPotential::default().with_metric(Metric::Dense));
    let compound = Compound::new().nuts_with(nuts, &[a, slope]).gibbs(&[k]);
    let chains = sample_chains(2, 41, &state, |state, rng| {
        let (samples, statistics) = compound.sample_with_statistics(
            &model,
            state,
            &[a, slope, k],
            3500,
            SamplingOptions::default().with_drop(500),
            rng,
        )?;
        let metric = statistics.nuts[0].dense_inverse_metric.clone().unwrap();
        assert_eq!(metric.shape(), [2, 2]);
        // the adapted metric has the posterior's strong negative correlation
        let m = metric.as_slice();
        assert!(m[1] / (m[0] * m[3]).sqrt() < -0.9, "{m:?}");
        Ok(samples)
    })
    .unwrap();

    // the exact posterior of (a, b): precision P = XᵀX + I / 100, mean P⁻¹ Xᵀ y
    let n = xs.len() as f64;
    let (sx, sxx) = (
        xs.iter().sum::<f64>(),
        xs.iter().map(|x| x * x).sum::<f64>(),
    );
    let (sy, sxy) = (
        ys.iter().sum::<f64>(),
        xs.iter().zip(&ys).map(|(x, y)| x * y).sum::<f64>(),
    );
    let p = [n + 0.01, sx, sx, sxx + 0.01];
    let det = p[0] * p[3] - p[1] * p[2];
    let covariance = [p[3] / det, -p[1] / det, -p[2] / det, p[0] / det];
    let mean = [
        covariance[0] * sy + covariance[1] * sxy,
        covariance[2] * sy + covariance[3] * sxy,
    ];
    for (id, (m, v)) in [
        (a, (mean[0], covariance[0])),
        (slope, (mean[1], covariance[3])),
    ] {
        let summary = chains.summarize(id, &[]).unwrap();
        assert!(
            (summary.mean - m).abs() < 5.0 * summary.mcse_mean,
            "{} vs {m} (mcse {})",
            summary.mean,
            summary.mcse_mean
        );
        let relative = summary.sd * summary.sd / v;
        assert!(
            (0.8..1.25).contains(&relative),
            "variance {relative} of the true one"
        );
        assert!(summary.r_hat < 1.01, "{}", summary.r_hat);
    }
    // k's posterior: P(k) ∝ Π N(w_j | k, 0.7)
    let log_weight = |k: f64| -ws.iter().map(|w| (w - k) * (w - k)).sum::<f64>() / (2.0 * 0.49);
    let weights: Vec<f64> = (0..3).map(|k| log_weight(k as f64).exp()).collect();
    let total: f64 = weights.iter().sum();
    let draws: Vec<i64> = chains
        .chains()
        .iter()
        .flat_map(|c| {
            c.i64(k)
                .unwrap()
                .as_slice()
                .iter()
                .map(|t| t.to_scalar().unwrap())
                .collect::<Vec<_>>()
        })
        .collect();
    for (value, weight) in weights.iter().enumerate() {
        let frequency =
            draws.iter().filter(|&&d| d == value as i64).count() as f64 / draws.len() as f64;
        let probability = weight / total;
        assert!(
            (frequency - probability).abs() < 0.03,
            "P(k = {value}): {frequency} vs {probability}"
        );
    }
}
