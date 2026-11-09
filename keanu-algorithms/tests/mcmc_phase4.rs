mod mcmc_common;

use keanu_algorithms::AlgorithmError;
use keanu_algorithms::mcmc::differential_evolution::DifferentialEvolutionMetropolis;
use keanu_algorithms::mcmc::hmc::Hmc;
use keanu_algorithms::mcmc::nuts::Nuts;
use keanu_algorithms::mcmc::slice::Slice;
use keanu_algorithms::mcmc::{SamplingAlgorithm, SamplingOptions, generate};
use keanu_algorithms::statistics::effective_sample_size;
use keanu_graph::{GraphBuilder, Model, NodeId, Space, State};
use keanu_tensor::DoubleTensor;
use mcmc_common::{assert_moments, beta_binomial_model, mean, rng, series};

fn correlated() -> (Model, State, [NodeId; 2]) {
    let b = GraphBuilder::new();
    let x = b.gaussian(0.0, 1.0);
    let y = b.gaussian(0.0, 1.0);
    b.set_value(x, DoubleTensor::scalar(0.5));
    b.set_value(y, DoubleTensor::scalar(-0.5));
    let z = b.gaussian(x + y, 0.5);
    b.observe(z, DoubleTensor::scalar(0.0));
    let ids = [x.id(), y.id()];
    let (graph, state) = b.build().unwrap();
    (Model::new(graph, &state).unwrap(), state, ids)
}

#[test]
fn the_new_samplers_match_nuts_and_the_exact_correlated_posterior() {
    let (model, state, ids) = correlated();
    let options = SamplingOptions::default().with_drop(300);
    let mut hmc_state = state.clone();
    let (hmc, statistics) = Hmc::default()
        .with_leapfrog_steps(5)
        .with_adapt_count(300)
        .sample_with_statistics(&model, &mut hmc_state, &ids, 3300, options, &mut rng(5))
        .unwrap();
    assert!(statistics.accepted > 2200 && statistics.accepted < 3300);
    assert!(statistics.step_size > 0.0 && statistics.step_size < 1.0);
    assert!(statistics.gradient_evaluations <= 1 + 3300 * 5);

    let mut slice_state = state.clone();
    let mut random = rng(6);
    let mut sampler = Slice::default()
        .sampler(&model, &mut slice_state, &ids, &mut random)
        .unwrap();
    let slice = generate(&mut sampler, &mut slice_state, 3300, options, &mut random).unwrap();
    assert!(sampler.evaluations() > 6600);

    let mut states = vec![state.clone(); 8];
    for (i, walker) in states.iter_mut().enumerate() {
        walker
            .set(
                model.graph(),
                ids[0],
                DoubleTensor::scalar((i as f64 - 3.5) / 2.0),
            )
            .unwrap();
    }
    let (walkers, statistics) = DifferentialEvolutionMetropolis::default()
        .sample_with_statistics(&model, &mut states, &ids, 3300, options, &mut rng(7))
        .unwrap();
    assert_eq!(statistics.evaluations, 8 * 3301);
    assert!(statistics.accepted.iter().all(|&n| n > 400 && n < 2500));

    let (nuts, _) = Nuts::default()
        .with_adapt_count(300)
        .sample_with_statistics(&model, &mut state.clone(), &ids, 3300, options, &mut rng(8))
        .unwrap();

    for (label, chain) in [
        ("HMC", &hmc),
        ("slice", &slice),
        ("DE walker 0", &walkers[0]),
        ("NUTS", &nuts),
    ] {
        let x = series(chain, ids[0]);
        let y = series(chain, ids[1]);
        assert_moments(&x, 0.0, 5.0 / 9.0, label);
        assert_moments(&y, 0.0, 5.0 / 9.0, label);
        let cross = x.iter().zip(&y).map(|(x, y)| x * y).collect::<Vec<_>>();
        assert!(
            (mean(&cross) + 4.0 / 9.0).abs() < 0.10,
            "{label} covariance {}",
            mean(&cross)
        );
    }
}

#[test]
fn warmup_stops_and_divergent_hmc_proposals_restore_the_state() {
    let (model, state, ids) = correlated();
    let mut start = state.clone();
    let mut random = rng(20);
    let mut sampler = Hmc::default()
        .with_step_size(1e6)
        .sampler(&model, &mut start, &ids, &mut random)
        .unwrap();
    let old_log_prob = model.log_prob(&start).unwrap();
    let old_position = ids.map(|id| start.f64(id).unwrap().clone());
    sampler.step(&mut start, &mut random).unwrap();
    assert_eq!(ids.map(|id| start.f64(id).unwrap().clone()), old_position);
    assert_eq!(model.log_prob(&start).unwrap(), old_log_prob);
    assert_eq!(sampler.statistics().accepted, 0);
    assert_eq!(sampler.statistics().divergences, 1);

    let mut sampler = Hmc::default()
        .with_adapt_count(20)
        .sampler(&model, &mut start, &ids, &mut random)
        .unwrap();
    for _ in 0..20 {
        sampler.step(&mut start, &mut random).unwrap();
    }
    let adapted = sampler.statistics().step_size;
    assert_ne!(adapted, Hmc::default().step_size);
    sampler.step(&mut start, &mut random).unwrap();
    assert_eq!(sampler.statistics().step_size, adapted);
}

#[test]
fn transformed_sampling_keeps_support_and_records_model_density() {
    let (model, state, id, expected_mean, expected_variance) = beta_binomial_model();
    let hmc = Hmc::default().with_leapfrog_steps(5);
    let mut hmc_state = state.clone();
    let draws = hmc
        .sample(&model, &mut hmc_state, &[id], 3000, &mut rng(9))
        .unwrap();
    let repeated = hmc
        .sample(&model, &mut state.clone(), &[id], 3000, &mut rng(9))
        .unwrap();
    assert_eq!(draws, repeated);

    let mut slice_state = state.clone();
    let slice = Slice::default()
        .sample(&model, &mut slice_state, &[id], 3000, &mut rng(10))
        .unwrap();
    let mut states = vec![state; 5];
    for (i, walker) in states.iter_mut().enumerate() {
        walker
            .set(
                model.graph(),
                id,
                DoubleTensor::scalar(0.2 + 0.1 * i as f64),
            )
            .unwrap();
    }
    let de = DifferentialEvolutionMetropolis::default();
    let walkers = de
        .sample(&model, &mut states, &[id], 3500, &mut rng(11))
        .unwrap();
    for (label, chain, end) in [
        ("HMC", &draws, &hmc_state),
        ("slice", &slice, &slice_state),
        ("DE", &walkers[0], &states[0]),
    ] {
        let draws = series(chain, id);
        assert!(draws.iter().all(|x| 0.0 < *x && *x < 1.0));
        let draws = &draws[500..];
        let squared = draws
            .iter()
            .map(|x| (x - expected_mean).powi(2))
            .collect::<Vec<_>>();
        // Negative correlation of successive draws can raise the ESS of the mean while lowering
        // the ESS of the variance. Check each moment using its own Monte Carlo error.
        let mean_tolerance = 5.0 * (expected_variance / effective_sample_size(draws)).sqrt();
        let variance_tolerance =
            6.0 * (mcmc_common::variance(&squared) / effective_sample_size(&squared)).sqrt();
        assert!(
            (mean(draws) - expected_mean).abs() < mean_tolerance,
            "{label}: mean {}",
            mean(draws)
        );
        assert!(
            (mean(&squared) - expected_variance).abs() < variance_tolerance,
            "{label}: variance {}",
            mean(&squared)
        );
        let recorded = chain.log_probs()[chain.len() - 1];
        assert!((recorded - model.log_prob(end).unwrap()).abs() < 1e-12);
    }
}

#[test]
fn slice_and_population_settings_and_recording_are_validated() {
    let (model, state, ids) = correlated();
    let mut states = vec![state.clone(); 3];
    let mut random = rng(12);
    let de = DifferentialEvolutionMetropolis::default();
    assert!(matches!(
        de.sample(&model, &mut states[..2], &ids, 10, &mut random),
        Err(AlgorithmError::InvalidArgument(_))
    ));
    assert!(matches!(
        Slice::default()
            .with_width(0.0)
            .sampler(&model, &mut state.clone(), &ids, &mut random),
        Err(AlgorithmError::InvalidSetting(_))
    ));
    let options = SamplingOptions::default()
        .with_drop(5)
        .with_down_sample_interval(2);
    let run = |states: &mut [State]| {
        de.sample_with_statistics(&model, states, &ids, 12, options, &mut rng(13))
            .unwrap()
    };
    let (draws, _) = run(&mut states);
    let (again, _) = run(&mut vec![state.clone(); 3]);
    assert_eq!(draws, again);
    assert!(draws.iter().all(|samples| samples.len() == 4));

    let observed = model.observed()[0];
    states[1]
        .observe(model.graph(), observed, DoubleTensor::scalar(3.0))
        .unwrap();
    assert!(
        de.sample(&model, &mut states, &ids, 10, &mut random)
            .is_err()
    );

    let mut state = state;
    let mut sampler = Slice::default()
        .with_space(Space::Constrained)
        .with_width(f64::MIN_POSITIVE)
        .sampler(&model, &mut state, &ids, &mut random)
        .unwrap();
    let start = state.f64(ids[0]).unwrap().clone();
    assert!(sampler.step(&mut state, &mut random).is_err());
    assert_eq!(state.f64(ids[0]).unwrap(), &start);
}

#[test]
fn gradient_free_samplers_cross_a_discontinuous_likelihood() {
    let b = GraphBuilder::new();
    let x = b.uniform(-2.0, 2.0);
    b.set_value(x, DoubleTensor::scalar(0.25));
    b.factor(b.where_(x.gt(0.0), b.scalar(0.0), b.scalar(-3.0_f64.ln())));
    let id = x.id();
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();

    let slice = Slice::default();
    let sample = || {
        slice
            .sample(&model, &mut state.clone(), &[id], 4000, &mut rng(14))
            .unwrap()
    };
    let draws = sample();
    assert_eq!(draws, sample());
    let mut states = vec![state; 6];
    for (i, walker) in states.iter_mut().enumerate() {
        walker
            .set(
                model.graph(),
                id,
                DoubleTensor::scalar(-1.0 + 0.4 * i as f64),
            )
            .unwrap();
    }
    let walkers = DifferentialEvolutionMetropolis::default()
        .sample(&model, &mut states, &[id], 4000, &mut rng(15))
        .unwrap();
    for (label, chain) in [("slice", &draws), ("DE", &walkers[0])] {
        let x = series(chain, id);
        // P(x > 0) = 3/4 with a uniform density on each half.
        assert_moments(&x[500..], 0.5, 4.0 / 3.0 - 0.25, label);
    }
}

#[test]
fn impossible_population_proposals_leave_walkers_and_recorded_density_at_the_old_position() {
    let b = GraphBuilder::new();
    let x = b.uniform(-1.0, 1.0);
    b.set_value(x, DoubleTensor::scalar(0.25));
    let id = x.id();
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let mut states = vec![state; 3];
    let (samples, statistics) = DifferentialEvolutionMetropolis::default()
        .with_space(Space::Constrained)
        .with_noise(1e8)
        .sample_with_statistics(
            &model,
            &mut states,
            &[id],
            4,
            SamplingOptions::default(),
            &mut rng(16),
        )
        .unwrap();
    assert_eq!(statistics.accepted, vec![0; 3]);
    for (chain, state) in samples.iter().zip(&states) {
        assert_eq!(series(chain, id), vec![0.25; 4]);
        assert_eq!(chain.log_probs(), &[model.log_prob(state).unwrap(); 4]);
        assert_eq!(state.f64(id).unwrap().to_scalar().unwrap(), 0.25);
    }
}
