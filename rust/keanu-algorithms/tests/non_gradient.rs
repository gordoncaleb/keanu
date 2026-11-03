//! The non-gradient optimizer (BOBYQA) on models with closed form answers and its failure modes.

use keanu_algorithms::AlgorithmError;
use keanu_algorithms::optimizer::{Bobyqa, NonGradientOptimizer, OptimizerBounds};
use keanu_graph::{GraphBuilder, Model, NodeId, State};
use keanu_tensor::DoubleTensor;

/// mu ~ N(2, 3), x_i ~ N(mu, 0.5) observed; also an unrelated latent `other` ~ N(0, 1) so there are two
/// dimensions, as Java's BOBYQA required.
fn conjugate_gaussian(observations: &[f64]) -> (Model, State, NodeId, NodeId) {
    let b = GraphBuilder::new();
    let mu = b.gaussian(2.0, 3.0).label("mu");
    b.set_value(mu, DoubleTensor::scalar(-1.0));
    let other = b.gaussian(0.0, 1.0).label("other");
    b.set_value(other, DoubleTensor::scalar(0.7));
    let x = b.gaussian_with_shape(&[observations.len()], mu, 0.5);
    b.observe(x, DoubleTensor::vector(observations.to_vec()));
    let (mu, other) = (mu.id(), other.id());
    let (graph, state) = b.build().unwrap();
    (Model::new(graph, &state).unwrap(), state, mu, other)
}

fn posterior_mode(observations: &[f64]) -> f64 {
    let (prior_mu, prior_precision) = (2.0, 1.0 / 9.0);
    let precision = 4.0;
    (prior_mu * prior_precision + precision * observations.iter().sum::<f64>())
        / (prior_precision + observations.len() as f64 * precision)
}

#[test]
fn finds_the_conjugate_gaussian_posterior_mode() {
    let observations = [0.3, 1.2, 0.9, 1.7];
    for bobyqa in [
        Bobyqa::default(),
        Bobyqa::default()
            .with_initial_trust_region_radius(0.5)
            .with_bounds_range(5.0),
        Bobyqa::default().with_interpolation_points(4),
        Bobyqa::default().with_interpolation_points(6),
    ] {
        let (model, mut state, mu, other) = conjugate_gaussian(&observations);
        let result = NonGradientOptimizer::new(&model)
            .algorithm(bobyqa.clone())
            .max_a_posteriori(&mut state)
            .unwrap();
        let found = result.get(mu).unwrap().to_scalar().unwrap();
        assert!(
            (found - posterior_mode(&observations)).abs() < 1e-6,
            "{bobyqa:?}: {found}"
        );
        assert!(result.get(other).unwrap().to_scalar().unwrap().abs() < 1e-6);
        assert!(result.converged);
        // the state is left at the optimum
        assert_eq!(state.f64(mu).unwrap().to_scalar().unwrap(), found);
        assert!((result.fitness - model.log_prob(&state).unwrap()).abs() < 1e-12);
    }
}

#[test]
fn maximum_likelihood_is_the_sample_mean() {
    let observations = [0.3, 1.2, 0.9, 1.7];
    let (model, mut state, mu, _) = conjugate_gaussian(&observations);
    let result = NonGradientOptimizer::new(&model)
        .max_likelihood(&mut state)
        .unwrap();
    let mean = observations.iter().sum::<f64>() / 4.0;
    assert!((result.get(mu).unwrap().to_scalar().unwrap() - mean).abs() < 1e-6);
}

#[test]
fn a_single_latent_dimension_works() {
    // Java's BOBYQA required two or more dimensions; PRIMA's does not
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 10.0);
    b.set_value(mu, DoubleTensor::scalar(3.0));
    let x = b.gaussian(mu, 1.0);
    b.observe(x, DoubleTensor::scalar(1.0));
    let mu = mu.id();
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let result = NonGradientOptimizer::new(&model)
        .max_a_posteriori(&mut state)
        .unwrap();
    // posterior mode: (0 / 100 + 1) / (1 / 100 + 1)
    assert!((result.get(mu).unwrap().to_scalar().unwrap() - 1.0 / 1.01).abs() < 1e-6);
}

#[test]
fn explicit_bounds_are_respected_and_can_be_active() {
    let observations = [0.3, 1.2, 0.9, 1.7];
    let (model, mut state, mu, other) = conjugate_gaussian(&observations);
    let bounds = OptimizerBounds::new()
        .add_scalar(mu, -5.0, 0.5)
        .add_scalar(other, 0.2, 3.0);
    let result = NonGradientOptimizer::new(&model)
        .bounds(bounds)
        .max_a_posteriori(&mut state)
        .unwrap();
    // both unconstrained optima are outside the bounds, so both bounds are active
    assert_eq!(result.get(mu).unwrap().to_scalar().unwrap(), 0.5);
    assert_eq!(result.get(other).unwrap().to_scalar().unwrap(), 0.2);
}

#[test]
fn bounds_range_limits_every_unbounded_latent() {
    let (model, mut state, mu, _) = conjugate_gaussian(&[5.0, 5.0]);
    // the start is -1, so the range [-1.5, -0.5] excludes the optimum near 5
    let result = NonGradientOptimizer::new(&model)
        .algorithm(
            Bobyqa::default()
                .with_bounds_range(0.5)
                .with_initial_trust_region_radius(0.1),
        )
        .max_a_posteriori(&mut state)
        .unwrap();
    assert!((result.get(mu).unwrap().to_scalar().unwrap() - -0.5).abs() < 1e-12);
}

#[test]
fn tensor_bounds_must_match_the_latent_shape() {
    let b = GraphBuilder::new();
    let w = b.gaussian_with_shape(&[3], 0.0, 1.0);
    b.set_value(w, DoubleTensor::vector(vec![0.1, 0.2, 0.3]));
    let w = w.id();
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();

    let good = OptimizerBounds::new().add(
        w,
        DoubleTensor::vector(vec![-1.0, 0.5, -1.0]),
        DoubleTensor::scalar(1.0),
    );
    let result = NonGradientOptimizer::new(&model)
        .bounds(good)
        .max_a_posteriori(&mut state)
        .unwrap();
    let values = result.get(w).unwrap().as_slice().to_vec();
    assert!(
        values[0].abs() < 1e-6 && values[1] == 0.5 && values[2].abs() < 1e-6,
        "{values:?}"
    );

    let bad = OptimizerBounds::new().add(
        w,
        DoubleTensor::vector(vec![-1.0, -1.0]),
        DoubleTensor::scalar(1.0),
    );
    let error = NonGradientOptimizer::new(&model)
        .bounds(bad)
        .max_a_posteriori(&mut state);
    assert!(matches!(error, Err(AlgorithmError::InvalidArgument(_))));

    let empty = OptimizerBounds::new().add_scalar(w, 1.0, 1.0);
    let error = NonGradientOptimizer::new(&model)
        .bounds(empty)
        .max_a_posteriori(&mut state);
    assert!(matches!(error, Err(AlgorithmError::InvalidArgument(_))));
}

#[test]
fn impossible_start_discrete_latents_and_no_latents_are_errors() {
    let b = GraphBuilder::new();
    let a = b.uniform(0.0, 1.0);
    b.set_value(a, DoubleTensor::scalar(5.0));
    let a = a.id();
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    assert_eq!(
        NonGradientOptimizer::new(&model)
            .max_a_posteriori(&mut state)
            .unwrap_err(),
        AlgorithmError::ImpossibleStart
    );
    // without the check the optimizer runs, and moves back into the support when an initial interpolation point
    // lands inside it (from 1.4 with radius 1 the point 0.4 does)
    state
        .set(model.graph(), a, DoubleTensor::scalar(1.4))
        .unwrap();
    let result = NonGradientOptimizer::new(&model)
        .check_initial_conditions(false)
        .algorithm(Bobyqa::default().with_initial_trust_region_radius(1.0))
        .max_a_posteriori(&mut state)
        .unwrap();
    assert!(result.fitness.is_finite());
    let value = result.get(a).unwrap().to_scalar().unwrap();
    assert!((0.0..=1.0).contains(&value), "{value}");

    let b = GraphBuilder::new();
    let p = b.uniform(0.0, 1.0);
    b.set_value(p, DoubleTensor::scalar(0.5));
    let _coin = b.bernoulli(p).label("coin");
    let (graph, mut state) = b.build().unwrap();
    let all: Vec<_> = graph.ids().collect();
    state
        .lazy_eval(
            &graph,
            &all,
            &mut keanu_distributions::KeanuRng::seed_from_u64(1),
        )
        .unwrap();
    let model = Model::new(graph, &state).unwrap();
    assert_eq!(
        NonGradientOptimizer::new(&model)
            .max_a_posteriori(&mut state)
            .unwrap_err(),
        AlgorithmError::DiscreteLatents(vec!["coin".into()])
    );

    let b = GraphBuilder::new();
    let x = b.gaussian(0.0, 1.0);
    b.observe(x, DoubleTensor::scalar(1.0));
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    assert_eq!(
        NonGradientOptimizer::new(&model)
            .max_a_posteriori(&mut state)
            .unwrap_err(),
        AlgorithmError::NoLatents
    );
}

#[test]
fn invalid_settings_are_errors() {
    let (model, mut state, _, _) = conjugate_gaussian(&[1.0]);
    for bad in [
        Bobyqa::default().with_max_evaluations(0),
        Bobyqa::default().with_bounds_range(0.0),
        Bobyqa::default().with_initial_trust_region_radius(-1.0),
        Bobyqa::default().with_stopping_trust_region_radius(100.0),
    ] {
        let error = NonGradientOptimizer::new(&model)
            .algorithm(bad)
            .max_a_posteriori(&mut state);
        assert!(matches!(error, Err(AlgorithmError::InvalidSetting(_))));
    }
}

#[test]
fn max_evaluations_stops_early_without_converging() {
    let (model, mut state, _, _) = conjugate_gaussian(&[0.3, 1.2]);
    let result = NonGradientOptimizer::new(&model)
        .algorithm(Bobyqa::default().with_max_evaluations(8))
        .max_a_posteriori(&mut state)
        .unwrap();
    assert!(!result.converged);
    // the initial-condition check, 8 solver evaluations and the final fitness
    assert_eq!(result.evaluations, 10);
}
