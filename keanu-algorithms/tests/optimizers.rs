//! Optimizer behaviour that has a closed form answer or a specific failure mode.

use keanu_algorithms::AlgorithmError;
use keanu_algorithms::Space;
use keanu_algorithms::optimizer::{
    Adam, ConjugateGradient, ConvergenceChecker, GradientOptimizer, Norm, UpdateFormula,
};
use keanu_graph::{GraphBuilder, Model, State};
use keanu_tensor::{DoubleTensor, IntTensor};

/// mu ~ N(prior_mu, prior_sigma), x_i ~ N(mu, sigma) observed. The posterior mode is the precision weighted mean.
fn conjugate_gaussian(observations: &[f64]) -> (Model, State, keanu_graph::NodeId) {
    let b = GraphBuilder::new();
    let mu = b.gaussian(2.0, 3.0).label("mu");
    b.set_value(mu, DoubleTensor::scalar(-1.0));
    let x = b.gaussian_with_shape(&[observations.len()], mu, 0.5);
    b.observe(x, DoubleTensor::vector(observations.to_vec()));
    let id = mu.id();
    let (graph, state) = b.build().unwrap();
    (Model::new(graph, &state).unwrap(), state, id)
}

fn posterior_mode(observations: &[f64]) -> f64 {
    let (prior_mu, prior_precision) = (2.0, 1.0 / 9.0);
    let precision = 1.0 / 0.25;
    let n = observations.len() as f64;
    (prior_mu * prior_precision + precision * observations.iter().sum::<f64>())
        / (prior_precision + n * precision)
}

#[test]
fn both_algorithms_find_the_conjugate_gaussian_posterior_mode() {
    let observations = [0.3, 1.2, 0.9, 1.7];
    let expected = posterior_mode(&observations);
    let algorithms: Vec<keanu_algorithms::optimizer::Algorithm> = vec![
        Adam::default().with_max_evaluations(100_000).into(),
        ConjugateGradient::default().into(),
        ConjugateGradient::default()
            .with_formula(UpdateFormula::FletcherReeves)
            .into(),
    ];
    for algorithm in algorithms {
        let (model, mut state, mu) = conjugate_gaussian(&observations);
        let result = GradientOptimizer::new(&model)
            .algorithm(algorithm.clone())
            .max_a_posteriori(&mut state)
            .unwrap();
        let found = result.get(mu).unwrap().to_scalar().unwrap();
        assert!(
            (found - expected).abs() < 1e-3,
            "{algorithm:?}: {found} vs {expected}"
        );
        // the state is left at the optimum, and the fitness is the one there
        assert_eq!(state.f64(mu).unwrap().to_scalar().unwrap(), found);
        assert_eq!(result.fitness, model.log_prob(&state).unwrap());
        assert!(result.converged);
    }
}

/// Regression: conjugate gradient re-evaluated points it had already evaluated (the start after the
/// initial-condition check, the line search's chosen step, the final point), and its line search spent 40 to 60
/// evaluations refining a step that could no longer change the fitness once the optimum was reached: a
/// one-dimensional Gaussian took 78 evaluations, a conjugate Gaussian 60.
#[test]
fn conjugate_gradient_does_not_waste_evaluations_at_the_optimum() {
    let b = GraphBuilder::new();
    let a = b.gaussian(0.0, 1.0);
    b.set_value(a, DoubleTensor::scalar(0.5));
    let id = a.id();
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let result = GradientOptimizer::new(&model)
        .max_a_posteriori(&mut state)
        .unwrap();
    assert!(result.converged);
    assert!(result.get(id).unwrap().to_scalar().unwrap().abs() < 1e-6);
    assert!(
        result.evaluations <= 20,
        "{} evaluations",
        result.evaluations
    );

    let observations = [0.3, 1.2, 0.9, 1.7];
    let (model, mut state, mu) = conjugate_gaussian(&observations);
    let result = GradientOptimizer::new(&model)
        .max_a_posteriori(&mut state)
        .unwrap();
    let found = result.get(mu).unwrap().to_scalar().unwrap();
    assert!((found - posterior_mode(&observations)).abs() < 1e-6);
    assert!(
        result.evaluations <= 20,
        "{} evaluations",
        result.evaluations
    );
}

#[test]
fn maximum_likelihood_is_the_sample_mean() {
    let observations = [0.3, 1.2, 0.9, 1.7];
    let (model, mut state, mu) = conjugate_gaussian(&observations);
    let result = GradientOptimizer::new(&model)
        .max_likelihood(&mut state)
        .unwrap();
    let mean = observations.iter().sum::<f64>() / 4.0;
    assert!((result.get(mu).unwrap().to_scalar().unwrap() - mean).abs() < 1e-4);
}

#[test]
fn optimizing_a_multivariate_regression() {
    // y = X w + noise with a flat-ish prior: the MAP is close to least squares
    let b = GraphBuilder::new();
    let x = b.constant(
        DoubleTensor::new(
            &[5, 2],
            vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0],
        )
        .unwrap(),
    );
    let w = b.gaussian_with_shape(&[2, 1], 0.0, 100.0);
    b.set_value(w, DoubleTensor::new(&[2, 1], vec![0.0, 0.0]).unwrap());
    let y = b.gaussian(x.matmul(w), 0.1);
    b.observe(
        y,
        DoubleTensor::new(&[5, 1], vec![1.0, 3.0, 5.0, 7.0, 9.0]).unwrap(),
    );
    let w_id = w.id();
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let result = GradientOptimizer::new(&model)
        .max_a_posteriori(&mut state)
        .unwrap();
    let found = result.get(w_id).unwrap().as_slice().to_vec();
    assert!(
        (found[0] - 1.0).abs() < 1e-3 && (found[1] - 2.0).abs() < 1e-3,
        "{found:?}"
    );
}

#[test]
fn adam_converges_when_the_optimum_is_zero() {
    // regression for Java's relative convergence check, whose floor never let a position near zero converge
    let b = GraphBuilder::new();
    let a = b.gaussian(0.0, 1.0);
    b.set_value(a, DoubleTensor::scalar(0.5));
    let id = a.id();
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let result = GradientOptimizer::new(&model)
        .algorithm(Adam::default().with_max_evaluations(100_000))
        .max_a_posteriori(&mut state)
        .unwrap();
    assert!(result.converged);
    assert!(result.evaluations < 100_000);
    assert!(result.get(id).unwrap().to_scalar().unwrap().abs() < 1e-2);
}

#[test]
fn a_steep_negative_gradient_is_not_flat() {
    let b = GraphBuilder::new();
    let a = b.gaussian(0.0, 1.0);
    b.set_value(a, DoubleTensor::scalar(100.0));
    let c = b.gaussian(0.0, 1.0);
    b.set_value(c, DoubleTensor::scalar(0.0));
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    assert!(
        GradientOptimizer::new(&model)
            .max_a_posteriori(&mut state)
            .is_ok()
    );
}

#[test]
fn a_flat_starting_gradient_is_an_error() {
    let b = GraphBuilder::new();
    let a = b.gaussian(0.0, 1.0);
    b.set_value(a, DoubleTensor::scalar(0.0));
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let error = GradientOptimizer::new(&model)
        .max_a_posteriori(&mut state)
        .unwrap_err();
    assert!(
        matches!(error, AlgorithmError::FlatGradient { .. }),
        "{error:?}"
    );
    // without the check it runs (and stays at the optimum)
    let result = GradientOptimizer::new(&model)
        .check_initial_conditions(false)
        .max_a_posteriori(&mut state)
        .unwrap();
    assert_eq!(result.values[0].1.to_scalar().unwrap(), 0.0);
}

#[test]
fn an_impossible_start_is_an_error() {
    let b = GraphBuilder::new();
    let a = b.uniform(0.0, 1.0);
    b.set_value(a, DoubleTensor::scalar(5.0));
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    assert_eq!(
        GradientOptimizer::new(&model)
            .max_a_posteriori(&mut state)
            .unwrap_err(),
        AlgorithmError::ImpossibleStart
    );
}

#[test]
fn discrete_latents_and_no_latents_are_errors() {
    let b = GraphBuilder::new();
    let p = b.uniform(0.0, 1.0);
    b.set_value(p, DoubleTensor::scalar(0.5));
    let _coin = b.bernoulli(p).label("coin");
    let (graph, state) = b.build().unwrap();
    let mut state = state;
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
        GradientOptimizer::new(&model)
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
        GradientOptimizer::new(&model)
            .max_a_posteriori(&mut state)
            .unwrap_err(),
        AlgorithmError::NoLatents
    );
}

#[test]
fn invalid_settings_are_errors() {
    let (model, mut state, _) = conjugate_gaussian(&[1.0]);
    let error = GradientOptimizer::new(&model)
        .algorithm(Adam::default().with_alpha(0.0))
        .max_a_posteriori(&mut state);
    assert!(matches!(error, Err(AlgorithmError::InvalidSetting(_))));
}

#[test]
fn max_evaluations_stops_early_without_converging() {
    let (model, mut state, _) = conjugate_gaussian(&[0.3, 1.2]);
    let result = GradientOptimizer::new(&model)
        .algorithm(Adam::default().with_max_evaluations(3))
        .max_a_posteriori(&mut state)
        .unwrap();
    assert!(!result.converged);
}

#[test]
fn convergence_checkers() {
    let a = [DoubleTensor::vector(vec![0.0, 2.0])];
    let b = [DoubleTensor::vector(vec![0.0, 2.000_000_1])];
    assert!(ConvergenceChecker::relative(Norm::L2, 1e-6).has_converged(&a, &b));
    assert!(ConvergenceChecker::relative(Norm::Max, 1e-6).has_converged(&a, &a));
    assert!(!ConvergenceChecker::absolute(Norm::Max, 1e-8).has_converged(&a, &b));
    assert!(ConvergenceChecker::absolute(Norm::L2, 1e-6).has_converged(&a, &b));
}

/// x ~ N(0, sigma) observed at 1, starting from sigma = 0: an impossible start whose gradient is not finite.
fn zero_sigma() -> (Model, State, keanu_graph::NodeId) {
    let b = GraphBuilder::new();
    let sigma = b.uniform(0.0, 2.0);
    b.set_value(sigma, DoubleTensor::scalar(0.0));
    let x = b.gaussian(0.0, sigma);
    b.observe(x, DoubleTensor::scalar(1.0));
    let id = sigma.id();
    let (graph, state) = b.build().unwrap();
    (Model::new(graph, &state).unwrap(), state, id)
}

/// Regression: conjugate gradient never stopped from an impossible start or along a NaN gradient (the
/// convergence test compared NaN), and left the latents at NaN. This would hang before the fix.
#[test]
fn conjugate_gradient_stops_at_an_impossible_start() {
    let (model, mut state, sigma) = zero_sigma();
    assert_eq!(
        GradientOptimizer::new(&model)
            .max_a_posteriori(&mut state.clone())
            .unwrap_err(),
        AlgorithmError::ImpossibleStart
    );
    // a value on the boundary of the support has no unconstrained point, so even without the check it is an
    // impossible start
    assert_eq!(
        GradientOptimizer::new(&model)
            .check_initial_conditions(false)
            .max_a_posteriori(&mut state.clone())
            .unwrap_err(),
        AlgorithmError::ImpossibleStart
    );
    // in constrained space the optimizer starts there, and stops at once
    let result = GradientOptimizer::new(&model)
        .space(Space::Constrained)
        .check_initial_conditions(false)
        .max_a_posteriori(&mut state)
        .unwrap();
    assert!(!result.converged);
    assert_eq!(result.get(sigma).unwrap().to_scalar(), Some(0.0));
    assert_eq!(state.f64(sigma).unwrap().to_scalar(), Some(0.0));
}

fn both_algorithms() -> Vec<keanu_algorithms::optimizer::Algorithm> {
    vec![
        Adam::default().with_max_evaluations(100_000).into(),
        ConjugateGradient::default().into(),
    ]
}

/// AE2: MAP in unconstrained space optimises without the Jacobian, so it finds the mode of the Gamma density,
/// (k − 1)θ = 0.5, not the mode of the density of log σ (kθ = 1).
#[test]
fn map_of_a_positive_latent_is_the_mode_of_its_density() {
    for algorithm in both_algorithms() {
        let b = GraphBuilder::new();
        let sigma = b.gamma(0.5, 2.0);
        b.set_value(sigma, DoubleTensor::scalar(2.0));
        let sigma = sigma.id();
        let (graph, mut state) = b.build().unwrap();
        let model = Model::new(graph, &state).unwrap();
        let result = GradientOptimizer::new(&model)
            .algorithm(algorithm.clone())
            .max_a_posteriori(&mut state)
            .unwrap();
        let mode = result.get(sigma).unwrap().to_scalar().unwrap();
        assert!(result.converged, "{algorithm:?}");
        assert!(result.boundary.is_none(), "{algorithm:?}");
        assert!((mode - 0.5).abs() < 2e-3, "{algorithm:?}: {mode}");
        assert_eq!(state.f64(sigma).unwrap().to_scalar(), Some(mode));
    }
}

/// AE5: with no data the mode of HalfGaussian(1) is on the boundary, 0, which is at −∞ in unconstrained space: the
/// optimizer stops and names the latent rather than running on.
#[test]
fn map_on_the_boundary_of_the_support_is_reported() {
    for algorithm in both_algorithms() {
        let b = GraphBuilder::new();
        let sigma = b.half_gaussian(1.0).label("sigma");
        b.set_value(sigma, DoubleTensor::scalar(1.0));
        let sigma = sigma.id();
        let (graph, mut state) = b.build().unwrap();
        let model = Model::new(graph, &state).unwrap();
        let result = GradientOptimizer::new(&model)
            .algorithm(algorithm.clone())
            .max_a_posteriori(&mut state)
            .unwrap();
        assert_eq!(result.boundary, Some(sigma), "{algorithm:?}");
        assert!(!result.converged, "{algorithm:?}");
        assert!(result.evaluations < 100_000, "{algorithm:?}");
        let value = result.get(sigma).unwrap().to_scalar().unwrap();
        assert!((0.0..1e-12).contains(&value), "{algorithm:?}: {value}");
    }
}

/// Maximum likelihood of a scale: the likelihood's gradient is pulled back through the transform, and the prior
/// plays no part. The estimate of sigma for y_i ~ N(0, sigma) is the root mean square of the data.
#[test]
fn maximum_likelihood_of_a_scale() {
    let data = [0.5, -1.5, 2.0, -0.5];
    for algorithm in both_algorithms() {
        let b = GraphBuilder::new();
        let sigma = b.half_gaussian(1.0);
        b.set_value(sigma, DoubleTensor::scalar(3.0));
        let y = b.gaussian_with_shape(&[data.len()], 0.0, sigma);
        b.observe(y, DoubleTensor::vector(data.to_vec()));
        let sigma = sigma.id();
        let (graph, mut state) = b.build().unwrap();
        let model = Model::new(graph, &state).unwrap();
        let result = GradientOptimizer::new(&model)
            .algorithm(algorithm.clone())
            .max_likelihood(&mut state)
            .unwrap();
        let expected = (data.iter().map(|y| y * y).sum::<f64>() / data.len() as f64).sqrt();
        let estimate = result.get(sigma).unwrap().to_scalar().unwrap();
        assert!(result.converged, "{algorithm:?}");
        assert!(
            (estimate - expected).abs() < 2e-3,
            "{algorithm:?}: {estimate} vs {expected}"
        );
    }
}

/// A probability on a simplex: in unconstrained space the optimizer stays on the simplex and finds the
/// Dirichlet–Multinomial posterior mode, (α + k − 1) / (Σα + n − K).
#[test]
fn map_of_a_simplex() {
    let alpha = [2.0, 3.0, 4.0];
    let counts = [5i64, 1, 8];
    for algorithm in both_algorithms() {
        let b = GraphBuilder::new();
        let p = b.dirichlet(DoubleTensor::vector(alpha.to_vec()));
        b.set_value(p, DoubleTensor::vector(vec![1.0 / 3.0; 3]));
        let k = b.multinomial(counts.iter().sum::<i64>(), p);
        b.observe(k, IntTensor::vector(counts.to_vec()));
        let p = p.id();
        let (graph, mut state) = b.build().unwrap();
        let model = Model::new(graph, &state).unwrap();
        let result = GradientOptimizer::new(&model)
            .algorithm(algorithm.clone())
            .max_a_posteriori(&mut state)
            .unwrap();
        let mode = result.get(p).unwrap();
        let denominator = alpha.iter().sum::<f64>() + counts.iter().sum::<i64>() as f64 - 3.0;
        for (i, (&a, &k)) in alpha.iter().zip(&counts).enumerate() {
            let expected = (a + k as f64 - 1.0) / denominator;
            let actual = mode.as_slice()[i];
            assert!(
                (actual - expected).abs() < 2e-3,
                "{algorithm:?} p[{i}]: {actual} vs {expected}"
            );
        }
        assert!((mode.sum() - 1.0).abs() < 1e-12);
    }
}

/// a ~ N(0, 1) at 0 and z ~ N(sqrt(a), 1) observed at 0: a finite fitness whose gradient is 0 · ∞ = NaN.
fn nan_gradient() -> (Model, State, keanu_graph::NodeId) {
    let b = GraphBuilder::new();
    let a = b.gaussian(0.0, 1.0);
    b.set_value(a, DoubleTensor::scalar(0.0));
    let z = b.gaussian(a.sqrt(), 1.0);
    b.observe(z, DoubleTensor::scalar(0.0));
    let id = a.id();
    let (graph, state) = b.build().unwrap();
    (Model::new(graph, &state).unwrap(), state, id)
}

/// Regression: the initial-condition check took the largest gradient element with f64::max, which skips NaN, so a
/// NaN gradient passed; and conjugate gradient then moved the latents to NaN.
#[test]
fn a_nan_starting_gradient_is_detected() {
    let (model, mut state, a) = nan_gradient();
    assert_eq!(
        GradientOptimizer::new(&model)
            .max_a_posteriori(&mut state.clone())
            .unwrap_err(),
        AlgorithmError::NonFiniteGradient
    );
    let result = GradientOptimizer::new(&model)
        .check_initial_conditions(false)
        .max_a_posteriori(&mut state)
        .unwrap();
    assert!(!result.converged);
    assert_eq!(result.get(a).unwrap().to_scalar(), Some(0.0));
    assert!(result.fitness.is_finite());
}

/// Regression: conjugate gradient only checked its evaluation budget between iterations, so the line search
/// could use many more (`with_max_evaluations(5)` used 19).
#[test]
fn conjugate_gradient_keeps_to_its_evaluation_budget() {
    let (model, mut state, _) = conjugate_gaussian(&[0.3, 1.2]);
    let result = GradientOptimizer::new(&model)
        .algorithm(ConjugateGradient::default().with_max_evaluations(5))
        .max_a_posteriori(&mut state)
        .unwrap();
    assert!(!result.converged);
    // the budget, with no final fitness evaluation
    assert!(result.evaluations <= 5, "{}", result.evaluations);
    assert!(result.fitness.is_finite());
}
