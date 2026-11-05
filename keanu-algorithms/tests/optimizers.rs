//! Optimizer behaviour that has a closed form answer or a specific failure mode.

use keanu_algorithms::AlgorithmError;
use keanu_algorithms::Space;
use keanu_algorithms::optimizer::{
    Adam, Algorithm, ConjugateGradient, ConvergenceChecker, GradientOptimizer, Lbfgs,
    NonGradientOptimizer, Norm, OptimizerBounds, UpdateFormula,
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
        Lbfgs::default().into(),
        Lbfgs::default().with_history_size(1).into(),
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
        .algorithm(ConjugateGradient::default())
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
        .algorithm(ConjugateGradient::default())
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
    // in constrained space conjugate gradient starts there, and stops at once
    let result = GradientOptimizer::new(&model)
        .algorithm(ConjugateGradient::default())
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
        Lbfgs::default().into(),
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

/// KTD11: L-BFGS is the default gradient algorithm.
#[test]
fn lbfgs_is_the_default() {
    assert_eq!(Algorithm::default(), Algorithm::from(Lbfgs::default()));
}

/// The generalised Rosenbrock function as a negative log likelihood: for each i < n,
/// a_i ~ N(10 (x_{i+1} − x_i²), 1/√2) observed at 0 and b_i ~ N(x_i, 1/√2) observed at 1, so the log likelihood
/// is −Σ [100 (x_{i+1} − x_i²)² + (1 − x_i)²] + const, maximised at x = (1, …, 1).
fn rosenbrock(start: &[f64]) -> (Model, State, Vec<keanu_graph::NodeId>) {
    let b = GraphBuilder::new();
    let sigma = std::f64::consts::FRAC_1_SQRT_2;
    let x: Vec<_> = start
        .iter()
        .map(|&v| {
            let x = b.gaussian(0.0, 1.0);
            b.set_value(x, DoubleTensor::scalar(v));
            x
        })
        .collect();
    for i in 0..start.len() - 1 {
        let a = b.gaussian((x[i + 1] - x[i] * x[i]) * 10.0, sigma);
        b.observe(a, DoubleTensor::scalar(0.0));
        let c = b.gaussian(x[i], sigma);
        b.observe(c, DoubleTensor::scalar(1.0));
    }
    let ids = x.iter().map(|x| x.id()).collect();
    let (graph, state) = b.build().unwrap();
    (Model::new(graph, &state).unwrap(), state, ids)
}

#[test]
fn lbfgs_minimises_rosenbrock() {
    let mut ten = [-1.2, 1.0].repeat(5);
    ten[9] = 1.0;
    for start in [vec![-1.2, 1.0], ten] {
        let (model, mut state, x) = rosenbrock(&start);
        let result = GradientOptimizer::new(&model)
            .algorithm(Lbfgs::default())
            .max_likelihood(&mut state)
            .unwrap();
        assert!(result.converged, "{} dimensions", start.len());
        for &id in &x {
            let v = result.get(id).unwrap().to_scalar().unwrap();
            assert!((v - 1.0).abs() < 1e-4, "{} dimensions: {v}", start.len());
        }
        // two evaluations (fitness and gradient) per trial point, well inside 2000 iterations
        assert!(result.evaluations < 2_000, "{}", result.evaluations);
    }
}

/// An impossible (−∞ or NaN) fitness at the start is an error for L-BFGS even without the initial-condition check
/// (conjugate gradient instead stops there).
#[test]
fn lbfgs_rejects_a_non_finite_start() {
    let (model, mut state, _) = zero_sigma();
    assert_eq!(
        GradientOptimizer::new(&model)
            .algorithm(Lbfgs::default())
            .space(Space::Constrained)
            .check_initial_conditions(false)
            .max_a_posteriori(&mut state)
            .unwrap_err(),
        AlgorithmError::ImpossibleStart
    );
}

/// In constrained space the first L-BFGS step from a = 0.5 (a step of 1 along the gradient) leaves the support of
/// Uniform(0, 1), where the fitness is −∞: the line search shrinks the step and the optimizer finds the mode, 0.9.
#[test]
fn lbfgs_shrinks_a_step_to_a_non_finite_point() {
    let b = GraphBuilder::new();
    let a = b.uniform(0.0, 1.0);
    b.set_value(a, DoubleTensor::scalar(0.5));
    let x = b.gaussian(a, 0.1);
    b.observe(x, DoubleTensor::scalar(0.9));
    let a = a.id();
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let result = GradientOptimizer::new(&model)
        .algorithm(Lbfgs::default())
        .space(Space::Constrained)
        .max_a_posteriori(&mut state)
        .unwrap();
    assert!(result.converged);
    let mode = result.get(a).unwrap().to_scalar().unwrap();
    assert!((mode - 0.9).abs() < 1e-6, "{mode}");
}

#[test]
fn invalid_lbfgs_settings_are_errors() {
    let (model, mut state, _) = conjugate_gaussian(&[1.0]);
    for lbfgs in [
        Lbfgs::default().with_history_size(0),
        Lbfgs::default().with_max_iterations(0),
        Lbfgs::default().with_tol_grad(-1.0),
        Lbfgs::default().with_tol_rel_obj(f64::NAN),
    ] {
        let error = GradientOptimizer::new(&model)
            .algorithm(lbfgs)
            .max_a_posteriori(&mut state);
        assert!(
            matches!(error, Err(AlgorithmError::InvalidSetting(_))),
            "{lbfgs:?}"
        );
    }
}

/// The iteration limit stops L-BFGS without converging.
#[test]
fn lbfgs_stops_at_its_iteration_limit() {
    let (model, mut state, _) = rosenbrock(&[-1.2, 1.0]);
    let result = GradientOptimizer::new(&model)
        .algorithm(Lbfgs::default().with_max_iterations(3))
        .max_likelihood(&mut state)
        .unwrap();
    assert!(!result.converged);
    assert!(result.fitness.is_finite());
}

/// mu ~ N(0, 10), x_i ~ N(mu, sigma) observed, where sigma = exp(log 0.5) is a deterministic node of constants only.
/// Building the graph does not compute sigma, so it has no value until something evaluates it.
fn constant_only_deterministic_scale() -> (Model, State, keanu_graph::NodeId, keanu_graph::NodeId) {
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 10.0).label("mu");
    b.set_value(mu, DoubleTensor::scalar(0.0));
    let sigma = b.constant(0.5f64.ln()).exp();
    let x = b.gaussian_with_shape(&[3], mu, sigma);
    b.observe(x, DoubleTensor::vector(vec![1.0, 2.0, 3.0]));
    let (mu, sigma) = (mu.id(), sigma.id());
    let (graph, state) = b.build().unwrap();
    (Model::new(graph, &state).unwrap(), state, mu, sigma)
}

/// Regression: the optimizers did not compute deterministic nodes that depend only on constants (as the samplers
/// do), so a model whose distribution parameters are such nodes (the constant parts of a Gaussian process kernel)
/// failed with a missing value unless every node was evaluated first.
#[test]
fn optimizers_compute_deterministic_nodes_of_constants_first() {
    // the posterior mode of mu: precision-weighted mean of the prior N(0, 10) and three observations with sd 0.5
    let expected = (6.0 / 0.25) / (1.0 / 100.0 + 3.0 / 0.25);
    for space in [Space::Unconstrained, Space::Constrained] {
        let (model, mut state, mu, sigma) = constant_only_deterministic_scale();
        assert!(!state.has_value(sigma));
        let result = GradientOptimizer::new(&model)
            .space(space)
            .max_a_posteriori(&mut state)
            .unwrap();
        let found = result.get(mu).unwrap().to_scalar().unwrap();
        assert!(
            (found - expected).abs() < 1e-4,
            "{space:?}: {found} vs {expected}"
        );
        assert!((state.f64(sigma).unwrap().to_scalar().unwrap() - 0.5).abs() < 1e-12);
    }

    let (model, mut state, mu, sigma) = constant_only_deterministic_scale();
    let result = NonGradientOptimizer::new(&model)
        .bounds(OptimizerBounds::new().add_scalar(mu, -10.0, 10.0))
        .max_a_posteriori(&mut state)
        .unwrap();
    let found = result.get(mu).unwrap().to_scalar().unwrap();
    assert!(
        (found - expected).abs() < 1e-3,
        "BOBYQA: {found} vs {expected}"
    );
    assert!(state.has_value(sigma));
}

/// A very narrow posterior, x ~ N(0, 1e-6) started at 10, has a starting gradient of norm 1e13: L-BFGS's first
/// step, 1 / |g|, is below the line search's absolute minimum step, so it must be judged relative to the initial
/// step rather than refused.
#[test]
fn lbfgs_converges_from_a_huge_gradient() {
    let b = GraphBuilder::new();
    let x = b.gaussian(0.0, 1e-6);
    b.set_value(x, DoubleTensor::scalar(10.0));
    let x = x.id();
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let result = GradientOptimizer::new(&model)
        .algorithm(Lbfgs::default())
        .max_a_posteriori(&mut state)
        .unwrap();
    assert!(result.converged);
    let value = result.get(x).unwrap().to_scalar().unwrap();
    assert!(value.abs() < 1e-8, "{value}");
}

/// An interior optimum far from 1 is at |z| > 30 in unconstrained space, but is not a boundary: MAP of
/// Gamma(θ, k = 3), whose mode is 2θ, for θ = 5e14 and 5e-16 (modes 1e15 and 1e-15).
#[test]
fn an_interior_optimum_far_from_one_is_not_a_boundary() {
    for theta in [5e14, 5e-16] {
        for algorithm in both_algorithms() {
            let b = GraphBuilder::new();
            let x = b.gamma(theta, 3.0);
            b.set_value(x, DoubleTensor::scalar(theta));
            let x = x.id();
            let (graph, mut state) = b.build().unwrap();
            let model = Model::new(graph, &state).unwrap();
            let result = GradientOptimizer::new(&model)
                .algorithm(algorithm.clone())
                .max_a_posteriori(&mut state)
                .unwrap();
            assert_eq!(result.boundary, None, "{theta} {algorithm:?}");
            assert!(result.converged, "{theta} {algorithm:?}");
            let mode = result.get(x).unwrap().to_scalar().unwrap();
            // Adam's convergence test is on the relative change in z (about ±35), so it stops a little short
            let tolerance = if matches!(algorithm, Algorithm::Adam(_)) {
                1e-2
            } else {
                2e-3
            };
            assert!(
                (mode / (2.0 * theta) - 1.0).abs() < tolerance,
                "{theta} {algorithm:?}: {mode}"
            );
        }
    }
}
