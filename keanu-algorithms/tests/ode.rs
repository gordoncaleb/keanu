//! ODE solves that fail in inference: the algorithms treat such points as impossible and carry on.
//!
//! The model is dy/dt = k y², y(0) = 1, observed at t = 1 with Gaussian noise. Its solution 1 / (1 − k t) blows up
//! before t = 1 for k ≥ 1, where the solver fails, so k's posterior on its prior range (0, 1.5) is cut off at 1.

mod mcmc_common;

use keanu_algorithms::laplace::Laplace;
use keanu_algorithms::mcmc::SamplingOptions;
use keanu_algorithms::mcmc::metropolis::MetropolisHastings;
use keanu_algorithms::mcmc::nuts::Nuts;
use keanu_algorithms::optimizer::{GradientOptimizer, Lbfgs, TrustRegion};
use keanu_algorithms::vi::Advi;
use keanu_algorithms::{AlgorithmError, HessianMethod};
use keanu_graph::ode::OdeOptions;
use keanu_graph::{GraphBuilder, GraphError, Model, NodeId, State};
use keanu_tensor::DoubleTensor;
use mcmc_common::*;

fn blow_up(noise: f64, start: f64) -> (Model, State, NodeId) {
    let b = GraphBuilder::new();
    let k = b.uniform(0.0, 1.5).label("k");
    b.set_value(k, DoubleTensor::scalar(start));
    let y = b.ode(
        b.constant(DoubleTensor::vector(vec![1.0])),
        b.constant(DoubleTensor::vector(vec![1.0])),
        b.scalar(0.0),
        &[k],
        OdeOptions::default(),
        |_, _, y, p| p[0] * y * y,
    );
    let observed = b.gaussian(y, noise);
    // y(1) = 2 at k = 0.5
    b.observe(observed, DoubleTensor::new(&[1, 1], vec![2.0]).unwrap());
    let k = k.id();
    let (graph, mut state) = b.build().unwrap();
    let all: Vec<NodeId> = graph.ids().collect();
    state.lazy_eval(&graph, &all, &mut rng(0)).unwrap();
    let model = Model::new(graph, &state).unwrap();
    (model, state, k)
}

#[test]
fn nuts_counts_a_failed_solve_as_a_divergence_and_continues() {
    // weak data put posterior mass near the cut-off, which trajectories cross
    let (model, mut state, k) = blow_up(20.0, 0.5);
    let (samples, statistics) = Nuts::default()
        .with_adapt_count(200)
        .sample_with_statistics(
            &model,
            &mut state,
            &[k],
            600,
            SamplingOptions::default()
                .with_drop(200)
                .with_down_sample_interval(1),
            &mut rng(5),
        )
        .unwrap();
    let draws = series(&samples, k);
    assert_eq!(draws.len(), 400);
    assert!(
        draws.iter().all(|&x| x > 0.0 && x < 1.0),
        "every draw solves"
    );
    assert!(
        statistics.divergence_count() > 0,
        "trajectories into k ≥ 1 diverge"
    );
    assert!(statistics.divergence_count() < 200);
}

#[test]
fn metropolis_hastings_rejects_a_failed_solve() {
    let (model, mut state, k) = blow_up(20.0, 0.5);
    let samples = MetropolisHastings::default()
        .sample_with(
            &model,
            &mut state,
            &[k],
            2000,
            SamplingOptions::default(),
            &mut rng(6),
        )
        .unwrap();
    let draws = series(&samples, k);
    assert!(draws.iter().all(|&x| x > 0.0 && x < 1.0));
    // proposals from the prior range reach k ≥ 1, about a third of it
    assert!(draws.windows(2).any(|w| w[0] == w[1]), "some rejections");
}

#[test]
fn an_optimizer_rejects_steps_into_a_failed_solve() {
    // from far below the optimum the first steps overshoot past the cut-off
    for start in [0.02, 0.95] {
        let (model, mut state, k) = blow_up(0.5, start);
        let result = GradientOptimizer::new(&model)
            .algorithm(Lbfgs::default())
            .max_a_posteriori(&mut state)
            .unwrap();
        let found = result.get(k).unwrap().to_scalar().unwrap();
        assert!((found - 0.5).abs() < 1e-3, "from {start}: {found}");
        assert!(result.fitness.is_finite());
    }
}

/// Trust-region Newton on an ODE model: the ODE has no second-order rule, so the Hessian-vector products are finite
/// differences of the gradient, and steps past the cut-off are rejected as impossible.
#[test]
fn trust_region_newton_maximizes_an_ode_model_by_finite_differences() {
    for start in [0.02, 0.95] {
        let (model, mut state, k) = blow_up(0.5, start);
        let result = GradientOptimizer::new(&model)
            .algorithm(TrustRegion::default())
            .max_a_posteriori(&mut state)
            .unwrap();
        let found = result.get(k).unwrap().to_scalar().unwrap();
        assert!(result.converged, "from {start}");
        assert!((found - 0.5).abs() < 1e-3, "from {start}: {found}");
        assert!(result.hessian_vector_products > 0);
    }
    // with exact products only, there is no rule for the ODE
    let (model, mut state, _) = blow_up(0.5, 0.4);
    let exact = GradientOptimizer::new(&model)
        .algorithm(TrustRegion::default().with_hessian_method(HessianMethod::Exact))
        .max_a_posteriori(&mut state);
    assert!(matches!(
        exact,
        Err(AlgorithmError::Graph(GraphError::NoSecondOrderRule(_)))
    ));
}

#[test]
fn advi_reports_draws_where_the_solve_fails_as_a_non_finite_elbo() {
    // the initial approximation (a unit Gaussian in unconstrained space) puts draws past the cut-off: impossible
    // draws, as for any model with zero-probability regions, are a non-finite ELBO rather than a solver error
    let (model, mut state, _) = blow_up(20.0, 0.5);
    let result = Advi::mean_field().fit(&model, &mut state, &mut rng(7));
    assert!(
        matches!(result, Err(AlgorithmError::NonFiniteElbo(_))),
        "{:?}",
        result.err()
    );
}

/// The ODE has no second-order rule, so the Laplace approximation takes its Hessian by finite differences of the
/// gradient, and agrees with the same model written with the closed-form solution y(1) = 1 / (1 − k), whose Hessian
/// is exact.
#[test]
fn the_laplace_approximation_of_an_ode_model_uses_finite_differences() {
    let (model, mut state, k) = blow_up(0.5, 0.4);
    let ode = Laplace::new().fit(&model, &mut state).unwrap();
    assert_eq!(ode.hessian_method(), HessianMethod::FiniteDifference);

    let b = GraphBuilder::new();
    let closed_k = b.uniform(0.0, 1.5);
    b.set_value(closed_k, DoubleTensor::scalar(0.4));
    let y = (b.scalar(1.0) - closed_k).reciprocal();
    b.observe(b.gaussian(y, 0.5), DoubleTensor::scalar(2.0));
    let closed_k = closed_k.id();
    let (graph, mut closed_state) = b.build().unwrap();
    let closed_model = Model::new(graph, &closed_state).unwrap();
    let closed = Laplace::new()
        .fit(&closed_model, &mut closed_state)
        .unwrap();
    assert_eq!(closed.hessian_method(), HessianMethod::Exact);

    let (a, c) = (
        ode.get(k).unwrap().to_scalar().unwrap(),
        closed.get(closed_k).unwrap().to_scalar().unwrap(),
    );
    assert!((a - c).abs() < 1e-5, "mode {a} vs {c}");
    let (a, c) = (
        ode.covariance().as_slice()[0],
        closed.covariance().as_slice()[0],
    );
    assert!((a / c - 1.0).abs() < 1e-3, "variance {a} vs {c}");

    // the exact Hessian has no rule for the ODE
    let (model, mut state, _) = blow_up(0.5, 0.4);
    match Laplace::new()
        .with_hessian_method(HessianMethod::Exact)
        .fit(&model, &mut state)
    {
        Err(AlgorithmError::Graph(GraphError::NoSecondOrderRule(message))) => {
            assert!(message.contains("ODE"), "{message}");
        }
        other => panic!("expected no second-order rule, got {:?}", other.err()),
    }
}

/// k and the solution y(1) recorded together by `sample`.
fn recorded(samples: &keanu_algorithms::NetworkSamples, k: NodeId, y: NodeId) -> Vec<(f64, f64)> {
    let ks = series(samples, k);
    let ys = samples
        .f64(y)
        .unwrap()
        .as_slice()
        .iter()
        .map(|t| t.as_slice()[0])
        .collect::<Vec<_>>();
    ks.into_iter().zip(ys).collect()
}

/// Draws from a fitted approximation where the solve fails are never recorded with the previous draw's solution:
/// Pathfinder's draws past the cut-off are skipped, and every recorded solution is the one of its own k.
#[test]
fn approximation_draws_where_the_solve_fails_are_not_recorded() {
    let (model, mut state, k) = blow_up(20.0, 0.5);
    let y = model
        .graph()
        .ids()
        .find(|&id| model.graph()[id].op().name() == "ode")
        .unwrap();
    let result = keanu_algorithms::vi::Pathfinder::default()
        .with_draws(400)
        .single_path(&model, &mut state, &mut rng(3))
        .unwrap();
    let impossible = result
        .log_density()
        .iter()
        .filter(|d| **d == f64::NEG_INFINITY)
        .count();
    assert!(impossible > 0, "some draws are past the cut-off");
    let samples = result.sample(&mut state, &[k, y]).unwrap();
    assert_eq!(samples.len(), 400 - impossible);
    for (k, y) in recorded(&samples, k, y) {
        assert!(k < 1.0, "k = {k}");
        // the solver's tolerance near the singularity; a stale solution is off by far more
        assert!(
            (y - 1.0 / (1.0 - k)).abs() < 1e-2 * y,
            "y(1) = {y} at k = {k}"
        );
    }
}

/// k ~ N(0, 1) with an observation N(k, 0.5) = 0.3, and dy/dt = −y · (k − m)/(k − m), y(0) = 1, observed at t = 1:
/// the same model whatever m, except that the solve fails (a NaN derivative) at exactly k = m.
#[allow(clippy::eq_op)] // (k − m)/(k − m) is 1 except at k = m, where it is NaN
fn fails_at(m: f64) -> (Model, State, NodeId) {
    let b = GraphBuilder::new();
    let k = b.gaussian(0.0, 1.0).label("k");
    b.set_value(k, DoubleTensor::scalar(-0.4));
    let observed = b.gaussian(k, 0.5);
    b.observe(observed, DoubleTensor::scalar(0.3));
    let y = b.ode(
        b.constant(DoubleTensor::vector(vec![1.0])),
        b.constant(DoubleTensor::vector(vec![1.0])),
        b.scalar(0.0),
        &[k],
        OdeOptions::default(),
        move |_, _, y, p| -y * ((p[0] - m) / (p[0] - m)),
    );
    let y_observed = b.gaussian(y, 1.0);
    b.observe(y_observed, DoubleTensor::new(&[1, 1], vec![0.4]).unwrap());
    let k = k.id();
    let (graph, mut state) = b.build().unwrap();
    let all: Vec<NodeId> = graph.ids().collect();
    state.lazy_eval(&graph, &all, &mut rng(0)).unwrap();
    let model = Model::new(graph, &state).unwrap();
    (model, state, k)
}

/// A fit whose final point (the fitted mean) can't be evaluated still succeeds: ADVI leaves the state at the
/// starting values and Pathfinder at the best approximation's L-BFGS iterate, both of which were evaluated.
#[test]
fn a_fit_whose_mean_cannot_be_evaluated_falls_back_to_an_evaluated_point() {
    let advi = Advi::mean_field().with_max_iterations(200);
    let (model, mut state, k) = fails_at(1e6);
    let mean = advi.fit(&model, &mut state, &mut rng(9)).unwrap().mean()[0];
    let (model, mut state, k_node) = fails_at(mean);
    assert_eq!(k_node, k);
    let fitted = advi.fit(&model, &mut state, &mut rng(9)).unwrap();
    assert_eq!(fitted.mean()[0], mean, "the same fit");
    assert_eq!(
        state.f64(k).unwrap().as_slice()[0],
        -0.4,
        "the starting value"
    );

    let pathfinder = keanu_algorithms::vi::Pathfinder::default();
    let (model, mut state, k) = fails_at(1e6);
    let path = pathfinder
        .single_path(&model, &mut state, &mut rng(9))
        .unwrap();
    let mean = path.paths()[0].mean()[0];
    let (model, mut state, _) = fails_at(mean);
    let fitted = pathfinder
        .single_path(&model, &mut state, &mut rng(9))
        .unwrap();
    let fitted_path = &fitted.paths()[0];
    assert_eq!(fitted_path.mean()[0], mean, "the same path");
    let value = state.f64(k).unwrap().as_slice()[0];
    assert!(value != mean && value.is_finite(), "{value}");
    assert!(model.log_prob(&state).unwrap().is_finite());
}

/// Prior draws where the solve fails (k ≥ 1, a third of the prior) are drawn again, as the approximations redraw
/// them: forward sampling and prior predictive draws are from the prior where the model can be evaluated, and no
/// draw pairs a k with another draw's solution.
#[test]
fn prior_draws_where_the_solve_fails_are_drawn_again() {
    let b = GraphBuilder::new();
    let k = b.uniform(0.0, 1.5).label("k");
    b.set_value(k, DoubleTensor::scalar(0.5));
    let y = b.ode(
        b.constant(DoubleTensor::vector(vec![1.0])),
        b.constant(DoubleTensor::vector(vec![1.0])),
        b.scalar(0.0),
        &[k],
        OdeOptions::default(),
        |_, _, y, p| p[0] * y * y,
    );
    let (k, y) = (k.id(), y.id());
    let (graph, mut state) = b.build().unwrap();
    let all: Vec<NodeId> = graph.ids().collect();
    state.lazy_eval(&graph, &all, &mut rng(0)).unwrap();
    let model = Model::new(graph, &state).unwrap();
    for calculate_log_prob in [false, true] {
        let samples = keanu_algorithms::ForwardSampler::new()
            .with_calculate_log_prob(calculate_log_prob)
            .sample(&model, &mut state, &[k, y], 300, &mut rng(1))
            .unwrap();
        assert_eq!(samples.len(), 300);
        let draws = recorded(&samples, k, y);
        assert!(
            draws.iter().any(|&(k, _)| k > 0.9),
            "draws near the cut-off"
        );
        for (k, y) in draws {
            assert!(k < 1.0, "k = {k}");
            assert!(
                (y - 1.0 / (1.0 - k)).abs() < 1e-2 * y,
                "y(1) = {y} at k = {k}"
            );
        }
    }

    // the prior predictive draws of an observation of the solution
    let b = GraphBuilder::new();
    let k = b.uniform(0.0, 1.5).label("k");
    b.set_value(k, DoubleTensor::scalar(0.5));
    let y = b.ode(
        b.constant(DoubleTensor::vector(vec![1.0])),
        b.constant(DoubleTensor::vector(vec![1.0])),
        b.scalar(0.0),
        &[k],
        OdeOptions::default(),
        |_, _, y, p| p[0] * y * y,
    );
    let observed = b.gaussian(y, 1e-6);
    b.observe(observed, DoubleTensor::new(&[1, 1], vec![2.0]).unwrap());
    let observed = observed.id();
    let (graph, mut state) = b.build().unwrap();
    let all: Vec<NodeId> = graph.ids().collect();
    state.lazy_eval(&graph, &all, &mut rng(0)).unwrap();
    let model = Model::new(graph, &state).unwrap();
    let draws = keanu_algorithms::PredictiveSampler::new()
        .prior(&model, &state, &[observed], 300, &mut rng(2))
        .unwrap();
    assert_eq!(draws.len(), 300);
    // y(1) = 1 / (1 − k) ≥ 1 for k in [0, 1)
    let values = draws.f64(observed).unwrap();
    assert!(values.as_slice().iter().all(|y| y.as_slice()[0] > 0.99));
}
