//! ODE solves that fail in inference: the algorithms treat such points as impossible and carry on.
//!
//! The model is dy/dt = k y², y(0) = 1, observed at t = 1 with Gaussian noise. Its solution 1 / (1 − k t) blows up
//! before t = 1 for k ≥ 1, where the solver fails, so k's posterior on its prior range (0, 1.5) is cut off at 1.

mod mcmc_common;

use keanu_algorithms::AlgorithmError;
use keanu_algorithms::laplace::Laplace;
use keanu_algorithms::mcmc::SamplingOptions;
use keanu_algorithms::mcmc::metropolis::MetropolisHastings;
use keanu_algorithms::mcmc::nuts::Nuts;
use keanu_algorithms::optimizer::{GradientOptimizer, Lbfgs};
use keanu_algorithms::vi::Advi;
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

#[test]
fn the_laplace_approximation_has_no_second_order_rule_for_an_ode() {
    let (model, mut state, _) = blow_up(0.5, 0.4);
    match Laplace::new().fit(&model, &mut state) {
        Err(AlgorithmError::Graph(GraphError::NoSecondOrderRule(message))) => {
            assert!(message.contains("ODE"), "{message}");
        }
        other => panic!("expected no second-order rule, got {:?}", other.err()),
    }
}
