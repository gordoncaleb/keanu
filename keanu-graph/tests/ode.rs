//! The ODE op: values against the solvers, first derivatives against closed forms and finite differences, the
//! missing second-order rule, failed solves as impossible points, and the rules on bodies and parents.

use keanu_distributions::KeanuRng;
use keanu_graph::ode::{FailureReason, OdeOptions, OdeSolver, integrate};
use keanu_graph::op::Op;
use keanu_graph::{
    DoubleVar, Graph, GraphBuilder, GraphError, Model, NodeId, Parameterisation, State,
    forward_partials, reverse_partials,
};
use keanu_tensor::DoubleTensor;

mod common;
use common::{assert_close, assert_close_rel, v};

fn rng() -> KeanuRng {
    KeanuRng::seed_from_u64(3)
}

fn evaluated(b: GraphBuilder) -> (Graph, State) {
    let (graph, mut state) = b.build().unwrap();
    let all: Vec<NodeId> = graph.ids().collect();
    state.lazy_eval(&graph, &all, &mut rng()).unwrap();
    (graph, state)
}

fn tight() -> OdeOptions {
    OdeOptions::new(OdeSolver::Rk45).with_tolerances(1e-12, 1e-12)
}

/// Lotka–Volterra: dx/dt = a x − b x y, dy/dt = c x y − d y, with θ = [a, b, c, d].
fn lotka_volterra<'s>(body: &'s GraphBuilder, y: DoubleVar<'s>, p: DoubleVar<'s>) -> DoubleVar<'s> {
    let (prey, predator) = (y.take(&[0]), y.take(&[1]));
    let p = |i: usize| p.take(&[i]);
    body.stack(
        0,
        &[
            p(0) * prey - p(1) * prey * predator,
            p(2) * prey * predator - p(3) * predator,
        ],
    )
}

const THETA: [f64; 4] = [1.1, 0.4, 0.1, 0.4];
const TIMES: [f64; 4] = [0.5, 1.5, 3.0, 4.0];
const Y0: [f64; 2] = [10.0, 5.0];

#[test]
fn the_value_matches_the_solver() {
    for solver in [OdeSolver::Rk45, OdeSolver::Stiff] {
        let options = OdeOptions::new(solver).with_tolerances(1e-8, 1e-8);
        let b = GraphBuilder::new();
        let theta = b.constant(v(&THETA));
        let y = b.ode(
            b.constant(v(&Y0)),
            b.constant(v(&TIMES)),
            b.scalar(0.0),
            &[theta],
            options,
            |body, _, y, p| lotka_volterra(body, y, p[0]),
        );
        assert_eq!(y.shape(), vec![4, 2]);
        let y = y.id();
        let (graph, state) = evaluated(b);
        assert!(matches!(graph[y].op(), Op::Ode(_)));
        assert_eq!(graph[y].op().name(), "ode");
        let [a, bb, c, d] = THETA;
        let expected = integrate(
            |_, y, f| {
                f[0] = a * y[0] - bb * y[0] * y[1];
                f[1] = c * y[0] * y[1] - d * y[1];
            },
            |_, y, j, ft| {
                j.copy_from_slice(&[a - bb * y[1], -bb * y[0], c * y[1], c * y[0] - d]);
                ft.fill(0.0);
            },
            0.0,
            &Y0,
            &TIMES,
            &options,
        )
        .unwrap();
        let expected: Vec<f64> = expected.into_iter().flatten().collect();
        let actual = state.f64(y).unwrap().as_slice();
        for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
            assert_close_rel(*a, *e, 1e-12, &format!("{solver:?} [{i}]"));
        }
    }
}

/// Exponential decay y' = −k y from y(t₀) = y₀ at times t, with every parent a latent, so all gradients flow.
fn decay(b: &GraphBuilder, options: OdeOptions) -> [NodeId; 5] {
    let y0 = b.gaussian_with_shape(&[1], 2.0, 1.0);
    let times = b.gaussian_with_shape(&[3], 0.0, 1.0);
    let t0 = b.gaussian(0.0, 1.0);
    let k = b.gaussian(0.0, 1.0);
    b.set_value(y0, v(&[2.0]));
    b.set_value(times, v(&[0.5, 1.0, 2.5]));
    b.set_value(t0, DoubleTensor::scalar(0.25));
    b.set_value(k, DoubleTensor::scalar(0.7));
    let y = b.ode(y0, times, t0, &[k], options, |_, _, y, p| -(p[0] * y));
    [y0.id(), times.id(), t0.id(), k.id(), y.id()]
}

#[test]
fn gradients_of_exponential_decay_match_the_closed_form() {
    for solver in [OdeSolver::Rk45, OdeSolver::Stiff] {
        let b = GraphBuilder::new();
        let [y0, times, t0, k, y] =
            decay(&b, OdeOptions::new(solver).with_tolerances(1e-10, 1e-10));
        let (graph, state) = evaluated(b);
        let (y0v, kv, t0v) = (2.0, 0.7, 0.25);
        let ts = [0.5, 1.0, 2.5];
        let exact = |t: f64| y0v * (-kv * (t - t0v)).exp();
        let reverse = reverse_partials(&graph, &state, y, &[y0, times, t0, k]).unwrap();
        for (wrt, name) in [(y0, "y0"), (times, "times"), (t0, "t0"), (k, "k")] {
            let forward = forward_partials(&graph, &state, wrt, &[y]).unwrap();
            for partials in [reverse.get(wrt).unwrap(), forward.get(y).unwrap()] {
                let data = partials.as_slice();
                for (i, &t) in ts.iter().enumerate() {
                    let expected = match name {
                        "y0" => exact(t) / y0v,
                        "t0" => kv * exact(t),
                        "k" => -(t - t0v) * exact(t),
                        _ => -kv * exact(t),
                    };
                    let value = if name == "times" {
                        data[i * 3 + i]
                    } else {
                        data[i]
                    };
                    assert_close_rel(value, expected, 1e-7, &format!("{solver:?} {name} {i}"));
                }
            }
        }
        // the times only move their own output
        let partial = reverse.get(times).unwrap();
        assert_eq!(partial.shape(), &[3, 1, 3]);
        assert_eq!(partial.as_slice()[1], 0.0);
    }
}

/// The Lotka–Volterra model with latent initial state, initial time and parameters, as a graph whose output is the
/// ODE solution. Returns (graph, state, [y0, t0, theta], solution).
fn latent_lotka_volterra(options: OdeOptions) -> (Graph, State, [NodeId; 3], NodeId) {
    let b = GraphBuilder::new();
    let y0 = b.gaussian_with_shape(&[2], 0.0, 1.0);
    let t0 = b.gaussian(0.0, 1.0);
    let theta = b.gaussian_with_shape(&[4], 0.0, 1.0);
    b.set_value(y0, v(&Y0));
    b.set_value(t0, DoubleTensor::scalar(0.1));
    b.set_value(theta, v(&THETA));
    let y = b.ode(
        y0,
        b.constant(v(&TIMES)),
        t0,
        &[theta],
        options,
        |body, _, y, p| lotka_volterra(body, y, p[0]),
    );
    let ids = [y0.id(), t0.id(), theta.id()];
    let y = y.id();
    let (graph, state) = evaluated(b);
    (graph, state, ids, y)
}

#[test]
fn gradients_match_finite_differences() {
    for solver in [OdeSolver::Rk45, OdeSolver::Stiff] {
        let options = OdeOptions::new(solver).with_tolerances(1e-12, 1e-12);
        let (graph, state, wrt, y) = latent_lotka_volterra(options);
        let reverse = reverse_partials(&graph, &state, y, &wrt).unwrap();
        let out_len = state.f64(y).unwrap().len();
        for id in wrt {
            let forward = forward_partials(&graph, &state, id, &[y]).unwrap();
            let x = state.f64(id).unwrap().clone();
            for m in 0..x.len() {
                let solution_at = |delta: f64| {
                    let mut data = x.as_slice().to_vec();
                    data[m] += delta;
                    let mut shifted = state.clone();
                    let value = DoubleTensor::new(x.shape(), data).unwrap();
                    shifted.set_and_cascade(&graph, id, value).unwrap();
                    shifted.f64(y).unwrap().as_slice().to_vec()
                };
                let h = 1e-5 * x.as_slice()[m].abs().max(1.0);
                let (up, down) = (solution_at(h), solution_at(-h));
                for o in 0..out_len {
                    let fd = (up[o] - down[o]) / (2.0 * h);
                    for (mode, partial) in [
                        ("reverse", reverse.get(id).unwrap()),
                        ("forward", forward.get(y).unwrap()),
                    ] {
                        let value = partial.as_slice()[o * x.len() + m];
                        // 1e-6 relative to the largest partial of this output, the scale finite differences resolve
                        let scale = up[o].abs().max(1.0);
                        assert_close(
                            value,
                            fd,
                            1e-6 * scale.max(fd.abs()),
                            &format!("{solver:?} {mode} d out[{o}] / d {id:?}[{m}]"),
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn gradients_through_a_log_probability_match_finite_differences() {
    // a Gaussian likelihood on the solution, with the gradient taken by the model in unconstrained space
    let b = GraphBuilder::new();
    let theta = b.uniform_with_shape(&[4], 0.0, 2.0).label("theta");
    b.set_value(theta, v(&THETA));
    let y = b.ode(
        b.constant(v(&Y0)),
        b.constant(v(&TIMES)),
        b.scalar(0.0),
        &[theta],
        tight(),
        |body, _, y, p| lotka_volterra(body, y, p[0]),
    );
    let observed = b.gaussian(y, 0.5);
    b.observe(
        observed,
        DoubleTensor::new(&[4, 2], vec![9.0, 8.0, 3.5, 9.0, 1.5, 6.0, 2.5, 3.0]).unwrap(),
    );
    let (graph, state) = evaluated(b);
    let model = Model::new(graph, &state).unwrap();
    let param = Parameterisation::new(&model, &state).unwrap();
    let mut state = state;
    let z = param.from_state(&state).unwrap();
    let (_, gradient) = param.log_prob_and_gradient(&z, true, &mut state).unwrap();
    let gradient = gradient.unwrap();
    for m in 0..z.len() {
        let at = |delta: f64, state: &mut State| {
            let mut point = z.clone();
            point[m] += delta;
            param.log_prob_and_gradient(&point, true, state).unwrap().0
        };
        let h = 1e-5;
        let fd = (at(h, &mut state) - at(-h, &mut state)) / (2.0 * h);
        assert_close_rel(gradient[m], fd, 1e-6, &format!("d log p / d z[{m}]"));
    }
}

#[test]
fn hessian_vector_products_have_no_second_order_rule() {
    let b = GraphBuilder::new();
    let k = b.uniform(0.0, 2.0);
    b.set_value(k, DoubleTensor::scalar(0.5));
    let y = b.ode(
        b.constant(v(&[1.0])),
        b.constant(v(&[1.0])),
        b.scalar(0.0),
        &[k],
        OdeOptions::default(),
        |_, _, y, p| -(p[0] * y),
    );
    b.observe(
        b.gaussian(y, 1.0),
        DoubleTensor::new(&[1, 1], vec![0.5]).unwrap(),
    );
    let (graph, mut state) = evaluated(b);
    let model = Model::new(graph, &state).unwrap();
    let param = Parameterisation::new(&model, &state).unwrap();
    let z = param.from_state(&state).unwrap();
    match param.hvp(&z, &[1.0], true, &mut state) {
        Err(GraphError::NoSecondOrderRule(message)) => {
            assert!(message.contains("ODE"), "{message}");
        }
        other => panic!("expected no second-order rule, got {other:?}"),
    }
}

/// dy/dt = k y², y(0) = 1, observed at t = 1: the solution 1 / (1 − k t) blows up before t = 1 for k > 1.
fn blow_up(b: &GraphBuilder, options: OdeOptions) -> (DoubleVar<'_>, DoubleVar<'_>) {
    let k = b.uniform(0.0, 1.5).label("k");
    b.set_value(k, DoubleTensor::scalar(0.5));
    let y = b.ode(
        b.constant(v(&[1.0])),
        b.constant(v(&[1.0])),
        b.scalar(0.0),
        &[k],
        options,
        |_, _, y, p| p[0] * y * y,
    );
    (k, y)
}

#[test]
fn a_failed_solve_is_an_impossible_point() {
    let b = GraphBuilder::new();
    let (k, y) = blow_up(&b, OdeOptions::default());
    let observed = b.gaussian(y, 10.0);
    b.observe(observed, DoubleTensor::new(&[1, 1], vec![2.0]).unwrap());
    let (k, y) = (k.id(), y.id());
    let (graph, mut state) = evaluated(b);
    assert_close_rel(
        state.f64(y).unwrap().as_slice()[0],
        2.0,
        1e-5,
        "y(1) at k = 0.5",
    );

    // evaluating directly fails with the solver's error
    let mut failing = state.clone();
    match failing.set_and_cascade(&graph, k, DoubleTensor::scalar(1.2)) {
        Err(e @ GraphError::SolverFailed(failed)) => {
            assert!(e.is_impossible_point());
            assert!(failed.t < 1.0, "{failed}");
            assert!(
                matches!(
                    failed.reason,
                    FailureReason::StepSizeUnderflow(_) | FailureReason::NonFinite
                ),
                "{failed:?}"
            );
        }
        other => panic!("expected a solver failure, got {other:?}"),
    }
    assert!(!GraphError::InvalidValue(String::new()).is_impossible_point());

    let model = Model::new(graph.clone(), &state).unwrap();
    // incrementally, as Metropolis–Hastings proposes
    let before = model.log_prob(&state).unwrap();
    let after = model
        .log_prob_after(&mut state, &[(k, DoubleTensor::scalar(1.2).into())], before)
        .unwrap();
    assert_eq!(after, f64::NEG_INFINITY);
    model
        .set_and_cascade(&mut state, [(k, DoubleTensor::scalar(0.5).into())])
        .unwrap();
    assert_eq!(model.log_prob(&state).unwrap(), before);

    // through the parameterisation, as the gradient-based algorithms evaluate
    let param = Parameterisation::new(&model, &state).unwrap();
    let mut z = param.from_state(&state).unwrap();
    let (log_prob, gradient) = param.log_prob_and_gradient(&z, true, &mut state).unwrap();
    assert!(log_prob.is_finite() && gradient.is_some());
    z[0] = 2.0; // k = 1.5 · sigmoid(2) ≈ 1.32
    assert_eq!(
        param.log_prob_and_gradient(&z, true, &mut state).unwrap(),
        (f64::NEG_INFINITY, None)
    );
    assert_eq!(
        param.log_prob(&z, true, &mut state).unwrap(),
        f64::NEG_INFINITY
    );
    assert!(!param.try_to_state(&z, &mut state).unwrap());
}

/// A cascade that fails at an impossible point leaves no stale value: the failed node and every deterministic node
/// downstream of it lose their values (nodes the failure doesn't reach are still recomputed), so nothing can read
/// the solution of the previous parameters. Reading the model there is an error rather than a stale value, and
/// setting a solvable value recomputes them; so does a later cascade from another node that needs them.
#[test]
fn a_failed_cascade_clears_the_failed_node_and_its_descendants() {
    let b = GraphBuilder::new();
    let (k, y) = blow_up(&b, OdeOptions::default());
    let doubled = (y * 2.0).label("doubled");
    let other = b.gaussian(0.0, 1.0).label("other");
    b.set_value(other, DoubleTensor::scalar(0.25));
    // depends on the solution and on another latent
    let both = (doubled + other).label("both");
    let k_squared = (k * k).label("k squared");
    let observed = b.gaussian(both, 10.0);
    b.observe(observed, DoubleTensor::new(&[1, 1], vec![2.0]).unwrap());
    let (k, y, doubled, other, both, k_squared) = (
        k.id(),
        y.id(),
        doubled.id(),
        other.id(),
        both.id(),
        k_squared.id(),
    );
    let (graph, mut state) = evaluated(b);
    let model = Model::new(graph.clone(), &state).unwrap();
    let error = state
        .set_and_cascade(&graph, k, DoubleTensor::scalar(1.2))
        .unwrap_err();
    assert!(error.is_impossible_point(), "{error}");
    for id in [y, doubled, both] {
        assert!(
            !state.has_value(id),
            "{} kept a stale value",
            graph.describe(id)
        );
    }
    // a sibling the failure doesn't reach is recomputed for the new value
    assert_close(
        state.f64(k_squared).unwrap().as_slice()[0],
        1.44,
        1e-15,
        "k²",
    );
    assert!(model.log_prob(&state).is_err());

    // a cascade from another latent recomputes the missing solution, failing again while k is unsolvable
    let error = state
        .set_and_cascade(&graph, other, DoubleTensor::scalar(0.5))
        .unwrap_err();
    assert!(error.is_impossible_point(), "{error}");
    assert!(!state.has_value(y) && !state.has_value(both));
    state.set(&graph, k, DoubleTensor::scalar(0.5)).unwrap();
    state
        .set_and_cascade(&graph, other, DoubleTensor::scalar(0.5))
        .unwrap();
    assert_close_rel(
        state.f64(both).unwrap().as_slice()[0],
        4.5,
        1e-5,
        "2 y(1) + 0.5",
    );
}

#[test]
fn the_step_limit_is_a_solver_failure() {
    let b = GraphBuilder::new();
    let (_, y) = blow_up(&b, OdeOptions::default().with_max_steps(3));
    let y = y.id();
    let (graph, mut state) = b.build().unwrap();
    match state.lazy_eval(&graph, &[y], &mut rng()) {
        Err(GraphError::SolverFailed(failed)) => {
            assert_eq!(failed.reason, FailureReason::MaxSteps(3));
        }
        other => panic!("expected the step limit, got {other:?}"),
    }
}

#[test]
fn a_probe_resamples_until_the_solve_succeeds() {
    let b = GraphBuilder::new();
    let (k, y) = blow_up(&b, OdeOptions::default());
    b.observe(
        b.gaussian(y, 10.0),
        DoubleTensor::new(&[1, 1], vec![2.0]).unwrap(),
    );
    let k = k.id();
    let (graph, mut state) = b.build().unwrap();
    state.clear(k).unwrap();
    let model = Model::new(graph, &state).unwrap();
    for seed in 0..20 {
        let mut probed = state.clone();
        model
            .probe_for_non_zero_probability(&mut probed, 100, &mut KeanuRng::seed_from_u64(seed))
            .unwrap();
        assert!(probed.f64(k).unwrap().as_slice()[0] < 1.0);
        assert!(model.log_prob(&probed).unwrap().is_finite());
    }
}

#[test]
fn non_increasing_times_are_an_error_not_an_impossible_point() {
    let b = GraphBuilder::new();
    let times = b.gaussian_with_shape(&[2], 0.0, 1.0);
    b.set_value(times, v(&[1.0, 0.5]));
    let y = b.ode(
        b.constant(v(&[1.0])),
        times,
        b.scalar(0.0),
        &[],
        OdeOptions::default(),
        |_, _, y, _| -y,
    );
    let y = y.id();
    let (graph, mut state) = b.build().unwrap();
    match state.lazy_eval(&graph, &[y], &mut rng()) {
        Err(e @ GraphError::InvalidNode { op: "ode", .. }) => assert!(!e.is_impossible_point()),
        other => panic!("expected an invalid ODE, got {other:?}"),
    }
}

#[test]
fn time_dependent_and_parameter_free_bodies() {
    // dy/dt = cos t, y(0) = 0: y = sin t, through a body with no parameters
    let b = GraphBuilder::new();
    let times = [0.5, 1.0, 3.0];
    let y = b.ode(
        b.constant(v(&[0.0])),
        b.constant(v(&times)),
        b.scalar(0.0),
        &[],
        tight(),
        |_, t, _, _| t.cos().reshape(&[1]),
    );
    let y = y.id();
    let (_, state) = evaluated(b);
    for (i, &t) in times.iter().enumerate() {
        assert_close(state.f64(y).unwrap().as_slice()[i], t.sin(), 1e-10, "sin t");
    }
}

#[test]
fn constant_parents_get_no_cotangent_but_latents_do() {
    // a constant initial state and a latent rate: the reverse sweep reaches the rate only
    let b = GraphBuilder::new();
    let k = b.gaussian(0.0, 1.0);
    b.set_value(k, DoubleTensor::scalar(0.3));
    let y0 = b.constant(v(&[1.0]));
    let y = b.ode(
        y0,
        b.constant(v(&[1.0])),
        b.scalar(0.0),
        &[k],
        tight(),
        |_, _, y, p| -(p[0] * y),
    );
    let (k, y0, y) = (k.id(), y0.id(), y.id());
    let (graph, state) = evaluated(b);
    let partials = reverse_partials(&graph, &state, y, &[k, y0]).unwrap();
    assert_close_rel(
        partials.get(k).unwrap().as_slice()[0],
        -(-0.3f64).exp(),
        1e-9,
        "dy/dk",
    );
    assert!(partials.get(y0).is_err());
}

#[test]
fn empty_output_times_give_an_empty_solution() {
    let b = GraphBuilder::new();
    let y = b.ode(
        b.constant(v(&[1.0, 2.0])),
        b.constant(v(&[])),
        b.scalar(0.0),
        &[],
        OdeOptions::default(),
        |_, _, y, _| -y,
    );
    assert_eq!(y.shape(), vec![0, 2]);
    let y = y.id();
    let (_, state) = evaluated(b);
    assert_eq!(state.f64(y).unwrap().shape(), &[0, 2]);
}

fn y0(b: &GraphBuilder) -> DoubleVar<'_> {
    b.constant(v(&[1.0, 2.0]))
}

fn times(b: &GraphBuilder) -> DoubleVar<'_> {
    b.constant(v(&[1.0]))
}

fn ode_error(build: impl FnOnce(&GraphBuilder)) -> String {
    let b = GraphBuilder::new();
    build(&b);
    match b.build() {
        Err(GraphError::InvalidNode { op: "ode", message }) => message,
        other => panic!("expected an invalid ODE, got {:?}", other.map(|_| ())),
    }
}

#[test]
fn bodies_and_parents_are_checked_when_building() {
    let message = ode_error(|b| {
        b.ode(
            y0(b),
            times(b),
            b.scalar(0.0),
            &[],
            OdeOptions::default(),
            |body, _, y, _| y + body.gaussian(0.0, 1.0),
        );
    });
    assert!(message.contains("probabilistic"), "{message}");
    let message = ode_error(|b| {
        b.ode(
            y0(b),
            times(b),
            b.scalar(0.0),
            &[],
            OdeOptions::default(),
            |_, _, y, _| y.sum(),
        );
    });
    assert!(message.contains("output"), "{message}");
    let message = ode_error(|b| {
        b.ode(
            b.constant(DoubleTensor::new(&[1, 2], vec![1.0, 2.0]).unwrap()),
            times(b),
            b.scalar(0.0),
            &[],
            OdeOptions::default(),
            |_, _, y, _| y,
        );
    });
    assert!(message.contains("vector"), "{message}");
    let message = ode_error(|b| {
        b.ode(
            y0(b),
            b.scalar(1.0),
            b.scalar(0.0),
            &[],
            OdeOptions::default(),
            |_, _, y, _| y,
        );
    });
    assert!(
        message.contains("output times must be a vector"),
        "{message}"
    );
    let message = ode_error(|b| {
        b.ode(
            y0(b),
            times(b),
            times(b),
            &[],
            OdeOptions::default(),
            |_, _, y, _| y,
        );
    });
    assert!(
        message.contains("initial time must be a scalar"),
        "{message}"
    );
    let message = ode_error(|b| {
        b.ode(
            y0(b),
            times(b),
            b.scalar(0.0),
            &[],
            OdeOptions::default().with_tolerances(0.0, 0.0),
            |_, _, y, _| y,
        );
    });
    assert!(message.contains("tolerances"), "{message}");
}
