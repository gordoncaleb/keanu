//! Structural time series: each component's matrices, the joint model's log density against a hand-built
//! linear-Gaussian state-space model, a regression's factor against the state-space density of the residual (with
//! its gradient against finite differences), and the builder's errors.

use keanu_distributions::{KeanuRng, LinearGaussianStateSpace};
use keanu_graph::op::DistributionKind;
use keanu_graph::sts::{Component, ComponentKind, StructuralModel, StructuralTimeSeries};
use keanu_graph::{GraphBuilder, GraphError, Model, NodeId, State};
use keanu_tensor::DoubleTensor;

mod common;
use common::{assert_close, t, v};

const Y: [f64; 10] = [1.0, 2.1, 0.4, 1.7, 2.6, 3.2, 1.5, 2.9, 3.8, 4.1];

/// Builds a model, gives every node a value and returns it with the description.
fn built(
    build: impl for<'b> FnOnce(&'b GraphBuilder) -> StructuralTimeSeries,
) -> (Model, State, StructuralTimeSeries) {
    let b = GraphBuilder::new();
    let sts = build(&b);
    let (graph, mut state) = b.build().unwrap();
    let all: Vec<NodeId> = graph.ids().collect();
    state
        .lazy_eval(&graph, &all, &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    (Model::new(graph, &state).unwrap(), state, sts)
}

/// The values of the state-space parameters.
fn matrices(state: &State, sts: &StructuralTimeSeries) -> Vec<DoubleTensor> {
    sts.state_space_parameters()
        .iter()
        .map(|&id| state.f64(id).unwrap().clone())
        .collect()
}

fn assert_tensor(actual: &DoubleTensor, expected: &DoubleTensor, what: &str) {
    assert_eq!(actual.shape(), expected.shape(), "{what}: shape");
    for (i, (a, e)) in actual
        .as_slice()
        .iter()
        .zip(expected.as_slice())
        .enumerate()
    {
        assert_close(*a, *e, 1e-12, &format!("{what}[{i}]"));
    }
}

/// The sample standard deviation of `Y`, the default priors' scale.
fn y_scale() -> f64 {
    let n = Y.len() as f64;
    let mean = Y.iter().sum::<f64>() / n;
    (Y.iter().map(|y| (y - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
}

#[test]
fn each_component_has_its_state_space_matrices() {
    // one component at a time, with fixed scales, against the matrices written out by hand
    let s = y_scale();
    type Case = (&'static str, usize, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);
    let cases: Vec<Case> = vec![
        // name, d, F, Q diagonal, H, initial mean
        ("level", 1, vec![1.0], vec![0.25], vec![1.0], vec![Y[0]]),
        (
            "trend",
            2,
            vec![1.0, 1.0, 0.0, 1.0],
            vec![0.25, 0.04],
            vec![1.0, 0.0],
            vec![Y[0], 0.0],
        ),
        (
            "seasonal",
            3,
            vec![-1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            vec![0.25, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0; 3],
        ),
        (
            "autoregressive",
            2,
            vec![0.6, -0.2, 1.0, 0.0],
            vec![0.25, 0.0],
            vec![1.0, 0.0],
            vec![0.0; 2],
        ),
    ];
    for (name, d, f, q, h, m0) in cases {
        let (_, state, sts) = built(|b| {
            let half = b.scalar(0.5);
            let component = match name {
                "level" => Component::local_level().with_scale(half),
                "trend" => Component::local_linear_trend()
                    .with_scale(half)
                    .with_slope_scale(b.scalar(0.2)),
                "seasonal" => Component::seasonal(4).with_scale(half),
                _ => Component::autoregressive(2)
                    .with_scale(half)
                    .with_coefficients(b.constant(DoubleTensor::vector(vec![0.6, -0.2]))),
            };
            StructuralModel::new(b, &Y)
                .with(component)
                .with_observation_noise_scale(b.scalar(0.3))
                .build()
                .unwrap()
        });
        assert_eq!(sts.state_dimension(), d, "{name}");
        let layout = &sts.components()[0];
        assert_eq!(layout.states(), 0..d, "{name}");
        assert_eq!(layout.observation(), h.as_slice(), "{name}");
        let m = matrices(&state, &sts);
        assert_tensor(&m[0], &DoubleTensor::vector(m0), &format!("{name} m0"));
        assert_tensor(
            &m[1],
            &DoubleTensor::eye(d).unwrap().map(|x| x * s * s),
            &format!("{name} P0"),
        );
        assert_tensor(&m[2], &t(&[d, d], &f), &format!("{name} F"));
        assert_tensor(
            &m[3],
            &DoubleTensor::vector(q).diag().unwrap(),
            &format!("{name} Q"),
        );
        assert_tensor(&m[4], &t(&[1, d], &h), &format!("{name} H"));
        assert_tensor(&m[5], &t(&[1, 1], &[0.09]), &format!("{name} R"));
    }
}

#[test]
fn a_trigonometric_seasonal_rotates_each_harmonic() {
    // period 4 with 2 harmonics: frequency π/2 (a rotating pair) and π (the Nyquist frequency, one state)
    let (_, state, sts) = built(|b| {
        StructuralModel::new(b, &Y)
            .with(Component::trigonometric_seasonal(4.0, 2).with_scale(b.scalar(0.5)))
            .build()
            .unwrap()
    });
    assert_eq!(sts.state_dimension(), 3);
    let m = matrices(&state, &sts);
    let (sin, cos) = (std::f64::consts::FRAC_PI_2).sin_cos();
    assert_tensor(
        &m[2],
        &t(&[3, 3], &[cos, sin, 0.0, -sin, cos, 0.0, 0.0, 0.0, -1.0]),
        "F",
    );
    assert_tensor(&m[3], &DoubleTensor::eye(3).unwrap().map(|x| x * 0.25), "Q");
    assert_tensor(&m[4], &t(&[1, 3], &[1.0, 0.0, 1.0]), "H");
    // a whole period of 12 with 6 harmonics has 11 states, as many as the dummy form of 12 seasons
    let (_, _, sts) = built(|b| {
        StructuralModel::new(b, &Y)
            .with(Component::trigonometric_seasonal(12.0, 6))
            .build()
            .unwrap()
    });
    assert_eq!(sts.state_dimension(), 11);
}

/// The local linear trend + seasonal(3) model with these scales, written out by hand.
fn hand_built(level: f64, slope: f64, drift: f64, noise: f64) -> LinearGaussianStateSpace<'static> {
    let s = y_scale();
    let q = [level * level, slope * slope, drift * drift, 0.0];
    #[rustfmt::skip]
    let f = [
        1.0, 1.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, -1.0, -1.0,
        0.0, 0.0, 1.0, 0.0,
    ];
    LinearGaussianStateSpace::new(
        DoubleTensor::vector(vec![Y[0], 0.0, 0.0, 0.0]),
        DoubleTensor::eye(4).unwrap().map(|x| x * s * s),
        t(&[4, 4], &f),
        DoubleTensor::vector(q.to_vec()).diag().unwrap(),
        t(&[1, 4], &[1.0, 0.0, 1.0, 0.0]),
        t(&[1, 1], &[noise * noise]),
    )
}

#[test]
fn a_trend_and_seasonal_model_is_the_hand_built_state_space_model() {
    let (model, mut state, sts) = built(|b| {
        StructuralModel::new(b, &Y)
            .with(Component::local_linear_trend())
            .with(Component::seasonal(3))
            .build()
            .unwrap()
    });
    assert_eq!(sts.state_dimension(), 4);
    assert!(!sts.has_regression());
    let names: Vec<String> = sts.parameters().into_iter().map(|(n, _)| n).collect();
    assert_eq!(
        names,
        [
            "trend.level_scale",
            "trend.slope_scale",
            "seasonal.drift_scale",
            "observation_noise_scale"
        ]
    );
    let graph = model.graph().clone();
    let y = DoubleTensor::new(&[Y.len(), 1], Y.to_vec()).unwrap();
    for values in [[0.3, 0.05, 0.2, 0.4], [1.1, 0.4, 0.01, 0.9]] {
        for ((_, id), value) in sts.parameters().iter().zip(values) {
            state
                .set_and_cascade(&graph, *id, DoubleTensor::scalar(value))
                .unwrap();
        }
        let [level, slope, drift, noise] = values;
        let expected = hand_built(level, slope, drift, noise).log_prob(&y).unwrap();
        let actual = keanu_graph::node_log_prob(&graph, &state, sts.likelihood()).unwrap();
        assert_close(actual, expected, 1e-10 * expected.abs(), "log density");
        // and the joint log probability adds the default priors: LogNormal(ln(0.05 s), 2) and LogNormal(ln(0.1 s), 2)
        let s = y_scale();
        let prior: f64 = values
            .iter()
            .zip([0.05, 0.05, 0.05, 0.1])
            .map(|(&x, m)| {
                let z = (x.ln() - (m * s).ln()) / 2.0;
                -0.5 * z * z - (2.0 * (2.0 * std::f64::consts::PI).sqrt()).ln() - x.ln()
            })
            .sum();
        assert_close(
            model.log_prob(&state).unwrap(),
            expected + prior,
            1e-9 * (expected + prior).abs(),
            "joint log probability",
        );
    }
}

#[test]
fn a_regression_is_an_offset_of_the_observations() {
    let x = t(
        &[10, 2],
        &[
            0.1, 1.0, 0.5, -1.0, 0.9, 0.3, -0.2, 0.7, 0.4, -0.5, 1.2, 0.0, 0.3, 0.6, -0.7, 1.1,
            0.8, -0.3, 0.2, 0.4,
        ],
    );
    let (model, mut state, sts) = built(|b| {
        StructuralModel::new(b, &Y)
            .with(Component::local_level().with_scale(b.scalar(0.4)))
            .with(Component::regression(x.clone()))
            .with_observation_noise_scale(b.scalar(0.5))
            .build()
            .unwrap()
    });
    assert!(sts.has_regression());
    let regression = sts.component("regression").unwrap();
    assert_eq!(
        regression.kind(),
        ComponentKind::Regression { covariates: 2 }
    );
    assert!(regression.states().is_empty());
    let weights = regression.weights().unwrap();
    let graph = model.graph().clone();
    assert!(model.factors().contains(&sts.likelihood()));
    let beta = [0.7, -1.3];
    state
        .set_and_cascade(&graph, weights, DoubleTensor::vector(beta.to_vec()))
        .unwrap();
    let residual: Vec<f64> = (0..10)
        .map(|i| Y[i] - x.as_slice()[2 * i] * beta[0] - x.as_slice()[2 * i + 1] * beta[1])
        .collect();
    let s = y_scale();
    let level = LinearGaussianStateSpace::new(
        DoubleTensor::vector(vec![Y[0]]),
        t(&[1, 1], &[s * s]),
        t(&[1, 1], &[1.0]),
        t(&[1, 1], &[0.16]),
        t(&[1, 1], &[1.0]),
        t(&[1, 1], &[0.25]),
    );
    let expected = level.log_prob(&t(&[10, 1], &residual)).unwrap();
    assert_close(
        state.f64(sts.likelihood()).unwrap().as_slice()[0],
        expected,
        1e-10 * expected.abs(),
        "factor",
    );
    // the gradient with respect to the weights against central differences of the log probability
    let gradient = model.log_prob_gradient(&state).unwrap();
    let g = gradient.get(weights).unwrap().as_slice().to_vec();
    for j in 0..2 {
        let h = 1e-6;
        let mut at = |delta: f64| {
            let mut b = beta;
            b[j] += delta;
            state
                .set_and_cascade(&graph, weights, DoubleTensor::vector(b.to_vec()))
                .unwrap();
            model.log_prob(&state).unwrap()
        };
        let fd = (at(h) - at(-h)) / (2.0 * h);
        assert_close(g[j], fd, 1e-6 * fd.abs().max(1.0), &format!("d/dβ{j}"));
    }
}

#[test]
fn a_state_space_log_density_node_checks_its_shapes() {
    let b = GraphBuilder::new();
    let one = || b.constant(t(&[1, 1], &[1.0]));
    let params = [
        b.constant(DoubleTensor::vector(vec![0.0])).id(),
        one().id(),
        one().id(),
        one().id(),
        one().id(),
        one().id(),
    ];
    let value = b.constant(t(&[3, 2], &[0.0; 6]));
    b.log_density(DistributionKind::LinearGaussianStateSpace, value, &params);
    match b.build() {
        Err(GraphError::InvalidNode { message, .. }) => {
            assert!(message.contains("[3, 2]"), "{message}")
        }
        other => panic!("expected a shape error, got {other:?}"),
    }
    let b = GraphBuilder::new();
    let one = || b.constant(t(&[1, 1], &[1.0]));
    let params = [
        b.constant(DoubleTensor::vector(vec![0.0])).id(),
        one().id(),
        one().id(),
        one().id(),
        one().id(),
        one().id(),
    ];
    let value = b.constant(t(&[3, 1], &[0.5, 0.1, -0.2]));
    let lp = b.log_density(DistributionKind::LinearGaussianStateSpace, value, &params);
    assert_eq!(lp.shape(), Vec::<usize>::new());
}

#[test]
fn the_builder_rejects_invalid_models() {
    let design = t(&[10, 1], &[1.0; 10]);
    type Build = fn(&GraphBuilder, DoubleTensor) -> Result<StructuralTimeSeries, GraphError>;
    let cases: Vec<(&str, Build, &str)> = vec![
        (
            "empty",
            |b, _| {
                StructuralModel::new(b, &[])
                    .with(Component::local_level())
                    .build()
            },
            "empty",
        ),
        (
            "no state",
            |b, x| {
                StructuralModel::new(b, &Y)
                    .with(Component::regression(x))
                    .build()
            },
            "at least one component",
        ),
        (
            "duplicate names",
            |b, _| {
                StructuralModel::new(b, &Y)
                    .with(Component::local_level())
                    .with(Component::local_level())
                    .build()
            },
            "two components",
        ),
        (
            "one season",
            |b, _| {
                StructuralModel::new(b, &Y)
                    .with(Component::seasonal(1))
                    .build()
            },
            "2 seasons",
        ),
        (
            "too many harmonics",
            |b, _| {
                StructuralModel::new(b, &Y)
                    .with(Component::trigonometric_seasonal(5.0, 3))
                    .build()
            },
            "harmonics",
        ),
        (
            "order 0",
            |b, _| {
                StructuralModel::new(b, &Y)
                    .with(Component::autoregressive(0))
                    .build()
            },
            "order",
        ),
        (
            "design rows",
            |b, _| {
                StructuralModel::new(b, &Y)
                    .with(Component::local_level())
                    .with(Component::regression(t(&[9, 1], &[1.0; 9])))
                    .build()
            },
            "one row per observed step",
        ),
        (
            "a zero covariate",
            |b, _| {
                StructuralModel::new(b, &Y)
                    .with(Component::local_level())
                    .with(Component::regression(t(&[10, 1], &[0.0; 10])))
                    .build()
            },
            "all zero",
        ),
        (
            "a slope scale on a level",
            |b, _| {
                StructuralModel::new(b, &Y)
                    .with(Component::local_level().with_slope_scale(b.scalar(1.0)))
                    .build()
            },
            "no slope scale",
        ),
        (
            "a vector scale",
            |b, _| {
                StructuralModel::new(b, &Y)
                    .with(
                        Component::local_level()
                            .with_scale(b.constant(DoubleTensor::vector(vec![1.0, 2.0]))),
                    )
                    .build()
            },
            "scalar",
        ),
        (
            "coefficients of the wrong length",
            |b, _| {
                StructuralModel::new(b, &Y)
                    .with(
                        Component::autoregressive(2)
                            .with_coefficients(b.constant(DoubleTensor::vector(vec![0.5]))),
                    )
                    .build()
            },
            "shape [2]",
        ),
    ];
    for (name, build, text) in cases {
        let b = GraphBuilder::new();
        let before = b.node_count();
        match build(&b, design.clone()) {
            Err(GraphError::InvalidNode { op: "sts", message }) => {
                assert!(
                    message.contains(text),
                    "{name}: {message:?} does not mention {text:?}"
                )
            }
            other => panic!("{name}: expected an sts error, got {other:?}"),
        }
        // nothing but the options' own constants was added
        assert!(b.node_count() <= before + 1, "{name}: nodes were added");
    }
}

/// Only the first level-type component starts at the series' first value: with a local level and a local linear
/// trend (in either order) the levels' initial means sum to y₀ rather than 2 y₀, and the seasonal states start at 0.
#[test]
fn only_the_first_level_component_starts_at_the_first_value() {
    for level_first in [true, false] {
        let (_, state, sts) = built(|b| {
            let level = Component::local_level().with_scale(b.scalar(0.5));
            let trend = Component::local_linear_trend()
                .with_scale(b.scalar(0.5))
                .with_slope_scale(b.scalar(0.2));
            let model = StructuralModel::new(b, &Y);
            let model = if level_first {
                model.with(level).with(trend)
            } else {
                model.with(trend).with(level)
            };
            model
                .with(Component::seasonal(3).with_scale(b.scalar(0.1)))
                .with_observation_noise_scale(b.scalar(0.3))
                .build()
                .unwrap()
        });
        let m0 = &matrices(&state, &sts)[0];
        // the first component's level is state 0 either way
        assert_tensor(m0, &v(&[Y[0], 0.0, 0.0, 0.0, 0.0]), "m0");
        let observed_mean: f64 = sts
            .components()
            .iter()
            .map(|c| {
                c.states()
                    .zip(c.observation())
                    .map(|(i, h)| h * m0.as_slice()[i])
                    .sum::<f64>()
            })
            .sum();
        assert_close(observed_mean, Y[0], 1e-15, "the initial observation mean");
    }
}
