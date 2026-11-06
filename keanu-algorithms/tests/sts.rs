//! Decomposition and forecasting of structural time series over posterior draws: the components sum to the
//! smoothed mean, single-draw results are the Kalman smoother's and forecast's Gaussian moments, mixture intervals
//! have the requested probability, and inputs are checked.

use keanu_algorithms::sts::StructuralPosterior;
use keanu_algorithms::{AlgorithmError, NetworkSamples};
use keanu_distributions::KeanuRng;
use keanu_graph::sts::{Component, StructuralModel, StructuralTimeSeries};
use keanu_graph::{GraphBuilder, Model, NodeId, State};
use keanu_tensor::DoubleTensor;
use keanu_tensor::special::{ndtr, ndtri};

const Y: [f64; 12] = [1.0, 2.1, 0.4, 1.7, 2.6, 3.2, 1.5, 2.9, 3.8, 4.1, 2.7, 4.4];

fn assert_close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{what}: {actual} is not within {tolerance} of {expected}"
    );
}

fn design(rows: usize, offset: usize) -> DoubleTensor {
    DoubleTensor::new(
        &[rows, 1],
        (0..rows)
            .map(|t| (((t + offset) as f64) * 0.9).sin())
            .collect(),
    )
    .unwrap()
}

/// A trend + seasonal(4) + AR(1) + regression model of `Y`.
fn model() -> (Model, State, StructuralTimeSeries) {
    let b = GraphBuilder::new();
    let sts = StructuralModel::new(&b, &Y)
        .with(Component::local_linear_trend())
        .with(Component::seasonal(4))
        .with(Component::autoregressive(1))
        .with(Component::regression(design(Y.len(), 0)))
        .build()
        .unwrap();
    let (graph, mut state) = b.build().unwrap();
    let all: Vec<NodeId> = graph.ids().collect();
    state
        .lazy_eval(&graph, &all, &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    (Model::new(graph, &state).unwrap(), state, sts)
}

/// Values of the parameters, in the order of `StructuralTimeSeries::parameters`: the trend's level and slope
/// scales, the seasonal drift, the AR coefficient and scale, the regression weight and the noise scale.
const DRAWS: [[f64; 7]; 3] = [
    [0.3, 0.05, 0.1, 0.5, 0.2, 0.8, 0.4],
    [0.6, 0.02, 0.3, -0.2, 0.4, 1.4, 0.3],
    [0.2, 0.10, 0.05, 0.1, 0.1, 0.2, 0.6],
];

fn samples(
    model: &Model,
    state: &mut State,
    sts: &StructuralTimeSeries,
    draws: &[[f64; 7]],
) -> NetworkSamples {
    let nodes: Vec<NodeId> = sts.parameters().into_iter().map(|(_, id)| id).collect();
    let mut samples = NetworkSamples::new(model.graph(), &nodes).unwrap();
    for draw in draws {
        for (&id, &value) in nodes.iter().zip(draw) {
            let value = if model.graph()[id].shape().is_empty() {
                DoubleTensor::scalar(value)
            } else {
                DoubleTensor::vector(vec![value])
            };
            state.set_and_cascade(model.graph(), id, value).unwrap();
        }
        samples
            .record(state, model.log_prob(state).unwrap())
            .unwrap();
    }
    samples
}

#[test]
fn the_components_sum_to_the_smoothed_mean() {
    let (model, mut state, sts) = model();
    let samples = samples(&model, &mut state, &sts, &DRAWS);
    let posterior = StructuralPosterior::new(&model, &state, &sts, &samples).unwrap();
    assert_eq!(posterior.len(), 3);
    let parts = posterior.decompose(0.8).unwrap();
    let names: Vec<&str> = parts.components.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["trend", "seasonal", "autoregressive", "regression"]);
    // the smoothed mean of the series at each draw, H m_t|T plus the regression, averaged
    let mut expected = vec![0.0; Y.len()];
    let x = design(Y.len(), 0);
    for (i, draw) in DRAWS.iter().enumerate() {
        let smoothed = posterior
            .model(i)
            .unwrap()
            .smooth(posterior.residual(i).unwrap())
            .unwrap();
        let d = sts.state_dimension();
        for (t, e) in expected.iter_mut().enumerate() {
            let m = &smoothed.means.as_slice()[t * d..(t + 1) * d];
            // H = [1, 0 | 1, 0, 0 | 1]
            *e += (m[0] + m[2] + m[5] + x.as_slice()[t] * draw[5]) / 3.0;
        }
    }
    for (t, &expected) in expected.iter().enumerate() {
        let sum: f64 = parts.components.iter().map(|(_, b)| b.mean[t]).sum();
        assert_close(sum, parts.total.mean[t], 1e-10, &format!("sum at {t}"));
        assert_close(
            parts.total.mean[t],
            expected,
            1e-10,
            &format!("smoothed mean at {t}"),
        );
        for (name, band) in &parts.components {
            assert!(
                band.lower[t] <= band.mean[t] + 1e-12 && band.mean[t] <= band.upper[t] + 1e-12,
                "{name} at {t}"
            );
        }
        // the regression is known exactly at each draw: its interval spans the draws' values of X β
        let reg = parts.component("regression").unwrap();
        let values: Vec<f64> = DRAWS.iter().map(|d| x.as_slice()[t] * d[5]).collect();
        let (lo, hi) = values
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                (a.min(v), b.max(v))
            });
        assert!(
            reg.lower[t] >= lo - 1e-9 && reg.upper[t] <= hi + 1e-9,
            "regression at {t}"
        );
    }
}

#[test]
fn one_draw_gives_the_gaussian_moments_and_intervals() {
    let (model, mut state, sts) = model();
    let samples = samples(&model, &mut state, &sts, &DRAWS[..1]);
    let posterior = StructuralPosterior::new(&model, &state, &sts, &samples).unwrap();
    let z = ndtri(0.95);
    let d = sts.state_dimension();
    // the smoother: the trend's contribution is the level state
    let smoothed = posterior
        .model(0)
        .unwrap()
        .smooth(posterior.residual(0).unwrap())
        .unwrap();
    let parts = posterior.decompose(0.9).unwrap();
    let trend = parts.component("trend").unwrap();
    for t in 0..Y.len() {
        let m = smoothed.means.as_slice()[t * d];
        let sd = smoothed.covariances.as_slice()[t * d * d].sqrt();
        assert_close(trend.mean[t], m, 1e-12, "trend mean");
        assert_close(trend.lower[t], m - z * sd, 1e-5 * sd, "trend lower");
        assert_close(trend.upper[t], m + z * sd, 1e-5 * sd, "trend upper");
    }
    // the forecast: the state-space forecast plus the future regression
    let horizon = 5;
    let future = design(horizon, Y.len());
    let forecast = posterior
        .forecast(horizon, std::slice::from_ref(&future), 0.9)
        .unwrap();
    let exact = posterior
        .model(0)
        .unwrap()
        .forecast(posterior.residual(0).unwrap(), horizon)
        .unwrap();
    for k in 0..horizon {
        let m = exact.observation_means.as_slice()[k] + future.as_slice()[k] * DRAWS[0][5];
        let sd = exact.observation_covariances.as_slice()[k].sqrt();
        let o = &forecast.observations;
        assert_close(o.mean[k], m, 1e-12, "forecast mean");
        assert_close(o.lower[k], m - z * sd, 1e-5 * sd, "forecast lower");
        assert_close(o.upper[k], m + z * sd, 1e-5 * sd, "forecast upper");
        let sum: f64 = forecast.components.iter().map(|(_, b)| b.mean[k]).sum();
        assert_close(sum, m, 1e-10, "forecast components");
    }
}

#[test]
fn mixture_intervals_have_the_requested_probability() {
    let (model, mut state, sts) = model();
    let samples = samples(&model, &mut state, &sts, &DRAWS);
    let posterior = StructuralPosterior::new(&model, &state, &sts, &samples).unwrap();
    let horizon = 4;
    let future = design(horizon, Y.len());
    let forecast = posterior
        .forecast(horizon, std::slice::from_ref(&future), 0.8)
        .unwrap();
    for k in 0..horizon {
        let moments: Vec<(f64, f64)> = (0..DRAWS.len())
            .map(|i| {
                let f = posterior
                    .model(i)
                    .unwrap()
                    .forecast(posterior.residual(i).unwrap(), horizon)
                    .unwrap();
                (
                    f.observation_means.as_slice()[k] + future.as_slice()[k] * DRAWS[i][5],
                    f.observation_covariances.as_slice()[k].sqrt(),
                )
            })
            .collect();
        let cdf = |x: f64| moments.iter().map(|(m, s)| ndtr((x - m) / s)).sum::<f64>() / 3.0;
        let o = &forecast.observations;
        assert_close(cdf(o.lower[k]), 0.1, 1e-6, "lower tail");
        assert_close(cdf(o.upper[k]), 0.9, 1e-6, "upper tail");
        let mean = moments.iter().map(|(m, _)| m).sum::<f64>() / 3.0;
        assert_close(o.mean[k], mean, 1e-12, "mixture mean");
    }
}

#[test]
fn simulated_forecast_paths_match_the_forecast_moments() {
    let (model, mut state, sts) = model();
    let draws = vec![DRAWS[1]; 4000];
    let samples = samples(&model, &mut state, &sts, &draws);
    let posterior = StructuralPosterior::new(&model, &state, &sts, &samples).unwrap();
    let horizon = 3;
    let future = design(horizon, Y.len());
    let forecast = posterior
        .forecast(horizon, std::slice::from_ref(&future), 0.5)
        .unwrap();
    let paths = posterior
        .sample_forecast(horizon, &[future], &mut KeanuRng::seed_from_u64(7))
        .unwrap();
    assert_eq!(paths.len(), 4000);
    for k in 0..horizon {
        let values: Vec<f64> = paths.iter().map(|p| p[k]).collect();
        let mean = values.iter().sum::<f64>() / 4000.0;
        let o = &forecast.observations;
        // the 50% interval of one Gaussian is mean ± 0.674 sd
        let sd = (o.upper[k] - o.lower[k]) / (2.0 * ndtri(0.75));
        assert_close(mean, o.mean[k], 4.0 * sd / 4000f64.sqrt(), "path mean");
        let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 3999.0;
        assert_close(var.sqrt(), sd, 0.05 * sd, "path sd");
    }
}

#[test]
fn inputs_are_checked() {
    let (model, mut state, sts) = model();
    let samples = samples(&model, &mut state, &sts, &DRAWS);
    let posterior = StructuralPosterior::new(&model, &state, &sts, &samples).unwrap();
    for level in [0.0, 1.0, f64::NAN] {
        assert!(matches!(
            posterior.decompose(level),
            Err(AlgorithmError::InvalidArgument(_))
        ));
    }
    // a regression needs its future covariates, of the right shape
    for covariates in [vec![], vec![design(3, 0)]] {
        match posterior.forecast(4, &covariates, 0.9) {
            Err(AlgorithmError::InvalidArgument(m)) => {
                assert!(m.contains("regression") || m.contains("[4, 1]"), "{m}")
            }
            other => panic!("expected an invalid argument, got {other:?}"),
        }
    }
    // the samples must record every parameter
    let nodes: Vec<NodeId> = sts.parameters().into_iter().map(|(_, id)| id).collect();
    let partial = NetworkSamples::new(model.graph(), &nodes[..2]).unwrap();
    match StructuralPosterior::new(&model, &state, &sts, &partial) {
        Err(AlgorithmError::InvalidArgument(m)) => assert!(m.contains("without samples"), "{m}"),
        other => panic!("expected missing latents, got {other:?}"),
    }
}
