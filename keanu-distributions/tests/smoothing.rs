//! Smoothing, decoding and forecasting: the Kalman (RTS) smoother against the exact Gaussian conditional of the
//! states given the observations, forward-filtering backward-sampling against the smoother, forecasts against
//! closed forms and simulation, and the hidden Markov chain's Viterbi path, posterior marginals and posterior
//! path draws against brute-force enumeration.

use keanu_distributions::{FutureStep, HiddenMarkovChain, KeanuRng, LinearGaussianStateSpace};
use keanu_tensor::DoubleTensor;

mod common;
use common::{assert_close, v};

fn m(rows: usize, cols: usize, values: &[f64]) -> DoubleTensor {
    DoubleTensor::new(&[rows, cols], values.to_vec()).unwrap()
}

/// A three-dimensional state observed in two dimensions.
fn model() -> LinearGaussianStateSpace<'static> {
    LinearGaussianStateSpace::new(
        v(&[0.5, -1.0, 0.2]),
        m(3, 3, &[1.0, 0.2, 0.0, 0.2, 0.8, 0.1, 0.0, 0.1, 0.5]),
        m(3, 3, &[0.9, 0.1, 0.0, -0.1, 0.8, 0.2, 0.0, 0.0, 0.7]),
        m(3, 3, &[0.3, 0.05, 0.0, 0.05, 0.2, 0.0, 0.0, 0.0, 0.1]),
        m(2, 3, &[1.0, 0.0, 0.5, 0.0, 1.0, -0.3]),
        m(2, 2, &[0.4, 0.1, 0.1, 0.3]),
    )
}

fn observations() -> DoubleTensor {
    m(
        5,
        2,
        &[0.7, -0.8, 0.2, -1.4, 1.1, -0.2, 0.4, 0.3, -0.5, 0.9],
    )
}

/// The exact posterior of the stacked states [x_0; ...; x_{T−1}] given the stacked observations, from the joint
/// Gaussian of states and observations: its mean [T d] and covariance [T d, T d].
fn exact_posterior(model: &LinearGaussianStateSpace<'_>, y: &DoubleTensor) -> (Vec<f64>, Vec<f64>) {
    let (t_len, m_obs) = (y.shape()[0], y.shape()[1]);
    let d = model.initial_mean.len();
    let f = &*model.transition_matrix;
    let h = &*model.observation_matrix;
    let mut means = vec![model.initial_mean.reshape(&[d as isize, 1]).unwrap()];
    let mut variances = vec![model.initial_covariance.clone().into_owned()];
    for t in 1..t_len {
        means.push(f.matmul(&means[t - 1]).unwrap());
        variances.push(
            f.matmul(&variances[t - 1])
                .unwrap()
                .matmul_transposed(f, false, true)
                .unwrap()
                .add(&model.transition_covariance)
                .unwrap(),
        );
    }
    // Cov(x_t, x_s) = F^(t − s) Var(x_s) for t ≥ s
    let cross = |t: usize, s: usize| -> DoubleTensor {
        if t >= s {
            let mut c = variances[s].clone();
            for _ in s..t {
                c = f.matmul(&c).unwrap();
            }
            c
        } else {
            let mut c = variances[t].clone();
            for _ in t..s {
                c = f.matmul(&c).unwrap();
            }
            c.transpose().unwrap()
        }
    };
    let (nx, ny) = (t_len * d, t_len * m_obs);
    let mut sxx = vec![0.0; nx * nx];
    let mut sxy = vec![0.0; nx * ny];
    let mut syy = vec![0.0; ny * ny];
    let mut mux = vec![0.0; nx];
    let mut muy = vec![0.0; ny];
    for t in 0..t_len {
        mux[t * d..(t + 1) * d].copy_from_slice(means[t].as_slice());
        muy[t * m_obs..(t + 1) * m_obs].copy_from_slice(h.matmul(&means[t]).unwrap().as_slice());
        for s in 0..t_len {
            let c = cross(t, s);
            let hx = c.matmul_transposed(h, false, true).unwrap(); // Cov(x_t, y_s)
            let mut yy = h.matmul(&hx).unwrap(); // Cov(y_t, y_s)
            if t == s {
                yy = yy.add(&model.observation_covariance).unwrap();
            }
            for i in 0..d {
                for j in 0..d {
                    sxx[(t * d + i) * nx + s * d + j] = c.as_slice()[i * d + j];
                }
                for j in 0..m_obs {
                    sxy[(t * d + i) * ny + s * m_obs + j] = hx.as_slice()[i * m_obs + j];
                }
            }
            for i in 0..m_obs {
                for j in 0..m_obs {
                    syy[(t * m_obs + i) * ny + s * m_obs + j] = yy.as_slice()[i * m_obs + j];
                }
            }
        }
    }
    let sxy = DoubleTensor::new(&[nx, ny], sxy).unwrap();
    let syy_inv = DoubleTensor::new(&[ny, ny], syy)
        .unwrap()
        .inverse()
        .unwrap();
    let residual: Vec<f64> = y.as_slice().iter().zip(&muy).map(|(a, b)| a - b).collect();
    let gain = sxy.matmul(&syy_inv).unwrap();
    let mean = gain
        .matmul(&DoubleTensor::new(&[ny, 1], residual).unwrap())
        .unwrap();
    let mean: Vec<f64> = mean
        .as_slice()
        .iter()
        .zip(&mux)
        .map(|(a, b)| a + b)
        .collect();
    let reduction = gain.matmul_transposed(&sxy, false, true).unwrap();
    let covariance: Vec<f64> = sxx
        .iter()
        .zip(reduction.as_slice())
        .map(|(a, b)| a - b)
        .collect();
    (mean, covariance)
}

#[test]
fn the_smoother_is_the_exact_gaussian_conditional() {
    let model = model();
    let y = observations();
    let smoothed = model.smooth(&y).unwrap();
    let (mean, covariance) = exact_posterior(&model, &y);
    let (t_len, d) = (5, 3);
    assert_eq!(smoothed.means.shape(), &[t_len, d]);
    assert_eq!(smoothed.covariances.shape(), &[t_len, d, d]);
    assert_eq!(smoothed.log_likelihood, model.log_prob(&y).unwrap());
    for t in 0..t_len {
        for i in 0..d {
            assert_close(
                smoothed.means.as_slice()[t * d + i],
                mean[t * d + i],
                1e-10,
                &format!("mean of x[{t}][{i}]"),
            );
            for j in 0..d {
                assert_close(
                    smoothed.covariances.as_slice()[(t * d + i) * d + j],
                    covariance[(t * d + i) * t_len * d + t * d + j],
                    1e-10,
                    &format!("covariance of x[{t}][{i}, {j}]"),
                );
            }
        }
    }
    // the last smoothed step is the last filtered one
    let filtered = model.filter(&y).unwrap();
    for (a, b) in smoothed.means.as_slice()[(t_len - 1) * d..]
        .iter()
        .zip(&filtered.means.as_slice()[(t_len - 1) * d..])
    {
        assert_eq!(a, b);
    }
}

#[test]
fn a_singular_predicted_covariance_is_smoothed_with_its_pseudo_inverse() {
    // a local level next to a state component that is known exactly (no initial or transition variance)
    let (q, r, p0) = (0.5, 0.3, 2.0);
    let model = LinearGaussianStateSpace::new(
        v(&[0.0, 1.5]),
        m(2, 2, &[p0, 0.0, 0.0, 0.0]),
        DoubleTensor::eye(2).unwrap(),
        m(2, 2, &[q, 0.0, 0.0, 0.0]),
        m(1, 2, &[1.0, 1.0]),
        m(1, 1, &[r]),
    );
    let y = m(4, 1, &[1.9, 2.4, 1.2, 2.0]);
    let smoothed = model.smooth(&y).unwrap();
    // the same local level, observed after subtracting the known component
    let level = LinearGaussianStateSpace::new(
        v(&[0.0]),
        m(1, 1, &[p0]),
        m(1, 1, &[1.0]),
        m(1, 1, &[q]),
        m(1, 1, &[1.0]),
        m(1, 1, &[r]),
    );
    let shifted = m(4, 1, &[0.4, 0.9, -0.3, 0.5]);
    let expected = level.smooth(&shifted).unwrap();
    for t in 0..4 {
        let (mean, var) = (smoothed.means.as_slice(), smoothed.covariances.as_slice());
        assert_close(mean[2 * t], expected.means.as_slice()[t], 1e-12, "level");
        assert_close(mean[2 * t + 1], 1.5, 1e-12, "known component");
        assert_close(
            var[4 * t],
            expected.covariances.as_slice()[t],
            1e-12,
            "level variance",
        );
        for i in 1..4 {
            assert_close(var[4 * t + i], 0.0, 1e-12, "known component variance");
        }
    }
    let draw = model
        .sample_posterior_states(&y, &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    for t in 0..4 {
        assert_close(
            draw.as_slice()[2 * t + 1],
            1.5,
            1e-12,
            "drawn known component",
        );
    }
}

#[test]
fn backward_sampled_states_have_the_posterior_moments() {
    let model = model();
    let y = observations();
    let (mean, covariance) = exact_posterior(&model, &y);
    let (t_len, d) = (5, 3);
    let n = t_len * d;
    let draws = 6000;
    let mut rng = KeanuRng::seed_from_u64(7);
    let mut sum = vec![0.0; n];
    let mut products = vec![0.0; n * n];
    for _ in 0..draws {
        let x = model.sample_posterior_states(&y, &mut rng).unwrap();
        assert_eq!(x.shape(), &[t_len, d]);
        let x = x.as_slice();
        for i in 0..n {
            sum[i] += x[i];
            for j in 0..n {
                products[i * n + j] += x[i] * x[j];
            }
        }
    }
    let count = draws as f64;
    for i in 0..n {
        let sample_mean = sum[i] / count;
        let sd = covariance[i * n + i].sqrt();
        // within 4.5 standard errors
        assert_close(
            sample_mean,
            mean[i],
            4.5 * sd / count.sqrt(),
            &format!("mean {i}"),
        );
    }
    // the joint covariance, including across steps (which the smoother's marginals don't give)
    for i in 0..n {
        for j in 0..n {
            let sample = products[i * n + j] / count - sum[i] * sum[j] / (count * count);
            let expected = covariance[i * n + j];
            let scale = (covariance[i * n + i] * covariance[j * n + j]).sqrt();
            // the standard error of a sample covariance is at most about scale · √(2 / n)
            assert_close(
                sample,
                expected,
                4.5 * scale * (2.0 / count).sqrt(),
                &format!("covariance [{i}, {j}]"),
            );
        }
    }
}

/// A local level model: x_{t+1} = x_t + N(0, q), y_t = x_t + N(0, r).
fn local_level(q: f64, r: f64) -> LinearGaussianStateSpace<'static> {
    LinearGaussianStateSpace::new(
        v(&[0.0]),
        m(1, 1, &[4.0]),
        m(1, 1, &[1.0]),
        m(1, 1, &[q]),
        m(1, 1, &[1.0]),
        m(1, 1, &[r]),
    )
}

#[test]
fn local_level_forecast_variance_grows_linearly() {
    let (q, r) = (0.3, 0.5);
    let model = local_level(q, r);
    let y = m(6, 1, &[0.2, 0.7, 0.4, 1.3, 1.1, 1.6]);
    let filtered = model.filter(&y).unwrap();
    let (last_mean, last_var) = (
        filtered.means.as_slice()[5],
        filtered.covariances.as_slice()[5],
    );
    let forecast = model.forecast(&y, 8).unwrap();
    assert_eq!(forecast.observation_means.shape(), &[8, 1]);
    assert_eq!(forecast.state_covariances.shape(), &[8, 1, 1]);
    for k in 0..8 {
        let steps = (k + 1) as f64;
        assert_close(
            forecast.state_means.as_slice()[k],
            last_mean,
            1e-12,
            "state mean",
        );
        assert_close(
            forecast.observation_means.as_slice()[k],
            last_mean,
            1e-12,
            "mean",
        );
        assert_close(
            forecast.state_covariances.as_slice()[k],
            last_var + steps * q,
            1e-12,
            "state variance",
        );
        assert_close(
            forecast.observation_covariances.as_slice()[k],
            last_var + steps * q + r,
            1e-12,
            "observation variance",
        );
    }
    // the same with the matrices given step by step
    let step = FutureStep::new(
        m(1, 1, &[1.0]),
        m(1, 1, &[q]),
        m(1, 1, &[1.0]),
        m(1, 1, &[r]),
    );
    let with = model.forecast_with(&y, &vec![step; 8]).unwrap();
    assert_eq!(with, forecast);
    // an empty series forecasts from the initial state
    let prior = model.forecast(&DoubleTensor::zeros(&[0, 1]), 2).unwrap();
    assert_eq!(prior.state_covariances.as_slice(), &[4.0, 4.0 + q]);
    assert_eq!(
        prior.observation_covariances.as_slice(),
        &[4.0 + r, 4.0 + q + r]
    );
}

#[test]
fn forecasts_follow_time_varying_future_matrices() {
    // a level observed with a known regressor in the future: y = level · z_k + noise
    let (q, r) = (0.2, 0.4);
    let model = local_level(q, r);
    let y = m(3, 1, &[1.0, 1.4, 0.9]);
    let filtered = model.filter(&y).unwrap();
    let (mean, var) = (
        filtered.means.as_slice()[2],
        filtered.covariances.as_slice()[2],
    );
    let regressors = [2.0, -1.0, 0.5];
    let future: Vec<FutureStep> = regressors
        .iter()
        .map(|&z| FutureStep::new(m(1, 1, &[1.0]), m(1, 1, &[q]), m(1, 1, &[z]), m(1, 1, &[r])))
        .collect();
    let forecast = model.forecast_with(&y, &future).unwrap();
    for (k, z) in regressors.iter().enumerate() {
        let state_var = var + (k + 1) as f64 * q;
        assert_close(
            forecast.observation_means.as_slice()[k],
            z * mean,
            1e-12,
            "mean",
        );
        assert_close(
            forecast.observation_covariances.as_slice()[k],
            z * z * state_var + r,
            1e-12,
            "variance",
        );
    }
    let wrong = [FutureStep::new(
        m(1, 1, &[1.0]),
        m(1, 1, &[q]),
        m(1, 2, &[1.0, 0.0]),
        m(1, 1, &[r]),
    )];
    assert!(model.forecast_with(&y, &wrong).is_err());
}

#[test]
fn sampled_forecasts_have_the_forecast_moments() {
    let model = model();
    let y = observations();
    let horizon = 3;
    let forecast = model.forecast(&y, horizon).unwrap();
    let mut rng = KeanuRng::seed_from_u64(11);
    let draws = 6000;
    let obs = 2;
    let mut sum = vec![0.0; horizon * obs];
    let mut squares = vec![0.0; horizon * obs];
    for _ in 0..draws {
        let path = model.sample_forecast(&y, horizon, &mut rng).unwrap();
        assert_eq!(path.states.shape(), &[horizon, 3]);
        assert_eq!(path.observations.shape(), &[horizon, obs]);
        for (i, x) in path.observations.as_slice().iter().enumerate() {
            sum[i] += x;
            squares[i] += x * x;
        }
    }
    let count = draws as f64;
    for k in 0..horizon {
        for i in 0..obs {
            let c = k * obs + i;
            let expected_mean = forecast.observation_means.as_slice()[c];
            let expected_var = forecast.observation_covariances.as_slice()[(k * obs + i) * obs + i];
            let sample_mean = sum[c] / count;
            let sample_var = squares[c] / count - sample_mean * sample_mean;
            let sd = expected_var.sqrt();
            assert_close(sample_mean, expected_mean, 4.5 * sd / count.sqrt(), "mean");
            assert_close(
                sample_var,
                expected_var,
                4.5 * expected_var * (2.0 / count).sqrt(),
                "variance",
            );
        }
    }
}

#[test]
fn impossible_observations_have_no_posterior() {
    // a deterministic observation of a deterministic state: the innovation covariance is zero
    let model = LinearGaussianStateSpace::new(
        v(&[0.0]),
        m(1, 1, &[0.0]),
        m(1, 1, &[1.0]),
        m(1, 1, &[0.0]),
        m(1, 1, &[1.0]),
        m(1, 1, &[0.0]),
    );
    let y = m(2, 1, &[0.1, 0.2]);
    assert!(model.smooth(&y).is_err());
    assert!(model.forecast(&y, 2).is_err());
    assert!(
        model
            .sample_posterior_states(&y, &mut KeanuRng::seed_from_u64(1))
            .is_err()
    );
    assert!(model.smooth(&m(2, 2, &[0.0; 4])).is_err());
}

// ---------------------------------------------------------------------------------------------------------------
// hidden Markov chains

fn chain() -> HiddenMarkovChain<'static> {
    HiddenMarkovChain::new(
        v(&[0.5, 0.3, 0.2]),
        m(3, 3, &[0.7, 0.2, 0.1, 0.25, 0.5, 0.25, 0.1, 0.3, 0.6]),
    )
}

/// Emission log probabilities [T, K] for T = 5, K = 3.
fn emissions() -> DoubleTensor {
    m(
        5,
        3,
        &[
            -0.3, -1.7, -2.2, //
            -2.5, -0.4, -1.1, //
            -1.0, -1.0, -0.2, //
            -0.6, -3.0, -1.4, //
            -2.0, -0.1, -0.9,
        ],
    )
}

/// Every path of `t_len` states with its joint probability p(z, x).
fn paths(chain: &HiddenMarkovChain<'_>, e: &DoubleTensor) -> Vec<(Vec<usize>, f64)> {
    let (t_len, k) = (e.shape()[0], e.shape()[1]);
    let (initial, transition, e) = (
        chain.initial.as_slice(),
        chain.transition.as_slice(),
        e.as_slice(),
    );
    (0..k.pow(t_len as u32))
        .map(|code| {
            let path: Vec<usize> = (0..t_len).map(|t| code / k.pow(t as u32) % k).collect();
            let mut p = initial[path[0]] * e[path[0]].exp();
            for t in 1..t_len {
                p *= transition[path[t - 1] * k + path[t]] * e[t * k + path[t]].exp();
            }
            (path, p)
        })
        .collect()
}

#[test]
fn viterbi_finds_the_most_probable_path() {
    let chain = chain();
    let e = emissions();
    let all = paths(&chain, &e);
    let (best, p) = all
        .iter()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .cloned()
        .unwrap();
    let (path, log_prob) = chain.viterbi(&e).unwrap();
    assert_eq!(path, best);
    assert_close(log_prob, p.ln(), 1e-12, "log joint probability");
    // a chain with a zero transition can't take it
    let blocked = HiddenMarkovChain::new(v(&[1.0, 0.0]), m(2, 2, &[0.0, 1.0, 1.0, 0.0]));
    let e = m(4, 2, &[0.0, 5.0, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0]);
    assert_eq!(blocked.viterbi(&e).unwrap().0, vec![0, 1, 0, 1]);
    // impossible observations
    let impossible = m(1, 2, &[f64::NEG_INFINITY, 0.0]);
    assert!(blocked.viterbi(&impossible).is_err());
    assert!(blocked.posterior_marginals(&impossible).is_err());
}

#[test]
fn posterior_marginals_are_enumerated_path_probabilities() {
    let chain = chain();
    let e = emissions();
    let all = paths(&chain, &e);
    let total: f64 = all.iter().map(|(_, p)| p).sum();
    let marginals = chain.posterior_marginals(&e).unwrap();
    assert_eq!(marginals.shape(), &[5, 3]);
    for t in 0..5 {
        for j in 0..3 {
            let expected: f64 = all
                .iter()
                .filter(|(path, _)| path[t] == j)
                .map(|(_, p)| p / total)
                .sum();
            assert_close(marginals.as_slice()[t * 3 + j], expected, 1e-12, "marginal");
        }
    }
    // they are the gradient with respect to the emission log probabilities
    let (_, gradient) = chain.log_likelihood_and_gradient(&e).unwrap();
    assert_eq!(gradient.unwrap().emission, marginals);
}

#[test]
fn backward_sampled_paths_follow_the_posterior() {
    let chain = chain();
    let e = emissions();
    let all = paths(&chain, &e);
    let total: f64 = all.iter().map(|(_, p)| p).sum();
    let k: usize = 3;
    let draws = 20000;
    let mut counts = vec![0usize; all.len()];
    let mut rng = KeanuRng::seed_from_u64(5);
    for _ in 0..draws {
        let path = chain.sample_posterior_states(&e, &mut rng).unwrap();
        let code: usize = path
            .iter()
            .enumerate()
            .map(|(t, s)| s * k.pow(t as u32))
            .sum();
        counts[code] += 1;
    }
    for (code, (_, p)) in all.iter().enumerate() {
        let expected = p / total;
        let observed = counts[code] as f64 / draws as f64;
        let se = (expected * (1.0 - expected) / draws as f64).sqrt();
        assert_close(observed, expected, 4.5 * se + 1e-4, &format!("path {code}"));
    }
}
