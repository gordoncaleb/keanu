//! Hidden Markov chains and linear-Gaussian state-space models: agreement with brute-force enumeration and with the
//! multivariate Gaussian of the stacked observations, numerical stability, gradients against finite differences,
//! simulated moments and invalid parameters.

use keanu_distributions::{
    DistributionError, HiddenMarkovChain, KeanuRng, LinearGaussianStateSpace, MultivariateGaussian,
};
use keanu_tensor::DoubleTensor;

mod common;
use common::{assert_close_1p, v};

fn m(rows: usize, cols: usize, values: &[f64]) -> DoubleTensor {
    DoubleTensor::new(&[rows, cols], values.to_vec()).unwrap()
}

/// A three-state chain.
fn chain() -> HiddenMarkovChain<'static> {
    HiddenMarkovChain::new(
        v(&[0.5, 0.3, 0.2]),
        m(3, 3, &[0.7, 0.2, 0.1, 0.25, 0.5, 0.25, 0.1, 0.3, 0.6]),
    )
}

/// Emission log probabilities [T, K] for T = 4, K = 3.
fn emissions() -> DoubleTensor {
    m(
        4,
        3,
        &[
            -0.3, -1.7, -2.2, //
            -2.5, -0.4, -1.1, //
            -1.0, -1.0, -0.2, //
            -0.6, -3.0, -1.4,
        ],
    )
}

#[test]
fn the_hmm_log_likelihood_sums_every_state_path() {
    let chain = chain();
    let e = emissions();
    let (initial, transition) = (chain.initial.as_slice(), chain.transition.as_slice());
    let (t_len, k): (usize, usize) = (4, 3);
    let mut total: f64 = 0.0;
    for code in 0..k.pow(t_len as u32) {
        let path: Vec<usize> = (0..t_len).map(|t| code / k.pow(t as u32) % k).collect();
        let mut p = initial[path[0]] * e.as_slice()[path[0]].exp();
        for t in 1..t_len {
            p *= transition[path[t - 1] * k + path[t]] * e.as_slice()[t * k + path[t]].exp();
        }
        total += p;
    }
    let log_likelihood = chain.log_likelihood(&e).unwrap();
    assert_close_1p(
        log_likelihood,
        total.ln(),
        1e-12,
        &format!("{log_likelihood} vs {}", total.ln()),
    );
}

#[test]
fn hmm_gradients_match_finite_differences() {
    let chain = chain();
    let e = emissions();
    let (log_likelihood, gradient) = chain.log_likelihood_and_gradient(&e).unwrap();
    let gradient = gradient.unwrap();
    assert_close_1p(
        log_likelihood,
        chain.log_likelihood(&e).unwrap(),
        1e-14,
        "value",
    );
    let h = 1e-6;
    let numeric = |f: &dyn Fn(f64) -> f64| (f(h) - f(-h)) / (2.0 * h);
    for i in 0..3 {
        let d = numeric(&|delta| {
            let mut initial = chain.initial.as_slice().to_vec();
            initial[i] += delta;
            // the likelihood is linear in each probability, so leaving the simplex is harmless here, but the
            // chain checks its probabilities: evaluate the unnormalised forward recursion directly instead
            unnormalised(&initial, chain.transition.as_slice(), &e)
        });
        assert_close_1p(
            gradient.initial.as_slice()[i],
            d,
            1e-6,
            &format!("initial {i}"),
        );
    }
    for i in 0..9 {
        let d = numeric(&|delta| {
            let mut transition = chain.transition.as_slice().to_vec();
            transition[i] += delta;
            unnormalised(chain.initial.as_slice(), &transition, &e)
        });
        assert_close_1p(
            gradient.transition.as_slice()[i],
            d,
            1e-6,
            &format!("transition {i}"),
        );
    }
    for i in 0..12 {
        let d = numeric(&|delta| {
            let mut emission = e.as_slice().to_vec();
            emission[i] += delta;
            chain
                .log_likelihood(&DoubleTensor::new(&[4, 3], emission).unwrap())
                .unwrap()
        });
        assert_close_1p(
            gradient.emission.as_slice()[i],
            d,
            1e-6,
            &format!("emission {i}"),
        );
    }
    // the emission gradient is the posterior state marginals: each row sums to 1
    for row in gradient.emission.as_slice().chunks(3) {
        assert_close_1p(row.iter().sum(), 1.0, 1e-12, "value");
    }
}

/// The log of Σ over paths of the (possibly unnormalised) path weights, by the plain forward recursion.
fn unnormalised(initial: &[f64], transition: &[f64], e: &DoubleTensor) -> f64 {
    let k = initial.len();
    let e = e.as_slice();
    let mut alpha: Vec<f64> = (0..k).map(|j| initial[j] * e[j].exp()).collect();
    for t in 1..e.len() / k {
        alpha = (0..k)
            .map(|j| {
                (0..k)
                    .map(|i| alpha[i] * transition[i * k + j])
                    .sum::<f64>()
                    * e[t * k + j].exp()
            })
            .collect();
    }
    alpha.iter().sum::<f64>().ln()
}

#[test]
fn zero_probabilities_give_finite_gradients() {
    let chain = HiddenMarkovChain::new(v(&[1.0, 0.0]), m(2, 2, &[0.0, 1.0, 1.0, 0.0]));
    let e = m(3, 2, &[-1.0, -2.0, -0.5, -0.1, -3.0, -0.2]);
    let (log_likelihood, gradient) = chain.log_likelihood_and_gradient(&e).unwrap();
    // the only path is 0, 1, 0
    assert_close_1p(log_likelihood, -1.0 - 0.1 - 3.0, 1e-12, "value");
    let gradient = gradient.unwrap();
    for g in [&gradient.initial, &gradient.transition, &gradient.emission] {
        assert!(g.as_slice().iter().all(|x| x.is_finite()), "{g:?}");
    }
    assert_eq!(
        gradient.emission.as_slice(),
        &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    );
    // an impossible sequence has no gradient
    let (log_likelihood, gradient) =
        HiddenMarkovChain::new(v(&[1.0, 0.0]), m(2, 2, &[1.0, 0.0, 0.0, 1.0]))
            .log_likelihood_and_gradient(&m(2, 2, &[f64::NEG_INFINITY, 0.0, 0.0, 0.0]))
            .unwrap();
    assert_eq!(log_likelihood, f64::NEG_INFINITY);
    assert!(gradient.is_none());
}

#[test]
fn simulated_states_follow_the_stationary_distribution() {
    // stationary distribution of [[0.9, 0.1], [0.3, 0.7]] is (0.75, 0.25)
    let chain = HiddenMarkovChain::new(v(&[0.75, 0.25]), m(2, 2, &[0.9, 0.1, 0.3, 0.7]));
    let mut rng = KeanuRng::seed_from_u64(3);
    let states = chain.sample_states(40_000, &mut rng).unwrap();
    let ones = states.iter().filter(|&&s| s == 1).count() as f64 / states.len() as f64;
    // autocorrelated draws: lag-1 correlation 0.6 inflates the variance by (1 + 0.6) / (1 - 0.6) = 4, so the
    // standard error is sqrt(4 × 0.1875 / 40000) ≈ 0.0043
    assert!((ones - 0.25).abs() < 0.02, "{ones}");
    let stays = states.windows(2).filter(|w| w[0] == 0 && w[1] == 0).count() as f64;
    let zeros = states[..states.len() - 1]
        .iter()
        .filter(|&&s| s == 0)
        .count() as f64;
    assert!((stays / zeros - 0.9).abs() < 0.01, "{}", stays / zeros);
}

#[test]
fn invalid_chains_are_errors() {
    let e = m(2, 2, &[0.0; 4]);
    let invalid = |chain: HiddenMarkovChain<'_>, e: &DoubleTensor| {
        matches!(
            chain.log_likelihood(e),
            Err(DistributionError::InvalidParameter(_) | DistributionError::InvalidValue(_))
        )
    };
    // probabilities off the simplex
    assert!(invalid(
        HiddenMarkovChain::new(v(&[0.6, 0.6]), m(2, 2, &[0.5; 4])),
        &e
    ));
    assert!(invalid(
        HiddenMarkovChain::new(v(&[0.5, 0.5]), m(2, 2, &[0.5, 0.5, 0.9, 0.2])),
        &e
    ));
    assert!(invalid(
        HiddenMarkovChain::new(v(&[1.5, -0.5]), m(2, 2, &[0.5; 4])),
        &e
    ));
    assert!(invalid(
        HiddenMarkovChain::new(v(&[f64::NAN, 0.5]), m(2, 2, &[0.5; 4])),
        &e
    ));
    // shapes
    assert!(invalid(
        HiddenMarkovChain::new(v(&[0.5, 0.5]), v(&[0.5; 4])),
        &e
    ));
    assert!(invalid(
        HiddenMarkovChain::new(v(&[0.5, 0.5]), m(2, 3, &[1.0 / 3.0; 6])),
        &e
    ));
    assert!(invalid(
        HiddenMarkovChain::new(v(&[0.5, 0.5]), m(3, 3, &[1.0 / 3.0; 9])),
        &e
    ));
    assert!(invalid(
        HiddenMarkovChain::new(v(&[]), m(0, 0, &[])),
        &m(2, 0, &[])
    ));
    let fine = || HiddenMarkovChain::new(v(&[0.5, 0.5]), m(2, 2, &[0.5; 4]));
    assert!(invalid(fine(), &v(&[0.0, 0.0])));
    assert!(invalid(fine(), &m(2, 3, &[0.0; 6])));
    assert!(invalid(fine(), &m(0, 2, &[])));
    assert!(
        fine()
            .sample_states(0, &mut KeanuRng::seed_from_u64(1))
            .is_err()
    );
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

#[test]
fn the_kalman_log_likelihood_is_the_stacked_gaussian_density() {
    let model = model();
    let y = observations();
    let (t_len, d, obs) = (5, 3, 2);
    let f = &*model.transition_matrix;
    let h = &*model.observation_matrix;
    // state means and covariances, and the cross covariances Cov(x_t, x_s) = F^(t - s) Var(x_s)
    let mut means = vec![model.initial_mean.reshape(&[3, 1]).unwrap()];
    let mut variances = vec![model.initial_covariance.clone().into_owned()];
    for t in 1..t_len {
        means.push(f.matmul(&means[t - 1]).unwrap());
        let next = f
            .matmul(&variances[t - 1])
            .unwrap()
            .matmul_transposed(f, false, true)
            .unwrap()
            .add(&model.transition_covariance)
            .unwrap();
        variances.push(next);
    }
    let mut mean = vec![0.0; t_len * obs];
    let mut covariance = vec![0.0; t_len * obs * t_len * obs];
    for t in 0..t_len {
        let mt = h.matmul(&means[t]).unwrap();
        mean[t * obs..(t + 1) * obs].copy_from_slice(mt.as_slice());
        for s in 0..=t {
            let mut cross = variances[s].clone();
            for _ in s..t {
                cross = f.matmul(&cross).unwrap();
            }
            let mut block = h
                .matmul(&cross)
                .unwrap()
                .matmul_transposed(h, false, true)
                .unwrap();
            if s == t {
                block = block.add(&model.observation_covariance).unwrap();
            }
            for i in 0..obs {
                for j in 0..obs {
                    let value = block.as_slice()[i * obs + j];
                    covariance[(t * obs + i) * t_len * obs + s * obs + j] = value;
                    covariance[(s * obs + j) * t_len * obs + t * obs + i] = value;
                }
            }
        }
    }
    let _ = d;
    let stacked = MultivariateGaussian::new(
        v(&mean),
        DoubleTensor::new(&[t_len * obs, t_len * obs], covariance).unwrap(),
    )
    .log_prob(&y.reshape(&[10]).unwrap())
    .unwrap();
    let log_likelihood = model.log_prob(&y).unwrap();
    assert_close_1p(
        log_likelihood,
        stacked,
        1e-10,
        &format!("{log_likelihood} vs {stacked}"),
    );
    let filtered = model.filter(&y).unwrap();
    assert_eq!(filtered.log_likelihood, log_likelihood);
    assert_eq!(filtered.means.shape(), &[5, 3]);
    assert_eq!(filtered.covariances.shape(), &[5, 3, 3]);
}

#[test]
fn kalman_gradients_match_finite_differences() {
    let model = model();
    let y = observations();
    let gradient = model.d_log_prob(&y).unwrap();
    let h = 1e-6;
    type Getter = fn(&LinearGaussianStateSpace<'static>) -> DoubleTensor;
    let parameters: [(&str, Getter, &DoubleTensor, bool); 6] = [
        (
            "initial_mean",
            |m| m.initial_mean.clone().into_owned(),
            &gradient.initial_mean,
            false,
        ),
        (
            "initial_covariance",
            |m| m.initial_covariance.clone().into_owned(),
            &gradient.initial_covariance,
            true,
        ),
        (
            "transition_matrix",
            |m| m.transition_matrix.clone().into_owned(),
            &gradient.transition_matrix,
            false,
        ),
        (
            "transition_covariance",
            |m| m.transition_covariance.clone().into_owned(),
            &gradient.transition_covariance,
            true,
        ),
        (
            "observation_matrix",
            |m| m.observation_matrix.clone().into_owned(),
            &gradient.observation_matrix,
            false,
        ),
        (
            "observation_covariance",
            |m| m.observation_covariance.clone().into_owned(),
            &gradient.observation_covariance,
            true,
        ),
    ];
    for (index, (name, get, analytic, symmetric)) in parameters.into_iter().enumerate() {
        let base = get(&model);
        assert_eq!(analytic.shape(), base.shape(), "{name}");
        let n = base.shape().last().copied().unwrap();
        for i in 0..base.as_slice().len() {
            // covariances are symmetric: move (i, j) and (j, i) together, whose derivative is g_ij + g_ji
            let mirror = if symmetric { (i % n) * n + i / n } else { i };
            let at = |delta: f64| {
                let mut values = base.as_slice().to_vec();
                values[i] += delta;
                if mirror != i {
                    values[mirror] += delta;
                }
                let changed = DoubleTensor::new(base.shape(), values).unwrap();
                let mut p: Vec<DoubleTensor> = parameters_of(&model);
                p[index] = changed;
                let [a, b, c, d, e, f] = <[DoubleTensor; 6]>::try_from(p).unwrap();
                LinearGaussianStateSpace::new(a, b, c, d, e, f)
                    .log_prob(&y)
                    .unwrap()
            };
            let numeric = (at(h) - at(-h)) / (2.0 * h);
            let g = analytic.as_slice();
            let expected = if mirror != i { g[i] + g[mirror] } else { g[i] };
            assert_close_1p(
                expected,
                numeric,
                1e-5,
                &format!("{name}[{i}]: {expected} vs {numeric}"),
            );
            if symmetric {
                assert_close_1p(
                    g[i],
                    g[mirror],
                    1e-12,
                    &format!("{name} gradient is symmetric"),
                );
            }
        }
    }
    for i in 0..10 {
        let at = |delta: f64| {
            let mut values = y.as_slice().to_vec();
            values[i] += delta;
            model
                .log_prob(&DoubleTensor::new(&[5, 2], values).unwrap())
                .unwrap()
        };
        let numeric = (at(h) - at(-h)) / (2.0 * h);
        assert_close_1p(gradient.x.as_slice()[i], numeric, 1e-5, &format!("x[{i}]"));
    }
}

/// The gradient's fields in the order (x, then each parameter as in `LinearGaussianStateSpace::new`).
fn gradient_fields(
    g: &keanu_distributions::LinearGaussianStateSpaceGradient,
) -> [&DoubleTensor; 7] {
    [
        &g.x,
        &g.initial_mean,
        &g.initial_covariance,
        &g.transition_matrix,
        &g.transition_covariance,
        &g.observation_matrix,
        &g.observation_covariance,
    ]
}

/// A direction for an input of `shape`, made symmetric for a covariance.
fn direction(shape: &[usize], seed: f64, symmetric: bool) -> Vec<f64> {
    let len: usize = shape.iter().product();
    let mut v: Vec<f64> = (0..len).map(|i| ((i as f64) * 0.7 + seed).sin()).collect();
    if symmetric {
        let n = shape[1];
        for i in 0..n {
            for j in 0..i {
                v[j * n + i] = v[i * n + j];
            }
        }
    }
    v
}

#[test]
fn kalman_hessian_vector_products_match_finite_differences_of_the_gradient() {
    let model = model();
    let y = observations();
    let mut inputs = vec![y.clone()];
    inputs.extend(parameters_of(&model));
    let symmetric = [false, false, true, false, true, false, true];
    let b = 2;
    // two directions, moving every input at once
    let directions: Vec<Vec<Vec<f64>>> = (0..b)
        .map(|k| {
            inputs
                .iter()
                .zip(symmetric)
                .enumerate()
                .map(|(u, (t, s))| direction(t.shape(), 0.3 + u as f64 + 1.7 * k as f64, s))
                .collect()
        })
        .collect();
    let tangents: Vec<DoubleTensor> = inputs
        .iter()
        .enumerate()
        .map(|(u, t)| {
            let mut shape = vec![b];
            shape.extend_from_slice(t.shape());
            let data = (0..b).flat_map(|k| directions[k][u].clone()).collect();
            DoubleTensor::new(&shape, data).unwrap()
        })
        .collect();
    let parameter_tangents: Vec<Option<&DoubleTensor>> = tangents[1..].iter().map(Some).collect();
    let hvp = model
        .hvp(&y, Some(&tangents[0]), &parameter_tangents, b)
        .unwrap();
    let gradient_at = |k: usize, step: f64| {
        let moved: Vec<DoubleTensor> = inputs
            .iter()
            .zip(&directions[k])
            .map(|(t, d)| {
                let values = t
                    .as_slice()
                    .iter()
                    .zip(d)
                    .map(|(x, d)| x + step * d)
                    .collect();
                DoubleTensor::new(t.shape(), values).unwrap()
            })
            .collect();
        let [x, a, bb, c, d, e, f] = <[DoubleTensor; 7]>::try_from(moved).unwrap();
        LinearGaussianStateSpace::new(a, bb, c, d, e, f)
            .d_log_prob(&x)
            .unwrap()
    };
    let h = 1e-5;
    for k in 0..b {
        let (plus, minus) = (gradient_at(k, h), gradient_at(k, -h));
        for (u, ((actual, p), m)) in gradient_fields(&hvp)
            .into_iter()
            .zip(gradient_fields(&plus))
            .zip(gradient_fields(&minus))
            .enumerate()
        {
            let len = p.len();
            assert_eq!(actual.shape()[0], b);
            assert_eq!(&actual.shape()[1..], p.shape(), "input {u}");
            for i in 0..len {
                let numeric = (p.as_slice()[i] - m.as_slice()[i]) / (2.0 * h);
                let exact = actual.as_slice()[k * len + i];
                assert_close_1p(
                    exact,
                    numeric,
                    1e-6,
                    &format!("direction {k}, input {u}, element {i}: {exact} vs {numeric}"),
                );
            }
        }
    }

    // a covariance tangent's symmetric part is used, and the covariance products are symmetric
    let lopsided = DoubleTensor::new(&[1, 2, 2], vec![0.4, 1.0, 0.0, -0.3]).unwrap();
    let balanced = DoubleTensor::new(&[1, 2, 2], vec![0.4, 0.5, 0.5, -0.3]).unwrap();
    let along = |r: &DoubleTensor| {
        let mut t: Vec<Option<&DoubleTensor>> = vec![None; 6];
        t[5] = Some(r);
        model.hvp(&y, None, &t, 1).unwrap()
    };
    let (a, s) = (along(&lopsided), along(&balanced));
    assert_eq!(a, s);
    let r = a.observation_covariance.as_slice();
    assert_close_1p(r[1], r[2], 1e-12, "value");
    // no tangents at all give zeros
    let zero = model.hvp(&y, None, &[None; 6], 3).unwrap();
    assert_eq!(zero.transition_matrix.shape(), &[3, 3, 3]);
    assert!(
        gradient_fields(&zero)
            .iter()
            .all(|t| t.as_slice().iter().all(|&v| v == 0.0))
    );
}

#[test]
fn bad_kalman_tangents_are_errors() {
    let model = model();
    let y = observations();
    assert!(model.hvp(&y, None, &[None; 5], 1).is_err());
    let wrong = DoubleTensor::zeros(&[2, 3]);
    let mut t: Vec<Option<&DoubleTensor>> = vec![None; 6];
    t[0] = Some(&wrong);
    assert!(model.hvp(&y, None, &t, 1).is_err());
    assert!(model.hvp(&y, Some(&wrong), &[None; 6], 1).is_err());
    // an impossible innovation covariance
    let degenerate = LinearGaussianStateSpace::new(
        v(&[0.0]),
        m(1, 1, &[0.0]),
        m(1, 1, &[1.0]),
        m(1, 1, &[0.0]),
        m(1, 1, &[1.0]),
        m(1, 1, &[0.0]),
    );
    assert!(
        degenerate
            .hvp(&m(1, 1, &[0.5]), None, &[None; 6], 1)
            .is_err()
    );
}

fn parameters_of(model: &LinearGaussianStateSpace<'_>) -> Vec<DoubleTensor> {
    [
        &model.initial_mean,
        &model.initial_covariance,
        &model.transition_matrix,
        &model.transition_covariance,
        &model.observation_matrix,
        &model.observation_covariance,
    ]
    .into_iter()
    .map(|p| p.clone().into_owned())
    .collect()
}

#[test]
fn the_filter_stays_finite_and_positive_semidefinite_over_long_series() {
    // a near-deterministic rotation observed almost exactly: the update nearly zeroes the covariance each step,
    // which loses positive definiteness without the Joseph form
    let theta: f64 = 0.1;
    let model = LinearGaussianStateSpace::new(
        v(&[1.0, 0.0]),
        m(2, 2, &[1.0, 0.0, 0.0, 1.0]),
        m(2, 2, &[theta.cos(), -theta.sin(), theta.sin(), theta.cos()]),
        m(2, 2, &[1e-12, 0.0, 0.0, 1e-12]),
        m(1, 2, &[1.0, 0.0]),
        m(1, 1, &[1e-10]),
    );
    let mut rng = KeanuRng::seed_from_u64(7);
    let y = model.sample(&[1000, 1], &mut rng).unwrap();
    let filtered = model.filter(&y).unwrap();
    assert!(
        filtered.log_likelihood.is_finite(),
        "{}",
        filtered.log_likelihood
    );
    for p in filtered.covariances.as_slice().chunks(4) {
        // a 2 × 2 symmetric matrix is positive semidefinite iff its trace and determinant are non-negative
        assert!(p.iter().all(|x| x.is_finite()));
        assert!((p[1] - p[2]).abs() <= 1e-12 * (p[0] + p[3]).abs().max(1e-300));
        let scale = p[0].abs().max(p[3].abs());
        assert!(p[0] >= -1e-9 * scale && p[3] >= -1e-9 * scale, "{p:?}");
        assert!(p[0] * p[3] - p[1] * p[2] >= -1e-9 * scale * scale, "{p:?}");
    }
    let gradient = model.d_log_prob(&y).unwrap();
    assert!(
        gradient
            .observation_matrix
            .as_slice()
            .iter()
            .all(|x| x.is_finite())
    );
}

#[test]
fn simulated_series_reproduce_the_stationary_moments() {
    // x_{t+1} = 0.8 x_t + w, w ~ N(0, 0.36): stationary variance 0.36 / (1 - 0.64) = 1; y = 2 x + v, v ~ N(0, 1)
    let model = LinearGaussianStateSpace::new(
        v(&[0.0]),
        m(1, 1, &[1.0]),
        m(1, 1, &[0.8]),
        m(1, 1, &[0.36]),
        m(1, 1, &[2.0]),
        m(1, 1, &[1.0]),
    );
    let mut rng = KeanuRng::seed_from_u64(11);
    let y = model.sample(&[50_000, 1], &mut rng).unwrap();
    let y = y.as_slice();
    let n = y.len() as f64;
    let mean = y.iter().sum::<f64>() / n;
    let variance = y.iter().map(|y| (y - mean).powi(2)).sum::<f64>() / n;
    let lag1 = y
        .windows(2)
        .map(|w| (w[0] - mean) * (w[1] - mean))
        .sum::<f64>()
        / n;
    // Var(y) = 4 + 1 = 5 and Cov(y_t, y_{t+1}) = 4 × 0.8 = 3.2. The series is autocorrelated (φ = 0.8), so the
    // effective sample size is about n (1 - 0.8) / (1 + 0.8) ≈ 5500; standard errors are ≈ 0.03 for the mean
    // and ≈ 0.1 for the variance
    assert!(mean.abs() < 0.12, "{mean}");
    assert!((variance - 5.0).abs() < 0.35, "{variance}");
    assert!((lag1 - 3.2).abs() < 0.3, "{lag1}");
}

#[test]
fn invalid_state_space_models_are_errors() {
    let y = observations();
    let p = parameters_of(&model());
    let with = |index: usize, t: DoubleTensor| {
        let mut p = p.clone();
        p[index] = t;
        let [a, b, c, d, e, f] = <[DoubleTensor; 6]>::try_from(p).unwrap();
        LinearGaussianStateSpace::new(a, b, c, d, e, f)
    };
    let is_error = |model: LinearGaussianStateSpace<'_>, y: &DoubleTensor| {
        model.log_prob(y).is_err() && model.d_log_prob(y).is_err() && model.filter(y).is_err()
    };
    assert!(is_error(with(0, v(&[0.0, 0.0])), &y));
    assert!(is_error(with(1, m(3, 2, &[0.0; 6])), &y));
    assert!(is_error(with(2, m(2, 3, &[0.0; 6])), &y));
    assert!(is_error(with(3, v(&[1.0; 3])), &y));
    assert!(is_error(with(4, m(2, 2, &[1.0; 4])), &y));
    assert!(is_error(with(5, m(3, 3, &[1.0; 9])), &y));
    assert!(is_error(model(), &v(&[1.0, 2.0])));
    assert!(is_error(model(), &m(5, 3, &[0.0; 15])));
    assert!(
        model()
            .sample(&[5, 3], &mut KeanuRng::seed_from_u64(1))
            .is_err()
    );
    assert!(
        model()
            .sample(&[5], &mut KeanuRng::seed_from_u64(1))
            .is_err()
    );
    // an innovation covariance that is not positive definite makes the observations impossible
    let singular = with(5, m(2, 2, &[-10.0, 0.0, 0.0, -10.0]));
    assert_eq!(singular.log_prob(&y).unwrap(), f64::NEG_INFINITY);
    assert!(singular.d_log_prob(&y).is_err());
    // no observations: probability 1
    assert_eq!(model().log_prob(&m(0, 2, &[])).unwrap(), 0.0);
}
