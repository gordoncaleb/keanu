//! Gaussian processes: kernels are symmetric and positive semi-definite, the marginal likelihood matches a hand
//! computation, the predictive interpolates noise-free training data (AE4), hyperparameter gradients match finite
//! differences, and malformed shapes are build errors rather than panics.

use keanu_distributions::KeanuRng;
use keanu_graph::gp::{GaussianProcess, Kernel, gp_marginal, gp_predict};
use keanu_graph::{
    DoubleVar, Graph, GraphBuilder, GraphError, NodeId, State, log_likelihood,
    log_likelihood_gradient, reverse_partials,
};
use keanu_tensor::DoubleTensor;

mod common;
use common::t;

/// Computes the parameters of an observed node (which keeps its value), so its log density can be evaluated.
fn evaluate_parents(graph: &Graph, state: &mut State, observed: NodeId) {
    let parents = graph[observed].parents().to_vec();
    state
        .eval(graph, &parents, &mut KeanuRng::seed_from_u64(1))
        .unwrap();
}

/// Every kind of kernel with fixed hyperparameters, for inputs with `d` features.
fn kernels<'b>(b: &'b GraphBuilder, d: usize) -> Vec<(&'static str, Kernel<'b>)> {
    let ard: Vec<f64> = (0..d).map(|i| 0.7 + 0.3 * i as f64).collect();
    let ard = DoubleTensor::vector(ard);
    vec![
        ("exp_quad", Kernel::exp_quad(b, 1.3, 0.8)),
        ("exp_quad_ard", Kernel::exp_quad(b, 1.3, ard.clone())),
        ("matern12", Kernel::matern12(b, 0.9, 1.1)),
        ("matern32", Kernel::matern32(b, 1.1, ard.clone())),
        ("matern52", Kernel::matern52(b, 1.2, 0.6)),
        (
            "rational_quadratic",
            Kernel::rational_quadratic(b, 1.0, 0.9, 2.5),
        ),
        ("periodic", Kernel::periodic(b, 0.8, 1.2, 1.7)),
        ("linear", Kernel::linear(b, 0.5, 1.5, 0.3)),
        (
            "sum",
            Kernel::exp_quad(b, 1.0, 2.0) + Kernel::periodic(b, 0.5, 1.0, 0.9),
        ),
        (
            "product",
            Kernel::linear(b, 0.2, 1.0, -0.5) * Kernel::matern52(b, 1.0, 1.5),
        ),
    ]
}

#[test]
fn every_kernel_is_symmetric_and_positive_semi_definite_on_random_inputs() {
    let mut rng = KeanuRng::seed_from_u64(7);
    for (n, d) in [(6, 1), (8, 2), (7, 3)] {
        let inputs = &rng.gaussian(&[n, d]) * 2.0;
        // 1-D inputs may also be given as a vector
        let inputs_1d = (d == 1).then(|| inputs.reshape(&[n as isize]).unwrap());
        let b = GraphBuilder::new();
        let mut outputs: Vec<(String, NodeId, NodeId)> = Vec::new();
        for (name, kernel) in kernels(&b, d) {
            let k = kernel.covariance(inputs.clone(), inputs.clone());
            outputs.push((name.to_string(), k.id(), k.eigvalsh().id()));
            if let Some(x) = &inputs_1d {
                let k = kernel.covariance(x.clone(), x.clone());
                outputs.push((format!("{name} (vector inputs)"), k.id(), k.eigvalsh().id()));
            }
        }
        let (graph, mut state) = b.build().expect("kernels build");
        let ids: Vec<NodeId> = outputs.iter().flat_map(|o| [o.1, o.2]).collect();
        state.lazy_eval(&graph, &ids, &mut rng).unwrap();
        for (name, k, eigenvalues) in &outputs {
            let k = state.f64(*k).unwrap();
            assert_eq!(k.shape(), &[n, n], "{name}");
            for i in 0..n {
                for j in 0..n {
                    let (a, c) = (k.as_slice()[i * n + j], k.as_slice()[j * n + i]);
                    assert!(
                        (a - c).abs() <= 1e-12,
                        "{name} is not symmetric: {a} != {c}"
                    );
                }
            }
            for &lambda in state.f64(*eigenvalues).unwrap().as_slice() {
                assert!(
                    lambda >= -1e-10,
                    "{name} has eigenvalue {lambda} (n {n}, d {d})"
                );
            }
        }
    }
}

type Formula<'a> = Box<dyn Fn(f64, f64) -> f64 + 'a>;

#[test]
fn cross_covariances_match_the_closed_forms() {
    let x1 = t(&[2], &[0.0, 1.0]);
    let x2 = t(&[3], &[0.5, -1.0, 2.0]);
    let (amp, ls) = (1.5_f64, 0.7_f64);
    let b = GraphBuilder::new();
    let cases: Vec<(Kernel<'_>, Formula)> = vec![
        (
            Kernel::exp_quad(&b, amp, ls),
            Box::new(move |a, c| amp * amp * (-0.5 * ((a - c) / ls).powi(2)).exp()),
        ),
        (
            Kernel::matern12(&b, amp, ls),
            Box::new(move |a, c| amp * amp * (-((a - c) / ls).abs()).exp()),
        ),
        (
            Kernel::matern32(&b, amp, ls),
            Box::new(move |a, c| {
                let r = 3f64.sqrt() * ((a - c) / ls).abs();
                amp * amp * (1.0 + r) * (-r).exp()
            }),
        ),
        (
            Kernel::matern52(&b, amp, ls),
            Box::new(move |a, c| {
                let r = 5f64.sqrt() * ((a - c) / ls).abs();
                amp * amp * (1.0 + r + r * r / 3.0) * (-r).exp()
            }),
        ),
        (
            Kernel::rational_quadratic(&b, amp, ls, 3.0),
            Box::new(move |a, c| amp * amp * (1.0 + ((a - c) / ls).powi(2) / 6.0).powf(-3.0)),
        ),
        (
            Kernel::periodic(&b, amp, ls, 1.3),
            Box::new(move |a, c| {
                let s = (std::f64::consts::PI * (a - c) / 1.3).sin() / ls;
                amp * amp * (-2.0 * s * s).exp()
            }),
        ),
        (
            Kernel::linear(&b, 0.4, amp, 0.25),
            Box::new(move |a, c| 0.16 + amp * amp * (a - 0.25) * (c - 0.25)),
        ),
    ];
    let ids: Vec<NodeId> = cases
        .iter()
        .map(|(k, _)| k.covariance(x1.clone(), x2.clone()).id())
        .collect();
    let formulas: Vec<_> = cases.into_iter().map(|(_, f)| f).collect();
    let (graph, mut state) = b.build().unwrap();
    state
        .lazy_eval(&graph, &ids, &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    for (k, (id, f)) in ids.iter().zip(&formulas).enumerate() {
        let value = state.f64(*id).unwrap();
        assert_eq!(value.shape(), &[2, 3]);
        for i in 0..2 {
            for j in 0..3 {
                let expected = f(x1.as_slice()[i], x2.as_slice()[j]);
                let actual = value.as_slice()[i * 3 + j];
                assert!(
                    (actual - expected).abs() < 1e-12,
                    "kernel {k} [{i}, {j}]: {actual} != {expected}"
                );
            }
        }
    }
}

/// ln N(y; m, Σ) for a 3 × 3 Σ, by the cofactor inverse and determinant (no factorisation).
fn hand_mvn_log_density(y: [f64; 3], m: f64, s: [[f64; 3]; 3]) -> f64 {
    let det = s[0][0] * (s[1][1] * s[2][2] - s[1][2] * s[2][1])
        - s[0][1] * (s[1][0] * s[2][2] - s[1][2] * s[2][0])
        + s[0][2] * (s[1][0] * s[2][1] - s[1][1] * s[2][0]);
    let mut inverse = [[0.0; 3]; 3];
    for (i, row) in inverse.iter_mut().enumerate() {
        for (j, entry) in row.iter_mut().enumerate() {
            // cofactor of s[j][i]
            let (r0, r1) = ((j + 1) % 3, (j + 2) % 3);
            let (c0, c1) = ((i + 1) % 3, (i + 2) % 3);
            *entry = (s[r0][c0] * s[r1][c1] - s[r0][c1] * s[r1][c0]) / det;
        }
    }
    let r: Vec<f64> = y.iter().map(|v| v - m).collect();
    let mut quadratic = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            quadratic += r[i] * inverse[i][j] * r[j];
        }
    }
    -0.5 * (quadratic + det.ln() + 3.0 * (2.0 * std::f64::consts::PI).ln())
}

#[test]
fn marginal_log_likelihood_matches_a_hand_computed_value_on_three_points() {
    let x = [0.0, 1.0, 3.0];
    let y = [0.5, -0.2, 1.1];
    let (amp, ls, noise, mean) = (1.2, 1.5, 0.3, 0.1);
    let b = GraphBuilder::new();
    let kernel = Kernel::exp_quad(&b, amp, ls);
    let observed = gp_marginal(
        DoubleTensor::vector(x.to_vec()),
        DoubleTensor::vector(y.to_vec()),
        mean,
        &kernel,
        noise,
    );
    assert_eq!(observed.shape(), &[3]);
    let observed = observed.id();
    let (graph, mut state) = b.build().unwrap();
    evaluate_parents(&graph, &mut state, observed);
    let mut s = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            s[i][j] = amp * amp * (-0.5 * ((x[i] - x[j]) / ls).powi(2)).exp();
        }
        s[i][i] += noise * noise + 1e-6;
    }
    let expected = hand_mvn_log_density(y, mean, s);
    let actual = log_likelihood(&graph, &state).unwrap();
    assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
}

#[test]
fn jitter_is_configurable() {
    let x = [0.0, 1.0, 3.0];
    let y = [0.5, -0.2, 1.1];
    let b = GraphBuilder::new();
    let gp = GaussianProcess::new(Kernel::matern32(&b, 1.0, 1.0)).with_jitter(0.01);
    assert_eq!(gp.jitter(), 0.01);
    let observed = gp
        .marginal(
            DoubleTensor::vector(x.to_vec()),
            DoubleTensor::vector(y.to_vec()),
        )
        .id();
    let (graph, mut state) = b.build().unwrap();
    evaluate_parents(&graph, &mut state, observed);
    let mut s = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let r = 3f64.sqrt() * (x[i] - x[j]).abs();
            s[i][j] = (1.0 + r) * (-r).exp();
        }
        s[i][i] += 0.01;
    }
    let expected = hand_mvn_log_density(y, 0.0, s);
    let actual = log_likelihood(&graph, &state).unwrap();
    assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
}

/// AE4: with zero noise the predictive at the training inputs returns the training targets with zero variance
/// (up to the jitter).
#[test]
fn predictive_at_noise_free_training_inputs_interpolates_the_targets() {
    let x = t(&[4, 2], &[0.0, 0.0, 1.0, 0.5, 2.5, -1.0, 4.0, 2.0]);
    let y = [0.3, -1.2, 0.8, 2.0];
    let b = GraphBuilder::new();
    let kernel = Kernel::exp_quad(&b, 1.5, 1.0) + Kernel::linear(&b, 0.1, 0.2, 0.0);
    let prediction = gp_predict(
        x.clone(),
        DoubleTensor::vector(y.to_vec()),
        x.clone(),
        0.0,
        &kernel,
        0.0,
    );
    let ids = [
        prediction.mean.id(),
        prediction.covariance.id(),
        prediction.variance().id(),
    ];
    let (graph, mut state) = b.build().unwrap();
    state
        .lazy_eval(&graph, &ids, &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    let mean = state.f64(ids[0]).unwrap();
    assert_eq!(mean.shape(), &[4]);
    for (m, target) in mean.as_slice().iter().zip(y) {
        assert!((m - target).abs() < 1e-4, "mean {m} != target {target}");
    }
    let covariance = state.f64(ids[1]).unwrap();
    assert_eq!(covariance.shape(), &[4, 4]);
    for c in covariance.as_slice() {
        assert!(c.abs() < 1e-5, "covariance entry {c} is not ~0");
    }
    for v in state.f64(ids[2]).unwrap().as_slice() {
        assert!((-1e-9..1e-5).contains(v), "variance {v} is not ~0");
    }
}

#[test]
fn predictive_matches_the_textbook_formula_at_new_inputs() {
    let x_train = [0.0, 0.8, 2.0, 3.1];
    let y_train = [1.0, 0.2, -0.5, 0.4];
    let x_new = [0.4, 2.5, 5.0];
    let (amp, ls, noise, mean) = (1.1_f64, 0.9_f64, 0.2_f64, 0.3);
    let k = |a: f64, c: f64| amp * amp * (-0.5 * ((a - c) / ls).powi(2)).exp();
    let matrix = |r: &[f64], c: &[f64]| {
        let data: Vec<f64> = r
            .iter()
            .flat_map(|&a| c.iter().map(move |&b| k(a, b)))
            .collect();
        t(&[r.len(), c.len()], &data)
    };
    let jitter = keanu_graph::gp::DEFAULT_JITTER;
    let k_train = matrix(&x_train, &x_train)
        .add(&(DoubleTensor::eye(4).unwrap() * (noise * noise + jitter)))
        .unwrap();
    let k_inverse = k_train.inverse().unwrap();
    let k_star = matrix(&x_new, &x_train);
    let residual = t(&[4, 1], &y_train.map(|v| v - mean));
    let expected_mean = &k_star
        .matmul(&k_inverse)
        .unwrap()
        .matmul(&residual)
        .unwrap()
        + mean;
    let expected_cov = matrix(&x_new, &x_new)
        .sub(
            &k_star
                .matmul(&k_inverse)
                .unwrap()
                .matmul_transposed(&k_star, false, true)
                .unwrap(),
        )
        .unwrap();

    let b = GraphBuilder::new();
    let gp = GaussianProcess::new(Kernel::exp_quad(&b, amp, ls))
        .with_mean(mean)
        .with_noise(noise);
    let prediction = gp.predict(
        DoubleTensor::vector(x_train.to_vec()),
        DoubleTensor::vector(y_train.to_vec()),
        DoubleTensor::vector(x_new.to_vec()),
    );
    let ids = [prediction.mean.id(), prediction.covariance.id()];
    let (graph, mut state) = b.build().unwrap();
    state
        .lazy_eval(&graph, &ids, &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    let close = |actual: &DoubleTensor, expected: &DoubleTensor| {
        for (a, e) in actual.as_slice().iter().zip(expected.as_slice()) {
            assert!((a - e).abs() < 1e-9, "{a} != {e}");
        }
    };
    close(state.f64(ids[0]).unwrap(), &expected_mean);
    assert_eq!(state.f64(ids[1]).unwrap().shape(), &[3, 3]);
    close(state.f64(ids[1]).unwrap(), &expected_cov);
}

#[test]
fn a_mean_function_shifts_the_marginal_and_the_predictive() {
    // y lies exactly on the mean function, so the predictive mean is the mean function everywhere
    let x_train = [0.0, 1.0, 2.0];
    let y_train: Vec<f64> = x_train.iter().map(|x| 2.0 * x - 1.0).collect();
    let b = GraphBuilder::new();
    let gp = GaussianProcess::new(Kernel::matern52(&b, 1.0, 1.0))
        .with_noise(0.1)
        .with_mean_fn(|x: DoubleVar<'_>| x * 2.0 - 1.0);
    let prediction = gp.predict(
        DoubleTensor::vector(x_train.to_vec()),
        DoubleTensor::vector(y_train.clone()),
        DoubleTensor::vector(vec![-3.0, 7.5]),
    );
    let id = prediction.mean.id();
    let (graph, mut state) = b.build().unwrap();
    state
        .lazy_eval(&graph, &[id], &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    let mean = state.f64(id).unwrap().as_slice().to_vec();
    assert!(
        (mean[0] + 7.0).abs() < 1e-12 && (mean[1] - 14.0).abs() < 1e-12,
        "{mean:?}"
    );
}

/// Builds a GP marginal whose hyperparameters are latents set to `values`, and checks the log likelihood gradient
/// with respect to each against central finite differences.
fn check_hyperparameter_gradients(
    name: &str,
    values: &[f64],
    kernel: impl for<'b> Fn(&'b GraphBuilder, &[DoubleVar<'b>]) -> (Kernel<'b>, DoubleVar<'b>),
) {
    let x = t(
        &[5, 2],
        &[0.0, 0.1, 0.7, -0.4, 1.5, 0.9, 2.2, 0.0, 3.0, 1.2],
    );
    let y = DoubleTensor::vector(vec![0.4, -0.3, 1.0, 0.6, -0.8]);
    let b = GraphBuilder::new();
    let latents: Vec<DoubleVar<'_>> = values
        .iter()
        .map(|&v| {
            let latent = b.uniform(-100.0, 100.0);
            b.set_value(latent, DoubleTensor::scalar(v));
            latent
        })
        .collect();
    let (kernel, noise) = kernel(&b, &latents);
    let mean = latents.last().copied().unwrap();
    let observed = gp_marginal(x, y, mean, &kernel, noise).id();
    let ids: Vec<NodeId> = latents.iter().map(|l| l.id()).collect();
    let (graph, mut state) = b.build().unwrap();
    evaluate_parents(&graph, &mut state, observed);
    let gradient = log_likelihood_gradient(&graph, &state).unwrap();
    let h = 1e-6;
    for (k, (&id, &v)) in ids.iter().zip(values).enumerate() {
        let at = |delta: f64| {
            let mut s = state.clone();
            s.set_and_cascade(&graph, id, DoubleTensor::scalar(v + delta))
                .unwrap();
            log_likelihood(&graph, &s).unwrap()
        };
        let numeric = (at(h) - at(-h)) / (2.0 * h);
        let analytic = gradient.get(id).unwrap().to_scalar().unwrap();
        assert!(
            (analytic - numeric).abs() <= 1e-5 * numeric.abs().max(1.0),
            "{name}: d/d hyperparameter {k}: {analytic} != finite difference {numeric}"
        );
    }
}

#[test]
fn hyperparameter_gradients_match_finite_differences() {
    // the last latent is always the (constant) mean and the noise is the second to last
    check_hyperparameter_gradients("exp_quad", &[1.3, 0.8, 0.2, 0.1], |b, h| {
        (Kernel::exp_quad(b, h[0], h[1]), h[2])
    });
    check_hyperparameter_gradients("matern12", &[0.9, 1.1, 0.3, -0.2], |b, h| {
        (Kernel::matern12(b, h[0], h[1]), h[2])
    });
    check_hyperparameter_gradients("matern32", &[1.1, 0.7, 0.25, 0.0], |b, h| {
        (Kernel::matern32(b, h[0], h[1]), h[2])
    });
    check_hyperparameter_gradients("matern52", &[1.2, 0.6, 0.2, 0.3], |b, h| {
        (Kernel::matern52(b, h[0], h[1]), h[2])
    });
    check_hyperparameter_gradients("rational_quadratic", &[1.0, 0.9, 2.5, 0.2, 0.0], |b, h| {
        (Kernel::rational_quadratic(b, h[0], h[1], h[2]), h[3])
    });
    check_hyperparameter_gradients("periodic", &[0.8, 1.2, 1.7, 0.3, 0.1], |b, h| {
        (Kernel::periodic(b, h[0], h[1], h[2]), h[3])
    });
    check_hyperparameter_gradients("linear", &[0.5, 0.4, 0.3, 0.5, 0.0], |b, h| {
        (Kernel::linear(b, h[0], h[1], h[2]), h[3])
    });
    check_hyperparameter_gradients(
        "sum and product",
        &[1.0, 0.8, 0.5, 1.5, 0.3, 0.2],
        |b, h| {
            (
                Kernel::exp_quad(b, h[0], h[1])
                    + Kernel::matern32(b, h[2], h[3]) * Kernel::exp_quad(b, 1.0, 2.0),
                h[4],
            )
        },
    );
    // ARD length scales: a [2] latent is checked through its sum (both elements scaled together)
    check_hyperparameter_gradients("exp_quad ard", &[1.3, 0.8, 0.2, 0.1], |b, h| {
        let scales = b.constant(DoubleTensor::vector(vec![1.0, 1.5])) * h[1];
        (Kernel::exp_quad(b, h[0], scales), h[2])
    });
}

#[test]
fn predictive_mean_gradients_match_finite_differences() {
    let b = GraphBuilder::new();
    let ls = b.uniform(0.0, 10.0);
    b.set_value(ls, DoubleTensor::scalar(0.9));
    let noise = b.uniform(0.0, 10.0);
    b.set_value(noise, DoubleTensor::scalar(0.3));
    let prediction = gp_predict(
        DoubleTensor::vector(vec![0.0, 1.0, 2.0, 3.5]),
        DoubleTensor::vector(vec![0.2, 0.9, -0.4, 0.1]),
        DoubleTensor::vector(vec![0.5, 2.7]),
        0.0,
        &Kernel::matern52(&b, 1.0, ls),
        noise,
    );
    let summary = (prediction.mean.sum() + prediction.covariance.sum()).id();
    let (ls, noise) = (ls.id(), noise.id());
    let (graph, mut state) = b.build().unwrap();
    state
        .lazy_eval(&graph, &[summary], &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    let partials = reverse_partials(&graph, &state, summary, &[ls, noise]).unwrap();
    for (id, v) in [(ls, 0.9), (noise, 0.3)] {
        let at = |delta: f64| {
            let mut s = state.clone();
            s.set_and_cascade(&graph, id, DoubleTensor::scalar(v + delta))
                .unwrap();
            s.f64(summary).unwrap().to_scalar().unwrap()
        };
        let numeric = (at(1e-6) - at(-1e-6)) / 2e-6;
        let analytic = partials.get(id).unwrap().to_scalar().unwrap();
        assert!(
            (analytic - numeric).abs() <= 1e-6 * numeric.abs().max(1.0),
            "{analytic} != finite difference {numeric}"
        );
    }
}

#[test]
fn kernel_gradients_are_finite_at_coincident_inputs() {
    // the Matérn kernels use |x - x'|, whose derivative is undefined at 0; the diagonal must not produce NaN
    let b = GraphBuilder::new();
    let ls = b.uniform(0.0, 10.0);
    b.set_value(ls, DoubleTensor::scalar(0.8));
    let x = DoubleTensor::vector(vec![0.0, 0.0, 1.0]);
    let total = (Kernel::matern12(&b, 1.0, ls).covariance(x.clone(), x.clone())
        + Kernel::matern32(&b, 1.0, ls).covariance(x.clone(), x.clone())
        + Kernel::matern52(&b, 1.0, ls).covariance(x.clone(), x))
    .sum()
    .id();
    let ls = ls.id();
    let (graph, mut state) = b.build().unwrap();
    state
        .lazy_eval(&graph, &[total], &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    let partial = reverse_partials(&graph, &state, total, &[ls]).unwrap();
    assert!(partial.get(ls).unwrap().to_scalar().unwrap().is_finite());
}

fn build_error(f: impl FnOnce(&GraphBuilder)) -> GraphError {
    let b = GraphBuilder::new();
    f(&b);
    b.build().expect_err("the build fails")
}

type Misuse<'a> = Box<dyn Fn(&GraphBuilder) + 'a>;

#[test]
fn shape_errors_are_errors_not_panics() {
    let v = |data: &[f64]| DoubleTensor::vector(data.to_vec());
    let m = |shape: &[usize]| DoubleTensor::zeros(shape);
    let cases: Vec<(&str, Misuse)> = vec![
        (
            "feature counts differ",
            Box::new(|b| {
                Kernel::exp_quad(b, 1.0, 1.0).covariance(m(&[3, 2]), m(&[4, 3]));
            }),
        ),
        (
            "rank 3 inputs",
            Box::new(|b| {
                Kernel::matern32(b, 1.0, 1.0).covariance(m(&[2, 2, 2]), m(&[2, 2, 2]));
            }),
        ),
        (
            "scalar inputs",
            Box::new(|b| {
                Kernel::periodic(b, 1.0, 1.0, 1.0).covariance(1.0, 2.0);
            }),
        ),
        (
            "length scale of the wrong size",
            Box::new(|b| {
                Kernel::exp_quad(b, 1.0, v(&[1.0, 2.0, 3.0])).covariance(m(&[3, 2]), m(&[3, 2]));
            }),
        ),
        (
            "matrix amplitude",
            Box::new(|b| {
                Kernel::matern12(b, m(&[2, 2]), 1.0).covariance(v(&[1.0]), v(&[2.0]));
            }),
        ),
        (
            "vector alpha",
            Box::new(|b| {
                Kernel::rational_quadratic(b, 1.0, 1.0, v(&[1.0, 2.0]))
                    .covariance(v(&[1.0]), v(&[2.0]));
            }),
        ),
        (
            "linear shift of the wrong size",
            Box::new(|b| {
                Kernel::linear(b, 1.0, 1.0, v(&[1.0, 2.0])).covariance(m(&[2, 3]), m(&[2, 3]));
            }),
        ),
        (
            "y of the wrong length",
            Box::new(|b| {
                gp_marginal(
                    v(&[0.0, 1.0, 2.0]),
                    v(&[1.0, 2.0]),
                    0.0,
                    &Kernel::exp_quad(b, 1.0, 1.0),
                    0.1,
                );
            }),
        ),
        (
            "matrix y",
            Box::new(|b| {
                gp_marginal(
                    v(&[0.0, 1.0]),
                    m(&[2, 1]),
                    0.0,
                    &Kernel::exp_quad(b, 1.0, 1.0),
                    0.1,
                );
            }),
        ),
        (
            "vector noise",
            Box::new(|b| {
                gp_marginal(
                    v(&[0.0, 1.0]),
                    v(&[1.0, 2.0]),
                    0.0,
                    &Kernel::exp_quad(b, 1.0, 1.0),
                    v(&[0.1, 0.1]),
                );
            }),
        ),
        (
            "mean of the wrong length",
            Box::new(|b| {
                gp_marginal(
                    v(&[0.0, 1.0]),
                    v(&[1.0, 2.0]),
                    v(&[0.0, 0.0, 0.0]),
                    &Kernel::exp_quad(b, 1.0, 1.0),
                    0.1,
                );
            }),
        ),
        (
            "mean function of the wrong shape",
            Box::new(|b| {
                GaussianProcess::new(Kernel::exp_quad(b, 1.0, 1.0))
                    .with_mean_fn(|x: DoubleVar<'_>| x.sum().reshape(&[1, 1]))
                    .marginal(v(&[0.0, 1.0]), v(&[1.0, 2.0]));
            }),
        ),
        (
            "empty training set",
            Box::new(|b| {
                gp_marginal(v(&[]), v(&[]), 0.0, &Kernel::exp_quad(b, 1.0, 1.0), 0.1);
            }),
        ),
        (
            "new inputs with other features",
            Box::new(|b| {
                gp_predict(
                    m(&[2, 2]),
                    v(&[1.0, 2.0]),
                    m(&[3, 1]),
                    0.0,
                    &Kernel::exp_quad(b, 1.0, 1.0),
                    0.1,
                );
            }),
        ),
        (
            "training targets of the wrong length",
            Box::new(|b| {
                gp_predict(
                    v(&[0.0, 1.0]),
                    v(&[1.0]),
                    v(&[0.5]),
                    0.0,
                    &Kernel::exp_quad(b, 1.0, 1.0),
                    0.1,
                );
            }),
        ),
        (
            "negative jitter",
            Box::new(|b| {
                GaussianProcess::new(Kernel::exp_quad(b, 1.0, 1.0))
                    .with_jitter(-1.0)
                    .marginal(v(&[0.0, 1.0]), v(&[1.0, 2.0]));
            }),
        ),
        (
            "NaN jitter",
            Box::new(|b| {
                GaussianProcess::new(Kernel::exp_quad(b, 1.0, 1.0))
                    .with_jitter(f64::NAN)
                    .predict(v(&[0.0, 1.0]), v(&[1.0, 2.0]), v(&[0.5]));
            }),
        ),
    ];
    for (name, case) in cases {
        let error = build_error(|b| case(b));
        assert!(
            matches!(
                error,
                GraphError::InvalidNode { .. } | GraphError::Tensor(_)
            ),
            "{name}: unexpected error {error:?}"
        );
    }
}

#[test]
fn kernels_from_another_builder_fail_the_build() {
    let (b1, b2) = (GraphBuilder::new(), GraphBuilder::new());
    let kernel = Kernel::exp_quad(&b1, 1.0, 1.0) + Kernel::exp_quad(&b2, 1.0, 1.0);
    kernel.covariance(
        DoubleTensor::vector(vec![0.0, 1.0]),
        DoubleTensor::vector(vec![0.5]),
    );
    assert!(matches!(b1.build(), Err(GraphError::WrongBuilder(_))));
}

#[test]
fn empty_inputs_give_empty_covariances() {
    let b = GraphBuilder::new();
    let k = Kernel::matern32(&b, 1.0, 1.0)
        .covariance(DoubleTensor::zeros(&[0, 2]), DoubleTensor::zeros(&[3, 2]));
    assert_eq!(k.shape(), &[0, 3]);
    let id = k.id();
    let (graph, mut state) = b.build().unwrap();
    state
        .lazy_eval(&graph, &[id], &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    assert_eq!(state.f64(id).unwrap().shape(), &[0, 3]);
}
