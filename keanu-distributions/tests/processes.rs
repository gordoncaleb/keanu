//! The Gaussian random walk, AR(p) and GARCH(1,1): log densities against the recursions written out, batches and
//! broadcasting, gradients and Hessian-vector products against finite differences, simulated statistics, and
//! invalid parameters and shapes.

use keanu_distributions::{Autoregressive, Garch11, Gaussian, GaussianRandomWalk, KeanuRng};
use keanu_tensor::DoubleTensor;

mod common;
use common::{assert_close, s, t, v};

fn normal(x: f64, mu: f64, sigma: f64) -> f64 {
    Gaussian::new(s(mu), s(sigma))
        .log_prob(&s(x))
        .unwrap()
        .sum()
}

#[test]
fn the_random_walk_density_sums_its_increments() {
    let x = [0.3, 0.9, 0.4, 1.6, 2.0];
    let (mu, sigma, m0, s0) = (0.2, 0.7, -0.1, 1.5);
    let mut expected = normal(x[0], m0, s0);
    for w in x.windows(2) {
        expected += normal(w[1] - w[0], mu, sigma);
    }
    let walk = GaussianRandomWalk::new(s(mu), s(sigma), s(m0), s(s0));
    assert_close(walk.log_prob(&v(&x)).unwrap(), expected, 1e-12, "walk");
    // a batch of two series with one drift each and shared scales
    let batch = t(&[2, 5], &[x, [1.0, 0.5, 0.0, -0.5, -1.0]].concat());
    let drifts = v(&[0.2, -0.4]);
    let both = GaussianRandomWalk::new(&drifts, s(sigma), s(m0), s(s0))
        .log_prob(&batch)
        .unwrap();
    let second = GaussianRandomWalk::new(s(-0.4), s(sigma), s(m0), s(s0))
        .log_prob(&v(&[1.0, 0.5, 0.0, -0.5, -1.0]))
        .unwrap();
    assert_close(both, expected + second, 1e-12, "batch");
    // one value is its initial distribution
    assert_close(
        walk.log_prob(&v(&[0.4])).unwrap(),
        normal(0.4, m0, s0),
        1e-12,
        "one value",
    );
    // a random walk is AR(1) with coefficient 1 and the drift as its constant
    let ar = Autoregressive::new(s(mu), v(&[1.0]), s(sigma), s(m0), s(s0));
    assert_close(ar.log_prob(&v(&x)).unwrap(), expected, 1e-12, "AR(1)");
}

#[test]
fn the_autoregressive_density_conditions_on_the_first_p_values() {
    let x = [0.3, -0.2, 0.5, 0.9, 0.1, -0.4];
    let (c, rho, sigma, m0, s0) = (0.1, [0.6, -0.3], 0.8, 0.0, 2.0);
    let mut expected = normal(x[0], m0, s0) + normal(x[1], m0, s0);
    for i in 2..x.len() {
        expected += normal(x[i], c + rho[0] * x[i - 1] + rho[1] * x[i - 2], sigma);
    }
    let ar = Autoregressive::new(s(c), v(&rho), s(sigma), s(m0), s(s0));
    assert_eq!(ar.order().unwrap(), 2);
    assert_close(ar.log_prob(&v(&x)).unwrap(), expected, 1e-12, "AR(2)");
    // coefficients per series, [2, p]
    let rhos = t(&[2, 2], &[0.6, -0.3, 0.2, 0.1]);
    let batch = t(&[2, 6], &[x, x].concat());
    let both = Autoregressive::new(s(c), &rhos, s(sigma), s(m0), s(s0))
        .log_prob(&batch)
        .unwrap();
    let second = Autoregressive::new(s(c), v(&[0.2, 0.1]), s(sigma), s(m0), s(s0))
        .log_prob(&v(&x))
        .unwrap();
    assert_close(both, expected + second, 1e-12, "batch");
    // a series no longer than the order is all initial values
    let short = ar.log_prob(&v(&[0.3, -0.2])).unwrap();
    assert_close(
        short,
        normal(0.3, m0, s0) + normal(-0.2, m0, s0),
        1e-12,
        "short",
    );
}

#[test]
fn the_garch_density_follows_the_variance_recursion() {
    let r = [0.3, -1.2, 0.5, 2.1, -0.4, 0.0, 0.8];
    let (omega, alpha, beta, v0) = (0.1, 0.25, 0.6, 0.9);
    let mut variance: f64 = v0 * v0;
    let mut expected = 0.0;
    for (i, &x) in r.iter().enumerate() {
        if i > 0 {
            variance = omega + alpha * r[i - 1] * r[i - 1] + beta * variance;
        }
        expected += normal(x, 0.0, variance.sqrt());
    }
    let garch = Garch11::new(s(omega), s(alpha), s(beta), s(v0));
    assert_close(garch.log_prob(&v(&r)).unwrap(), expected, 1e-12, "GARCH");
}

/// The inputs of a process, x first, and functions of them.
struct Case {
    name: &'static str,
    inputs: Vec<DoubleTensor>,
    log_prob: fn(&[DoubleTensor]) -> f64,
    gradient: fn(&[DoubleTensor]) -> Vec<DoubleTensor>,
    /// Hessian-vector products along one direction ([1, shape] tangents), shaped [1, shape].
    hvp: fn(&[DoubleTensor], &[DoubleTensor]) -> Vec<DoubleTensor>,
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "random walk",
            inputs: vec![
                t(&[2, 4], &[0.3, 0.9, 0.4, 1.6, -0.5, 0.1, 0.6, 0.2]),
                v(&[0.2, -0.1]),
                s(0.7),
                s(-0.1),
                v(&[1.5, 0.8]),
            ],
            log_prob: |i| {
                GaussianRandomWalk::new(&i[1], &i[2], &i[3], &i[4])
                    .log_prob(&i[0])
                    .unwrap()
            },
            gradient: |i| {
                let g = GaussianRandomWalk::new(&i[1], &i[2], &i[3], &i[4])
                    .d_log_prob(&i[0])
                    .unwrap();
                vec![g.x, g.mu, g.sigma, g.init_mu, g.init_sigma]
            },
            hvp: |i, d| {
                let tangents: Vec<Option<&DoubleTensor>> = d[1..].iter().map(Some).collect();
                let g = GaussianRandomWalk::new(&i[1], &i[2], &i[3], &i[4])
                    .hvp(&i[0], Some(&d[0]), &tangents, 1)
                    .unwrap();
                vec![g.x, g.mu, g.sigma, g.init_mu, g.init_sigma]
            },
        },
        Case {
            name: "AR(2)",
            inputs: vec![
                t(
                    &[2, 5],
                    &[0.3, -0.2, 0.5, 0.9, 0.1, 1.0, 0.4, -0.3, 0.2, 0.7],
                ),
                s(0.1),
                t(&[2, 2], &[0.6, -0.3, 0.2, 0.1]),
                v(&[0.8, 1.1]),
                s(0.2),
                s(2.0),
            ],
            log_prob: |i| {
                Autoregressive::new(&i[1], &i[2], &i[3], &i[4], &i[5])
                    .log_prob(&i[0])
                    .unwrap()
            },
            gradient: |i| {
                let g = Autoregressive::new(&i[1], &i[2], &i[3], &i[4], &i[5])
                    .d_log_prob(&i[0])
                    .unwrap();
                vec![g.x, g.constant, g.rho, g.sigma, g.init_mu, g.init_sigma]
            },
            hvp: |i, d| {
                let tangents: Vec<Option<&DoubleTensor>> = d[1..].iter().map(Some).collect();
                let g = Autoregressive::new(&i[1], &i[2], &i[3], &i[4], &i[5])
                    .hvp(&i[0], Some(&d[0]), &tangents, 1)
                    .unwrap();
                vec![g.x, g.constant, g.rho, g.sigma, g.init_mu, g.init_sigma]
            },
        },
        Case {
            name: "GARCH(1,1)",
            inputs: vec![
                t(
                    &[2, 5],
                    &[0.3, -1.2, 0.5, 2.1, -0.4, 0.1, 0.9, -0.7, 0.2, 1.4],
                ),
                s(0.1),
                v(&[0.25, 0.1]),
                s(0.6),
                s(0.9),
            ],
            log_prob: |i| {
                Garch11::new(&i[1], &i[2], &i[3], &i[4])
                    .log_prob(&i[0])
                    .unwrap()
            },
            gradient: |i| {
                let g = Garch11::new(&i[1], &i[2], &i[3], &i[4])
                    .d_log_prob(&i[0])
                    .unwrap();
                vec![g.x, g.omega, g.alpha, g.beta, g.initial_vol]
            },
            hvp: |i, d| {
                let tangents: Vec<Option<&DoubleTensor>> = d[1..].iter().map(Some).collect();
                let g = Garch11::new(&i[1], &i[2], &i[3], &i[4])
                    .hvp(&i[0], Some(&d[0]), &tangents, 1)
                    .unwrap();
                vec![g.x, g.omega, g.alpha, g.beta, g.initial_vol]
            },
        },
    ]
}

/// `inputs` with element j of input k moved by h.
fn moved(inputs: &[DoubleTensor], k: usize, j: usize, h: f64) -> Vec<DoubleTensor> {
    let mut out = inputs.to_vec();
    let mut data = out[k].as_slice().to_vec();
    data[j] += h;
    out[k] = DoubleTensor::new(out[k].shape(), data).unwrap();
    out
}

#[test]
fn gradients_match_finite_differences() {
    for case in cases() {
        let gradient = (case.gradient)(&case.inputs);
        for (k, input) in case.inputs.iter().enumerate() {
            assert_eq!(
                gradient[k].shape(),
                input.shape(),
                "{} input {k}",
                case.name
            );
            for j in 0..input.len() {
                let h = 1e-6;
                let fd = ((case.log_prob)(&moved(&case.inputs, k, j, h))
                    - (case.log_prob)(&moved(&case.inputs, k, j, -h)))
                    / (2.0 * h);
                assert_close(
                    gradient[k].as_slice()[j],
                    fd,
                    1e-6 * (1.0 + fd.abs()),
                    &format!("{} input {k}[{j}]", case.name),
                );
            }
        }
    }
}

#[test]
fn hessian_vector_products_match_finite_differences_of_the_gradient() {
    for case in cases() {
        // one direction over every input
        let direction: Vec<DoubleTensor> = case
            .inputs
            .iter()
            .enumerate()
            .map(|(k, input)| {
                let data = (0..input.len())
                    .map(|j| 0.3 * ((j as f64) * 1.7 + k as f64).sin())
                    .collect();
                DoubleTensor::new(input.shape(), data).unwrap()
            })
            .collect();
        let tangents: Vec<DoubleTensor> = direction
            .iter()
            .map(|d| {
                let mut shape = vec![1];
                shape.extend_from_slice(d.shape());
                d.reshape(&shape.iter().map(|&s| s as isize).collect::<Vec<_>>())
                    .unwrap()
            })
            .collect();
        let hvp = (case.hvp)(&case.inputs, &tangents);
        let h = 1e-5;
        let along = |sign: f64| -> Vec<DoubleTensor> {
            let inputs: Vec<DoubleTensor> = case
                .inputs
                .iter()
                .zip(&direction)
                .map(|(x, d)| x.add(&(d * (sign * h))).unwrap())
                .collect();
            (case.gradient)(&inputs)
        };
        let (plus, minus) = (along(1.0), along(-1.0));
        for k in 0..case.inputs.len() {
            let mut shape = vec![1];
            shape.extend_from_slice(case.inputs[k].shape());
            assert_eq!(hvp[k].shape(), shape.as_slice(), "{} input {k}", case.name);
            for j in 0..case.inputs[k].len() {
                let fd = (plus[k].as_slice()[j] - minus[k].as_slice()[j]) / (2.0 * h);
                assert_close(
                    hvp[k].as_slice()[j],
                    fd,
                    1e-6 * (1.0 + fd.abs()),
                    &format!("{} input {k}[{j}]", case.name),
                );
            }
        }
        // two directions at once give each direction's products
        let doubled: Vec<DoubleTensor> = tangents
            .iter()
            .map(|t| DoubleTensor::concat(0, &[t, &(t * 2.0)]).unwrap())
            .collect();
        let refs: Vec<Option<&DoubleTensor>> = doubled[1..].iter().map(Some).collect();
        if case.name == "GARCH(1,1)" {
            let i = &case.inputs;
            let g = Garch11::new(&i[1], &i[2], &i[3], &i[4])
                .hvp(&i[0], Some(&doubled[0]), &refs, 2)
                .unwrap();
            let one = hvp[0].as_slice();
            let n = one.len();
            for (j, &h) in one.iter().enumerate() {
                assert_close(g.x.as_slice()[j], h, 1e-12, "first direction");
                assert_close(g.x.as_slice()[n + j], 2.0 * h, 1e-12, "second");
            }
        }
    }
}

#[test]
fn simulated_walks_have_the_increments_moments() {
    let (mu, sigma, m0, s0) = (0.3, 0.5, 2.0, 0.1);
    let walk = GaussianRandomWalk::new(s(mu), s(sigma), s(m0), s(s0));
    let mut rng = KeanuRng::seed_from_u64(1);
    let x = walk.sample(&[400, 50], &mut rng).unwrap();
    let mut increments = Vec::new();
    let mut first = Vec::new();
    for series in x.as_slice().chunks(50) {
        first.push(series[0]);
        increments.extend(series.windows(2).map(|w| w[1] - w[0]));
    }
    let (mean, var) = moments(&increments);
    let n = increments.len() as f64;
    assert_close(mean, mu, 4.5 * sigma / n.sqrt(), "increment mean");
    assert_close(
        var,
        sigma * sigma,
        4.5 * sigma * sigma * (2.0 / n).sqrt(),
        "increment variance",
    );
    let (mean, _) = moments(&first);
    assert_close(mean, m0, 4.5 * s0 / 20.0, "first value");
}

#[test]
fn simulated_ar_series_reach_the_stationary_moments() {
    // AR(2) with c = 0.5: mean c / (1 − ρ₁ − ρ₂), and the stationary variance
    // σ² (1 − ρ₂) / ((1 + ρ₂) ((1 − ρ₂)² − ρ₁²))
    let (c, r1, r2, sigma) = (0.5, 0.5, -0.3, 1.0);
    let ar = Autoregressive::new(s(c), v(&[r1, r2]), s(sigma), s(0.0), s(1.0));
    let mut rng = KeanuRng::seed_from_u64(2);
    let x = ar.sample(&[200, 100], &mut rng).unwrap();
    // drop each series' burn-in
    let late: Vec<f64> = x
        .as_slice()
        .chunks(100)
        .flat_map(|s| s[50..].to_vec())
        .collect();
    let (mean, var) = moments(&late);
    let expected_mean = c / (1.0 - r1 - r2);
    let expected_var = sigma * sigma * (1.0 - r2) / ((1.0 + r2) * ((1.0 - r2).powi(2) - r1 * r1));
    // the series are autocorrelated, so allow for an effective sample size of a few thousand
    assert_close(mean, expected_mean, 0.06, "mean");
    assert_close(var, expected_var, 0.08 * expected_var, "variance");
}

#[test]
fn simulated_garch_returns_have_the_unconditional_variance() {
    let (omega, alpha, beta) = (0.2, 0.1, 0.7);
    let garch = Garch11::new(s(omega), s(alpha), s(beta), s(1.0));
    let mut rng = KeanuRng::seed_from_u64(3);
    let r = garch.sample(&[400, 100], &mut rng).unwrap();
    let late: Vec<f64> = r
        .as_slice()
        .chunks(100)
        .flat_map(|s| s[50..].to_vec())
        .collect();
    let (mean, var) = moments(&late);
    let expected = omega / (1.0 - alpha - beta);
    assert_close(mean, 0.0, 0.03, "mean");
    assert_close(var, expected, 0.06 * expected, "variance");
    // volatility clusters: squared returns are positively autocorrelated, the returns themselves are not
    let squares: Vec<f64> = late.iter().map(|x| x * x).collect();
    assert!(lag_one_correlation(&squares, 50) > 0.05);
    assert!(lag_one_correlation(&late, 50).abs() < 0.03);
}

fn moments(x: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    (mean, var)
}

/// The lag-one autocorrelation within consecutive series of `len` values.
fn lag_one_correlation(x: &[f64], len: usize) -> f64 {
    let (mean, var) = moments(x);
    let mut total = 0.0;
    let mut count = 0.0;
    for series in x.chunks(len) {
        for w in series.windows(2) {
            total += (w[0] - mean) * (w[1] - mean);
            count += 1.0;
        }
    }
    total / count / var
}

#[test]
fn garch_parameters_outside_the_stationary_region_are_impossible() {
    let r = v(&[0.3, -0.5, 0.2]);
    let invalid = [
        (0.1, 0.5, 0.5, 1.0),  // α + β = 1
        (0.1, 0.6, 0.5, 1.0),  // α + β > 1
        (0.0, 0.1, 0.5, 1.0),  // ω = 0
        (0.1, -0.1, 0.5, 1.0), // α < 0
        (0.1, 0.1, -0.5, 1.0), // β < 0
        (0.1, 0.1, 0.5, 0.0),  // σ₀ = 0
        (f64::NAN, 0.1, 0.5, 1.0),
    ];
    let mut rng = KeanuRng::seed_from_u64(1);
    for (omega, alpha, beta, v0) in invalid {
        let garch = Garch11::new(s(omega), s(alpha), s(beta), s(v0));
        assert_eq!(
            garch.log_prob(&r).unwrap(),
            f64::NEG_INFINITY,
            "{omega} {alpha} {beta} {v0}"
        );
        assert!(garch.sample(&[3], &mut rng).is_err());
    }
    // one invalid series in a batch makes the whole value impossible
    let batch = Garch11::new(s(0.1), v(&[0.2, 0.9]), s(0.5), s(1.0));
    assert_eq!(
        batch.log_prob(&t(&[2, 3], &[0.1; 6])).unwrap(),
        f64::NEG_INFINITY
    );
}

#[test]
fn invalid_scales_and_shapes() {
    let walk = GaussianRandomWalk::new(s(0.0), s(-1.0), s(0.0), s(1.0));
    assert!(walk.log_prob(&v(&[0.1, 0.2])).unwrap().is_nan());
    assert!(walk.sample(&[3], &mut KeanuRng::seed_from_u64(1)).is_err());
    let ar = Autoregressive::new(s(0.0), v(&[0.5]), s(1.0), s(0.0), s(0.0));
    assert!(ar.log_prob(&v(&[0.1, 0.2])).unwrap().is_nan());
    let ok = GaussianRandomWalk::new(s(0.0), s(1.0), s(0.0), s(1.0));
    // a scalar or an empty series
    assert!(ok.log_prob(&s(0.3)).is_err());
    assert!(ok.log_prob(&DoubleTensor::zeros(&[0])).is_err());
    // a parameter that doesn't broadcast to the batch
    let wide = GaussianRandomWalk::new(v(&[0.0, 1.0, 2.0]), s(1.0), s(0.0), s(1.0));
    assert!(wide.log_prob(&t(&[2, 3], &[0.0; 6])).is_err());
    // AR coefficients need a lag axis
    let scalar_rho = Autoregressive::new(s(0.0), s(0.5), s(1.0), s(0.0), s(1.0));
    assert!(scalar_rho.order().is_err());
    assert!(scalar_rho.log_prob(&v(&[0.1, 0.2])).is_err());
    // the wrong number of tangents
    assert!(ok.hvp(&v(&[0.1, 0.2]), None, &[None], 1).is_err());
}
