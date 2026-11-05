//! The zero-sum normal: its density is N(0, σ² I) in an orthonormal basis of the zero-sum subspace, values off the
//! subspace are impossible, gradients match finite differences, and draws sum to zero with marginal variance
//! σ² (N − 1) / N.

use keanu_distributions::{DistributionError, KeanuRng, ZeroSumNormal};
use keanu_tensor::DoubleTensor;
use std::f64::consts::PI;

fn t(shape: &[usize], data: &[f64]) -> DoubleTensor {
    DoubleTensor::new(shape, data.to_vec()).unwrap()
}

#[test]
fn the_density_is_a_gaussian_on_the_subspace() {
    let sigma = DoubleTensor::scalar(1.7);
    let d = ZeroSumNormal::new(&sigma);
    let x = [0.4, -1.1, 0.2, 0.5];
    let squares: f64 = x.iter().map(|v| v * v).sum();
    // three standard Gaussian coordinates of variance σ²
    let expected = -0.5 * squares / (1.7 * 1.7) - 3.0 * (1.7f64.ln() + 0.5 * (2.0 * PI).ln());
    let actual = d.log_prob(&t(&[4], &x)).unwrap();
    assert!((actual - expected).abs() < 1e-12, "{actual} vs {expected}");
    // a vector of two is one Gaussian coordinate, (x₀ − x₁)/√2 = √2 x₀
    let two = d.log_prob(&t(&[2], &[0.3, -0.3])).unwrap();
    let coordinate = 0.3 * 2f64.sqrt();
    let expected = -0.5 * (coordinate / 1.7f64).powi(2) - (1.7f64.ln() + 0.5 * (2.0 * PI).ln());
    assert!((two - expected).abs() < 1e-12);
    // a batch with one sigma per vector is the sum of each vector's density
    let sigmas = DoubleTensor::vector(vec![0.5, 2.0]);
    let batch = ZeroSumNormal::new(&sigmas)
        .log_prob(&t(&[2, 2], &[0.3, -0.3, 1.0, -1.0]))
        .unwrap();
    let each = ZeroSumNormal::new(DoubleTensor::scalar(0.5))
        .log_prob(&t(&[2], &[0.3, -0.3]))
        .unwrap()
        + ZeroSumNormal::new(DoubleTensor::scalar(2.0))
            .log_prob(&t(&[2], &[1.0, -1.0]))
            .unwrap();
    assert!((batch - each).abs() < 1e-12);
    // a single element must be 0
    assert_eq!(d.log_prob(&t(&[1], &[0.0])).unwrap(), 0.0);
}

#[test]
fn values_off_the_subspace_are_impossible_and_bad_parameters_are_nan_or_errors() {
    let d = ZeroSumNormal::new(DoubleTensor::scalar(1.0));
    assert_eq!(
        d.log_prob(&t(&[3], &[1.0, -0.5, -0.4])).unwrap(),
        f64::NEG_INFINITY
    );
    assert_eq!(
        d.log_prob(&t(&[2], &[f64::INFINITY, f64::NEG_INFINITY]))
            .unwrap(),
        f64::NEG_INFINITY
    );
    // rounding error is not an error: a sum of 1e-14 at values of order 1e3
    assert!(
        d.log_prob(&t(&[2], &[1e3, -1e3 + 1e-11]))
            .unwrap()
            .is_finite()
    );
    assert!(
        ZeroSumNormal::new(DoubleTensor::scalar(-1.0))
            .log_prob(&t(&[2], &[1.0, -1.0]))
            .unwrap()
            .is_nan()
    );
    assert!(d.log_prob(&DoubleTensor::scalar(0.0)).is_err());
    assert!(d.log_prob(&DoubleTensor::zeros(&[2, 0])).is_err());
    let mismatched = ZeroSumNormal::new(DoubleTensor::vector(vec![1.0, 2.0, 3.0]));
    assert!(mismatched.log_prob(&DoubleTensor::zeros(&[2, 4])).is_err());
    let wider = ZeroSumNormal::new(DoubleTensor::zeros(&[2, 1]));
    assert!(matches!(
        wider.log_prob(&DoubleTensor::zeros(&[2, 4])),
        Err(DistributionError::InvalidParameter(_))
    ));
    let mut rng = KeanuRng::seed_from_u64(1);
    assert!(
        ZeroSumNormal::new(DoubleTensor::scalar(0.0))
            .sample(&[3], &mut rng)
            .is_err()
    );
}

#[test]
fn gradients_match_finite_differences() {
    let sigmas = DoubleTensor::vector(vec![0.8, 1.6]);
    let x = t(&[2, 3], &[0.3, -0.5, 0.2, 1.5, -2.0, 0.5]);
    let g = ZeroSumNormal::new(&sigmas).d_log_prob(&x).unwrap();
    let h = 1e-6;
    for k in 0..2 {
        let at = |delta: f64| {
            let mut s = sigmas.as_slice().to_vec();
            s[k] += delta;
            ZeroSumNormal::new(DoubleTensor::vector(s))
                .log_prob(&x)
                .unwrap()
        };
        let fd = (at(h) - at(-h)) / (2.0 * h);
        assert!((g.sigma.as_slice()[k] - fd).abs() < 1e-6, "sigma[{k}]");
    }
    // along a zero-sum direction the value's gradient is the directional derivative
    let direction = [0.6, -0.1, -0.5, -0.2, 0.9, -0.7];
    let at = |delta: f64| {
        let moved: Vec<f64> = x
            .as_slice()
            .iter()
            .zip(&direction)
            .map(|(x, d)| x + delta * d)
            .collect();
        ZeroSumNormal::new(&sigmas)
            .log_prob(&t(&[2, 3], &moved))
            .unwrap()
    };
    let fd = (at(h) - at(-h)) / (2.0 * h);
    let analytic: f64 =
        g.x.as_slice()
            .iter()
            .zip(&direction)
            .map(|(g, d)| g * d)
            .sum();
    assert!((analytic - fd).abs() < 1e-6, "{analytic} vs {fd}");
    // a scalar sigma gets the sum over the batch
    let scalar = ZeroSumNormal::new(DoubleTensor::scalar(1.2))
        .d_log_prob(&x)
        .unwrap();
    assert_eq!(scalar.sigma.shape(), &[] as &[usize]);
}

#[test]
fn draws_sum_to_zero_with_marginal_variance_sigma_squared_times_n_minus_one_over_n() {
    let (n, sigma, draws) = (5, 2.0, 40_000);
    let mut rng = KeanuRng::seed_from_u64(7);
    let samples = ZeroSumNormal::new(DoubleTensor::scalar(sigma))
        .sample(&[draws, n], &mut rng)
        .unwrap();
    let mut sums = vec![0.0; n];
    let mut squares = vec![0.0; n];
    for row in samples.as_slice().chunks_exact(n) {
        let total: f64 = row.iter().sum();
        assert!(total.abs() < 1e-12, "{row:?} sums to {total}");
        for (i, v) in row.iter().enumerate() {
            sums[i] += v;
            squares[i] += v * v;
        }
    }
    let expected = sigma * sigma * (n - 1) as f64 / n as f64;
    for i in 0..n {
        let mean = sums[i] / draws as f64;
        let variance = squares[i] / draws as f64 - mean * mean;
        // the variance's standard error is about expected·√(2 / draws) = 0.023
        assert!(mean.abs() < 0.03, "mean of element {i}: {mean}");
        assert!(
            (variance - expected).abs() < 0.1,
            "variance of element {i}: {variance} vs {expected}"
        );
    }
}
