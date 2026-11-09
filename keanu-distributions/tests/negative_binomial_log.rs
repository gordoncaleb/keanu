use keanu_distributions::{KeanuRng, NegativeBinomial, NegativeBinomialLog};
use keanu_tensor::{DoubleTensor, IntTensor};

#[test]
fn log_mean_matches_the_mean_parameterization_and_handles_extreme_predictors() {
    let dispersion = DoubleTensor::scalar(2.0);
    let eta = DoubleTensor::vector(vec![-2.0, 0.0, 3.0]);
    let mean = eta.exp();
    let counts = IntTensor::vector(vec![0, 3, 17]);
    let ordinary = NegativeBinomial::new(&mean, &dispersion);
    let log_mean = NegativeBinomialLog::new(&eta, &dispersion);
    let expected = ordinary.log_prob(&counts).unwrap();
    for (actual, expected) in log_mean
        .log_prob(&counts)
        .unwrap()
        .as_slice()
        .iter()
        .zip(expected.as_slice())
    {
        assert!((actual - expected).abs() < 1e-12);
    }
    let expected = ordinary.d_log_prob(&counts).unwrap();
    let actual = log_mean.d_log_prob(&counts).unwrap();
    for (actual, expected) in actual
        .log_mu
        .as_slice()
        .iter()
        .zip(expected.mu.mul(&mean).unwrap().as_slice())
    {
        assert!((actual - expected).abs() < 1e-12);
    }

    // For alpha=2, the mass is (k+1) q² (1-q)^k. At these predictors the limiting expressions are
    // exact to f64 precision, although exp(eta) overflows or underflows.
    let extreme = NegativeBinomialLog::new(
        DoubleTensor::vector(vec![-1000.0, 1000.0, 1e300, f64::NEG_INFINITY]),
        &dispersion,
    );
    let counts = IntTensor::vector(vec![3, 3, 3, 0]);
    let lp = extreme.log_prob(&counts).unwrap();
    assert!(
        (lp.as_slice()[0] - (4.0_f64.ln() + 3.0 * (-1000.0 - std::f64::consts::LN_2))).abs()
            < 1e-12
    );
    assert!((lp.as_slice()[1] - (16.0_f64.ln() - 2000.0)).abs() < 1e-12);
    assert!(lp.as_slice()[2].is_finite());
    assert_eq!(lp.as_slice()[3], 0.0);
    assert_eq!(
        extreme.d_log_prob(&counts).unwrap().log_mu.as_slice(),
        &[3.0, -2.0, -2.0, 0.0]
    );

    let tiny_odds =
        NegativeBinomialLog::new(DoubleTensor::scalar(-150.0), DoubleTensor::scalar(1e300));
    assert_eq!(
        tiny_odds
            .log_prob(&IntTensor::scalar(0))
            .unwrap()
            .as_slice(),
        &[-(-150.0_f64).exp()]
    );
}

#[test]
fn log_mean_draws_follow_gamma_poisson_moments() {
    let mut rng = KeanuRng::seed_from_u64(15);
    let distribution = NegativeBinomialLog::new(
        DoubleTensor::scalar(4.0_f64.ln()),
        DoubleTensor::scalar(2.0),
    );
    let draws = distribution.sample(&[40_000], &mut rng).unwrap();
    let n = draws.len() as f64;
    let mean = draws.as_slice().iter().map(|&x| x as f64).sum::<f64>() / n;
    let variance = draws
        .as_slice()
        .iter()
        .map(|&x| (x as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    assert!((mean - 4.0).abs() < 0.08, "{mean}");
    assert!((variance - 12.0).abs() < 0.5, "{variance}");

    let zero = NegativeBinomialLog::new(
        DoubleTensor::scalar(f64::NEG_INFINITY),
        DoubleTensor::scalar(2.0),
    );
    assert_eq!(zero.sample(&[3], &mut rng).unwrap().as_slice(), &[0; 3]);
    let too_large =
        NegativeBinomialLog::new(DoubleTensor::scalar(1000.0), DoubleTensor::scalar(2.0));
    assert!(too_large.sample(&[], &mut rng).is_err());
}
