use keanu_distributions::{KeanuRng, Multinomial, MultinomialLogit, MultivariateGaussianPrecision};
use keanu_tensor::{DoubleTensor, IntTensor};

#[test]
fn multinomial_logit_matches_an_exact_polynomial_and_handles_extreme_logits() {
    let logits = DoubleTensor::vector(vec![2.0_f64.ln(), 3.0_f64.ln(), 5.0_f64.ln()]);
    let dist = MultinomialLogit::new(IntTensor::scalar(6), &logits).unwrap();
    let x = IntTensor::vector(vec![1, 2, 3]);
    let expected = (60.0 * 0.2 * 0.3_f64.powi(2) * 0.5_f64.powi(3)).ln();
    assert!((dist.log_prob(&x).unwrap().to_scalar().unwrap() - expected).abs() < 1e-13);
    let gradient = dist.d_log_prob(&x).unwrap().logits;
    for (&g, expected) in gradient.as_slice().iter().zip([-0.2, 0.2, 0.0]) {
        assert!((g - expected).abs() < 1e-13);
    }

    let extreme = DoubleTensor::vector(vec![1000.0, -1000.0, -1000.0]);
    let dist = MultinomialLogit::new(IntTensor::scalar(1), &extreme).unwrap();
    let rare = IntTensor::vector(vec![0, 1, 0]);
    assert_eq!(dist.log_prob(&rare).unwrap().to_scalar(), Some(-2000.0));
    assert_eq!(
        dist.log_prob(&IntTensor::vector(vec![1, 0, 0]))
            .unwrap()
            .to_scalar(),
        Some(0.0)
    );
    assert!(
        Multinomial::new(IntTensor::scalar(1), extreme.softmax(-1).unwrap())
            .unwrap()
            .log_prob(&rare)
            .unwrap()
            .to_scalar()
            .unwrap()
            .is_infinite()
    );
    let huge = i64::MAX;
    let dist = MultinomialLogit::new(IntTensor::scalar(huge), &extreme).unwrap();
    assert_eq!(
        dist.log_prob(&IntTensor::vector(vec![huge, 1, 0]))
            .unwrap()
            .to_scalar(),
        Some(f64::NEG_INFINITY)
    );
}

#[test]
fn multinomial_batches_broadcast_and_sampling_preserves_each_total() {
    let totals = IntTensor::vector(vec![0, 17, 30]);
    let logits = DoubleTensor::vector(vec![0.0, 1.0, 2.0]);
    let dist = MultinomialLogit::new(&totals, &logits).unwrap();
    let mut rng = KeanuRng::seed_from_u64(9);
    let counts = dist.sample(&[500, 3, 3], &mut rng).unwrap();
    assert_eq!(dist.log_prob(&counts).unwrap().shape(), &[500, 3]);
    assert_eq!(dist.d_log_prob(&counts).unwrap().logits.shape(), &[3]);
    for (i, row) in counts.as_slice().chunks_exact(3).enumerate() {
        assert_eq!(row.iter().sum::<i64>(), totals.as_slice()[i % 3]);
    }
    assert!(dist.sample(&[3], &mut rng).is_err());
    let successes: i64 = counts
        .as_slice()
        .chunks_exact(9)
        .map(|batch| batch[8])
        .sum();
    assert!(
        (successes as f64 / 15000.0 - 2.0_f64.exp() / (1.0 + 1.0_f64.exp() + 2.0_f64.exp())).abs()
            < 0.02
    );
}

#[test]
fn precision_gaussian_matches_a_closed_form_density_and_draw_covariance() {
    let mu = DoubleTensor::vector(vec![0.0, 0.0]);
    // P = [[2, 1], [1, 3]], determinant 5, inverse [[3, -1], [-1, 2]] / 5.
    let p = DoubleTensor::new(&[2, 2], vec![2.0, 99.0, 1.0, 3.0]).unwrap();
    let dist = MultivariateGaussianPrecision::new(&mu, &p);
    let x = DoubleTensor::new(&[2, 2], vec![1.0, -2.0, 0.0, 0.0]).unwrap();
    let terms = dist.log_prob_terms(&x).unwrap();
    let at_zero = 0.5 * 5.0_f64.ln() - std::f64::consts::TAU.ln();
    for (&actual, expected) in terms.as_slice().iter().zip([at_zero - 5.0, at_zero]) {
        assert!((actual - expected).abs() < 1e-12);
    }
    let (lp, gradient) = dist.log_prob_and_gradient(&x).unwrap();
    assert_eq!(lp, terms.sum());
    let gradient = gradient.unwrap();
    for (&actual, expected) in gradient
        .precision
        .as_slice()
        .iter()
        .zip([0.1, 0.0, 1.6, -1.6])
    {
        assert!((actual - expected).abs() < 1e-12);
    }
    let mut rng = KeanuRng::seed_from_u64(10);
    let sample = dist.sample(&[15000, 2], &mut rng).unwrap();
    let xy = sample
        .as_slice()
        .chunks_exact(2)
        .map(|r| r[0] * r[1])
        .sum::<f64>()
        / 15000.0;
    assert!((xy + 0.2).abs() < 0.02);
    let invalid = MultivariateGaussianPrecision::new(&mu, DoubleTensor::zeros(&[2, 2]));
    assert_eq!(invalid.log_prob(&x).unwrap(), f64::NEG_INFINITY);
    assert!(invalid.log_prob_and_gradient(&x).unwrap().1.is_none());
    assert!(invalid.sample(&[2], &mut rng).is_err());
}

#[test]
fn precision_parameters_broadcast_without_reusing_random_variates() {
    let mu = DoubleTensor::new(&[2, 1, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let precision = DoubleTensor::eye(2)
        .unwrap()
        .broadcast_to(&[3, 2, 2])
        .unwrap();
    let dist = MultivariateGaussianPrecision::new(&mu, &precision);
    let x = DoubleTensor::vector(vec![0.0, 0.0]);
    let gradient = dist.d_log_prob(&x).unwrap();
    assert_eq!(dist.log_prob_terms(&x).unwrap().shape(), &[2, 3]);
    assert_eq!(gradient.x.shape(), &[2]);
    assert_eq!(gradient.mu.shape(), &[2, 1, 2]);
    assert_eq!(gradient.precision.shape(), &[3, 2, 2]);
    let mut rng = KeanuRng::seed_from_u64(11);
    assert!(dist.sample(&[1, 3, 2], &mut rng).is_err());
    let draws = dist.sample(&[2, 3, 2], &mut rng).unwrap();
    assert_ne!(&draws.as_slice()[..2], &draws.as_slice()[2..4]);
}
