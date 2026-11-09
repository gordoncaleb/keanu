use keanu_distributions::{KeanuRng, PolyaGamma1};
use keanu_tensor::DoubleTensor;

#[test]
fn density_agrees_with_the_inverse_gaussian_series_on_both_sides_of_the_switch() {
    // The reference uses the inverse-Gaussian series at every x, evaluated with 65 decimal digits.
    for (x, z, expected) in [
        (0.03, 0.0, 0.174_231_646_108_623_1),
        (0.16, 2.5, 1.178_618_054_952_898_6),
        (0.16, 0.0, 1.042_875_501_220_294_3),
        (0.4, -2.5, -0.750_301_676_003_557_4),
        (1.1, 0.0, -3.590_405_354_189_801_7),
        (0.0005, 1000.0, 9.789_267_975_548_505),
    ] {
        let d = PolyaGamma1::log_prob_derivatives(x, [z]);
        assert!((d.value - expected).abs() < 4e-14, "{d:?}");
        let tilt_score = 0.5 * (0.5 * z).tanh() - x * z;
        let tilt_curvature = 0.25 / (0.5 * z).cosh().powi(2) - x;
        assert!((d.gradient[1] - tilt_score).abs() < 2e-14, "{d:?}");
        assert!((d.hessian[1][1] - tilt_curvature).abs() < 2e-14, "{d:?}");
        assert!((d.hessian[0][1] + z).abs() < 1e-11, "{d:?}");
    }
}

#[test]
fn density_normalizes_and_matches_the_laplace_transform() {
    for z in [0.0_f64, 3.5] {
        let step = 0.001;
        let x = DoubleTensor::vector((1..=8000).map(|i| f64::from(i) * step).collect());
        let log_density = PolyaGamma1::new(DoubleTensor::scalar(z))
            .log_prob(&x)
            .unwrap();
        let mass = log_density.exp().sum() * step;
        assert!((mass - 1.0).abs() < 2e-12, "{mass}");
        let laplace = log_density.sub(&(&x * 4.0)).unwrap().exp().sum() * step;
        let c = z / 2.0;
        let expected = c.cosh() / (c * c + 2.0).sqrt().cosh();
        assert!(
            (laplace - expected).abs() < 2e-12,
            "{laplace} vs {expected}"
        );
    }
}

#[test]
fn exact_sampler_broadcasts_and_recovers_moments_across_tilts() {
    let z = [0.0_f64, 3.0, -25.0, 1000.0];
    let samples = PolyaGamma1::new(DoubleTensor::vector(z.to_vec()))
        .sample(&[25_000, 4], &mut KeanuRng::seed_from_u64(409))
        .unwrap();
    assert_eq!(samples.shape(), &[25_000, 4]);
    assert!(samples.as_slice().iter().all(|&x| x > 0.0 && x.is_finite()));
    for (column, z) in z.into_iter().enumerate() {
        let c = 0.5 * z;
        let expected_mean = if z == 0.0 { 0.25 } else { c.tanh() / (2.0 * z) };
        let s = 4.0 * (1.0 + z.abs());
        let expected_laplace = c.cosh() / (c * c + s / 2.0).sqrt().cosh();
        let (sum, laplace_sum) = samples
            .as_slice()
            .chunks_exact(4)
            .map(|row| (row[column], (-s * row[column]).exp()))
            .fold((0.0, 0.0), |(sx, sl), (x, l)| (sx + x, sl + l));
        assert!(
            (sum / 25_000.0 / expected_mean - 1.0).abs() < 0.024,
            "z={z}"
        );
        assert!(
            (laplace_sum / 25_000.0 - expected_laplace).abs() < 0.008,
            "z={z}"
        );
    }
}

#[test]
fn support_and_extreme_tilts() {
    let d = PolyaGamma1::new(DoubleTensor::scalar(0.0));
    let x = DoubleTensor::vector(vec![f64::NEG_INFINITY, -1.0, 0.0, f64::INFINITY]);
    assert_eq!(d.log_prob(&x).unwrap().as_slice(), &[f64::NEG_INFINITY; 4]);
    let mut rng = KeanuRng::seed_from_u64(410);
    let huge = PolyaGamma1::new(DoubleTensor::scalar(f64::MAX));
    assert!(
        huge.sample(&[4], &mut rng)
            .unwrap()
            .as_slice()
            .iter()
            .all(|&x| x > 0.0)
    );
    let invalid = PolyaGamma1::new(DoubleTensor::scalar(f64::INFINITY));
    assert!(invalid.sample(&[], &mut rng).is_err());
    assert!(
        invalid
            .log_prob(&DoubleTensor::scalar(0.1))
            .unwrap()
            .as_slice()[0]
            .is_nan()
    );
}
