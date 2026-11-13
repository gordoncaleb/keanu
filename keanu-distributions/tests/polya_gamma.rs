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

#[test]
fn general_shape_matches_high_precision_density_and_tails() {
    use keanu_distributions::PolyaGamma;
    // Independent 70-digit de Hoog inversion of the analytic Laplace transform.
    for (b, z, x, expected) in [
        (
            0.2,
            0.0,
            0.001,
            [
                2.278738728386476,
                -6.320983018038133,
                -0.0017997936560496323,
            ],
        ),
        (
            0.2,
            0.0,
            0.6,
            [-3.6947506715170703, -0.004130548756295552, -5.4914095735472],
        ),
        (
            0.7,
            -2.5,
            0.1,
            [1.489790280247668, -0.5570095559500473, -0.8507834361685346],
        ),
        (
            1.3,
            4.0,
            0.55,
            [
                -3.1122982936656374,
                -0.003597055837486742,
                -5.629437579841717,
            ],
        ),
        (
            4.5,
            1.5,
            0.9,
            [
                0.16489868105572458,
                -0.6922540060516361,
                -0.6940411535421865,
            ],
        ),
        (
            32.0,
            0.0,
            8.0,
            [-1.065009521726868, -0.6481048405445047, -0.7403144195513349],
        ),
        (
            32.0,
            0.0,
            16.0,
            [
                -15.546666519688562,
                -6.245748618386016e-8,
                -16.588_779_763_946_35,
            ],
        ),
        (
            3.1,
            40.0,
            0.02,
            [
                -8.675149094131298,
                -16.398467025158907,
                -7.555031435241168e-8,
            ],
        ),
        (
            3.1,
            40.0,
            0.1,
            [
                -27.03930596915823,
                -2.589109655470472e-15,
                -33.58746234066568,
            ],
        ),
        (
            0.04,
            0.0,
            20.0,
            [
                -104.69541274118948,
                -6.822241264581982e-47,
                -106.301_311_321_681_8,
            ],
        ),
    ] {
        let d = PolyaGamma::new(DoubleTensor::scalar(b), DoubleTensor::scalar(z));
        let t = DoubleTensor::scalar(x);
        for (actual, expected) in [d.log_prob(&t), d.log_cdf(&t), d.log_ccdf(&t)]
            .into_iter()
            .zip(expected)
        {
            let actual = actual.unwrap().as_slice()[0];
            assert!(
                (actual - expected).abs() < 4e-9 * expected.abs().max(1e-10),
                "b={b}, z={z}, x={x}, {actual} vs {expected}"
            );
        }
    }
}

#[test]
fn general_shape_sampler_recovers_moments_and_laplace_transform() {
    use keanu_distributions::PolyaGamma;
    let b = DoubleTensor::vector(vec![0.23, 5.0, 20.5, 3.1]);
    let z = DoubleTensor::vector(vec![0.0, 1.2, 2.5, 40.0]);
    let samples = PolyaGamma::new(&b, &z)
        .sample(&[20_000, 4], &mut KeanuRng::seed_from_u64(657))
        .unwrap();
    assert_eq!(samples.shape(), &[20_000, 4]);
    for i in 0..4 {
        let (b, c) = (b.as_slice()[i], 0.5 * z.as_slice()[i]);
        let mean = if c == 0.0 {
            b / 4.0
        } else {
            b * c.tanh() / (4.0 * c)
        };
        let variance = if c == 0.0 {
            b / 24.0
        } else {
            b * (c.tanh() - c / c.cosh().powi(2)) / (16.0 * c.powi(3))
        };
        let s = 1.0 / mean;
        let expected = (c.cosh() / (c * c + 0.5 * s).sqrt().cosh()).powf(b);
        let (first, second, laplace) = samples
            .as_slice()
            .chunks_exact(4)
            .map(|row| row[i])
            .fold((0.0, 0.0, 0.0), |(first, second, l), x| {
                (first + x, second + (x - mean).powi(2), l + (-s * x).exp())
            });
        assert!((first / 20_000.0 - mean).abs() < 5.0 * (variance / 20_000.0).sqrt());
        assert!((second / 20_000.0 / variance - 1.0).abs() < 0.09);
        assert!((laplace / 20_000.0 - expected).abs() < 0.009);
    }
}

#[test]
fn general_shape_one_keeps_log_tails_below_underflow() {
    use keanu_distributions::PolyaGamma;
    let general = PolyaGamma::new(DoubleTensor::scalar(1.0), DoubleTensor::scalar(0.0));
    let specialized = PolyaGamma1::new(DoubleTensor::scalar(0.0));
    let early = DoubleTensor::scalar(0.0001);
    let late = DoubleTensor::scalar(200.0);
    // The first image contributes twice the one-boundary Brownian hitting probability.
    let early_cdf =
        2.0 * std::f64::consts::LN_2 + keanu_tensor::special::log_ndtr(-0.5 / 0.0001_f64.sqrt());
    let late_survival = specialized.log_prob(&late).unwrap().as_slice()[0]
        - (std::f64::consts::PI.powi(2) / 2.0).ln();
    for (actual, expected) in [
        (general.log_cdf(&early), early_cdf),
        (general.log_ccdf(&late), late_survival),
        (
            general.log_prob(&early),
            specialized.log_prob(&early).unwrap().as_slice()[0],
        ),
    ] {
        let actual = actual.unwrap().as_slice()[0];
        assert!(expected < -750.0);
        assert!((actual - expected).abs() < 2e-8, "{actual} vs {expected}");
    }
}
