use keanu_distributions::{KeanuRng, LocationScaleStudentT, SkewStudentT};
use keanu_tensor::DoubleTensor;

fn close(a: f64, b: f64) {
    assert!((a - b).abs() <= 2e-12 * b.abs().max(1.0), "{a} vs {b}");
}

fn skew(a: f64, b: f64, mu: f64, sigma: f64) -> SkewStudentT<'static> {
    SkewStudentT::new(
        DoubleTensor::scalar(a),
        DoubleTensor::scalar(b),
        DoubleTensor::scalar(mu),
        DoubleTensor::scalar(sigma),
    )
}

#[test]
fn skew_student_t_matches_the_beta_polynomial_reference() {
    let (mu, sigma) = (0.4, 1.7);
    let dist = skew(3.0, 2.0, mu, sigma);
    let xs = [-12.0, -2.0, mu, 1.4, 5.0, 28.0];
    let x = DoubleTensor::vector(xs.to_vec());
    let (density, cdf, sf) = (
        dist.log_prob(&x).unwrap(),
        dist.log_cdf(&x).unwrap(),
        dist.log_ccdf(&x).unwrap(),
    );
    for (i, x) in xs.into_iter().enumerate() {
        let z = (x - mu) / sigma;
        let r = z.hypot(5.0_f64.sqrt());
        let u = (1.0 + z / r) * 0.5;
        // Beta(3, 2) CDF and its complementary polynomial.
        close(cdf.as_slice()[i], (u.powi(3) * (4.0 - 3.0 * u)).ln());
        close(
            sf.as_slice()[i],
            ((1.0 - u).powi(2) * (1.0 + 2.0 * u + 3.0 * u * u)).ln(),
        );
        close(
            density.as_slice()[i],
            (24.0 * u.powi(3) * (1.0 - u).powi(2) / (sigma * r)).ln(),
        );
    }
}

#[test]
fn equal_shapes_agree_with_location_scale_student_t() {
    let a = 2.7;
    let dist = skew(a, a, -0.3, 1.4);
    let student = LocationScaleStudentT::new(
        DoubleTensor::scalar(2.0 * a),
        DoubleTensor::scalar(-0.3),
        DoubleTensor::scalar(1.4),
    );
    let x = DoubleTensor::vector(vec![-100.0, -0.3, 0.4, 5.0, 100.0]);
    for (a, b) in [
        (dist.log_prob(&x), student.log_prob(&x)),
        (dist.log_cdf(&x), student.log_cdf(&x)),
        (dist.log_ccdf(&x), student.log_ccdf(&x)),
    ] {
        for (&a, &b) in a.unwrap().as_slice().iter().zip(b.unwrap().as_slice()) {
            close(a, b);
        }
    }
}

#[test]
fn tails_remain_scoreable_beyond_beta_coordinate_underflow() {
    let dist = skew(3.0, 2.0, 0.0, 1.0);
    let x = DoubleTensor::scalar(1e200);
    // 1 − u ∼ (a+b)/(4 x²), and the beta survival is asymptotic to 6(1 − u)².
    let expected = 6.0_f64.ln() + 2.0 * ((5.0_f64 / 4.0).ln() - 2.0 * 1e200_f64.ln());
    close(dist.log_ccdf(&x).unwrap().to_scalar().unwrap(), expected);
    assert!(
        dist.d_log_ccdf(&x)
            .unwrap()
            .b
            .to_scalar()
            .unwrap()
            .is_finite()
    );
    assert!(dist.log_prob(&x).unwrap().to_scalar().unwrap().is_finite());
}

#[test]
fn sampling_matches_the_beta_probability_at_the_location() {
    let dist = skew(3.0, 2.0, 0.4, 1.7);
    let samples = dist
        .sample(&[12000], &mut KeanuRng::seed_from_u64(75))
        .unwrap();
    let left = samples.as_slice().iter().filter(|&&x| x <= 0.4).count() as f64 / 12000.0;
    assert!((left - 0.3125).abs() < 0.015, "{left}");
    assert!(
        skew(0.0, 2.0, 0.4, 1.7)
            .sample(&[1], &mut KeanuRng::seed_from_u64(76))
            .is_err()
    );
}

#[test]
fn large_equal_shapes_keep_the_normal_limit_of_the_density() {
    let dist = skew(1e12, 1e12, 0.0, 1.0);
    for x in [-2.0_f64, 0.0, 1.3] {
        let actual = dist
            .log_prob(&DoubleTensor::scalar(x))
            .unwrap()
            .to_scalar()
            .unwrap();
        let expected = -0.5 * (std::f64::consts::TAU.ln() + x * x);
        assert!((actual - expected).abs() < 5e-9, "{actual} vs {expected}");
    }
}
