use keanu_distributions::{KeanuRng, LocationScaleStudentT};
use keanu_tensor::DoubleTensor;

#[test]
fn density_and_tails_match_high_precision_values() {
    let distribution = LocationScaleStudentT::new(
        DoubleTensor::scalar(3.7),
        DoubleTensor::scalar(1.2),
        DoubleTensor::scalar(0.8),
    );
    // Independent 70-digit mpmath references using loggamma and regularized betainc.
    for (x, density, cdf, ccdf) in [
        (
            -2.0,
            -4.692_471_961_138_375,
            -4.667_635_656_379_306,
            -0.009_438_861_235_540_465,
        ),
        (
            1.2,
            -0.762_600_293_318_029_5,
            -std::f64::consts::LN_2,
            -std::f64::consts::LN_2,
        ),
        (
            1.200_01,
            -0.762_600_293_417_269_3,
            -0.693_137_851_564_724_3,
            -0.693_156_509_642_197_2,
        ),
        (
            2.3,
            -1.732_582_223_270_234_6,
            -0.131_537_111_193_174_54,
            -2.093_513_995_144_026,
        ),
        (
            1e180,
            -1_946.723_781_531_279_7,
            0.0,
            -1_533.566_797_612_001_4,
        ),
    ] {
        let value = DoubleTensor::scalar(x);
        for (actual, expected) in [
            (distribution.log_prob(&value), density),
            (distribution.log_cdf(&value), cdf),
            (distribution.log_ccdf(&value), ccdf),
        ] {
            let actual = actual.unwrap().to_scalar().unwrap();
            assert!(
                (actual - expected).abs() < 1e-11,
                "x={x}, {actual} vs {expected}"
            );
        }
    }
}

#[test]
fn samples_have_the_expected_location_and_interquartile_range() {
    let distribution = LocationScaleStudentT::new(
        DoubleTensor::scalar(3.7),
        DoubleTensor::scalar(1.2),
        DoubleTensor::scalar(0.8),
    );
    let draws = distribution
        .sample(&[25_000], &mut KeanuRng::seed_from_u64(4))
        .unwrap();
    let mut draws = draws.into_vec();
    draws.sort_unstable_by(f64::total_cmp);
    let median = draws[12_500];
    let iqr = draws[18_750] - draws[6_250];
    assert!((median - 1.2).abs() < 0.03, "{median}");
    assert!((iqr - 1.194_376_245_571_93).abs() < 0.05, "{iqr}");
}

#[test]
fn density_approaches_the_gaussian_at_large_degrees_of_freedom() {
    let distribution = LocationScaleStudentT::new(
        DoubleTensor::scalar(1e20),
        DoubleTensor::scalar(0.0),
        DoubleTensor::scalar(1.0),
    );
    let density = distribution
        .log_prob(&DoubleTensor::scalar(1.5))
        .unwrap()
        .to_scalar()
        .unwrap();
    let gaussian = -0.5 * (std::f64::consts::TAU.ln() + 1.5_f64.powi(2));
    assert!(
        (density - gaussian).abs() < 1e-14,
        "{density} vs {gaussian}"
    );
}
