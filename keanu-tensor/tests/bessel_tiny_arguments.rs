//! ln K_ν(x) and ln I_ν(x) for x down to the least positive float, against 50-digit values (mpmath): they are
//! finite there, not ∞, −∞ or NaN.

// the reference values keep the digits they were computed to
#![allow(clippy::excessive_precision)]

use keanu_tensor::special::{log_bessel_i, log_bessel_k};

fn check(name: &str, actual: f64, expected: f64) {
    let error = (actual - expected).abs() / expected.abs().max(1.0);
    assert!(
        error <= 1e-14,
        "{name}: {actual} vs {expected} (error {error:.1e})"
    );
}

#[test]
fn log_bessel_k_is_accurate_down_to_the_least_positive_float() {
    for (nu, x, expected) in [
        (29.5, 1e-308, 21010.611930510681163),
        (0.5, 5e-324, 372.44582731333535859),
        (0.0, 5e-324, 6.6127880721788323524),
        (1e-10, 1e-200, 6.1326015222019161001),
        (1e-3, 1e-300, 6.6163099588068663509),
        (0.3, 1e-200, 138.76570054806884977),
        (0.999, 1e-160, 368.04508615552059325),
        (1.0, 1e-300, 690.77552789821370518),
        (2.0, 5e-324, 1489.5732910233224699),
        (200.0, 1e-200, 93099.273678519113918),
        (2000.0, 1e-300, 1396135.5804584210057),
        (0.5, 1e-151, 174.07096587369517661),
        (7.25, 1e-155, 2598.9143535812970349),
        (0.05, 1e-151, 19.694906831624827824),
        (0.0999, 1e-151, 36.364119626812510497),
        (0.1, 1e-151, 36.397915093440346892),
        (0.6, 1e-300, 414.58629172477346456),
        (1e-6, 5e-324, 6.6127881645735929497),
    ] {
        check(&format!("ln K({nu}, {x})"), log_bessel_k(nu, x), expected);
        check(
            &format!("ln K({}, {x})", -nu),
            log_bessel_k(-nu, x),
            expected,
        );
    }
}

#[test]
fn log_bessel_i_is_accurate_down_to_the_least_positive_float() {
    for (nu, x, expected) in [
        (30.0, 1e-309, -21440.416463820431958),
        (0.0, 5e-324, 0.0),
        (0.5, 5e-324, -372.44582731333535859),
        (1e-3, 1e-300, -0.69089228148049029533),
        (2.5, 1e-200, -1154.2263880507697796),
        (2000.0, 1e-300, -1396143.8745080611077),
        (30.0, 1e-151, -10526.163123028655474),
    ] {
        check(&format!("ln I({nu}, {x})"), log_bessel_i(nu, x), expected);
    }
}
