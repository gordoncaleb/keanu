//! The Wiener first-passage density with a start very close to the other boundary (the upper boundary's density
//! for a tiny w, the lower one's for w near 1), against 60-digit values of the large-time series (mpmath), and
//! sampling with a tiny but nonzero drift.

// the reference values keep the digits they were computed to
#![allow(clippy::excessive_precision)]

use keanu_distributions::{KeanuRng, Wiener};
use keanu_tensor::DoubleTensor;

/// (x, a, t0, w, v, the upper boundary's log density, the lower boundary's), with sv = 0.
const CASES: [(f64, f64, f64, f64, f64, f64, f64); 9] = [
    (
        0.5,
        1.0,
        0.1,
        1e-17,
        0.8,
        -38.167186206928151173,
        -38.945741641487693215,
    ),
    (
        0.5,
        1.0,
        0.1,
        1e-10,
        0.8,
        -22.049090556049831412,
        -22.827645990609373454,
    ),
    (
        0.15,
        1.0,
        0.1,
        1e-17,
        0.8,
        -41.147700544046078654,
        -35.585286703772463153,
    ),
    (
        1.2,
        1.5,
        0.2,
        1e-17,
        -0.5,
        -40.739230566741353621,
        -39.978124732076813243,
    ),
    (
        0.13,
        1.0,
        0.1,
        1e-12,
        0.3,
        -35.488893599424997216,
        -23.291472803153548399,
    ),
    (
        0.5,
        1.0,
        0.1,
        0.3,
        0.8,
        -0.61544829847419342377,
        -1.4028438015955545752,
    ),
    (
        0.12,
        1.0,
        0.1,
        0.3,
        0.8,
        -7.1039789690013655182,
        1.2487231706116101064,
    ),
    (
        0.5,
        1.0,
        0.1,
        0.999999,
        0.8,
        -13.617304818526132366,
        -14.438749383966484581,
    ),
    (
        0.12,
        1.0,
        0.1,
        0.9999999,
        0.8,
        -11.17539959654737858,
        -32.390432117872822086,
    ),
];

#[test]
fn the_density_keeps_its_precision_for_a_start_next_to_the_other_boundary() {
    for (x, a, t0, w, v, upper, lower) in CASES {
        for (boundary, expected) in [(1, upper), (0, lower)] {
            let (value, gradient) = Wiener::log_prob_and_gradient(x, [a, t0, w, v, 0.0], boundary);
            let error = (value - expected).abs() / expected.abs().max(1.0);
            assert!(
                error <= 1e-12,
                "x = {x}, a = {a}, w = {w}, v = {v}, boundary {boundary}: {value} vs {expected} ({error:.1e})"
            );
            assert!(gradient.iter().all(|g| g.is_finite()), "{gradient:?}");
            // the second-order path agrees
            let derivatives = Wiener::log_prob_derivatives(x, [a, t0, w, v, 0.0], boundary);
            assert!(
                (derivatives.value - value).abs() <= 1e-12 * value.abs().max(1.0),
                "{} vs {value}",
                derivatives.value
            );
        }
    }
}

#[test]
fn sampling_works_with_a_tiny_nonzero_drift() {
    let s = DoubleTensor::scalar;
    let mean_time = |v: f64, boundary: i64| {
        let d = Wiener::new(
            s(1.0),
            s(0.0),
            s(0.4),
            s(v),
            s(0.0),
            keanu_tensor::IntTensor::scalar(boundary),
        );
        let draws = d
            .sample(&[20_000], &mut KeanuRng::seed_from_u64(5))
            .unwrap();
        assert!(draws.as_slice().iter().all(|t| t.is_finite() && *t > 0.0));
        draws.as_slice().iter().sum::<f64>() / 20_000.0
    };
    for boundary in [0, 1] {
        let (tiny, zero) = (mean_time(1e-200, boundary), mean_time(0.0, boundary));
        // the same distribution: within a few standard errors (the times' sd is below their mean)
        assert!(
            (tiny - zero).abs() < 0.05 * zero,
            "boundary {boundary}: mean {tiny} with a tiny drift, {zero} without"
        );
    }
}
