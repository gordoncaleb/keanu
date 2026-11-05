//! Helpers shared by this crate's integration tests (each test file includes them with `mod common;`).
#![allow(dead_code)]

use keanu_tensor::DoubleTensor;

/// A scalar tensor.
pub fn s(value: f64) -> DoubleTensor {
    DoubleTensor::scalar(value)
}

/// A vector tensor.
pub fn v(values: &[f64]) -> DoubleTensor {
    DoubleTensor::vector(values.to_vec())
}

/// A tensor of the given shape, row-major.
pub fn t(shape: &[usize], data: &[f64]) -> DoubleTensor {
    DoubleTensor::new(shape, data.to_vec()).unwrap()
}

/// Asserts |actual − expected| ≤ tolerance (equal values, infinities included, always pass).
pub fn assert_close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!(
        actual == expected || (actual - expected).abs() <= tolerance,
        "{what}: {actual} is not within {tolerance} of {expected}"
    );
}

/// Asserts |actual − expected| ≤ tolerance · max(1, |expected|): an absolute tolerance near zero and a relative one
/// elsewhere (equal values, infinities included, always pass).
pub fn assert_close_rel(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!(
        actual == expected || (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
        "{what}: {actual} is not within {tolerance} (relative) of {expected}"
    );
}

/// [`assert_close`] with the tolerance scaled by 1 + |expected|.
pub fn assert_close_1p(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert_close(actual, expected, tolerance * (1.0 + expected.abs()), what);
}

/// [`assert_close`] for each element, after checking the lengths agree.
pub fn assert_all_close(actual: &[f64], expected: &[f64], tolerance: f64, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: lengths differ");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert_close(a, e, tolerance, &format!("{what}[{i}]"));
    }
}

/// [`assert_close_rel`] for each element, after checking the lengths agree.
pub fn assert_all_close_rel(actual: &[f64], expected: &[f64], tolerance: f64, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: lengths differ");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert_close_rel(a, e, tolerance, &format!("{what}[{i}]"));
    }
}

/// [`assert_close_1p`] for each element, after checking the lengths agree.
pub fn assert_all_close_1p(actual: &[f64], expected: &[f64], tolerance: f64, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: lengths differ");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert_close_1p(a, e, tolerance, &format!("{what}[{i}]"));
    }
}
