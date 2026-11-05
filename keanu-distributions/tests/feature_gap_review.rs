//! Regression tests for the residual feature-gap review findings (Weibull and negative binomial edge cases,
//! ordered logistic cutpoints outside their valid region).

use keanu_distributions::*;
use keanu_tensor::{DoubleTensor, IntTensor};

fn s(v: f64) -> DoubleTensor {
    DoubleTensor::scalar(v)
}

fn close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!(
        actual == expected || (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
        "{what}: {actual} != {expected}"
    );
}

#[test]
fn negative_binomial_gradients_at_zero_mean() {
    let alpha = 2.0;
    let dist = NegativeBinomial::new(s(0.0), s(alpha));
    // k = 0: log P = -α ln(1 + μ/α), so d/dμ = -1/(1 + μ/α) = -1 and d/dα = -ln(1 + μ/α) + μ/(α + μ) = 0
    let g = dist.d_log_prob(&IntTensor::scalar(0)).unwrap();
    close(g.mu.as_slice()[0], -1.0, 1e-12, "d/dmu at k = 0");
    close(g.alpha.as_slice()[0], 0.0, 1e-12, "d/dalpha at k = 0");
    // k > 0 is impossible at μ = 0; log P ≈ k ln μ, so d/dμ → +∞, and d/dα has the finite limit
    // ψ(k + α) - ψ(α) - k/α
    let k = 3.0;
    let g = dist.d_log_prob(&IntTensor::scalar(3)).unwrap();
    assert_eq!(g.mu.as_slice()[0], f64::INFINITY);
    let digamma = keanu_tensor::special::digamma;
    close(
        g.alpha.as_slice()[0],
        digamma(k + alpha) - digamma(alpha) - k / alpha,
        1e-12,
        "d/dalpha at k = 3",
    );
    // the CDF: P(X <= k) = 1 at μ = 0. P(X <= 0) = (α / (α + μ))^α, with the same derivatives as P(0); for k >= 1,
    // 1 - P(X <= k) = O(μ^(k+1)), so both derivatives are 0
    let g = dist.d_log_cdf(&IntTensor::vector(vec![0, 1, 4])).unwrap();
    for (i, expected) in [-1.0, 0.0, 0.0].into_iter().enumerate() {
        close(g.mu.as_slice()[i], expected, 1e-9, "d log cdf / dmu");
        close(g.alpha.as_slice()[i], 0.0, 1e-9, "d log cdf / dalpha");
    }
    // the survival function is 0 at μ = 0 (log -∞): no NaN
    let g = dist.d_log_ccdf(&IntTensor::vector(vec![0, 1, 4])).unwrap();
    for d in g.mu.as_slice().iter().chain(g.alpha.as_slice()) {
        assert!(!d.is_nan(), "d log ccdf at mu = 0: {d}");
    }
}

