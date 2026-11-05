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

#[test]
fn weibull_log_cdf_gradient_where_the_cdf_underflows() {
    // t = (x/β)^α underflows to 0: ln F ≈ ln t = α (ln x - ln β), so d/dx = α/x, d/dα = ln(x/β), d/dβ = -α/β
    for (a, b, x) in [(2.0, 1.0, 1e-200), (40.0, 3.0, 1e-10), (1.0, 10.0, 5e-324)] {
        let dist = Weibull::new(s(a), s(b));
        let g = dist.d_log_cdf(&s(x)).unwrap();
        let what = format!("alpha {a}, beta {b}, x {x}");
        close(g.x.as_slice()[0], a / x, 1e-9, &format!("d/dx, {what}"));
        close(
            g.alpha.as_slice()[0],
            x.ln() - f64::ln(b),
            1e-9,
            &format!("d/dalpha, {what}"),
        );
        close(g.beta.as_slice()[0], -a / b, 1e-9, &format!("d/dbeta, {what}"));
    }
}

#[test]
fn weibull_log_prob_gradient_at_and_near_zero() {
    // ln p = ln α - ln β + (α - 1) ln(x/β) - (x/β)^α:
    // d/dx = (α - 1)/x - (α/β)(x/β)^(α-1), d/dα = 1/α + ln(x/β)(1 - (x/β)^α), d/dβ = (α/β)((x/β)^α - 1)
    let expected = |a: f64, b: f64, x: f64| {
        let lz = x.ln() - b.ln();
        [
            (a - 1.0) / x - a / b * ((a - 1.0) * lz).exp(),
            1.0 / a + lz * (1.0 - (a * lz).exp()),
            a / b * ((a * lz).exp() - 1.0),
        ]
    };
    for (a, b, x) in [
        (1.0, 10.0, 5e-324),
        (2.0, 1e10, 1e-300),
        (0.5, 1e10, 1e-300),
        (3.0, 2.0, 1e-100),
    ] {
        let dist = Weibull::new(s(a), s(b));
        let g = dist.d_log_prob(&s(x)).unwrap();
        let [dx, da, db] = expected(a, b, x);
        let what = format!("alpha {a}, beta {b}, x {x}");
        close(g.x.as_slice()[0], dx, 1e-9, &format!("d/dx, {what}"));
        close(g.alpha.as_slice()[0], da, 1e-9, &format!("d/dalpha, {what}"));
        close(g.beta.as_slice()[0], db, 1e-9, &format!("d/dbeta, {what}"));
    }
    // at x = 0, the limits: d/dx is -1/β for α = 1 and ±∞ (the sign of α - 1) otherwise; d/dα → -∞ (the
    // (α - 1) ln x term); d/dβ = -α/β
    for (a, b, dx) in [
        (1.0, 2.0, -0.5),
        (2.0, 2.0, f64::INFINITY),
        (0.5, 2.0, f64::NEG_INFINITY),
    ] {
        let g = Weibull::new(s(a), s(b)).d_log_prob(&s(0.0)).unwrap();
        assert_eq!(g.x.as_slice()[0], dx, "d/dx at x = 0, alpha {a}");
        assert_eq!(g.alpha.as_slice()[0], f64::NEG_INFINITY, "d/dalpha at 0");
        close(g.beta.as_slice()[0], -a / b, 1e-12, "d/dbeta at x = 0");
    }
}

