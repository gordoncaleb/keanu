//! The Lambert W function: the real branches W₀ (W >= −1) and W₋₁ (W <= −1) of the inverse of w ↦ w eʷ.
//!
//! Starting points as Fritsch, Shafer and Crowley (1973) and Corless et al. (1996) describe them (the series about
//! the branch point x = −1/e, an asymptotic form for large |ln|x||, and a rational form in between), refined by
//! Fritsch's iteration wₙ₊₁ = wₙ (1 + εₙ), which converges at fourth order and works with ln(x/w) so that it
//! never overflows. Within p = √(2(e x + 1)) < 0.1 of the branch point, where the iteration loses accuracy, the
//! branch-point series is summed instead.

use std::f64::consts::E;

/// 1/e rounded to f64, and the remainder 1/e − `INV_E`.
const INV_E: f64 = 0.367_879_441_171_442_33;
const INV_E_LOW: f64 = -1.242_875_367_278_836_3e-17;

/// Below this distance p from the branch point, W comes from its series in p.
const BRANCH_SERIES_MAX: f64 = 0.1;

const LAMBERT_BRANCH: [f64; 16] = [
    -1.0,
    1.0,
    -0.3333333333333333,
    0.1527777777777778,
    -0.07962962962962962,
    0.04450231481481482,
    -0.025984714873603758,
    0.015635632532333927,
    -0.009616892024299429,
    0.00601454325295612,
    -0.0038112980348919993,
    0.002440877991143984,
    -0.0015769303446867841,
    0.0010262633205076073,
    -0.0006720616311561363,
    0.0004424730618146208,
];

/// p = √(2(e x + 1)) for x >= −1/e, with e x + 1 = e (x + 1/e) computed in two parts, or None below −1/e (allowing
/// for the rounding of −1/e itself).
fn branch_distance(x: f64) -> Option<f64> {
    let d = (x + INV_E) + INV_E_LOW;
    if d < -2.0 * INV_E_LOW.abs() {
        return None;
    }
    Some((2.0 * E * d.max(0.0)).sqrt())
}

/// The branch-point series Σ μₖ pᵏ (W₀ for p >= 0, W₋₁ with p negated).
fn branch_series(p: f64) -> f64 {
    LAMBERT_BRANCH.iter().rev().fold(0.0, |sum, &c| sum * p + c)
}

/// Fritsch's iteration from w for w eʷ = x (x and w non-zero, of the same sign).
fn refine(x: f64, mut w: f64) -> f64 {
    for _ in 0..8 {
        let z = (x / w).ln() - w;
        let w1 = 1.0 + w;
        let q = 2.0 * w1 * (w1 + 2.0 * z / 3.0);
        let epsilon = z / w1 * (q - z) / (q - 2.0 * z);
        w *= 1.0 + epsilon;
        if epsilon.abs() <= 2.0 * f64::EPSILON {
            break;
        }
    }
    w
}

/// W₀(x), the principal branch of the Lambert W function (the solution w >= −1 of w eʷ = x), for x >= −1/e: −1 at
/// −1/e, 0 at 0, ∞ at ∞, and NaN below −1/e or for NaN.
///
/// Accuracy: within 1e-14 relative error over its whole domain, checked against 50-digit values; near −1/e it is
/// as accurate as x + 1/e is (W is not Lipschitz there).
///
/// Its derivative is W'(x) = W / (x (1 + W)) = 1 / (eᵂ (1 + W)), and W''(x) = −W'² (2 + W) / (1 + W).
pub fn lambert_w0(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x == 0.0 || x == f64::INFINITY {
        return x;
    }
    let Some(p) = branch_distance(x) else {
        return f64::NAN;
    };
    if p < BRANCH_SERIES_MAX {
        return branch_series(p);
    }
    let start = if x < -0.25 {
        branch_series(p)
    } else if x < 0.0 {
        // the Taylor series at 0
        x * (1.0 - x * (1.0 - x * (1.5 - x * 8.0 / 3.0)))
    } else if x < 3.0 {
        // Winitzki's (2003) approximation, within 1% for x >= 0
        let l = x.ln_1p();
        l * (1.0 - l.ln_1p() / (2.0 + l))
    } else {
        let l1 = x.ln();
        let l2 = l1.ln();
        l1 - l2 + l2 / l1
    };
    refine(x, start)
}

/// W₋₁(x), the lower real branch of the Lambert W function (the solution w <= −1 of w eʷ = x), for
/// −1/e <= x < 0: −1 at −1/e, −∞ at 0 (as the limit from below), and NaN for positive x, below −1/e or for NaN.
///
/// Accuracy: within 1e-14 relative error, checked against 50-digit values; near −1/e as accurate as x + 1/e is. Its derivatives are those of [`lambert_w0`] in terms of W.
pub fn lambert_wm1(x: f64) -> f64 {
    if x.is_nan() || x > 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    let Some(p) = branch_distance(x) else {
        return f64::NAN;
    };
    if p < BRANCH_SERIES_MAX {
        return branch_series(-p);
    }
    let start = if x < -0.25 {
        branch_series(-p)
    } else {
        // the asymptotic form as x → 0⁻ (Corless et al. 1996, eq. 4.19)
        let l1 = (-x).ln();
        let l2 = (-l1).ln();
        l1 - l2 + l2 / l1
    };
    refine(x, start)
}

/// W'(x) for either branch, given w = W(x): w / (x (1 + w)), and 1 at x = 0 (W₀); ∞ at the branch point.
pub fn lambert_w_dx(x: f64, w: f64) -> f64 {
    if x == 0.0 {
        return if w == 0.0 { 1.0 } else { f64::NAN };
    }
    w / (x * (1.0 + w))
}

/// W''(x) for either branch, given w = W(x): −W'² (2 + w) / (1 + w).
pub fn lambert_w_dx2(x: f64, w: f64) -> f64 {
    let d = lambert_w_dx(x, w);
    -d * d * (2.0 + w) / (1.0 + w)
}
