//! The Gauss hypergeometric function ₂F₁(a, b; c; z) for real parameters and real z < 1.
//!
//! - For 0 <= z <= ½, the Gauss series Σ (a)ₖ(b)ₖ / ((c)ₖ k!) zᵏ (DLMF 15.2.1).
//! - For ½ < z < 1, the series at z₀ = ½ gives F and F' = (ab/c) ₂F₁(a + 1, b + 1; c + 1; z₀), and the solution
//!   is continued along the real axis by Taylor series of the hypergeometric differential equation
//!   z(1 − z)F'' + (c − (a + b + 1)z)F' − abF = 0 (DLMF 15.10.1), each step going half the way to the singular
//!   point 1, so that each series converges at least as fast as 2⁻ⁿ. Unlike the connection formulas at z = 1
//!   (Abramowitz and Stegun 15.3.6), this has no special cases where c − a − b is an integer.
//! - For z < 0, Pfaff's transformation ₂F₁(a, b; c; z) = (1 − z)^{−a} ₂F₁(a, c − b; c; z/(z − 1)) (DLMF 15.8.1),
//!   with a and b swapped if that keeps the new parameters non-negative, maps z to (0, 1), where the terms of the
//!   series are all positive when the parameters are.
//!
//! Every step is written over second-order dual numbers ([`Jet`]), so the same arithmetic gives the derivatives
//! with respect to a, b, c and z exactly (to rounding and the series' truncation), not by finite differences.

use super::Jet;

/// The most terms of any one series.
const MAX_TERMS: usize = 100_000;

/// Up to this z, ₂F₁ is summed directly.
const SERIES_MAX_Z: f64 = 0.5;

/// Whether a term is negligible against a sum: each component is at most a rounding error of the sum's own, or of
/// the sum's value (so a derivative that is zero, or nearly, doesn't hold up convergence).
fn negligible<const K: usize>(term: &Jet<K>, sum: &Jet<K>) -> bool {
    let tolerance = 0.5 * f64::EPSILON;
    let value = sum.v.abs();
    let small = |t: f64, s: f64| t.abs() <= tolerance * (s.abs() + value);
    term.v.abs() <= tolerance * value
        && (0..K)
            .all(|i| small(term.g[i], sum.g[i]) && (0..K).all(|j| small(term.h[i][j], sum.h[i][j])))
}

/// The Gauss series, None if it hasn't converged within [`MAX_TERMS`] terms. Convergence isn't tested until every
/// Pochhammer symbol has passed its last sign change, where a term can be small without the rest being so.
fn series<const K: usize>(a: Jet<K>, b: Jet<K>, c: Jet<K>, z: Jet<K>) -> Option<Jet<K>> {
    let start = (-a.v).max(-b.v).max(-c.v).max(0.0).ceil();
    let mut term = Jet::constant(1.0);
    let mut sum = term;
    for k in 0..MAX_TERMS {
        let k = k as f64;
        term = term * (a + k) * (b + k) / ((c + k) * (k + 1.0)) * z;
        sum = sum + term;
        if k >= start && negligible(&term, &sum) {
            return Some(sum);
        }
    }
    None
}

/// The solution's value and derivative at z₀ + h, from those at z₀ (with 0 < z₀ < 1 and |h| <= (1 − z₀)/2), by
/// its Taylor series from the differential equation. The coefficients are scaled by powers of the radius of
/// convergence 1 − z₀, so that they don't overflow as z₀ nears 1.
fn taylor_step<const K: usize>(
    (a, b, c): (Jet<K>, Jet<K>, Jet<K>),
    z0: f64,
    f: Jet<K>,
    df: Jet<K>,
    h: Jet<K>,
) -> Option<(Jet<K>, Jet<K>)> {
    // z(1 − z) = p0 + p1 t − t² and c − (a + b + 1)z = q0 + q1 t, with t = z − z₀
    let radius = 1.0 - z0;
    let p0 = z0 * radius;
    let p1 = 1.0 - 2.0 * z0;
    let q1 = -(a + b + 1.0);
    let q0 = c + q1 * z0;
    let ab = a * b;
    // the scaled coefficients gₙ = fₙ Rⁿ of F = Σ fₙ tⁿ, with R the radius, and the power (h/R)ⁿ
    let (mut g0, mut g1) = (f, df * radius);
    let ratio = h / radius;
    let mut power = Jet::constant(1.0);
    let mut value = f;
    let mut derivative = df;
    let mut quiet = 0;
    for n in 0..MAX_TERMS {
        let m = n as f64;
        // the coefficient recurrence p0 (n+2)(n+1) f_{n+2} + (p1 n + q0)(n+1) f_{n+1} + (q1 n − n(n−1) − ab) fₙ = 0
        let g2 = -((p1 * m + q0) * (m + 1.0) * radius * g1
            + (q1 * m - m * (m - 1.0) - ab) * (radius * radius) * g0)
            / (p0 * (m + 2.0) * (m + 1.0));
        let next_power = power * ratio;
        // the terms g_{n+1} (h/R)^{n+1} of the value and (n + 2) g_{n+2} (h/R)^{n+1} / R of the derivative
        let value_term = g1 * next_power;
        let derivative_term = g2 * next_power * ((m + 2.0) / radius);
        value = value + value_term;
        derivative = derivative + derivative_term;
        if negligible(&value_term, &value) && negligible(&derivative_term, &derivative) {
            quiet += 1;
            if quiet == 2 {
                return Some((value, derivative));
            }
        } else {
            quiet = 0;
        }
        (g0, g1, power) = (g1, g2, next_power);
    }
    None
}

/// ₂F₁ for ½ < z < 1, by continuation from z₀ = ½.
fn continued<const K: usize>(a: Jet<K>, b: Jet<K>, c: Jet<K>, z: Jet<K>) -> Option<Jet<K>> {
    let mut z0 = SERIES_MAX_Z;
    let mut f = series(a, b, c, Jet::constant(z0))?;
    let mut df = a * b / c * series(a + 1.0, b + 1.0, c + 1.0, Jet::constant(z0))?;
    loop {
        let step = 0.5 * (1.0 - z0);
        if z.v - z0 <= step {
            return Some(taylor_step((a, b, c), z0, f, df, z - z0)?.0);
        }
        (f, df) = taylor_step((a, b, c), z0, f, df, Jet::constant(step))?;
        z0 += step;
    }
}

/// Whether x is 0 or a negative integer.
fn non_positive_integer(x: f64) -> bool {
    x <= 0.0 && x == x.floor()
}

/// ₂F₁ for 0 <= z < 1.
fn unit_interval<const K: usize>(a: Jet<K>, b: Jet<K>, c: Jet<K>, z: Jet<K>) -> Option<Jet<K>> {
    if z.v <= SERIES_MAX_Z {
        return series(a, b, c, z);
    }
    // Where ₂F₁ or its Euler transform (1 − z)^{c−a−b} ₂F₁(c − a, c − b; c; z) (DLMF 15.8.1) is a polynomial,
    // sum it: the continuation would lose accuracy for these, which have no component (or only one) of one of
    // the two solutions at z = 1. (With derivatives the series doesn't terminate, and may not converge in time,
    // in which case the continuation is used after all.)
    if non_positive_integer(a.v) || non_positive_integer(b.v) {
        if let Some(f) = series(a, b, c, z) {
            return Some(f);
        }
    } else if non_positive_integer(c.v - a.v) || non_positive_integer(c.v - b.v) {
        if let Some(f) = series(c - a, c - b, c, z) {
            return Some(((1.0 - z).ln() * (c - a - b)).exp() * f);
        }
    }
    continued(a, b, c, z)
}

/// ₂F₁(a, b; c; z) with its derivatives in the jets' variables; NaN outside the supported region.
fn hyp2f1_jet<const K: usize>(a: Jet<K>, b: Jet<K>, c: Jet<K>, z: Jet<K>) -> Jet<K> {
    let finite = [a.v, b.v, c.v, z.v].iter().all(|v| v.is_finite());
    if !finite || z.v >= 1.0 || non_positive_integer(c.v) {
        return Jet::NAN;
    }
    let result = if z.v >= 0.0 {
        unit_interval(a, b, c, z)
    } else {
        // Pfaff: (1 − z)^{−p} ₂F₁(p, c − q; c; z/(z − 1)), with (p, q) = (a, b) or (b, a): the one whose new
        // parameters are non-negative (so the terms don't change sign), or else whose second is the larger
        let (p, q) = if c.v - b.v >= 0.0 && a.v >= 0.0 {
            (a, b)
        } else if (c.v - a.v >= 0.0 && b.v >= 0.0) || c.v - a.v > c.v - b.v {
            (b, a)
        } else {
            (a, b)
        };
        let w = z / (z - 1.0);
        let prefactor = ((1.0 - z).ln() * -p).exp();
        unit_interval(p, c - q, c, w).map(|f| prefactor * f)
    };
    result.unwrap_or(Jet::NAN)
}

/// The Gauss hypergeometric function ₂F₁(a, b; c; z) = Σ (a)ₖ(b)ₖ / ((c)ₖ k!) zᵏ, for real a, b and c (c not 0 or a
/// negative integer) and real z < 1; NaN for z >= 1, non-finite arguments or such a c. Used by, for example, the
/// Pareto/NBD customer-lifetime likelihood.
///
/// Accuracy: within 1e-14 relative error for −1 <= z < 0.999 and 5e-14 for −100 <= z <= 1 − 1e-10, checked
/// against 50-digit values for |a|, |b| <= 25 and 0.3 <= c <= 26 (including negative and polynomial parameters).
/// Parameters for which the terms of the series (after Pfaff's transformation for z < 0) change sign many times
/// lose accuracy to cancellation, as do large parameters. The cost is a series of about 50 terms per halving of
/// 1 − z beyond ½, more where |ab/c| is large.
pub fn hyp2f1(a: f64, b: f64, c: f64, z: f64) -> f64 {
    let j = Jet::<0>::constant;
    hyp2f1_jet(j(a), j(b), j(c), j(z)).v
}

/// ∂₂F₁(a, b; c; z)/∂z = (ab/c) ₂F₁(a + 1, b + 1; c + 1; z) (DLMF 15.5.1), with the accuracy of [`hyp2f1`].
pub fn hyp2f1_dz(a: f64, b: f64, c: f64, z: f64) -> f64 {
    if c.is_finite() && c <= 0.0 && c == c.floor() {
        return f64::NAN;
    }
    a * b / c * hyp2f1(a + 1.0, b + 1.0, c + 1.0, z)
}

/// ₂F₁(a, b; c; z) with its gradient and Hessian with respect to (a, b, c, z), in that order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hyp2f1Derivatives {
    /// ₂F₁(a, b; c; z).
    pub value: f64,
    /// (∂/∂a, ∂/∂b, ∂/∂c, ∂/∂z).
    pub gradient: [f64; 4],
    /// The second derivatives, symmetric, in the same order.
    pub hessian: [[f64; 4]; 4],
}

/// ₂F₁(a, b; c; z) and its first and second derivatives with respect to a, b, c and z (the parameter derivatives
/// have no closed form), from differentiating the algorithm of [`hyp2f1`] exactly (forward-mode dual numbers, not
/// finite differences), on the same region. Accuracy, checked against 50-digit values as [`hyp2f1`]: first
/// derivatives within 1e-12 relative error for z <= 0.999 (1e-11 up to 1 − 1e-10); second derivatives within 1e-11
/// for z <= 0.999, losing accuracy closer to 1 (1e-9 at 1 − z = 1e-6, 1e-5 at 1 − z = 1e-10), where the
/// differential equation gives F'' through a cancellation. It costs about thirty times as much as [`hyp2f1`]. Every
/// component is NaN outside the region.
pub fn hyp2f1_derivatives(a: f64, b: f64, c: f64, z: f64) -> Hyp2f1Derivatives {
    let f = hyp2f1_jet(
        Jet::<4>::var(a, 0),
        Jet::var(b, 1),
        Jet::var(c, 2),
        Jet::var(z, 3),
    );
    // the dual-number arithmetic is symmetric only to rounding
    let hessian = std::array::from_fn(|i| std::array::from_fn(|j| 0.5 * (f.h[i][j] + f.h[j][i])));
    Hyp2f1Derivatives {
        value: f.v,
        gradient: f.g,
        hessian,
    }
}
