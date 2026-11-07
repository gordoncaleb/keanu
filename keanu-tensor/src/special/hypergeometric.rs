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
//! - A polynomial (a or b a non-positive integer) is summed as it is for z < 0, and for ½ < z < 1 in powers of
//!   1 − z (DLMF 15.8.7).
//!
//! Every step is written generically over the number type. On second-order dual numbers ([`Jet`]) the same
//! arithmetic gives the derivatives with respect to a, b, c and z exactly (to rounding and the series'
//! truncation), not by finite differences. [`hyp2f1`]'s value is computed in f64 only where the terms are positive
//! (0 <= z <= ½, a, b >= 0, c > 0) and otherwise in double-double arithmetic ([`DoubleDouble`], about 32 digits),
//! so that the cancellation of terms that change sign, and of the continuation, costs digits beyond f64's.

use super::Jet;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// The most terms of any one series.
const MAX_TERMS: usize = 100_000;

/// Up to this z, ₂F₁ is summed directly.
const SERIES_MAX_Z: f64 = 0.5;

/// The numbers the algorithm runs on: [`Jet`]s (plain numbers and numbers with derivatives) and [`DoubleDouble`]s
/// (plain numbers in about twice the precision of f64).
trait Real:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Add<f64, Output = Self>
    + Sub<f64, Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
{
    fn constant(v: f64) -> Self;
    /// The value, rounded to f64.
    fn value(&self) -> f64;
    fn ln(self) -> Self;
    fn exp(self) -> Self;
    /// Whether a term is negligible against a sum, to this type's precision.
    fn negligible(&self, sum: &Self) -> bool;
    /// Whether the number has no derivatives (or carries none).
    fn is_constant(&self) -> bool;
    /// self^exponent, for a positive self.
    fn pow(self, exponent: Self) -> Self {
        (self.ln() * exponent).exp()
    }
}

impl<const K: usize> Real for Jet<K> {
    fn constant(v: f64) -> Self {
        Jet::constant(v)
    }

    fn value(&self) -> f64 {
        self.v
    }

    fn ln(self) -> Self {
        Jet::ln(self)
    }

    fn exp(self) -> Self {
        Jet::exp(self)
    }

    /// Each component is at most a rounding error of the sum's own, or of the sum's value (so a derivative that is
    /// zero, or nearly, doesn't hold up convergence).
    fn negligible(&self, sum: &Self) -> bool {
        let tolerance = 0.5 * f64::EPSILON;
        let value = sum.v.abs();
        let small = |t: f64, s: f64| t.abs() <= tolerance * (s.abs() + value);
        self.v.abs() <= tolerance * value
            && (0..K).all(|i| {
                small(self.g[i], sum.g[i]) && (0..K).all(|j| small(self.h[i][j], sum.h[i][j]))
            })
    }

    fn is_constant(&self) -> bool {
        self.g.iter().all(|&g| g == 0.0) && self.h.iter().flatten().all(|&h| h == 0.0)
    }
}

/// A double-double number: the unevaluated sum of two f64s, `hi + lo` with |lo| at most half an ulp of `hi`, which
/// carries about 32 significant digits (Dekker 1971; the algorithms of Hida, Li and Bailey's QD library). The
/// series and continuation run on it where their terms can cancel, so that the cancellation costs digits beyond
/// f64's. Its logarithm and exponential are those of f64: they only form prefactors, whose relative error is not
/// amplified.
#[derive(Debug, Clone, Copy)]
struct DoubleDouble {
    hi: f64,
    lo: f64,
}

/// a + b exactly, as the rounded sum and its error.
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let v = s - a;
    (s, (a - (s - v)) + (b - v))
}

/// a + b exactly, for |a| >= |b|.
fn quick_two_sum(a: f64, b: f64) -> DoubleDouble {
    let s = a + b;
    DoubleDouble {
        hi: s,
        lo: b - (s - a),
    }
}

/// a · b exactly, as the rounded product and its error (by a fused multiply-add).
fn two_product(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

impl Add for DoubleDouble {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let (s, e) = two_sum(self.hi, other.hi);
        let (t, f) = two_sum(self.lo, other.lo);
        let r = quick_two_sum(s, e + t);
        quick_two_sum(r.hi, r.lo + f)
    }
}

impl Add<f64> for DoubleDouble {
    type Output = Self;
    fn add(self, other: f64) -> Self {
        let (s, e) = two_sum(self.hi, other);
        quick_two_sum(s, e + self.lo)
    }
}

impl Neg for DoubleDouble {
    type Output = Self;
    fn neg(self) -> Self {
        DoubleDouble {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
}

impl Sub for DoubleDouble {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        self + -other
    }
}

impl Sub<f64> for DoubleDouble {
    type Output = Self;
    fn sub(self, other: f64) -> Self {
        self + -other
    }
}

impl Mul for DoubleDouble {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        let (p, e) = two_product(self.hi, other.hi);
        quick_two_sum(p, e + (self.hi * other.lo + self.lo * other.hi))
    }
}

impl Mul<f64> for DoubleDouble {
    type Output = Self;
    fn mul(self, other: f64) -> Self {
        let (p, e) = two_product(self.hi, other);
        quick_two_sum(p, e + self.lo * other)
    }
}

impl Div for DoubleDouble {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        // long division: a first quotient, then the remainder's
        let q1 = self.hi / other.hi;
        let r = self - other * q1;
        let q2 = r.hi / other.hi;
        let r = r - other * q2;
        let q3 = r.hi / other.hi;
        quick_two_sum(q1, q2) + q3
    }
}

impl Div<f64> for DoubleDouble {
    type Output = Self;
    fn div(self, other: f64) -> Self {
        self / DoubleDouble::constant(other)
    }
}

impl Real for DoubleDouble {
    fn constant(v: f64) -> Self {
        DoubleDouble { hi: v, lo: 0.0 }
    }

    fn value(&self) -> f64 {
        self.hi + self.lo
    }

    fn ln(self) -> Self {
        Self::constant(self.value().ln())
    }

    fn exp(self) -> Self {
        Self::constant(self.value().exp())
    }

    fn negligible(&self, sum: &Self) -> bool {
        self.hi.abs() <= 1e-32 * sum.hi.abs()
    }

    fn is_constant(&self) -> bool {
        true
    }

    /// By f64's `powf`, which is accurate to about an ulp even for large exponents, unlike the exponential of
    /// the product of a rounded logarithm and the exponent.
    fn pow(self, exponent: Self) -> Self {
        Self::constant(self.value().powf(exponent.value()))
    }
}

/// The Gauss series, None if it hasn't converged within [`MAX_TERMS`] terms. Convergence isn't tested until every
/// Pochhammer symbol has passed its last sign change, where a term can be small without the rest being so.
fn series<T: Real>(a: T, b: T, c: T, z: T) -> Option<T> {
    let start = (-a.value()).max(-b.value()).max(-c.value()).max(0.0).ceil();
    let mut term = T::constant(1.0);
    let mut sum = term;
    for k in 0..MAX_TERMS {
        let k = k as f64;
        term = term * (a + k) * (b + k) / ((c + k) * (k + 1.0)) * z;
        sum = sum + term;
        if k >= start && term.negligible(&sum) {
            return Some(sum);
        }
    }
    None
}

/// The solution's value and derivative at z₀ + h, from those at z₀ (with 0 < z₀ < 1 and |h| <= (1 − z₀)/2), by
/// its Taylor series from the differential equation. The coefficients are scaled by powers of the radius of
/// convergence 1 − z₀, so that they don't overflow as z₀ nears 1.
fn taylor_step<T: Real>((a, b, c): (T, T, T), z0: T, f: T, df: T, h: T) -> Option<(T, T)> {
    // z(1 − z) = p0 + p1 t − t² and c − (a + b + 1)z = q0 + q1 t, with t = z − z₀
    let radius = -(z0 - 1.0);
    let p0 = z0 * radius;
    let p1 = -(z0 * 2.0 - 1.0);
    let q1 = -(a + b + 1.0);
    let q0 = c + q1 * z0;
    let ab = a * b;
    // the scaled coefficients gₙ = fₙ Rⁿ of F = Σ fₙ tⁿ, with R the radius, and the power (h/R)ⁿ
    let (mut g0, mut g1) = (f, df * radius);
    let ratio = h / radius;
    let mut power = T::constant(1.0);
    let mut value = f;
    let mut derivative = df;
    let mut quiet = 0;
    for n in 0..MAX_TERMS {
        let m = n as f64;
        // the coefficient recurrence p0 (n+2)(n+1) f_{n+2} + (p1 n + q0)(n+1) f_{n+1} + (q1 n − n(n−1) − ab) fₙ = 0
        let g2 = -((q0 + p1 * m) * (m + 1.0) * radius * g1
            + (q1 * m - m * (m - 1.0) - ab) * (radius * radius) * g0)
            / (p0 * ((m + 2.0) * (m + 1.0)));
        let next_power = power * ratio;
        // the terms g_{n+1} (h/R)^{n+1} of the value and (n + 2) g_{n+2} (h/R)^{n+1} / R of the derivative
        let value_term = g1 * next_power;
        let derivative_term = g2 * next_power * (m + 2.0) / radius;
        value = value + value_term;
        derivative = derivative + derivative_term;
        if value_term.negligible(&value) && derivative_term.negligible(&derivative) {
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
fn continued<T: Real>(a: T, b: T, c: T, z: T) -> Option<T> {
    let mut z0 = SERIES_MAX_Z;
    let mut f = series(a, b, c, T::constant(z0))?;
    let mut df = a * b / c * series(a + 1.0, b + 1.0, c + 1.0, T::constant(z0))?;
    loop {
        let step = 0.5 * (1.0 - z0);
        if z.value() - z0 <= step {
            return Some(taylor_step((a, b, c), T::constant(z0), f, df, z - z0)?.0);
        }
        (f, df) = taylor_step((a, b, c), T::constant(z0), f, df, T::constant(step))?;
        z0 += step;
    }
}

/// Whether x is 0 or a negative integer.
fn non_positive_integer(x: f64) -> bool {
    x <= 0.0 && x == x.floor()
}

/// ₂F₁(−m, b; c; z) by DLMF 15.8.7 for a constant non-positive integer `a` = −m, or None if `a` isn't one or the
/// new denominator parameter b − c − m + 1 is a non-positive integer above −m (making the new series' terms
/// infinite).
fn polynomial_from_one<T: Real>(a: T, b: T, c: T, z: T) -> Option<T> {
    let m = -a.value();
    if !(non_positive_integer(a.value()) && a.is_constant()) || m > MAX_TERMS as f64 {
        return None;
    }
    let d = b - c + a + 1.0;
    if non_positive_integer(d.value()) && -d.value() < m {
        return None;
    }
    let mut ratio = T::constant(1.0);
    for k in 0..m as usize {
        let k = k as f64;
        ratio = ratio * (c - b + k) / (c + k);
    }
    Some(ratio * series(a, b, d, -(z - 1.0))?)
}

/// ₂F₁ for 0 <= z < 1.
fn unit_interval<T: Real>(a: T, b: T, c: T, z: T) -> Option<T> {
    if z.value() <= SERIES_MAX_Z {
        return series(a, b, c, z);
    }
    // a polynomial in z (a or b a non-positive integer −m, held constant) is one in 1 − z too:
    // ₂F₁(−m, b; c; z) = (c − b)ₘ/(c)ₘ ₂F₁(−m, b; b − c − m + 1; 1 − z) (DLMF 15.8.7), whose terms are small near
    // z = 1, where those of the series in z are large and cancel
    for (m, other) in [(a, b), (b, a)] {
        if let Some(f) = polynomial_from_one(m, other, c, z) {
            return Some(f);
        }
    }
    // Where ₂F₁ or its Euler transform (1 − z)^{c−a−b} ₂F₁(c − a, c − b; c; z) (DLMF 15.8.1) is a polynomial,
    // sum it: the continuation would lose accuracy for these, which have no component (or only one) of one of
    // the two solutions at z = 1. (With derivatives the series doesn't terminate, and may not converge in time,
    // in which case the continuation is used after all.)
    if non_positive_integer(a.value()) || non_positive_integer(b.value()) {
        if let Some(f) = series(a, b, c, z) {
            return Some(f);
        }
    } else if non_positive_integer((c - a).value()) || non_positive_integer((c - b).value()) {
        if let Some(f) = series(c - a, c - b, c, z) {
            return Some((-(z - 1.0)).pow(c - a - b) * f);
        }
    }
    continued(a, b, c, z)
}

/// ₂F₁(a, b; c; z) on the numbers `T`; NaN outside the supported region.
fn hyp2f1_in<T: Real>(a: T, b: T, c: T, z: T) -> T {
    let (av, bv, cv, zv) = (a.value(), b.value(), c.value(), z.value());
    let finite = [av, bv, cv, zv].iter().all(|v| v.is_finite());
    if !finite || zv >= 1.0 || non_positive_integer(cv) {
        return T::constant(f64::NAN);
    }
    // a polynomial (a or b a non-positive integer, held constant) for z < 0 is summed as it is: the sum is finite,
    // and Pfaff's transformation, unless it keeps the integer parameter first, would make it an infinite series
    // near w = 1
    let polynomial = (non_positive_integer(av) && a.is_constant())
        || (non_positive_integer(bv) && b.is_constant());
    let result = if zv >= 0.0 {
        unit_interval(a, b, c, z)
    } else if polynomial {
        series(a, b, c, z)
    } else {
        // Pfaff: (1 − z)^{−p} ₂F₁(p, c − q; c; z/(z − 1)), with (p, q) = (a, b) or (b, a): the one whose new
        // parameters are non-negative (so the terms don't change sign), or else whose second is the larger
        let (p, q) = if cv - bv >= 0.0 && av >= 0.0 {
            (a, b)
        } else if (cv - av >= 0.0 && bv >= 0.0) || cv - av > cv - bv {
            (b, a)
        } else {
            (a, b)
        };
        let w = z / (z - 1.0);
        let prefactor = (-(z - 1.0)).pow(-p);
        unit_interval(p, c - q, c, w).map(|f| prefactor * f)
    };
    result.unwrap_or(T::constant(f64::NAN))
}

/// Whether f64 suffices for [`hyp2f1`]: for 0 <= z <= ½ with a, b >= 0 and c > 0, where the series' terms are
/// positive, so nothing is lost to cancellation. (For z < 0 the transformation's prefactor and the rounding of
/// z/(z − 1) cost a few ulps even then, and beyond ½ so does the continuation.)
fn f64_suffices(a: f64, b: f64, c: f64, z: f64) -> bool {
    a >= 0.0 && b >= 0.0 && c > 0.0 && (0.0..=SERIES_MAX_Z).contains(&z)
}

/// ₂F₁(a, b; c; z) with its derivatives in the jets' variables; NaN outside the supported region.
fn hyp2f1_jet<const K: usize>(a: Jet<K>, b: Jet<K>, c: Jet<K>, z: Jet<K>) -> Jet<K> {
    let result = hyp2f1_in(a, b, c, z);
    if result.v.is_nan() { Jet::NAN } else { result }
}

/// The Gauss hypergeometric function ₂F₁(a, b; c; z) = Σ (a)ₖ(b)ₖ / ((c)ₖ k!) zᵏ, for real a, b and c (c not 0 or a
/// negative integer) and real z < 1; NaN for z >= 1, non-finite arguments or such a c. Used by, for example, the
/// Pareto/NBD customer-lifetime likelihood.
///
/// Accuracy: within 1e-14 relative error for −1 <= z < 0.999 and 5e-14 for −100 <= z <= 1 − 1e-10, checked
/// against 50- and 60-digit values for |a|, |b| <= 25 and 0.3 <= c <= 26, including negative and polynomial
/// parameters whose series' terms change sign and cancel by up to 15 digits (where the arithmetic is double-double:
/// see the module documentation). Cancellation beyond double-double's precision loses accuracy; so far that has
/// been seen only for a polynomial whose parameters are within rounding of a degenerate case (a a negative integer
/// −m and b − c − m + 1 an integer above −m, as for ₂F₁(−21, 5.03; 1.03; 0.999), off by 2e-3), and large
/// parameters cost accuracy too. The cost is a series of about 50 terms per halving of 1 − z beyond ½, more where
/// |ab/c| is large, and several times that in double-double arithmetic (everywhere except 0 <= z <= ½ with a,
/// b >= 0 and c > 0).
pub fn hyp2f1(a: f64, b: f64, c: f64, z: f64) -> f64 {
    if f64_suffices(a, b, c, z) {
        let j = Jet::<0>::constant;
        hyp2f1_jet(j(a), j(b), j(c), j(z)).v
    } else {
        let d = DoubleDouble::constant;
        hyp2f1_in(d(a), d(b), d(c), d(z)).value()
    }
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
/// finite differences), on the same region, in f64 arithmetic. Accuracy, checked against 50-digit values for the
/// parameter sets of `tests/fixtures/special_extra.py` (|a|, |b| <= 25, 0.3 <= c <= 26, a few of them negative):
/// first derivatives within 1e-12 relative error for z <= 0.999 (1e-11 up to 1 − 1e-10); second derivatives within
/// 1e-11 for z <= 0.999, losing accuracy closer to 1 (1e-9 at 1 − z = 1e-6, 1e-5 at 1 − z = 1e-10), where the
/// differential equation gives F'' through a cancellation. Unlike [`hyp2f1`] it is not computed in double-double
/// arithmetic, so where the series' terms change sign and cancel (negative or polynomial parameters, c below a or
/// b) its value and derivatives lose to the cancellation what f64 can't hold: up to all their digits, as for
/// ₂F₁(−14, −8.99; 9.77; −44.3). Its `value` then differs from [`hyp2f1`]'s. It costs about thirty times as much
/// as [`hyp2f1`] in f64. Every component is NaN outside the region.
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
