//! Owen's T function T(h, a) = (1/2π) ∫₀ᵃ exp(−½h²(1 + x²)) / (1 + x²) dx.
//!
//! Following Patefield and Tandy (2000), the arguments are first reduced to h >= 0 and 0 <= a <= 1 by symmetry and
//! the identity T(h, a) + T(ah, 1/a) = ½Φ(h) + ½Φ(ah) − Φ(h)Φ(ah) for a > 1, evaluated with Φ − ½ for small h and
//! with the upper tail 1 − Φ otherwise (their §2), so that no cancellation occurs. Patefield and Tandy then choose
//! among six series and quadrature methods by region; here the remaining integral, which is smooth on the reduced
//! region, is evaluated by one 32-point Gauss–Legendre rule instead: in x for h <= 1, and in y = h x over
//! [0, min(ah, 9.5)] for h > 1 (the rest of the integral is below e^{−45} of it). Its poles at x = ±i (y = ±ih) lie
//! at least the interval's length from it, so the rule converges to rounding.

use super::{erf, ndtr};
use std::f64::consts::{FRAC_1_SQRT_2, PI};

/// The positive nodes and their weights of the 32-point Gauss–Legendre rule on [−1, 1] (the rule is symmetric).
const GAUSS_LEGENDRE_32: [(f64, f64); 16] = [
    (0.9972638618494816, 0.007018610009470096),
    (0.9856115115452684, 0.01627439473090567),
    (0.9647622555875064, 0.02539206530926206),
    (0.9349060759377397, 0.03427386291302143),
    (0.8963211557660521, 0.04283589802222668),
    (0.84936761373257, 0.050998059262376175),
    (0.7944837959679424, 0.058684093478535544),
    (0.7321821187402897, 0.06582222277636185),
    (0.6630442669302152, 0.0723457941088485),
    (0.5877157572407623, 0.07819389578707031),
    (0.5068999089322294, 0.08331192422694675),
    (0.42135127613063533, 0.08765209300440381),
    (0.33186860228212767, 0.09117387869576389),
    (0.23928736225213706, 0.09384439908080457),
    (0.1444719615827965, 0.09563872007927486),
    (0.04830766568773832, 0.0965400885147278),
];

/// The argument h beyond which T(h, a) is below the smallest f64 for every a.
const OWENS_T_MAX_H: f64 = 40.0;

/// The upper limit of the integral in y = h x: the rest is below e^{−45} of it.
const OWENS_T_MAX_Y: f64 = 9.5;

/// ∫₀ᴸ f(t) dt by the 32-point Gauss–Legendre rule.
fn gauss_legendre(length: f64, f: impl Fn(f64) -> f64) -> f64 {
    let half = 0.5 * length;
    let sum: f64 = GAUSS_LEGENDRE_32
        .iter()
        .map(|&(x, w)| w * (f(half * (1.0 - x)) + f(half * (1.0 + x))))
        .sum();
    sum * half
}

/// T(h, a) for h >= 0 and 0 <= a <= 1.
fn reduced(h: f64, a: f64) -> f64 {
    if a == 0.0 || h >= OWENS_T_MAX_H {
        return 0.0;
    }
    if h == 0.0 {
        return a.atan() / (2.0 * PI);
    }
    let q = 0.5 * h * h;
    if h <= 1.0 {
        let integral = gauss_legendre(a, |x| (-q * x * x).exp() / (1.0 + x * x));
        (-q).exp() / (2.0 * PI) * integral
    } else {
        let upper = (a * h).min(OWENS_T_MAX_Y);
        let integral = gauss_legendre(upper, |y| {
            let r = y / h;
            (-0.5 * y * y).exp() / (1.0 + r * r)
        });
        (-q).exp() / (2.0 * PI * h) * integral
    }
}

/// Owen's T function T(h, a) = (1/2π) ∫₀ᵃ exp(−½h²(1 + x²)) / (1 + x²) dx, for real h and a (including ±∞):
/// the probability that X > h and 0 < Y < aX for independent standard normals X and Y, used by the skew normal's
/// CDF. Even in h and odd in a; T(0, a) = arctan(a)/2π, T(h, 1) = ½Φ(h)(1 − Φ(h)) and T(h, ∞) = ½(1 − Φ(|h|)).
///
/// Accuracy: within 2e-14 relative error wherever it is above 1e-300 (see the module notes), checked against
/// 50-digit values for |h| up to 35 and |a| from 1e-10 to 1e5.
pub fn owens_t(h: f64, a: f64) -> f64 {
    if h.is_nan() || a.is_nan() {
        return f64::NAN;
    }
    if a < 0.0 {
        return -owens_t(h, -a);
    }
    let h = h.abs();
    if a.is_infinite() {
        return 0.5 * ndtr(-h);
    }
    if a <= 1.0 {
        return reduced(h, a);
    }
    // T(h, a) = ½Φ(h) + ½Φ(ah) − Φ(h)Φ(ah) − T(ah, 1/a)
    let ah = a * h;
    let rest = reduced(ah, 1.0 / a);
    if h <= 0.67 {
        // with Φ − ½ = ½ erf(·/√2): ¼ − (Φ(h) − ½)(Φ(ah) − ½)
        let (u, v) = (0.5 * erf(h * FRAC_1_SQRT_2), 0.5 * erf(ah * FRAC_1_SQRT_2));
        0.25 - u * v - rest
    } else {
        // with the upper tails c = 1 − Φ: ½(c(h) + c(ah)) − c(h)c(ah)
        let (ch, cah) = (ndtr(-h), ndtr(-ah));
        0.5 * (ch + cah) - ch * cah - rest
    }
}

/// ∂T(h, a)/∂h = −½ φ(h) erf(ah/√2), with φ the standard normal density.
pub fn owens_t_dh(h: f64, a: f64) -> f64 {
    let phi = (-0.5 * h * h).exp() / (2.0 * PI).sqrt();
    if a.is_infinite() && h == 0.0 {
        return -0.5 * phi * a.signum();
    }
    -0.5 * phi * erf(a * h * FRAC_1_SQRT_2)
}

/// ∂T(h, a)/∂a = exp(−½h²(1 + a²)) / (2π(1 + a²)).
pub fn owens_t_da(h: f64, a: f64) -> f64 {
    let s = 1.0 + a * a;
    (-0.5 * h * h * s).exp() / (2.0 * PI * s)
}
