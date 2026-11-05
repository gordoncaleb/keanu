//! Special functions.

use std::f64::consts::PI;

/// The Euler–Mascheroni constant. Defined here rather than taken from `std::f64::consts::EULER_GAMMA`, which needs
/// Rust 1.94, to keep the minimum supported Rust version lower.
#[allow(clippy::approx_constant)]
const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// Lanczos approximation coefficients (g = 607/128, 15 terms; Godfrey).
const LANCZOS_G: f64 = 607.0 / 128.0;
const LANCZOS: [f64; 15] = [
    0.999_999_999_999_997_1,
    57.156_235_665_862_92,
    -59.597_960_355_475_49,
    14.136_097_974_741_746,
    -0.491_913_816_097_620_2,
    0.339_946_499_848_118_9e-4,
    0.465_236_289_270_485_8e-4,
    -0.983_744_753_048_795_6e-4,
    0.158_088_703_224_912_5e-3,
    -0.210_264_441_724_104_9e-3,
    0.217_439_618_115_212_6e-3,
    -0.164_318_106_536_763_9e-3,
    0.844_182_239_838_527_4e-4,
    -0.261_908_384_015_814_1e-4,
    0.368_991_826_595_316_2e-5,
];

/// ln Γ(x) for x > 0, NaN for x <= 0 or NaN, and infinity at infinity.
pub fn ln_gamma(x: f64) -> f64 {
    if x.is_nan() || x <= 0.0 {
        return f64::NAN;
    }
    if x.is_infinite() {
        return f64::INFINITY;
    }
    if x < 0.5 {
        // reflection: Γ(x) Γ(1 - x) = π / sin(πx)
        return (PI / (PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let z = x - 1.0;
    let mut sum = LANCZOS[0];
    for (i, &c) in LANCZOS.iter().enumerate().skip(1) {
        sum += c / (z + i as f64);
    }
    let t = z + LANCZOS_G + 0.5;
    0.5 * (2.0 * PI).ln() + (z + 0.5) * t.ln() - t + sum.ln()
}

/// The digamma function ψ(x) = d/dx ln Γ(x).
pub fn digamma(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() {
        return x;
    }
    if x < 0.0 {
        if x == x.floor() {
            // a pole: the limit through the recurrence from the right, as for ψ(+0.0)
            return f64::NEG_INFINITY;
        }
        // reflection: ψ(1 - x) - ψ(x) = π cot(πx). cot has period 1, so reduce x to (-0.5, 0.5] first to keep
        // sin(πx) accurate for large |x|.
        let r = x - x.round();
        return digamma(1.0 - x) - PI * (PI * r).cos() / (PI * r).sin();
    }
    // recurrence ψ(x) = ψ(x + 1) - 1/x until x is large enough for the asymptotic series
    let mut shifts = 0usize;
    let mut z = x;
    let base = loop {
        if z > 0.0 && z <= 1e-5 {
            break -EULER_GAMMA - 1.0 / z;
        }
        if z >= 49.0 {
            let inv = 1.0 / (z * z);
            break z.ln() - 0.5 / z - inv * (1.0 / 12.0 - inv * (1.0 / 120.0 - inv / 252.0));
        }
        z += 1.0;
        shifts += 1;
    };
    let mut value = base;
    for k in (0..shifts).rev() {
        // use x itself for the first term so the sign of -0.0 is kept (ψ(-0.0) = +∞, the limit from the left)
        let xk = if k == 0 { x } else { x + k as f64 };
        value -= 1.0 / xk;
    }
    value
}

/// The trigamma function ψ₁(x) = d/dx ψ(x).
pub fn trigamma(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() {
        return x;
    }
    if x < 0.0 {
        if x == x.floor() {
            return f64::INFINITY;
        }
        // reflection: ψ₁(1 - x) + ψ₁(x) = π² / sin²(πx), with x reduced as in digamma
        let sin = (PI * (x - x.round())).sin();
        return PI * PI / (sin * sin) - trigamma(1.0 - x);
    }
    let mut shifts = 0usize;
    let mut z = x;
    let base = loop {
        if z > 0.0 && z <= 1e-5 {
            break 1.0 / (z * z);
        }
        if z >= 49.0 {
            let inv = 1.0 / (z * z);
            break 1.0 / z + inv / 2.0 + inv / z * (1.0 / 6.0 - inv * (1.0 / 30.0 - inv / 42.0));
        }
        z += 1.0;
        shifts += 1;
    };
    let mut value = base;
    for k in (0..shifts).rev() {
        let xk = x + k as f64;
        value += 1.0 / (xk * xk);
    }
    value
}

/// The tetragamma function ψ₂(x) = d/dx ψ₁(x), NaN at the poles.
pub fn tetragamma(x: f64) -> f64 {
    if x.is_nan() || (x <= 0.0 && x == x.floor()) {
        return f64::NAN;
    }
    if x < 0.0 {
        // reflection: ψ₂(1 - x) - ψ₂(x) = 2π³ cot(πx) / sin²(πx)
        let sin = (PI * x).sin();
        let cot = (PI * x).cos() / sin;
        return tetragamma(1.0 - x) - 2.0 * PI.powi(3) * cot / (sin * sin);
    }
    let mut result = 0.0;
    let mut z = x;
    while z < 10.0 {
        result -= 2.0 / (z * z * z);
        z += 1.0;
    }
    let z2 = 1.0 / (z * z);
    let series = -z2
        * (1.0
            + 1.0 / z
            + z2 * (0.5
                + z2 * (-1.0 / 6.0 + z2 * (1.0 / 6.0 + z2 * (-3.0 / 10.0 + z2 * (5.0 / 6.0))))));
    result + series
}

// ---------------------------------------------------------------------------------------------------------------
// Shared numerical machinery

/// 1 / √π.
const FRAC_1_SQRT_PI: f64 = 0.5 * std::f64::consts::FRAC_2_SQRT_PI;
/// √π.
const SQRT_PI: f64 = 1.772_453_850_905_516;
/// ln √(2π).
const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_8;
/// Iteration cap for the series and continued fractions; far beyond what any argument in range needs.
const MAX_ITERATIONS: usize = 1_000_000;

/// The Stirling correction δ(z) = ln Γ(z) - (z - ½) ln z + z - ½ ln 2π for z >= 10, from its asymptotic series
/// (accurate to a few ulps there).
fn stirling_correction(z: f64) -> f64 {
    let r = 1.0 / (z * z);
    (1.0 / 12.0
        - r * (1.0 / 360.0
            - r * (1.0 / 1260.0
                - r * (1.0 / 1680.0 - r * (1.0 / 1188.0 - r * (691.0 / 360_360.0 - r / 156.0))))))
        / z
}

/// ln(1 + d) - d for |d| <= 1/4, from its series -d²/2 + d³/3 - d⁴/4 + ... to avoid cancellation.
fn ln_1p_minus(d: f64) -> f64 {
    let mut power = d * d;
    let mut sum = 0.0;
    for k in 2..60 {
        let term = power / k as f64;
        sum += if k % 2 == 0 { -term } else { term };
        if term.abs() <= 1e-17 * sum.abs() {
            break;
        }
        power *= d;
    }
    sum
}

/// Coefficients (-1)ᵏ ζ(k) / k, k = 2, 3, ..., of the Maclaurin series of ln Γ(1 + a).
const LN_GAMMA_1P: [f64; 25] = [
    0.822_467_033_424_113_2,
    -0.400_685_634_386_531_43,
    0.270_580_808_427_784_54,
    -0.207_385_551_028_673_98,
    0.169_557_176_997_408_2,
    -0.144_049_896_768_846_1,
    0.125_509_669_524_743_04,
    -0.111_334_265_869_564_69,
    0.100_099_457_512_781_8,
    -0.090_954_017_145_829_04,
    0.083_353_840_546_109,
    -0.076_932_516_411_352_2,
    0.071_432_946_295_361_33,
    -0.066_668_705_882_420_46,
    0.062_500_955_141_213_04,
    -0.058_823_978_658_684_585,
    0.055_555_767_627_403_614,
    -0.052_631_679_379_616_66,
    0.050_000_047_698_101_69,
    -0.047_619_070_330_142_226,
    0.045_454_556_293_204_67,
    -0.043_478_266_053_040_26,
    0.041_666_669_150_341_21,
    -0.040_000_001_192_140_14,
    0.038_461_539_034_675_18,
];

/// ln Γ(1 + a), accurate relative to its (small) value as a → 0.
fn ln_gamma_1p(a: f64) -> f64 {
    if a.abs() > 0.2 {
        return ln_gamma(1.0 + a);
    }
    // -γa + Σ (-1)ᵏ ζ(k) aᵏ / k, by Horner
    let mut sum = 0.0;
    for &c in LN_GAMMA_1P.iter().rev() {
        sum = sum * a + c;
    }
    a * (a * sum - EULER_GAMMA)
}

/// Evaluates the continued fraction b₀ + a₁ / (b₁ + a₂ / (b₂ + ...)) by the modified Lentz algorithm (Lentz 1976;
/// Thompson and Barnett 1986), together with its derivatives with respect to `K` parameters by differentiating
/// each step (forward mode). `coefficient(n)` gives (aₙ, ∂aₙ, bₙ, ∂bₙ) for n >= 1; `b0` must be non-zero.
fn continued_fraction<const K: usize>(
    b0: f64,
    db0: [f64; K],
    mut coefficient: impl FnMut(usize) -> (f64, [f64; K], f64, [f64; K]),
) -> (f64, [f64; K]) {
    const TINY: f64 = 1e-300;
    let mut f = b0;
    let mut df = db0;
    let (mut c, mut dc) = (f, df);
    let (mut d, mut dd) = (0.0, [0.0; K]);
    for n in 1..=MAX_ITERATIONS {
        let (a, da, b, db) = coefficient(n);
        // Dₙ = 1 / (bₙ + aₙ Dₙ₋₁)
        let mut den = b + a * d;
        if den == 0.0 {
            den = TINY;
        }
        let d_next = 1.0 / den;
        // Cₙ = bₙ + aₙ / Cₙ₋₁
        let mut c_next = b + a / c;
        if c_next == 0.0 {
            c_next = TINY;
        }
        let delta = c_next * d_next;
        let mut derivatives_converged = true;
        for k in 0..K {
            let dd_next = -(db[k] + da[k] * d + a * dd[k]) * d_next * d_next;
            let dc_next = db[k] + da[k] / c - a * dc[k] / (c * c);
            // fₙ = fₙ₋₁ Cₙ Dₙ
            let df_next = df[k] * delta + f * (dc_next * d_next + c_next * dd_next);
            derivatives_converged &= (df_next - df[k]).abs() <= 4.0 * f64::EPSILON * df_next.abs();
            (dd[k], dc[k], df[k]) = (dd_next, dc_next, df_next);
        }
        (c, d) = (c_next, d_next);
        f *= delta;
        if (delta - 1.0).abs() <= f64::EPSILON && derivatives_converged {
            break;
        }
    }
    (f, df)
}

// ---------------------------------------------------------------------------------------------------------------
// Error functions

/// Below this |x| the error function is summed from its series and erfc is 1 - erf; above it erfc comes from its
/// continued fraction and erf is 1 - erfc. Either way the result keeps full relative precision.
const ERF_SWITCH: f64 = 1.5;

/// erf(x) for |x| < 1.5 from the series 2/√π e^(-x²) Σ 2ⁿ x^(2n+1) / (1·3·...·(2n+1)) (Abramowitz and Stegun
/// 7.1.6), whose terms are all of one sign.
fn erf_series(x: f64) -> f64 {
    let z = x * x;
    let mut term = x;
    let mut sum = x;
    for n in 1..100 {
        term *= 2.0 * z / (2 * n + 1) as f64;
        sum += term;
        if term.abs() <= 1e-17 * sum.abs() {
            break;
        }
    }
    std::f64::consts::FRAC_2_SQRT_PI * (-z).exp() * sum
}

/// e^(x²) erfc(x) for x >= 1.5 from the even part of Laplace's continued fraction (Abramowitz and Stegun 7.1.14):
/// erfc(x) = x e^(-x²) / √π / (x² + ½ - (1·2/4) / (x² + 5/2 - (3·4/4) / (x² + 9/2 - ...))).
fn erfcx_cf(x: f64) -> f64 {
    if x > 1e8 {
        // two terms of the asymptotic series; the third is below the rounding error
        return FRAC_1_SQRT_PI / x * (1.0 - 0.5 / (x * x));
    }
    let z = x * x;
    let (f, []) = continued_fraction::<0>(z + 0.5, [], |n| {
        let n = n as f64;
        (-n * (2.0 * n - 1.0) / 2.0, [], z + 0.5 + 2.0 * n, [])
    });
    x * FRAC_1_SQRT_PI / f
}

/// e^(-x²/2), with x² split so that its rounding error doesn't reach the exponential (as in Cody's CALERF).
fn exp_neg_half_square(x: f64) -> f64 {
    let hi = (x * 16.0).trunc() / 16.0;
    (-0.5 * hi * hi).exp() * (-0.5 * (x - hi) * (x + hi)).exp()
}

/// e^(-x²), with x² split as in [`exp_neg_half_square`].
fn exp_neg_square(x: f64) -> f64 {
    let hi = (x * 16.0).trunc() / 16.0;
    (-hi * hi).exp() * (-(x - hi) * (x + hi)).exp()
}

/// The error function erf(x) = 2/√π ∫₀ˣ e^(-t²) dt, to full relative precision.
///
/// Summed from its positive-term series for |x| < 1.5 and taken as 1 - erfc(|x|) beyond, with erfc from Laplace's
/// continued fraction.
pub fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x.abs() < ERF_SWITCH {
        erf_series(x)
    } else {
        (1.0 - erfc(x.abs())).copysign(x)
    }
}

/// The complementary error function erfc(x) = 1 - erf(x), to full relative precision including the upper tail
/// (where it underflows to 0 beyond x ≈ 27.2).
pub fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x <= -ERF_SWITCH {
        2.0 - erfc(-x)
    } else if x < ERF_SWITCH {
        1.0 - erf_series(x)
    } else if x.is_infinite() {
        0.0
    } else {
        erfcx_cf(x) * exp_neg_square(x)
    }
}

/// The scaled complementary error function erfcx(x) = e^(x²) erfc(x), which does not underflow for large x
/// (erfcx(x) ~ 1 / (x√π)); it overflows to infinity for x below about -26.6.
pub fn erfcx(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x < 0.0 {
        2.0 * (x * x).exp() - erfcx(-x)
    } else if x < ERF_SWITCH {
        (x * x).exp() * (1.0 - erf_series(x))
    } else if x.is_infinite() {
        0.0
    } else {
        erfcx_cf(x)
    }
}

/// erf⁻¹(y) for |y| <= ½ by Halley's iteration on erf(w) - y, from the first two terms of the Maclaurin series.
fn erf_inv_central(y: f64) -> f64 {
    if y == 0.0 {
        return y;
    }
    let mut w = 0.5 * SQRT_PI * y * (1.0 + std::f64::consts::PI / 12.0 * y * y);
    for _ in 0..20 {
        let f = erf(w) - y;
        // Halley: f'(w) = 2/√π e^(-w²) and f''(w) = -2w f'(w)
        let step = f / (std::f64::consts::FRAC_2_SQRT_PI * (-w * w).exp() + w * f);
        w -= step;
        if step.abs() <= 1e-16 * w.abs() {
            break;
        }
    }
    w
}

/// erfc⁻¹(q) for 0 <= q <= ½ by Newton's iteration on ln erfc(w) - ln q, which is concave and decreasing in w.
/// It starts from √(-ln q), at or beyond the root since erfc(w) <= e^(-w²), so the iterates decrease
/// monotonically to the root. Working with logarithms keeps the iteration exact for subnormal q.
fn erfc_inv_tail(q: f64) -> f64 {
    if q == 0.0 {
        return f64::INFINITY;
    }
    let ln_q = q.ln();
    let mut w = (-ln_q).sqrt();
    for _ in 0..100 {
        let scaled = erfcx(w);
        // d/dw ln erfc(w) = -2 / (√π erfcx(w))
        let step = (scaled.ln() - w * w - ln_q) * scaled * 0.5 * SQRT_PI;
        w += step;
        if step.abs() <= 1e-16 * w {
            break;
        }
    }
    w
}

/// The inverse error function: the w with erf(w) = y, for -1 <= y <= 1 (±∞ at ±1, NaN outside).
///
/// Computed by Halley's iteration on erf for |y| <= ½ and from [`erfc_inv`] of the exactly computed 1 - |y|
/// beyond, so the result has full relative precision everywhere, including as y → ±1.
pub fn erf_inv(y: f64) -> f64 {
    if y.is_nan() || y.abs() > 1.0 {
        return f64::NAN;
    }
    if y.abs() <= 0.5 {
        erf_inv_central(y)
    } else {
        erfc_inv_tail(1.0 - y.abs()).copysign(y)
    }
}

/// The inverse complementary error function: the w with erfc(w) = q, for 0 <= q <= 2 (+∞ at 0, -∞ at 2, NaN
/// outside), to full relative precision including for subnormal q.
pub fn erfc_inv(q: f64) -> f64 {
    if q.is_nan() || !(0.0..=2.0).contains(&q) {
        return f64::NAN;
    }
    if q < 0.5 {
        erfc_inv_tail(q)
    } else if q > 1.5 {
        -erfc_inv_tail(2.0 - q)
    } else {
        erf_inv_central(1.0 - q)
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Standard normal distribution functions

/// The standard normal cumulative distribution function Φ(x) = ½ erfc(-x / √2), to full relative precision in the
/// lower tail (it underflows to 0 below x ≈ -38.5).
pub fn ndtr(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    let z = -x * std::f64::consts::FRAC_1_SQRT_2;
    if z >= ERF_SWITCH {
        if z.is_infinite() {
            return 0.0;
        }
        0.5 * erfcx_cf(z) * exp_neg_half_square(x)
    } else {
        0.5 * erfc(z)
    }
}

/// ln Φ(x), the log of the standard normal cumulative distribution function. Finite and accurate far into the
/// lower tail, where Φ underflows: there ln Φ(x) = ln(½ erfcx(-x/√2)) - x²/2, with erfcx from its continued
/// fraction (which converges faster than the asymptotic series, and to full precision). In the upper half it is
/// log1p(-Φ(-x)).
pub fn log_ndtr(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    let z = -x * std::f64::consts::FRAC_1_SQRT_2;
    if z >= ERF_SWITCH {
        (0.5 * erfcx_cf(z)).ln() - 0.5 * x * x
    } else if x < 0.0 {
        (0.5 * erfc(z)).ln()
    } else {
        (-0.5 * erfc(-z)).ln_1p()
    }
}

/// The standard normal quantile function Φ⁻¹(p) = -√2 erfc⁻¹(2p) for 0 <= p <= 1: -∞ at 0, +∞ at 1 and NaN
/// outside, with full relative precision in both tails.
pub fn ndtri(p: f64) -> f64 {
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    // adding 0 turns -0.0 at p = ½ into 0.0
    -std::f64::consts::SQRT_2 * erfc_inv(2.0 * p) + 0.0
}

// ---------------------------------------------------------------------------------------------------------------
// Log-space helpers

/// softplus(x) = ln(1 + eˣ), without overflow for large x or loss of precision for very negative x.
pub fn softplus(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// logit(p) = ln(p / (1 - p)) for 0 <= p <= 1 (-∞ at 0, +∞ at 1, NaN outside), accurate near ½ as well as in
/// the tails.
pub fn logit(p: f64) -> f64 {
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if (0.25..=0.75).contains(&p) {
        // 2p - 1 is exact here
        ((2.0 * p - 1.0) / (1.0 - p)).ln_1p()
    } else {
        (p / (1.0 - p)).ln()
    }
}

/// ln(1 - eˣ) for x <= 0 (-∞ at 0, NaN for x > 0), choosing between ln(-expm1(x)) and log1p(-eˣ) as in
/// Mächler (2012), "Accurately computing log(1 − exp(−|a|))".
pub fn log1mexp(x: f64) -> f64 {
    if x.is_nan() || x > 0.0 {
        return f64::NAN;
    }
    if x > -std::f64::consts::LN_2 {
        (-x.exp_m1()).ln()
    } else {
        (-x.exp()).ln_1p()
    }
}

/// The log of the beta function, ln B(a, b) = ln Γ(a) + ln Γ(b) - ln Γ(a + b), for a, b > 0 (+∞ if either is 0,
/// NaN if either is negative).
///
/// When an argument is large the Stirling terms are combined analytically (as in Stan's `lbeta`) so that the
/// large ln Γ values don't cancel, keeping the result accurate for, say, ln B(0.5, 10¹⁰).
pub fn log_beta(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() || a < 0.0 || b < 0.0 {
        return f64::NAN;
    }
    let (x, y) = if a < b { (a, b) } else { (b, a) };
    if x == 0.0 {
        return f64::INFINITY;
    }
    if y.is_infinite() {
        return f64::NEG_INFINITY;
    }
    if y < 10.0 {
        return ln_gamma(x) + ln_gamma(y) - ln_gamma(x + y);
    }
    let corrections = stirling_correction(y) - stirling_correction(x + y);
    if x < 10.0 {
        // ln Γ(y) - ln Γ(x + y) by Stirling's formula, with (y - ½) ln(y / (x + y)) = -(y - ½) ln(1 + x/y)
        let difference = -(y - 0.5) * (x / y).ln_1p() - x * (x + y).ln() + x + corrections;
        return ln_gamma(x) + difference;
    }
    let ratio = x / (x + y);
    LN_SQRT_2PI - 0.5 * y.ln()
        + (x - 0.5) * ratio.ln()
        + y * (-ratio).ln_1p()
        + stirling_correction(x)
        + corrections
}

// ---------------------------------------------------------------------------------------------------------------
// Regularised incomplete gamma function

/// xᵃ e⁻ˣ / Γ(a), computed for large a as √(a/2π) e^(a (ln(1 + d) - d) - δ(a)) with d = (x - a)/a, so that the
/// large terms of a ln x - x - ln Γ(a) cancel analytically (as in Temme's and DiDonato and Morris's methods).
fn gamma_prefix(a: f64, x: f64) -> f64 {
    if a < 10.0 {
        (a * x.ln() - x - ln_gamma(a)).exp()
    } else {
        // a (ln(x/a) - d) = a ln(x/a) + a - x
        let d = (x - a) / a;
        let exponent = if d.abs() <= 0.25 {
            a * ln_1p_minus(d)
        } else {
            a * (x / a).ln() + (a - x)
        };
        (exponent - stirling_correction(a)).exp() * (a / (2.0 * std::f64::consts::PI)).sqrt()
    }
}

/// (P(a, x), Q(a, x), ∂P/∂a), the derivative only when asked for (NaN otherwise).
///
/// For x < a + 1, P comes from the series P = xᵃe⁻ˣ/Γ(a+1) Σ xⁿ / ((a+1)...(a+n)) (DLMF 8.7.1), and
/// its derivative from differentiating the series term by term, whose terms are then all of one sign. Q is
/// 1 - P, unless P > 0.9, when Q is computed directly from the series of DiDonato and Morris (1986) as
/// 1 - xᵃ/Γ(1+a) - xᵃ/Γ(1+a) a Σₙ₌₁ (-x)ⁿ / ((a+n) n!), keeping its relative precision for small a.
/// For x >= a + 1, Q comes from Legendre's continued fraction (DLMF 8.9.2), its derivative from differentiating
/// the modified Lentz recurrences, and P is 1 - Q.
///
/// Stan's `grad_reg_inc_gamma` (Stan Math, BSD-3) instead uses the DLMF 8.7.3 series and a ten-term asymptotic
/// expansion for z >= 8; the step-by-step derivatives used here are accurate to rounding over the whole range.
fn inc_gamma(a: f64, x: f64, derivative: bool) -> (f64, f64, f64) {
    if a.is_nan() || x.is_nan() || a <= 0.0 || x < 0.0 || a.is_infinite() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    if x == 0.0 {
        return (0.0, 1.0, 0.0);
    }
    if x.is_infinite() {
        return (1.0, 0.0, 0.0);
    }
    if x < a + 1.0 {
        let front = gamma_prefix(a, x) / a;
        let (mut term, mut sum) = (1.0, 1.0);
        // ∂term/∂a and its sum
        let (mut dterm, mut dsum) = (0.0f64, 0.0f64);
        for n in 1..=MAX_ITERATIONS {
            let an = a + n as f64;
            term *= x / an;
            dterm = dterm * x / an - term / an;
            sum += term;
            dsum += dterm;
            if term <= 1e-17 * sum && dterm.abs() <= 1e-17 * dsum.abs() {
                break;
            }
        }
        let p = front * sum;
        let dp = if derivative {
            p * (x.ln() - digamma(a + 1.0)) + front * dsum
        } else {
            f64::NAN
        };
        let q = if p <= 0.9 {
            1.0 - p
        } else {
            let u = a * x.ln() - ln_gamma_1p(a);
            let (mut c, mut s) = (1.0, 0.0f64);
            for n in 1..=MAX_ITERATIONS {
                c *= -x / n as f64;
                let t = c / (a + n as f64);
                s += t;
                if t.abs() <= 1e-17 * s.abs() {
                    break;
                }
            }
            -u.exp_m1() - u.exp() * a * s
        };
        (p, q, dp)
    } else {
        let (g, [dg]) = continued_fraction(x + 1.0 - a, [-1.0], |n| {
            let n = n as f64;
            (-n * (n - a), [n], x + 2.0 * n + 1.0 - a, [-1.0])
        });
        let q = gamma_prefix(a, x) / g;
        let dq = if derivative {
            q * (x.ln() - digamma(a)) - q * dg / g
        } else {
            f64::NAN
        };
        (1.0 - q, q, -dq)
    }
}

/// The regularised lower incomplete gamma function P(a, x) = γ(a, x) / Γ(a) for a > 0 and x >= 0 (NaN
/// otherwise): the CDF at x of a gamma distribution with shape a and unit rate.
pub fn gamma_p(a: f64, x: f64) -> f64 {
    inc_gamma(a, x, false).0
}

/// The regularised upper incomplete gamma function Q(a, x) = Γ(a, x) / Γ(a) = 1 - P(a, x), computed directly so
/// that it keeps its relative precision in the upper tail.
pub fn gamma_q(a: f64, x: f64) -> f64 {
    inc_gamma(a, x, false).1
}

/// ∂P(a, x)/∂a, the derivative of the regularised lower incomplete gamma function with respect to its shape
/// (∂Q/∂a is its negation).
pub fn gamma_p_da(a: f64, x: f64) -> f64 {
    inc_gamma(a, x, true).2
}

// ---------------------------------------------------------------------------------------------------------------
// Regularised incomplete beta function

/// (I_x(a, b), ∂I/∂a, ∂I/∂b) from the continued fraction DLMF 8.17.22 (evaluated by modified Lentz), for x at
/// most about the mean, given ln x and ln(1 - x) for accuracy.
fn inc_beta_cf(a: f64, b: f64, x: f64, ln_x: f64, ln_y: f64, derivative: bool) -> (f64, f64, f64) {
    let front = (a * ln_x + b * ln_y - log_beta(a, b)).exp() / a;
    let (g, [dga, dgb]) = continued_fraction(1.0, [0.0, 0.0], |n| {
        let m = (n / 2) as f64;
        if n % 2 == 1 {
            // d_(2m+1) = -(a+m)(a+b+m) x / ((a+2m)(a+2m+1))
            let den = (a + 2.0 * m) * (a + 2.0 * m + 1.0);
            let d = -(a + m) * (a + b + m) * x / den;
            let dda = d
                * (1.0 / (a + m) + 1.0 / (a + b + m)
                    - 1.0 / (a + 2.0 * m)
                    - 1.0 / (a + 2.0 * m + 1.0));
            let ddb = -(a + m) * x / den;
            (d, [dda, ddb], 1.0, [0.0, 0.0])
        } else {
            // d_(2m) = m(b-m) x / ((a+2m-1)(a+2m))
            let den = (a + 2.0 * m - 1.0) * (a + 2.0 * m);
            let d = m * (b - m) * x / den;
            let dda = -d * (1.0 / (a + 2.0 * m - 1.0) + 1.0 / (a + 2.0 * m));
            let ddb = m * x / den;
            (d, [dda, ddb], 1.0, [0.0, 0.0])
        }
    });
    let value = front / g;
    if !derivative {
        return (value, f64::NAN, f64::NAN);
    }
    // ∂/∂a ln(xᵃ yᵇ / (a B(a, b))) = ln x - ψ(a + 1) + ψ(a + b), and likewise for b
    let shared = digamma(a + b);
    let da = value * (ln_x - digamma(a + 1.0) + shared - dga / g);
    let db = value * (ln_y - digamma(b) + shared - dgb / g);
    (value, da, db)
}

/// (I_x(a, b), ∂I/∂a, ∂I/∂b), the derivatives only when asked for (NaN otherwise).
///
/// Uses the continued fraction for x <= (a+1)/(a+b+2) and the symmetry I_x(a, b) = 1 - I_(1-x)(b, a) beyond, so
/// the continued fraction converges quickly. The shape derivatives come from differentiating the continued fraction
/// step by step, an exact alternative to the hypergeometric series of Stan's `inc_beta_dda` / `inc_beta_ddb`
/// (Stan Math, BSD-3), which converges slowly for x near 1 and large b.
fn inc_beta_all(a: f64, b: f64, x: f64, derivative: bool) -> (f64, f64, f64) {
    if a.is_nan()
        || b.is_nan()
        || x.is_nan()
        || a <= 0.0
        || b <= 0.0
        || a.is_infinite()
        || b.is_infinite()
        || !(0.0..=1.0).contains(&x)
    {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    if x == 0.0 {
        return (0.0, 0.0, 0.0);
    }
    if x == 1.0 {
        return (1.0, 0.0, 0.0);
    }
    let y = 1.0 - x;
    let ln_x = x.ln();
    let ln_y = if x < 0.5 { (-x).ln_1p() } else { y.ln() };
    if x > (a + 1.0) / (a + b + 2.0) {
        let (v, dvb, dva) = inc_beta_cf(b, a, y, ln_y, ln_x, derivative);
        (1.0 - v, -dva, -dvb)
    } else {
        inc_beta_cf(a, b, x, ln_x, ln_y, derivative)
    }
}

/// The regularised incomplete beta function I_x(a, b) = B(x; a, b) / B(a, b) for a, b > 0 and 0 <= x <= 1 (NaN
/// otherwise): the CDF at x of a beta distribution.
pub fn inc_beta(a: f64, b: f64, x: f64) -> f64 {
    inc_beta_all(a, b, x, false).0
}

/// ∂I_x(a, b)/∂a, the derivative of the regularised incomplete beta function with respect to a.
pub fn inc_beta_da(a: f64, b: f64, x: f64) -> f64 {
    inc_beta_all(a, b, x, true).1
}

/// ∂I_x(a, b)/∂b, the derivative of the regularised incomplete beta function with respect to b.
pub fn inc_beta_db(a: f64, b: f64, x: f64) -> f64 {
    inc_beta_all(a, b, x, true).2
}

// ---------------------------------------------------------------------------------------------------------------
// Modified Bessel functions

/// ln I_ν(κ) for ν ∈ {0, 1} and κ >= 0: the power series (DLMF 10.25.2) up to κ = 30, where all its terms are
/// positive, and the Hankel asymptotic expansion (DLMF 10.40.1) beyond, which is accurate to rounding there and
/// never overflows.
fn log_bessel_i(nu: u8, kappa: f64) -> f64 {
    if kappa.is_infinite() {
        return f64::INFINITY;
    }
    let nu = f64::from(nu);
    if kappa <= 30.0 {
        // ln of the first term, (κ/2)^ν / ν!, plus ln(1 + the rest relative to it)
        let q = 0.25 * kappa * kappa;
        let mut term = 1.0;
        let mut rest = 0.0;
        for j in 1..200 {
            let j = j as f64;
            term *= q / (j * (j + nu));
            rest += term;
            if term <= 1e-17 * (1.0 + rest) {
                break;
            }
        }
        let first = if nu == 0.0 { 0.0 } else { (0.5 * kappa).ln() };
        first + rest.ln_1p()
    } else {
        let mu = 4.0 * nu * nu;
        let mut term = 1.0f64;
        let mut sum = 1.0;
        for k in 1..60 {
            let odd = (2 * k - 1) as f64;
            term *= -(mu - odd * odd) / (8.0 * k as f64 * kappa);
            sum += term;
            if term.abs() <= 1e-17 * sum {
                break;
            }
        }
        kappa - 0.5 * (2.0 * std::f64::consts::PI * kappa).ln() + sum.ln()
    }
}

/// ln I₀(κ), the log of the modified Bessel function of the first kind of order 0 (even in κ). Stable for large κ,
/// where I₀ overflows, as needed by the von Mises normaliser.
pub fn log_bessel_i0(kappa: f64) -> f64 {
    if kappa.is_nan() {
        return kappa;
    }
    log_bessel_i(0, kappa.abs())
}

/// ln I₁(κ), the log of the modified Bessel function of the first kind of order 1, for κ >= 0 (-∞ at 0, NaN for
/// κ < 0, where I₁ is negative). Stable for large κ.
pub fn log_bessel_i1(kappa: f64) -> f64 {
    if kappa.is_nan() || kappa < 0.0 {
        return f64::NAN;
    }
    log_bessel_i(1, kappa)
}
