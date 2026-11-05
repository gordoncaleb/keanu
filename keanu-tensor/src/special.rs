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
/// `None` if it has not converged within [`MAX_ITERATIONS`] steps.
fn continued_fraction<const K: usize>(
    b0: f64,
    db0: [f64; K],
    mut coefficient: impl FnMut(usize) -> (f64, [f64; K], f64, [f64; K]),
) -> Option<(f64, [f64; K])> {
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
            return Some((f, df));
        }
    }
    None
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
    // converges within a few hundred steps for x >= 1.5
    continued_fraction::<0>(z + 0.5, [], |n| {
        let n = n as f64;
        (-n * (2.0 * n - 1.0) / 2.0, [], z + 0.5 + 2.0 * n, [])
    })
    .map_or(f64::NAN, |(f, [])| x * FRAC_1_SQRT_PI / f)
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

/// ln Σ eᵗ over `terms`, computed stably by shifting by the largest term: -∞ for no terms or when every term is
/// -∞, +∞ if any term is +∞ (and none is NaN), and NaN if any term is NaN, as
/// [`DoubleTensor::log_sum_exp`](crate::DoubleTensor::log_sum_exp) gives for a tensor.
///
/// ```
/// use keanu_tensor::special::log_sum_exp;
///
/// assert!((log_sum_exp([1.0_f64.ln(), 3.0_f64.ln()]) - 4.0_f64.ln()).abs() < 1e-15);
/// assert_eq!(log_sum_exp([f64::NEG_INFINITY, f64::NEG_INFINITY]), f64::NEG_INFINITY);
/// assert!(log_sum_exp([f64::NAN, f64::NEG_INFINITY]).is_nan());
/// ```
pub fn log_sum_exp<I>(terms: I) -> f64
where
    I: IntoIterator<Item = f64>,
    I::IntoIter: Clone,
{
    let terms = terms.into_iter();
    let mut max = f64::NEG_INFINITY;
    for t in terms.clone() {
        if t.is_nan() {
            return f64::NAN;
        }
        max = max.max(t);
    }
    if max.is_infinite() {
        return max;
    }
    max + terms.map(|t| (t - max).exp()).sum::<f64>().ln()
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

/// xᵃ e⁻ˣ / Γ(a) as (e, m) with the value e^e m, computed for large a as √(a/2π) e^(a (ln(1 + d) - d) - δ(a))
/// with d = (x - a)/a, so that the large terms of a ln x - x - ln Γ(a) cancel analytically (as in Temme's and
/// DiDonato and Morris's methods). Its log is e + ln m, which stays finite where the value underflows.
fn gamma_prefix_parts(a: f64, x: f64) -> (f64, f64) {
    if a < 10.0 {
        (a * x.ln() - x - ln_gamma(a), 1.0)
    } else {
        // a (ln(x/a) - d) = a ln(x/a) + a - x
        let d = (x - a) / a;
        let exponent = if d.abs() <= 0.25 {
            a * ln_1p_minus(d)
        } else {
            a * (x / a).ln() + (a - x)
        };
        (
            exponent - stirling_correction(a),
            (a / (2.0 * std::f64::consts::PI)).sqrt(),
        )
    }
}

/// The smallest shape for which the incomplete gamma functions use Temme's uniform asymptotic expansion.
const TEMME_MIN_SHAPE: f64 = 100.0;
/// The largest |x - a| / a for which they do.
const TEMME_MAX_DEVIATION: f64 = 0.4;

/// Taylor coefficients in η of the first eight coefficient functions C₀(η), ..., C₇(η) of Temme's expansion
/// (DLMF 8.12.3-8.12.9), each row in increasing powers of η; computed exactly in rational arithmetic by
/// `tests/fixtures/temme.py`. With a >= 100 and |x - a| <= 0.4 a, the truncated expansion is accurate to about
/// 2e-17 relative to the smaller of P and Q (checked against mpmath).
#[allow(clippy::excessive_precision)]
const TEMME: [[f64; 21]; 8] = [
    [
        -0.3333333333333333,
        0.08333333333333333,
        -0.014814814814814815,
        0.0011574074074074073,
        0.0003527336860670194,
        -0.0001787551440329218,
        3.919263178522438e-05,
        -2.185448510679992e-06,
        -1.85406221071516e-06,
        8.296711340953087e-07,
        -1.7665952736826078e-07,
        6.707853543401498e-09,
        1.0261809784240309e-08,
        -4.382036018453353e-09,
        9.14769958223679e-10,
        -2.5514193994946248e-11,
        -5.830772132550426e-11,
        2.4361948020667415e-11,
        -5.0276692801141755e-12,
        1.1004392031956135e-13,
        3.371763262400985e-13,
    ],
    [
        -0.001851851851851852,
        -0.003472222222222222,
        0.0026455026455026454,
        -0.0009902263374485596,
        0.00020576131687242798,
        -4.018775720164609e-07,
        -1.8098550334489977e-05,
        7.64916091608111e-06,
        -1.6120900894563446e-06,
        4.647127802807434e-09,
        1.378633446915721e-07,
        -5.752545603517705e-08,
        1.1951628599778148e-08,
        -1.7543241719747647e-11,
        -1.0091543710600413e-09,
        4.162792991842583e-10,
        -8.56390702649298e-11,
        6.067215101604758e-14,
        7.1624989648114856e-12,
        -2.933186643771437e-12,
        5.996696365683689e-13,
    ],
    [
        0.004133597883597883,
        -0.0026813271604938273,
        0.0007716049382716049,
        2.0093878600823047e-06,
        -0.0001073665322636516,
        5.2923448829120125e-05,
        -1.2760635188618728e-05,
        3.423578734096138e-08,
        1.3721957309062934e-06,
        -6.298992138380055e-07,
        1.4280614206064242e-07,
        -2.0477098421990866e-10,
        -1.409252991086752e-08,
        6.228974084922022e-09,
        -1.3670488396617114e-09,
        9.428356159014678e-13,
        1.2872252400089318e-10,
        -5.5645956134363323e-11,
        1.197593554636698e-11,
        -4.1689782251838634e-15,
        -1.0940640427884595e-12,
    ],
    [
        0.0006494341563786008,
        0.00022947209362139917,
        -0.0004691894943952557,
        0.00026772063206283885,
        -7.561801671883977e-05,
        -2.396505113867297e-07,
        1.1082654115347302e-05,
        -5.6749528269915965e-06,
        1.4230900732435883e-06,
        -2.7861080291528143e-11,
        -1.6958404091930278e-07,
        8.099464905388083e-08,
        -1.9111168485973655e-08,
        2.3928620439808118e-12,
        2.0620131815488797e-09,
        -9.460496661855133e-10,
        2.1541049775774907e-10,
        -1.388823336813903e-14,
        -2.1894761681963938e-11,
        9.790998951171684e-12,
        -2.178219188018096e-12,
    ],
    [
        -0.0008618882909167117,
        0.0007840392217200666,
        -0.0002990724803031902,
        -1.4638452578843418e-06,
        6.641498215465122e-05,
        -3.968365047179435e-05,
        1.1375726970678419e-05,
        2.507497226237533e-10,
        -1.6954149536558305e-06,
        8.907507532205309e-07,
        -2.292934834000805e-07,
        2.956794137544049e-11,
        2.8865829742708783e-08,
        -1.4189739437803219e-08,
        3.4463580499464896e-09,
        -2.3024517174528067e-13,
        -3.9409233028046403e-10,
        1.86023389685045e-10,
        -4.356323005056618e-11,
        1.278600101629623e-15,
        4.67927502665792e-12,
    ],
    [
        -0.00033679855336635813,
        -6.972813758365857e-05,
        0.0002772753244959392,
        -0.00019932570516188847,
        6.797780477937208e-05,
        1.419062920643967e-07,
        -1.3594048189768693e-05,
        8.018470256334202e-06,
        -2.291481176508095e-06,
        -3.252473551298454e-10,
        3.4652846491085265e-07,
        -1.8447187191171344e-07,
        4.8240967037894184e-08,
        -1.7989466721743514e-14,
        -6.306194500013523e-09,
        3.162417628774568e-09,
        -7.840924253697429e-10,
        5.192679165254041e-15,
        9.358944242306784e-11,
        -4.513426216163278e-11,
        1.0799129993116828e-11,
    ],
    [
        0.0005313079364639922,
        -0.0005921664373536939,
        0.0002708782096718045,
        7.902353232660328e-07,
        -8.153969367561969e-05,
        5.61168275310625e-05,
        -1.8329116582843375e-05,
        -3.0796134506033047e-09,
        3.465155368803609e-06,
        -2.0291327396058603e-06,
        5.788792863149004e-07,
        2.338630673826657e-13,
        -8.828600746330484e-08,
        4.7435958880408125e-08,
        -1.2545415020710383e-08,
        8.649648858010293e-14,
        1.6846058979264062e-09,
        -8.575492823577594e-10,
        2.1598224929232125e-10,
        -7.613230520476153e-16,
        -2.6639822008536144e-11,
    ],
    [
        0.00034436760689237765,
        5.171790908260592e-05,
        -0.00033493161081142234,
        0.0002812695154763237,
        -0.00010976582244684731,
        -1.2741009095484485e-07,
        2.7744451511563645e-05,
        -1.8263488805711332e-05,
        5.7876949497350525e-06,
        4.93875893393627e-10,
        -1.0595367014026043e-06,
        6.166714376110408e-07,
        -1.7562973359060463e-07,
        -1.297447328701544e-12,
        2.695423606288966e-08,
        -1.4578352908731272e-08,
        3.887645959386175e-09,
        -3.881002251019412e-17,
        -5.327994173877286e-10,
        2.7437977643314844e-10,
        -6.995796092070568e-11,
    ],
];

/// The regularised incomplete gamma functions at one point, directly and in log space, with their derivatives
/// with respect to the shape (NaN unless asked for).
#[derive(Clone, Copy)]
struct IncGamma {
    p: f64,
    q: f64,
    log_p: f64,
    log_q: f64,
    /// ∂P/∂a.
    dp: f64,
    /// ∂ ln P/∂a.
    dlog_p: f64,
    /// ∂ ln Q/∂a.
    dlog_q: f64,
}

impl IncGamma {
    const NAN: IncGamma = IncGamma {
        p: f64::NAN,
        q: f64::NAN,
        log_p: f64::NAN,
        log_q: f64::NAN,
        dp: f64::NAN,
        dlog_p: f64::NAN,
        dlog_q: f64::NAN,
    };
}

/// Temme's uniform asymptotic expansion (DLMF 8.12.3; Temme 1979, as used by DiDonato and Morris 1986),
/// Q(a, x) = ½ erfc(η √(a/2)) + e^(-aη²/2) / √(2πa) Σₖ Cₖ(η) a⁻ᵏ with η = sign(x - a) √(2 (λ - 1 - ln λ)) and
/// λ = x/a, for a >= [`TEMME_MIN_SHAPE`] and |λ - 1| <= [`TEMME_MAX_DEVIATION`]. The smaller of P and Q is formed
/// as e^(-aη²/2) / √(2πa) (√(πa/2) erfcx(|η| √(a/2)) ± S), which keeps its relative precision (and its log stays
/// finite) far into the tails. The shape derivative differentiates the expansion analytically, with
/// ∂η/∂a = -(λ - 1) / (η a).
fn inc_gamma_temme(a: f64, x: f64, derivative: bool) -> IncGamma {
    let d = (x - a) / a;
    // λ - 1 - ln λ = η²/2
    let half_eta2 = if d.abs() <= 0.25 {
        -ln_1p_minus(d)
    } else {
        d - d.ln_1p()
    };
    let eta = (2.0 * half_eta2).sqrt().copysign(d);
    // S = Σ Cₖ(η) a⁻ᵏ, its derivative in η, and its derivative in a at fixed η
    let inv = 1.0 / a;
    let (mut s, mut s_eta, mut s_a) = (0.0, 0.0, 0.0);
    for (k, row) in TEMME.iter().enumerate().rev() {
        let (mut c, mut dc) = (0.0, 0.0);
        for &coefficient in row.iter().rev() {
            dc = dc * eta + c;
            c = c * eta + coefficient;
        }
        let power = inv.powi(k as i32);
        s += c * power;
        s_eta += dc * power;
        s_a -= k as f64 * c * power * inv;
    }
    let log_scale = -a * half_eta2 - 0.5 * (2.0 * PI * a).ln();
    let scale = log_scale.exp();
    let erfc_part = (0.5 * PI * a).sqrt() * erfcx((a * half_eta2).sqrt());
    // the smaller of P and Q is scale * bracket
    let upper = d >= 0.0;
    let bracket = if upper { erfc_part + s } else { erfc_part - s };
    let small = scale * bracket;
    let log_small = log_scale + bracket.ln();
    let large = 1.0 - small;
    let log_large = (-small).ln_1p();
    let (p, q, log_p, log_q) = if upper {
        (large, small, log_large, log_small)
    } else {
        (small, large, log_small, log_large)
    };
    if !derivative {
        return IncGamma {
            p,
            q,
            log_p,
            log_q,
            ..IncGamma::NAN
        };
    }
    // ∂Q/∂a = scale * t, where (λ - 1)/η → 1 as η → 0
    let ratio = if eta == 0.0 { 1.0 } else { d / eta };
    let t = ratio - 0.5 * eta + (d - half_eta2 - 0.5 * inv) * s - ratio * inv * s_eta + s_a;
    let dq = scale * t;
    let (dlog_p, dlog_q) = if upper {
        (-dq / p, t / bracket)
    } else {
        (-t / bracket, dq / q)
    };
    IncGamma {
        p,
        q,
        log_p,
        log_q,
        dp: -dq,
        dlog_p,
        dlog_q,
    }
}

/// The incomplete gamma functions P and Q, their logs, and (when asked for) their shape derivatives.
///
/// For a >= 100 and x within 40% of a, from Temme's uniform asymptotic expansion ([`inc_gamma_temme`]). Otherwise,
/// for x < a + 1, P comes from the series P = xᵃe⁻ˣ/Γ(a+1) Σ xⁿ / ((a+1)...(a+n)) (DLMF 8.7.1), and
/// its derivative from differentiating the series term by term, whose terms are then all of one sign. Q is
/// 1 - P, unless P > 0.9, when Q is computed directly from the series of DiDonato and Morris (1986) as
/// 1 - xᵃ/Γ(1+a) - xᵃ/Γ(1+a) a Σₙ₌₁ (-x)ⁿ / ((a+n) n!), keeping its relative precision for small a.
/// For x >= a + 1, Q comes from Legendre's continued fraction (DLMF 8.9.2), its derivative from differentiating
/// the modified Lentz recurrences, and P is 1 - Q. The logs use the prefactor xᵃe⁻ˣ/Γ(a) in log space, so they
/// stay finite where P or Q underflows.
///
/// Everything is NaN if a series or continued fraction fails to converge within [`MAX_ITERATIONS`] terms
/// (which the switch to Temme's expansion for large a near the transition is there to prevent).
///
/// Stan's `grad_reg_inc_gamma` (Stan Math, BSD-3) instead uses the DLMF 8.7.3 series and a ten-term asymptotic
/// expansion for z >= 8; the step-by-step derivatives used here are accurate to rounding over the whole range.
fn inc_gamma(a: f64, x: f64, derivative: bool) -> IncGamma {
    if a.is_nan() || x.is_nan() || a <= 0.0 || x < 0.0 || a.is_infinite() {
        return IncGamma::NAN;
    }
    if x == 0.0 {
        return IncGamma {
            p: 0.0,
            q: 1.0,
            log_p: f64::NEG_INFINITY,
            log_q: 0.0,
            dp: 0.0,
            dlog_p: f64::NAN,
            dlog_q: 0.0,
        };
    }
    if x.is_infinite() {
        return IncGamma {
            p: 1.0,
            q: 0.0,
            log_p: 0.0,
            log_q: f64::NEG_INFINITY,
            dp: 0.0,
            dlog_p: 0.0,
            dlog_q: f64::NAN,
        };
    }
    if a >= TEMME_MIN_SHAPE && ((x - a) / a).abs() <= TEMME_MAX_DEVIATION {
        return inc_gamma_temme(a, x, derivative);
    }
    let (exponent, multiplier) = gamma_prefix_parts(a, x);
    let log_prefix = exponent + multiplier.ln();
    let prefix = exponent.exp() * multiplier;
    if x < a + 1.0 {
        let front = prefix / a;
        let (mut term, mut sum) = (1.0, 1.0);
        // ∂term/∂a and its sum
        let (mut dterm, mut dsum) = (0.0f64, 0.0f64);
        let mut converged = false;
        for n in 1..=MAX_ITERATIONS {
            let an = a + n as f64;
            term *= x / an;
            dterm = dterm * x / an - term / an;
            sum += term;
            dsum += dterm;
            if term <= 1e-17 * sum && dterm.abs() <= 1e-17 * dsum.abs() {
                converged = true;
                break;
            }
        }
        if !converged {
            return IncGamma::NAN;
        }
        let p = front * sum;
        let log_p = log_prefix - a.ln() + sum.ln();
        let q = if p <= 0.9 {
            1.0 - p
        } else {
            let u = a * x.ln() - ln_gamma_1p(a);
            let (mut c, mut s) = (1.0, 0.0f64);
            let mut converged = false;
            for n in 1..=MAX_ITERATIONS {
                c *= -x / n as f64;
                let t = c / (a + n as f64);
                s += t;
                if t.abs() <= 1e-17 * s.abs() {
                    converged = true;
                    break;
                }
            }
            if !converged {
                return IncGamma::NAN;
            }
            -u.exp_m1() - u.exp() * a * s
        };
        let (dp, dlog_p, dlog_q) = if derivative {
            let dlog_p = x.ln() - digamma(a + 1.0) + dsum / sum;
            let dp = p * dlog_p;
            (dp, dlog_p, -dp / q)
        } else {
            (f64::NAN, f64::NAN, f64::NAN)
        };
        IncGamma {
            p,
            q,
            log_p,
            // 1 - P rounds away a tiny P, but ln(1 - P) keeps it
            log_q: if p <= 0.9 { (-p).ln_1p() } else { q.ln() },
            dp,
            dlog_p,
            dlog_q,
        }
    } else {
        let Some((g, [dg])) = continued_fraction(x + 1.0 - a, [-1.0], |n| {
            let n = n as f64;
            (-n * (n - a), [n], x + 2.0 * n + 1.0 - a, [-1.0])
        }) else {
            return IncGamma::NAN;
        };
        let q = prefix / g;
        let p = 1.0 - q;
        let (dp, dlog_p, dlog_q) = if derivative {
            let dlog_q = x.ln() - digamma(a) - dg / g;
            let dp = -q * dlog_q;
            (dp, dp / p, dlog_q)
        } else {
            (f64::NAN, f64::NAN, f64::NAN)
        };
        IncGamma {
            p,
            q,
            log_p: (-q).ln_1p(),
            log_q: log_prefix - g.ln(),
            dp,
            dlog_p,
            dlog_q,
        }
    }
}

/// The regularised lower incomplete gamma function P(a, x) = γ(a, x) / Γ(a) for a > 0 and x >= 0 (NaN
/// otherwise): the CDF at x of a gamma distribution with shape a and unit rate.
///
/// Accurate for large shapes too, through Temme's uniform asymptotic expansion near x = a for a >= 100. NaN in
/// the (not expected to be reachable) case that the series or continued fraction fails to converge.
pub fn gamma_p(a: f64, x: f64) -> f64 {
    inc_gamma(a, x, false).p
}

/// The regularised upper incomplete gamma function Q(a, x) = Γ(a, x) / Γ(a) = 1 - P(a, x), computed directly so
/// that it keeps its relative precision in the upper tail.
pub fn gamma_q(a: f64, x: f64) -> f64 {
    inc_gamma(a, x, false).q
}

/// ∂P(a, x)/∂a, the derivative of the regularised lower incomplete gamma function with respect to its shape
/// (∂Q/∂a is its negation).
pub fn gamma_p_da(a: f64, x: f64) -> f64 {
    inc_gamma(a, x, true).dp
}

/// ln P(a, x), the log of the regularised lower incomplete gamma function, finite and accurate far into the lower
/// tail where P itself underflows (-∞ at x = 0, NaN where [`gamma_p`] is).
pub fn log_gamma_p(a: f64, x: f64) -> f64 {
    inc_gamma(a, x, false).log_p
}

/// ln Q(a, x), the log of the regularised upper incomplete gamma function, finite and accurate far into the upper
/// tail where Q itself underflows (-∞ at x = ∞, NaN where [`gamma_q`] is).
pub fn log_gamma_q(a: f64, x: f64) -> f64 {
    inc_gamma(a, x, false).log_q
}

/// ∂ ln P(a, x)/∂a, formed as a ratio that stays finite where P and ∂P/∂a underflow (NaN at x = 0, where
/// ln P = -∞).
pub fn log_gamma_p_da(a: f64, x: f64) -> f64 {
    inc_gamma(a, x, true).dlog_p
}

/// ∂ ln Q(a, x)/∂a, formed as a ratio that stays finite where Q and ∂Q/∂a underflow (NaN at x = ∞, where
/// ln Q = -∞).
pub fn log_gamma_q_da(a: f64, x: f64) -> f64 {
    inc_gamma(a, x, true).dlog_q
}

// ---------------------------------------------------------------------------------------------------------------
// Regularised incomplete beta function

/// The regularised incomplete beta function at one point, directly and in log space, with its derivatives with
/// respect to the shapes (NaN unless asked for).
#[derive(Clone, Copy)]
struct IncBeta {
    value: f64,
    log_value: f64,
    da: f64,
    db: f64,
    /// ∂ ln I/∂a.
    dlog_a: f64,
    /// ∂ ln I/∂b.
    dlog_b: f64,
}

impl IncBeta {
    const NAN: IncBeta = IncBeta {
        value: f64::NAN,
        log_value: f64::NAN,
        da: f64::NAN,
        db: f64::NAN,
        dlog_a: f64::NAN,
        dlog_b: f64::NAN,
    };

    /// I and its derivatives from 1 - I, which is not small.
    fn complement(self) -> IncBeta {
        let value = 1.0 - self.value;
        IncBeta {
            value,
            log_value: (-self.value).ln_1p(),
            da: -self.da,
            db: -self.db,
            dlog_a: -self.da / value,
            dlog_b: -self.db / value,
        }
    }
}

/// ln(xᵃ yᵇ / B(a, b)) with y = 1 - x, given ln x and ln y. For a, b >= 10 the Stirling terms of ln B are
/// combined with the powers analytically: with x₀ = a/(a+b) and y₀ = b/(a+b), a ln(x/x₀) + b ln(y/y₀) =
/// a (ln(1 + d₁) - d₁) + b (ln(1 + d₂) - d₂) for d₁ = x/x₀ - 1 and d₂ = y/y₀ - 1 (since a d₁ + b d₂ = 0), so
/// the large terms don't cancel numerically when a and b are large and x is near x₀.
fn log_beta_prefix(a: f64, b: f64, x: f64, ln_x: f64, ln_y: f64) -> f64 {
    if a < 10.0 || b < 10.0 {
        return a * ln_x + b * ln_y - log_beta(a, b);
    }
    let total = a + b;
    let term = |shape: f64, ln_v: f64, d: f64| {
        if d.abs() <= 0.25 {
            shape * ln_1p_minus(d)
        } else {
            shape * (ln_v - (shape / total).ln()) - shape * d
        }
    };
    // x/x₀ - 1 = (x (a + b) - a) / a, and y/y₀ - 1 = (a - x (a + b)) / b with y = 1 - x
    let shift = x * total;
    let d1 = (shift - a) / a;
    let d2 = (a - shift) / b;
    // ln Γ(a + b) - ln Γ(a) - ln Γ(b) by Stirling's formula, less the (a+b)^(a+b) / (aᵃ bᵇ) absorbed above
    let stirling = 0.5 * (a * b / total).ln() - LN_SQRT_2PI + stirling_correction(total)
        - stirling_correction(a)
        - stirling_correction(b);
    term(a, ln_x, d1) + term(b, ln_y, d2) + stirling
}

/// I_x(a, b) and its log and shape derivatives from the continued fraction DLMF 8.17.22 (evaluated by modified
/// Lentz), for x at most about the mean, given ln x and ln(1 - x) for accuracy.
fn inc_beta_cf(a: f64, b: f64, x: f64, ln_x: f64, ln_y: f64, derivative: bool) -> IncBeta {
    let log_front = log_beta_prefix(a, b, x, ln_x, ln_y) - a.ln();
    let Some((g, [dga, dgb])) = continued_fraction(1.0, [0.0, 0.0], |n| {
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
    }) else {
        return IncBeta::NAN;
    };
    let log_value = log_front - g.ln();
    let value = log_value.exp();
    if !derivative {
        return IncBeta {
            value,
            log_value,
            ..IncBeta::NAN
        };
    }
    // ∂/∂a ln(xᵃ yᵇ / (a B(a, b))) = ln x - ψ(a + 1) + ψ(a + b), and likewise for b
    let shared = digamma(a + b);
    let dlog_a = ln_x - digamma(a + 1.0) + shared - dga / g;
    let dlog_b = ln_y - digamma(b) + shared - dgb / g;
    IncBeta {
        value,
        log_value,
        da: value * dlog_a,
        db: value * dlog_b,
        dlog_a,
        dlog_b,
    }
}

/// I_x(a, b), its log and (when asked for) its shape derivatives.
///
/// Uses the continued fraction for x <= (a+1)/(a+b+2) and the symmetry I_x(a, b) = 1 - I_(1-x)(b, a) beyond, so
/// the continued fraction converges quickly. The shape derivatives come from differentiating the continued fraction
/// step by step, an exact alternative to the hypergeometric series of Stan's `inc_beta_dda` / `inc_beta_ddb`
/// (Stan Math, BSD-3), which converges slowly for x near 1 and large b. For large shapes the prefactor is formed
/// without cancellation ([`log_beta_prefix`]), and the rounding errors of the continued fraction's many steps
/// near the mean then limit the relative accuracy to about 1e-11 at shapes of 10¹⁰ (1e-8 at 10¹⁴). Everything is
/// NaN if the continued fraction fails to converge within [`MAX_ITERATIONS`] steps.
fn inc_beta_all(a: f64, b: f64, x: f64, derivative: bool) -> IncBeta {
    if a.is_nan()
        || b.is_nan()
        || x.is_nan()
        || a <= 0.0
        || b <= 0.0
        || a.is_infinite()
        || b.is_infinite()
        || !(0.0..=1.0).contains(&x)
    {
        return IncBeta::NAN;
    }
    if x == 0.0 {
        return IncBeta {
            value: 0.0,
            log_value: f64::NEG_INFINITY,
            da: 0.0,
            db: 0.0,
            dlog_a: f64::NAN,
            dlog_b: f64::NAN,
        };
    }
    if x == 1.0 {
        return IncBeta {
            value: 1.0,
            log_value: 0.0,
            da: 0.0,
            db: 0.0,
            dlog_a: 0.0,
            dlog_b: 0.0,
        };
    }
    let y = 1.0 - x;
    let ln_x = x.ln();
    let ln_y = if x < 0.5 { (-x).ln_1p() } else { y.ln() };
    if x > (a + 1.0) / (a + b + 2.0) {
        let swapped = inc_beta_cf(b, a, y, ln_y, ln_x, derivative);
        IncBeta {
            da: swapped.db,
            db: swapped.da,
            ..swapped
        }
        .complement()
    } else {
        inc_beta_cf(a, b, x, ln_x, ln_y, derivative)
    }
}

/// The regularised incomplete beta function I_x(a, b) = B(x; a, b) / B(a, b) for a, b > 0 and 0 <= x <= 1 (NaN
/// otherwise): the CDF at x of a beta distribution.
///
/// Accurate to about 1e-11 relative for shapes up to 10¹⁰, degrading slowly beyond (1e-8 at 10¹⁴). NaN in the
/// (not expected to be reachable) case that its continued fraction fails to converge.
pub fn inc_beta(a: f64, b: f64, x: f64) -> f64 {
    inc_beta_all(a, b, x, false).value
}

/// ∂I_x(a, b)/∂a, the derivative of the regularised incomplete beta function with respect to a.
pub fn inc_beta_da(a: f64, b: f64, x: f64) -> f64 {
    inc_beta_all(a, b, x, true).da
}

/// ∂I_x(a, b)/∂b, the derivative of the regularised incomplete beta function with respect to b.
pub fn inc_beta_db(a: f64, b: f64, x: f64) -> f64 {
    inc_beta_all(a, b, x, true).db
}

/// ln I_x(a, b), the log of the regularised incomplete beta function, finite and accurate far into the lower tail
/// where I itself underflows (-∞ at x = 0, NaN where [`inc_beta`] is).
pub fn log_inc_beta(a: f64, b: f64, x: f64) -> f64 {
    inc_beta_all(a, b, x, false).log_value
}

/// ∂ ln I_x(a, b)/∂a, formed as a ratio that stays finite where I and ∂I/∂a underflow (NaN at x = 0).
pub fn log_inc_beta_da(a: f64, b: f64, x: f64) -> f64 {
    inc_beta_all(a, b, x, true).dlog_a
}

/// ∂ ln I_x(a, b)/∂b, formed as a ratio that stays finite where I and ∂I/∂b underflow (NaN at x = 0).
pub fn log_inc_beta_db(a: f64, b: f64, x: f64) -> f64 {
    inc_beta_all(a, b, x, true).dlog_b
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_continued_fraction_that_does_not_converge_gives_none() {
        // 1 - 1/(1 - 1/(1 - ...)) cycles through 0, ∞, 1 for ever
        let result = continued_fraction::<1>(1.0, [0.0], |_| (-1.0, [0.0], 1.0, [0.0]));
        assert!(result.is_none());
        // and one that converges: 1 + 1/(1 + 1/(1 + ...)) is the golden ratio
        let (f, _) = continued_fraction::<1>(1.0, [0.0], |_| (1.0, [0.0], 1.0, [0.0])).unwrap();
        assert!((f - (1.0 + 5f64.sqrt()) / 2.0).abs() < 1e-15);
    }

    #[test]
    fn log_sum_exp_propagates_nan_and_infinities() {
        let lse = |terms: &[f64]| log_sum_exp(terms.iter().copied());
        // a NaN term wins even where the others alone give an infinite result
        assert!(lse(&[f64::NAN, f64::NEG_INFINITY]).is_nan());
        assert!(lse(&[f64::NAN, f64::INFINITY]).is_nan());
        assert!(lse(&[0.0, f64::NAN]).is_nan());
        assert_eq!(lse(&[]), f64::NEG_INFINITY);
        assert_eq!(lse(&[f64::NEG_INFINITY; 3]), f64::NEG_INFINITY);
        assert_eq!(lse(&[f64::INFINITY, 1.0]), f64::INFINITY);
        // stable far from 0, and agreeing with the tensor reduction
        assert!((lse(&[-1000.0, -1000.0]) - (-1000.0 + std::f64::consts::LN_2)).abs() < 1e-12);
        let values = [0.3, -2.0, 5.0, f64::NEG_INFINITY];
        let tensor = crate::DoubleTensor::vector(values.to_vec()).log_sum_exp();
        assert!((lse(&values) - tensor).abs() < 1e-14);
    }
}
