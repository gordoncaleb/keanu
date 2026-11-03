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
    // recurrence ψ(x) = ψ(x + 1) - 1/x until x is large enough for the asymptotic series
    let mut shifts = 0usize;
    let mut z = x;
    let base = loop {
        if z > 0.0 && z <= 1e-5 {
            break -EULER_GAMMA - 1.0 / z;
        }
        if z >= 49.0 {
            let inv = 1.0 / (z * z);
            break z.ln() - 0.5 / z - inv * ((1.0 / 12.0) + inv * (1.0 / 120.0 - inv / 252.0));
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
    let mut shifts = 0usize;
    let mut z = x;
    let base = loop {
        if z > 0.0 && z <= 1e-5 {
            break 1.0 / (z * z);
        }
        if z >= 49.0 {
            let inv = 1.0 / (z * z);
            break 1.0 / z + inv / 2.0 + inv / z * (1.0 / 6.0 - inv * (1.0 / 30.0 + inv / 42.0));
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
