//! The shape-one Pólya–Gamma distribution and the alternating-series sampler.

use super::*;
use crate::util::sigmoid;
use keanu_tensor::special::{log_ndtr, log_sum_exp};
use std::f64::consts::FRAC_PI_2;

// The switch in Polson, Scott and Windle (2013), Appendix S1, on the J* = 4 PG scale.
const SWITCH: f64 = 0.64;
const RATE: f64 = PI * PI / 8.0;

derived_distribution! {
    /// The shape-one Pólya–Gamma distribution PG(1, z) on x > 0. The tilt z is real,
    /// and the law is symmetric in z. Its mean is tanh(z/2)/(2z), with limit 1/4.
    ///
    /// See [Polson, Scott and Windle (2013)](https://arxiv.org/abs/1205.0310).
    PolyaGamma1, PolyaGamma1Gradient, inputs = 2,
    (
        /// The exponential tilt.
        z,
    ),
    log_prob = polya_gamma_log_prob,
}

fn polya_gamma_log_prob<const K: usize>(x: Jet<K>, [z]: [Jet<K>; 1]) -> Jet<K> {
    if !z.v.is_finite() || x.v.is_nan() {
        return Jet::constant(f64::NAN);
    }
    let t = 4.0 * x;
    if t.v <= 0.0 || !t.v.is_finite() {
        return Jet::constant(f64::NEG_INFINITY);
    }
    let c = if z.v < 0.0 { -0.5 * z } else { 0.5 * z };
    let normalizer = LN_2 + (-2.0 * c).softplus();
    let small = t.v <= SWITCH;
    let leading = if small {
        // Combine the tilt and inverse-Gaussian exponent before rounding near c*t = 1.
        let centered = (c * t - 1.0) / t.sqrt();
        LN_2 - LN_SQRT_2PI - 1.5 * t.ln() - (0.5 * centered) * centered
    } else {
        let scaled = c * t.sqrt();
        c + FRAC_PI_2.ln() - RATE * t - (0.5 * scaled) * scaled
    };
    let mut correction = Jet::constant(0.0);
    for n in 1.. {
        let k = f64::from(n) * (f64::from(n) + 1.0);
        let exponent = if small {
            (-2.0 * k) / t
        } else {
            (-4.0 * RATE * k) * t
        };
        // At the switch the fourth correction and its first two derivatives are below roundoff.
        if exponent.v < -60.0 {
            break;
        }
        let term = (2.0 * f64::from(n) + 1.0) * exponent.exp();
        correction = if n % 2 == 0 {
            correction + term
        } else {
            correction - term
        };
    }
    normalizer + leading + correction.ln_1p()
}

fn exponential(rng: &mut KeanuRng) -> f64 {
    -rng.next_f64_non_zero().ln()
}

fn truncated_inverse_gaussian(c: f64, rng: &mut KeanuRng) -> f64 {
    if c * SWITCH < 1.0 {
        loop {
            let e = exponential(rng);
            if e * e <= 2.0 * exponential(rng) / SWITCH {
                let x = SWITCH / (1.0 + SWITCH * e).powi(2);
                if rng.next_f64() <= (-0.5 * c * c * x).exp() {
                    return x;
                }
            }
        }
    }
    let mu = 1.0 / c;
    loop {
        // Rationalize the smaller root to keep it positive when the roots are far apart.
        let u = 0.5 * mu * rng.next_gaussian().powi(2);
        let root_ratio = 1.0 + u + (u * (u + 2.0)).sqrt();
        let x = if rng.next_f64() <= 1.0 / (1.0 + 1.0 / root_ratio) {
            mu / root_ratio
        } else {
            mu * root_ratio
        };
        if x <= SWITCH {
            return x;
        }
    }
}

pub(super) fn sample_polya_gamma(z: f64, rng: &mut KeanuRng) -> f64 {
    let c = 0.5 * z.abs();
    let rate = RATE + (0.5 * c) * c;
    let log_right = FRAC_PI_2.ln() - rate.ln() - rate * SWITCH;
    let root = SWITCH.sqrt();
    let log_left = LN_2
        + log_sum_exp([
            log_ndtr(c * root - 1.0 / root) - c,
            log_ndtr(-c * root - 1.0 / root) + c,
        ]);
    let right_probability = sigmoid(log_right - log_left);

    loop {
        let x = if rng.next_f64() < right_probability {
            SWITCH + exponential(rng) / rate
        } else {
            truncated_inverse_gaussian(c, rng)
        };
        let u = rng.next_f64();
        let decay = if x <= SWITCH { 2.0 / x } else { 4.0 * RATE * x };
        // Compare against alternating lower and upper bounds relative to the leading coefficient.
        let mut sum = 1.0;
        for n in 1.. {
            let k = f64::from(n);
            let term = (2.0 * k + 1.0) * (-decay * k * (k + 1.0)).exp();
            if n % 2 == 1 {
                sum -= term;
                if u <= sum {
                    return x / 4.0;
                }
            } else {
                sum += term;
                if u > sum {
                    break;
                }
            }
        }
    }
}

impl PolyaGamma1<'_> {
    /// Samples using alternating-series rejection. The requested shape broadcasts with z.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        sample_each(shape, &[&self.z], |p| {
            let z = p[0];
            if !z.is_finite() {
                return invalid_parameter("the Pólya–Gamma tilt must be finite");
            }
            Ok(sample_polya_gamma(z, rng))
        })
    }
}
