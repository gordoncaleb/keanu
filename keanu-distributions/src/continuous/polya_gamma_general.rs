use super::polya_gamma::sample_polya_gamma;
use super::positive::sample_gamma;
use super::*;
use crate::quadrature::integrate_signed;

const POLE: f64 = -PI * PI / 4.0;

derived_distribution! {
    /// Pólya–Gamma PG(b, z) for real shape b > 0 and finite tilt z.
    ///
    /// The density and both tails use numerical inversion of the Laplace transform.
    /// See [Polson, Scott and Windle (2013)](https://arxiv.org/abs/1205.0310).
    PolyaGamma, PolyaGammaGradient, inputs = 3,
    (
        /// The positive shape.
        b,
        /// The exponential tilt.
        z,
    ),
    log_prob = density,
    log_cdf = tail,
}

fn density<const K: usize>(x: Jet<K>, [b, z]: [Jet<K>; 2]) -> Jet<K> {
    evaluate(x, b, z, None)
}

fn tail<const K: usize>(x: Jet<K>, [b, z]: [Jet<K>; 2], upper: bool) -> Jet<K> {
    evaluate(x, b, z, Some(upper))
}

fn valid(b: f64, z: f64) -> bool {
    b > 0.0 && b.is_finite() && z.is_finite()
}

// Mean and variance per unit shape after shifting the Laplace contour to q.
fn moments(q: f64) -> (f64, f64) {
    if q.abs() < 1e-3 {
        let mean = 0.25 * (1.0 + q * (-1.0 / 3.0 + q * (2.0 / 15.0 - q * 17.0 / 315.0)));
        let variance = 1.0 / 24.0 + q * (-1.0 / 30.0 + q * 17.0 / 840.0);
        return (mean, variance);
    }
    let r = q.abs().sqrt();
    let (f, difference) = if q > 0.0 {
        let t = r.tanh();
        (t, t - r * (1.0 - t * t))
    } else {
        let t = r.tan();
        (t, r * (1.0 + t * t) - t)
    };
    (0.25 * (f / r), ((difference / r) / r) / (16.0 * r))
}

fn log_cosh<const K: usize>(r: Jet<K>) -> Jet<K> {
    let r = if r.v < 0.0 { -r } else { r };
    r - LN_2 + (-2.0 * r).softplus()
}

fn real_log_cosh_sqrt(q: f64) -> f64 {
    if q >= 0.0 {
        log_cosh(Jet::<0>::constant(q.sqrt())).v
    } else {
        (-q).sqrt().cos().ln()
    }
}

/// Analytic log(cosh(sqrt(q))) in the upper half plane. The imaginary part
/// retains its winding number around each zero on the negative real axis.
fn log_cosh_sqrt<const K: usize>(real: Jet<K>, imag: f64) -> (Jet<K>, Jet<K>) {
    let modulus = real.v.hypot(imag);
    let radius = real.chain(
        modulus,
        real.v / modulus,
        (imag / modulus).powi(2) / modulus,
    );
    let (r, i) = if real.v >= 0.0 {
        let r = (0.5 * radius + 0.5 * real).sqrt();
        (r, (0.5 * imag) / r)
    } else {
        let i = (0.5 * radius - 0.5 * real).sqrt();
        ((0.5 * imag) / i, i)
    };
    let decay = (-2.0 * r).exp();
    let gap = -(-2.0 * r).exp_m1();
    let cosine = (i + std::f64::consts::FRAC_PI_2).sin();
    let projection = 2.0 * decay * cosine.square();
    let slope = -decay * (2.0 * i).sin() / (gap + projection);
    let inverse = 1.0 / (1.0 + slope.v * slope.v);
    let angle = slope.chain(slope.v.atan(), inverse, -2.0 * slope.v * inverse * inverse);
    (
        r - LN_2 + 0.5 * (gap.square() + 2.0 * projection).ln(),
        i + angle,
    )
}

fn evaluate<const K: usize>(x: Jet<K>, b: Jet<K>, z: Jet<K>, tail: Option<bool>) -> Jet<K> {
    if !valid(b.v, z.v) || x.v.is_nan() {
        return Jet::constant(f64::NAN);
    }
    if x.v <= 0.0 || x.v.is_infinite() {
        let certain = tail.is_some_and(|upper| upper == (x.v <= 0.0));
        return Jet::constant(if certain { 0.0 } else { f64::NEG_INFINITY });
    }
    let qz = (0.5 * z.v).powi(2);
    let (mean, variance) = moments(qz);
    let upper = x.v / b.v >= mean;
    let target = x.v / b.v;
    let mut lo = POLE;
    let mut hi = (0.25 / target).powi(2);
    if !hi.is_finite() || !qz.is_finite() {
        return Jet::constant(f64::NAN);
    }
    // Exponential tilting places the inversion near its saddle and keeps a far-tail
    // probability on its own scale. Move the tail contour clear of the pole at zero.
    for _ in 0..100 {
        let mid = 0.5 * lo + 0.5 * hi;
        if mid == lo || mid == hi {
            break;
        }
        if moments(mid).0 > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let Some(requested) = tail else {
        return invert(x, b, z, hi, qz, None);
    };
    let margin = 0.25 / (b.v * variance).sqrt();
    let integrate_tail = |upper: bool| {
        let q0 = if upper {
            hi.min(qz - margin.min(0.5 * (qz - POLE)))
        } else {
            hi.max(qz + margin)
        };
        invert(x, b, z, q0, qz, Some(upper))
    };
    let mut side = upper;
    let mut result = integrate_tail(side);
    // The mean only identifies the smaller tail approximately for a skewed law.
    if result.v > -LN_2 || result.v.is_nan() {
        let other = integrate_tail(!side);
        if other.v < result.v || result.v.is_nan() && other.v.is_finite() {
            side = !side;
            result = other;
        }
    }
    if requested == side {
        result
    } else {
        result.log1mexp()
    }
}

fn invert<const K: usize>(
    x: Jet<K>,
    b: Jet<K>,
    z: Jet<K>,
    q0: f64,
    qz: f64,
    tail: Option<bool>,
) -> Jet<K> {
    let shift = 2.0 * (q0 - qz);
    let scale = 1.0 / (b.v * moments(q0).1).sqrt();
    if !scale.is_finite() || scale <= 0.0 || !shift.is_finite() {
        return Jet::constant(f64::NAN);
    }
    let tilt = log_cosh(0.5 * z);
    let base = tilt.v - real_log_cosh_sqrt(q0);
    let height = shift * x.v + b.v * base;
    let delta_q = 0.25 * (z - z.v) * (z + z.v);
    let (direction, shift_scaled) = (if tail == Some(true) { -1.0 } else { 1.0 }, shift / scale);
    let area = integrate_signed(&|r| {
        // Bend the contour left without crossing a singularity. Bound its slope
        // so the real part of the exponent stays near the saddle at small times.
        let u = r / (1.0 - r);
        let bend = 0.25 * u / (1.0 + u);
        let offset = bend * u;
        let curve = scale * offset;
        let frequency = scale * u;
        let (amplitude, argument) = log_cosh_sqrt(q0 - 0.5 * curve + delta_q, 0.5 * frequency);
        let log_weight = shift * (x - x.v) + (b - b.v) * base + b * (tilt - amplitude - base)
            - curve * x
            - 2.0 * (1.0 - r).ln();
        if log_weight.v < -700.0 {
            return Jet::constant(0.0);
        }
        let phase = frequency * x - b * argument;
        let (mut real, mut imag) = (1.0, bend * (1.0 + 1.0 / (1.0 + u)));
        if tail.is_some() {
            let s = shift_scaled - offset;
            let denom = s * s + u * u;
            (real, imag) = (
                direction * (s + imag * u) / denom,
                direction * (imag * s - u) / denom,
            );
        }
        log_weight.exp() * (real * (phase + std::f64::consts::FRAC_PI_2).sin() - imag * phase.sin())
    });
    let result = height - PI.ln() + area.ln();
    if tail.is_none() {
        result + scale.ln()
    } else {
        result
    }
}

impl PolyaGamma<'_> {
    /// Samples positive real shapes. Small integer shapes sum exact PG(1, z) draws.
    /// Other shapes use 64 gamma terms plus a gamma approximation to the remaining
    /// series with its analytic mean and variance.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        sample_each(shape, &[&self.b, &self.z], |p| {
            let (b, z) = (p[0], p[1]);
            if !valid(b, z) {
                return invalid_parameter(
                    "Pólya–Gamma needs a finite positive shape and finite tilt",
                );
            }
            if b <= 16.0 && b.fract() == 0.0 {
                return Ok((0..b as usize).map(|_| sample_polya_gamma(z, rng)).sum());
            }
            let c = 0.5 * z.abs();
            let (mean, variance) = moments(c * c);
            let mut remainder_mean = mean;
            let mut remainder_variance = variance;
            let mut total = 0.0;
            for n in 0..64 {
                let root = c.hypot((f64::from(n) + 0.5) * PI);
                let scale = (0.5 / root) / root;
                remainder_mean -= scale;
                remainder_variance -= scale * scale;
                total += sample_gamma(scale, b, rng)?;
            }
            if remainder_variance > 0.0 && remainder_mean > 0.0 {
                let scale = remainder_variance / remainder_mean;
                total += sample_gamma(scale, b * (remainder_mean / scale), rng)?;
            } else {
                total += b * remainder_mean.max(0.0);
            }
            Ok(total)
        })
    }
}
