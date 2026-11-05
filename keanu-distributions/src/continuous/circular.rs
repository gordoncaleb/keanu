//! Distributions of angles.

use super::*;

/// Von Mises distribution on the circle with mean direction `mu` and concentration `kappa`:
/// pdf = e^(κ cos(x - μ)) / (2π I₀(κ)), periodic in x with period 2π.
///
/// The normaliser uses ln I₀(κ) directly, so the density is finite for any concentration (I₀ itself overflows
/// beyond κ ≈ 700). With κ = 0 it is uniform on the circle. Samples are angles in [-π, π).
#[derive(Debug, Clone, PartialEq)]
pub struct VonMises<'a> {
    /// The mean direction μ.
    pub mu: Param<'a>,
    /// The concentration κ (non-negative).
    pub kappa: Param<'a>,
}

/// Derivatives of [`VonMises::log_prob`].
#[derive(Debug, Clone, PartialEq)]
pub struct VonMisesGradient {
    /// With respect to x.
    pub x: DoubleTensor,
    /// With respect to `mu`.
    pub mu: DoubleTensor,
    /// With respect to `kappa`.
    pub kappa: DoubleTensor,
}

impl<'a> VonMises<'a> {
    /// Creates the distribution from its parameters.
    pub fn new(mu: impl IntoParam<'a, f64>, kappa: impl IntoParam<'a, f64>) -> Self {
        Self {
            mu: mu.into_param(),
            kappa: kappa.into_param(),
        }
    }

    /// κ cos(x - μ) - ln(2π) - ln I₀(κ) element-wise, broadcasting x with the parameters (-infinity for a
    /// negative κ).
    pub fn log_prob(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        map((x, &*self.mu, &*self.kappa), |(x, mu, kappa)| {
            if kappa < 0.0 || kappa.is_nan() {
                return f64::NEG_INFINITY;
            }
            kappa * (x - mu).cos() - LN_2PI - log_bessel_i0(kappa)
        })
    }

    /// The element-wise derivatives of [`Self::log_prob`] with respect to x and each parameter.
    pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<VonMisesGradient> {
        let [x, mu, kappa] = map_n((x, &*self.mu, &*self.kappa), |(x, mu, kappa)| {
            let (sin, cos) = (x - mu).sin_cos();
            // d ln I₀(κ) / dκ = I₁(κ) / I₀(κ)
            let ratio = (log_bessel_i1(kappa) - log_bessel_i0(kappa)).exp();
            [-kappa * sin, kappa * sin, cos - ratio]
        })?;
        Ok(VonMisesGradient { x, mu, kappa })
    }

    /// Draws a sample of the given shape (broadcast with the parameters' shapes), as angles in [-π, π). An error
    /// unless kappa is non-negative and finite.
    ///
    /// Each draw is by Best and Fisher's (1979) wrapped Cauchy rejection method, uniform for κ < 1e-8 and a
    /// wrapped normal with variance 1/κ beyond κ = 1e6, where the two agree to rounding.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        sample_each(shape, &[&self.mu, &self.kappa], |p| {
            sample_von_mises(p[0], p[1], rng)
        })
    }
}

const LN_2PI: f64 = 1.837_877_066_409_345_5;

/// The angle wrapped to [-π, π).
fn wrap_angle(x: f64) -> f64 {
    let wrapped = (x + PI).rem_euclid(2.0 * PI) - PI;
    // rounding can give exactly π
    if wrapped >= PI { -PI } else { wrapped }
}

fn sample_von_mises(mu: f64, kappa: f64, rng: &mut KeanuRng) -> Result<f64> {
    // `!(..)` also rejects NaN
    if !(kappa >= 0.0 && kappa.is_finite()) {
        return invalid_parameter(format!(
            "kappa must be non-negative and finite, got {kappa}"
        ));
    }
    if kappa < 1e-8 {
        return Ok(wrap_angle(mu + PI * (2.0 * rng.next_f64() - 1.0)));
    }
    if kappa > 1e6 {
        return Ok(wrap_angle(mu + rng.next_gaussian() / kappa.sqrt()));
    }
    // the wrapped Cauchy envelope's parameter, from its second-order expansion for small κ where the closed form
    // cancels
    let s = if kappa < 1e-5 {
        1.0 / kappa + kappa
    } else {
        let r = 1.0 + (1.0 + 4.0 * kappa * kappa).sqrt();
        let rho = (r - (2.0 * r).sqrt()) / (2.0 * kappa);
        (1.0 + rho * rho) / (2.0 * rho)
    };
    loop {
        let z = (PI * rng.next_f64()).cos();
        let w = (1.0 + s * z) / (s + z);
        let y = kappa * (s - w);
        let u = rng.next_f64();
        if y * (2.0 - y) - u > 0.0 || (y / u).ln() + 1.0 - y >= 0.0 {
            let angle = w.clamp(-1.0, 1.0).acos();
            let angle = if rng.next_f64() < 0.5 { -angle } else { angle };
            return Ok(wrap_angle(mu + angle));
        }
    }
}
