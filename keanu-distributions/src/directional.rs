//! Distributions of directions in more than two dimensions: the von Mises–Fisher distribution on the unit sphere.

// the negated comparisons (`!(kappa >= 0.0)`) are deliberate: they also catch NaN
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use crate::continuous::sample_gamma;
use crate::error::{DistributionError, Result, invalid_parameter};
use crate::event::{EventFamily, EventGradient, event_distribution};
use crate::param::{IntoParam, Param};
use crate::real::Real;
use crate::rng::KeanuRng;
use keanu_tensor::DoubleTensor;
use keanu_tensor::special::ln_gamma;
use std::f64::consts::{LN_2, PI};

/// How far a unit vector's squared length may be from 1 for [`VonMisesFisher::log_prob`] to treat it as in the
/// support.
pub const UNIT_VECTOR_TOLERANCE: f64 = 1e-8;

/// How far the mean direction's length may be from 1 before the parameters are invalid.
const MEAN_DIRECTION_TOLERANCE: f64 = 1e-6;

struct VonMisesFisherKernel;

impl EventFamily for VonMisesFisherKernel {
    const EVENT_RANK: usize = 1;

    fn validate_event(&self, event: &[usize]) -> Result<()> {
        if event[0] < 2 {
            return Err(DistributionError::InvalidValue(format!(
                "a von Mises-Fisher value needs at least two dimensions, not {}",
                event[0]
            )));
        }
        Ok(())
    }

    fn parameter_events(&self, event: &[usize]) -> Vec<Vec<usize>> {
        vec![event.to_vec(), vec![]]
    }

    fn check(&self, _: &[usize], x: &[f64], p: &[&[f64]]) -> Option<f64> {
        let (mu, kappa) = (p[0], p[1][0]);
        let length: f64 = mu.iter().map(|m| m * m).sum::<f64>().sqrt();
        if !(kappa >= 0.0 && kappa.is_finite())
            || !((length - 1.0).abs() <= MEAN_DIRECTION_TOLERANCE)
        {
            return Some(f64::NAN);
        }
        let squared: f64 = x.iter().map(|v| v * v).sum();
        (!((squared - 1.0).abs() <= UNIT_VECTOR_TOLERANCE)).then_some(f64::NEG_INFINITY)
    }

    fn gradient<S: Real>(&self, event: &[usize], x: &[S], p: &[&[S]]) -> EventGradient<S> {
        let dimension = event[0] as f64;
        let nu = dimension / 2.0 - 1.0;
        let (mu, kappa) = (p[0], p[1][0]);
        let t = x.iter().zip(mu).fold(S::zero(), |s, (&x, &m)| s + x * m);
        // ln C_p(κ) = ν ln κ − p/2 ln 2π − ln I_ν(κ), whose limit at κ = 0 is minus the log surface area
        let log_normaliser = if kappa.value() == 0.0 {
            S::constant(ln_gamma(dimension / 2.0) - LN_2 - dimension / 2.0 * PI.ln())
        } else {
            kappa.ln() * nu - kappa.log_bessel_i(nu) - dimension / 2.0 * (2.0 * PI).ln()
        };
        EventGradient {
            log_prob: kappa * t + log_normaliser,
            x: mu.iter().map(|&m| kappa * m).collect(),
            parameters: vec![
                x.iter().map(|&x| kappa * x).collect(),
                vec![t - kappa.bessel_i_ratio(nu)],
            ],
        }
    }
}

event_distribution! {
    /// The von Mises–Fisher distribution on the unit sphere in p ≥ 2 dimensions, with mean direction μ (a unit
    /// vector) and concentration κ ≥ 0, as SciPy's `vonmises_fisher`:
    ///
    /// ```text
    /// p(x) = C_p(κ) exp(κ μᵀx),  C_p(κ) = κ^(p/2 − 1) / ((2π)^(p/2) I_(p/2 − 1)(κ))
    /// ```
    ///
    /// with respect to the surface measure of the sphere, I_ν the modified Bessel function of the first kind (in
    /// logs, [`keanu_tensor::special::log_bessel_i`]). κ = 0 is the uniform distribution, and p = 2 is the
    /// [`VonMises`](crate::VonMises) of the angle of x. The mean resultant length E(μᵀx) is
    /// I_(p/2)(κ)/I_(p/2 − 1)(κ).
    ///
    /// The value and `mu` are [..., p] and `kappa` a scalar or one per vector. A value whose squared length is not
    /// within 1e-8 of 1 is impossible, and a mean direction whose length is not within 1e-6 of 1 (or a negative κ)
    /// gives NaN. The gradient with respect to x and μ is in the ambient space (κμ and κx). Draws use Wood's (1994)
    /// rejection sampler for the component along μ and a uniform direction orthogonal to it, rotated onto μ by a
    /// Householder reflection.
    VonMisesFisher, VonMisesFisherGradient,
    kernel = VonMisesFisherKernel, what = "von Mises-Fisher",
    (
        /// The mean direction μ, a unit vector [..., p].
        mu,
        /// The concentration κ >= 0.
        kappa,
    )
}

impl VonMisesFisher<'_> {
    /// Draws unit vectors of the given shape ([..., p], p >= 2). An error unless κ >= 0 and μ is a unit vector.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<DoubleTensor> {
        self.events().sample(shape, rng, |event, p, out, rng| {
            let mut probe = vec![0.0; event[0]];
            probe[0] = 1.0;
            if VonMisesFisherKernel.check(event, &probe, p).is_some() {
                return invalid_parameter(
                    "a von Mises-Fisher needs kappa >= 0 and a unit mean direction",
                );
            }
            sample_von_mises_fisher(p[0], p[1][0], out, rng)
        })
    }
}

/// One draw by Wood's (1994) algorithm, written into `out`.
fn sample_von_mises_fisher(
    mu: &[f64],
    kappa: f64,
    out: &mut [f64],
    rng: &mut KeanuRng,
) -> Result<()> {
    let p = mu.len();
    let m = (p - 1) as f64;
    // the component w along μ
    let w = if kappa == 0.0 {
        2.0 * rng.next_f64() - 1.0
    } else {
        // b = (−2κ + √(4κ² + m²)) / m, written without cancellation
        let b = m / (2.0 * kappa + (4.0 * kappa * kappa + m * m).sqrt());
        let x0 = (1.0 - b) / (1.0 + b);
        let c = kappa * x0 + m * (1.0 - x0 * x0).ln();
        loop {
            let (g1, g2) = (
                sample_gamma(1.0, m / 2.0, rng)?,
                sample_gamma(1.0, m / 2.0, rng)?,
            );
            let z = g1 / (g1 + g2);
            let w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z);
            if kappa * w + m * (1.0 - x0 * w).ln() - c >= rng.next_f64_non_zero().ln() {
                break w;
            }
        }
    };
    // a uniform direction v orthogonal to e₁, so the draw is w e₁ + √(1 − w²) v
    let mut v: Vec<f64> = (0..p - 1).map(|_| rng.next_gaussian()).collect();
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let radial = (1.0 - w * w).max(0.0).sqrt();
    for x in &mut v {
        *x *= radial / norm;
    }
    out[0] = w;
    out[1..].copy_from_slice(&v);
    // the Householder reflection taking e₁ to μ: H = I − 2uuᵀ with u = (e₁ − μ)/‖e₁ − μ‖
    let mut u: Vec<f64> = mu.iter().map(|&m| -m).collect();
    u[0] += 1.0;
    let length2: f64 = u.iter().map(|x| x * x).sum();
    if length2 > 1e-30 {
        let projection = 2.0 * out.iter().zip(&u).map(|(o, u)| o * u).sum::<f64>() / length2;
        for (o, u) in out.iter_mut().zip(&u) {
            *o -= projection * u;
        }
    }
    Ok(())
}
