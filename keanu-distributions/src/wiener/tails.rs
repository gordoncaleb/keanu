//! Joint first-passage time tails, integrating the positive density in log space.

use super::{Wiener, WienerGradient, log_density, valid};
use crate::Result;
use crate::jet::{self, ElementDerivatives, Jet};
use crate::quadrature::integrate_log;
use crate::util::{map, map_n};
use keanu_tensor::DoubleTensor;

impl Wiener<'_> {
    /// Elementwise log P(T ≤ x, B = boundary), broadcasting x with the parameters.
    ///
    /// The limit at infinity is the log probability of the selected boundary. The density
    /// is integrated numerically, including the existing across-trial variability in drift.
    pub fn log_cdf(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        self.log_tail(x, false)
    }

    /// Elementwise log P(T > x, B = boundary), broadcasting x with the parameters.
    ///
    /// Integrates the right tail directly. For a timeout whose boundary is unknown, add
    /// the two boundary probabilities in log space.
    pub fn log_ccdf(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        self.log_tail(x, true)
    }

    fn log_tail(&self, x: &DoubleTensor, upper: bool) -> Result<DoubleTensor> {
        map(
            (
                x,
                &*self.a,
                &*self.t0,
                &*self.w,
                &*self.v,
                &*self.sv,
                &*self.boundary,
            ),
            |(x, a, t0, w, v, sv, boundary)| {
                jet::value(x, [a, t0, w, v, sv], |x, p| log_tail(x, p, boundary, upper))
            },
        )
    }

    /// Elementwise derivatives of the joint log CDF.
    pub fn d_log_cdf(&self, x: &DoubleTensor) -> Result<WienerGradient> {
        self.d_log_tail(x, false)
    }

    /// Elementwise derivatives of the joint log survival probability.
    pub fn d_log_ccdf(&self, x: &DoubleTensor) -> Result<WienerGradient> {
        self.d_log_tail(x, true)
    }

    fn d_log_tail(&self, x: &DoubleTensor, upper: bool) -> Result<WienerGradient> {
        let [x, a, t0, w, v, sv] = map_n(
            (
                x,
                &*self.a,
                &*self.t0,
                &*self.w,
                &*self.v,
                &*self.sv,
                &*self.boundary,
            ),
            |(x, a, t0, w, v, sv, boundary)| {
                Self::log_tail_derivatives(x, [a, t0, w, v, sv], boundary, upper).gradient
            },
        )?;
        Ok(WienerGradient { x, a, t0, w, v, sv })
    }

    /// One joint log CDF or log survival term and its gradient and Hessian.
    ///
    /// The derivative order is `[x, a, t0, w, v, sv]`. Set `upper` for the survival term.
    /// Derivatives are taken through the numerical integral with its moving limit.
    pub fn log_tail_derivatives(
        x: f64,
        parameters: [f64; 5],
        boundary: i64,
        upper: bool,
    ) -> ElementDerivatives<6> {
        jet::derivatives(x, parameters, |x, p| log_tail(x, p, boundary, upper))
    }
}

pub(super) fn log_tail<const K: usize, const H: usize>(
    x: Jet<K, H>,
    [a, t0, w, v, sv]: [Jet<K, H>; 5],
    boundary: i64,
    upper: bool,
) -> Jet<K, H> {
    if !valid(a.v, t0.v, w.v, v.v, sv.v) || !matches!(boundary, 0 | 1) || x.v.is_nan() {
        return Jet::constant(f64::NAN);
    }
    let zero = Jet::constant(0.0);
    let infinity = Jet::constant(f64::INFINITY);
    let mut t = ((x - t0) / a) / a;
    if t.v <= 0.0 {
        t = zero;
    } else if x.v.is_infinite() || t.v.is_infinite() {
        t = infinity;
    }
    let (lo, hi) = if upper { (t, infinity) } else { (zero, t) };
    if lo.v >= hi.v {
        return Jet::constant(f64::NEG_INFINITY);
    }

    let drift = a * v;
    let spread = a * sv;
    let distance = if boundary == 0 { w.v } else { 1.0 - w.v };
    let knee = distance / (1.0 / distance + drift.v.hypot(spread.v));
    let one = Jet::constant(1.0);
    let density = |time| log_density(time, [one, zero, w, drift, spread], boundary);
    let mut total = Jet::constant(f64::NEG_INFINITY);
    if lo.v == 0.0 {
        let end = if hi.v <= knee {
            hi
        } else {
            Jet::constant(knee)
        };
        if end.v > 0.0 {
            // Place nodes close to the deadline in the exponentially small left tail.
            let scale = end.v / (end.v + 0.5 * distance * distance);
            let log_weight = end.ln() + scale.ln();
            total = integrate_log(&|r| {
                let q = 1.0 - r + scale * r;
                density(end * ((1.0 - r) / q)) + log_weight - 2.0 * q.ln()
            });
        }
    }

    let start = if lo.v > 0.0 { lo } else { Jet::constant(knee) };
    let end = if hi.v <= 1.0 { hi } else { one };
    if start.v > 0.0 && end.v > start.v {
        let log_start = start.ln();
        let log_width = if end.v / start.v < 2.0 {
            ((end - start) / start).ln_1p()
        } else {
            end.ln() - log_start
        };
        total = total.log_add_exp(integrate_log(&|r| {
            let log_time = log_start + r * log_width;
            density(log_time.exp()) + log_time + log_width.ln()
        }));
    }

    let start = if lo.v >= 1.0 { lo } else { one };
    if hi.v > start.v {
        // The late-time rate is at least the lowest diffusion eigenvalue. Concentrate
        // nodes closer to the limit when drift adds a faster decay.
        let adjusted_drift = drift.v / (1.0 + (spread.v * start.v.sqrt()).powi(2));
        let scale = 1.0 / (1.0 + (0.5 * adjusted_drift) * adjusted_drift);
        let extent = if hi.v.is_infinite() {
            one
        } else {
            let width = hi - start;
            width / (scale + width)
        };
        total = total.log_add_exp(integrate_log(&|r| {
            let q = extent * r;
            density(start + scale * q / (1.0 - q)) + scale.ln() + extent.ln() - 2.0 * (-q).ln_1p()
        }));
    }
    total
}
