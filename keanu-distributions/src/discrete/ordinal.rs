//! The ordered probit distribution.

use super::{increasing, select_category};
use crate::error::{Result, invalid_parameter};
use crate::jet::ElementDerivatives;
use crate::param::{IntoParam, Param};
use crate::rng::KeanuRng;
use crate::util::{broadcast_all, operand};
use keanu_tensor::special::{erf, erfcx, log_ndtr, log1mexp};
use keanu_tensor::{DoubleTensor, IntTensor, Shape};
use std::f64::consts::{FRAC_1_SQRT_2, PI};

/// ln √(2π).
const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_8;

/// Ordered probit distribution (Stan's `ordered_probit`) over categories 0..K: a latent η is compared with K − 1
/// increasing cutpoints c, giving
///
/// ```text
/// P(0) = 1 − Φ(η − c₁),  P(k) = Φ(η − c_k) − Φ(η − c_(k+1)),  P(K − 1) = Φ(η − c_(K−1))
/// ```
///
/// (one-based cutpoints), with Φ the standard normal CDF: [`OrderedLogistic`](crate::OrderedLogistic) with a
/// probit link, the model of a standard normal latent y* = η + ε cut at the cutpoints.
///
/// The cutpoints are a tensor whose last axis is the K − 1 cutpoints; its leading axes and η's shape broadcast with
/// the value's shape. Probabilities are computed in logs from ln Φ, and a difference Φ(a) − Φ(b) (a > b) from the
/// side where it doesn't cancel: as Φ(−b) − Φ(−a) in logs when both are in the upper half, as Φ(a) − Φ(b) in logs
/// when both are in the lower half, and as ½(erf(a/√2) − erf(b/√2)) when they straddle 0, so the probabilities stay
/// accurate far into either tail.
#[derive(Debug, Clone, PartialEq)]
pub struct OrderedProbit<'a> {
    /// The location η.
    pub eta: Param<'a>,
    /// The cutpoints, strictly increasing along the last axis.
    pub cutpoints: Param<'a>,
}

/// Derivatives of [`OrderedProbit::log_prob`].
#[derive(Debug, Clone, PartialEq)]
pub struct OrderedProbitGradient {
    /// With respect to `eta`, shaped like the log probability.
    pub eta: DoubleTensor,
    /// With respect to `cutpoints`, shaped like the log probability with the cutpoints' last axis appended.
    pub cutpoints: DoubleTensor,
}

/// ln Φ(z) and the ratio φ(z)/Φ(z), the latter from erfcx in the lower tail, where φ and Φ both underflow.
fn log_ndtr_ratio(z: f64) -> (f64, f64) {
    let value = log_ndtr(z);
    let ratio = if z < -1.0 {
        (2.0 / PI).sqrt() / erfcx(-z * FRAC_1_SQRT_2)
    } else {
        (-0.5 * z * z - LN_SQRT_2PI - value).exp()
    };
    (value, ratio)
}

/// ln(Φ(a) − Φ(b)) for a > b, from the side on which the difference doesn't cancel.
fn log_ndtr_difference(a: f64, b: f64) -> f64 {
    if b >= 0.0 {
        let upper = log_ndtr(-b);
        upper + log1mexp(log_ndtr(-a) - upper)
    } else if a <= 0.0 {
        let upper = log_ndtr(a);
        upper + log1mexp(log_ndtr(b) - upper)
    } else {
        (0.5 * (erf(a * FRAC_1_SQRT_2) - erf(b * FRAC_1_SQRT_2))).ln()
    }
}

/// ln P as a function f(a, b) of a = η − c_(k−1) and b = η − c_k (the lower and upper cutpoints of category k, a
/// missing one at the ends taken as an infinite a or b): f, f_a, f_b, f_aa, f_ab, f_bb.
fn probit_parts(a: Option<f64>, b: Option<f64>) -> [f64; 6] {
    match (a, b) {
        // the first category: f = ln Φ(−b), with ln Φ's derivatives r and −r(z + r) at z = −b
        (None, Some(b)) => {
            let z = -b;
            let (f, r) = log_ndtr_ratio(z);
            [f, 0.0, -r, 0.0, 0.0, -r * (z + r)]
        }
        // the last category: f = ln Φ(a)
        (Some(a), None) => {
            let (f, r) = log_ndtr_ratio(a);
            [f, r, 0.0, -r * (a + r), 0.0, 0.0]
        }
        // between two cutpoints: f = ln D, D = Φ(a) − Φ(b), f_a = φ(a)/D and f_b = −φ(b)/D
        (Some(a), Some(b)) => {
            let log_d = log_ndtr_difference(a, b);
            let fa = (-0.5 * a * a - LN_SQRT_2PI - log_d).exp();
            let fb = -(-0.5 * b * b - LN_SQRT_2PI - log_d).exp();
            [
                log_d,
                fa,
                fb,
                -a * fa - fa * fa,
                -fa * fb,
                -b * fb - fb * fb,
            ]
        }
        // a single category (no cutpoints) has probability 1
        (None, None) => [0.0; 6],
    }
}

impl<'a> OrderedProbit<'a> {
    /// Creates the distribution from its parameters.
    pub fn new(eta: impl IntoParam<'a, f64>, cutpoints: impl IntoParam<'a, f64>) -> Self {
        Self {
            eta: eta.into_param(),
            cutpoints: cutpoints.into_param(),
        }
    }

    /// The log probability of category `k` given η and one set of (strictly increasing) `cutpoints`, with its
    /// gradient and Hessian with respect to η (index 0), the cutpoint below the category, c_(k−1) (index 1), and
    /// the one above it, c_k (index 2), zero-based. At the ends one of the cutpoints is missing (below the first
    /// category, above the last) and its derivatives are zero. −∞ (with zero derivatives) for an index that is not
    /// a category.
    pub fn category_derivatives(k: i64, eta: f64, cutpoints: &[f64]) -> ElementDerivatives<3> {
        let count = cutpoints.len();
        let impossible = ElementDerivatives {
            value: f64::NEG_INFINITY,
            gradient: [0.0; 3],
            hessian: [[0.0; 3]; 3],
        };
        let Ok(k) = usize::try_from(k) else {
            return impossible;
        };
        if k > count {
            return impossible;
        }
        let a = (k > 0).then(|| eta - cutpoints[k - 1]);
        let b = (k < count).then(|| eta - cutpoints[k]);
        let [f, fa, fb, faa, fab, fbb] = probit_parts(a, b);
        // a = η − c_lo and b = η − c_hi
        ElementDerivatives {
            value: f,
            gradient: [fa + fb, -fa, -fb],
            hessian: [
                [faa + 2.0 * fab + fbb, -(faa + fab), -(fab + fbb)],
                [-(faa + fab), faa, fab],
                [-(fab + fbb), fab, fbb],
            ],
        }
    }

    /// The batch shape (the broadcast of the value's shape, η's shape and the cutpoints' leading axes) and the
    /// number of cutpoints, checking the cutpoints are a non-empty last axis.
    fn layout(&self, value: &[usize]) -> Result<(Shape, usize)> {
        let Some((&count, leading)) = self.cutpoints.shape().split_last() else {
            return invalid_parameter("the cutpoints must have at least one axis");
        };
        if count == 0 {
            return invalid_parameter("there must be at least one cutpoint");
        }
        Ok((broadcast_all(&[value, self.eta.shape(), leading])?, count))
    }

    /// The cutpoints broadcast to `batch` with their last axis appended.
    fn cutpoints_for(&self, batch: &[usize], count: usize) -> Result<Vec<f64>> {
        let mut shape = batch.to_vec();
        shape.push(count);
        Ok(self.cutpoints.broadcast_to(&shape)?.into_vec())
    }

    /// The log probability of each category index k, broadcasting k with η and the cutpoints' leading axes
    /// (−∞ for an index that is not a category). Where the cutpoints are not strictly increasing the log
    /// probability is −∞ (with a zero gradient), as for [`OrderedLogistic`](crate::OrderedLogistic); an error if
    /// the cutpoints have no last axis or it is empty.
    pub fn log_prob(&self, k: &IntTensor) -> Result<DoubleTensor> {
        Ok(self.log_prob_and_gradient(k, false)?.0)
    }

    /// The derivatives of [`Self::log_prob`] with respect to each parameter.
    pub fn d_log_prob(&self, k: &IntTensor) -> Result<OrderedProbitGradient> {
        Ok(self.log_prob_and_gradient(k, true)?.1)
    }

    fn log_prob_and_gradient(
        &self,
        k: &IntTensor,
        gradient: bool,
    ) -> Result<(DoubleTensor, OrderedProbitGradient)> {
        let (batch, count) = self.layout(k.shape())?;
        let cutpoints = self.cutpoints_for(&batch, count)?;
        let k = operand(k, &batch)?;
        let eta = operand(&self.eta, &batch)?;
        let len: usize = batch.iter().product();
        let mut log_prob = Vec::with_capacity(len);
        let mut d_eta = vec![0.0; if gradient { len } else { 0 }];
        let mut d_cutpoints = vec![0.0; if gradient { len * count } else { 0 }];
        for i in 0..len {
            let cuts = &cutpoints[i * count..(i + 1) * count];
            if !increasing(cuts) {
                log_prob.push(f64::NEG_INFINITY);
                continue;
            }
            let category = k.at(i);
            let d = Self::category_derivatives(category, eta.at(i), cuts);
            log_prob.push(d.value);
            if gradient && d.value > f64::NEG_INFINITY {
                d_eta[i] = d.gradient[0];
                let c = category as usize;
                if c > 0 {
                    d_cutpoints[i * count + c - 1] = d.gradient[1];
                }
                if c < count {
                    d_cutpoints[i * count + c] = d.gradient[2];
                }
            }
        }
        let mut cut_shape = batch.to_vec();
        cut_shape.push(count);
        let gradients = if gradient {
            OrderedProbitGradient {
                eta: DoubleTensor::new(&batch, d_eta)?,
                cutpoints: DoubleTensor::new(&cut_shape, d_cutpoints)?,
            }
        } else {
            OrderedProbitGradient {
                eta: DoubleTensor::scalar(0.0),
                cutpoints: DoubleTensor::scalar(0.0),
            }
        };
        Ok((DoubleTensor::new(&batch, log_prob)?, gradients))
    }

    /// The probability of each category, shaped like the batch with a last axis of the K categories. An error
    /// unless the cutpoints are strictly increasing.
    pub fn probabilities(&self) -> Result<DoubleTensor> {
        let (batch, count) = self.layout(&[])?;
        let cutpoints = self.cutpoints_for(&batch, count)?;
        let eta = operand(&self.eta, &batch)?;
        let len: usize = batch.iter().product();
        let mut out = Vec::with_capacity(len * (count + 1));
        for i in 0..len {
            let cuts = &cutpoints[i * count..(i + 1) * count];
            if !increasing(cuts) {
                return invalid_parameter("the cutpoints must be strictly increasing");
            }
            for k in 0..=count {
                out.push(
                    Self::category_derivatives(k as i64, eta.at(i), cuts)
                        .value
                        .exp(),
                );
            }
        }
        let mut shape = batch.to_vec();
        shape.push(count + 1);
        Ok(DoubleTensor::new(&shape, out)?)
    }

    /// Draws a sample of the given shape (broadcast with η's shape and the cutpoints' leading axes). An error
    /// unless the cutpoints are strictly increasing.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<IntTensor> {
        let (batch, count) = self.layout(shape)?;
        let cutpoints = self.cutpoints_for(&batch, count)?;
        let eta = operand(&self.eta, &batch)?;
        let len: usize = batch.iter().product();
        let mut samples = Vec::with_capacity(len);
        for i in 0..len {
            let cuts = &cutpoints[i * count..(i + 1) * count];
            if !increasing(cuts) {
                return invalid_parameter("the cutpoints must be strictly increasing");
            }
            let probabilities = (0..=count).map(|k| {
                Self::category_derivatives(k as i64, eta.at(i), cuts)
                    .value
                    .exp()
            });
            samples.push(select_category(rng.next_f64(), probabilities) as i64);
        }
        Ok(IntTensor::new(&batch, samples)?)
    }
}
