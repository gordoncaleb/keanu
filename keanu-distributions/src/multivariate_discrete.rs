//! Discrete distributions over vectors: the Dirichlet-multinomial (counts over categories) and the Plackett–Luce
//! (rankings).

// the negated comparisons (`!(kappa >= 0.0)`) are deliberate: they also catch NaN
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use crate::continuous::sample_dirichlet;
use crate::discrete::sample_binomial;
use crate::error::{DistributionError, Result, invalid_parameter};
use crate::event::{EventFamily, EventGradient, Events};
use crate::param::{IntoParam, Param};
use crate::real::Real;
use crate::rng::KeanuRng;
use keanu_tensor::{DoubleTensor, IntTensor};

/// A non-empty last axis, or an error naming the distribution.
fn non_empty(what: &str, event: &[usize]) -> Result<()> {
    if event[0] == 0 {
        return Err(DistributionError::InvalidValue(format!(
            "a {what} value needs at least one element"
        )));
    }
    Ok(())
}

/// The first of a batch's parts (the gradient or Hessian-vector product of every parameter after the first).
fn parameter(parts: (DoubleTensor, Vec<DoubleTensor>), index: usize) -> DoubleTensor {
    parts
        .1
        .into_iter()
        .nth(index)
        .unwrap_or_else(|| DoubleTensor::scalar(0.0))
}

/// Integer draws written as f64 by an event sampler, as an integer tensor.
fn to_int(values: DoubleTensor) -> Result<IntTensor> {
    let data = values.as_slice().iter().map(|&v| v as i64).collect();
    Ok(IntTensor::new(values.shape(), data)?)
}

// ---------------------------------------------------------------------------------------------------------------
// Dirichlet-multinomial

struct DirichletMultinomialKernel;

impl EventFamily for DirichletMultinomialKernel {
    const EVENT_RANK: usize = 1;

    fn validate_event(&self, event: &[usize]) -> Result<()> {
        non_empty("Dirichlet-multinomial", event)
    }

    fn parameter_events(&self, event: &[usize]) -> Vec<Vec<usize>> {
        vec![vec![], event.to_vec()]
    }

    fn check(&self, _: &[usize], x: &[f64], p: &[&[f64]]) -> Option<f64> {
        let (n, alpha) = (p[0][0], p[1]);
        if !(n >= 0.0) || alpha.iter().any(|&a| !(a > 0.0 && a.is_finite())) {
            return Some(f64::NAN);
        }
        let valid = x.iter().all(|&v| v >= 0.0) && x.iter().sum::<f64>() == n;
        (!valid).then_some(f64::NEG_INFINITY)
    }

    fn gradient<S: Real>(&self, _: &[usize], x: &[S], p: &[&[S]]) -> EventGradient<S> {
        let (n, alpha) = (p[0][0], p[1]);
        let total = alpha.iter().fold(S::zero(), |s, &a| s + a);
        let mut log_prob = (n + 1.0).ln_gamma() + total.ln_gamma() - (n + total).ln_gamma();
        let shared = total.digamma() - (n + total).digamma();
        let mut d_alpha = Vec::with_capacity(alpha.len());
        for (&x, &a) in x.iter().zip(alpha) {
            log_prob = log_prob + (x + a).ln_gamma() - a.ln_gamma() - (x + 1.0).ln_gamma();
            d_alpha.push(shared + (x + a).digamma() - a.digamma());
        }
        EventGradient {
            log_prob,
            x: vec![S::zero(); x.len()],
            parameters: vec![vec![S::zero()], d_alpha],
        }
    }
}

/// The Dirichlet-multinomial distribution: counts of `n` draws over K categories whose probabilities are
/// Dirichlet(α) distributed, as PyMC's `DirichletMultinomial(n, a)` (Stan's `dirichlet_multinomial` takes n from
/// the counts):
///
/// ```text
/// P(x) = n! Γ(α₀) / Γ(n + α₀) Π_k Γ(x_k + α_k) / (x_k! Γ(α_k)),  α₀ = Σ_k α_k
/// ```
///
/// for counts x_k >= 0 summing to n. It is an over-dispersed multinomial with mean nα/α₀. The value is
/// [..., K], `alpha` [..., K] (positive) and `n` a scalar or one per vector. Counts that are negative or don't sum to
/// n are impossible; a negative n or a non-positive α gives NaN. Draws are a Dirichlet draw followed by a
/// multinomial one.
#[derive(Debug, Clone, PartialEq)]
pub struct DirichletMultinomial<'a> {
    /// The number of draws n >= 0.
    pub n: Param<'a, i64>,
    /// The concentrations α > 0, [..., K].
    pub alpha: Param<'a>,
}

/// Derivatives of [`DirichletMultinomial::log_prob`] with respect to `alpha` (shaped like it), or its
/// Hessian-vector products along `b` directions ([b, alpha shape...]).
#[derive(Debug, Clone, PartialEq)]
pub struct DirichletMultinomialGradient {
    /// With respect to `alpha`.
    pub alpha: DoubleTensor,
}

impl<'a> DirichletMultinomial<'a> {
    /// Creates the distribution from its parameters.
    pub fn new(n: impl IntoParam<'a, i64>, alpha: impl IntoParam<'a, f64>) -> Self {
        Self {
            n: n.into_param(),
            alpha: alpha.into_param(),
        }
    }

    fn events<'e>(&'e self, n: &'e DoubleTensor) -> Events<'e, DirichletMultinomialKernel> {
        Events {
            family: DirichletMultinomialKernel,
            what: "Dirichlet-multinomial",
            parameters: vec![("n", n), ("alpha", &self.alpha)],
        }
    }

    /// The total log probability of every count vector in x ([..., K]), summed over the batch. An error if the
    /// shapes don't fit.
    pub fn log_prob(&self, x: &IntTensor) -> Result<f64> {
        let n = self.n.to_double();
        self.events(&n).log_prob(&x.to_double())
    }

    /// The derivative of [`Self::log_prob`] with respect to `alpha`.
    pub fn d_log_prob(&self, x: &IntTensor) -> Result<DirichletMultinomialGradient> {
        let n = self.n.to_double();
        Ok(DirichletMultinomialGradient {
            alpha: parameter(self.events(&n).d_log_prob(&x.to_double())?, 1),
        })
    }

    /// Hessian-vector products of [`Self::log_prob`] along `b` directions given as tangents of `alpha`
    /// ([b, alpha shape], None for zero).
    ///
    /// # Errors
    ///
    /// An error if the tangent has the wrong shape.
    pub fn hvp(
        &self,
        x: &IntTensor,
        alpha_tangent: Option<&DoubleTensor>,
        b: usize,
    ) -> Result<DirichletMultinomialGradient> {
        let n = self.n.to_double();
        let parts = self
            .events(&n)
            .hvp(&x.to_double(), None, &[None, alpha_tangent], b)?;
        Ok(DirichletMultinomialGradient {
            alpha: parameter(parts, 1),
        })
    }

    /// Draws count vectors of the given shape ([..., K]). An error unless n >= 0 and every α is positive.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<IntTensor> {
        let n = self.n.to_double();
        let draws = self.events(&n).sample(shape, rng, |_, p, out, rng| {
            let (n, alpha) = (p[0][0], p[1]);
            if DirichletMultinomialKernel
                .check(&[], &[], &[&[0.0], alpha])
                .is_some()
                || n < 0.0
            {
                return invalid_parameter(
                    "a Dirichlet-multinomial needs n >= 0 and positive concentrations",
                );
            }
            // the Dirichlet draw from the Gamma draws' logs, so it is a probability vector even when every Gamma
            // draw would underflow (every α well below 1)
            let probabilities = sample_dirichlet(alpha, rng)?;
            // the multinomial by conditional binomials: each count out of what is left, with its share of what is
            // left of the probability; the last category takes the remainder, so the counts always sum to n
            let mut remaining = n as i64;
            let mut left = 1.0;
            let last = out.len().saturating_sub(1);
            for (i, (o, p)) in out.iter_mut().zip(&probabilities).enumerate() {
                let count = if i == last {
                    remaining
                } else {
                    let share = if left > 0.0 { (p / left).min(1.0) } else { 0.0 };
                    sample_binomial(remaining, share, rng)
                };
                *o = count as f64;
                remaining -= count;
                left -= p;
            }
            Ok(())
        })?;
        to_int(draws)
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Plackett-Luce

struct PlackettLuceKernel;

/// The items of a ranking as indices, if it is a permutation of 0..K.
fn permutation(x: &[f64]) -> Option<Vec<usize>> {
    let k = x.len();
    let mut seen = vec![false; k];
    let mut order = Vec::with_capacity(k);
    for &v in x {
        if !(v >= 0.0 && v < k as f64 && v.fract() == 0.0) || seen[v as usize] {
            return None;
        }
        seen[v as usize] = true;
        order.push(v as usize);
    }
    Some(order)
}

impl EventFamily for PlackettLuceKernel {
    const EVENT_RANK: usize = 1;

    fn validate_event(&self, event: &[usize]) -> Result<()> {
        non_empty("Plackett-Luce", event)
    }

    fn parameter_events(&self, event: &[usize]) -> Vec<Vec<usize>> {
        vec![event.to_vec()]
    }

    fn check(&self, _: &[usize], x: &[f64], p: &[&[f64]]) -> Option<f64> {
        if p[0].iter().any(|&w| !(w > 0.0 && w.is_finite())) {
            return Some(f64::NAN);
        }
        permutation(x).is_none().then_some(f64::NEG_INFINITY)
    }

    fn gradient<S: Real>(&self, _: &[usize], x: &[S], p: &[&[S]]) -> EventGradient<S> {
        let w = p[0];
        let k = w.len();
        let values: Vec<f64> = x.iter().map(|v| v.value()).collect();
        let Some(order) = permutation(&values) else {
            let nan = S::constant(f64::NAN);
            return EventGradient {
                log_prob: nan,
                x: vec![S::zero(); k],
                parameters: vec![vec![nan; k]],
            };
        };
        // S_j = Σ_{l >= j} w_{o_l}, the total worth of the items not yet ranked at position j
        let mut remaining = vec![S::zero(); k];
        let mut total = S::zero();
        for j in (0..k).rev() {
            total = total + w[order[j]];
            remaining[j] = total;
        }
        let mut log_prob = S::zero();
        let mut d = vec![S::zero(); k];
        let mut cumulative = S::zero();
        for j in 0..k {
            let item = order[j];
            // the last position's term, ln w − ln w, is zero
            if j + 1 < k {
                log_prob = log_prob + w[item].ln() - remaining[j].ln();
                cumulative = cumulative + S::constant(1.0) / remaining[j];
                d[item] = d[item] + S::constant(1.0) / w[item];
            }
            d[item] = d[item] - cumulative;
        }
        EventGradient {
            log_prob,
            x: vec![S::zero(); k],
            parameters: vec![d],
        }
    }
}

/// The Plackett–Luce distribution over rankings of K items with positive worths w: the items are chosen one at a
/// time, each with probability proportional to its worth among those not yet chosen, so a ranking x (a
/// permutation of 0..K, x₀ the index of the item ranked first, x₁ the second, ...) has
///
/// ```text
/// P(x) = Π_{j=0}^{K−2} w_{x_j} / Σ_{l>=j} w_{x_l}
/// ```
///
/// The worths are identified only up to a common factor; a linear predictor η gives worths e^η. The value is a
/// zero-based ranking (i64 [..., K], the order of the items, not each item's rank) and `worths` [..., K]. A value
/// that is not a permutation of 0..K is impossible and a non-positive worth gives NaN. Draws sort the items by
/// ln w plus independent Gumbel noise (the Gumbel-max trick), which gives exactly this distribution.
#[derive(Debug, Clone, PartialEq)]
pub struct PlackettLuce<'a> {
    /// The positive worths of the items, [..., K].
    pub worths: Param<'a>,
}

/// Derivatives of [`PlackettLuce::log_prob`] with respect to `worths` (shaped like it), or its Hessian-vector
/// products along `b` directions ([b, worths shape...]).
#[derive(Debug, Clone, PartialEq)]
pub struct PlackettLuceGradient {
    /// With respect to `worths`.
    pub worths: DoubleTensor,
}

impl<'a> PlackettLuce<'a> {
    /// Creates the distribution from its parameter.
    pub fn new(worths: impl IntoParam<'a, f64>) -> Self {
        Self {
            worths: worths.into_param(),
        }
    }

    fn events(&self) -> Events<'_, PlackettLuceKernel> {
        Events {
            family: PlackettLuceKernel,
            what: "Plackett-Luce",
            parameters: vec![("worths", &self.worths)],
        }
    }

    /// The total log probability of every ranking in x ([..., K]), summed over the batch. An error if the shapes
    /// don't fit.
    pub fn log_prob(&self, x: &IntTensor) -> Result<f64> {
        self.events().log_prob(&x.to_double())
    }

    /// The derivative of [`Self::log_prob`] with respect to the worths (NaN for a value that is not a ranking).
    pub fn d_log_prob(&self, x: &IntTensor) -> Result<PlackettLuceGradient> {
        Ok(PlackettLuceGradient {
            worths: parameter_first(self.events().d_log_prob(&x.to_double())?),
        })
    }

    /// Hessian-vector products of [`Self::log_prob`] along `b` directions given as tangents of the worths
    /// ([b, worths shape], None for zero).
    ///
    /// # Errors
    ///
    /// An error if the tangent has the wrong shape.
    pub fn hvp(
        &self,
        x: &IntTensor,
        worths_tangent: Option<&DoubleTensor>,
        b: usize,
    ) -> Result<PlackettLuceGradient> {
        let parts = self
            .events()
            .hvp(&x.to_double(), None, &[worths_tangent], b)?;
        Ok(PlackettLuceGradient {
            worths: parameter_first(parts),
        })
    }

    /// Draws rankings of the given shape ([..., K]). An error unless every worth is positive.
    pub fn sample(&self, shape: &[usize], rng: &mut KeanuRng) -> Result<IntTensor> {
        let draws = self.events().sample(shape, rng, |_, p, out, rng| {
            let w = p[0];
            if w.iter().any(|&w| !(w > 0.0 && w.is_finite())) {
                return invalid_parameter("Plackett-Luce worths must be positive");
            }
            let mut keys: Vec<(f64, usize)> = w
                .iter()
                .enumerate()
                .map(|(i, &w)| (w.ln() - (-rng.next_f64_non_zero().ln()).ln(), i))
                .collect();
            keys.sort_by(|a, b| b.0.total_cmp(&a.0));
            for (o, (_, i)) in out.iter_mut().zip(keys) {
                *o = i as f64;
            }
            Ok(())
        })?;
        to_int(draws)
    }
}

/// The gradient of the only parameter.
fn parameter_first(parts: (DoubleTensor, Vec<DoubleTensor>)) -> DoubleTensor {
    parameter(parts, 0)
}
