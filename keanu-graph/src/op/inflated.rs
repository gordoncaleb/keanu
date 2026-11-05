//! Zero-inflated and hurdle distributions, built around an inner distribution.
//!
//! Both are kinds built around another one with the layout described in the wrappers module: the node's parents are
//! the inner kind's parameters, in its order, followed by their own parameter `psi` (ψ, in [0, 1]). Both are
//! element-wise, so they can be mixture components, hidden Markov model emissions and log densities, and the
//! second-order rules contract a per-element Hessian.
//!
//! With p the inner kind's mass (or density) and a = ln p(0):
//!
//! - **zero-inflated** (a count inner kind): ln P(0) = ln(ψ + (1 − ψ) eᵃ), computed as a logsumexp, and
//!   ln P(k) = ln(1 − ψ) + ln p(k) for k ≠ 0. A zero is a structural zero with probability ψ / P(0);
//! - **hurdle** with a count inner kind: ln P(0) = ln ψ and ln P(k) = ln(1 − ψ) + ln p(k) − ln(1 − eᵃ) for k > 0,
//!   the inner kind truncated to the positive values (by its own mass at zero, so no CDF is needed);
//! - **hurdle** with a positive continuous inner kind: a value of exactly 0 has probability ψ, and a positive value
//!   the density (1 − ψ) p(x), the inner kind's support having no mass at 0.
//!
//! Negative values are impossible for a hurdle. A ψ outside [0, 1] (or NaN) makes every value impossible, and
//! sampling with it is an error.

use super::distribution::{DistributionKind, LogProbGradientParts, Univariate};
use super::distribution_hvp::rules::{self, Hessian};
use super::distribution_hvp::{self, Elementwise};
use super::wrappers::{self, Evaluated};
use super::{Support, broadcast, invalid_parameter};
use crate::error::{GraphError, Result};
use crate::value::{DType, Value};
use keanu_distributions::KeanuRng;
use keanu_tensor::special::log1mexp;
use keanu_tensor::{DoubleTensor, IntTensor, Shape, broadcast_shape};

/// The number of parameters a zero-inflated or hurdle kind adds after its inner kind's: `psi`.
pub(crate) const OWN_PARAMETERS: usize = 1;

/// Which form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Inflation {
    ZeroInflated,
    Hurdle,
}

/// Whether a continuous kind's support is the positive reals (or a part of them starting at 0), so that it can be a
/// hurdle's inner kind with its point mass at exactly 0.
fn positive_continuous(kind: &DistributionKind) -> bool {
    use DistributionKind as K;
    matches!(
        kind,
        K::HalfGaussian
            | K::HalfCauchy
            | K::ChiSquared
            | K::Exponential
            | K::Gamma
            | K::InverseGamma
            | K::LogNormal
            | K::Weibull
            | K::Beta
    )
}

/// Checks that `inner` can be wrapped, returning the reason it can't.
pub(crate) fn check_inner(
    inflation: Inflation,
    inner: &DistributionKind,
) -> std::result::Result<(), String> {
    let elementwise = inner.is_elementwise();
    match (inflation, inner.value_type()) {
        (_, DType::I64) if elementwise => Ok(()),
        (Inflation::Hurdle, DType::F64) if elementwise && positive_continuous(inner) => Ok(()),
        (Inflation::ZeroInflated, _) => Err(format!(
            "the inner distribution ({}) must be an element-wise count distribution (integer values, one per \
             element), such as a Poisson, negative binomial or binomial",
            inner.name()
        )),
        (Inflation::Hurdle, _) => Err(format!(
            "the inner distribution ({}) must be an element-wise count distribution or a positive continuous one \
             (half Gaussian, half Cauchy, chi-squared, exponential, gamma, inverse gamma, log-normal, Weibull or \
             beta)",
            inner.name()
        )),
    }
}

/// The support: discrete for a count inner kind, the inner kind's for a continuous one (the point mass at 0 is on
/// its boundary, so a hurdle on continuous values is meant to be observed).
pub(crate) fn support(
    _: Inflation,
    inner: &DistributionKind,
    parameters: &[&Value],
) -> Result<Support> {
    let (base, _) = wrappers::split(parameters, OWN_PARAMETERS)?;
    if inner.value_type() == DType::F64 {
        inner.support(base)
    } else {
        Ok(Support::Discrete)
    }
}

/// Whether a ψ is a probability (false for NaN).
fn valid(psi: f64) -> bool {
    (0.0..=1.0).contains(&psi)
}

/// Draws a value: for a zero-inflated kind a draw of the inner kind, for a hurdle a draw of the inner kind
/// restricted to positive values (by rejection for a count, see the wrappers module), each replaced by 0 with
/// probability ψ.
pub(crate) fn sample(
    inflation: Inflation,
    inner: &DistributionKind,
    shape: &[usize],
    parameters: &[&Value],
    rng: &mut KeanuRng,
) -> Result<Value> {
    let (base, own) = wrappers::split(parameters, OWN_PARAMETERS)?;
    let psi = broadcast(own[0].f64()?, shape)?;
    if let Some(bad) = psi.iter().find(|&&p| !valid(p)) {
        return invalid_parameter(format!("psi must be in [0, 1], got {bad}"));
    }
    let draw = match (inflation, inner.value_type()) {
        (Inflation::Hurdle, DType::I64) => {
            let (one, infinity) = (
                Value::F64(DoubleTensor::scalar(1.0)),
                Value::F64(DoubleTensor::scalar(f64::INFINITY)),
            );
            let mut truncated: Vec<&Value> = base.to_vec();
            truncated.extend([&one, &infinity]);
            wrappers::sample_truncated(inner, shape, &truncated, rng)?
        }
        _ => inner.sample(shape, base, rng)?,
    };
    let u = rng.uniform(shape);
    let zero = |i: usize| u.as_slice()[i] < psi[i];
    Ok(match draw {
        Value::I64(t) => {
            let t = broadcast_int(&t, shape)?;
            let values = t
                .into_iter()
                .enumerate()
                .map(|(i, k)| if zero(i) { 0 } else { k })
                .collect();
            Value::I64(IntTensor::new(shape, values)?)
        }
        Value::F64(t) => {
            let values = broadcast(&t, shape)?
                .into_iter()
                .enumerate()
                .map(|(i, x)| if zero(i) { 0.0 } else { x })
                .collect();
            Value::F64(DoubleTensor::new(shape, values)?)
        }
        Value::Bool(_) => {
            return Err(GraphError::InvalidValue(
                "a zero-inflated or hurdle distribution has no boolean values".into(),
            ));
        }
    })
}

fn broadcast_int(t: &IntTensor, shape: &[usize]) -> Result<Vec<i64>> {
    Ok(if t.shape() == shape {
        t.as_slice().to_vec()
    } else {
        t.broadcast_to(shape)?.into_vec()
    })
}

/// The pieces of an element-wise evaluation: the broadcast shape, the value as f64s, ψ, the inner kind at the value
/// and (for a count inner kind) at 0.
struct Pieces {
    shape: Shape,
    x: Vec<f64>,
    psi: Vec<f64>,
    at_value: Evaluated,
    at_zero: Option<Evaluated>,
}

fn pieces(
    inflation: Inflation,
    inner: &DistributionKind,
    value: &Value,
    parameters: &[&Value],
    gradient: bool,
) -> Result<Pieces> {
    let (base, own) = wrappers::split(parameters, OWN_PARAMETERS)?;
    let mut shape = Shape::from_slice(value.shape());
    for p in parameters {
        shape = broadcast_shape(&shape, p.shape())?;
    }
    let x = broadcast(&distribution_hvp::as_f64(value), &shape)?;
    let psi = broadcast(own[0].f64()?, &shape)?;
    let at_value = wrappers::evaluate(
        inner,
        Univariate::Density,
        value,
        base,
        gradient,
        &shape,
        None,
    )?;
    let needs_zero = match inflation {
        Inflation::ZeroInflated => true,
        Inflation::Hurdle => inner.value_type() == DType::I64,
    };
    let at_zero = if needs_zero {
        let zero = Value::I64(IntTensor::scalar(0));
        Some(wrappers::evaluate(
            inner,
            Univariate::Density,
            &zero,
            base,
            gradient,
            &shape,
            None,
        )?)
    } else {
        None
    };
    Ok(Pieces {
        shape,
        x,
        psi,
        at_value,
        at_zero,
    })
}

/// How an element's log probability is made up, from which its value and derivatives follow.
#[derive(Debug, Clone, Copy)]
enum Case {
    /// −∞.
    Impossible,
    /// A zero-inflated zero: ln(ψ + (1 − ψ)eᵃ) = `log_prob`, with r = (1 − ψ)eᵃ / P(0) the probability that it is
    /// the inner kind's zero.
    InflatedZero { log_prob: f64, r: f64 },
    /// ln ψ: a hurdle's zero.
    HurdleZero,
    /// ln(1 − ψ) + ln p(x), less ln(1 − eᵃ) for a hurdle on counts, whose derivative in a is `s` = eᵃ / (1 − eᵃ)
    /// (0 for the other kinds).
    Inner { s: f64 },
}

/// The case of element `i`.
fn case(inflation: Inflation, p: &Pieces, i: usize) -> Case {
    let (x, psi) = (p.x[i], p.psi[i]);
    if !valid(psi) {
        return Case::Impossible;
    }
    let a = p.at_zero.as_ref().map(|z| z.values[i]);
    // no mass left for the non-zero values, or none at zero for a hurdle
    if (psi == 1.0 && x != 0.0) || (inflation == Inflation::Hurdle && psi == 0.0 && x == 0.0) {
        return Case::Impossible;
    }
    match inflation {
        Inflation::ZeroInflated if x == 0.0 => {
            let a = a.expect("evaluated at zero");
            let structural = psi.ln();
            let inner = (-psi).ln_1p() + a;
            let log_prob = keanu_tensor::special::log_sum_exp([structural, inner]);
            if log_prob == f64::NEG_INFINITY || log_prob.is_nan() {
                return Case::Impossible;
            }
            Case::InflatedZero {
                log_prob,
                r: (inner - log_prob).exp(),
            }
        }
        Inflation::ZeroInflated => Case::Inner { s: 0.0 },
        Inflation::Hurdle if x == 0.0 => Case::HurdleZero,
        Inflation::Hurdle if x < 0.0 || x.is_nan() => Case::Impossible,
        Inflation::Hurdle => match a {
            // all of the inner kind's mass at zero leaves none for positive values
            Some(a) if a >= 0.0 => Case::Impossible,
            Some(a) if a == f64::NEG_INFINITY => Case::Inner { s: 0.0 },
            Some(a) => Case::Inner {
                s: 1.0 / (-a).exp_m1(),
            },
            None => Case::Inner { s: 0.0 },
        },
    }
}

/// The log probability of each element of `value` broadcast with the parameters and, with `gradient`, the gradient
/// parts in that broadcast shape: the inner kind's parameters (None where it has no derivative), then ψ, and the
/// value for a continuous inner kind.
pub(crate) fn elementwise(
    inflation: Inflation,
    inner: &DistributionKind,
    value: &Value,
    parameters: &[&Value],
    gradient: bool,
) -> Result<(DoubleTensor, Option<LogProbGradientParts>)> {
    let p = pieces(inflation, inner, value, parameters, gradient)?;
    let len = p.x.len();
    let cases: Vec<Case> = (0..len).map(|i| case(inflation, &p, i)).collect();
    let log_probs: Vec<f64> = cases
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let a = p
                .at_zero
                .as_ref()
                .map_or(f64::NEG_INFINITY, |z| z.values[i]);
            match *c {
                Case::Impossible => f64::NEG_INFINITY,
                Case::InflatedZero { log_prob, .. } => log_prob,
                Case::HurdleZero => p.psi[i].ln(),
                Case::Inner { .. } => {
                    let truncation = match (inflation, &p.at_zero) {
                        (Inflation::Hurdle, Some(_)) if a > f64::NEG_INFINITY => log1mexp(a),
                        _ => 0.0,
                    };
                    (-p.psi[i]).ln_1p() + p.at_value.values[i] - truncation
                }
            }
        })
        .collect();
    let values = DoubleTensor::new(&p.shape, log_probs)?;
    if !gradient {
        return Ok((values, None));
    }
    let mut inner_gradients: Vec<Option<Vec<f64>>> = p
        .at_value
        .parameters
        .iter()
        .map(|g| g.as_ref().map(|_| vec![0.0; len]))
        .collect();
    let mut psi_gradient = vec![0.0; len];
    let mut x_gradient = p.at_value.x.as_ref().map(|_| vec![0.0; len]);
    for (i, c) in cases.iter().enumerate() {
        let psi = p.psi[i];
        match *c {
            Case::Impossible => {}
            Case::InflatedZero { log_prob, r } => {
                let a = p.at_zero.as_ref().expect("evaluated at zero").values[i];
                psi_gradient[i] = -a.exp_m1() * (-log_prob).exp();
                for (j, gradient) in inner_gradients.iter_mut().enumerate() {
                    if let (Some(g), Some(d)) = (
                        gradient.as_mut(),
                        p.at_zero.as_ref().and_then(|z| z.parameters[j].as_ref()),
                    ) {
                        g[i] = r * d[i];
                    }
                }
            }
            Case::HurdleZero => psi_gradient[i] = 1.0 / psi,
            Case::Inner { s } => {
                psi_gradient[i] = -1.0 / (1.0 - psi);
                for (j, gradient) in inner_gradients.iter_mut().enumerate() {
                    let Some(g) = gradient.as_mut() else {
                        continue;
                    };
                    let at_value = p.at_value.parameters[j].as_ref().map_or(0.0, |d| d[i]);
                    let at_zero = p
                        .at_zero
                        .as_ref()
                        .and_then(|z| z.parameters[j].as_ref())
                        .map_or(0.0, |d| d[i]);
                    g[i] = at_value + s * at_zero;
                }
                if let (Some(g), Some(d)) = (x_gradient.as_mut(), p.at_value.x.as_ref()) {
                    g[i] = d[i];
                }
            }
        }
    }
    inner_gradients.push(Some(psi_gradient));
    Ok((
        values,
        Some(wrappers::parts(&p.shape, x_gradient, inner_gradients)?),
    ))
}

/// The second-order parts at `value` broadcast to `shape`, along the inputs' `tangents` (the value's, each inner
/// parameter's, then ψ's; see [`distribution_hvp::elementwise`]). Inputs are numbered 0 for the value, 1..=m for the
/// inner parameters and m + 1 for ψ; H_a is the inner log density's Hessian at 0 and ∇a its gradient there.
///
/// - zero-inflated zero, L = ln(ψ + (1 − ψ)eᵃ): with D = eᴸ and r = (1 − ψ)eᵃ / D, L_a = r, L_aa = r(1 − r),
///   L_ψ = (1 − eᵃ)/D, L_ψψ = −L_ψ², L_ψa = −eᵃ/D², so H = r(1 − r)∇a∇aᵀ + r H_a in the inner parameters,
///   −eᵃ/D² ∇a between them and ψ and −L_ψ² for ψ;
/// - hurdle zero, ln ψ: −1/ψ²;
/// - otherwise ln(1 − ψ) + ln p(x) − ln(1 − eᵃ): the inner Hessian at x, −1/(1 − ψ)² for ψ, and for a hurdle on
///   counts s H_a + s(1 + s)∇a∇aᵀ with s = eᵃ/(1 − eᵃ).
///
/// # Errors
///
/// [`GraphError::NoSecondOrderRule`] naming the inner kind if it has no element-wise second-order rule.
pub(crate) fn second_order(
    inflation: Inflation,
    inner: &DistributionKind,
    value: &Value,
    parameters: &[&Value],
    tangents: &[Option<Vec<f64>>],
    b: usize,
    shape: &[usize],
) -> Result<Elementwise> {
    let Some(rule) = rules::density_rule(inner) else {
        return Err(GraphError::NoSecondOrderRule(format!(
            "the {} distribution",
            inner.name()
        )));
    };
    let (values, parts) = elementwise(inflation, inner, value, parameters, true)?;
    let values = distribution_hvp::spread(&values, shape)?;
    let (x_gradient, parameter_gradients) =
        parts.unwrap_or_else(|| (None, smallvec::smallvec![None; parameters.len()]));
    let gradients: Vec<Option<Vec<f64>>> = std::iter::once(x_gradient)
        .chain(parameter_gradients)
        .map(|g| g.map(|g| distribution_hvp::spread(&g, shape)).transpose())
        .collect::<Result<_>>()?;
    let p = pieces(inflation, inner, value, parameters, true)?;
    let (base, _) = wrappers::split(parameters, OWN_PARAMETERS)?;
    let m = base.len();
    let psi_index = m + 1;
    let inputs: Vec<Vec<f64>> = base
        .iter()
        .map(|v| broadcast(&distribution_hvp::as_f64(v), shape))
        .collect::<Result<_>>()?;
    let len = values.len();
    let cases: Vec<Case> = (0..len).map(|i| case(inflation, &p, i)).collect();
    let differentiable: Vec<bool> = gradients.iter().map(Option::is_some).collect();
    let mut point = vec![0.0; m];
    let mut at_zero = Hessian::new(m + 2);
    let zero_gradient = |i: usize| -> Vec<f64> {
        (0..m)
            .map(|j| {
                p.at_zero
                    .as_ref()
                    .and_then(|z| z.parameters[j].as_ref())
                    .map_or(0.0, |d| d[i])
            })
            .collect()
    };
    let hvps = distribution_hvp::contract(&differentiable, tangents, len, b, |i, h| {
        for (q, input) in point.iter_mut().zip(&inputs) {
            *q = input[i];
        }
        let psi = p.psi[i];
        // adds `scale` times the inner Hessian at 0 (its parameter block), and `outer` times ∇a∇aᵀ
        let mut add_zero = |h: &mut Hessian, scale: f64, outer: f64| {
            at_zero.clear();
            rule(0.0, &point, &mut at_zero);
            let g = zero_gradient(i);
            for u in 0..m {
                for v in u..m {
                    h.add(
                        u + 1,
                        v + 1,
                        scale * at_zero.get(u + 1, v + 1) + outer * g[u] * g[v],
                    );
                }
            }
            g
        };
        match cases[i] {
            Case::Impossible => {}
            Case::InflatedZero { log_prob, r } => {
                let a = p.at_zero.as_ref().expect("evaluated at zero").values[i];
                let g = add_zero(h, r, r * (1.0 - r));
                let d_psi = -a.exp_m1() * (-log_prob).exp();
                h.add(psi_index, psi_index, -d_psi * d_psi);
                let cross = -(a - 2.0 * log_prob).exp();
                for (u, gu) in g.iter().enumerate() {
                    h.add(u + 1, psi_index, cross * gu);
                }
            }
            Case::HurdleZero => h.add(psi_index, psi_index, -1.0 / (psi * psi)),
            Case::Inner { s } => {
                rule(p.x[i], &point, h);
                h.add(psi_index, psi_index, -1.0 / ((1.0 - psi) * (1.0 - psi)));
                if s != 0.0 {
                    add_zero(h, s, s * (1.0 + s));
                }
            }
        }
        Ok(())
    })?;
    Ok(Elementwise {
        values,
        gradients,
        hvps,
    })
}
