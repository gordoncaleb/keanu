//! Truncated and censored distributions, built around a base distribution.
//!
//! # Nested distribution kinds
//!
//! A distribution kind built around another one (here [`DistributionKind::Truncated`] and
//! [`DistributionKind::Censored`], and [`DistributionKind::Mixture`] in the mixture module) holds its base as an
//! `Arc<DistributionKind>` and takes the node's parents as the base's parameters, in the base's own order, followed
//! by the wrapper's own parameters. So:
//!
//! - the base sees exactly the parameter slice it would as a node of its own ([`split`] separates the two), and its
//!   values, supports and gradient parts map one to one onto the leading parents;
//! - the wrapper's own parameters are appended after them, and its gradient parts are the base's followed by its
//!   own;
//! - the parameter types are the base's followed by the wrapper's own, and the value type is the base's.
//!
//! A mixture's base parameters carry a trailing component axis, and its gradient parts for them are shaped like the
//! parameters broadcast with the value and that axis. A hidden Markov model ([`DistributionKind::HiddenMarkovModel`],
//! in the hmm module) does the same with its emission's parameters and a trailing state axis. The saved schema
//! stores the base as a nested distribution type, recursively, so any nesting the graph accepts (a mixture of
//! truncated distributions) saves and loads.

use super::distribution::{DistributionKind, LogProbGradientParts, ParameterGradients, Univariate};
use super::distribution_hvp::rules::{self, Hessian, MAX_INPUTS};
use super::distribution_hvp::{self, Elementwise, LogProbHvpParts};
use super::{Support, broadcast, invalid_parameter};
use crate::error::{GraphError, Result};
use crate::value::{DType, Value};
use keanu_distributions::{BoundedPoissonBinomial, KeanuRng, PoissonBinomial};
use keanu_tensor::special::log1mexp;
use keanu_tensor::{DoubleTensor, Shape, broadcast_shape};
use std::f64::consts::LN_2;

/// The number of parameters a truncated or censored kind adds after its base's: `lower` and `upper`.
pub(crate) const OWN_PARAMETERS: usize = 2;

/// How many times a truncated distribution without an inverse CDF draws from its base before giving up.
const REJECTION_ATTEMPTS: usize = 1000;

/// Which wrapper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Wrapper {
    Truncated,
    Censored,
}

/// Splits a wrapper's parameters into its base's and its own `own` trailing ones.
pub(crate) fn split<'p, 'v>(
    parameters: &'p [&'v Value],
    own: usize,
) -> Result<(&'p [&'v Value], &'p [&'v Value])> {
    if parameters.len() < own {
        return Err(GraphError::InvalidValue(format!(
            "a wrapped distribution needs at least {own} parameters, got {}",
            parameters.len()
        )));
    }
    Ok(parameters.split_at(parameters.len() - own))
}

/// A wrapper's base parameters and bounds, checked: for a discrete base each finite bound must be a whole number,
/// and (for [`Bounds::new`], which sampling and the support use) each lower bound must be less than its upper bound
/// (neither NaN).
///
/// The log density instead takes bounds that are not ordered as an impossible state, with log probability −∞
/// ([`Bounds::unordered`]), as a uniform distribution does: bounds that depend on latents can cross during
/// inference, and a sampler such as Metropolis–Hastings then rejects the state rather than failing. Constant bounds
/// that are not ordered are an error when the graph is built.
struct Bounds<'p, 'v> {
    base: &'p [&'v Value],
    lower: &'v DoubleTensor,
    upper: &'v DoubleTensor,
    discrete: bool,
}

/// Whether a lower and an upper bound are ordered (false if either is NaN).
fn ordered(lower: f64, upper: f64) -> bool {
    lower < upper
}

impl<'p, 'v> Bounds<'p, 'v> {
    /// The bounds, which must be ordered.
    fn new(base_kind: &DistributionKind, parameters: &'p [&'v Value]) -> Result<Self> {
        Self::checked(base_kind, parameters, true)
    }

    /// The bounds, which need not be ordered (for the log density).
    fn unordered(base_kind: &DistributionKind, parameters: &'p [&'v Value]) -> Result<Self> {
        Self::checked(base_kind, parameters, false)
    }

    fn checked(
        base_kind: &DistributionKind,
        parameters: &'p [&'v Value],
        check_order: bool,
    ) -> Result<Self> {
        let (base, own) = split(parameters, OWN_PARAMETERS)?;
        let (lower, upper) = (own[0].f64()?, own[1].f64()?);
        let discrete = base_kind.value_type() != DType::F64;
        let shape = broadcast_shape(lower.shape(), upper.shape())?;
        let (l, u) = (lower.broadcast_to(&shape)?, upper.broadcast_to(&shape)?);
        for (&l, &u) in l.as_slice().iter().zip(u.as_slice()) {
            if check_order && !ordered(l, u) {
                return invalid_parameter(format!(
                    "the lower bound must be less than the upper bound, got [{l}, {u}]"
                ));
            }
            if discrete && [l, u].iter().any(|b| b.is_finite() && b.fract() != 0.0) {
                return invalid_parameter(format!(
                    "the bounds of a discrete distribution must be whole numbers or infinite, got [{l}, {u}]"
                ));
            }
        }
        Ok(Self {
            base,
            lower,
            upper,
            discrete,
        })
    }
}

/// Keep the endpoints of joint tails, where the boundary probability can depend on the parameters.
fn bound_points(
    base: &DistributionKind,
    bound: &DoubleTensor,
    shift: f64,
    discrete: bool,
) -> Value {
    if base.has_joint_tails() {
        return Value::F64(bound.clone());
    }
    let finite = bound.map(|b| if b.is_infinite() { 0.0 } else { b + shift });
    if discrete {
        Value::I64(finite.to_int())
    } else {
        Value::F64(finite)
    }
}

fn from_cdf(base: &DistributionKind, cdf: (f64, f64), survival: (f64, f64)) -> bool {
    if !base.has_joint_tails() {
        return cdf.0 < -LN_2;
    }
    if cdf.1 == f64::NEG_INFINITY {
        return false;
    }
    if survival.0 == f64::NEG_INFINITY {
        return true;
    }
    cdf.0 - cdf.1 < survival.1 - survival.0
}

/// A function of the base evaluated element-wise and broadcast to the node's shape.
pub(crate) struct Evaluated {
    pub(crate) values: Vec<f64>,
    /// With respect to the point it was evaluated at (None for discrete values).
    pub(crate) x: Option<Vec<f64>>,
    /// With respect to each base parameter (None where there is no derivative).
    pub(crate) parameters: Vec<Option<Vec<f64>>>,
}

/// `function` of the base at `at`, broadcast to `shape`. Where `infinite` (broadcast) holds, the value is
/// `at_infinity` and the derivatives are 0.
pub(crate) fn evaluate(
    base: &DistributionKind,
    function: Univariate,
    at: &Value,
    parameters: &[&Value],
    gradient: bool,
    shape: &[usize],
    infinite: Option<(&DoubleTensor, f64)>,
) -> Result<Evaluated> {
    let (values, parts) = base.univariate(function, at, parameters, gradient)?;
    let mut values = broadcast(&values, shape)?;
    let (mut x, mut parameters) = match parts {
        Some((x, parameters)) => (
            x.map(|x| broadcast(&x, shape)).transpose()?,
            parameters
                .into_iter()
                .map(|p| p.map(|p| broadcast(&p, shape)).transpose())
                .collect::<Result<Vec<_>>>()?,
        ),
        None => (None, vec![None; parameters.len()]),
    };
    if let Some((bound, at_infinity)) = infinite.filter(|_| !base.has_joint_tails()) {
        let infinite = broadcast(
            &bound.map(|b| if b.is_infinite() { 1.0 } else { 0.0 }),
            shape,
        )?;
        for (i, _) in infinite.iter().enumerate().filter(|(_, f)| **f == 1.0) {
            values[i] = at_infinity;
            for g in x.iter_mut().chain(parameters.iter_mut().flatten()) {
                g[i] = 0.0;
            }
        }
    }
    Ok(Evaluated {
        values,
        x,
        parameters,
    })
}

/// The node's value as f64s.
fn value_as_f64(value: &Value) -> Result<DoubleTensor> {
    Ok(match value {
        Value::F64(t) => t.clone(),
        Value::I64(t) => t.to_double(),
        Value::Bool(_) => {
            return Err(GraphError::InvalidValue(
                "a truncated or censored distribution has no boolean values".into(),
            ));
        }
    })
}

/// The log probability of a truncated or censored node at `value` (summed over its elements) and, with
/// `gradient`, its gradient parts: the base's parameters followed by the lower and upper bounds (whose derivatives
/// exist only for a continuous base).
pub(crate) fn log_prob(
    wrapper: Wrapper,
    base: &DistributionKind,
    value: &Value,
    parameters: &[&Value],
    gradient: bool,
) -> Result<(f64, Option<LogProbGradientParts>)> {
    let (log_probs, gradient) = elementwise(wrapper, base, value, parameters, gradient)?;
    Ok((log_probs.sum(), gradient))
}

/// [`log_prob`] element-wise: the log probability of each element of `value` broadcast with the parameters, and
/// the gradient parts in that broadcast shape (what [`DistributionKind::univariate`] gives for the log density).
/// For Poisson-binomial, gradients are reduced to the original parameter shape, including its trial axis.
pub(crate) fn elementwise(
    wrapper: Wrapper,
    base: &DistributionKind,
    value: &Value,
    parameters: &[&Value],
    gradient: bool,
) -> Result<(DoubleTensor, Option<LogProbGradientParts>)> {
    if matches!(base, DistributionKind::PoissonBinomial) {
        let x = value.i64()?;
        return poisson_binomial(wrapper, parameters, |d| {
            if gradient {
                let (values, grad) = d.log_prob_and_gradient(x)?;
                Ok((
                    values,
                    Some((None, smallvec::smallvec![Some(grad.p), None, None])),
                ))
            } else {
                Ok((d.log_prob(x)?, None))
            }
        });
    }
    let bounds = Bounds::unordered(base, parameters)?;
    let mut shape = Shape::from_slice(value.shape());
    for p in parameters {
        shape = broadcast_shape(&shape, p.shape())?;
    }
    let x = value_as_f64(value)?;
    let x = if x.shape() == shape.as_slice() {
        x
    } else {
        x.broadcast_to(&shape)?
    };
    let lower = broadcast(bounds.lower, &shape)?;
    let upper = broadcast(bounds.upper, &shape)?;
    let density = evaluate(
        base,
        Univariate::Density,
        value,
        bounds.base,
        gradient,
        &shape,
        None,
    )?;
    let parts = match wrapper {
        Wrapper::Truncated => {
            truncated(&bounds, base, &x, &lower, &upper, density, gradient, &shape)?
        }
        Wrapper::Censored => {
            censored(&bounds, base, &x, &lower, &upper, density, gradient, &shape)?
        }
    };
    let (log_probs, gradient) = parts;
    Ok((DoubleTensor::new(&shape, log_probs)?, gradient))
}

fn poisson_binomial<T>(
    wrapper: Wrapper,
    parameters: &[&Value],
    f: impl FnOnce(BoundedPoissonBinomial<'_, '_>) -> Result<T>,
) -> Result<T> {
    let &[p, lower, upper] = parameters else {
        return Err(GraphError::InvalidValue(
            "bounded Poisson-binomial needs p, lower and upper".into(),
        ));
    };
    let d = PoissonBinomial::new(p.f64()?)?;
    let (lower, upper) = (lower.f64()?, upper.f64()?);
    f(match wrapper {
        Wrapper::Censored => d.censored(lower, upper),
        Wrapper::Truncated => d.truncated(lower, upper),
    })
}

pub(crate) fn poisson_binomial_hvp(
    wrapper: Wrapper,
    value: &Value,
    parameters: &[&Value],
    parameter_tangents: &[Option<&DoubleTensor>],
) -> Result<LogProbHvpParts> {
    let Some(tangent) = parameter_tangents[0] else {
        return Ok((None, smallvec::smallvec![None, None, None]));
    };
    let x = value.i64()?;
    let product = poisson_binomial(wrapper, parameters, |d| Ok(d.hvp(x, tangent)?))?;
    Ok((None, smallvec::smallvec![Some(product), None, None]))
}

/// Element-wise log probabilities and gradient parts.
type Parts = (Vec<f64>, Option<LogProbGradientParts>);

/// Assembles gradient parts from per-element vectors.
pub(crate) fn parts(
    shape: &[usize],
    x: Option<Vec<f64>>,
    parameters: Vec<Option<Vec<f64>>>,
) -> Result<LogProbGradientParts> {
    let tensor = |v: Vec<f64>| -> Result<DoubleTensor> { Ok(DoubleTensor::new(shape, v)?) };
    Ok((
        x.map(tensor).transpose()?,
        parameters
            .into_iter()
            .map(|p| p.map(tensor).transpose())
            .collect::<Result<ParameterGradients>>()?,
    ))
}

/// The probability the base gives the bounds: ln P(lower <= X <= upper) element-wise, and (with `gradient`) its
/// derivatives with respect to each base parameter and to the bounds.
struct Normaliser {
    log_z: Vec<f64>,
    parameters: Vec<Option<Vec<f64>>>,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

/// ln Z = ln(F(u) - F(l)) = ln(S(l) - S(u)) for each element, from the CDF where the lower point is in the lower
/// half of the distribution and from the survival function otherwise, so that neither difference cancels.
fn normaliser(
    bounds: &Bounds<'_, '_>,
    base: &DistributionKind,
    gradient: bool,
    shape: &[usize],
) -> Result<Normaliser> {
    // P(l <= X <= u) is F(u) - F(l) for a continuous base and F(u) - F(l - 1) for a discrete one
    let shift = if bounds.discrete { -1.0 } else { 0.0 };
    let low = bound_points(base, bounds.lower, shift, bounds.discrete);
    let high = bound_points(base, bounds.upper, 0.0, bounds.discrete);
    let eval = |function, at: &Value, bound: &DoubleTensor, at_infinity| {
        evaluate(
            base,
            function,
            at,
            bounds.base,
            gradient,
            shape,
            Some((bound, at_infinity)),
        )
    };
    // at an infinite lower bound F = 0 and S = 1, at an infinite upper bound F = 1 and S = 0
    let cdf_low = eval(Univariate::Cdf, &low, bounds.lower, f64::NEG_INFINITY)?;
    let ccdf_low = eval(Univariate::Ccdf, &low, bounds.lower, 0.0)?;
    let cdf_high = eval(Univariate::Cdf, &high, bounds.upper, 0.0)?;
    let ccdf_high = eval(Univariate::Ccdf, &high, bounds.upper, f64::NEG_INFINITY)?;
    let len = cdf_low.values.len();
    let mut log_z = Vec::with_capacity(len);
    let mut parameters: Vec<Option<Vec<f64>>> = cdf_low
        .parameters
        .iter()
        .map(|p| p.as_ref().map(|_| vec![0.0; len]))
        .collect();
    let mut lower = vec![0.0; len];
    let mut upper = vec![0.0; len];
    for i in 0..len {
        // d ln Z = w_high d ln G(high) - w_low d ln G(low), with G the CDF or the survival function, w = G / Z
        let (from_cdf, (low_g, high_g), sign) = if from_cdf(
            base,
            (cdf_low.values[i], cdf_high.values[i]),
            (ccdf_low.values[i], ccdf_high.values[i]),
        ) {
            (true, (&cdf_low, &cdf_high), 1.0)
        } else {
            (false, (&ccdf_low, &ccdf_high), -1.0)
        };
        let (g_low, g_high) = (low_g.values[i], high_g.values[i]);
        // the larger of the two, which Z is a fraction of
        let (big, small) = if from_cdf {
            (g_high, g_low)
        } else {
            (g_low, g_high)
        };
        let z = if big == f64::NEG_INFINITY {
            f64::NEG_INFINITY
        } else {
            big + log1mexp((small - big).min(0.0))
        };
        log_z.push(z);
        if !gradient || z == f64::NEG_INFINITY {
            continue;
        }
        let w_high = (g_high - z).exp();
        let w_low = (g_low - z).exp();
        for (j, p) in parameters.iter_mut().enumerate() {
            if let (Some(p), Some(d_high), Some(d_low)) = (
                p.as_mut(),
                high_g.parameters[j].as_ref(),
                low_g.parameters[j].as_ref(),
            ) {
                p[i] = sign * (w_high * d_high[i] - w_low * d_low[i]);
            }
        }
        if let (Some(d_high), Some(d_low)) = (&high_g.x, &low_g.x) {
            upper[i] = sign * w_high * d_high[i];
            lower[i] = -sign * w_low * d_low[i];
        }
    }
    Ok(Normaliser {
        log_z,
        parameters,
        lower,
        upper,
    })
}

#[allow(clippy::too_many_arguments)]
fn truncated(
    bounds: &Bounds<'_, '_>,
    base: &DistributionKind,
    x: &DoubleTensor,
    lower: &[f64],
    upper: &[f64],
    density: Evaluated,
    gradient: bool,
    shape: &[usize],
) -> Result<Parts> {
    let z = normaliser(bounds, base, gradient, shape)?;
    let x = x.as_slice();
    // (bounds that are not ordered hold no value with Z > 0)
    let inside: Vec<bool> = (0..x.len())
        .map(|i| x[i] >= lower[i] && x[i] <= upper[i] && z.log_z[i] > f64::NEG_INFINITY)
        .collect();
    let log_probs = (0..x.len())
        .map(|i| {
            if inside[i] {
                density.values[i] - z.log_z[i]
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect();
    if !gradient {
        return Ok((log_probs, None));
    }
    let masked = |v: Vec<f64>| -> Vec<f64> {
        v.into_iter()
            .zip(&inside)
            .map(|(d, &inside)| if inside { d } else { 0.0 })
            .collect()
    };
    let mut parameters: Vec<Option<Vec<f64>>> = density
        .parameters
        .into_iter()
        .zip(z.parameters)
        .map(|(d, dz)| match (d, dz) {
            (Some(d), Some(dz)) => Some(masked(d.iter().zip(&dz).map(|(d, dz)| d - dz).collect())),
            _ => None,
        })
        .collect();
    if bounds.discrete {
        parameters.extend([None, None]);
    } else {
        parameters.push(Some(masked(z.lower.iter().map(|d| -d).collect())));
        parameters.push(Some(masked(z.upper.iter().map(|d| -d).collect())));
    }
    let x_gradient = density.x.map(masked);
    Ok((log_probs, Some(parts(shape, x_gradient, parameters)?)))
}

#[allow(clippy::too_many_arguments)]
fn censored(
    bounds: &Bounds<'_, '_>,
    base: &DistributionKind,
    x: &DoubleTensor,
    lower: &[f64],
    upper: &[f64],
    density: Evaluated,
    gradient: bool,
    shape: &[usize],
) -> Result<Parts> {
    // P(X <= l) at the lower bound, and P(X >= u) at the upper: S(u) for a continuous base, S(u - 1) for a
    // discrete one
    let shift = if bounds.discrete { -1.0 } else { 0.0 };
    let low = bound_points(base, bounds.lower, 0.0, bounds.discrete);
    let high = bound_points(base, bounds.upper, shift, bounds.discrete);
    let at_low = evaluate(
        base,
        Univariate::Cdf,
        &low,
        bounds.base,
        gradient,
        shape,
        None,
    )?;
    let at_high = evaluate(
        base,
        Univariate::Ccdf,
        &high,
        bounds.base,
        gradient,
        shape,
        None,
    )?;
    #[derive(Clone, Copy, PartialEq)]
    enum Case {
        Outside,
        Lower,
        Upper,
        Inside,
    }
    let x = x.as_slice();
    let cases: Vec<Case> = (0..x.len())
        .map(|i| {
            if !ordered(lower[i], upper[i]) {
                Case::Outside
            } else if x[i] == lower[i] && lower[i].is_finite() {
                Case::Lower
            } else if x[i] == upper[i] && upper[i].is_finite() {
                Case::Upper
            } else if x[i] > lower[i] && x[i] < upper[i] {
                Case::Inside
            } else {
                Case::Outside
            }
        })
        .collect();
    let log_probs = cases
        .iter()
        .enumerate()
        .map(|(i, case)| match case {
            Case::Outside => f64::NEG_INFINITY,
            Case::Lower => at_low.values[i],
            Case::Upper => at_high.values[i],
            Case::Inside => density.values[i],
        })
        .collect();
    if !gradient {
        return Ok((log_probs, None));
    }
    let select = |inside: Option<&Vec<f64>>, low: Option<&Vec<f64>>, high: Option<&Vec<f64>>| {
        let (inside, low, high) = (inside?, low?, high?);
        Some(
            cases
                .iter()
                .enumerate()
                .map(|(i, case)| match case {
                    Case::Outside => 0.0,
                    Case::Lower => low[i],
                    Case::Upper => high[i],
                    Case::Inside => inside[i],
                })
                .collect::<Vec<f64>>(),
        )
    };
    let only = |wanted: Case, d: Option<&Vec<f64>>| {
        d.map(|d| {
            cases
                .iter()
                .enumerate()
                .map(|(i, &case)| if case == wanted { d[i] } else { 0.0 })
                .collect::<Vec<f64>>()
        })
    };
    let mut parameters: Vec<Option<Vec<f64>>> = (0..density.parameters.len())
        .map(|j| {
            select(
                density.parameters[j].as_ref(),
                at_low.parameters[j].as_ref(),
                at_high.parameters[j].as_ref(),
            )
        })
        .collect();
    if bounds.discrete {
        parameters.extend([None, None]);
    } else {
        parameters.push(only(Case::Lower, at_low.x.as_ref()));
        parameters.push(only(Case::Upper, at_high.x.as_ref()));
    }
    let x_gradient = only(Case::Inside, density.x.as_ref());
    Ok((log_probs, Some(parts(shape, x_gradient, parameters)?)))
}

/// The support of a truncated or censored distribution: discrete for a discrete base, else the base's support
/// intersected with the bounds.
pub(crate) fn support(base: &DistributionKind, parameters: &[&Value]) -> Result<Support> {
    let bounds = Bounds::new(base, parameters)?;
    let base_support = base.support(bounds.base)?;
    if bounds.discrete {
        return Ok(Support::Discrete);
    }
    let Some((base_lower, base_upper)) = support_bounds(base_support.clone()) else {
        return Ok(base_support);
    };
    let lower = base_lower.zip_with(bounds.lower, f64::max)?;
    let upper = base_upper.zip_with(bounds.upper, f64::min)?;
    Ok(support_between(lower, upper))
}

/// The bounds of a real support, with infinite bounds as infinities: (-∞, ∞) for [`Support::Real`]; None for a
/// support that isn't a range of real numbers.
pub(crate) fn support_bounds(support: Support) -> Option<(DoubleTensor, DoubleTensor)> {
    let infinity = |v: f64| DoubleTensor::scalar(v);
    Some(match support {
        Support::Real => (infinity(f64::NEG_INFINITY), infinity(f64::INFINITY)),
        Support::GreaterThan(a) => (a, infinity(f64::INFINITY)),
        Support::LessThan(b) => (infinity(f64::NEG_INFINITY), b),
        Support::Interval(a, b) => (a, b),
        _ => return None,
    })
}

/// The real support between element-wise bounds, either of which may be infinite: [`Support::Real`],
/// [`Support::GreaterThan`], [`Support::LessThan`] or [`Support::Interval`] as every element's bounds allow.
pub(crate) fn support_between(lower: DoubleTensor, upper: DoubleTensor) -> Support {
    let all = |t: &DoubleTensor, v: f64| t.as_slice().iter().all(|&x| x == v);
    match (all(&lower, f64::NEG_INFINITY), all(&upper, f64::INFINITY)) {
        (true, true) => Support::Real,
        (false, true) => Support::GreaterThan(lower),
        (true, false) => Support::LessThan(upper),
        (false, false) => Support::Interval(lower, upper),
    }
}

/// Draws truncated values by inverting a tail when a closed-form or numerical inverse is available.
/// Other bases use rejection with a fixed proposal limit.
pub(crate) fn sample_truncated(
    base: &DistributionKind,
    shape: &[usize],
    parameters: &[&Value],
    rng: &mut KeanuRng,
) -> Result<Value> {
    if matches!(base, DistributionKind::PoissonBinomial) {
        return poisson_binomial(Wrapper::Truncated, parameters, |d| {
            Ok(Value::I64(d.sample(shape, rng)?))
        });
    }
    let bounds = Bounds::new(base, parameters)?;
    let lower = broadcast(bounds.lower, shape)?;
    let upper = broadcast(bounds.upper, shape)?;
    if !bounds.discrete {
        if let Some(value) = sample_by_inverse(&bounds, base, shape, &lower, &upper, rng)? {
            return Ok(value);
        }
    }
    let len: usize = shape.iter().product();
    let mut values = vec![0.0; len];
    let mut missing: Vec<usize> = (0..len).collect();
    for _ in 0..REJECTION_ATTEMPTS {
        let draw = value_as_f64(&base.sample(shape, bounds.base, rng)?)?;
        let draw = draw.as_slice();
        missing.retain(|&i| {
            let accepted = draw[i] >= lower[i] && draw[i] <= upper[i];
            if accepted {
                values[i] = draw[i];
            }
            !accepted
        });
        if missing.is_empty() {
            return assemble(values, shape, bounds.discrete);
        }
    }
    invalid_parameter(format!(
        "no draw from the base distribution fell within the truncation bounds in {REJECTION_ATTEMPTS} attempts; \
         the bounds hold too little of its probability to sample by rejection"
    ))
}

fn assemble(values: Vec<f64>, shape: &[usize], discrete: bool) -> Result<Value> {
    let t = DoubleTensor::new(shape, values)?;
    Ok(if discrete {
        Value::I64(t.to_int())
    } else {
        Value::F64(t)
    })
}

/// Below this log probability a tail probability is too close to the smallest normal number to interpolate and
/// invert as a plain number, so the draw is found in log space instead.
const LOG_TINY: f64 = -700.0;

/// The most iterations of the log-space inversion.
const LOG_INVERSION_ITERATIONS: usize = 200;

/// ln((1 − w)·eˡᵒ + w·eʰⁱ) for `lo <= hi` and `w` in (0, 1], without leaving log space.
fn log_interpolate(w: f64, lo: f64, hi: f64) -> f64 {
    if hi == f64::NEG_INFINITY {
        return hi;
    }
    hi + (w + (1.0 - w) * (lo - hi).exp()).ln()
}

/// Inverse CDF sampling: u uniform between F(l) and F(u) and x = F⁻¹(u), or between S(u) and S(l) with
/// x = S⁻¹(u) where the lower bound is in the upper half of the distribution, so tail truncations keep their
/// precision. Where the tail probabilities are too small to handle as plain numbers (or the closed-form inverse
/// gives no finite value), the draw solves ln F(x) = t or ln S(x) = t in log space instead, with t interpolated in
/// log space ([`solve_in_log_space`]). Joint Wiener tails always use the numerical inverse.
fn sample_by_inverse(
    bounds: &Bounds<'_, '_>,
    base: &DistributionKind,
    shape: &[usize],
    lower: &[f64],
    upper: &[f64],
    rng: &mut KeanuRng,
) -> Result<Option<Value>> {
    if !base.has_joint_tails()
        && base
            .inverse_cdf(&DoubleTensor::scalar(0.5), false, bounds.base)?
            .is_none()
    {
        return Ok(None);
    }
    let low = bound_points(base, bounds.lower, 0.0, false);
    let high = bound_points(base, bounds.upper, 0.0, false);
    let eval = |function, at: &Value, bound: &DoubleTensor, at_infinity| {
        evaluate(
            base,
            function,
            at,
            bounds.base,
            false,
            shape,
            Some((bound, at_infinity)),
        )
        .map(|e| e.values)
    };
    let cdf_low = eval(Univariate::Cdf, &low, bounds.lower, f64::NEG_INFINITY)?;
    let cdf_high = eval(Univariate::Cdf, &high, bounds.upper, 0.0)?;
    let ccdf_low = eval(Univariate::Ccdf, &low, bounds.lower, 0.0)?;
    let ccdf_high = eval(Univariate::Ccdf, &high, bounds.upper, f64::NEG_INFINITY)?;
    let len: usize = shape.iter().product();
    let u = rng.uniform(shape);
    let mut p = vec![0.5; len];
    let mut q = vec![0.5; len];
    let mut cdf_side = vec![true; len];
    // the log of the tail probability each draw is at, and whether it must be inverted in log space
    let mut targets = vec![0.0; len];
    let mut in_log_space = vec![false; len];
    for i in 0..len {
        let u = u.as_slice()[i];
        let (lo, hi) = if from_cdf(base, (cdf_low[i], cdf_high[i]), (ccdf_low[i], ccdf_high[i])) {
            let (a, b) = (cdf_low[i].exp(), cdf_high[i].exp());
            p[i] = a + u * (b - a);
            (cdf_low[i], cdf_high[i])
        } else {
            let (a, b) = (ccdf_high[i].exp(), ccdf_low[i].exp());
            q[i] = a + u * (b - a);
            cdf_side[i] = false;
            (ccdf_high[i], ccdf_low[i])
        };
        if lo.partial_cmp(&hi) != Some(std::cmp::Ordering::Less) {
            return invalid_parameter(
                "the truncation bounds hold none of the base distribution's probability".into(),
            );
        }
        // the same draw as above, ln((1 − u)·eˡᵒ + u·eʰⁱ) with u replaced by 1 − u so its weight is never 0
        targets[i] = log_interpolate(1.0 - u, lo, hi);
        in_log_space[i] = hi < LOG_TINY;
    }
    if base.has_joint_tails() {
        let values = solve_in_log_space(
            bounds,
            base,
            shape,
            (lower, upper),
            &cdf_side,
            &targets,
            &vec![true; len],
        )?;
        return assemble(values, shape, false).map(Some);
    }
    let p = DoubleTensor::new(shape, p)?;
    let q = DoubleTensor::new(shape, q)?;
    let (Some(x_cdf), Some(x_ccdf)) = (
        base.inverse_cdf(&p, false, bounds.base)?,
        base.inverse_cdf(&q, true, bounds.base)?,
    ) else {
        return Ok(None);
    };
    let (x_cdf, x_ccdf) = (broadcast(&x_cdf, shape)?, broadcast(&x_ccdf, shape)?);
    let mut values: Vec<f64> = (0..len)
        .map(|i| {
            let x = if cdf_side[i] { x_cdf[i] } else { x_ccdf[i] };
            if !x.is_finite() {
                in_log_space[i] = true;
            }
            // rounding can step just outside the bounds
            x.max(lower[i]).min(upper[i])
        })
        .collect();
    if in_log_space.iter().any(|&b| b) {
        let solved = solve_in_log_space(
            bounds,
            base,
            shape,
            (lower, upper),
            &cdf_side,
            &targets,
            &in_log_space,
        )?;
        for i in 0..len {
            if in_log_space[i] {
                values[i] = solved[i];
            }
        }
    }
    assemble(values, shape, false).map(Some)
}

/// Solves ln F(x) = t (where `from_cdf`) or ln S(x) = t for the elements of `todo`, each between its bounds, by
/// Newton's method on the log tail probability (whose derivative is ±p(x)/F(x) or ±p(x)/S(x), from the base's log
/// density and log tail probability) safeguarded by bisection of a bracket. An error if an element doesn't converge
/// to a finite value, never an infinite or clipped one.
fn solve_in_log_space(
    bounds: &Bounds<'_, '_>,
    base: &DistributionKind,
    shape: &[usize],
    (lower, upper): (&[f64], &[f64]),
    from_cdf: &[bool],
    targets: &[f64],
    todo: &[bool],
) -> Result<Vec<f64>> {
    let len = targets.len();
    let mut lo = lower.to_vec();
    let mut hi = upper.to_vec();
    // start between the bounds, or at the finite one
    let mut x: Vec<f64> = (0..len)
        .map(|i| match (lower[i].is_finite(), upper[i].is_finite()) {
            (true, true) => lower[i] + (upper[i] - lower[i]) / 2.0,
            (true, false) => lower[i],
            (false, true) => upper[i],
            (false, false) => 0.0,
        })
        .collect();
    let mut active: Vec<bool> = todo.to_vec();
    let evaluate_at = |function, x: &[f64]| -> Result<Vec<f64>> {
        let at = Value::F64(DoubleTensor::new(shape, x.to_vec())?);
        Ok(evaluate(base, function, &at, bounds.base, false, shape, None)?.values)
    };
    for _ in 0..LOG_INVERSION_ITERATIONS {
        if !active.iter().any(|&a| a) {
            break;
        }
        let log_cdf = evaluate_at(Univariate::Cdf, &x)?;
        let log_ccdf = evaluate_at(Univariate::Ccdf, &x)?;
        let log_density = evaluate_at(Univariate::Density, &x)?;
        for i in 0..len {
            if !active[i] {
                continue;
            }
            let (tail, sign) = if from_cdf[i] {
                (log_cdf[i], 1.0)
            } else {
                (log_ccdf[i], -1.0)
            };
            let g = tail - targets[i];
            if g == 0.0 {
                active[i] = false;
                continue;
            }
            // the log tail increases with x for the CDF and decreases for the survival function
            if (g > 0.0) == (sign > 0.0) {
                hi[i] = x[i];
            } else {
                lo[i] = x[i];
            }
            let slope = sign * (log_density[i] - tail).exp();
            let mut next = x[i] - g / slope;
            if !(next > lo[i] && next < hi[i]) {
                next = match (lo[i].is_finite(), hi[i].is_finite()) {
                    (true, true) => lo[i] + (hi[i] - lo[i]) / 2.0,
                    (true, false) => x[i] + x[i].abs().max(1.0),
                    (false, true) => x[i] - x[i].abs().max(1.0),
                    (false, false) => f64::NAN,
                };
            }
            let tolerance = 4.0 * f64::EPSILON * next.abs().max(1.0);
            let converged = (next - x[i]).abs() <= tolerance || hi[i] - lo[i] <= tolerance;
            x[i] = next;
            if !next.is_finite() || converged {
                active[i] = false;
            }
        }
    }
    for i in (0..len).filter(|&i| todo[i]) {
        if active[i] || !(x[i] >= lower[i] && x[i] <= upper[i] && x[i].is_finite()) {
            return invalid_parameter(format!(
                "could not draw from the truncation to [{}, {}]: its probability is too far in the base \
                 distribution's tail to invert",
                lower[i], upper[i]
            ));
        }
    }
    Ok(x)
}

/// Draws a censored value of `shape`: a draw from the base, with values beyond a bound recorded as the bound.
pub(crate) fn sample_censored(
    base: &DistributionKind,
    shape: &[usize],
    parameters: &[&Value],
    rng: &mut KeanuRng,
) -> Result<Value> {
    if matches!(base, DistributionKind::PoissonBinomial) {
        return poisson_binomial(Wrapper::Censored, parameters, |d| {
            Ok(Value::I64(d.sample(shape, rng)?))
        });
    }
    let bounds = Bounds::new(base, parameters)?;
    let lower = broadcast(bounds.lower, shape)?;
    let upper = broadcast(bounds.upper, shape)?;
    let draw = value_as_f64(&base.sample(shape, bounds.base, rng)?)?;
    let values = draw
        .as_slice()
        .iter()
        .enumerate()
        .map(|(i, &x)| x.max(lower[i]).min(upper[i]))
        .collect();
    assemble(values, shape, bounds.discrete)
}

// ---------------------------------------------------------------------------------------------------------------
// second derivatives

/// The second-order parts of a truncated or censored distribution at `value` broadcast to `shape`, along the
/// inputs' `tangents` (the value's, each base parameter's, then the bounds'; see
/// [`distribution_hvp::elementwise`]).
///
/// Inside the bounds, a truncated log density is ln p(x) − ln Z with Z = e^P − e^Q, P and Q the base's log CDF at
/// the upper and lower bound (or its log survival function at the lower and upper bound, as [`normaliser`]
/// chooses). With w_P = e^P / Z and w_Q = e^Q / Z, ∇ ln Z = w_P ∇P − w_Q ∇Q and
/// ∇² ln Z = w_P (∇²P + ∇P ∇Pᵀ) − w_Q (∇²Q + ∇Q ∇Qᵀ) − ∇ ln Z ∇ ln Zᵀ, where P and Q depend on the base parameters
/// and their own bound. A censored value at a bound has the base's log CDF (or log survival function) there, a
/// function of the base parameters and that bound. The log CDFs' second derivatives come from
/// [`rules::cdf_rule`], which has them with respect to every differentiable parameter (the shapes of the gamma,
/// beta and negative binomial CDFs included).
///
/// # Errors
///
/// [`GraphError::NoSecondOrderRule`] naming the base if it has no log density or log CDF rule. A transformed base
/// has the rules of the distribution it transforms (see [`transformed::wrapped_second_order`]).
///
/// [`transformed::wrapped_second_order`]: super::transformed::wrapped_second_order
pub(crate) fn second_order(
    wrapper: Wrapper,
    base: &DistributionKind,
    value: &Value,
    parameters: &[&Value],
    tangents: &[Option<Vec<f64>>],
    b: usize,
    shape: &[usize],
) -> Result<Elementwise> {
    if matches!(base, DistributionKind::Transformed(..)) {
        return super::transformed::wrapped_second_order(
            wrapper, base, value, parameters, tangents, b, shape,
        );
    }
    let bounds = Bounds::unordered(base, parameters)?;
    let m = bounds.base.len();
    let (lower_index, upper_index) = (m + 1, m + 2);
    let (Some(density_rule), Some(cdf_rule)) = (rules::density_rule(base), rules::cdf_rule(base))
    else {
        return Err(GraphError::NoSecondOrderRule(format!(
            "the {} distribution's CDF",
            base.name()
        )));
    };
    let (values, parts) = elementwise(wrapper, base, value, parameters, true)?;
    let values = distribution_hvp::spread(&values, shape)?;
    let (x_gradient, parameter_gradients) =
        parts.unwrap_or_else(|| (None, smallvec::smallvec![None; parameters.len()]));
    let gradients: Vec<Option<Vec<f64>>> = std::iter::once(x_gradient)
        .chain(parameter_gradients)
        .map(|g| g.map(|g| distribution_hvp::spread(&g, shape)).transpose())
        .collect::<Result<_>>()?;
    let len = values.len();
    let x = broadcast(&value_as_f64(value)?, shape)?;
    let base_inputs: Vec<Vec<f64>> = bounds
        .base
        .iter()
        .map(|p| broadcast(&distribution_hvp::as_f64(p), shape))
        .collect::<Result<_>>()?;
    let lower = broadcast(bounds.lower, shape)?;
    let upper = broadcast(bounds.upper, shape)?;
    let eval = |function, at: &Value, infinite| {
        evaluate(base, function, at, bounds.base, true, shape, infinite)
    };
    let points = |at: &Value| broadcast(&distribution_hvp::as_f64(at), shape);

    // the log CDFs needed, with their points and the input of the bound each is at
    struct Side {
        evaluated: Evaluated,
        points: Vec<f64>,
        /// Whether this endpoint needs its tail Hessian. Joint tails can depend on parameters at infinity.
        use_rule: Vec<bool>,
        bound: usize,
        upper: bool,
    }
    let use_rule = |bound: &DoubleTensor| -> Result<Vec<bool>> {
        Ok(broadcast(bound, shape)?
            .into_iter()
            .map(|b| b.is_finite() || base.has_joint_tails())
            .collect())
    };
    let shift = if bounds.discrete { -1.0 } else { 0.0 };
    #[derive(Clone, Copy, PartialEq)]
    enum Case {
        Outside,
        Lower,
        Upper,
        Inside,
    }
    let (sides, cases) = match wrapper {
        Wrapper::Truncated => {
            let low = bound_points(base, bounds.lower, shift, bounds.discrete);
            let high = bound_points(base, bounds.upper, 0.0, bounds.discrete);
            let side = |function, at: &Value, bound, upper, infinite, bound_values| {
                Ok::<_, GraphError>(Side {
                    evaluated: eval(function, at, Some(infinite))?,
                    points: points(at)?,
                    use_rule: use_rule(bound_values)?,
                    bound,
                    upper,
                })
            };
            // at an infinite lower bound F = 0 and S = 1, at an infinite upper bound F = 1 and S = 0
            let sides = vec![
                side(
                    Univariate::Cdf,
                    &low,
                    lower_index,
                    false,
                    (bounds.lower, f64::NEG_INFINITY),
                    bounds.lower,
                )?,
                side(
                    Univariate::Ccdf,
                    &low,
                    lower_index,
                    true,
                    (bounds.lower, 0.0),
                    bounds.lower,
                )?,
                side(
                    Univariate::Cdf,
                    &high,
                    upper_index,
                    false,
                    (bounds.upper, 0.0),
                    bounds.upper,
                )?,
                side(
                    Univariate::Ccdf,
                    &high,
                    upper_index,
                    true,
                    (bounds.upper, f64::NEG_INFINITY),
                    bounds.upper,
                )?,
            ];
            let cases = (0..len)
                .map(|i| {
                    if ordered(lower[i], upper[i]) && x[i] >= lower[i] && x[i] <= upper[i] {
                        Case::Inside
                    } else {
                        Case::Outside
                    }
                })
                .collect::<Vec<_>>();
            (sides, cases)
        }
        Wrapper::Censored => {
            let low = bound_points(base, bounds.lower, 0.0, bounds.discrete);
            let high = bound_points(base, bounds.upper, shift, bounds.discrete);
            let sides = vec![
                Side {
                    evaluated: eval(Univariate::Cdf, &low, None)?,
                    points: points(&low)?,
                    use_rule: use_rule(bounds.lower)?,
                    bound: lower_index,
                    upper: false,
                },
                Side {
                    evaluated: eval(Univariate::Ccdf, &high, None)?,
                    points: points(&high)?,
                    use_rule: use_rule(bounds.upper)?,
                    bound: upper_index,
                    upper: true,
                },
            ];
            let cases: Vec<Case> = (0..len)
                .map(|i| {
                    if !ordered(lower[i], upper[i]) {
                        Case::Outside
                    } else if x[i] == lower[i] && lower[i].is_finite() {
                        Case::Lower
                    } else if x[i] == upper[i] && upper[i].is_finite() {
                        Case::Upper
                    } else if x[i] > lower[i] && x[i] < upper[i] {
                        Case::Inside
                    } else {
                        Case::Outside
                    }
                })
                .collect();
            (sides, cases)
        }
    };
    let differentiable: Vec<bool> = gradients.iter().map(Option::is_some).collect();

    let n = m + 3;
    let mut point = vec![0.0; m];
    let mut cdf = Hessian::new(m + 1);
    let mut cdf_gradient = vec![0.0; m + 1];
    // a side's gradient at element i over the wrapper's inputs, and its Hessian (into `cdf`, inputs 0..=m with 0
    // the point)
    let mut side_derivatives =
        |side: &Side, i: usize, point: &[f64], cdf: &mut Hessian| -> [f64; MAX_INPUTS] {
            let mut g = [0.0; MAX_INPUTS];
            let e = &side.evaluated;
            cdf_gradient[0] = e.x.as_ref().map_or(0.0, |d| d[i]);
            g[side.bound] = cdf_gradient[0];
            for j in 0..m {
                cdf_gradient[j + 1] = e.parameters[j].as_ref().map_or(0.0, |d| d[i]);
                g[j + 1] = cdf_gradient[j + 1];
            }
            cdf.clear();
            if e.values[i].is_finite() && side.use_rule[i] {
                cdf_rule(side.points[i], point, side.upper, &cdf_gradient, cdf);
            }
            g
        };
    // adds `scale` times a side's Hessian, with its point moved to its bound
    let embed = |h: &mut Hessian, cdf: &Hessian, bound: usize, scale: f64| {
        let at = |a: usize| if a == 0 { bound } else { a };
        for a in 0..=m {
            for c in a..=m {
                h.add(at(a), at(c), scale * cdf.get(a, c));
            }
        }
    };
    let hvps = distribution_hvp::contract(&differentiable, tangents, len, b, |i, h| {
        for (p, input) in point.iter_mut().zip(&base_inputs) {
            *p = input[i];
        }
        match (wrapper, cases[i]) {
            (_, Case::Outside) => {}
            (Wrapper::Censored, Case::Inside) => density_rule(x[i], &point, h),
            (Wrapper::Censored, case) => {
                let side = &sides[usize::from(case == Case::Upper)];
                side_derivatives(side, i, &point, &mut cdf);
                embed(h, &cdf, side.bound, 1.0);
            }
            (Wrapper::Truncated, _) => {
                // Use the same tail as the first-order normalizer.
                let (big, small) = if from_cdf(
                    base,
                    (sides[0].evaluated.values[i], sides[2].evaluated.values[i]),
                    (sides[1].evaluated.values[i], sides[3].evaluated.values[i]),
                ) {
                    (&sides[2], &sides[0])
                } else {
                    (&sides[1], &sides[3])
                };
                let (p, q) = (big.evaluated.values[i], small.evaluated.values[i]);
                let log_z = if p == f64::NEG_INFINITY {
                    f64::NEG_INFINITY
                } else {
                    p + log1mexp((q - p).min(0.0))
                };
                if log_z == f64::NEG_INFINITY {
                    return Ok(());
                }
                density_rule(x[i], &point, h);
                let mut g = [0.0; MAX_INPUTS];
                for (side, weight, sign) in [
                    (big, (p - log_z).exp(), 1.0),
                    (small, (q - log_z).exp(), -1.0),
                ] {
                    if weight == 0.0 {
                        continue;
                    }
                    let gs = side_derivatives(side, i, &point, &mut cdf);
                    // − (±w (∇²G + ∇G ∇Gᵀ))
                    embed(h, &cdf, side.bound, -sign * weight);
                    for a in 0..n {
                        for c in a..n {
                            h.add(a, c, -sign * weight * gs[a] * gs[c]);
                        }
                        g[a] += sign * weight * gs[a];
                    }
                }
                for a in 0..n {
                    for c in a..n {
                        h.add(a, c, g[a] * g[c]);
                    }
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
