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
//! parameters broadcast with the value and that axis. Other kinds wrapping a base distribution (hidden Markov model
//! emissions) use the same layout. The saved schema stores the base as a nested distribution type, recursively, so
//! any nesting the graph accepts (a mixture of truncated distributions) saves and loads.

use super::Support;
use super::distribution::{DistributionKind, LogProbGradientParts, ParameterGradients, Univariate};
use crate::error::{GraphError, Result};
use crate::value::{DType, Value};
use keanu_distributions::{DistributionError, KeanuRng};
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

fn invalid_bounds<T>(message: String) -> Result<T> {
    Err(GraphError::Distribution(
        DistributionError::InvalidParameter(message),
    ))
}

/// A wrapper's base parameters and bounds, checked: each lower bound must be less than its upper bound (neither
/// NaN), and for a discrete base each finite bound must be a whole number.
struct Bounds<'p, 'v> {
    base: &'p [&'v Value],
    lower: &'v DoubleTensor,
    upper: &'v DoubleTensor,
    discrete: bool,
}

impl<'p, 'v> Bounds<'p, 'v> {
    fn new(base_kind: &DistributionKind, parameters: &'p [&'v Value]) -> Result<Self> {
        let (base, own) = split(parameters, OWN_PARAMETERS)?;
        let (lower, upper) = (own[0].f64()?, own[1].f64()?);
        let discrete = base_kind.value_type() != DType::F64;
        let shape = broadcast_shape(lower.shape(), upper.shape())?;
        let (l, u) = (lower.broadcast_to(&shape)?, upper.broadcast_to(&shape)?);
        for (&l, &u) in l.as_slice().iter().zip(u.as_slice()) {
            // NaN bounds fail too
            if l >= u || l.is_nan() || u.is_nan() {
                return invalid_bounds(format!(
                    "the lower bound must be less than the upper bound, got [{l}, {u}]"
                ));
            }
            if discrete && [l, u].iter().any(|b| b.is_finite() && b.fract() != 0.0) {
                return invalid_bounds(format!(
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

/// A value of the base's type at each element of `bound + shift`, with infinite bounds replaced by 0 (the result
/// there is overridden or unused).
fn bound_points(bound: &DoubleTensor, shift: f64, discrete: bool) -> Value {
    let finite = bound.map(|b| if b.is_infinite() { 0.0 } else { b + shift });
    if discrete {
        Value::I64(finite.to_int())
    } else {
        Value::F64(finite)
    }
}

/// A function of the base evaluated element-wise and broadcast to the node's shape.
struct Evaluated {
    values: Vec<f64>,
    /// With respect to the point it was evaluated at (None for discrete values).
    x: Option<Vec<f64>>,
    /// With respect to each base parameter (None where there is no derivative).
    parameters: Vec<Option<Vec<f64>>>,
}

fn broadcast(t: &DoubleTensor, shape: &[usize]) -> Result<Vec<f64>> {
    Ok(if t.shape() == shape {
        t.as_slice().to_vec()
    } else {
        t.broadcast_to(shape)?.into_vec()
    })
}

/// `function` of the base at `at`, broadcast to `shape`. Where `infinite` (broadcast) holds, the value is
/// `at_infinity` and the derivatives are 0.
fn evaluate(
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
    if let Some((bound, at_infinity)) = infinite {
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
pub(crate) fn elementwise(
    wrapper: Wrapper,
    base: &DistributionKind,
    value: &Value,
    parameters: &[&Value],
    gradient: bool,
) -> Result<(DoubleTensor, Option<LogProbGradientParts>)> {
    let bounds = Bounds::new(base, parameters)?;
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

/// Element-wise log probabilities and gradient parts.
type Parts = (Vec<f64>, Option<LogProbGradientParts>);

/// Assembles gradient parts from per-element vectors.
fn parts(
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
    let low = bound_points(bounds.lower, shift, bounds.discrete);
    let high = bound_points(bounds.upper, 0.0, bounds.discrete);
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
        let (from_cdf, (low_g, high_g), sign) = if cdf_low.values[i] < -LN_2 {
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
    let low = bound_points(bounds.lower, 0.0, bounds.discrete);
    let high = bound_points(bounds.upper, shift, bounds.discrete);
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
            if x[i] == lower[i] {
                Case::Lower
            } else if x[i] == upper[i] {
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

/// Draws a truncated value of `shape`: by inverting the base's CDF where it has a closed-form inverse, else by
/// rejection (an error if fewer than all elements land within the bounds after a fixed number of attempts).
pub(crate) fn sample_truncated(
    base: &DistributionKind,
    shape: &[usize],
    parameters: &[&Value],
    rng: &mut KeanuRng,
) -> Result<Value> {
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
    invalid_bounds(format!(
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

/// Inverse CDF sampling: u uniform between F(l) and F(u) and x = F⁻¹(u), or between S(u) and S(l) with
/// x = S⁻¹(u) where the lower bound is in the upper half of the distribution, so tail truncations keep their
/// precision. None if the base has no closed-form inverse.
fn sample_by_inverse(
    bounds: &Bounds<'_, '_>,
    base: &DistributionKind,
    shape: &[usize],
    lower: &[f64],
    upper: &[f64],
    rng: &mut KeanuRng,
) -> Result<Option<Value>> {
    if base
        .inverse_cdf(&DoubleTensor::scalar(0.5), false, bounds.base)?
        .is_none()
    {
        return Ok(None);
    }
    let z = normaliser(bounds, base, false, shape)?;
    let low = bound_points(bounds.lower, 0.0, false);
    let high = bound_points(bounds.upper, 0.0, false);
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
    let mut from_cdf = vec![true; len];
    for i in 0..len {
        if z.log_z[i] == f64::NEG_INFINITY {
            return invalid_bounds(
                "the truncation bounds hold none of the base distribution's probability".into(),
            );
        }
        let u = u.as_slice()[i];
        if cdf_low[i] < -LN_2 {
            let (a, b) = (cdf_low[i].exp(), cdf_high[i].exp());
            p[i] = a + u * (b - a);
        } else {
            let (a, b) = (ccdf_high[i].exp(), ccdf_low[i].exp());
            q[i] = a + u * (b - a);
            from_cdf[i] = false;
        }
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
    let values = (0..len)
        .map(|i| {
            let x = if from_cdf[i] { x_cdf[i] } else { x_ccdf[i] };
            // rounding can step just outside the bounds
            x.max(lower[i]).min(upper[i])
        })
        .collect();
    assemble(values, shape, false).map(Some)
}

/// Draws a censored value of `shape`: a draw from the base, with values beyond a bound recorded as the bound.
pub(crate) fn sample_censored(
    base: &DistributionKind,
    shape: &[usize],
    parameters: &[&Value],
    rng: &mut KeanuRng,
) -> Result<Value> {
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
