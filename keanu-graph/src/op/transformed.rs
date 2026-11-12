//! Transformed distributions: a base distribution pushed through an element-wise bijector
//! ([`DistributionKind::Transformed`]).
//!
//! The value is x = f(y) for a draw y of the base, where f is the forward map of an [`ElementBijector`] (the
//! element-wise transforms of the [`transform`](crate::transform) module's bounded bijectors, plus an affine map).
//! With g = f⁻¹, the log density is
//!
//! log q(x) = log p(g(x)) + log |g′(x)|,
//!
//! and its derivatives follow by the chain rule from the base's element-wise rules:
//!
//! - ∂/∂x = p_y g′ + g″/g′, the parameters' derivatives being the base's at g(x);
//! - ∂²/∂x² = p_yy g′² + p_y g″ + (g‴g′ − g″²)/g′², ∂²/∂x∂θ = p_yθ g′, and ∂²/∂θ∂θ′ the base's.
//!
//! Values outside the bijector's image are impossible (log density −∞, with zero derivatives). The support is the
//! image of the base's support, so the base's support must be a range of real numbers.

use super::distribution::{DistributionKind, LogProbGradientParts, Univariate};
use super::distribution_hvp::{self, Elementwise};
use super::{Support, broadcast, invalid_parameter, wrappers};
use crate::error::Result;
use crate::value::Value;
use keanu_distributions::KeanuRng;
use keanu_tensor::DoubleTensor;

/// An element-wise bijector for a [transformed distribution](DistributionKind::Transformed): a smooth, strictly
/// monotone map f from the real line onto an interval, applied to each element of a base distribution's draws.
///
/// The bounded forms are those of the [`transform`](crate::transform) module's [`LowerBound`], [`UpperBound`] and
/// [`Interval`] bijectors, with constant scalar bounds:
///
/// | Bijector | Forward map f(y) | Inverse g(x) | log \|g′(x)\| |
/// |---|---|---|---|
/// | `LowerBound(a)` | a + exp(y) | ln(x − a) | −ln(x − a) |
/// | `UpperBound(b)` | b − exp(y) | ln(b − x) | −ln(b − x) |
/// | `Interval(a, b)` | a + (b − a) logistic(y) | ln(x − a) − ln(b − x) | ln(b − a) − ln(x − a) − ln(b − x) |
/// | `Affine { loc, scale }` | loc + scale · y | (x − loc) / scale | −ln \|scale\| |
///
/// So a Gaussian through `LowerBound(0.0)` is a log-normal, and through `Interval(0.0, 1.0)` a logit-normal.
///
/// [`LowerBound`]: crate::transform::LowerBound
/// [`UpperBound`]: crate::transform::UpperBound
/// [`Interval`]: crate::transform::Interval
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum ElementBijector {
    /// x = a + exp(y), onto (a, ∞).
    LowerBound(f64),
    /// x = b − exp(y), onto (−∞, b).
    UpperBound(f64),
    /// x = a + (b − a) logistic(y), onto (a, b).
    Interval(f64, f64),
    /// x = loc + scale · y, onto the real line (decreasing for a negative scale).
    Affine {
        /// The shift.
        loc: f64,
        /// The scale, finite and non-zero.
        scale: f64,
    },
}

/// The inverse map's value and first three derivatives at a point, and the log absolute first derivative.
#[derive(Debug, Clone, Copy)]
struct Inverse {
    y: f64,
    d1: f64,
    d2: f64,
    d3: f64,
    log_d1: f64,
}

impl ElementBijector {
    /// Why the bijector is invalid (a NaN or infinite bound, an empty interval, a zero or non-finite scale), if
    /// it is.
    pub(crate) fn check(self) -> std::result::Result<(), String> {
        let ok = match self {
            ElementBijector::LowerBound(a) | ElementBijector::UpperBound(a) => a.is_finite(),
            ElementBijector::Interval(a, b) => a.is_finite() && b.is_finite() && a < b,
            ElementBijector::Affine { loc, scale } => {
                loc.is_finite() && scale.is_finite() && scale != 0.0
            }
        };
        if ok {
            Ok(())
        } else {
            Err(format!(
                "the bijector {self:?} needs finite bounds (lower below upper) or a finite location and a finite, \
                 non-zero scale"
            ))
        }
    }

    /// The forward map f(y), with f(±∞) the ends of the image.
    pub fn forward(self, y: f64) -> f64 {
        match self {
            ElementBijector::LowerBound(a) => a + y.exp(),
            ElementBijector::UpperBound(b) => b - y.exp(),
            // computed from the nearer bound, so values near either keep their precision
            ElementBijector::Interval(a, b) if y > 0.0 => b - (b - a) * logistic(-y),
            ElementBijector::Interval(a, b) => a + (b - a) * logistic(y),
            ElementBijector::Affine { loc, scale } => loc + scale * y,
        }
    }

    /// Whether the forward map is increasing (it is decreasing otherwise).
    pub fn is_increasing(self) -> bool {
        match self {
            ElementBijector::UpperBound(_) => false,
            ElementBijector::Affine { scale, .. } => scale > 0.0,
            ElementBijector::LowerBound(_) | ElementBijector::Interval(..) => true,
        }
    }

    /// The inverse map at x and its derivatives, None where x is outside the image (or NaN).
    fn inverse(self, x: f64) -> Option<Inverse> {
        let inverse = match self {
            ElementBijector::LowerBound(a) => {
                let u = x - a;
                (u > 0.0).then(|| Inverse {
                    y: u.ln(),
                    d1: 1.0 / u,
                    d2: -1.0 / (u * u),
                    d3: 2.0 / (u * u * u),
                    log_d1: -u.ln(),
                })?
            }
            ElementBijector::UpperBound(b) => {
                let u = b - x;
                (u > 0.0).then(|| Inverse {
                    y: u.ln(),
                    d1: -1.0 / u,
                    d2: -1.0 / (u * u),
                    d3: -2.0 / (u * u * u),
                    log_d1: -u.ln(),
                })?
            }
            ElementBijector::Interval(a, b) => {
                let (u, v) = (x - a, b - x);
                (u > 0.0 && v > 0.0).then(|| Inverse {
                    y: u.ln() - v.ln(),
                    d1: 1.0 / u + 1.0 / v,
                    d2: -1.0 / (u * u) + 1.0 / (v * v),
                    d3: 2.0 / (u * u * u) + 2.0 / (v * v * v),
                    log_d1: (b - a).ln() - u.ln() - v.ln(),
                })?
            }
            ElementBijector::Affine { loc, scale } => {
                let y = (x - loc) / scale;
                y.is_finite().then(|| Inverse {
                    y,
                    d1: 1.0 / scale,
                    d2: 0.0,
                    d3: 0.0,
                    log_d1: -scale.abs().ln(),
                })?
            }
        };
        inverse.y.is_finite().then_some(inverse)
    }
}

/// `1 / (1 + exp(-x))` without overflow.
fn logistic(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Checks that `base` can be transformed: an element-wise kind with f64 values and a density (not a censored or
/// hurdle kind, whose point masses have no Jacobian), and a valid bijector.
pub(crate) fn check(
    base: &DistributionKind,
    bijector: ElementBijector,
) -> std::result::Result<(), String> {
    bijector.check()?;
    if let DistributionKind::Transformed(inner, bijector) = base {
        check(inner, *bijector)?;
    }
    let point_masses = matches!(
        base,
        DistributionKind::Censored(_) | DistributionKind::Hurdle(_)
    );
    if !base.is_elementwise() || base.value_type() != crate::value::DType::F64 || point_masses {
        return Err(format!(
            "the base distribution ({}) must be a continuous element-wise distribution with a density (one f64 \
             value per element, not censored or a hurdle), so it cannot be transformed",
            base.name()
        ));
    }
    Ok(())
}

/// Each element of `x` (of x's shape): the inverse map's parts, None outside the image.
fn inverses(bijector: ElementBijector, x: &DoubleTensor) -> Vec<Option<Inverse>> {
    x.as_slice().iter().map(|&x| bijector.inverse(x)).collect()
}

/// The base's points g(x), as a value of x's shape, with 0 where x is outside the image (those elements are
/// masked afterwards).
fn base_points(x: &DoubleTensor, inverses: &[Option<Inverse>]) -> Result<Value> {
    Ok(Value::F64(DoubleTensor::new(
        x.shape(),
        inverses.iter().map(|i| i.map_or(0.0, |i| i.y)).collect(),
    )?))
}

/// One of the inverse's parts at each element, as a tensor of x's shape (`missing` outside the image).
fn part(
    x: &DoubleTensor,
    inverses: &[Option<Inverse>],
    missing: f64,
    f: impl Fn(&Inverse) -> f64,
) -> Result<DoubleTensor> {
    Ok(DoubleTensor::new(
        x.shape(),
        inverses
            .iter()
            .map(|i| i.as_ref().map_or(missing, &f))
            .collect(),
    )?)
}

/// The log density of each element of `value` (broadcast with the parameters) and, with `gradient`, its
/// element-wise gradient parts.
pub(crate) fn elementwise(
    base: &DistributionKind,
    bijector: ElementBijector,
    value: &Value,
    parameters: &[&Value],
    gradient: bool,
) -> Result<(DoubleTensor, Option<LogProbGradientParts>)> {
    evaluate(
        base,
        bijector,
        Univariate::Density,
        value,
        parameters,
        gradient,
    )
}

/// Evaluate a density or tail through the inverse map. Only a density acquires a Jacobian term.
pub(crate) fn evaluate(
    base: &DistributionKind,
    bijector: ElementBijector,
    function: Univariate,
    value: &Value,
    parameters: &[&Value],
    gradient: bool,
) -> Result<(DoubleTensor, Option<LogProbGradientParts>)> {
    let x = value.f64()?;
    let inverses = inverses(bijector, x);
    let y = base_points(x, &inverses)?;
    let base_function = match (function, bijector.is_increasing()) {
        (Univariate::Cdf, false) => Univariate::Ccdf,
        (Univariate::Ccdf, false) => Univariate::Cdf,
        _ => function,
    };
    let (terms, parts) = base.univariate(base_function, &y, parameters, gradient)?;
    let density = function == Univariate::Density;
    let shape = keanu_tensor::broadcast_shape(terms.shape(), x.shape())?;
    let inside = broadcast(&part(x, &inverses, 0.0, |_| 1.0)?, &shape)?;
    let mut values = broadcast(&terms, &shape)?;
    if density {
        let log_d1 = broadcast(&part(x, &inverses, 0.0, |i| i.log_d1)?, &shape)?;
        for (i, value) in values.iter_mut().enumerate() {
            *value = if inside[i] == 1.0 {
                *value + log_d1[i]
            } else {
                f64::NEG_INFINITY
            };
        }
    } else {
        let points = broadcast(x, &shape)?;
        for (i, value) in values.iter_mut().enumerate() {
            if inside[i] == 0.0 {
                let x = points[i];
                let below = match bijector {
                    ElementBijector::LowerBound(a) | ElementBijector::Interval(a, _) => x <= a,
                    ElementBijector::UpperBound(b) => x < b,
                    ElementBijector::Affine { loc, .. } => x < loc,
                };
                *value = if x.is_nan() {
                    f64::NAN
                } else if below == (function == Univariate::Cdf) {
                    f64::NEG_INFINITY
                } else {
                    0.0
                };
            }
        }
    }
    let values = DoubleTensor::new(&shape, values)?;
    let Some((dy, dparameters)) = parts.filter(|_| gradient) else {
        return Ok((values, None));
    };
    // zero outside the image
    let masked = |g: Vec<f64>| -> Vec<f64> {
        g.into_iter()
            .zip(&inside)
            .map(|(g, &inside)| if inside == 1.0 { g } else { 0.0 })
            .collect()
    };
    let dx = dy
        .map(|dy| -> Result<DoubleTensor> {
            let d1 = broadcast(&part(x, &inverses, 1.0, |i| i.d1)?, &shape)?;
            let mut dx = broadcast(&dy, &shape)?;
            for (dx, &d1) in dx.iter_mut().zip(&d1) {
                *dx *= d1;
            }
            if density {
                let d2 = broadcast(&part(x, &inverses, 0.0, |i| i.d2)?, &shape)?;
                for ((dx, d2), d1) in dx.iter_mut().zip(d2).zip(d1) {
                    *dx += d2 / d1;
                }
            }
            Ok(DoubleTensor::new(&shape, masked(dx))?)
        })
        .transpose()?;
    let dparameters = dparameters
        .into_iter()
        .map(|d| {
            d.map(|d| -> Result<DoubleTensor> {
                Ok(DoubleTensor::new(&shape, masked(broadcast(&d, &shape)?))?)
            })
            .transpose()
        })
        .collect::<Result<_>>()?;
    Ok((values, Some((dx, dparameters))))
}

/// The second-order parts of a transformed kind at `value` broadcast with the parameters to `shape`, along the
/// inputs' `tangents` (see [`distribution_hvp::elementwise`]): the base's at g(x), with the value's tangent
/// scaled by g′(x), then mapped back to x by the chain rule of the module documentation.
pub(crate) fn second_order(
    base: &DistributionKind,
    bijector: ElementBijector,
    value: &Value,
    parameters: &[&Value],
    tangents: &[Option<Vec<f64>>],
    b: usize,
    shape: &[usize],
) -> Result<Elementwise> {
    let x = value.f64()?;
    let inverses = inverses(bijector, x);
    let y = base_points(x, &inverses)?;
    let len: usize = shape.iter().product();
    let inside = broadcast(&part(x, &inverses, 0.0, |_| 1.0)?, shape)?;
    let d1 = broadcast(&part(x, &inverses, 1.0, |i| i.d1)?, shape)?;
    let d2 = broadcast(&part(x, &inverses, 0.0, |i| i.d2)?, shape)?;
    let d3 = broadcast(&part(x, &inverses, 0.0, |i| i.d3)?, shape)?;
    let log_d1 = broadcast(&part(x, &inverses, 0.0, |i| i.log_d1)?, shape)?;
    // ẏ = g′(x) ẋ
    let mut base_tangents = tangents.to_vec();
    if let Some(Some(t)) = base_tangents.first_mut() {
        for (j, t) in t.iter_mut().enumerate() {
            *t *= d1[j % len];
        }
    }
    let parts = distribution_hvp::elementwise(base, &y, parameters, &base_tangents, b, shape)?;
    let mut values = parts.values;
    for (i, v) in values.iter_mut().enumerate() {
        *v = if inside[i] == 1.0 {
            *v + log_d1[i]
        } else {
            f64::NEG_INFINITY
        };
    }
    let mut gradients = parts.gradients;
    let mut hvps = parts.hvps;
    let dy = gradients[0].clone();
    if let Some(g) = gradients.first_mut().and_then(Option::as_mut) {
        for i in 0..len {
            g[i] = g[i] * d1[i] + d2[i] / d1[i];
        }
    }
    if let (Some(h), Some(dy)) = (hvps.first_mut().and_then(Option::as_mut), &dy) {
        let tx = tangents.first().and_then(Option::as_ref);
        for k in 0..b {
            for i in 0..len {
                let j = k * len + i;
                let curvature = dy[i] * d2[i] + (d3[i] * d1[i] - d2[i] * d2[i]) / (d1[i] * d1[i]);
                h[j] = d1[i] * h[j] + tx.map_or(0.0, |t| curvature * t[j]);
            }
        }
    }
    // zero outside the image
    for g in gradients.iter_mut().flatten() {
        for (g, &inside) in g.iter_mut().zip(&inside) {
            if inside != 1.0 {
                *g = 0.0;
            }
        }
    }
    for h in hvps.iter_mut().flatten() {
        for (j, h) in h.iter_mut().enumerate() {
            if inside[j % len] != 1.0 {
                *h = 0.0;
            }
        }
    }
    Ok(Elementwise {
        values,
        gradients,
        hvps,
    })
}

/// The base's point for a bound `x` of a truncated or censored transformed kind: g(x) inside the image, and the
/// matching end of the base's real line (a constant) beyond it.
fn bound_point(bijector: ElementBijector, x: f64) -> (f64, Option<Inverse>) {
    if let Some(inverse) = bijector.inverse(x) {
        return (inverse.y, Some(inverse));
    }
    if x.is_nan() {
        return (f64::NAN, None);
    }
    let below = match bijector {
        ElementBijector::LowerBound(a) | ElementBijector::Interval(a, _) => x <= a,
        ElementBijector::UpperBound(b) => x < b,
        ElementBijector::Affine { loc, .. } => x < loc,
    };
    let y = if below == bijector.is_increasing() {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    };
    (y, None)
}

/// The second-order parts of a truncated or censored transformed kind (`wrapper` around `base` pushed through
/// `bijector`) at `value` broadcast to `shape`, along the inputs' `tangents`: the value, the base parameters, then
/// the lower and upper bound.
///
/// The value and both bounds map through g to the wrapper of the base (with the bounds swapped for a decreasing
/// map), whose second-order parts give those at x by the chain rule: for each point-like input v with u = g(v),
/// (H_v t)_v = g′(v) (H_u (g′ ⊙ t))_u + ∂_u F g″(v) t_v. Where the value has a density, its Jacobian log g′(x)
/// adds (g‴g′ − g″²)/g′² ẋ. Values, gradients and the zero parts outside the image come from first order.
pub(crate) fn wrapped_second_order(
    wrapper: wrappers::Wrapper,
    transformed: &DistributionKind,
    value: &Value,
    parameters: &[&Value],
    tangents: &[Option<Vec<f64>>],
    b: usize,
    shape: &[usize],
) -> Result<Elementwise> {
    let DistributionKind::Transformed(base, bijector) = transformed else {
        return Err(crate::error::GraphError::NoSecondOrderRule(format!(
            "the {} distribution is not transformed",
            transformed.name()
        )));
    };
    let bijector = *bijector;
    let (base_parameters, own) = wrappers::split(parameters, 2)?;
    let m = base_parameters.len();
    let (lower_index, upper_index) = (m + 1, m + 2);
    let len: usize = shape.iter().product();
    let increasing = bijector.is_increasing();
    // the inputs in the base's coordinates: the image of x, then of each bound
    // the base's points, then g′, g″ and g‴ broadcast to `shape`
    type Mapped = (Value, Vec<f64>, Vec<f64>, Vec<f64>);
    let map = |x: &DoubleTensor| -> Result<Mapped> {
        let mapped: Vec<(f64, Option<Inverse>)> = x
            .as_slice()
            .iter()
            .map(|&x| bound_point(bijector, x))
            .collect();
        let field = |f: &dyn Fn(&Option<Inverse>) -> f64| -> Result<Vec<f64>> {
            broadcast(
                &DoubleTensor::new(x.shape(), mapped.iter().map(|(_, i)| f(i)).collect())?,
                shape,
            )
        };
        let d1 = field(&|i| i.map_or(0.0, |i| i.d1))?;
        let d2 = field(&|i| i.map_or(0.0, |i| i.d2))?;
        let d3 = field(&|i| i.map_or(0.0, |i| i.d3))?;
        let y = Value::F64(DoubleTensor::new(
            x.shape(),
            mapped.iter().map(|(y, _)| *y).collect(),
        )?);
        Ok((y, d1, d2, d3))
    };
    let x = value.f64()?;
    let (y, x_d1, x_d2, x_d3) = map(x)?;
    let (lower, lower_d1, lower_d2, _) = map(own[0].f64()?)?;
    let (upper, upper_d1, upper_d2, _) = map(own[1].f64()?)?;
    // a decreasing map takes the lower bound to the base's upper bound
    let (base_lower, base_upper) = if increasing {
        (&lower, &upper)
    } else {
        (&upper, &lower)
    };
    let (base_lower_index, base_upper_index) = if increasing {
        (lower_index, upper_index)
    } else {
        (upper_index, lower_index)
    };
    let mut mapped_parameters: Vec<&Value> = base_parameters.to_vec();
    mapped_parameters.push(base_lower);
    mapped_parameters.push(base_upper);
    // each point-like input: its index here, its index in the base's wrapper and g′ and g″ there
    let points = [
        (0, 0, &x_d1, &x_d2),
        (lower_index, base_lower_index, &lower_d1, &lower_d2),
        (upper_index, base_upper_index, &upper_d1, &upper_d2),
    ];
    let mut base_tangents = tangents.to_vec();
    for &(here, there, d1, _) in &points {
        base_tangents[there] = tangents[here]
            .as_ref()
            .map(|t| t.iter().enumerate().map(|(j, t)| t * d1[j % len]).collect());
    }
    let parts = wrappers::second_order(
        wrapper,
        base,
        &y,
        &mapped_parameters,
        &base_tangents,
        b,
        shape,
    )?;
    let (values, first) = wrappers::elementwise(wrapper, transformed, value, parameters, true)?;
    let values = distribution_hvp::spread(&values, shape)?;
    let (x_gradient, parameter_gradients) =
        first.unwrap_or_else(|| (None, smallvec::smallvec![None; parameters.len()]));
    let gradients: Vec<Option<Vec<f64>>> = std::iter::once(x_gradient)
        .chain(parameter_gradients)
        .map(|g| g.map(|g| distribution_hvp::spread(&g, shape)).transpose())
        .collect::<Result<_>>()?;
    let mut hvps = gradients
        .iter()
        .map(|g| g.as_ref().map(|_| vec![0.0; b * len]))
        .collect::<Vec<_>>();
    for (out, h) in hvps.iter_mut().zip(&parts.hvps).take(m + 1).skip(1) {
        if let (Some(out), Some(h)) = (out.as_mut(), h.as_ref()) {
            out.copy_from_slice(h);
        }
    }
    for &(here, there, d1, d2) in &points {
        let (Some(out), Some(h)) = (hvps[here].as_mut(), parts.hvps[there].as_ref()) else {
            continue;
        };
        let base_gradient = parts.gradients[there].as_ref();
        let tangent = tangents[here].as_ref();
        for (j, out) in out.iter_mut().enumerate() {
            let i = j % len;
            let curvature = base_gradient.map_or(0.0, |g| g[i] * d2[i]);
            *out = d1[i] * h[j] + tangent.map_or(0.0, |t| curvature * t[j]);
        }
    }
    // the Jacobian of a value with a density
    if let (Some(out), Some(t)) = (hvps[0].as_mut(), tangents[0].as_ref()) {
        let lower = broadcast(own[0].f64()?, shape)?;
        let upper = broadcast(own[1].f64()?, shape)?;
        let xs = broadcast(x, shape)?;
        for (j, out) in out.iter_mut().enumerate() {
            let i = j % len;
            let mass =
                wrapper == wrappers::Wrapper::Censored && (xs[i] == lower[i] || xs[i] == upper[i]);
            if !values[i].is_finite() || x_d1[i] == 0.0 || mass {
                continue;
            }
            let (d1, d2, d3) = (x_d1[i], x_d2[i], x_d3[i]);
            *out += (d3 * d1 - d2 * d2) / (d1 * d1) * t[j];
        }
    }
    // zero where the value is impossible
    for out in hvps.iter_mut().flatten() {
        for (j, out) in out.iter_mut().enumerate() {
            if !values[j % len].is_finite() {
                *out = 0.0;
            }
        }
    }
    Ok(Elementwise {
        values,
        gradients,
        hvps,
    })
}

/// The support: the image of the base's support under the forward map.
pub(crate) fn support(
    base: &DistributionKind,
    bijector: ElementBijector,
    parameters: &[&Value],
) -> Result<Support> {
    let base_support = base.support(parameters)?;
    let Some((lower, upper)) = wrappers::support_bounds(base_support.clone()) else {
        return invalid_parameter(format!(
            "the base distribution's support ({base_support:?}) is not a range of real numbers, so it can't be \
             transformed"
        ));
    };
    let (lower, upper) = (
        lower.map(|v| bijector.forward(v)),
        upper.map(|v| bijector.forward(v)),
    );
    Ok(if bijector.is_increasing() {
        wrappers::support_between(lower, upper)
    } else {
        wrappers::support_between(upper, lower)
    })
}

/// Draws a value of `shape`: the forward map of a draw of the base.
pub(crate) fn sample(
    base: &DistributionKind,
    bijector: ElementBijector,
    shape: &[usize],
    parameters: &[&Value],
    rng: &mut KeanuRng,
) -> Result<Value> {
    let draws = base.sample(shape, parameters, rng)?;
    Ok(Value::F64(draws.f64()?.map(|y| bijector.forward(y))))
}
