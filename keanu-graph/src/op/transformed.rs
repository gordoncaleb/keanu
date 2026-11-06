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
    let x = value.f64()?;
    let inverses = inverses(bijector, x);
    let y = base_points(x, &inverses)?;
    let (densities, parts) = base.univariate(Univariate::Density, &y, parameters, gradient)?;
    let shape = keanu_tensor::broadcast_shape(densities.shape(), x.shape())?;
    let inside = broadcast(&part(x, &inverses, 0.0, |_| 1.0)?, &shape)?;
    let log_d1 = broadcast(&part(x, &inverses, 0.0, |i| i.log_d1)?, &shape)?;
    let values: Vec<f64> = broadcast(&densities, &shape)?
        .into_iter()
        .zip(&log_d1)
        .zip(&inside)
        .map(|((d, l), &inside)| {
            if inside == 1.0 {
                d + l
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect();
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
            let d2 = broadcast(&part(x, &inverses, 0.0, |i| i.d2)?, &shape)?;
            let dx = broadcast(&dy, &shape)?
                .into_iter()
                .zip(d1.iter().zip(&d2))
                .map(|(dy, (d1, d2))| dy * d1 + d2 / d1)
                .collect();
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
