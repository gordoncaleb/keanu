//! Same-family mixtures: C distributions of one component kind, weighted.
//!
//! A [`DistributionKind::Mixture`] is a kind built around another one, with the layout described in the wrappers
//! module: the node's parents are the component kind's parameters, in its order, followed by the mixture's own
//! parameter, the `weights`. The weights' last axis is the component axis, of length C; each component parameter
//! has the same trailing axis (of length C, or 1, or no axes at all, to share one value between the components).
//! The node's value has the batch shape: the parameters' shapes without their component axes, broadcast together
//! (and with a requested shape).
//!
//! With `x` the value extended by a trailing axis of length 1, so it broadcasts against the component axis, and
//! `log p_c(x)` the component kind's element-wise log density with the c-th components' parameters:
//!
//! - log p(x) = logsumexp_c(log w_c + log p_c(x)), element-wise;
//! - with the responsibilities r_c = w_c p_c(x) / p(x), the gradient is Σ_c r_c ∂ log p_c(x) with respect to x,
//!   r_c ∂ log p_c(x) with respect to the c-th component's parameters, and p_c(x) / p(x) with respect to w_c;
//! - a draw picks a component by its weight, then draws from it.
//!
//! The support is the component kind's support widened to hold every component's: the smallest lower bound and the
//! largest upper bound across the components. Where components with parameter-dependent bounds don't overlap, the
//! density is 0 between them although the support includes them.

use super::distribution::{DistributionKind, LogProbGradientParts, ParameterGradients, Univariate};
use super::distribution_hvp::{LogProbHvpParts, elementwise, reduce, spread_tangent};
use super::wrappers;
use super::{Support, broadcast, invalid_parameter};
use crate::autodiff::{batched_shape, reduce_to};
use crate::error::Result;
use crate::value::{DType, Value};
use keanu_distributions::KeanuRng;
use keanu_tensor::{DoubleTensor, Shape, broadcast_shape};
use smallvec::SmallVec;

/// The number of parameters a mixture adds after its component's: `weights`.
pub(crate) const OWN_PARAMETERS: usize = 1;

/// How far each set of weights may sum from 1.
const WEIGHT_TOLERANCE: f64 = 1e-5;

/// The value shape of a mixture node with `parameters` (the component's then the weights, all already checked to
/// have the right types) and an optional requested shape: the batch shapes of the parameters, broadcast together and
/// with the requested shape.
pub(crate) fn value_shape(
    requested: Option<&[usize]>,
    parameters: &[(DType, &[usize])],
) -> std::result::Result<Shape, String> {
    let Some(((_, weights), components)) = parameters.split_last() else {
        return Err("a mixture needs weights".into());
    };
    let Some((&count, weight_batch)) = weights.split_last() else {
        return Err("the weights must have a component axis (their last)".into());
    };
    if count == 0 {
        return Err("there must be at least one component".into());
    }
    let mut shape = Shape::from_slice(requested.unwrap_or(&[]));
    let mut add = |batch: &[usize]| -> std::result::Result<(), String> {
        shape = broadcast_shape(&shape, batch).map_err(|e| e.to_string())?;
        Ok(())
    };
    add(weight_batch)?;
    for (i, (_, parameter)) in components.iter().enumerate() {
        if let Some((&last, batch)) = parameter.split_last() {
            if last != count && last != 1 {
                return Err(format!(
                    "component parameter {i} has shape {parameter:?}, whose last axis should be the component \
                     axis, of length {count} (the weights' last axis) or 1"
                ));
            }
            add(batch)?;
        }
    }
    Ok(shape)
}

/// A mixture's component parameters and weights, with the weights checked: non-negative and summing to 1 (within
/// [`WEIGHT_TOLERANCE`]) along their last axis.
struct Parameters<'p, 'v> {
    components: &'p [&'v Value],
    weights: &'v DoubleTensor,
    count: usize,
}

impl<'p, 'v> Parameters<'p, 'v> {
    /// The parameters, an error unless the weights are valid.
    fn new(parameters: &'p [&'v Value]) -> Result<Self> {
        let p = Self::unchecked(parameters)?;
        if let Some(message) = p.invalid_weights() {
            return invalid_parameter(message);
        }
        Ok(p)
    }

    /// The parameters with only the weights' layout checked (a non-empty component axis), not their values.
    fn unchecked(parameters: &'p [&'v Value]) -> Result<Self> {
        let (components, own) = wrappers::split(parameters, OWN_PARAMETERS)?;
        let weights = own[0].f64()?;
        let Some(&count) = weights.shape().last().filter(|&&c| c > 0) else {
            return invalid_parameter("the mixture weights need a non-empty component axis".into());
        };
        Ok(Self {
            components,
            weights,
            count,
        })
    }

    /// Why the weights are outside their valid region (the simplex), if they are.
    fn invalid_weights(&self) -> Option<String> {
        invalid_weights(self.weights, self.count)
    }
}

/// Why mixture weights with `count` components along their last axis are outside their valid region (the
/// simplex: non-negative and summing to 1 within [`WEIGHT_TOLERANCE`]), if they are.
pub(crate) fn invalid_weights(weights: &DoubleTensor, count: usize) -> Option<String> {
    weights.as_slice().chunks(count.max(1)).find_map(|row| {
        // written so that NaN weights fail too
        if !row.iter().all(|&w| w >= 0.0) {
            return Some(format!("mixture weights must be non-negative, got {row:?}"));
        }
        let total: f64 = row.iter().sum();
        ((total - 1.0).abs() > WEIGHT_TOLERANCE)
            .then(|| format!("mixture weights must sum to 1, got {row:?}"))
    })
}

/// The log contribution of a component. A zero weight excludes even a singular density.
pub(super) fn weighted_log_term(weight: f64, log_density: f64) -> f64 {
    if weight == 0.0 {
        f64::NEG_INFINITY
    } else {
        weight.ln() + log_density
    }
}

/// logsumexp of the weighted component terms.
pub(super) fn log_sum_exp(terms: &[f64]) -> f64 {
    let max = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max == f64::NEG_INFINITY || max.is_nan() {
        max
    } else {
        max + terms.iter().map(|t| (t - max).exp()).sum::<f64>().ln()
    }
}

/// A derivative with a zero coefficient contributes nothing, even at a singularity.
pub(super) fn scale_derivative(coefficient: f64, derivative: f64) -> f64 {
    if coefficient == 0.0 {
        0.0
    } else {
        coefficient * derivative
    }
}

/// A value with a trailing axis of length 1 added, so that it broadcasts against the component axis.
pub(crate) fn with_component_axis(value: &Value) -> Result<Value> {
    let mut shape: Vec<isize> = value.shape().iter().map(|&d| d as isize).collect();
    shape.push(1);
    Ok(match value {
        Value::F64(t) => Value::F64(t.reshape(&shape)?),
        Value::I64(t) => Value::I64(t.reshape(&shape)?),
        Value::Bool(t) => Value::Bool(t.reshape(&shape)?),
    })
}

/// The log probability of a mixture node at `value` (summed over its elements) and, with `gradient`, its gradient
/// parts: with respect to the value (summed over the components), to each component parameter (shaped like the
/// value with the component axis) and to the weights.
///
/// Weights off the simplex are outside the parameters' valid region: the log probability is -infinity there (with
/// no gradient), so that a sampler proposing such weights rejects the proposal, as for any other impossible state.
/// Sampling such a mixture is an error.
pub(crate) fn log_prob(
    component: &DistributionKind,
    value: &Value,
    parameters: &[&Value],
    gradient: bool,
) -> Result<(f64, Option<LogProbGradientParts>)> {
    let p = Parameters::unchecked(parameters)?;
    if p.invalid_weights().is_some() {
        return Ok((f64::NEG_INFINITY, None));
    }
    let x = with_component_axis(value)?;
    let (densities, parts) =
        component.univariate(Univariate::Density, &x, p.components, gradient)?;
    // the value, the parameters and the weights, broadcast, with the component axis last
    let mut full = broadcast_shape(densities.shape(), p.weights.shape())?;
    full = broadcast_shape(&full, x.shape())?;
    let count = p.count;
    if full.last() != Some(&count) {
        return invalid_parameter(format!(
            "the component parameters don't broadcast to the {count} components of the weights"
        ));
    }
    let densities = broadcast(&densities, &full)?;
    let weights = broadcast(p.weights, &full)?;
    let rows = densities.len() / count;
    let mut log_probs = Vec::with_capacity(rows);
    // r_c, and p_c(x) / p(x), the derivative with respect to w_c
    let mut responsibilities = vec![0.0; densities.len()];
    let mut ratios = vec![0.0; densities.len()];
    let mut terms = Vec::with_capacity(count);
    for row in 0..rows {
        let range = row * count..(row + 1) * count;
        terms.clear();
        terms.extend(
            range
                .clone()
                .map(|i| weighted_log_term(weights[i], densities[i])),
        );
        let total = log_sum_exp(&terms);
        log_probs.push(total);
        if !gradient || !total.is_finite() {
            continue;
        }
        for (c, i) in range.enumerate() {
            responsibilities[i] = (terms[c] - total).exp();
            ratios[i] = (densities[i] - total).exp();
        }
    }
    let log_prob = log_probs.iter().sum();
    let Some((dx, dparameters)) = parts.filter(|_| gradient) else {
        return Ok((log_prob, None));
    };
    // r_c times a component's derivative, 0 where r_c is (whatever the derivative is there)
    let weighted = |d: &DoubleTensor| -> Result<Vec<f64>> {
        Ok(broadcast(d, &full)?
            .into_iter()
            .zip(&responsibilities)
            .map(|(d, &r)| scale_derivative(r, d))
            .collect())
    };
    let batch = &full[..full.len() - 1];
    let dx = dx
        .map(|dx| -> Result<DoubleTensor> {
            let summed = weighted(&dx)?
                .chunks(count)
                .map(|row| row.iter().sum())
                .collect();
            Ok(DoubleTensor::new(batch, summed)?)
        })
        .transpose()?;
    let mut gradients: ParameterGradients = dparameters
        .iter()
        .map(|d| {
            d.as_ref()
                .map(|d| Ok(DoubleTensor::new(&full, weighted(d)?)?))
                .transpose()
        })
        .collect::<Result<_>>()?;
    gradients.push(Some(DoubleTensor::new(&full, ratios)?));
    Ok((log_prob, Some((dx, gradients))))
}

/// The log probability of each element of the value (the mixture's batch shape): logsumexp_c(log w_c + log p_c(x)),
/// as [`log_prob`] sums them; −∞ everywhere for weights off the simplex.
pub(crate) fn log_prob_terms(
    component: &DistributionKind,
    value: &Value,
    parameters: &[&Value],
) -> Result<DoubleTensor> {
    let p = Parameters::unchecked(parameters)?;
    let x = with_component_axis(value)?;
    let (densities, _) = component.univariate(Univariate::Density, &x, p.components, false)?;
    let mut full = broadcast_shape(densities.shape(), p.weights.shape())?;
    full = broadcast_shape(&full, x.shape())?;
    let count = p.count;
    if full.last() != Some(&count) {
        return invalid_parameter(format!(
            "the component parameters don't broadcast to the {count} components of the weights"
        ));
    }
    let batch = &full[..full.len() - 1];
    if p.invalid_weights().is_some() {
        return Ok(DoubleTensor::full(batch, f64::NEG_INFINITY));
    }
    let densities = broadcast(&densities, &full)?;
    let weights = broadcast(p.weights, &full)?;
    let mut terms = Vec::with_capacity(count);
    let log_probs = densities
        .chunks(count)
        .zip(weights.chunks(count))
        .map(|(densities, weights)| {
            terms.clear();
            terms.extend(
                densities
                    .iter()
                    .zip(weights)
                    .map(|(&d, &w)| weighted_log_term(w, d)),
            );
            log_sum_exp(&terms)
        })
        .collect();
    Ok(DoubleTensor::new(batch, log_probs)?)
}

/// The support of a mixture: the component kind's support with each bound reduced over the component axis (the
/// smallest lower bound, the largest upper bound), so that it holds every component's support. The weights don't
/// bound the support, so they aren't checked here (the log probability and sampling check them): weights that are
/// latent can be perturbed off the simplex without the support appearing to move.
pub(crate) fn support(component: &DistributionKind, parameters: &[&Value]) -> Result<Support> {
    let (components, _) = wrappers::split(parameters, OWN_PARAMETERS)?;
    let support = component.support(components)?;
    let Some((lower, upper)) = wrappers::support_bounds(support.clone()) else {
        return Ok(support);
    };
    Ok(wrappers::support_between(
        reduce_components(&lower, f64::min)?,
        reduce_components(&upper, f64::max)?,
    ))
}

/// A bound reduced over its last (component) axis; a bound without axes is shared by every component already.
pub(crate) fn reduce_components(
    bound: &DoubleTensor,
    f: fn(f64, f64) -> f64,
) -> Result<DoubleTensor> {
    let Some((&count, batch)) = bound.shape().split_last() else {
        return Ok(bound.clone());
    };
    let reduced = bound
        .as_slice()
        .chunks(count.max(1))
        .map(|row| row.iter().copied().reduce(f).unwrap_or(f64::NAN))
        .collect();
    Ok(DoubleTensor::new(batch, reduced)?)
}

/// Draws a mixture value of `shape`: a draw of every component at every element, of which each element keeps the
/// one of a component picked by its weight.
pub(crate) fn sample(
    component: &DistributionKind,
    shape: &[usize],
    parameters: &[&Value],
    rng: &mut KeanuRng,
) -> Result<Value> {
    let p = Parameters::new(parameters)?;
    let count = p.count;
    let mut full = shape.to_vec();
    full.push(count);
    let draws = component.sample(&full, p.components, rng)?;
    if draws.shape() != full.as_slice() {
        return invalid_parameter(format!(
            "the component parameters don't broadcast to the mixture's shape {shape:?} with {count} components"
        ));
    }
    let weights = broadcast(p.weights, &full)?;
    let picks: Vec<usize> = weights
        .chunks(count)
        .enumerate()
        .map(|(row, w)| {
            let u = rng.next_f64();
            let mut cumulative = 0.0;
            // the last component with any weight, should rounding leave u above the cumulative total
            let mut pick = w.iter().rposition(|&w| w > 0.0).unwrap_or(0);
            for (c, &w) in w.iter().enumerate() {
                cumulative += w;
                if u < cumulative && w > 0.0 {
                    pick = c;
                    break;
                }
            }
            row * count + pick
        })
        .collect();
    gather(&draws, shape, &picks)
}

/// The elements of `draws` at the flat indices `picks`, as a value of `shape`.
pub(crate) fn gather(draws: &Value, shape: &[usize], picks: &[usize]) -> Result<Value> {
    Ok(match draws {
        Value::F64(t) => Value::F64(DoubleTensor::new(
            shape,
            picks.iter().map(|&i| t.as_slice()[i]).collect(),
        )?),
        Value::I64(t) => Value::I64(keanu_tensor::IntTensor::new(
            shape,
            picks.iter().map(|&i| t.as_slice()[i]).collect(),
        )?),
        Value::Bool(t) => Value::Bool(keanu_tensor::BoolTensor::new(
            shape,
            picks.iter().map(|&i| t.as_slice()[i]).collect(),
        )?),
    })
}

/// The Hessian-vector products of a mixture's log density along `b` directions (see
/// [`DistributionKind::log_prob_hvp`]). With l_c = ln w_c + ln p_c(x) for each component and L = logsumexp_c l_c,
/// the Hessian of L is Σ_c r_c (∇²l_c + ∇l_c ∇l_cᵀ) − ∇L ∇Lᵀ with ∇L = Σ_c r_c ∇l_c, so along a direction with
/// l̇_c = ∇l_c · v and L̇ = Σ_c r_c l̇_c the product is Σ_c r_c (∇²l_c v + ∇l_c (l̇_c − L̇)). The components' own
/// gradients and products come from the component kind's element-wise rule; ln w_c adds 1/w_c and −1/w_c² for
/// w_c.
pub(crate) fn log_prob_hvp(
    component: &DistributionKind,
    value: &Value,
    parameters: &[&Value],
    value_tangent: Option<&DoubleTensor>,
    parameter_tangents: &[Option<&DoubleTensor>],
    b: usize,
) -> Result<LogProbHvpParts> {
    let p = Parameters::new(parameters)?;
    let x = with_component_axis(value)?;
    let m = p.components.len();
    let count = p.count;
    let mut full = Shape::from_slice(x.shape());
    for parameter in p.components {
        full = broadcast_shape(&full, parameter.shape())?;
    }
    full = broadcast_shape(&full, p.weights.shape())?;
    if full.last() != Some(&count) {
        return invalid_parameter(format!(
            "the component parameters don't broadcast to the {count} components of the weights"
        ));
    }
    let value_tangent = value_tangent
        .map(|t| -> Result<DoubleTensor> {
            let mut shape: Vec<isize> = t.shape().iter().map(|&d| d as isize).collect();
            shape.push(1);
            Ok(t.reshape(&shape)?)
        })
        .transpose()?;
    let tangents: Vec<Option<Vec<f64>>> = std::iter::once(value_tangent.as_ref())
        .chain(parameter_tangents[..m].iter().copied())
        .map(|t| spread_tangent(t, b, &full))
        .collect::<Result<_>>()?;
    let parts = elementwise(component, &x, p.components, &tangents, b, &full)?;
    let weights = broadcast(p.weights, &full)?;
    let weight_tangent = spread_tangent(parameter_tangents[m], b, &full)?;
    let len = weights.len();
    // the responsibilities r_c = w_c p_c / p and the ratios ρ_c = p_c / p (so r_c = w_c ρ_c), with p the mixture's
    // density: the derivatives are written with ρ_c rather than r_c / w_c, so a weight of 0 (or a subnormal one)
    // needs no division by it
    let mut responsibilities = vec![0.0; len];
    let mut ratios = vec![0.0; len];
    let mut terms = Vec::with_capacity(count);
    for row in (0..len).step_by(count) {
        let range = row..row + count;
        terms.clear();
        terms.extend(
            range
                .clone()
                .map(|i| weighted_log_term(weights[i], parts.values[i])),
        );
        let total = log_sum_exp(&terms);
        if !total.is_finite() {
            continue;
        }
        for (c, i) in range.enumerate() {
            responsibilities[i] = (terms[c] - total).exp();
            ratios[i] = (parts.values[i] - total).exp();
        }
    }
    let mut outputs: Vec<Option<Vec<f64>>> = parts
        .hvps
        .iter()
        .map(|h| h.as_ref().map(|_| vec![0.0; b * len]))
        .collect();
    let mut weight_output = vec![0.0; b * len];
    let mut directional = vec![0.0; count];
    for k in 0..b {
        for row in (0..len).step_by(count) {
            // ṗ_c / p_c (the component's directional log derivative, without the weights), and L̇ = ṗ / p
            let mut mean = 0.0;
            for (c, d) in directional.iter_mut().enumerate() {
                let i = row + c;
                let j = k * len + i;
                *d = 0.0;
                // an impossible component (ρ_c = 0) contributes nothing, and its derivatives may not be finite
                if ratios[i] == 0.0 {
                    continue;
                }
                for (g, t) in parts.gradients.iter().zip(&tangents) {
                    if let (Some(g), Some(t)) = (g, t) {
                        *d += g[i] * t[j];
                    }
                }
                let dw = weight_tangent.as_ref().map_or(0.0, |t| t[j]);
                mean += scale_derivative(dw, ratios[i]) + scale_derivative(responsibilities[i], *d);
            }
            for (c, &d) in directional.iter().enumerate() {
                let i = row + c;
                let j = k * len + i;
                let (r, ratio) = (responsibilities[i], ratios[i]);
                if ratio == 0.0 {
                    continue;
                }
                let spread = d - mean;
                let dw = weight_tangent.as_ref().map_or(0.0, |t| t[j]);
                // the derivative of r_c g_c is r_c ġ_c + g_c ṙ_c, with ṙ_c = ρ_c ẇ_c + r_c (ṗ_c / p_c − L̇)
                let dr = scale_derivative(dw, ratio) + scale_derivative(r, spread);
                for (u, out) in outputs.iter_mut().enumerate() {
                    let (Some(out), Some(g)) = (out, &parts.gradients[u]) else {
                        continue;
                    };
                    let own = parts.hvps[u].as_ref().map_or(0.0, |h| h[j]);
                    out[j] = scale_derivative(r, own) + scale_derivative(dr, g[i]);
                }
                // the derivative of ∂L/∂w_c = ρ_c is ρ_c (ṗ_c / p_c − L̇)
                weight_output[j] = scale_derivative(spread, ratio);
            }
        }
    }
    let mut outputs = outputs.into_iter();
    // the value's products, summed over the component axis
    let value_hvp = outputs
        .next()
        .flatten()
        .map(|h| -> Result<DoubleTensor> {
            let summed: Vec<f64> = h.chunks(count).map(|row| row.iter().sum()).collect();
            let batch = &full[..full.len() - 1];
            reduce_to(
                DoubleTensor::new(&batched_shape(b, batch), summed)?,
                value.shape(),
            )
        })
        .transpose()?;
    let mut parameter_hvps: LogProbHvpParts = (value_hvp, SmallVec::new());
    for (h, parameter) in outputs.zip(p.components) {
        parameter_hvps
            .1
            .push(reduce(h, b, &full, parameter.shape())?);
    }
    parameter_hvps
        .1
        .push(reduce(Some(weight_output), b, &full, p.weights.shape())?);
    Ok(parameter_hvps)
}
