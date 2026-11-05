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

use super::Support;
use super::distribution::{DistributionKind, LogProbGradientParts, ParameterGradients, Univariate};
use super::wrappers;
use crate::error::{GraphError, Result};
use crate::value::{DType, Value};
use keanu_distributions::{DistributionError, KeanuRng};
use keanu_tensor::{DoubleTensor, Shape, broadcast_shape};

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

fn invalid<T>(message: String) -> Result<T> {
    Err(GraphError::Distribution(
        DistributionError::InvalidParameter(message),
    ))
}

/// A mixture's component parameters and weights, with the weights checked: non-negative and summing to 1 (within
/// [`WEIGHT_TOLERANCE`]) along their last axis.
struct Parameters<'p, 'v> {
    components: &'p [&'v Value],
    weights: &'v DoubleTensor,
    count: usize,
}

impl<'p, 'v> Parameters<'p, 'v> {
    fn new(parameters: &'p [&'v Value]) -> Result<Self> {
        let (components, own) = wrappers::split(parameters, OWN_PARAMETERS)?;
        let weights = own[0].f64()?;
        let Some(&count) = weights.shape().last().filter(|&&c| c > 0) else {
            return invalid("the mixture weights need a non-empty component axis".into());
        };
        for row in weights.as_slice().chunks(count) {
            // written so that NaN weights fail too
            if !row.iter().all(|&w| w >= 0.0) {
                return invalid(format!("mixture weights must be non-negative, got {row:?}"));
            }
            let total: f64 = row.iter().sum();
            if (total - 1.0).abs() > WEIGHT_TOLERANCE {
                return invalid(format!("mixture weights must sum to 1, got {row:?}"));
            }
        }
        Ok(Self {
            components,
            weights,
            count,
        })
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

/// A tensor's elements broadcast to `shape`.
pub(crate) fn broadcast(t: &DoubleTensor, shape: &[usize]) -> Result<Vec<f64>> {
    Ok(if t.shape() == shape {
        t.as_slice().to_vec()
    } else {
        t.broadcast_to(shape)?.into_vec()
    })
}

/// The log probability of a mixture node at `value` (summed over its elements) and, with `gradient`, its gradient
/// parts: with respect to the value (summed over the components), to each component parameter (shaped like the
/// value with the component axis) and to the weights.
pub(crate) fn log_prob(
    component: &DistributionKind,
    value: &Value,
    parameters: &[&Value],
    gradient: bool,
) -> Result<(f64, Option<LogProbGradientParts>)> {
    let p = Parameters::new(parameters)?;
    let x = with_component_axis(value)?;
    let (densities, parts) =
        component.univariate(Univariate::Density, &x, p.components, gradient)?;
    // the value, the parameters and the weights, broadcast, with the component axis last
    let mut full = broadcast_shape(densities.shape(), p.weights.shape())?;
    full = broadcast_shape(&full, x.shape())?;
    let count = p.count;
    if full.last() != Some(&count) {
        return invalid(format!(
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
    for row in 0..rows {
        let range = row * count..(row + 1) * count;
        let terms: Vec<f64> = range
            .clone()
            .map(|i| weights[i].ln() + densities[i])
            .collect();
        let max = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let total = if max == f64::NEG_INFINITY || max.is_nan() {
            max
        } else {
            max + terms.iter().map(|t| (t - max).exp()).sum::<f64>().ln()
        };
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
            .map(|(d, &r)| if r == 0.0 { 0.0 } else { r * d })
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

/// The support of a mixture: the component kind's support with each bound reduced over the component axis (the
/// smallest lower bound, the largest upper bound), so that it holds every component's support.
pub(crate) fn support(component: &DistributionKind, parameters: &[&Value]) -> Result<Support> {
    let p = Parameters::new(parameters)?;
    let support = component.support(p.components)?;
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
        return invalid(format!(
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
