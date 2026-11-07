//! Heterogeneous mixtures: components of different kinds, weighted ([`DistributionKind::MixtureOf`]).
//!
//! The node's parents are the `weights` (the component axis last, of length C, the number of components), then
//! each component's parameters in its own order, component after component. The components are element-wise kinds
//! with the same value type (a gamma and a log-normal, a Poisson and a negative binomial, a censored Weibull and a
//! censored uniform, ...), each with parameters of its own that have no component axis. The value has the batch
//! shape: the weights' shape without the component axis and every component parameter's shape, broadcast together
//! (and with a requested shape).
//!
//! With `log p_c(x)` the c-th component's element-wise log density:
//!
//! - log p(x) = logsumexp_c(log w_c + log p_c(x)), element-wise;
//! - with the responsibilities r_c = w_c p_c(x) / p(x), the gradient is Σ_c r_c ∂ log p_c(x) with respect to x,
//!   r_c ∂ log p_c(x) with respect to the c-th component's parameters, and p_c(x) / p(x) with respect to w_c;
//! - the Hessian-vector products are those of a logsumexp (see the mixture module), with each component's own
//!   element-wise second-order rule;
//! - a draw picks a component by its weight, then keeps that component's draw.
//!
//! The support is the hull of the components' supports (each must be a range of real numbers, or all discrete):
//! the smallest lower bound and the largest upper bound across them. Where the components' supports don't overlap
//! the density is 0 between them although the support includes them.

use super::distribution::{DistributionKind, LogProbGradientParts, ParameterGradients, Univariate};
use super::distribution_hvp::{LogProbHvpParts, elementwise, reduce, spread, spread_tangent};
use super::mixture::{gather, invalid_weights};
use super::{Support, broadcast, invalid_parameter, wrappers};
use crate::autodiff::{batched_shape, fit, reduce_to};
use crate::error::{GraphError, Result};
use crate::value::{DType, Value};
use keanu_distributions::KeanuRng;
use keanu_tensor::{DoubleTensor, Shape, broadcast_shape};

/// Checks that `components` can be mixed: at least one, each an element-wise kind with a fixed parameter list,
/// all with the same value type.
pub(crate) fn check(components: &[DistributionKind]) -> std::result::Result<(), String> {
    let Some(first) = components.first() else {
        return Err("a mixture needs at least one component".into());
    };
    for component in components {
        if !component.is_elementwise() || component.parameter_types().is_none() {
            return Err(format!(
                "the component distribution ({}) does not have one value per element with an element-wise \
                 density, so it cannot be mixed",
                component.name()
            ));
        }
        if component.value_type() != first.value_type() {
            return Err(format!(
                "the components' values must have one type, but {} has {:?} and {} {:?}",
                first.name(),
                first.value_type(),
                component.name(),
                component.value_type()
            ));
        }
    }
    Ok(())
}

/// The number of parameters of each component.
fn counts(components: &[DistributionKind]) -> impl Iterator<Item = usize> + '_ {
    components
        .iter()
        .map(|c| c.parameter_types().map_or(0, |t| t.len()))
}

/// The parameter types: the weights' (f64), then each component's.
pub(crate) fn parameter_types(components: &[DistributionKind]) -> Option<Vec<DType>> {
    let mut types = vec![DType::F64];
    for component in components {
        types.extend_from_slice(&component.parameter_types()?);
    }
    Some(types)
}

/// The parameter names: `weights`, then each component's.
pub(crate) fn parameter_names(components: &[DistributionKind]) -> Option<Vec<&'static str>> {
    let mut names = vec!["weights"];
    for component in components {
        names.extend_from_slice(&component.parameter_names()?);
    }
    Some(names)
}

/// The value type: the components'.
pub(crate) fn value_type(components: &[DistributionKind]) -> DType {
    components.first().map_or(DType::F64, |c| c.value_type())
}

/// The value shape of a node with `parameters` (the weights, then each component's, checked to have the right
/// types) and an optional requested shape.
pub(crate) fn value_shape(
    components: &[DistributionKind],
    requested: Option<&[usize]>,
    parameters: &[(DType, &[usize])],
) -> std::result::Result<Shape, String> {
    let Some(((_, weights), rest)) = parameters.split_first() else {
        return Err("a mixture needs weights".into());
    };
    let Some((&count, weight_batch)) = weights.split_last() else {
        return Err("the weights must have a component axis (their last)".into());
    };
    if count != components.len() {
        return Err(format!(
            "the weights' last axis has length {count}, but there are {} components",
            components.len()
        ));
    }
    let mut shape = Shape::from_slice(requested.unwrap_or(&[]));
    for batch in std::iter::once(weight_batch).chain(rest.iter().map(|(_, s)| *s)) {
        shape = broadcast_shape(&shape, batch).map_err(|e| e.to_string())?;
    }
    Ok(shape)
}

/// The weights and each component's parameters.
struct Layout<'p, 'v> {
    weights: &'v DoubleTensor,
    components: Vec<&'p [&'v Value]>,
}

fn layout<'p, 'v>(
    components: &[DistributionKind],
    parameters: &'p [&'v Value],
) -> Result<Layout<'p, 'v>> {
    let wrong = || GraphError::InvalidNode {
        op: "mixture_of",
        message: format!(
            "needs parameters of types {:?}",
            parameter_types(components)
        ),
    };
    let Some((weights, mut rest)) = parameters.split_first() else {
        return Err(wrong());
    };
    let weights = weights.f64()?;
    if weights.shape().last() != Some(&components.len()) {
        return invalid_parameter(format!(
            "the mixture weights need a last axis of length {}, the number of components, not shape {:?}",
            components.len(),
            weights.shape()
        ));
    }
    let mut split = Vec::with_capacity(components.len());
    for count in counts(components) {
        if rest.len() < count {
            return Err(wrong());
        }
        let (these, others) = rest.split_at(count);
        split.push(these);
        rest = others;
    }
    if !rest.is_empty() {
        return Err(wrong());
    }
    Ok(Layout {
        weights,
        components: split,
    })
}

/// The batch shape: the value's, the weights' without their component axis and every component parameter's,
/// broadcast together.
fn batch_shape(value: &Value, layout: &Layout<'_, '_>) -> Result<Shape> {
    let weights = layout.weights.shape();
    let mut shape = broadcast_shape(value.shape(), &weights[..weights.len() - 1])?;
    for parameter in layout.components.iter().flat_map(|p| p.iter()) {
        shape = broadcast_shape(&shape, parameter.shape())?;
    }
    Ok(shape)
}

/// `[shape..., count]`.
fn with_axis(shape: &[usize], count: usize) -> Shape {
    let mut full = Shape::from_slice(shape);
    full.push(count);
    full
}

/// logsumexp of `terms`, −∞ for none finite (and NaN for a NaN term).
fn log_sum_exp(terms: &[f64]) -> f64 {
    let max = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max == f64::NEG_INFINITY || max.is_nan() {
        max
    } else {
        max + terms.iter().map(|t| (t - max).exp()).sum::<f64>().ln()
    }
}

/// The element-wise log density over the batch shape and, with `gradient`, its gradient parts: with respect to
/// the value and each component parameter over the batch shape, and to the weights over the batch shape with the
/// component axis. The weights must be valid.
fn density(
    components: &[DistributionKind],
    value: &Value,
    layout: &Layout<'_, '_>,
    gradient: bool,
) -> Result<(Shape, Vec<f64>, Option<LogProbGradientParts>)> {
    let shape = batch_shape(value, layout)?;
    let count = components.len();
    let n: usize = shape.iter().product();
    let full = with_axis(&shape, count);
    let weights = broadcast(layout.weights, &full)?;
    let mut densities = Vec::with_capacity(count);
    let mut parts = Vec::with_capacity(count);
    for (component, parameters) in components.iter().zip(&layout.components) {
        let (d, p) = component.univariate(Univariate::Density, value, parameters, gradient)?;
        densities.push(broadcast(&d, &shape)?);
        parts.push(p);
    }
    let mut values = Vec::with_capacity(n);
    let mut responsibilities = vec![0.0; n * count];
    let mut ratios = vec![0.0; n * count];
    let mut terms = vec![0.0; count];
    for i in 0..n {
        for (c, t) in terms.iter_mut().enumerate() {
            *t = weights[i * count + c].ln() + densities[c][i];
        }
        let total = log_sum_exp(&terms);
        values.push(total);
        if gradient && total.is_finite() {
            for c in 0..count {
                responsibilities[i * count + c] = (terms[c] - total).exp();
                ratios[i * count + c] = (densities[c][i] - total).exp();
            }
        }
    }
    if !gradient {
        return Ok((shape, values, None));
    }
    // r_c times a component's derivative, 0 where r_c is (whatever the derivative is there)
    let weighted = |c: usize, d: &DoubleTensor| -> Result<Vec<f64>> {
        Ok(broadcast(d, &shape)?
            .into_iter()
            .enumerate()
            .map(|(i, d)| {
                let r = responsibilities[i * count + c];
                if r == 0.0 { 0.0 } else { r * d }
            })
            .collect())
    };
    let mut dx: Option<Vec<f64>> = None;
    let mut gradients: ParameterGradients =
        smallvec::smallvec![Some(DoubleTensor::new(&full, ratios)?)];
    for (c, (p, parameters)) in parts.into_iter().zip(&layout.components).enumerate() {
        let (component_dx, component_parameters) =
            p.unwrap_or_else(|| (None, smallvec::smallvec![None; parameters.len()]));
        if let Some(d) = component_dx {
            let d = weighted(c, &d)?;
            match &mut dx {
                Some(total) => total.iter_mut().zip(d).for_each(|(t, d)| *t += d),
                None => dx = Some(d),
            }
        }
        for d in component_parameters {
            gradients.push(
                d.map(|d| Ok::<_, GraphError>(DoubleTensor::new(&shape, weighted(c, &d)?)?))
                    .transpose()?,
            );
        }
    }
    let dx = dx.map(|d| DoubleTensor::new(&shape, d)).transpose()?;
    Ok((shape, values, Some((dx, gradients))))
}

/// The log probability of a node at `value` (summed over its elements) and, with `gradient`, its gradient parts
/// (shaped as the batch, the weights' with the component axis). Weights off the simplex make it −∞, with no
/// gradient, as for a same-family mixture.
pub(crate) fn log_prob(
    components: &[DistributionKind],
    value: &Value,
    parameters: &[&Value],
    gradient: bool,
) -> Result<(f64, Option<LogProbGradientParts>)> {
    let layout = layout(components, parameters)?;
    if invalid_weights(layout.weights, components.len()).is_some() {
        return Ok((f64::NEG_INFINITY, None));
    }
    let (_, values, parts) = density(components, value, &layout, gradient)?;
    Ok((values.iter().sum(), parts))
}

/// The log probability of each element (the batch shape); −∞ everywhere for weights off the simplex.
pub(crate) fn log_prob_terms(
    components: &[DistributionKind],
    value: &Value,
    parameters: &[&Value],
) -> Result<DoubleTensor> {
    let layout = layout(components, parameters)?;
    if invalid_weights(layout.weights, components.len()).is_some() {
        let shape = batch_shape(value, &layout)?;
        return Ok(DoubleTensor::full(&shape, f64::NEG_INFINITY));
    }
    let (shape, values, _) = density(components, value, &layout, false)?;
    Ok(DoubleTensor::new(&shape, values)?)
}

/// The support: discrete for discrete components, else the hull of the components' supports.
pub(crate) fn support(components: &[DistributionKind], parameters: &[&Value]) -> Result<Support> {
    let layout = layout(components, parameters)?;
    if value_type(components) != DType::F64 {
        return Ok(Support::Discrete);
    }
    let mut hull: Option<(DoubleTensor, DoubleTensor)> = None;
    for (component, parameters) in components.iter().zip(&layout.components) {
        let support = component.support(parameters)?;
        let Some((lower, upper)) = wrappers::support_bounds(support.clone()) else {
            return invalid_parameter(format!(
                "the {} component's support ({support:?}) is not a range of real numbers, so the mixture has no \
                 support",
                component.name()
            ));
        };
        hull = Some(match hull {
            None => (lower, upper),
            Some((l, u)) => (l.zip_with(&lower, f64::min)?, u.zip_with(&upper, f64::max)?),
        });
    }
    let (lower, upper) = hull.expect("at least one component");
    Ok(wrappers::support_between(lower, upper))
}

/// Draws a value of `shape`: a draw of every component, of which each element keeps the one of a component picked
/// by its weight.
pub(crate) fn sample(
    components: &[DistributionKind],
    shape: &[usize],
    parameters: &[&Value],
    rng: &mut KeanuRng,
) -> Result<Value> {
    let layout = layout(components, parameters)?;
    let count = components.len();
    if let Some(message) = invalid_weights(layout.weights, count) {
        return invalid_parameter(message);
    }
    let n: usize = shape.iter().product();
    let draws = components
        .iter()
        .zip(&layout.components)
        .map(|(component, parameters)| {
            let draw = component.sample(shape, parameters, rng)?;
            if draw.shape() != shape {
                return invalid_parameter(format!(
                    "the {} component's parameters don't broadcast to the mixture's shape {shape:?}",
                    component.name()
                ));
            }
            Ok(draw)
        })
        .collect::<Result<Vec<_>>>()?;
    let weights = broadcast(layout.weights, &with_axis(shape, count))?;
    let mut picks = Vec::with_capacity(n);
    for (i, w) in weights.chunks(count).enumerate() {
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
        picks.push((pick, i));
    }
    // the picked elements, from a stack of the draws [C, shape...]
    let stacked = match &draws[0] {
        Value::F64(_) => Value::F64(DoubleTensor::new(
            &[count * n],
            draws
                .iter()
                .map(|d| d.f64().map(|t| t.as_slice().to_vec()))
                .collect::<Result<Vec<_>>>()?
                .concat(),
        )?),
        Value::I64(_) => Value::I64(keanu_tensor::IntTensor::new(
            &[count * n],
            draws
                .iter()
                .map(|d| d.i64().map(|t| t.as_slice().to_vec()))
                .collect::<Result<Vec<_>>>()?
                .concat(),
        )?),
        Value::Bool(_) => Value::Bool(keanu_tensor::BoolTensor::new(
            &[count * n],
            draws
                .iter()
                .map(|d| d.bool().map(|t| t.as_slice().to_vec()))
                .collect::<Result<Vec<_>>>()?
                .concat(),
        )?),
    };
    let flat: Vec<usize> = picks.iter().map(|&(c, i)| c * n + i).collect();
    gather(&stacked, shape, &flat)
}

/// The unreduced second-order parts over the batch shape: the products with respect to the value and each
/// component parameter (`[b, batch...]` flattened) and to the weights (`[b, batch..., C]` flattened), None where
/// there is no derivative.
struct SecondOrder {
    shape: Shape,
    value: Option<Vec<f64>>,
    weights: Vec<f64>,
    parameters: Vec<Option<Vec<f64>>>,
}

/// The Hessian-vector products of the element-wise log density along `b` directions, unreduced. With
/// l_c = ln w_c + ln p_c(x), L = logsumexp_c l_c, l̇_c = ∇l_c · v and L̇ = Σ_c r_c l̇_c, the product with respect to
/// an input u is Σ_c r_c (∇²l_c v + ∇l_c (l̇_c − L̇))_u.
fn second_order(
    components: &[DistributionKind],
    value: &Value,
    parameters: &[&Value],
    value_tangent: Option<&DoubleTensor>,
    parameter_tangents: &[Option<&DoubleTensor>],
    b: usize,
) -> Result<SecondOrder> {
    let layout = layout(components, parameters)?;
    let count = components.len();
    if let Some(message) = invalid_weights(layout.weights, count) {
        return invalid_parameter(message);
    }
    let shape = batch_shape(value, &layout)?;
    let n: usize = shape.iter().product();
    let full = with_axis(&shape, count);
    let x_tangent = spread_tangent(value_tangent, b, &shape)?;
    let weights = spread(layout.weights, &full)?;
    let weight_tangent = spread_tangent(parameter_tangents[0], b, &full)?;
    // each component's parts, with its inputs' tangents (the value's, then its parameters')
    let mut offset = 1;
    let mut parts = Vec::with_capacity(count);
    let mut tangents = Vec::with_capacity(count);
    for (component, parameters) in components.iter().zip(&layout.components) {
        let these: Vec<Option<Vec<f64>>> = std::iter::once(Ok(x_tangent.clone()))
            .chain(
                parameter_tangents[offset..offset + parameters.len()]
                    .iter()
                    .map(|t| spread_tangent(*t, b, &shape)),
            )
            .collect::<Result<_>>()?;
        offset += parameters.len();
        parts.push(elementwise(
            component, value, parameters, &these, b, &shape,
        )?);
        tangents.push(these);
    }
    // the responsibilities r_c = w_c p_c / p and the ratios ρ_c = p_c / p (so r_c = w_c ρ_c), with p the mixture's
    // density: the derivatives are written with ρ_c rather than r_c / w_c, so a weight of 0 (or a subnormal one)
    // needs no division by it
    let mut responsibilities = vec![0.0; n * count];
    let mut ratios = vec![0.0; n * count];
    let mut terms = vec![0.0; count];
    for i in 0..n {
        for (c, t) in terms.iter_mut().enumerate() {
            *t = weights[i * count + c].ln() + parts[c].values[i];
        }
        let total = log_sum_exp(&terms);
        if total.is_finite() {
            for c in 0..count {
                responsibilities[i * count + c] = (terms[c] - total).exp();
                ratios[i * count + c] = (parts[c].values[i] - total).exp();
            }
        }
    }
    let differentiable_value = parts.iter().any(|p| p.gradients[0].is_some());
    let mut value_out = differentiable_value.then(|| vec![0.0; b * n]);
    let mut weight_out = vec![0.0; b * n * count];
    let mut parameter_out: Vec<Vec<Option<Vec<f64>>>> = parts
        .iter()
        .map(|p| {
            p.gradients[1..]
                .iter()
                .map(|g| g.as_ref().map(|_| vec![0.0; b * n]))
                .collect()
        })
        .collect();
    let mut directional = vec![0.0; count];
    for k in 0..b {
        for i in 0..n {
            let j = k * n + i;
            // ṗ_c / p_c (the component's directional log derivative, without the weights), and L̇ = ṗ / p
            let mut mean = 0.0;
            for c in 0..count {
                let w = i * count + c;
                directional[c] = 0.0;
                // an impossible component (ρ_c = 0) contributes nothing, and its derivatives may not be finite
                if ratios[w] == 0.0 {
                    continue;
                }
                let mut d = 0.0;
                for (g, t) in parts[c].gradients.iter().zip(&tangents[c]) {
                    if let (Some(g), Some(t)) = (g, t) {
                        d += g[i] * t[j];
                    }
                }
                directional[c] = d;
                let dw = weight_tangent
                    .as_ref()
                    .map_or(0.0, |t| t[k * n * count + w]);
                mean += ratios[w] * dw + responsibilities[w] * d;
            }
            for c in 0..count {
                let w = i * count + c;
                let (r, ratio) = (responsibilities[w], ratios[w]);
                if ratio == 0.0 {
                    continue;
                }
                let spread = directional[c] - mean;
                let dw = weight_tangent
                    .as_ref()
                    .map_or(0.0, |t| t[k * n * count + w]);
                // the derivative of r_c g_c is r_c ġ_c + g_c ṙ_c, with ṙ_c = ρ_c ẇ_c + r_c (ṗ_c / p_c − L̇)
                let dr = ratio * dw + r * spread;
                let product = |u: usize| -> Option<f64> {
                    let g = parts[c].gradients[u].as_ref()?;
                    let own = parts[c].hvps[u].as_ref().map_or(0.0, |h| h[j]);
                    Some(r * own + g[i] * dr)
                };
                if let (Some(out), Some(p)) = (&mut value_out, product(0)) {
                    out[j] += p;
                }
                for (u, out) in parameter_out[c].iter_mut().enumerate() {
                    if let (Some(out), Some(p)) = (out, product(u + 1)) {
                        out[j] = p;
                    }
                }
                // the derivative of ∂L/∂w_c = ρ_c is ρ_c (ṗ_c / p_c − L̇)
                weight_out[k * n * count + w] = ratio * spread;
            }
        }
    }
    Ok(SecondOrder {
        shape,
        value: value_out,
        weights: weight_out,
        parameters: parameter_out.into_iter().flatten().collect(),
    })
}

/// The Hessian-vector products of a node's log density (see [`DistributionKind::log_prob_hvp`]).
pub(crate) fn log_prob_hvp(
    components: &[DistributionKind],
    value: &Value,
    parameters: &[&Value],
    value_tangent: Option<&DoubleTensor>,
    parameter_tangents: &[Option<&DoubleTensor>],
    b: usize,
) -> Result<LogProbHvpParts> {
    let parts = second_order(
        components,
        value,
        parameters,
        value_tangent,
        parameter_tangents,
        b,
    )?;
    let full = with_axis(&parts.shape, components.len());
    let value_hvp = reduce(parts.value, b, &parts.shape, value.shape())?;
    let mut parameter_hvps = smallvec::smallvec![reduce(
        Some(parts.weights),
        b,
        &full,
        parameters[0].shape()
    )?];
    for (h, p) in parts.parameters.into_iter().zip(&parameters[1..]) {
        parameter_hvps.push(reduce(h, b, &parts.shape, p.shape())?);
    }
    Ok((value_hvp, parameter_hvps))
}

// The log-density op of a heterogeneous mixture: element-wise over the batch shape, its inputs the value, the
// weights (with their component axis) and the components' parameters.

/// The log density of each element of `inputs[0]` given the weights and the components' parameters.
pub(crate) fn density_eval(components: &[DistributionKind], inputs: &[&Value]) -> Result<Value> {
    let (value, parameters) = inputs
        .split_first()
        .ok_or_else(|| GraphError::InvalidValue("a log density needs its value".into()))?;
    Ok(Value::F64(log_prob_terms(components, value, parameters)?))
}

/// The derivatives of the element-wise log density with respect to each input (the value, the weights, each
/// component parameter), over the batch shape (with the component axis for the weights), None where there are none
/// (and for every input where the weights are invalid).
fn density_derivatives(
    components: &[DistributionKind],
    inputs: &[&Value],
) -> Result<Vec<Option<DoubleTensor>>> {
    let (value, parameters) = inputs
        .split_first()
        .ok_or_else(|| GraphError::InvalidValue("a log density needs its value".into()))?;
    let (_, gradient) = log_prob(components, value, parameters, true)?;
    let Some((dx, dparameters)) = gradient else {
        return Ok(vec![None; inputs.len()]);
    };
    Ok(std::iter::once(dx)
        .chain(dparameters)
        .zip(inputs)
        .map(|(d, input)| d.filter(|_| input.dtype() == DType::F64))
        .collect())
}

/// Forward mode of the log-density op (see the log density module's `jvp`).
pub(crate) fn density_jvp(
    components: &[DistributionKind],
    inputs: &[&Value],
    t: &[Option<&DoubleTensor>],
    b: usize,
    out_shape: &[usize],
) -> Result<Option<DoubleTensor>> {
    if t.iter().all(Option::is_none) {
        return Ok(None);
    }
    let count = components.len();
    let mut total: Option<DoubleTensor> = None;
    for (u, (d, t)) in density_derivatives(components, inputs)?
        .into_iter()
        .zip(t)
        .enumerate()
    {
        let (Some(d), Some(t)) = (d, t) else { continue };
        let term = if u == 1 {
            let product = fit((*t).clone(), b, &with_axis(out_shape, count))?.mul(&d)?;
            let summed = product
                .as_slice()
                .chunks(count)
                .map(|row| row.iter().sum())
                .collect();
            DoubleTensor::new(&batched_shape(b, out_shape), summed)?
        } else {
            fit((*t).clone(), b, out_shape)?.mul(&d)?
        };
        total = Some(match total {
            Some(sum) => sum.add(&term)?,
            None => term,
        });
    }
    Ok(total)
}

/// `g` (`[b, out...]`) with a trailing axis of length 1, to broadcast against the weights' component axis.
fn with_unit_axis(g: &DoubleTensor) -> Result<DoubleTensor> {
    let mut shape: Vec<isize> = g.shape().iter().map(|&d| d as isize).collect();
    shape.push(1);
    Ok(g.reshape(&shape)?)
}

/// Reverse mode of the log-density op (see the log density module's `vjp`).
pub(crate) fn density_vjp(
    components: &[DistributionKind],
    inputs: &[&Value],
    g: &DoubleTensor,
) -> Result<Vec<Option<DoubleTensor>>> {
    density_derivatives(components, inputs)?
        .into_iter()
        .zip(inputs)
        .enumerate()
        .map(|(u, (d, input))| {
            d.map(|d| {
                let product = if u == 1 {
                    with_unit_axis(g)?.mul(&d)?
                } else {
                    g.mul(&d)?
                };
                reduce_to(product, input.shape())
            })
            .transpose()
        })
        .collect()
}

/// The derivative of [`density_vjp`] for the cotangent `g` (`[1, out...]`) along the inputs' tangents `t` (see the
/// log density module's `vjp_tangent`).
pub(crate) fn density_vjp_tangent(
    components: &[DistributionKind],
    inputs: &[&Value],
    g: &DoubleTensor,
    t: &[Option<&DoubleTensor>],
    b: usize,
) -> Result<Vec<Option<DoubleTensor>>> {
    let (value, parameters) = inputs
        .split_first()
        .ok_or_else(|| GraphError::InvalidValue("a log density needs its value".into()))?;
    let Some((&value_tangent, parameter_tangents)) = t.split_first() else {
        return Err(GraphError::InvalidValue(
            "a log density needs its value's tangent".into(),
        ));
    };
    let parts = second_order(
        components,
        value,
        parameters,
        value_tangent,
        parameter_tangents,
        b,
    )?;
    let count = components.len();
    let cotangent = g.as_slice();
    let n = cotangent.len().max(1);
    let weigh = |mut h: Vec<f64>, per: usize| -> Vec<f64> {
        for (j, x) in h.iter_mut().enumerate() {
            *x *= cotangent[(j / per) % n];
        }
        h
    };
    let full = with_axis(&parts.shape, count);
    let mut out = vec![
        reduce(
            parts.value.map(|h| weigh(h, 1)),
            b,
            &parts.shape,
            value.shape(),
        )?
        .filter(|_| value.dtype() == DType::F64),
        reduce(
            Some(weigh(parts.weights, count)),
            b,
            &full,
            parameters[0].shape(),
        )?,
    ];
    for (h, p) in parts.parameters.into_iter().zip(&parameters[1..]) {
        out.push(
            reduce(h.map(|h| weigh(h, 1)), b, &parts.shape, p.shape())?
                .filter(|_| p.dtype() == DType::F64),
        );
    }
    Ok(out)
}
