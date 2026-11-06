//! The log-density op ([`Op::LogDensity`](super::Op::LogDensity)): the element-wise log density of a value under an
//! element-wise distribution, as a deterministic node. Its value, first derivatives and second derivatives reuse the
//! distribution's element-wise rules (`DistributionKind::univariate` and the scalar second-order rules), so a
//! log density on the graph differentiates exactly as the same distribution's node does. The Bernoulli and its logit
//! form, whose values are boolean and which therefore can't be mixture components, have element-wise densities too:
//! their values and derivatives come from their nodes' unreduced log probability terms and gradients, which are
//! element-wise.
//!
//! The linear Gaussian state space model has a *joint* density instead: its log density node is a scalar, the
//! Kalman filter's log likelihood of the whole series, and its derivatives are the distribution's gradient and
//! Hessian-vector products. A model whose observations are a deterministic function of latents (a structural time
//! series with a regression offset, see [`crate::sts`]) uses it in a factor.

use super::distribution::{DistributionKind, DistributionOp, LogProbGradientParts, Univariate};
use super::distribution_hvp::{elementwise, spread, spread_tangent};
use crate::autodiff::{batched_shape, fit, lift, reduce_to};
use crate::error::{GraphError, Result};
use crate::value::{DType, Value};
use keanu_tensor::{DoubleTensor, Shape, broadcast_shape};

fn invalid(message: String) -> GraphError {
    GraphError::InvalidNode {
        op: "log_density",
        message,
    }
}

/// Whether `kind` has an element-wise log density, so that a log density node of it can be built: the
/// [element-wise](DistributionKind::is_elementwise) kinds, the Bernoulli and its logit form.
pub(crate) fn has_elementwise_density(kind: &DistributionKind) -> bool {
    kind.is_elementwise()
        || matches!(
            kind,
            DistributionKind::Bernoulli | DistributionKind::BernoulliLogit
        )
}

/// Whether `kind` has a joint log density, so that a log density node of it is the scalar log probability of the
/// whole value: the linear Gaussian state space model.
pub(crate) fn has_joint_density(kind: &DistributionKind) -> bool {
    matches!(kind, DistributionKind::LinearGaussianStateSpace)
}

/// The element-wise log density of `value` given the parameters and, with `gradient`, its element-wise derivatives
/// with respect to the value and each parameter.
fn density(
    kind: &DistributionKind,
    value: &Value,
    parameters: &[&Value],
    gradient: bool,
) -> Result<(DoubleTensor, Option<LogProbGradientParts>)> {
    if kind.is_elementwise() {
        return kind.univariate(Univariate::Density, value, parameters, gradient);
    }
    // the Bernoulli and its logit form: their terms and gradients are element-wise, broadcast with the parameter
    let terms = kind.log_prob_terms(value, parameters)?;
    let parts = if gradient {
        Some(kind.log_prob_gradient(value, parameters)?)
    } else {
        None
    };
    Ok((terms, parts))
}

/// The number of parents of a log density of `kind` (the value and each parameter), if fixed.
pub(crate) fn arity(kind: &DistributionKind) -> Option<usize> {
    kind.parameter_types().map(|types| types.len() + 1)
}

/// The type and shape of a log density of `kind` given its parents' (the value's, then each parameter's): f64, the
/// broadcast of all their shapes (a scalar for a kind with a [joint density](has_joint_density), whose value and
/// parameters must have the shapes a node of the kind would). Fails for a kind that is neither element-wise nor
/// joint and for parents of the wrong types.
pub(crate) fn output_type(
    kind: &DistributionKind,
    parents: &[(DType, &[usize])],
) -> Result<(DType, Shape)> {
    let joint = has_joint_density(kind);
    if !has_elementwise_density(kind) && !joint {
        return Err(invalid(format!(
            "the {} distribution does not have one value per element with an element-wise density, so it has no \
             element-wise log density",
            kind.name()
        )));
    }
    let types = kind.parameter_types().ok_or_else(|| {
        invalid(format!(
            "the {} distribution has no parameter types",
            kind.name()
        ))
    })?;
    let Some(((value_type, value_shape), parameters)) = parents.split_first() else {
        return Err(invalid("needs the value as its first parent".into()));
    };
    if *value_type != kind.value_type() {
        return Err(invalid(format!(
            "the {} distribution's values are {:?}, not {value_type:?}",
            kind.name(),
            kind.value_type()
        )));
    }
    if parameters.len() != types.len()
        || parameters
            .iter()
            .zip(types.iter())
            .any(|((t, _), e)| t != e)
    {
        return Err(invalid(format!(
            "the {} distribution needs parameters of types {types:?}",
            kind.name()
        )));
    }
    if joint {
        let shape = DistributionOp::value_shape(kind, Some(value_shape), parameters)?;
        if shape.as_slice() != *value_shape {
            return Err(invalid(format!(
                "the {} distribution's value has shape {:?} for these parameters, not {value_shape:?}",
                kind.name(),
                shape.as_slice()
            )));
        }
        return Ok((DType::F64, Shape::new()));
    }
    let mut shape = Shape::from_slice(value_shape);
    for (_, p) in parameters {
        shape = broadcast_shape(&shape, p)?;
    }
    Ok((DType::F64, shape))
}

/// The broadcast shape of the inputs (the value and the parameters).
fn shape_of(inputs: &[&Value]) -> Result<Shape> {
    let mut shape = Shape::new();
    for input in inputs {
        shape = broadcast_shape(&shape, input.shape())?;
    }
    Ok(shape)
}

/// The log density of each element of the value (`inputs[0]`) given the parameters (`inputs[1..]`).
pub(crate) fn eval(kind: &DistributionKind, inputs: &[&Value]) -> Result<Value> {
    let (value, parameters) = split(kind, inputs)?;
    if has_joint_density(kind) {
        return Ok(Value::F64(DoubleTensor::scalar(
            kind.log_prob(value, parameters)?,
        )));
    }
    let shape = shape_of(inputs)?;
    let (densities, _) = density(kind, value, parameters, false)?;
    Ok(Value::F64(DoubleTensor::new(
        &shape,
        spread(&densities, &shape)?,
    )?))
}

fn split<'a, 'v>(
    kind: &DistributionKind,
    inputs: &'a [&'v Value],
) -> Result<(&'v Value, &'a [&'v Value])> {
    match inputs.split_first() {
        Some((value, parameters)) => Ok((value, parameters)),
        None => Err(kind.wrong_parameters()),
    }
}

/// The element-wise derivative of the log density with respect to each input (the value, then each parameter),
/// broadcast to the output shape; None for inputs it has none with respect to (non-f64 inputs among them).
fn derivatives(
    kind: &DistributionKind,
    inputs: &[&Value],
    shape: &[usize],
) -> Result<Vec<Option<DoubleTensor>>> {
    let (value, parameters) = split(kind, inputs)?;
    let (_, parts) = density(kind, value, parameters, true)?;
    let (value_gradient, parameter_gradients) =
        parts.unwrap_or_else(|| (None, smallvec::smallvec![None; parameters.len()]));
    std::iter::once(value_gradient)
        .chain(parameter_gradients)
        .zip(inputs)
        .map(|(g, input)| match g {
            Some(g) if input.dtype() == DType::F64 => {
                Ok(Some(DoubleTensor::new(shape, spread(&g, shape)?)?))
            }
            _ => Ok(None),
        })
        .collect()
}

/// Forward mode: Σᵢ ∂ log p / ∂ inputᵢ · ṫᵢ, element-wise, for the inputs' batched tangents `t` (`[b, input
/// shape...]`, None for a zero tangent).
pub(crate) fn jvp(
    kind: &DistributionKind,
    inputs: &[&Value],
    t: &[Option<&DoubleTensor>],
    b: usize,
    out_shape: &[usize],
) -> Result<Option<DoubleTensor>> {
    if t.iter().all(Option::is_none) {
        return Ok(None);
    }
    if has_joint_density(kind) {
        return joint_jvp(kind, inputs, t, b);
    }
    let mut total: Option<DoubleTensor> = None;
    for (d, t) in derivatives(kind, inputs, out_shape)?.into_iter().zip(t) {
        let (Some(d), Some(t)) = (d, t) else { continue };
        let term = fit((*t).clone(), b, out_shape)?.mul(&d)?;
        total = Some(match total {
            Some(sum) => sum.add(&term)?,
            None => term,
        });
    }
    Ok(total)
}

/// Reverse mode: for each input, the cotangent `g` (`[b, out...]`) times ∂ log p / ∂ input, summed to the input's
/// shape; None for inputs without a derivative.
pub(crate) fn vjp(
    kind: &DistributionKind,
    inputs: &[&Value],
    g: &DoubleTensor,
    out_shape: &[usize],
) -> Result<Vec<Option<DoubleTensor>>> {
    if has_joint_density(kind) {
        return joint_vjp(kind, inputs, g);
    }
    derivatives(kind, inputs, out_shape)?
        .into_iter()
        .zip(inputs)
        .map(|(d, input)| d.map(|d| reduce_to(g.mul(&d)?, input.shape())).transpose())
        .collect()
}

/// The derivative of [`vjp`] for the cotangent `g` (`[1, out...]`) along the inputs' forward tangents `t`: for
/// each input u, `g ⊙ Σᵥ ∂² log p / ∂u ∂v · ṫᵥ`, element-wise, summed to the input's shape (`[b, input...]`); None
/// where it is zero or the input has no derivative.
pub(crate) fn vjp_tangent(
    kind: &DistributionKind,
    inputs: &[&Value],
    g: &DoubleTensor,
    t: &[Option<&DoubleTensor>],
    b: usize,
    out_shape: &[usize],
) -> Result<Vec<Option<DoubleTensor>>> {
    let (value, parameters) = split(kind, inputs)?;
    if has_joint_density(kind) {
        return joint_vjp_tangent(kind, value, parameters, g, t, b);
    }
    let tangents: Vec<Option<Vec<f64>>> = t
        .iter()
        .map(|t| spread_tangent(*t, b, out_shape))
        .collect::<Result<_>>()?;
    let parts = elementwise(kind, value, parameters, &tangents, b, out_shape)?;
    let weights = g.as_slice();
    let len = weights.len();
    let batched = batched_shape(b, out_shape);
    parts
        .hvps
        .into_iter()
        .zip(inputs)
        .map(|(h, input)| match h {
            Some(mut h) if input.dtype() == DType::F64 => {
                for (j, x) in h.iter_mut().enumerate() {
                    *x *= weights[j % len];
                }
                Ok(Some(reduce_to(
                    DoubleTensor::new(&batched, h)?,
                    input.shape(),
                )?))
            }
            _ => Ok(None),
        })
        .collect()
}

/// A joint density's gradient with respect to each input (the value, then each parameter), each shaped like the
/// input; None for inputs it has none with respect to.
fn joint_derivatives(
    kind: &DistributionKind,
    inputs: &[&Value],
) -> Result<Vec<Option<DoubleTensor>>> {
    let (value, parameters) = split(kind, inputs)?;
    let (value_gradient, parameter_gradients) = kind.log_prob_gradient(value, parameters)?;
    Ok(std::iter::once(value_gradient)
        .chain(parameter_gradients)
        .zip(inputs)
        .map(|(g, input)| g.filter(|_| input.dtype() == DType::F64))
        .collect())
}

/// Forward mode for a joint density: Σᵢ ⟨∂ log p / ∂ inputᵢ, ṫᵢ⟩ for each of the `b` directions, `[b]`.
fn joint_jvp(
    kind: &DistributionKind,
    inputs: &[&Value],
    t: &[Option<&DoubleTensor>],
    b: usize,
) -> Result<Option<DoubleTensor>> {
    let mut total = vec![0.0; b];
    for ((d, t), input) in joint_derivatives(kind, inputs)?
        .into_iter()
        .zip(t)
        .zip(inputs)
    {
        let (Some(d), Some(t)) = (d, t) else { continue };
        let t = fit((*t).clone(), b, input.shape())?;
        let n = d.len();
        for (k, direction) in t.as_slice().chunks(n.max(1)).enumerate().take(b) {
            total[k] += direction
                .iter()
                .zip(d.as_slice())
                .map(|(a, b)| a * b)
                .sum::<f64>();
        }
    }
    Ok(Some(DoubleTensor::new(&[b], total)?))
}

/// Reverse mode for a joint density: for each input, each cotangent of `g` (`[b]`) times the gradient,
/// `[b, input...]`.
fn joint_vjp(
    kind: &DistributionKind,
    inputs: &[&Value],
    g: &DoubleTensor,
) -> Result<Vec<Option<DoubleTensor>>> {
    joint_derivatives(kind, inputs)?
        .into_iter()
        .map(|d| {
            d.map(|d| -> Result<DoubleTensor> { Ok(lift(g, d.rank())?.mul(&d)?) })
                .transpose()
        })
        .collect()
}

/// The derivative of [`joint_vjp`] for the cotangent `g` (`[1]`) along the inputs' tangents: `g` times the
/// distribution's Hessian-vector products, `[b, input...]`.
fn joint_vjp_tangent(
    kind: &DistributionKind,
    value: &Value,
    parameters: &[&Value],
    g: &DoubleTensor,
    t: &[Option<&DoubleTensor>],
    b: usize,
) -> Result<Vec<Option<DoubleTensor>>> {
    let Some((&value_tangent, parameter_tangents)) = t.split_first() else {
        return Err(kind.wrong_parameters());
    };
    let weight = g.as_slice().first().copied().unwrap_or(0.0);
    let (value_hvp, parameter_hvps) =
        kind.log_prob_hvp(value, parameters, value_tangent, parameter_tangents, b)?;
    std::iter::once(value_hvp)
        .chain(parameter_hvps)
        .map(|h| h.map(|h| Ok(h.map(|x| x * weight))).transpose())
        .collect()
}
