//! The log-density op ([`Op::LogDensity`](super::Op::LogDensity)): the element-wise log density of a value under an
//! element-wise distribution, as a deterministic node. Its value, first derivatives and second derivatives reuse the
//! distribution's element-wise rules (`DistributionKind::univariate` and the scalar second-order rules), so a
//! log density on the graph differentiates exactly as the same distribution's node does. The Bernoulli and its logit
//! form, whose values are boolean and which therefore can't be mixture components, have element-wise densities too:
//! their values and derivatives come from their nodes' unreduced log probability terms and gradients, which are
//! element-wise.

use super::distribution::{DistributionKind, LogProbGradientParts, Univariate};
use super::distribution_hvp::{elementwise, spread, spread_tangent};
use crate::autodiff::{batched_shape, fit, reduce_to};
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
/// broadcast of all their shapes. Fails for a kind that isn't element-wise and for parents of the wrong types.
pub(crate) fn output_type(
    kind: &DistributionKind,
    parents: &[(DType, &[usize])],
) -> Result<(DType, Shape)> {
    if !has_elementwise_density(kind) {
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
