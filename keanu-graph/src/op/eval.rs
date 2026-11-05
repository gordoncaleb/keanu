//! Evaluation of every deterministic op from its parents' values.

use super::*;
use crate::error::GraphError;
use keanu_tensor::{BoolTensor, DoubleTensor, IntTensor};
use smallvec::SmallVec;

fn invalid(op: &Op, message: impl Into<String>) -> GraphError {
    GraphError::InvalidNode {
        op: op.name(),
        message: message.into(),
    }
}

fn unary_f64(op: UnaryOp, x: &DoubleTensor) -> DoubleTensor {
    match op {
        UnaryOp::Abs => x.abs(),
        UnaryOp::Neg => x.neg(),
        UnaryOp::Sign => x.sign(),
        UnaryOp::Reciprocal => x.reciprocal(),
        UnaryOp::Exp => x.exp(),
        UnaryOp::Exp2 => x.exp2(),
        UnaryOp::ExpM1 => x.exp_m1(),
        UnaryOp::Ln => x.ln(),
        UnaryOp::Ln1p => x.ln_1p(),
        UnaryOp::Log2 => x.log2(),
        UnaryOp::Log10 => x.log10(),
        UnaryOp::Sqrt => x.sqrt(),
        UnaryOp::Sin => x.sin(),
        UnaryOp::Cos => x.cos(),
        UnaryOp::Tan => x.tan(),
        UnaryOp::Asin => x.asin(),
        UnaryOp::Acos => x.acos(),
        UnaryOp::Atan => x.atan(),
        UnaryOp::Sinh => x.sinh(),
        UnaryOp::Cosh => x.cosh(),
        UnaryOp::Tanh => x.tanh(),
        UnaryOp::Asinh => x.asinh(),
        UnaryOp::Acosh => x.acosh(),
        UnaryOp::Atanh => x.atanh(),
        UnaryOp::Sigmoid => x.sigmoid(),
        UnaryOp::Floor => x.floor(),
        UnaryOp::Ceil => x.ceil(),
        UnaryOp::Round => x.round(),
        UnaryOp::LnGamma => x.ln_gamma(),
        UnaryOp::Digamma => x.digamma(),
        UnaryOp::Trigamma => x.trigamma(),
        UnaryOp::Erf => x.erf(),
        UnaryOp::Erfc => x.erfc(),
        UnaryOp::ErfInv => x.erf_inv(),
        UnaryOp::LogNdtr => x.log_ndtr(),
        UnaryOp::Ndtri => x.ndtri(),
        UnaryOp::Softplus => x.softplus(),
        UnaryOp::Logit => x.logit(),
        UnaryOp::Log1mExp => x.log1mexp(),
    }
}

fn binary_f64(op: BinaryOp, a: &DoubleTensor, b: &DoubleTensor) -> Result<Option<DoubleTensor>> {
    Ok(Some(match op {
        BinaryOp::Add => a.add(b)?,
        BinaryOp::Sub => a.sub(b)?,
        BinaryOp::Mul => a.mul(b)?,
        BinaryOp::Div => a.div(b)?,
        BinaryOp::Pow => a.pow(b)?,
        BinaryOp::Max => a.maximum(b)?,
        BinaryOp::Min => a.minimum(b)?,
        BinaryOp::Atan2 => a.atan2(b)?,
        BinaryOp::LogAddExp => a.log_add_exp(b)?,
        BinaryOp::LogAddExp2 => a.log_add_exp2(b)?,
        BinaryOp::SafeLnTimes => a.safe_ln_times(b)?,
        BinaryOp::LogBeta => a.log_beta(b)?,
        BinaryOp::Rem => return Ok(None),
    }))
}

fn binary_i64(op: BinaryOp, a: &IntTensor, b: &IntTensor) -> Result<Option<IntTensor>> {
    Ok(Some(match op {
        BinaryOp::Add => a.add(b)?,
        BinaryOp::Sub => a.sub(b)?,
        BinaryOp::Mul => a.mul(b)?,
        BinaryOp::Div => a.div(b)?,
        BinaryOp::Pow => a.pow(b)?,
        BinaryOp::Max => a.maximum(b)?,
        BinaryOp::Min => a.minimum(b)?,
        BinaryOp::Rem => a.rem(b)?,
        _ => return Ok(None),
    }))
}

fn compare<T: keanu_tensor::Numeric>(
    op: CompareOp,
    a: &keanu_tensor::Tensor<T>,
    b: &keanu_tensor::Tensor<T>,
) -> Result<BoolTensor> {
    Ok(match op {
        CompareOp::Gt => a.gt(b)?,
        CompareOp::Ge => a.ge(b)?,
        CompareOp::Lt => a.lt(b)?,
        CompareOp::Le => a.le(b)?,
        CompareOp::Eq => a.elementwise_eq(b)?,
        CompareOp::Ne => a.elementwise_ne(b)?,
    })
}

fn same_dtype(op: &Op, inputs: &[&Value]) -> Result<DType> {
    let first = inputs
        .first()
        .ok_or_else(|| invalid(op, "needs at least one parent"))?
        .dtype();
    if inputs.iter().any(|v| v.dtype() != first) {
        return Err(invalid(op, "parents must have the same type"));
    }
    Ok(first)
}

/// Applies a shape operation that is defined for every element type.
macro_rules! any_dtype {
    ($value:expr, |$t:ident| $body:expr) => {
        match $value {
            Value::F64($t) => Value::F64($body),
            Value::I64($t) => Value::I64($body),
            Value::Bool($t) => Value::Bool($body),
        }
    };
}

/// Evaluates a deterministic op. Distribution nodes are sampled elsewhere; proxies are resolved by the caller.
pub(crate) fn eval(op: &Op, inputs: &[&Value]) -> Result<Value> {
    Ok(match op {
        Op::Constant(v) => v.clone(),

        Op::Unary(u) => {
            expect_parents(op, inputs, 1)?;
            match inputs[0] {
                Value::F64(x) => Value::F64(unary_f64(*u, x)),
                Value::I64(x) if u.applies_to_integers() => Value::I64(match u {
                    UnaryOp::Abs => x.abs(),
                    UnaryOp::Neg => x.neg(),
                    _ => x.sign(),
                }),
                other => {
                    return Err(invalid(
                        op,
                        format!("{u:?} does not apply to {:?}", other.dtype()),
                    ));
                }
            }
        }

        Op::Binary(b) => {
            expect_parents(op, inputs, 2)?;
            match (inputs[0], inputs[1]) {
                (Value::F64(x), Value::F64(y)) => binary_f64(*b, x, y)?.map(Value::F64),
                (Value::I64(x), Value::I64(y)) => binary_i64(*b, x, y)?.map(Value::I64),
                _ => None,
            }
            .ok_or_else(|| {
                invalid(
                    op,
                    format!(
                        "{b:?} does not apply to {:?} and {:?}",
                        inputs[0].dtype(),
                        inputs[1].dtype()
                    ),
                )
            })?
        }

        Op::Compare(c) => {
            expect_parents(op, inputs, 2)?;
            Value::Bool(match (inputs[0], inputs[1]) {
                (Value::F64(x), Value::F64(y)) => compare(*c, x, y)?,
                (Value::I64(x), Value::I64(y)) => compare(*c, x, y)?,
                (Value::Bool(x), Value::Bool(y)) if matches!(c, CompareOp::Eq | CompareOp::Ne) => {
                    let eq = matches!(c, CompareOp::Eq);
                    x.zip_with(y, |a, b| (a == b) == eq)?
                }
                _ => return Err(invalid(op, "operands must have the same numeric type")),
            })
        }

        Op::Mask(c) => {
            expect_parents(op, inputs, 2)?;
            let (x, y) = (inputs[0].f64()?, inputs[1].f64()?);
            Value::F64(match c {
                CompareOp::Gt => x.gt_mask(y)?,
                CompareOp::Ge => x.ge_mask(y)?,
                CompareOp::Lt => x.lt_mask(y)?,
                CompareOp::Le => x.le_mask(y)?,
                CompareOp::Eq => x.zip_with(y, |a, b| f64::from(u8::from(a == b)))?,
                CompareOp::Ne => x.zip_with(y, |a, b| f64::from(u8::from(a != b)))?,
            })
        }

        Op::ApproxEq(epsilon) => {
            expect_parents(op, inputs, 2)?;
            let e = *epsilon;
            Value::Bool(match (inputs[0], inputs[1]) {
                (Value::F64(x), Value::F64(y)) => x.zip_with(y, |a, b| (a - b).abs() <= e)?,
                (Value::I64(x), Value::I64(y)) => {
                    x.zip_with(y, |a, b| ((a - b) as f64).abs() <= e)?
                }
                _ => return Err(invalid(op, "operands must have the same numeric type")),
            })
        }

        Op::Logic(l) => {
            expect_parents(op, inputs, 2)?;
            let (x, y) = (inputs[0].bool()?, inputs[1].bool()?);
            Value::Bool(match l {
                LogicOp::And => x.and(y)?,
                LogicOp::Or => x.or(y)?,
                LogicOp::Xor => x.xor(y)?,
            })
        }
        Op::Not => Value::Bool(inputs[0].bool()?.not()),
        Op::All => Value::Bool(BoolTensor::scalar(inputs[0].bool()?.all())),
        Op::Any => Value::Bool(BoolTensor::scalar(inputs[0].bool()?.any())),
        Op::Predicate(p) => {
            let x = inputs[0].f64()?;
            Value::Bool(match p {
                Predicate::IsNan => x.map(f64::is_nan),
                Predicate::NotNan => x.map(|v| !v.is_nan()),
                Predicate::IsFinite => x.map(f64::is_finite),
                Predicate::IsInfinite => x.map(f64::is_infinite),
                Predicate::IsPositiveInfinity => x.map(|v| v == f64::INFINITY),
                Predicate::IsNegativeInfinity => x.map(|v| v == f64::NEG_INFINITY),
            })
        }

        Op::Reduce { kind, axes } => {
            expect_parents(op, inputs, 1)?;
            reduce(op, *kind, axes.as_deref(), inputs[0])?
        }
        Op::Standardize => Value::F64(inputs[0].f64()?.standardize()),
        Op::Softmax { axis } => Value::F64(inputs[0].f64()?.softmax(*axis)?),
        Op::LogSoftmax { axis } => Value::F64(inputs[0].f64()?.log_softmax(*axis)?),
        Op::Cumulative { kind, axis } => match (inputs[0], kind) {
            (Value::F64(x), CumulativeOp::CumSum) => Value::F64(x.cumsum(*axis)?),
            (Value::F64(x), CumulativeOp::CumProd) => Value::F64(x.cumprod(*axis)?),
            (Value::I64(x), CumulativeOp::CumSum) => Value::I64(x.cumsum(*axis)?),
            (Value::I64(x), CumulativeOp::CumProd) => Value::I64(x.cumprod(*axis)?),
            _ => return Err(invalid(op, "cumulative sums and products apply to numbers")),
        },
        Op::ArgReduce { kind, axis } => arg_reduce(op, *kind, *axis, inputs[0])?,

        Op::Reshape(shape) => any_dtype!(inputs[0], |t| t.reshape(shape)?),
        Op::Permute(p) => any_dtype!(inputs[0], |t| t.permute(p)?),
        Op::Transpose => any_dtype!(inputs[0], |t| t.transpose()?),
        Op::Broadcast(shape) => any_dtype!(inputs[0], |t| t.broadcast_to(shape)?),
        Op::Slice(spec) => any_dtype!(inputs[0], |t| t.slice(spec)?),
        Op::Concat(axis) => match same_dtype(op, inputs)? {
            DType::F64 => Value::F64(DoubleTensor::concat(
                *axis,
                &inputs.iter().map(|v| v.f64()).collect::<Result<Vec<_>>>()?,
            )?),
            DType::I64 => Value::I64(IntTensor::concat(
                *axis,
                &inputs.iter().map(|v| v.i64()).collect::<Result<Vec<_>>>()?,
            )?),
            DType::Bool => Value::Bool(BoolTensor::concat(
                *axis,
                &inputs
                    .iter()
                    .map(|v| v.bool())
                    .collect::<Result<Vec<_>>>()?,
            )?),
        },
        Op::Gather { axis } => {
            let indices = inputs[1].i64()?;
            any_dtype!(inputs[0], |t| t.gather(*axis, indices)?)
        }
        Op::ScatterAdd { axis, size } => {
            let indices = inputs[1].i64()?;
            match inputs[0] {
                Value::F64(t) => Value::F64(t.scatter_add(*axis, indices, *size)?),
                Value::I64(t) => Value::I64(t.scatter_add(*axis, indices, *size)?),
                Value::Bool(_) => return Err(invalid(op, "scatter-add applies to numbers")),
            }
        }
        Op::Stack(axis) => match same_dtype(op, inputs)? {
            DType::F64 => Value::F64(DoubleTensor::stack(
                *axis,
                &inputs.iter().map(|v| v.f64()).collect::<Result<Vec<_>>>()?,
            )?),
            DType::I64 => Value::I64(IntTensor::stack(
                *axis,
                &inputs.iter().map(|v| v.i64()).collect::<Result<Vec<_>>>()?,
            )?),
            DType::Bool => Value::Bool(BoolTensor::stack(
                *axis,
                &inputs
                    .iter()
                    .map(|v| v.bool())
                    .collect::<Result<Vec<_>>>()?,
            )?),
        },
        Op::Take(index) => {
            let take = |shape: &[usize]| {
                invalid(
                    op,
                    format!("index {index:?} is out of range for shape {shape:?}"),
                )
            };
            match inputs[0] {
                Value::F64(t) => Value::F64(DoubleTensor::scalar(
                    t.get(index).ok_or_else(|| take(t.shape()))?,
                )),
                Value::I64(t) => Value::I64(IntTensor::scalar(
                    t.get(index).ok_or_else(|| take(t.shape()))?,
                )),
                Value::Bool(t) => Value::Bool(BoolTensor::scalar(
                    t.get(index).ok_or_else(|| take(t.shape()))?,
                )),
            }
        }
        Op::Diag => any_dtype!(inputs[0], |t| t.diag()?),
        Op::DiagPart => any_dtype!(inputs[0], |t| t.diag_part()?),
        Op::Tril(k) => any_dtype!(inputs[0], |t| t.tril(*k)?),
        Op::Triu(k) => any_dtype!(inputs[0], |t| t.triu(*k)?),
        Op::FillTriangular { upper, lower } => {
            any_dtype!(inputs[0], |t| t.fill_triangular(*upper, *lower)?)
        }
        Op::TrianglePart { upper } => any_dtype!(inputs[0], |t| t.triangle_part(*upper)?),

        Op::MatMul {
            transpose_left,
            transpose_right,
        } => Value::F64(inputs[0].f64()?.matmul_transposed(
            inputs[1].f64()?,
            *transpose_left,
            *transpose_right,
        )?),
        Op::TensorMultiply {
            left_axes,
            right_axes,
        } => Value::F64(inputs[0].f64()?.tensor_multiply(
            inputs[1].f64()?,
            left_axes,
            right_axes,
        )?),
        Op::Inverse => Value::F64(inputs[0].f64()?.inverse()?),
        Op::Determinant => Value::F64(inputs[0].f64()?.determinant()?),
        Op::Cholesky => Value::F64(inputs[0].f64()?.cholesky()?),
        Op::CholeskyInverse => Value::F64(inputs[0].f64()?.cholesky_inverse()?),
        Op::SolveTriangular { lower, transpose } => Value::F64(inputs[0].f64()?.solve_triangular(
            inputs[1].f64()?,
            *lower,
            *transpose,
        )?),
        Op::Solve => Value::F64(inputs[0].f64()?.solve(inputs[1].f64()?)?),
        Op::LogAbsDeterminant => Value::F64(inputs[0].f64()?.slogdet()?.1),
        Op::DeterminantSign => Value::F64(inputs[0].f64()?.slogdet()?.0),
        Op::Eigvalsh => Value::F64(inputs[0].f64()?.eigh()?.0),
        Op::EighVectors => Value::F64(inputs[0].f64()?.eigh()?.1),

        Op::Cast(to) => cast(inputs[0], *to),

        Op::Where => {
            expect_parents(op, inputs, 3)?;
            let predicate = inputs[0].bool()?;
            match (inputs[1], inputs[2]) {
                (Value::F64(a), Value::F64(b)) => Value::F64(select(predicate, a, b)?),
                (Value::I64(a), Value::I64(b)) => Value::I64(select(predicate, a, b)?),
                (Value::Bool(a), Value::Bool(b)) => Value::Bool(select(predicate, a, b)?),
                _ => return Err(invalid(op, "then and else must have the same type")),
            }
        }
        Op::Multiplexer => {
            inputs[multiplexer_selected(inputs).map_err(|e| invalid(op, e))?].clone()
        }
        Op::Cpt { conditions, keys } => {
            inputs[cpt_selected(*conditions, keys, inputs).map_err(|e| invalid(op, e))?].clone()
        }
        Op::SetWithMask => {
            expect_parents(op, inputs, 3)?;
            let (operand, mask) = (inputs[0].f64()?, inputs[1].f64()?);
            let value = inputs[2]
                .f64()?
                .to_scalar()
                .ok_or_else(|| invalid(op, "the value must be a scalar"))?;
            if operand.shape() != mask.shape() {
                return Err(invalid(
                    op,
                    format!(
                        "operand {:?} and mask {:?} must have the same shape",
                        operand.shape(),
                        mask.shape()
                    ),
                ));
            }
            Value::F64(operand.zip_with(mask, |x, m| if m == 1.0 { value } else { x })?)
        }
        Op::ReplaceNan(with) => {
            let with = *with;
            Value::F64(inputs[0].f64()?.map(|x| if x.is_nan() { with } else { x }))
        }
        Op::Proxy => {
            expect_parents(op, inputs, 1)?;
            inputs[0].clone()
        }
        Op::Scan(body) => body.eval(inputs)?,
        Op::Lambda(lambda) => {
            let out = (lambda.function)(inputs).map_err(GraphError::Lambda)?;
            if out.dtype() != lambda.dtype || out.shape() != lambda.shape.as_slice() {
                return Err(GraphError::Lambda(format!(
                    "{} returned a {:?} value of shape {:?}, declared {:?} of shape {:?}",
                    lambda.name,
                    out.dtype(),
                    out.shape(),
                    lambda.dtype,
                    lambda.shape
                )));
            }
            out
        }
        Op::Assert(message) => {
            let predicate = inputs[0].bool()?;
            if !predicate.all() {
                return Err(GraphError::AssertionFailed(
                    message
                        .clone()
                        .unwrap_or_else(|| "predicate is false".into()),
                ));
            }
            inputs[0].clone()
        }
        Op::Distribution(_) => {
            return Err(invalid(
                op,
                "probabilistic nodes are sampled, not calculated",
            ));
        }
    })
}

/// The input a multiplexer selects: its scalar integer selector (input 0) indexes the options (inputs 1..).
pub(crate) fn multiplexer_selected(inputs: &[&Value]) -> std::result::Result<usize, String> {
    let selector = inputs[0]
        .as_i64()
        .and_then(|t| t.to_scalar())
        .ok_or("the selector must be a scalar integer")?;
    let options = inputs.len() - 1;
    usize::try_from(selector)
        .ok()
        .filter(|&i| i < options)
        .map(|i| i + 1)
        .ok_or_else(|| format!("selector {selector} is out of range for {options} options"))
}

/// The input a CPT selects: the outcome (after the `conditions` inputs) whose key matches the conditions' values
/// (booleans as 0 and 1), otherwise the default (the last input).
pub(crate) fn cpt_selected(
    conditions: usize,
    keys: &[Vec<i64>],
    inputs: &[&Value],
) -> std::result::Result<usize, String> {
    let condition: SmallVec<[i64; 4]> = inputs[..conditions]
        .iter()
        .map(|v| match v {
            Value::Bool(t) => t.to_scalar().map(i64::from),
            Value::I64(t) => t.to_scalar(),
            Value::F64(_) => None,
        })
        .collect::<Option<_>>()
        .ok_or("conditions must be scalar booleans or integers")?;
    Ok(match keys.iter().position(|k| k[..] == condition[..]) {
        Some(i) => conditions + i,
        None => inputs.len() - 1,
    })
}

fn select<T: keanu_tensor::Element>(
    predicate: &BoolTensor,
    a: &keanu_tensor::Tensor<T>,
    b: &keanu_tensor::Tensor<T>,
) -> Result<keanu_tensor::Tensor<T>> {
    let shape = keanu_tensor::broadcast_shape(
        &keanu_tensor::broadcast_shape(predicate.shape(), a.shape())?,
        b.shape(),
    )?;
    let (p, a, b) = (
        predicate.broadcast_to(&shape)?,
        a.broadcast_to(&shape)?,
        b.broadcast_to(&shape)?,
    );
    let data = p
        .as_slice()
        .iter()
        .zip(a.as_slice())
        .zip(b.as_slice())
        .map(|((&p, &x), &y)| if p { x } else { y })
        .collect();
    Ok(keanu_tensor::Tensor::new(&shape, data)?)
}

fn cast(value: &Value, to: DType) -> Value {
    match (value, to) {
        (Value::F64(t), DType::F64) => Value::F64(t.clone()),
        (Value::F64(t), DType::I64) => Value::I64(t.to_int()),
        (Value::F64(t), DType::Bool) => Value::Bool(t.map(|x| x == 1.0)),
        (Value::I64(t), DType::F64) => Value::F64(t.to_double()),
        (Value::I64(t), DType::I64) => Value::I64(t.clone()),
        (Value::I64(t), DType::Bool) => Value::Bool(t.map(|x| x == 1)),
        (Value::Bool(t), DType::F64) => Value::F64(t.to_double_mask()),
        (Value::Bool(t), DType::I64) => Value::I64(t.to_int_mask()),
        (Value::Bool(t), DType::Bool) => Value::Bool(t.clone()),
    }
}

fn reduce(op: &Op, kind: ReduceOp, axes: Option<&[isize]>, input: &Value) -> Result<Value> {
    let none = || invalid(op, "cannot reduce an empty tensor");
    Ok(match (input, kind, axes) {
        (Value::F64(x), ReduceOp::Sum, None) => Value::F64(DoubleTensor::scalar(x.sum())),
        (Value::F64(x), ReduceOp::Sum, Some(a)) => Value::F64(x.sum_axes(a)?),
        (Value::F64(x), ReduceOp::Product, None) => Value::F64(DoubleTensor::scalar(x.product())),
        (Value::F64(x), ReduceOp::Product, Some(a)) => Value::F64(x.product_axes(a)?),
        (Value::F64(x), ReduceOp::Mean, None) => Value::F64(DoubleTensor::scalar(x.mean())),
        (Value::F64(x), ReduceOp::Mean, Some(a)) => Value::F64(x.mean_axes(a)?),
        (Value::F64(x), ReduceOp::Max, None) => {
            Value::F64(DoubleTensor::scalar(x.max().ok_or_else(none)?))
        }
        (Value::F64(x), ReduceOp::Min, None) => {
            Value::F64(DoubleTensor::scalar(x.min().ok_or_else(none)?))
        }
        (Value::F64(x), ReduceOp::StdDev, None) => Value::F64(DoubleTensor::scalar(x.std_dev())),
        (Value::F64(x), ReduceOp::LogSumExp, None) => {
            Value::F64(DoubleTensor::scalar(x.log_sum_exp()))
        }
        (Value::F64(x), ReduceOp::LogSumExp, Some(a)) => Value::F64(x.log_sum_exp_axes(a)?),
        (Value::I64(x), ReduceOp::Sum, None) => Value::I64(IntTensor::scalar(x.sum())),
        (Value::I64(x), ReduceOp::Sum, Some(a)) => Value::I64(x.sum_axes(a)?),
        (Value::I64(x), ReduceOp::Product, None) => Value::I64(IntTensor::scalar(x.product())),
        (Value::I64(x), ReduceOp::Product, Some(a)) => Value::I64(x.product_axes(a)?),
        (Value::I64(x), ReduceOp::Max, None) => {
            Value::I64(IntTensor::scalar(x.max().ok_or_else(none)?))
        }
        (Value::I64(x), ReduceOp::Min, None) => {
            Value::I64(IntTensor::scalar(x.min().ok_or_else(none)?))
        }
        (v, k, a) => {
            return Err(invalid(
                op,
                format!("{k:?} over {a:?} does not apply to {:?}", v.dtype()),
            ));
        }
    })
}

fn arg_reduce(op: &Op, kind: ArgOp, axis: Option<isize>, input: &Value) -> Result<Value> {
    let none = || invalid(op, "cannot take the arg max or min of an empty tensor");
    let flat = |i: Option<usize>| -> Result<Value> {
        Ok(Value::I64(IntTensor::scalar(i.ok_or_else(none)? as i64)))
    };
    match (input, kind, axis) {
        (Value::F64(x), ArgOp::ArgMax, None) => flat(x.argmax()),
        (Value::F64(x), ArgOp::ArgMin, None) => flat(x.argmin()),
        (Value::F64(x), ArgOp::NanArgMax, None) => flat(x.nan_argmax()),
        (Value::F64(x), ArgOp::NanArgMin, None) => flat(x.nan_argmin()),
        (Value::F64(x), ArgOp::ArgMax, Some(a)) => Ok(Value::I64(x.argmax_axis(a)?)),
        (Value::F64(x), ArgOp::ArgMin, Some(a)) => Ok(Value::I64(x.argmin_axis(a)?)),
        (Value::I64(x), ArgOp::ArgMax, None) => flat(x.argmax()),
        (Value::I64(x), ArgOp::ArgMin, None) => flat(x.argmin()),
        (Value::I64(x), ArgOp::ArgMax, Some(a)) => Ok(Value::I64(x.argmax_axis(a)?)),
        (Value::I64(x), ArgOp::ArgMin, Some(a)) => Ok(Value::I64(x.argmin_axis(a)?)),
        (v, k, a) => Err(invalid(
            op,
            format!("{k:?} over {a:?} does not apply to {:?}", v.dtype()),
        )),
    }
}
