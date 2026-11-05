//! Output type and shape of a node, computed when the node is created.

use super::*;
use crate::error::GraphError;
use keanu_tensor::{Shape, broadcast_shape};

fn invalid(op: &Op, message: impl Into<String>) -> GraphError {
    GraphError::InvalidNode {
        op: op.name(),
        message: message.into(),
    }
}

/// The most elements a node's value may have: at 8 bytes an element (the size of an `f64` or `i64`), the value must
/// fit in `isize::MAX` bytes, the most one allocation can hold. Node creation rejects larger shapes, so a graph (or a
/// loaded file) never asks evaluation for an allocation that can't exist.
pub(crate) const MAX_ELEMENTS: usize = isize::MAX as usize / 8;

/// The number of elements of a shape, or `None` if it exceeds [`MAX_ELEMENTS`].
pub(crate) fn element_count(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .filter(|&n| n <= MAX_ELEMENTS)
}

fn check_elements(op: &Op, shape: &[usize]) -> Result<()> {
    match element_count(shape) {
        Some(_) => Ok(()),
        None => Err(invalid(
            op,
            format!("the shape {shape:?} has too many elements (at most {MAX_ELEMENTS})"),
        )),
    }
}

fn square_batch(op: &Op, shape: &[usize]) -> Result<()> {
    let r = shape.len();
    if r < 2 || shape[r - 1] != shape[r - 2] {
        return Err(invalid(
            op,
            format!("needs square matrices, got shape {shape:?}"),
        ));
    }
    Ok(())
}

/// The type and shape of a node's value given its parents' types and shapes.
///
/// Most ops are checked by evaluating them on dummy values of the parents' shapes, which guarantees the shape
/// agrees with evaluation. Ops where dummy values could fail or be expensive have explicit rules. Every output shape
/// has at most [`MAX_ELEMENTS`] elements.
pub(crate) fn output_type(op: &Op, parents: &[(DType, &[usize])]) -> Result<(DType, Shape)> {
    let (dtype, shape) = infer(op, parents)?;
    check_elements(op, &shape)?;
    Ok((dtype, shape))
}

fn infer(op: &Op, parents: &[(DType, &[usize])]) -> Result<(DType, Shape)> {
    let f64_parents = |n: usize| -> Result<()> {
        if parents.len() != n || parents.iter().any(|(t, _)| *t != DType::F64) {
            return Err(invalid(op, format!("needs {n} f64 parents")));
        }
        Ok(())
    };
    check_arity(op, parents.len())?;
    match op {
        Op::Constant(v) => Ok((v.dtype(), Shape::from_slice(v.shape()))),
        Op::Proxy | Op::Distribution(_) => Err(invalid(
            op,
            "proxy and probabilistic nodes are created with add_proxy and add_distribution",
        )),
        Op::Lambda(l) => Ok((l.dtype, Shape::from_slice(&l.shape))),
        Op::Scan(body) => body.output_type(parents),
        Op::MatMul {
            transpose_left,
            transpose_right,
        } => {
            f64_parents(2)?;
            let (a, b) = (parents[0].1, parents[1].1);
            if a.len() < 2 || b.len() < 2 {
                return Err(invalid(
                    op,
                    format!("needs rank >= 2 operands, got {a:?} and {b:?}"),
                ));
            }
            let (ar, ac) = (a[a.len() - 2], a[a.len() - 1]);
            let (br, bc) = (b[b.len() - 2], b[b.len() - 1]);
            let (m, k) = if *transpose_left { (ac, ar) } else { (ar, ac) };
            let (k2, n) = if *transpose_right { (bc, br) } else { (br, bc) };
            if k != k2 {
                return Err(invalid(op, format!("cannot multiply {a:?} by {b:?}")));
            }
            let mut shape = broadcast_shape(&a[..a.len() - 2], &b[..b.len() - 2])?;
            shape.extend([m, n]);
            Ok((DType::F64, shape))
        }
        Op::Inverse | Op::Cholesky | Op::CholeskyInverse => {
            f64_parents(1)?;
            square_batch(op, parents[0].1)?;
            Ok((DType::F64, Shape::from_slice(parents[0].1)))
        }
        Op::Determinant | Op::LogAbsDeterminant | Op::DeterminantSign => {
            f64_parents(1)?;
            let s = parents[0].1;
            square_batch(op, s)?;
            Ok((DType::F64, Shape::from_slice(&s[..s.len() - 2])))
        }
        Op::EighVectors => {
            f64_parents(1)?;
            square_batch(op, parents[0].1)?;
            Ok((DType::F64, Shape::from_slice(parents[0].1)))
        }
        Op::Eigvalsh => {
            f64_parents(1)?;
            let s = parents[0].1;
            square_batch(op, s)?;
            Ok((DType::F64, Shape::from_slice(&s[..s.len() - 1])))
        }
        Op::SolveTriangular { .. } | Op::Solve => {
            // explicit, as a dummy matrix of ones is singular
            f64_parents(2)?;
            let (a, b) = (parents[0].1, parents[1].1);
            square_batch(op, a)?;
            let n = a[a.len() - 1];
            // a vector right-hand side [n], or matrices [..., n, k]
            let (b_batch, tail) = match b {
                [] => return Err(invalid(op, "needs a right-hand side of rank >= 1")),
                [_] => (&b[..0], b),
                _ => b.split_at(b.len() - 2),
            };
            if tail[0] != n {
                return Err(invalid(
                    op,
                    format!("cannot solve {a:?} against right-hand side {b:?}"),
                ));
            }
            let mut shape = broadcast_shape(&a[..a.len() - 2], b_batch)?;
            shape.extend_from_slice(tail);
            Ok((DType::F64, shape))
        }
        Op::Multiplexer => {
            if parents.len() < 2 || parents[0] != (DType::I64, &[][..]) {
                return Err(invalid(
                    op,
                    "needs a scalar i64 selector and at least one option",
                ));
            }
            let first = parents[1];
            if parents[1..].iter().any(|p| *p != first) {
                return Err(invalid(op, "options must have the same type and shape"));
            }
            Ok((first.0, Shape::from_slice(first.1)))
        }
        Op::Cpt { conditions, keys } => {
            let (conditions, outcomes) = (*conditions, keys.len());
            if conditions >= parents.len() || parents.len() - conditions != outcomes + 1 {
                return Err(invalid(
                    op,
                    "needs the conditions, one outcome per key and a default",
                ));
            }
            if parents[..conditions]
                .iter()
                .any(|(t, s)| *t == DType::F64 || !s.iter().all(|&d| d == 1))
            {
                return Err(invalid(
                    op,
                    "conditions must be scalar booleans or integers",
                ));
            }
            if keys.iter().any(|k| k.len() != conditions) {
                return Err(invalid(op, "every key needs one value per condition"));
            }
            let default = parents[parents.len() - 1];
            if parents[conditions..].iter().any(|p| *p != default) {
                return Err(invalid(
                    op,
                    "outcomes and default must have the same type and shape",
                ));
            }
            Ok((default.0, Shape::from_slice(default.1)))
        }
        Op::TensorMultiply { .. } => {
            f64_parents(2)?;
            dummy(op, parents)
        }
        Op::Broadcast(target) => {
            // explicit, so a large target is checked before anything of its size is allocated
            check_elements(op, target)?;
            let source = parents[0].1;
            let shape = broadcast_shape(source, target)?;
            if shape.as_slice() != target.as_slice() {
                return Err(keanu_tensor::TensorError::IncompatibleShapes {
                    left: source.to_vec(),
                    right: target.clone(),
                }
                .into());
            }
            Ok((parents[0].0, shape))
        }
        Op::Gather { .. } | Op::ScatterAdd { .. } => {
            if let Op::ScatterAdd { axis, size } = op {
                // the output replaces the axis's length with `size`: check it before the dummy evaluation allocates it
                let source = parents[0].1;
                let rank = source.len() as isize;
                if (-rank..rank).contains(axis) {
                    let mut shape = source.to_vec();
                    shape[axis.rem_euclid(rank) as usize] = *size;
                    check_elements(op, &shape)?;
                }
            }
            if parents[1].0 != DType::I64 {
                return Err(invalid(op, "the indices must be i64"));
            }
            // zero indices are in range wherever any index could be, so the dummy evaluation checks only shapes
            let values = [
                Value::ones(parents[0].0, parents[0].1),
                Value::I64(keanu_tensor::IntTensor::full(parents[1].1, 0)),
            ];
            let out = eval(op, &[&values[0], &values[1]])?;
            Ok((out.dtype(), Shape::from_slice(out.shape())))
        }
        _ => dummy(op, parents),
    }
}

/// The number of parents each op takes: exact, or at least `min` (`max` None). Every deterministic op's parent
/// count is checked here, once, when the node is created, so evaluation and auto diff can index parents freely.
fn check_arity(op: &Op, count: usize) -> Result<()> {
    let (min, max) = match op {
        Op::Constant(_) => (0, Some(0)),
        Op::Unary(_)
        | Op::Not
        | Op::All
        | Op::Any
        | Op::Predicate(_)
        | Op::Reduce { .. }
        | Op::Standardize
        | Op::Softmax { .. }
        | Op::LogSoftmax { .. }
        | Op::Cumulative { .. }
        | Op::ArgReduce { .. }
        | Op::Reshape(_)
        | Op::Permute(_)
        | Op::Transpose
        | Op::Broadcast(_)
        | Op::Slice(_)
        | Op::Take(_)
        | Op::Diag
        | Op::DiagPart
        | Op::Tril(_)
        | Op::Triu(_)
        | Op::FillTriangular { .. }
        | Op::TrianglePart { .. }
        | Op::Inverse
        | Op::Determinant
        | Op::Cholesky
        | Op::CholeskyInverse
        | Op::LogAbsDeterminant
        | Op::DeterminantSign
        | Op::Eigvalsh
        | Op::EighVectors
        | Op::Cast(_)
        | Op::ReplaceNan(_)
        | Op::Assert(_) => (1, Some(1)),
        Op::Binary(_)
        | Op::Compare(_)
        | Op::Mask(_)
        | Op::ApproxEq(_)
        | Op::Logic(_)
        | Op::MatMul { .. }
        | Op::TensorMultiply { .. }
        | Op::SolveTriangular { .. }
        | Op::Solve
        | Op::Gather { .. }
        | Op::ScatterAdd { .. } => (2, Some(2)),
        Op::Where | Op::SetWithMask => (3, Some(3)),
        Op::Concat(_) | Op::Stack(_) => (1, None),
        Op::Multiplexer => (2, None),
        Op::Cpt { conditions, keys } => {
            let Some(n) = conditions
                .checked_add(keys.len())
                .and_then(|n| n.checked_add(1))
            else {
                return Err(invalid(
                    op,
                    format!("{conditions} conditions is more parents than a node can have"),
                ));
            };
            (n, Some(n))
        }
        Op::Scan(body) => {
            let n = body.parent_count();
            (n, Some(n))
        }
        // a lambda takes any parents; proxies and distributions are rejected by the caller
        Op::Lambda(_) | Op::Proxy | Op::Distribution(_) => (0, None),
    };
    let ok = count >= min && max.is_none_or(|max| count <= max);
    if ok {
        return Ok(());
    }
    Err(invalid(
        op,
        match max {
            Some(max) if max == min => format!("needs {min} parents, got {count}"),
            _ => format!("needs at least {min} parents, got {count}"),
        },
    ))
}

fn dummy(op: &Op, parents: &[(DType, &[usize])]) -> Result<(DType, Shape)> {
    let values: Vec<Value> = parents.iter().map(|(t, s)| Value::ones(*t, s)).collect();
    let refs: Vec<&Value> = values.iter().collect();
    let out = eval(op, &refs)?;
    Ok((out.dtype(), Shape::from_slice(out.shape())))
}
