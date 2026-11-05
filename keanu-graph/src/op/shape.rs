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
/// agrees with evaluation. Ops where dummy values could fail or be expensive have explicit rules.
pub(crate) fn output_type(op: &Op, parents: &[(DType, &[usize])]) -> Result<(DType, Shape)> {
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
            if parents.len() != conditions + outcomes + 1 {
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
        Op::Gather { .. } | Op::ScatterAdd { .. } => {
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
            let n = conditions + keys.len() + 1;
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
