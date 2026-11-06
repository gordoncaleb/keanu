//! Output type and shape of a node, computed when the node is created.

use super::*;
use crate::error::GraphError;
use keanu_tensor::ops::shape_ops::{reshape_shape, slice_shape};
use keanu_tensor::{Shape, TensorError, broadcast_shape};

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

/// An error message if a node of `shape` would have more than [`MAX_ELEMENTS`] elements.
pub(crate) fn check_element_count(shape: &[usize]) -> std::result::Result<(), String> {
    match element_count(shape) {
        Some(_) => Ok(()),
        None => Err(format!(
            "the shape {shape:?} has too many elements (at most {MAX_ELEMENTS})"
        )),
    }
}

fn check_elements(op: &Op, shape: &[usize]) -> Result<()> {
    check_element_count(shape).map_err(|message| invalid(op, message))
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
/// Every op has an explicit rule computing its output from the parents' types and shapes alone: nothing is
/// evaluated, so creating a node allocates nothing of the size of its value however large its shape (a lambda's
/// type and shape are declared, and its function only runs when the node is evaluated). The rules give the type and
/// shape evaluation gives, and fail where evaluation fails for every value of the parents' shapes; a unit test
/// checks them against evaluation. Every output shape has at most [`MAX_ELEMENTS`] elements.
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
        Op::Ode(body) => body.output_type(parents),
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
        Op::Inverse | Op::Cholesky | Op::CholeskyInverse | Op::MatrixExp => {
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
        Op::SparseMatMul(a) => {
            f64_parents(1)?;
            let s = parents[0].1;
            match s {
                [n] if *n == a.cols() => Ok((DType::F64, Shape::from_slice(&[a.rows()]))),
                [n, k] if *n == a.cols() => Ok((DType::F64, Shape::from_slice(&[a.rows(), *k]))),
                _ => Err(invalid(
                    op,
                    format!(
                        "cannot multiply a {} × {} sparse matrix by a tensor of shape {s:?} (it needs [{}] or \
                         [{}, k])",
                        a.rows(),
                        a.cols(),
                        a.cols(),
                        a.cols()
                    ),
                )),
            }
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
        Op::TensorMultiply {
            left_axes,
            right_axes,
        } => {
            f64_parents(2)?;
            let (a, b) = (parents[0].1, parents[1].1);
            if left_axes.len() != right_axes.len() {
                return Err(TensorError::InvalidArgument(
                    "the same number of axes must be given for both tensors".into(),
                )
                .into());
            }
            let left = axes(left_axes, a.len())?;
            let right = axes(right_axes, b.len())?;
            if !distinct(&left) || !distinct(&right) {
                return Err(invalid(op, "an axis is contracted more than once"));
            }
            if left.iter().zip(&right).any(|(&l, &r)| a[l] != b[r]) {
                return Err(incompatible(a, b));
            }
            let shape = (0..a.len())
                .filter(|d| !left.contains(d))
                .map(|d| a[d])
                .chain((0..b.len()).filter(|d| !right.contains(d)).map(|d| b[d]))
                .collect();
            Ok((DType::F64, shape))
        }
        Op::Broadcast(target) => {
            check_elements(op, target)?;
            let source = parents[0].1;
            let shape = broadcast_shape(source, target)?;
            if shape.as_slice() != target.as_slice() {
                return Err(incompatible(source, target));
            }
            Ok((parents[0].0, shape))
        }
        Op::Gather { axis } => {
            let ((dtype, input), (index_type, indices)) = (parents[0], parents[1]);
            if index_type != DType::I64 {
                return Err(invalid(op, "the indices must be i64"));
            }
            let a = axis_of(*axis, input.len())?;
            check_index_range(indices, input[a])?;
            let shape = input[..a]
                .iter()
                .chain(indices)
                .chain(&input[a + 1..])
                .copied()
                .collect();
            Ok((dtype, shape))
        }
        Op::ScatterAdd { axis, size } => {
            let ((dtype, input), (index_type, indices)) = (parents[0], parents[1]);
            if index_type != DType::I64 {
                return Err(invalid(op, "the indices must be i64"));
            }
            if dtype == DType::Bool {
                return Err(invalid(op, "scatter-add applies to numbers"));
            }
            let k = indices.len();
            let out_rank = (input.len() + 1)
                .checked_sub(k)
                .ok_or_else(|| incompatible(input, indices))?;
            let a = axis_of(*axis, out_rank)?;
            if input[a..a + k] != *indices {
                return Err(incompatible(input, indices));
            }
            check_index_range(indices, *size)?;
            let shape = input[..a]
                .iter()
                .chain([size])
                .chain(&input[a + k..])
                .copied()
                .collect();
            Ok((dtype, shape))
        }
        Op::Unary(u) => {
            let (dtype, shape) = parents[0];
            if dtype != DType::F64 && !(dtype == DType::I64 && u.applies_to_integers()) {
                return Err(invalid(op, format!("{u:?} does not apply to {dtype:?}")));
            }
            Ok((dtype, Shape::from_slice(shape)))
        }
        Op::Binary(b) => {
            let ((left, a), (right, c)) = (parents[0], parents[1]);
            let applies = match (left, right) {
                (DType::F64, DType::F64) => *b != BinaryOp::Rem,
                (DType::I64, DType::I64) => matches!(
                    b,
                    BinaryOp::Add
                        | BinaryOp::Sub
                        | BinaryOp::Mul
                        | BinaryOp::Div
                        | BinaryOp::Pow
                        | BinaryOp::Max
                        | BinaryOp::Min
                        | BinaryOp::Rem
                ),
                _ => false,
            };
            if !applies {
                return Err(invalid(
                    op,
                    format!("{b:?} does not apply to {left:?} and {right:?}"),
                ));
            }
            Ok((left, broadcast_shape(a, c)?))
        }
        Op::Compare(c) => {
            let ((left, a), (right, b)) = (parents[0], parents[1]);
            let applies = left == right
                && (left != DType::Bool || matches!(c, CompareOp::Eq | CompareOp::Ne));
            if !applies {
                return Err(invalid(op, "operands must have the same numeric type"));
            }
            Ok((DType::Bool, broadcast_shape(a, b)?))
        }
        Op::ApproxEq(_) => {
            let ((left, a), (right, b)) = (parents[0], parents[1]);
            if left != right || left == DType::Bool {
                return Err(invalid(op, "operands must have the same numeric type"));
            }
            Ok((DType::Bool, broadcast_shape(a, b)?))
        }
        Op::Mask(_) => {
            let a = expect(parents[0], DType::F64)?;
            let b = expect(parents[1], DType::F64)?;
            Ok((DType::F64, broadcast_shape(a, b)?))
        }
        Op::Logic(_) => {
            let a = expect(parents[0], DType::Bool)?;
            let b = expect(parents[1], DType::Bool)?;
            Ok((DType::Bool, broadcast_shape(a, b)?))
        }
        Op::Not | Op::Assert(_) => Ok((DType::Bool, expect(parents[0], DType::Bool)?.into())),
        Op::All | Op::Any => {
            expect(parents[0], DType::Bool)?;
            Ok((DType::Bool, Shape::new()))
        }
        Op::Predicate(_) => Ok((DType::Bool, expect(parents[0], DType::F64)?.into())),
        Op::LogDensity(kind) => log_density::output_type(kind, parents),
        Op::Factor => {
            expect(parents[0], DType::F64)?;
            Ok((DType::F64, Shape::new()))
        }
        Op::Standardize | Op::ReplaceNan(_) => {
            Ok((DType::F64, expect(parents[0], DType::F64)?.into()))
        }
        Op::Softmax { axis } | Op::LogSoftmax { axis } => {
            let shape = expect(parents[0], DType::F64)?;
            axis_of(*axis, shape.len())?;
            Ok((DType::F64, shape.into()))
        }
        Op::Cumulative { axis, .. } => {
            let (dtype, shape) = parents[0];
            if dtype == DType::Bool {
                return Err(invalid(op, "cumulative sums and products apply to numbers"));
            }
            axis_of(*axis, shape.len())?;
            Ok((dtype, shape.into()))
        }
        Op::Reduce { kind, axes: a } => {
            let (dtype, shape) = parents[0];
            let applies = matches!(
                (dtype, kind, a),
                (
                    DType::F64,
                    ReduceOp::Sum | ReduceOp::Product | ReduceOp::Mean | ReduceOp::LogSumExp,
                    _
                ) | (
                    DType::F64,
                    ReduceOp::Max | ReduceOp::Min | ReduceOp::StdDev,
                    None
                ) | (DType::I64, ReduceOp::Sum | ReduceOp::Product, _)
                    | (DType::I64, ReduceOp::Max | ReduceOp::Min, None)
            );
            if !applies {
                return Err(invalid(
                    op,
                    format!("{kind:?} over {a:?} does not apply to {dtype:?}"),
                ));
            }
            match a {
                None => {
                    if matches!(kind, ReduceOp::Max | ReduceOp::Min) && shape.contains(&0) {
                        return Err(invalid(op, "cannot reduce an empty tensor"));
                    }
                    Ok((dtype, Shape::new()))
                }
                Some(a) => {
                    let reduced = axes(a, shape.len())?;
                    let shape = (0..shape.len())
                        .filter(|d| !reduced.contains(d))
                        .map(|d| shape[d])
                        .collect();
                    Ok((dtype, shape))
                }
            }
        }
        Op::ArgReduce { kind, axis } => {
            let (dtype, shape) = parents[0];
            let applies = matches!(
                (dtype, kind, axis),
                (DType::F64, _, None) | (DType::F64 | DType::I64, ArgOp::ArgMax | ArgOp::ArgMin, _)
            );
            if !applies {
                return Err(invalid(
                    op,
                    format!("{kind:?} over {axis:?} does not apply to {dtype:?}"),
                ));
            }
            match axis {
                None => {
                    if shape.contains(&0) {
                        return Err(invalid(
                            op,
                            "cannot take the arg max or min of an empty tensor",
                        ));
                    }
                    Ok((DType::I64, Shape::new()))
                }
                Some(axis) => {
                    let a = axis_of(*axis, shape.len())?;
                    if shape[a] == 0 {
                        return Err(TensorError::InvalidArgument(format!(
                            "cannot take the argmax or argmin along empty axis {a} of shape {shape:?}"
                        ))
                        .into());
                    }
                    let shape = (0..shape.len())
                        .filter(|&d| d != a)
                        .map(|d| shape[d])
                        .collect();
                    Ok((DType::I64, shape))
                }
            }
        }
        Op::Reshape(to) => Ok((parents[0].0, reshape_shape(parents[0].1, to)?)),
        Op::Permute(permutation) => {
            let (dtype, shape) = parents[0];
            let rank = shape.len();
            let valid = permutation.len() == rank
                && permutation.iter().all(|&p| p < rank)
                && distinct(permutation);
            if !valid {
                return Err(TensorError::InvalidPermutation {
                    permutation: permutation.clone(),
                    rank,
                }
                .into());
            }
            Ok((dtype, permutation.iter().map(|&p| shape[p]).collect()))
        }
        Op::Transpose => {
            let (dtype, shape) = parents[0];
            let rank = shape.len();
            if rank < 2 {
                return Err(TensorError::InvalidArgument(
                    "transpose needs a tensor of rank >= 2".into(),
                )
                .into());
            }
            let mut shape = Shape::from_slice(shape);
            shape.swap(rank - 1, rank - 2);
            Ok((dtype, shape))
        }
        Op::Slice(spec) => Ok((parents[0].0, slice_shape(spec, parents[0].1)?)),
        Op::Concat(axis) => {
            let (dtype, first) = parents[0];
            same_dtype(op, parents)?;
            let a = axis_of(*axis, first.len())?;
            let mut length = 0usize;
            for &(_, shape) in parents {
                let compatible = shape.len() == first.len()
                    && (0..shape.len()).all(|d| d == a || shape[d] == first[d]);
                if !compatible {
                    return Err(incompatible(first, shape));
                }
                length = length
                    .checked_add(shape[a])
                    .ok_or_else(|| incompatible(first, shape))?;
            }
            let mut shape = Shape::from_slice(first);
            shape[a] = length;
            Ok((dtype, shape))
        }
        Op::Stack(axis) => {
            let (dtype, first) = parents[0];
            same_dtype(op, parents)?;
            let a = axis_of(*axis, first.len() + 1)?;
            if let Some(&(_, shape)) = parents.iter().find(|(_, s)| *s != first) {
                return Err(incompatible(first, shape));
            }
            let mut shape = Shape::from_slice(first);
            shape.insert(a, parents.len());
            Ok((dtype, shape))
        }
        Op::Take(index) => {
            let (dtype, shape) = parents[0];
            if index.len() != shape.len() || index.iter().zip(shape).any(|(&i, &d)| i >= d) {
                return Err(invalid(
                    op,
                    format!("index {index:?} is out of range for shape {shape:?}"),
                ));
            }
            Ok((dtype, Shape::new()))
        }
        Op::Diag => {
            let (dtype, shape) = parents[0];
            let Some(&n) = shape.last() else {
                return Err(TensorError::InvalidArgument(
                    "diag needs a tensor of rank >= 1".into(),
                )
                .into());
            };
            let mut shape = Shape::from_slice(shape);
            shape.push(n);
            Ok((dtype, shape))
        }
        Op::DiagPart => {
            let (dtype, shape) = parents[0];
            let (rows, cols) = matrix_dims("diag_part", shape)?;
            let mut out = Shape::from_slice(&shape[..shape.len() - 2]);
            out.push(rows.min(cols));
            Ok((dtype, out))
        }
        Op::Tril(_) | Op::Triu(_) => {
            let (dtype, shape) = parents[0];
            let name = if matches!(op, Op::Tril(_)) {
                "tril"
            } else {
                "triu"
            };
            matrix_dims(name, shape)?;
            Ok((dtype, shape.into()))
        }
        Op::FillTriangular { .. } => {
            let (dtype, shape) = parents[0];
            let Some(&m) = shape.last() else {
                return Err(TensorError::InvalidArgument(
                    "fill_triangular needs a tensor of rank >= 1".into(),
                )
                .into());
            };
            // as Tensor::fill_triangular; m is at most MAX_ELEMENTS, so 8m + 1 fits
            let n = (((8 * m + 1) as f64).sqrt() as usize).saturating_sub(1) / 2;
            if n * (n + 1) / 2 != m {
                return Err(TensorError::InvalidArgument(format!(
                    "{m} is not a triangular number"
                ))
                .into());
            }
            let mut out = Shape::from_slice(&shape[..shape.len() - 1]);
            out.extend([n, n]);
            Ok((dtype, out))
        }
        Op::TrianglePart { .. } => {
            let (dtype, shape) = parents[0];
            let (rows, cols) = matrix_dims("triangle_part", shape)?;
            if rows != cols {
                return Err(TensorError::InvalidArgument(
                    "triangle_part needs square matrices".into(),
                )
                .into());
            }
            // n² is at most MAX_ELEMENTS, so n(n + 1) fits
            let mut out = Shape::from_slice(&shape[..shape.len() - 2]);
            out.push(rows * (rows + 1) / 2);
            Ok((dtype, out))
        }
        Op::Cast(to) => Ok((*to, parents[0].1.into())),
        Op::Where => {
            let predicate = expect(parents[0], DType::Bool)?;
            let ((then_type, then), (else_type, otherwise)) = (parents[1], parents[2]);
            if then_type != else_type {
                return Err(invalid(op, "then and else must have the same type"));
            }
            let shape = broadcast_shape(&broadcast_shape(predicate, then)?, otherwise)?;
            Ok((then_type, shape))
        }
        Op::SetWithMask => {
            let operand = expect(parents[0], DType::F64)?;
            let mask = expect(parents[1], DType::F64)?;
            let value = expect(parents[2], DType::F64)?;
            if element_count(value) != Some(1) {
                return Err(invalid(op, "the value must be a scalar"));
            }
            if operand != mask {
                return Err(invalid(
                    op,
                    format!("operand {operand:?} and mask {mask:?} must have the same shape"),
                ));
            }
            Ok((DType::F64, operand.into()))
        }
    }
}

/// The parent's shape if it has type `want`, else the error evaluation gives for a value of the wrong type.
fn expect((dtype, shape): (DType, &[usize]), want: DType) -> Result<&[usize]> {
    if dtype != want {
        let want = match want {
            DType::F64 => "f64",
            DType::I64 => "i64",
            DType::Bool => "bool",
        };
        return Err(GraphError::InvalidValue(format!(
            "expected {want}, got {dtype:?}"
        )));
    }
    Ok(shape)
}

fn same_dtype(op: &Op, parents: &[(DType, &[usize])]) -> Result<()> {
    if parents.iter().any(|(t, _)| *t != parents[0].0) {
        return Err(invalid(op, "parents must have the same type"));
    }
    Ok(())
}

fn incompatible(left: &[usize], right: &[usize]) -> GraphError {
    TensorError::IncompatibleShapes {
        left: left.to_vec(),
        right: right.to_vec(),
    }
    .into()
}

/// A possibly negative axis (counting from the end) resolved against a rank.
fn axis_of(axis: isize, rank: usize) -> Result<usize> {
    let resolved = if axis < 0 {
        axis.checked_add_unsigned(rank)
    } else {
        Some(axis)
    };
    match resolved {
        Some(a) if a >= 0 && (a as usize) < rank => Ok(a as usize),
        _ => Err(TensorError::InvalidAxis { axis, rank }.into()),
    }
}

/// Possibly negative axes resolved against a rank.
fn axes(axes: &[isize], rank: usize) -> Result<Vec<usize>> {
    axes.iter().map(|&a| axis_of(a, rank)).collect()
}

/// Whether no value repeats.
fn distinct(values: &[usize]) -> bool {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted.windows(2).all(|w| w[0] != w[1])
}

/// The last two dimensions of a tensor of rank >= 2.
fn matrix_dims(name: &str, shape: &[usize]) -> Result<(usize, usize)> {
    match shape {
        [.., rows, cols] => Ok((*rows, *cols)),
        _ => {
            Err(TensorError::InvalidArgument(format!("{name} needs a tensor of rank >= 2")).into())
        }
    }
}

/// An error unless indices of this shape can be in range for an axis of length `n`: when the axis is empty every
/// index is out of range, so only an empty set of indices is. The index values are checked by evaluation.
fn check_index_range(indices: &[usize], n: usize) -> Result<()> {
    if n == 0 && !indices.contains(&0) {
        return Err(TensorError::InvalidArgument(
            "index 0 is out of range for an axis of length 0".into(),
        )
        .into());
    }
    Ok(())
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
        | Op::MatrixExp
        | Op::SparseMatMul(_)
        | Op::Cast(_)
        | Op::ReplaceNan(_)
        | Op::Assert(_)
        | Op::Factor => (1, Some(1)),
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
        Op::Ode(body) => {
            let n = body.parent_count();
            (n, Some(n))
        }
        // a kind without a fixed parameter list isn't element-wise, which inference reports
        Op::LogDensity(kind) => match log_density::arity(kind) {
            Some(n) => (n, Some(n)),
            None => (1, None),
        },
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

#[cfg(test)]
mod tests {
    use super::*;
    use keanu_tensor::IntTensor;

    const SHAPES: &[&[usize]] = &[
        &[],
        &[0],
        &[1],
        &[3],
        &[6],
        &[1, 3],
        &[3, 1],
        &[2, 3],
        &[3, 3],
        &[2, 0],
        &[0, 0],
        &[2, 3, 3],
        &[4, 1, 3],
    ];
    const SMALL: &[&[usize]] = &[&[], &[0], &[3], &[1, 3], &[2, 3]];
    const DTYPES: [DType; 3] = [DType::F64, DType::I64, DType::Bool];

    /// Evaluation of the op on values of the parents' types and shapes: ones, except zero indices (in range wherever
    /// any index could be).
    fn evaluated(op: &Op, parents: &[(DType, &[usize])]) -> Result<(DType, Shape)> {
        let values: Vec<Value> = parents
            .iter()
            .enumerate()
            .map(|(i, &(t, s))| match (op, i, t) {
                (Op::Gather { .. } | Op::ScatterAdd { .. }, 1, DType::I64) => {
                    Value::I64(IntTensor::full(s, 0))
                }
                _ => Value::ones(t, s),
            })
            .collect();
        let refs: Vec<&Value> = values.iter().collect();
        let out = eval(op, &refs)?;
        Ok((out.dtype(), Shape::from_slice(out.shape())))
    }

    fn check(op: &Op, parents: &[(DType, &[usize])]) {
        let inferred = output_type(op, parents);
        let evaluated = evaluated(op, parents);
        match (&inferred, &evaluated) {
            (Ok(a), Ok(b)) => assert_eq!(a, b, "{op:?} on {parents:?}"),
            (Err(_), Err(_)) => {}
            // stricter than evaluation, which skips its checks of the axes when either operand is empty
            (Err(_), Ok(_)) if matches!(op, Op::TensorMultiply { left_axes, .. } if left_axes == &[0, 0]) =>
                {}
            _ => panic!("{op:?} on {parents:?}: inferred {inferred:?}, evaluated {evaluated:?}"),
        }
    }

    fn typed(shapes: &[&'static [usize]]) -> Vec<(DType, &'static [usize])> {
        DTYPES
            .iter()
            .flat_map(|&t| shapes.iter().map(move |&s| (t, s)))
            .collect()
    }

    fn unary_ops() -> Vec<Op> {
        let mut ops = vec![
            Op::Unary(UnaryOp::Exp),
            Op::Unary(UnaryOp::Abs),
            Op::Unary(UnaryOp::Sign),
            Op::Not,
            Op::All,
            Op::Any,
            Op::Predicate(Predicate::IsNan),
            Op::Standardize,
            Op::Transpose,
            Op::Diag,
            Op::DiagPart,
            Op::Tril(0),
            Op::Triu(-1),
            Op::FillTriangular {
                upper: true,
                lower: false,
            },
            Op::TrianglePart { upper: false },
            Op::ReplaceNan(0.0),
            Op::Assert(None),
        ];
        for dtype in DTYPES {
            ops.push(Op::Cast(dtype));
        }
        for axis in [-4, -1, 0, 1, 2, 3] {
            ops.extend([
                Op::Softmax { axis },
                Op::LogSoftmax { axis },
                Op::Cumulative {
                    kind: CumulativeOp::CumSum,
                    axis,
                },
                Op::Cumulative {
                    kind: CumulativeOp::CumProd,
                    axis,
                },
                Op::Concat(axis),
                Op::Stack(axis),
            ]);
            for kind in [ArgOp::ArgMax, ArgOp::NanArgMin] {
                ops.push(Op::ArgReduce {
                    kind,
                    axis: Some(axis),
                });
            }
        }
        for kind in [
            ReduceOp::Sum,
            ReduceOp::Product,
            ReduceOp::Mean,
            ReduceOp::Max,
            ReduceOp::Min,
            ReduceOp::StdDev,
            ReduceOp::LogSumExp,
        ] {
            for axes in [
                None,
                Some(vec![]),
                Some(vec![0]),
                Some(vec![-1, 0]),
                Some(vec![1, 1]),
                Some(vec![3]),
            ] {
                ops.push(Op::Reduce { kind, axes });
            }
        }
        for kind in [
            ArgOp::ArgMax,
            ArgOp::ArgMin,
            ArgOp::NanArgMax,
            ArgOp::NanArgMin,
        ] {
            ops.push(Op::ArgReduce { kind, axis: None });
        }
        for shape in [
            vec![-1],
            vec![3, -1],
            vec![2, 3],
            vec![0, -1],
            vec![-1, -1],
            vec![-2],
            vec![],
        ] {
            ops.push(Op::Reshape(shape));
        }
        for permutation in [
            vec![],
            vec![0],
            vec![1, 0],
            vec![0, 0],
            vec![2, 0, 1],
            vec![3, 0, 1],
        ] {
            ops.push(Op::Permute(permutation));
        }
        for spec in [
            "",
            "1",
            "-1",
            "::2",
            "1:, ::-1",
            "..., 0",
            "0, 0, 0, 0",
            ":0",
            "5:",
        ] {
            ops.push(Op::Slice(spec.into()));
        }
        for index in [vec![], vec![0], vec![2], vec![1, 2], vec![0, 0, 0]] {
            ops.push(Op::Take(index));
        }
        ops.push(Op::Broadcast(vec![2, 3]));
        ops.push(Op::Broadcast(vec![4, 0, 3]));
        ops
    }

    #[test]
    fn unary_rules_agree_with_evaluation() {
        for op in unary_ops() {
            for &(t, s) in &typed(SHAPES) {
                check(&op, &[(t, s)]);
            }
        }
    }

    #[test]
    fn binary_rules_agree_with_evaluation() {
        let mut ops = vec![
            Op::Binary(BinaryOp::Add),
            Op::Binary(BinaryOp::Rem),
            Op::Binary(BinaryOp::Atan2),
            Op::Compare(CompareOp::Gt),
            Op::Compare(CompareOp::Eq),
            Op::Mask(CompareOp::Le),
            Op::ApproxEq(0.1),
            Op::Logic(LogicOp::Xor),
            Op::MatMul {
                transpose_left: false,
                transpose_right: true,
            },
            Op::Concat(0),
            Op::Concat(-1),
            Op::Stack(1),
        ];
        for (left_axes, right_axes) in [
            (vec![], vec![]),
            (vec![1], vec![0]),
            (vec![-1], vec![-1]),
            (vec![0, 1], vec![0, 1]),
            (vec![0, 0], vec![0, 1]),
            (vec![0], vec![]),
            (vec![2], vec![0]),
        ] {
            ops.push(Op::TensorMultiply {
                left_axes,
                right_axes,
            });
        }
        for axis in [-1, 0, 1, 2] {
            ops.push(Op::Gather { axis });
            for size in [0, 2] {
                ops.push(Op::ScatterAdd { axis, size });
            }
        }
        let parents = typed(SHAPES);
        for op in &ops {
            for &a in &parents {
                for &b in &parents {
                    check(op, &[a, b]);
                }
            }
        }
    }

    #[test]
    fn ternary_rules_agree_with_evaluation() {
        let parents = typed(SMALL);
        for op in [Op::Where, Op::SetWithMask, Op::Concat(0), Op::Stack(-1)] {
            for &a in &parents {
                for &b in &parents {
                    for &c in &parents {
                        check(&op, &[a, b, c]);
                    }
                }
            }
        }
    }
}
