//! Forward and reverse mode automatic differentiation over the graph.
//!
//! Partial derivatives are returned shaped `[of shape..., wrt shape...]`: element `[i..., j...]` is
//! ∂of[i...]/∂wrt[j...].
//!
//! Internally every tangent (forward mode) and cotangent (reverse mode) carries a leading batch axis: forward
//! tangents are `[wrt length, node shape...]` and reverse cotangents `[of length, node shape...]`. This is the Java
//! implementation's layout (`[wrt..., of...]` forward, `[of..., wrt...]` reverse) with the outer shape flattened.
//!
//! Traversal follows the Java `Differentiator`:
//! - forward mode starts at the `wrt` node (identity, or nothing if it is observed) and visits descendants that
//!   are non-probabilistic and differentiable;
//! - reverse mode starts at the `of` node and visits differentiable ancestors in decreasing id order, recording and
//!   stopping at `wrt` nodes, and stopping at probabilistic nodes;
//! - floor, ceil, round, sign, comparisons, masks, casts, arg reductions, multiplexers, `set_with_mask` and
//!   `replace_nan` are not differentiable, so gradients stop there.
//!
//! Every value used must already be in the state: evaluate the graph first.

use crate::error::{GraphError, Result};
use crate::graph::{Graph, Node, NodeId};
use crate::op::{BinaryOp, Op, ReduceOp, ScanOp, UnaryOp, eval};
use crate::state::State;
use crate::value::{DType, Value};
use keanu_tensor::{DoubleTensor, special};
use std::collections::HashMap;

/// Partial derivatives keyed by node: each shaped `[of shape..., wrt shape...]`.
#[derive(Debug, Clone)]
pub struct Partials {
    partials: HashMap<NodeId, DoubleTensor>,
    /// the fixed end of the derivative: the `wrt` node in forward mode, the `of` node in reverse mode
    fixed: NodeId,
    forward: bool,
}

impl Partials {
    /// The partial for a node: in forward mode ∂node/∂wrt, in reverse mode ∂of/∂node.
    pub fn get(&self, node: NodeId) -> Result<&DoubleTensor> {
        self.partials.get(&node).ok_or(if self.forward {
            GraphError::NoPartial {
                of: node.0,
                wrt: self.fixed.0,
            }
        } else {
            GraphError::NoPartial {
                of: self.fixed.0,
                wrt: node.0,
            }
        })
    }

    /// The nodes that have a partial.
    pub fn nodes(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.partials.keys().copied()
    }

    /// Every partial, keyed by node.
    pub fn into_map(self) -> HashMap<NodeId, DoubleTensor> {
        self.partials
    }
}

/// True if gradients can flow through this node (to its parents, or from it as a leaf).
pub fn is_differentiable(graph: &Graph, id: NodeId) -> bool {
    let node = graph.node(id);
    node.dtype == DType::F64
        && match &node.op {
            Op::Constant(_) | Op::Distribution(_) | Op::Proxy => true,
            Op::Unary(u) => !matches!(
                u,
                UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Round | UnaryOp::Sign
            ),
            Op::Binary(b) => !matches!(b, BinaryOp::Rem),
            Op::Reduce { .. } | Op::Standardize | Op::Scan { .. } => true,
            Op::Reshape(_)
            | Op::Permute(_)
            | Op::Transpose
            | Op::Broadcast(_)
            | Op::Slice(_)
            | Op::Concat(_)
            | Op::Take(_) => true,
            Op::Diag
            | Op::DiagPart
            | Op::Tril(_)
            | Op::Triu(_)
            | Op::FillTriangular { .. }
            | Op::TrianglePart { .. } => true,
            Op::MatMul { .. }
            | Op::TensorMultiply { .. }
            | Op::Inverse
            | Op::Determinant
            | Op::Cholesky
            | Op::CholeskyInverse => true,
            Op::Where | Op::Cpt { .. } => true,
            Op::Lambda(l) => l.jacobian.is_some(),
            _ => false,
        }
}

fn value(state: &State, id: NodeId) -> Result<&Value> {
    state.value(id).ok_or(GraphError::MissingValue(id.0))
}

/// Forward mode: ∂of/∂wrt for each node in `of`, shaped `[of shape..., wrt shape...]`.
pub fn forward_partials(
    graph: &Graph,
    state: &State,
    wrt: NodeId,
    of: &[NodeId],
) -> Result<Partials> {
    let wrt_node = graph.node(wrt);
    if wrt_node.dtype != DType::F64 {
        return Err(GraphError::NotDifferentiable(format!(
            "cannot differentiate with respect to {:?} node #{}",
            wrt_node.dtype, wrt.0
        )));
    }
    let wrt_shape = wrt_node.shape.to_vec();
    let n = wrt_shape.iter().product::<usize>();
    let mut tangents: Vec<Option<DoubleTensor>> = vec![None; graph.len()];
    let mut result = Partials {
        partials: HashMap::new(),
        fixed: wrt,
        forward: true,
    };
    if state.is_observed(wrt) || !is_differentiable(graph, wrt) {
        return Ok(result);
    }
    tangents[wrt.0] = Some(identity(n, &wrt_shape)?);

    let last = of.iter().map(|id| id.0).max().unwrap_or(wrt.0);
    for i in wrt.0 + 1..=last.max(wrt.0) {
        let id = NodeId(i);
        let node = graph.node(id);
        if node.is_probabilistic() || !is_differentiable(graph, id) {
            continue;
        }
        let parent_tangents: Vec<Option<&DoubleTensor>> = node
            .parents
            .iter()
            .map(|p| tangents[p.0].as_ref())
            .collect();
        if parent_tangents.iter().all(Option::is_none) {
            continue;
        }
        let parent_values = node
            .parents
            .iter()
            .map(|&p| value(state, p))
            .collect::<Result<Vec<_>>>()?;
        let out = value(state, id)?;
        tangents[i] = jvp(node, &parent_values, out, &parent_tangents, n)?;
    }

    for &o in of {
        if let Some(t) = &tangents[o.0] {
            // [n, of...] -> [wrt..., of...] -> [of..., wrt...]
            let of_shape = graph.node(o).shape.to_vec();
            let full: Vec<isize> = wrt_shape
                .iter()
                .chain(&of_shape)
                .map(|&d| d as isize)
                .collect();
            let t = t.reshape(&full)?;
            let (w, r) = (wrt_shape.len(), of_shape.len());
            let permutation: Vec<usize> = (w..w + r).chain(0..w).collect();
            result.partials.insert(o, t.permute(&permutation)?);
        }
    }
    Ok(result)
}

/// Reverse mode: ∂of/∂wrt for each node in `wrt`, shaped `[of shape..., wrt shape...]`. Empty if `of` is observed.
pub fn reverse_partials(
    graph: &Graph,
    state: &State,
    of: NodeId,
    wrt: &[NodeId],
) -> Result<Partials> {
    let of_shape = graph.node(of).shape.to_vec();
    if state.is_observed(of) {
        return Ok(Partials {
            partials: HashMap::new(),
            fixed: of,
            forward: false,
        });
    }
    let n = of_shape.iter().product::<usize>();
    let upstream = identity(n, &of_shape)?.reshape(
        &of_shape
            .iter()
            .chain(&of_shape)
            .map(|&d| d as isize)
            .collect::<Vec<_>>(),
    )?;
    reverse_partials_from(graph, state, of, &upstream, wrt)
}

/// Reverse mode starting from a given partial of some outer quantity with respect to `of`, shaped
/// `[outer shape..., of shape...]`. Returns, for each node in `wrt`, the partial of the outer quantity shaped
/// `[outer shape..., wrt shape...]`.
pub fn reverse_partials_from(
    graph: &Graph,
    state: &State,
    of: NodeId,
    upstream: &DoubleTensor,
    wrt: &[NodeId],
) -> Result<Partials> {
    let of_shape = graph.node(of).shape.to_vec();
    let up_shape = upstream.shape();
    if up_shape.len() < of_shape.len()
        || up_shape[up_shape.len() - of_shape.len()..] != of_shape[..]
    {
        return Err(GraphError::InvalidValue(format!(
            "upstream partial of shape {up_shape:?} does not end with the shape {of_shape:?} of node #{}",
            of.0
        )));
    }
    let outer: Vec<usize> = up_shape[..up_shape.len() - of_shape.len()].to_vec();
    let b = outer.iter().product::<usize>();
    let mut result = Partials {
        partials: HashMap::new(),
        fixed: of,
        forward: false,
    };

    let mut cotangents: Vec<Option<DoubleTensor>> = vec![None; of.0 + 1];
    cotangents[of.0] = Some(upstream.reshape(&batched(b, &of_shape))?);

    for i in (0..=of.0).rev() {
        let id = NodeId(i);
        let Some(cotangent) = cotangents[i].take() else {
            continue;
        };
        if wrt.contains(&id) {
            let shape: Vec<isize> = outer
                .iter()
                .chain(graph.node(id).shape.iter())
                .map(|&d| d as isize)
                .collect();
            result.partials.insert(id, cotangent.reshape(&shape)?);
            continue;
        }
        let node = graph.node(id);
        if node.is_probabilistic() || !is_differentiable(graph, id) {
            continue;
        }
        let parent_values = node
            .parents
            .iter()
            .map(|&p| value(state, p))
            .collect::<Result<Vec<_>>>()?;
        let out = value(state, id)?;
        let contributions = vjp(node, &parent_values, out, &cotangent, b)?;
        for (&parent, contribution) in node.parents.iter().zip(contributions) {
            let Some(c) = contribution else { continue };
            if !is_differentiable(graph, parent) {
                continue;
            }
            cotangents[parent.0] = Some(match cotangents[parent.0].take() {
                Some(existing) => existing.add(&c)?,
                None => c,
            });
        }
    }
    Ok(result)
}

// ---------------------------------------------------------------------------------------------------------------
// helpers on batched tensors [B, shape...]

fn batched(b: usize, shape: &[usize]) -> Vec<isize> {
    std::iter::once(b)
        .chain(shape.iter().copied())
        .map(|d| d as isize)
        .collect()
}

/// The identity as a batched tangent: [n, shape...] with element [k, j...] = 1 where j is the k-th element.
fn identity(n: usize, shape: &[usize]) -> Result<DoubleTensor> {
    Ok(DoubleTensor::eye(n).reshape(&batched(n, shape))?)
}

/// Reshapes a batched tensor [B, s...] to [B, 1..., s...] with `rank` non-batch dims, so it broadcasts against
/// tensors of that rank.
fn lift(t: &DoubleTensor, rank: usize) -> Result<DoubleTensor> {
    let shape = t.shape();
    let inner = &shape[1..];
    if inner.len() >= rank {
        return Ok(t.clone());
    }
    let mut new_shape = vec![shape[0] as isize];
    new_shape.extend(std::iter::repeat_n(1, rank - inner.len()));
    new_shape.extend(inner.iter().map(|&d| d as isize));
    Ok(t.reshape(&new_shape)?)
}

/// Sums a batched tensor [B, out...] over the dimensions that `target` was broadcast along, giving [B, target...].
fn reduce_to(t: &DoubleTensor, target: &[usize]) -> Result<DoubleTensor> {
    let inner = &t.shape()[1..];
    if inner == target {
        return Ok(t.clone());
    }
    let rank = inner.len();
    let offset = rank - target.len();
    let axes: Vec<isize> = (0..rank)
        .filter(|&i| i < offset || (target[i - offset] == 1 && inner[i] != 1))
        .map(|i| i as isize + 1)
        .collect();
    let summed = if axes.is_empty() {
        t.clone()
    } else {
        t.sum_axes(&axes)?
    };
    Ok(summed.reshape(&batched(t.shape()[0], target))?)
}

/// Multiplies a batched parent tangent by an output-shaped local derivative, broadcasting to [B, out...].
fn scale(t: &DoubleTensor, d: &DoubleTensor) -> Result<DoubleTensor> {
    Ok(lift(t, d.rank())?.mul(d)?)
}

fn add_opt(a: Option<DoubleTensor>, b: Option<DoubleTensor>) -> Result<Option<DoubleTensor>> {
    Ok(match (a, b) {
        (Some(a), Some(b)) => Some(a.add(&b)?),
        (a, None) => a,
        (None, b) => b,
    })
}

fn f64v(v: &Value) -> Result<&DoubleTensor> {
    v.as_f64().ok_or_else(|| {
        GraphError::InvalidValue(format!("expected an f64 value, got {:?}", v.dtype()))
    })
}

/// Applies a Jacobian J [n_out, n_in] in forward mode: [B, in...] -> [B, out...].
fn jacobian_forward(
    t: &DoubleTensor,
    j: &DoubleTensor,
    out_shape: &[usize],
) -> Result<DoubleTensor> {
    let b = t.shape()[0];
    let n_in = j.shape()[1] as isize;
    Ok(t.reshape(&[b as isize, n_in])?
        .matmul_transposed(j, false, true)?
        .reshape(&batched(b, out_shape))?)
}

/// Applies a Jacobian J [n_out, n_in] in reverse mode: [B, out...] -> [B, in...].
fn jacobian_reverse(
    g: &DoubleTensor,
    j: &DoubleTensor,
    in_shape: &[usize],
) -> Result<DoubleTensor> {
    let b = g.shape()[0];
    let n_out = j.shape()[0] as isize;
    Ok(g.reshape(&[b as isize, n_out])?
        .matmul(j)?
        .reshape(&batched(b, in_shape))?)
}

/// The exact Jacobian [n_out, n_in] of an op with respect to parent `p`, for ops that are affine in each element of
/// that parent when everything else is fixed (sums, products, scans, tensor contractions). Column j is
/// op(x with x_j = 1) - op(x with x_j = 0), which is exact and well defined where elements are zero.
fn affine_jacobian(op: &Op, parents: &[&Value], p: usize) -> Result<DoubleTensor> {
    let x = f64v(parents[p])?;
    let n_in = x.len();
    let mut columns = Vec::with_capacity(n_in);
    for j in 0..n_in {
        let mut one = x.as_slice().to_vec();
        let mut zero = one.clone();
        one[j] = 1.0;
        zero[j] = 0.0;
        let eval_with = |data: Vec<f64>| -> Result<DoubleTensor> {
            let replaced = Value::F64(DoubleTensor::new(x.shape(), data)?);
            let inputs: Vec<&Value> = parents
                .iter()
                .enumerate()
                .map(|(i, v)| if i == p { &replaced } else { *v })
                .collect();
            Ok(f64v(&eval(op, &inputs)?)?.clone())
        };
        let column = eval_with(one)?.sub(&eval_with(zero)?)?;
        columns.push(column.into_vec());
    }
    let n_out = columns.first().map_or(0, Vec::len);
    let mut data = vec![0.0; n_out * n_in];
    for (j, column) in columns.iter().enumerate() {
        for (i, &v) in column.iter().enumerate() {
            data[i * n_in + j] = v;
        }
    }
    Ok(DoubleTensor::new(&[n_out, n_in], data)?)
}

/// For ops whose every output element is a copy of one input element or zero (reshape, slice, concat, ...): for each
/// output element, the (parent, flat index) it copies, found by evaluating the op on inputs holding their own
/// (offset) indices.
fn index_map(op: &Op, parents: &[&Value]) -> Result<Vec<Option<(usize, usize)>>> {
    let mut offsets = Vec::with_capacity(parents.len());
    let mut total = 0usize;
    let encoded: Vec<Value> = parents
        .iter()
        .map(|v| {
            let x = f64v(v)?;
            offsets.push(total);
            let data = (0..x.len()).map(|i| (total + i + 1) as f64).collect();
            total += x.len();
            Ok(Value::F64(DoubleTensor::new(x.shape(), data)?))
        })
        .collect::<Result<_>>()?;
    let inputs: Vec<&Value> = encoded.iter().collect();
    let out = eval(op, &inputs)?;
    Ok(f64v(&out)?
        .as_slice()
        .iter()
        .map(|&v| {
            if v == 0.0 {
                return None;
            }
            let global = v as usize - 1;
            let parent = offsets.iter().rposition(|&o| o <= global).unwrap();
            Some((parent, global - offsets[parent]))
        })
        .collect())
}

fn gather_forward(
    map: &[Option<(usize, usize)>],
    tangents: &[Option<&DoubleTensor>],
    b: usize,
    out_shape: &[usize],
) -> Result<Option<DoubleTensor>> {
    if tangents.iter().all(Option::is_none) {
        return Ok(None);
    }
    let n_out = map.len();
    let mut data = vec![0.0; b * n_out];
    for (o, source) in map.iter().enumerate() {
        let Some((p, idx)) = *source else { continue };
        let Some(t) = tangents[p] else { continue };
        let t = t.as_slice();
        let n_p = t.len() / b;
        for k in 0..b {
            data[k * n_out + o] = t[k * n_p + idx];
        }
    }
    Ok(Some(DoubleTensor::new(
        &batched(b, out_shape)
            .iter()
            .map(|&d| d as usize)
            .collect::<Vec<_>>(),
        data,
    )?))
}

fn gather_reverse(
    map: &[Option<(usize, usize)>],
    g: &DoubleTensor,
    parents: &[&Value],
    b: usize,
) -> Result<Vec<Option<DoubleTensor>>> {
    let g = g.as_slice();
    let n_out = map.len();
    let mut results: Vec<Vec<f64>> = parents
        .iter()
        .map(|v| vec![0.0; b * v.shape().iter().product::<usize>()])
        .collect();
    for (o, source) in map.iter().enumerate() {
        let Some((p, idx)) = *source else { continue };
        let n_p = results[p].len() / b;
        for k in 0..b {
            results[p][k * n_p + idx] += g[k * n_out + o];
        }
    }
    parents
        .iter()
        .zip(results)
        .map(|(v, data)| {
            let shape: Vec<usize> = batched(b, v.shape()).iter().map(|&d| d as usize).collect();
            Ok(Some(DoubleTensor::new(&shape, data)?))
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------------------------
// element-wise derivatives

fn unary_derivative(u: UnaryOp, x: &DoubleTensor, y: &DoubleTensor) -> Result<DoubleTensor> {
    use std::f64::consts::{LN_2, LN_10};
    Ok(match u {
        UnaryOp::Abs => x.sign(),
        UnaryOp::Neg => x.map(|_| -1.0),
        UnaryOp::Reciprocal => y.map(|y| -y * y),
        UnaryOp::Exp => y.clone(),
        UnaryOp::Exp2 => y.map(|y| y * LN_2),
        UnaryOp::ExpM1 => y.map(|y| y + 1.0),
        UnaryOp::Ln => x.reciprocal(),
        UnaryOp::Ln1p => x.map(|x| 1.0 / (1.0 + x)),
        UnaryOp::Log2 => x.map(|x| 1.0 / (x * LN_2)),
        UnaryOp::Log10 => x.map(|x| 1.0 / (x * LN_10)),
        UnaryOp::Sqrt => y.map(|y| 0.5 / y),
        UnaryOp::Sin => x.cos(),
        UnaryOp::Cos => x.map(|x| -x.sin()),
        UnaryOp::Tan => x.map(|x| 1.0 / (x.cos() * x.cos())),
        UnaryOp::Asin => x.map(|x| 1.0 / (1.0 - x * x).sqrt()),
        UnaryOp::Acos => x.map(|x| -1.0 / (1.0 - x * x).sqrt()),
        UnaryOp::Atan => x.map(|x| 1.0 / (1.0 + x * x)),
        UnaryOp::Sinh => x.cosh(),
        UnaryOp::Cosh => x.sinh(),
        UnaryOp::Tanh => y.map(|y| 1.0 - y * y),
        UnaryOp::Asinh => x.map(|x| 1.0 / (x * x + 1.0).sqrt()),
        UnaryOp::Acosh => x.map(|x| 1.0 / (x * x - 1.0).sqrt()),
        UnaryOp::Atanh => x.map(|x| 1.0 / (1.0 - x * x)),
        UnaryOp::Sigmoid => y.map(|y| y * (1.0 - y)),
        UnaryOp::LnGamma => x.digamma(),
        UnaryOp::Digamma => x.trigamma(),
        UnaryOp::Trigamma => x.map(special::tetragamma),
        UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Round | UnaryOp::Sign => {
            return Err(GraphError::NotDifferentiable(format!(
                "{u:?} is not differentiable"
            )));
        }
    })
}

/// (∂y/∂left, ∂y/∂right), each shaped like the output.
fn binary_derivatives(
    op: BinaryOp,
    l: &DoubleTensor,
    r: &DoubleTensor,
    y: &DoubleTensor,
) -> Result<(DoubleTensor, DoubleTensor)> {
    let ones = || DoubleTensor::ones(y.shape());
    let full = |t: DoubleTensor| -> Result<DoubleTensor> { Ok(t.mul(&ones())?) };
    Ok(match op {
        BinaryOp::Add => (ones(), ones()),
        BinaryOp::Sub => (ones(), ones().neg()),
        BinaryOp::Mul => (full(r.clone())?, full(l.clone())?),
        BinaryOp::Div => (
            full(r.reciprocal())?,
            l.zip_with(r, |l, r| -l / (r * r))?.mul(&ones())?,
        ),
        BinaryOp::Pow => (
            l.zip_with(r, |l, r| r * l.powf(r - 1.0))?.mul(&ones())?,
            full(l.ln())?.mul(y)?,
        ),
        BinaryOp::Max => {
            let m = l
                .zip_with(r, |l, r| f64::from(u8::from(l >= r)))?
                .mul(&ones())?;
            (m.clone(), m.map(|m| 1.0 - m))
        }
        BinaryOp::Min => {
            let m = l
                .zip_with(r, |l, r| f64::from(u8::from(l <= r)))?
                .mul(&ones())?;
            (m.clone(), m.map(|m| 1.0 - m))
        }
        // left is x and right is y: atan2(y, x)
        BinaryOp::Atan2 => (
            l.zip_with(r, |x, y| -y / (x * x + y * y))?.mul(&ones())?,
            l.zip_with(r, |x, y| x / (x * x + y * y))?.mul(&ones())?,
        ),
        BinaryOp::LogAddExp => (
            full(l.clone())?.sub(y)?.exp(),
            full(r.clone())?.sub(y)?.exp(),
        ),
        BinaryOp::LogAddExp2 => (
            full(l.clone())?.sub(y)?.exp2(),
            full(r.clone())?.sub(y)?.exp2(),
        ),
        // y * ln(x): d/dx = y / x; d/dy = ln(x), NaN where y == 0 (as Java)
        BinaryOp::SafeLnTimes => (
            l.zip_with(r, |x, y| y / x)?.mul(&ones())?,
            l.zip_with(r, |x, y| if y == 0.0 { f64::NAN } else { x.ln() })?
                .mul(&ones())?,
        ),
        BinaryOp::Rem => {
            return Err(GraphError::NotDifferentiable(
                "rem is not differentiable".into(),
            ));
        }
        #[allow(unreachable_patterns)]
        _ => {
            return Err(GraphError::NotDifferentiable(format!(
                "{op:?} is not differentiable"
            )));
        }
    })
}

// ---------------------------------------------------------------------------------------------------------------
// Jacobians of reductions

fn reduce_jacobian(
    kind: ReduceOp,
    axes: &Option<Vec<isize>>,
    op: &Op,
    parents: &[&Value],
) -> Result<Option<DoubleTensor>> {
    let x = f64v(parents[0])?;
    let n = x.len();
    Ok(match kind {
        ReduceOp::Product => Some(affine_jacobian(op, parents, 0)?),
        ReduceOp::Max | ReduceOp::Min => {
            // gradient to the selected element, split equally between ties
            let best = if kind == ReduceOp::Max {
                x.max()
            } else {
                x.min()
            }
            .unwrap_or(f64::NAN);
            let selected = x.map(|v| f64::from(u8::from(v == best)));
            let count: f64 = selected.sum();
            Some((selected / count).reshape(&[1, n as isize])?)
        }
        ReduceOp::StdDev => {
            let s = x.std_dev();
            let mean = x.mean();
            let d = x.map(|v| (v - mean) / ((n as f64 - 1.0) * s));
            Some(d.reshape(&[1, n as isize])?)
        }
        ReduceOp::Sum | ReduceOp::Mean => {
            let _ = axes;
            None
        }
    })
}

fn standardize_jacobian(x: &DoubleTensor) -> Result<DoubleTensor> {
    // z_i = (x_i - m) / s; dz_i/dx_j = (δij - 1/n) / s - (x_i - m)(x_j - m) / ((n - 1) s³)
    let n = x.len();
    let s = x.std_dev();
    let mean = x.mean();
    let c = x.as_slice().iter().map(|&v| v - mean).collect::<Vec<_>>();
    let mut data = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let delta = if i == j { 1.0 } else { 0.0 };
            data[i * n + j] =
                (delta - 1.0 / n as f64) / s - c[i] * c[j] / ((n as f64 - 1.0) * s * s * s);
        }
    }
    Ok(DoubleTensor::new(&[n, n], data)?)
}

/// Axes of the input reduced by a Sum/Mean, normalised, plus the element count per output.
fn reduced_axes(
    axes: &Option<Vec<isize>>,
    rank: usize,
    shape: &[usize],
) -> Result<(Vec<usize>, usize)> {
    let axes: Vec<usize> = match axes {
        None => (0..rank).collect(),
        Some(a) => a
            .iter()
            .map(|&ax| {
                let v = if ax < 0 { ax + rank as isize } else { ax };
                if v < 0 || v as usize >= rank {
                    Err(GraphError::InvalidValue(format!("axis {ax} out of range")))
                } else {
                    Ok(v as usize)
                }
            })
            .collect::<Result<_>>()?,
    };
    let count = axes.iter().map(|&a| shape[a]).product();
    Ok((axes, count))
}

// ---------------------------------------------------------------------------------------------------------------
// linear algebra helpers (all on the last two dimensions, broadcasting leading ones)

fn tr(t: &DoubleTensor) -> Result<DoubleTensor> {
    Ok(t.transpose()?)
}

/// phi(X): the lower triangle with the diagonal halved
fn phi(x: &DoubleTensor) -> Result<DoubleTensor> {
    Ok(x.tril(0)?.add(&x.tril(-1)?)? * 0.5)
}

fn symmetric_from_lower(x: &DoubleTensor) -> Result<DoubleTensor> {
    Ok(x.tril(0)?.add(&tr(&x.tril(-1)?)?)?)
}

// ---------------------------------------------------------------------------------------------------------------
// forward mode rules

fn jvp(
    node: &Node,
    parents: &[&Value],
    out: &Value,
    t: &[Option<&DoubleTensor>],
    b: usize,
) -> Result<Option<DoubleTensor>> {
    let out_shape = node.shape.to_vec();
    let y = f64v(out)?;
    let op = &node.op;
    Ok(match op {
        Op::Proxy => t[0].cloned(),
        Op::Unary(u) => {
            let Some(t0) = t[0] else { return Ok(None) };
            Some(scale(t0, &unary_derivative(*u, f64v(parents[0])?, y)?)?)
        }
        Op::Binary(bop) => {
            let (dl, dr) = binary_derivatives(*bop, f64v(parents[0])?, f64v(parents[1])?, y)?;
            let from_left = t[0].map(|tl| scale(tl, &dl)).transpose()?;
            let from_right = t[1].map(|tr| scale(tr, &dr)).transpose()?;
            add_opt(from_left, from_right)?
        }
        Op::Reduce { kind, axes } => {
            let Some(t0) = t[0] else { return Ok(None) };
            match kind {
                ReduceOp::Sum | ReduceOp::Mean => {
                    let x = f64v(parents[0])?;
                    let (axes, count) = reduced_axes(axes, x.rank(), x.shape())?;
                    let shifted: Vec<isize> = axes.iter().map(|&a| a as isize + 1).collect();
                    let summed = if shifted.is_empty() {
                        t0.clone()
                    } else {
                        t0.sum_axes(&shifted)?
                    };
                    let summed = summed.reshape(&batched(b, &out_shape))?;
                    Some(if *kind == ReduceOp::Mean {
                        summed / count as f64
                    } else {
                        summed
                    })
                }
                _ => {
                    let j = reduce_jacobian(*kind, axes, op, parents)?.unwrap();
                    Some(jacobian_forward(t0, &j, &out_shape)?)
                }
            }
        }
        Op::Standardize => {
            let Some(t0) = t[0] else { return Ok(None) };
            Some(jacobian_forward(
                t0,
                &standardize_jacobian(f64v(parents[0])?)?,
                &out_shape,
            )?)
        }
        Op::Scan { .. } => {
            let Some(t0) = t[0] else { return Ok(None) };
            Some(jacobian_forward(
                t0,
                &affine_jacobian(op, parents, 0)?,
                &out_shape,
            )?)
        }
        Op::Reshape(_)
        | Op::Permute(_)
        | Op::Transpose
        | Op::Broadcast(_)
        | Op::Slice(_)
        | Op::Concat(_)
        | Op::Take(_)
        | Op::Diag
        | Op::DiagPart
        | Op::Tril(_)
        | Op::Triu(_)
        | Op::FillTriangular { .. }
        | Op::TrianglePart { .. } => gather_forward(&index_map(op, parents)?, t, b, &out_shape)?,
        Op::TensorMultiply { .. } => {
            let mut total = None;
            for (p, tp) in t.iter().enumerate() {
                let Some(tp) = tp else { continue };
                let contribution =
                    jacobian_forward(tp, &affine_jacobian(op, parents, p)?, &out_shape)?;
                total = add_opt(total, Some(contribution))?;
            }
            total
        }
        Op::MatMul {
            transpose_left,
            transpose_right,
        } => {
            let (a, bm) = (f64v(parents[0])?, f64v(parents[1])?);
            let rank = out_shape.len();
            let from_left = t[0]
                .map(|ta| -> Result<DoubleTensor> {
                    Ok(lift(ta, rank)?.matmul_transposed(bm, *transpose_left, *transpose_right)?)
                })
                .transpose()?;
            let from_right = t[1]
                .map(|tb| -> Result<DoubleTensor> {
                    Ok(a.matmul_transposed(&lift(tb, rank)?, *transpose_left, *transpose_right)?)
                })
                .transpose()?;
            // results may lack broadcast batch dims of the other operand; bring both to [B, out...]
            let fit = |x: DoubleTensor| -> Result<DoubleTensor> {
                Ok(x.broadcast_to(
                    &batched(b, &out_shape)
                        .iter()
                        .map(|&d| d as usize)
                        .collect::<Vec<_>>(),
                )?)
            };
            add_opt(
                from_left.map(fit).transpose()?,
                from_right.map(fit).transpose()?,
            )?
        }
        Op::Inverse => {
            let Some(t0) = t[0] else { return Ok(None) };
            // dX = -X dA X
            Some(y.matmul(&lift(t0, out_shape.len())?)?.matmul(y)?.neg())
        }
        Op::Determinant => {
            let Some(t0) = t[0] else { return Ok(None) };
            // d det = det * tr(A⁻¹ dA) = det * sum_ij (A⁻ᵀ)_ij dA_ij
            let a = f64v(parents[0])?;
            let inv_t = tr(&a.inverse()?)?;
            let rank = a.rank();
            let products = lift(t0, rank)?.mul(&inv_t)?;
            let summed = products.sum_axes(&[rank as isize - 1, rank as isize])?;
            Some(summed.reshape(&batched(b, &out_shape))?.mul(y)?)
        }
        Op::Cholesky => {
            let Some(t0) = t[0] else { return Ok(None) };
            // dL = L phi(L⁻¹ dA L⁻ᵀ), dA the symmetric matrix read from the lower triangle of the tangent
            let l = y;
            let l_inv = l.inverse()?;
            let da = symmetric_from_lower(&lift(t0, out_shape.len())?)?;
            let inner = l_inv.matmul(&da)?.matmul_transposed(&l_inv, false, true)?;
            Some(l.matmul(&phi(&inner)?)?)
        }
        Op::CholeskyInverse => {
            let Some(t0) = t[0] else { return Ok(None) };
            // X = (L Lᵀ)⁻¹ with L = tril(input): dX = -X (dL Lᵀ + L dLᵀ) X
            let l = f64v(parents[0])?.tril(0)?;
            let dl = lift(t0, out_shape.len())?.tril(0)?;
            let dl_lt = dl.matmul_transposed(&l, false, true)?;
            let da = dl_lt.add(&tr(&dl_lt)?)?;
            Some(y.matmul(&da)?.matmul(y)?.neg())
        }
        Op::Where => {
            let mask = parents[0]
                .as_bool()
                .ok_or_else(|| GraphError::InvalidValue("where predicate must be bool".into()))?
                .to_double_mask();
            let m = mask.mul(&DoubleTensor::ones(&out_shape))?;
            let from_then = t[1].map(|tt| scale(tt, &m)).transpose()?;
            let from_else = t[2].map(|te| scale(te, &m.map(|m| 1.0 - m))).transpose()?;
            let fit = |x: DoubleTensor| -> Result<DoubleTensor> {
                Ok(x.broadcast_to(
                    &batched(b, &out_shape)
                        .iter()
                        .map(|&d| d as usize)
                        .collect::<Vec<_>>(),
                )?)
            };
            add_opt(
                from_then.map(fit).transpose()?,
                from_else.map(fit).transpose()?,
            )?
        }
        Op::Cpt { conditions, keys } => {
            // the selected outcome's tangent; zero (rather than none) if only unselected outcomes depend on wrt
            let selected = cpt_selected(*conditions, keys, parents)?;
            match t[selected] {
                Some(ts) => Some(ts.clone()),
                None if t[*conditions..].iter().any(Option::is_some) => Some(DoubleTensor::zeros(
                    &batched(b, &out_shape)
                        .iter()
                        .map(|&d| d as usize)
                        .collect::<Vec<_>>(),
                )),
                None => None,
            }
        }
        Op::Lambda(lambda) => {
            let jacobian = lambda.jacobian.as_ref().ok_or_else(|| {
                GraphError::NotDifferentiable(format!("lambda {} has no jacobian", lambda.name))
            })?;
            let values: Vec<Value> = parents.iter().map(|v| (*v).clone()).collect();
            let jacobians = jacobian(&values).map_err(GraphError::Lambda)?;
            let n_out = out_shape.iter().product::<usize>() as isize;
            let mut total = None;
            for ((tp, j), v) in t.iter().zip(jacobians).zip(parents) {
                let (Some(tp), Some(j)) = (tp, j) else {
                    continue;
                };
                let n_in = v.shape().iter().product::<usize>() as isize;
                let contribution = jacobian_forward(tp, &j.reshape(&[n_out, n_in])?, &out_shape)?;
                total = add_opt(total, Some(contribution))?;
            }
            total
        }
        _ => None,
    })
}

// ---------------------------------------------------------------------------------------------------------------
// reverse mode rules

fn vjp(
    node: &Node,
    parents: &[&Value],
    out: &Value,
    g: &DoubleTensor,
    b: usize,
) -> Result<Vec<Option<DoubleTensor>>> {
    let out_shape = node.shape.to_vec();
    let y = f64v(out)?;
    let op = &node.op;
    let none = || vec![None; parents.len()];
    Ok(match op {
        Op::Proxy => vec![Some(g.clone())],
        Op::Unary(u) => {
            let x = f64v(parents[0])?;
            vec![Some(reduce_to(
                &g.mul(&unary_derivative(*u, x, y)?)?,
                x.shape(),
            )?)]
        }
        Op::Binary(bop) => {
            let (l, r) = (f64v(parents[0])?, f64v(parents[1])?);
            let (dl, dr) = binary_derivatives(*bop, l, r, y)?;
            vec![
                Some(reduce_to(&g.mul(&dl)?, l.shape())?),
                Some(reduce_to(&g.mul(&dr)?, r.shape())?),
            ]
        }
        Op::Reduce { kind, axes } => {
            let x = f64v(parents[0])?;
            match kind {
                ReduceOp::Sum | ReduceOp::Mean => {
                    let (axes, count) = reduced_axes(axes, x.rank(), x.shape())?;
                    let kept: Vec<usize> = x
                        .shape()
                        .iter()
                        .enumerate()
                        .map(|(i, &d)| if axes.contains(&i) { 1 } else { d })
                        .collect();
                    let expanded = g.reshape(&batched(b, &kept))?.broadcast_to(
                        &batched(b, x.shape())
                            .iter()
                            .map(|&d| d as usize)
                            .collect::<Vec<_>>(),
                    )?;
                    vec![Some(if *kind == ReduceOp::Mean {
                        expanded / count as f64
                    } else {
                        expanded
                    })]
                }
                _ => {
                    let j = reduce_jacobian(*kind, axes, op, parents)?.unwrap();
                    vec![Some(jacobian_reverse(g, &j, x.shape())?)]
                }
            }
        }
        Op::Standardize => {
            let x = f64v(parents[0])?;
            vec![Some(jacobian_reverse(
                g,
                &standardize_jacobian(x)?,
                x.shape(),
            )?)]
        }
        Op::Scan {
            kind: ScanOp::CumSum | ScanOp::CumProd,
            ..
        } => {
            let x = f64v(parents[0])?;
            vec![Some(jacobian_reverse(
                g,
                &affine_jacobian(op, parents, 0)?,
                x.shape(),
            )?)]
        }
        Op::Reshape(_)
        | Op::Permute(_)
        | Op::Transpose
        | Op::Broadcast(_)
        | Op::Slice(_)
        | Op::Concat(_)
        | Op::Take(_)
        | Op::Diag
        | Op::DiagPart
        | Op::Tril(_)
        | Op::Triu(_)
        | Op::FillTriangular { .. }
        | Op::TrianglePart { .. } => gather_reverse(&index_map(op, parents)?, g, parents, b)?,
        Op::TensorMultiply { .. } => (0..parents.len())
            .map(|p| {
                Ok(Some(jacobian_reverse(
                    g,
                    &affine_jacobian(op, parents, p)?,
                    parents[p].shape(),
                )?))
            })
            .collect::<Result<_>>()?,
        Op::MatMul {
            transpose_left,
            transpose_right,
        } => {
            let (a, bm) = (f64v(parents[0])?, f64v(parents[1])?);
            let (tl, tr_) = (*transpose_left, *transpose_right);
            // C = op(A) op(B): d op(A) = G op(B)ᵀ, d op(B) = op(A)ᵀ G
            let d_op_a = g.matmul_transposed(bm, false, !tr_)?;
            let d_op_b = a.matmul_transposed(g, !tl, false)?;
            let da = if tl { tr(&d_op_a)? } else { d_op_a };
            let db = if tr_ { tr(&d_op_b)? } else { d_op_b };
            vec![
                Some(reduce_to(&da, a.shape())?),
                Some(reduce_to(&db, bm.shape())?),
            ]
        }
        Op::Inverse => {
            // dA = -Xᵀ G Xᵀ
            let xt = tr(y)?;
            vec![Some(xt.matmul(g)?.matmul(&xt)?.neg())]
        }
        Op::Determinant => {
            let a = f64v(parents[0])?;
            let inv_t = tr(&a.inverse()?)?;
            // G [B, batch...] -> [B, batch..., 1, 1] * det * A⁻ᵀ
            let mut shape = batched(b, &out_shape);
            shape.extend([1, 1]);
            let det = y.reshape(
                &out_shape
                    .iter()
                    .map(|&d| d as isize)
                    .chain([1, 1])
                    .collect::<Vec<_>>(),
            )?;
            vec![Some(g.reshape(&shape)?.mul(&det)?.mul(&inv_t)?)]
        }
        Op::Cholesky => {
            // S = L⁻ᵀ phi(Lᵀ Lbar) L⁻¹; dA (read from the lower triangle) = tril(S + Sᵀ) - diag(S)
            let l = y;
            let l_inv = l.inverse()?;
            let l_bar = g.tril(0)?;
            let inner = phi(&l.matmul_transposed(&l_bar, true, false)?)?;
            let s = l_inv
                .matmul_transposed(&inner, true, false)?
                .matmul(&l_inv)?;
            let sum = s.add(&tr(&s)?)?;
            let diag = s.tril(0)?.sub(&s.tril(-1)?)?;
            vec![Some(sum.tril(0)?.sub(&diag)?)]
        }
        Op::CholeskyInverse => {
            // <G, dX> with dX = -X (dL Lᵀ + L dLᵀ) X: M = -X G X, Lbar = (M + Mᵀ) L, read from tril(input)
            let l = f64v(parents[0])?.tril(0)?;
            let m = y.matmul(g)?.matmul(y)?.neg();
            let l_bar = m.add(&tr(&m)?)?.matmul(&l)?;
            vec![Some(l_bar.tril(0)?)]
        }
        Op::Where => {
            let mask = parents[0]
                .as_bool()
                .ok_or_else(|| GraphError::InvalidValue("where predicate must be bool".into()))?
                .to_double_mask();
            let m = mask.mul(&DoubleTensor::ones(&out_shape))?;
            let then_shape = parents[1].shape();
            let else_shape = parents[2].shape();
            vec![
                None,
                Some(reduce_to(&g.mul(&m)?, then_shape)?),
                Some(reduce_to(&g.mul(&m.map(|m| 1.0 - m))?, else_shape)?),
            ]
        }
        Op::Cpt { conditions, keys } => {
            // the selected outcome gets the cotangent, the other outcomes zero (as forward mode)
            let selected = cpt_selected(*conditions, keys, parents)?;
            let mut out = none();
            for (i, slot) in out.iter_mut().enumerate().skip(*conditions) {
                *slot = Some(if i == selected {
                    g.clone()
                } else {
                    g.map(|_| 0.0)
                });
            }
            out
        }
        Op::Lambda(lambda) => {
            let jacobian = lambda.jacobian.as_ref().ok_or_else(|| {
                GraphError::NotDifferentiable(format!("lambda {} has no jacobian", lambda.name))
            })?;
            let values: Vec<Value> = parents.iter().map(|v| (*v).clone()).collect();
            let jacobians = jacobian(&values).map_err(GraphError::Lambda)?;
            let n_out = out_shape.iter().product::<usize>() as isize;
            jacobians
                .into_iter()
                .zip(parents)
                .map(|(j, v)| {
                    j.map(|j| -> Result<DoubleTensor> {
                        let n_in = v.shape().iter().product::<usize>() as isize;
                        jacobian_reverse(g, &j.reshape(&[n_out, n_in])?, v.shape())
                    })
                    .transpose()
                })
                .collect::<Result<_>>()?
        }
        _ => none(),
    })
}

/// The parent index of the outcome a CPT node selects (the same choice as its evaluation).
fn cpt_selected(conditions: usize, keys: &[Vec<i64>], parents: &[&Value]) -> Result<usize> {
    let condition: Vec<i64> = parents[..conditions]
        .iter()
        .map(|v| match v {
            Value::Bool(t) => t.to_scalar().map(i64::from),
            Value::I64(t) => t.to_scalar(),
            Value::F64(_) => None,
        })
        .collect::<Option<_>>()
        .ok_or_else(|| {
            GraphError::InvalidValue("CPT conditions must be scalar booleans or integers".into())
        })?;
    Ok(match keys.iter().position(|k| *k == condition) {
        Some(i) => conditions + i,
        None => parents.len() - 1,
    })
}
