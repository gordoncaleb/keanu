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
//! - floor, ceil, round, sign, comparisons, masks, casts and arg reductions are not differentiable, so gradients
//!   stop there. Multiplexers pass the gradient of the selected option, `set_with_mask` the operand's gradient
//!   where the mask isn't 1 and the set value's where it is, and `replace_nan` the input's gradient where it wasn't
//!   NaN (zero where it was replaced).
//!
//! Every value used must already be in the state: evaluate the graph first.

use crate::error::{GraphError, Result};
use crate::graph::{Graph, Node, NodeId};
use crate::op::eval::{cpt_selected, multiplexer_selected};
use crate::op::{BinaryOp, Op, ReduceOp, ScanOp, UnaryOp, eval};
use crate::state::State;
use crate::value::{DType, Value};
use keanu_tensor::{DoubleTensor, special};
use smallvec::SmallVec;
use std::borrow::Cow;
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
                of: node,
                wrt: self.fixed,
            }
        } else {
            GraphError::NoPartial {
                of: self.fixed,
                wrt: node,
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

/// True if gradients can flow through this node (to its parents, or from it as a leaf). False if the id is not a
/// node of the graph.
pub fn is_differentiable(graph: &Graph, id: NodeId) -> bool {
    let Some(node) = graph.node(id) else {
        return false;
    };
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
            Op::Where | Op::Cpt { .. } | Op::Multiplexer => true,
            Op::SetWithMask | Op::ReplaceNan(_) => true,
            Op::Lambda(l) => l.jacobian.is_some(),
            _ => false,
        }
}

fn value(state: &State, id: NodeId) -> Result<&Value> {
    state.value(id).ok_or(GraphError::MissingValue(id))
}

/// Forward mode: ∂of/∂wrt for each node in `of`, shaped `[of shape..., wrt shape...]`.
pub fn forward_partials(
    graph: &Graph,
    state: &State,
    wrt: NodeId,
    of: &[NodeId],
) -> Result<Partials> {
    graph.check_ids(std::iter::once(&wrt).chain(of))?;
    let wrt_node = &graph[wrt];
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
        let node = &graph[id];
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
            .collect::<Result<SmallVec<[&Value; 4]>>>()?;
        let out = value(state, id)?;
        tangents[i] = jvp(graph, id, &parent_values, out, &parent_tangents, n)?;
    }

    for &o in of {
        if let Some(t) = &tangents[o.0] {
            // [n, of...] -> [wrt..., of...] -> [of..., wrt...]
            let of_shape = graph[o].shape.to_vec();
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
    graph.check_ids(std::iter::once(&of).chain(wrt))?;
    let of_shape = graph[of].shape.to_vec();
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
    graph.check_ids(std::iter::once(&of).chain(wrt))?;
    let of_shape = graph[of].shape.to_vec();
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
    let mut is_wrt = vec![false; of.0 + 1];
    for &w in wrt {
        if w.0 <= of.0 {
            is_wrt[w.0] = true;
        }
    }
    let seed = upstream.reshape(&batched(b, &of_shape))?;
    reverse_sweep(graph, state, [(of, seed)], b, &is_wrt, |id, cotangent| {
        let shape: Dims = outer
            .iter()
            .chain(graph[id].shape.iter())
            .map(|&d| d as isize)
            .collect();
        result.partials.insert(id, cotangent.into_reshape(&shape)?);
        Ok(())
    })?;
    Ok(result)
}

/// Reverse mode from several scalar-valued seeds at once: each seed is ∂f/∂node for one node (shaped like that
/// node), and the result is ∂f/∂w for every node w in `wrt` reachable from a seed, shaped like w. Equivalent to
/// summing [`reverse_partials_from`] over the seeds, but with a single sweep over the graph.
pub(crate) fn reverse_from_seeds(
    graph: &Graph,
    state: &State,
    seeds: impl IntoIterator<Item = (NodeId, DoubleTensor)>,
    is_wrt: &[bool],
    mut emit: impl FnMut(NodeId, DoubleTensor) -> Result<()>,
) -> Result<()> {
    let seeds = seeds
        .into_iter()
        .map(|(id, seed)| Ok((id, seed.into_reshape(&batched(1, &graph[id].shape))?)))
        .collect::<Result<SmallVec<[(NodeId, DoubleTensor); 8]>>>()?;
    reverse_sweep(graph, state, seeds, 1, is_wrt, |id, cotangent| {
        let shape: Dims = graph[id].shape.iter().map(|&d| d as isize).collect();
        emit(id, cotangent.into_reshape(&shape)?)
    })
}

/// The reverse sweep: from batched cotangents `[b, node shape...]` at the seed nodes (summed where a node is
/// seeded twice), visits differentiable ancestors in decreasing id order, stopping at probabilistic nodes and at
/// the nodes marked in `is_wrt`, whose batched cotangents it emits.
fn reverse_sweep(
    graph: &Graph,
    state: &State,
    seeds: impl IntoIterator<Item = (NodeId, DoubleTensor)>,
    b: usize,
    is_wrt: &[bool],
    mut emit: impl FnMut(NodeId, DoubleTensor) -> Result<()>,
) -> Result<()> {
    let mut cotangents: SmallVec<[Option<DoubleTensor>; 8]> = SmallVec::new();
    for (id, seed) in seeds {
        if cotangents.len() <= id.0 {
            cotangents.resize(id.0 + 1, None);
        }
        accumulate(&mut cotangents[id.0], seed)?;
    }
    for i in (0..cotangents.len()).rev() {
        let Some(cotangent) = cotangents[i].take() else {
            continue;
        };
        let id = NodeId(i);
        if is_wrt.get(i).copied().unwrap_or(false) {
            emit(id, cotangent)?;
            continue;
        }
        let node = &graph[id];
        if node.is_probabilistic() || !is_differentiable(graph, id) {
            continue;
        }
        let parent_values = node
            .parents
            .iter()
            .map(|&p| value(state, p))
            .collect::<Result<SmallVec<[&Value; 4]>>>()?;
        let out = value(state, id)?;
        let contributions = match scalar_vjp(node, &parent_values, out, &cotangent) {
            Some(fast) => fast,
            None => vjp(graph, id, &parent_values, out, cotangent, b)?
                .into_iter()
                .collect(),
        };
        for (&parent, contribution) in node.parents.iter().zip(contributions) {
            let Some(c) = contribution else { continue };
            if is_differentiable(graph, parent) {
                accumulate(&mut cotangents[parent.0], c)?;
            }
        }
    }
    Ok(())
}

/// Adds a contribution to a node's (co)tangent.
fn accumulate(slot: &mut Option<DoubleTensor>, contribution: DoubleTensor) -> Result<()> {
    *slot = Some(match slot.take() {
        Some(existing) => existing.add(&contribution)?,
        None => contribution,
    });
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------------
// helpers on batched tensors [B, shape...]

/// A small list of dimensions, inline for the ranks typical of models.
type Dims = SmallVec<[isize; 6]>;

fn batched(b: usize, shape: &[usize]) -> Dims {
    batched_shape(b, shape)
        .iter()
        .map(|&d| d as isize)
        .collect()
}

/// The shape [b, shape...] of a batched tensor.
fn batched_shape(b: usize, shape: &[usize]) -> SmallVec<[usize; 6]> {
    std::iter::once(b).chain(shape.iter().copied()).collect()
}

/// Broadcasts a batched tensor [b, s...] to [b, out...] (results may lack broadcast dims of other operands).
fn fit(x: DoubleTensor, b: usize, out_shape: &[usize]) -> Result<DoubleTensor> {
    let shape = batched_shape(b, out_shape);
    if x.shape() == shape.as_slice() {
        return Ok(x);
    }
    Ok(lift(&x, out_shape.len())?.broadcast_to(&shape)?)
}

/// The identity as a batched tangent: [n, shape...] with element [k, j...] = 1 where j is the k-th element.
fn identity(n: usize, shape: &[usize]) -> Result<DoubleTensor> {
    Ok(DoubleTensor::eye(n)?.reshape(&batched(n, shape))?)
}

/// Reshapes a batched tensor [B, s...] to [B, 1..., s...] with `rank` non-batch dims, so it broadcasts against
/// tensors of that rank.
fn lift(t: &DoubleTensor, rank: usize) -> Result<Cow<'_, DoubleTensor>> {
    let shape = t.shape();
    let inner = &shape[1..];
    if inner.len() >= rank {
        return Ok(Cow::Borrowed(t));
    }
    let mut new_shape: Dims = smallvec::smallvec![shape[0] as isize];
    new_shape.extend(std::iter::repeat_n(1, rank - inner.len()));
    new_shape.extend(inner.iter().map(|&d| d as isize));
    Ok(Cow::Owned(t.reshape(&new_shape)?))
}

/// Sums a batched tensor [B, out...] over the dimensions that `target` was broadcast along, giving [B, target...].
fn reduce_to(t: DoubleTensor, target: &[usize]) -> Result<DoubleTensor> {
    let inner = &t.shape()[1..];
    if inner == target {
        return Ok(t);
    }
    let rank = inner.len();
    let offset = rank - target.len();
    let axes: Dims = (0..rank)
        .filter(|&i| i < offset || (target[i - offset] == 1 && inner[i] != 1))
        .map(|i| i as isize + 1)
        .collect();
    let shape = batched(t.shape()[0], target);
    let summed = if axes.is_empty() {
        t
    } else {
        t.sum_axes(&axes)?
    };
    Ok(summed.into_reshape(&shape)?)
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

// ---------------------------------------------------------------------------------------------------------------
// products, scans and contractions: closed forms that never divide by an element (so are exact where elements are
// zero) and never build a dense Jacobian

/// Resolves a possibly negative axis of a tensor of the given rank.
fn resolve_axis(axis: isize, rank: usize) -> Result<usize> {
    let a = if axis < 0 { axis + rank as isize } else { axis };
    if a < 0 || a as usize >= rank {
        return Err(GraphError::InvalidValue(format!(
            "axis {axis} out of range for rank {rank}"
        )));
    }
    Ok(a as usize)
}

/// The lanes of a tensor of `shape` along `axis`: (number of lanes before the axis, lane length, stride). Lane
/// `(o, i)` holds the elements at `(o * n + k) * stride + i` for `k` in `0..n`.
fn lanes(shape: &[usize], axis: usize) -> (usize, usize, usize) {
    (
        shape[..axis].iter().product(),
        shape[axis],
        shape[axis + 1..].iter().product(),
    )
}

fn batched_tensor(b: usize, shape: &[usize], data: Vec<f64>) -> Result<DoubleTensor> {
    Ok(DoubleTensor::new(&batched_shape(b, shape), data)?)
}

/// Reverse mode of a cumulative sum along non-batch `axis` of a batched cotangent [B, shape...]: the reverse
/// cumulative sum, x̄_i = Σ_{k ≥ i} ȳ_k.
fn reverse_cumsum(g: &DoubleTensor, axis: usize) -> Result<DoubleTensor> {
    let (outer, n, stride) = lanes(g.shape(), axis + 1);
    let mut data = g.as_slice().to_vec();
    for o in 0..outer {
        for i in 0..stride {
            let base = o * n * stride + i;
            for k in (0..n.saturating_sub(1)).rev() {
                data[base + k * stride] += data[base + (k + 1) * stride];
            }
        }
    }
    Ok(DoubleTensor::new(g.shape(), data)?)
}

/// Forward mode of y = cumprod(x) along `axis` for a batched tangent [B, shape...], from the recurrence
/// y_k = y_{k-1} x_k: ẏ_k = ẏ_{k-1} x_k + y_{k-1} ẋ_k (with y_{-1} = 1, ẏ_{-1} = 0).
fn cumprod_forward(
    t: &DoubleTensor,
    x: &DoubleTensor,
    y: &DoubleTensor,
    axis: usize,
) -> Result<DoubleTensor> {
    let (outer, n, stride) = lanes(x.shape(), axis);
    let b = t.shape()[0];
    let (xs, ys, ts) = (x.as_slice(), y.as_slice(), t.as_slice());
    let mut data = vec![0.0; ts.len()];
    for ob in 0..b * outer {
        let xo = (ob % outer) * n * stride;
        let to = ob * n * stride;
        for i in 0..stride {
            let (mut dy, mut prev) = (0.0, 1.0);
            for k in 0..n {
                let (xi, ti) = (xo + k * stride + i, to + k * stride + i);
                dy = dy * xs[xi] + prev * ts[ti];
                data[ti] = dy;
                prev = ys[xi];
            }
        }
    }
    batched_tensor(b, x.shape(), data)
}

/// Reverse mode of y = cumprod(x) along `axis` for a batched cotangent [B, shape...]:
/// x̄_i = Σ_{k ≥ i} ȳ_k Π_{j ≤ k, j ≠ i} x_j = y_{i-1} a_i, where a_i = ȳ_i + x_{i+1} a_{i+1} is a reverse scan.
fn cumprod_reverse(
    g: &DoubleTensor,
    x: &DoubleTensor,
    y: &DoubleTensor,
    axis: usize,
) -> Result<DoubleTensor> {
    let (outer, n, stride) = lanes(x.shape(), axis);
    let b = g.shape()[0];
    let (xs, ys, gs) = (x.as_slice(), y.as_slice(), g.as_slice());
    let mut data = vec![0.0; gs.len()];
    for ob in 0..b * outer {
        let xo = (ob % outer) * n * stride;
        let go = ob * n * stride;
        for i in 0..stride {
            let mut a = 0.0;
            for k in (0..n).rev() {
                let (xi, gi) = (xo + k * stride + i, go + k * stride + i);
                if k + 1 < n {
                    a *= xs[xi + stride];
                }
                a += gs[gi];
                data[gi] = if k == 0 { a } else { ys[xi - stride] * a };
            }
        }
    }
    batched_tensor(b, x.shape(), data)
}

/// The distinct axes a product reduces, sorted.
fn product_axes(axes: &Option<Vec<isize>>, x: &DoubleTensor) -> Result<Vec<usize>> {
    let (mut axes, _) = reduced_axes(axes, x.rank(), x.shape())?;
    axes.sort_unstable();
    axes.dedup();
    Ok(axes)
}

/// For a product over `axes` (distinct and in range) of x: each element's product of the other elements of its
/// group, from exclusive prefix and suffix products, so exact where elements are zero.
fn product_of_others(x: &DoubleTensor, axes: &[usize]) -> Result<DoubleTensor> {
    let rank = x.rank();
    let permutation: Vec<usize> = (0..rank)
        .filter(|a| !axes.contains(a))
        .chain(axes.iter().copied())
        .collect();
    let group: usize = axes.iter().map(|&a| x.shape()[a]).product();
    let permuted = x.permute(&permutation)?;
    let values = permuted.as_slice();
    let mut others = vec![1.0; values.len()];
    if group > 0 {
        for (v, o) in values.chunks(group).zip(others.chunks_mut(group)) {
            let mut running = 1.0;
            for (o, &v) in o.iter_mut().zip(v) {
                *o = running;
                running *= v;
            }
            running = 1.0;
            for (o, &v) in o.iter_mut().zip(v).rev() {
                *o *= running;
                running *= v;
            }
        }
    }
    let mut inverse = vec![0; rank];
    for (i, &p) in permutation.iter().enumerate() {
        inverse[p] = i;
    }
    Ok(DoubleTensor::new(permuted.shape(), others)?.permute(&inverse)?)
}

/// A tensor_multiply's contracted axes of each operand, resolved, and each operand's kept (uncontracted) axes.
struct Contraction {
    left: Vec<usize>,
    right: Vec<usize>,
    kept_left: Vec<usize>,
    kept_right: Vec<usize>,
}

impl Contraction {
    fn new(
        left_axes: &[isize],
        right_axes: &[isize],
        l_rank: usize,
        r_rank: usize,
    ) -> Result<Self> {
        let resolve = |axes: &[isize], rank: usize| -> Result<Vec<usize>> {
            axes.iter().map(|&a| resolve_axis(a, rank)).collect()
        };
        let left = resolve(left_axes, l_rank)?;
        let right = resolve(right_axes, r_rank)?;
        let kept_left = (0..l_rank).filter(|d| !left.contains(d)).collect();
        let kept_right = (0..r_rank).filter(|d| !right.contains(d)).collect();
        Ok(Self {
            left,
            right,
            kept_left,
            kept_right,
        })
    }

    /// The permutation taking a cotangent contracted with the other operand, laid out
    /// [B, kept..., the other's contracted axes in increasing order], to [B, operand...]; `contracted[k]` of this
    /// operand pairs with `other[k]` of the other.
    fn to_operand(kept: &[usize], contracted: &[usize], other: &[usize]) -> Vec<usize> {
        let mut sorted_other = other.to_vec();
        sorted_other.sort_unstable();
        let rank = kept.len() + contracted.len();
        let mut permutation = vec![0; rank + 1];
        for (d, slot) in permutation.iter_mut().enumerate().skip(1) {
            let d = d - 1;
            *slot = 1 + match kept.iter().position(|&k| k == d) {
                Some(p) => p,
                None => {
                    let k = contracted.iter().position(|&c| c == d).unwrap_or(0);
                    kept.len()
                        + sorted_other
                            .iter()
                            .position(|&o| o == other[k])
                            .unwrap_or(0)
                }
            };
        }
        permutation
    }
}

fn signed(axes: &[usize], offset: usize) -> Vec<isize> {
    axes.iter().map(|&a| (a + offset) as isize).collect()
}

/// For ops whose every output element is a copy of one input element or zero (reshape, slice, concat, ...): for each
/// output element, the (parent, flat index) it copies, found by evaluating the op on inputs holding their own
/// (offset) indices.
pub(crate) fn index_map(op: &Op, parents: &[&Value]) -> Result<Vec<Option<(usize, usize)>>> {
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
        if b == 0 {
            // an empty "with respect to": every partial is empty
            break;
        }
        let n_p = t.len() / b;
        for k in 0..b {
            data[k * n_out + o] = t[k * n_p + idx];
        }
    }
    Ok(Some(DoubleTensor::new(&batched_shape(b, out_shape), data)?))
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
    // parent lengths come from their shapes: dividing the buffer length by `b` fails when `b` is 0 (an empty "of")
    let parent_lens: Vec<usize> = parents.iter().map(|v| v.shape().iter().product()).collect();
    for (o, source) in map.iter().enumerate() {
        let Some((p, idx)) = *source else { continue };
        let n_p = parent_lens[p];
        for k in 0..b {
            results[p][k * n_p + idx] += g[k * n_out + o];
        }
    }
    parents
        .iter()
        .zip(results)
        .map(|(v, data)| Ok(Some(DoubleTensor::new(&batched_shape(b, v.shape()), data)?)))
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

/// A local derivative ∂y/∂operand of an element-wise binary op, broadcastable against the output.
enum Local<'a> {
    /// ∂y/∂operand = 1 (add, sub's left operand)
    Identity,
    /// ∂y/∂operand = -1 (sub's right operand)
    Negated,
    /// ∂y/∂operand is this tensor
    Scale(Cow<'a, DoubleTensor>),
}

impl Local<'_> {
    /// Forward mode: the operand's batched tangent [B, operand...] times the derivative, as [B, out...].
    fn forward(&self, t: &DoubleTensor, b: usize, out_shape: &[usize]) -> Result<DoubleTensor> {
        Ok(match self {
            Local::Identity => fit(t.clone(), b, out_shape)?,
            Local::Negated => fit(t.neg(), b, out_shape)?,
            Local::Scale(d) => fit(scale(t, d)?, b, out_shape)?,
        })
    }

    /// Reverse mode: the batched output cotangent [B, out...] times the derivative, as [B, out...].
    fn reverse(&self, g: &DoubleTensor) -> Result<DoubleTensor> {
        Ok(match self {
            Local::Identity => g.clone(),
            Local::Negated => g.neg(),
            Local::Scale(d) => g.mul(d)?,
        })
    }
}

/// (∂y/∂left, ∂y/∂right), each broadcastable against the output.
fn binary_derivatives<'a>(
    op: BinaryOp,
    l: &'a DoubleTensor,
    r: &'a DoubleTensor,
    y: &DoubleTensor,
) -> Result<(Local<'a>, Local<'a>)> {
    let owned = |t: DoubleTensor| Local::Scale(Cow::Owned(t));
    Ok(match op {
        BinaryOp::Add => (Local::Identity, Local::Identity),
        BinaryOp::Sub => (Local::Identity, Local::Negated),
        BinaryOp::Mul => (
            Local::Scale(Cow::Borrowed(r)),
            Local::Scale(Cow::Borrowed(l)),
        ),
        BinaryOp::Div => (
            owned(r.reciprocal()),
            owned(l.zip_with(r, |l, r| -l / (r * r))?),
        ),
        BinaryOp::Pow => (
            owned(l.zip_with(r, |l, r| r * l.powf(r - 1.0))?),
            owned(l.ln().mul(y)?),
        ),
        BinaryOp::Max | BinaryOp::Min => {
            let max = op == BinaryOp::Max;
            let m = l.zip_with(r, |l, r| {
                f64::from(u8::from(if max { l >= r } else { l <= r }))
            })?;
            let other = m.map(|m| 1.0 - m);
            (owned(m), owned(other))
        }
        // left is x and right is y: atan2(y, x)
        BinaryOp::Atan2 => (
            owned(l.zip_with(r, |x, y| -y / (x * x + y * y))?),
            owned(l.zip_with(r, |x, y| x / (x * x + y * y))?),
        ),
        BinaryOp::LogAddExp => (owned(l.sub(y)?.exp()), owned(r.sub(y)?.exp())),
        BinaryOp::LogAddExp2 => (owned(l.sub(y)?.exp2()), owned(r.sub(y)?.exp2())),
        // y * ln(x): d/dx = y / x; d/dy = ln(x), NaN where y == 0 (as Java)
        BinaryOp::SafeLnTimes => (
            owned(l.zip_with(r, |x, y| y / x)?),
            owned(l.zip_with(r, |x, y| if y == 0.0 { f64::NAN } else { x.ln() })?),
        ),
        BinaryOp::Rem => {
            return Err(GraphError::NotDifferentiable(
                "rem is not differentiable".into(),
            ));
        }
    })
}

// ---------------------------------------------------------------------------------------------------------------
// scalar fast path: unary and binary ops whose operands and output are single f64 values (most nodes of typical
// models). The formulas are exactly those of `unary_derivative` / `binary_derivatives`, so results are bit-for-bit
// the same as the tensor path.

fn sign_of(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        x
    }
}

/// d y / d x for a unary op at x with y = op(x), or None if the op isn't differentiable.
fn unary_derivative_scalar(u: UnaryOp, x: f64, y: f64) -> Option<f64> {
    use std::f64::consts::{LN_2, LN_10};
    Some(match u {
        UnaryOp::Abs => sign_of(x),
        UnaryOp::Neg => -1.0,
        UnaryOp::Reciprocal => -y * y,
        UnaryOp::Exp => y,
        UnaryOp::Exp2 => y * LN_2,
        UnaryOp::ExpM1 => y + 1.0,
        UnaryOp::Ln => 1.0 / x,
        UnaryOp::Ln1p => 1.0 / (1.0 + x),
        UnaryOp::Log2 => 1.0 / (x * LN_2),
        UnaryOp::Log10 => 1.0 / (x * LN_10),
        UnaryOp::Sqrt => 0.5 / y,
        UnaryOp::Sin => x.cos(),
        UnaryOp::Cos => -x.sin(),
        UnaryOp::Tan => 1.0 / (x.cos() * x.cos()),
        UnaryOp::Asin => 1.0 / (1.0 - x * x).sqrt(),
        UnaryOp::Acos => -1.0 / (1.0 - x * x).sqrt(),
        UnaryOp::Atan => 1.0 / (1.0 + x * x),
        UnaryOp::Sinh => x.cosh(),
        UnaryOp::Cosh => x.sinh(),
        UnaryOp::Tanh => 1.0 - y * y,
        UnaryOp::Asinh => 1.0 / (x * x + 1.0).sqrt(),
        UnaryOp::Acosh => 1.0 / (x * x - 1.0).sqrt(),
        UnaryOp::Atanh => 1.0 / (1.0 - x * x),
        UnaryOp::Sigmoid => y * (1.0 - y),
        UnaryOp::LnGamma => special::digamma(x),
        UnaryOp::Digamma => special::trigamma(x),
        UnaryOp::Trigamma => special::tetragamma(x),
        _ => return None,
    })
}

/// (d y / d l, d y / d r) for a binary op at (l, r) with y = op(l, r), or None if it isn't differentiable.
fn binary_derivatives_scalar(op: BinaryOp, l: f64, r: f64, y: f64) -> Option<(f64, f64)> {
    Some(match op {
        BinaryOp::Add => (1.0, 1.0),
        BinaryOp::Sub => (1.0, -1.0),
        BinaryOp::Mul => (r, l),
        BinaryOp::Div => (1.0 / r, -l / (r * r)),
        BinaryOp::Pow => (r * l.powf(r - 1.0), l.ln() * y),
        BinaryOp::Max => {
            let m = f64::from(u8::from(l >= r));
            (m, 1.0 - m)
        }
        BinaryOp::Min => {
            let m = f64::from(u8::from(l <= r));
            (m, 1.0 - m)
        }
        BinaryOp::Atan2 => (-r / (l * l + r * r), l / (l * l + r * r)),
        BinaryOp::LogAddExp => ((l - y).exp(), (r - y).exp()),
        BinaryOp::LogAddExp2 => ((l - y).exp2(), (r - y).exp2()),
        BinaryOp::SafeLnTimes => (r / l, if r == 0.0 { f64::NAN } else { l.ln() }),
        _ => return None,
    })
}

/// The scalar fast path of `vjp` for a cotangent of one element: contributions shaped [1] (a batch of one
/// scalar), or None when the node isn't a scalar unary/binary op over scalar f64 parents.
fn scalar_vjp(
    node: &Node,
    parents: &[&Value],
    out: &Value,
    g: &DoubleTensor,
) -> Option<SmallVec<[Option<DoubleTensor>; 4]>> {
    if g.len() != 1 || !node.shape.is_empty() {
        return None;
    }
    let scalar = |v: &Value| match v {
        Value::F64(t) if t.rank() == 0 => Some(t.as_slice()[0]),
        _ => None,
    };
    let one = |v: f64| Some(DoubleTensor::from_fn(&[1], |_| v));
    let g = g.as_slice()[0];
    let y = scalar(out)?;
    match &node.op {
        Op::Unary(u) => {
            let x = scalar(parents[0])?;
            let d = unary_derivative_scalar(*u, x, y)?;
            Some(smallvec::smallvec![one(g * d)])
        }
        Op::Binary(b) => {
            let (l, r) = (scalar(parents[0])?, scalar(parents[1])?);
            let (dl, dr) = binary_derivatives_scalar(*b, l, r, y)?;
            Some(smallvec::smallvec![one(g * dl), one(g * dr)])
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Jacobians of reductions

/// The Jacobian [1, n] of the max (or min) of every element: the gradient goes to the selected element, split
/// equally between ties. (Max, Min and StdDev only reduce over every element.)
fn extremum_jacobian(x: &DoubleTensor, max: bool) -> Result<DoubleTensor> {
    let best = if max { x.max() } else { x.min() }.unwrap_or(f64::NAN);
    let selected = x.map(|v| f64::from(u8::from(v == best)));
    let count: f64 = selected.sum();
    Ok((selected / count).reshape(&[1, x.len() as isize])?)
}

/// The Jacobian [1, n] of the standard deviation of every element.
fn std_dev_jacobian(x: &DoubleTensor) -> Result<DoubleTensor> {
    let n = x.len();
    let s = x.std_dev();
    let mean = x.mean();
    let d = x.map(|v| (v - mean) / ((n as f64 - 1.0) * s));
    Ok(d.reshape(&[1, n as isize])?)
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
    graph: &Graph,
    id: NodeId,
    parents: &[&Value],
    out: &Value,
    t: &[Option<&DoubleTensor>],
    b: usize,
) -> Result<Option<DoubleTensor>> {
    let node = &graph[id];
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
            let from_left = t[0].map(|tl| dl.forward(tl, b, &out_shape)).transpose()?;
            let from_right = t[1].map(|tr| dr.forward(tr, b, &out_shape)).transpose()?;
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
                ReduceOp::Product => {
                    // ẏ = Σ over each group of ẋ_i × (the product of the group's other elements)
                    let x = f64v(parents[0])?;
                    let axes = product_axes(axes, x)?;
                    let products = t0.mul(&product_of_others(x, &axes)?)?;
                    let summed = if axes.is_empty() {
                        products
                    } else {
                        products.sum_axes(&signed(&axes, 1))?
                    };
                    Some(summed.reshape(&batched(b, &out_shape))?)
                }
                ReduceOp::Max | ReduceOp::Min => {
                    let j = extremum_jacobian(f64v(parents[0])?, *kind == ReduceOp::Max)?;
                    Some(jacobian_forward(t0, &j, &out_shape)?)
                }
                ReduceOp::StdDev => {
                    let j = std_dev_jacobian(f64v(parents[0])?)?;
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
        Op::Scan { kind, axis } => {
            let Some(t0) = t[0] else { return Ok(None) };
            let x = f64v(parents[0])?;
            let axis = resolve_axis(*axis, x.rank())?;
            match kind {
                ScanOp::CumSum => Some(t0.cumsum(axis as isize + 1)?),
                ScanOp::CumProd => Some(cumprod_forward(t0, x, y, axis)?),
            }
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
        | Op::TrianglePart { .. } => {
            gather_forward(graph.index_map(id, parents)?, t, b, &out_shape)?
        }
        Op::TensorMultiply {
            left_axes,
            right_axes,
        } => {
            // d(L·R) = dL·R + L·dR, contracting the same axes (shifted past the batch axis of the tangent)
            let (l, r) = (f64v(parents[0])?, f64v(parents[1])?);
            let c = Contraction::new(left_axes, right_axes, l.rank(), r.rank())?;
            let from_left = t[0]
                .map(|tl| tl.tensor_multiply(r, &signed(&c.left, 1), &signed(&c.right, 0)))
                .transpose()?;
            let from_right = t[1]
                .map(|tr| -> Result<DoubleTensor> {
                    // [kept left..., B, kept right...] -> [B, kept left..., kept right...]
                    let product =
                        l.tensor_multiply(tr, &signed(&c.left, 0), &signed(&c.right, 1))?;
                    let kl = c.kept_left.len();
                    let permutation: Vec<usize> = std::iter::once(kl)
                        .chain(0..kl)
                        .chain(kl + 1..product.rank())
                        .collect();
                    Ok(product.permute(&permutation)?)
                })
                .transpose()?;
            add_opt(from_left, from_right)?
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
                    Ok(a.matmul_transposed(&*lift(tb, rank)?, *transpose_left, *transpose_right)?)
                })
                .transpose()?;
            // results may lack broadcast batch dims of the other operand; bring both to [B, out...]
            add_opt(
                from_left.map(|x| fit(x, b, &out_shape)).transpose()?,
                from_right.map(|x| fit(x, b, &out_shape)).transpose()?,
            )?
        }
        Op::Inverse => {
            let Some(t0) = t[0] else { return Ok(None) };
            // dX = -X dA X
            Some(y.matmul(&*lift(t0, out_shape.len())?)?.matmul(y)?.neg())
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
            let da = symmetric_from_lower(&*lift(t0, out_shape.len())?)?;
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
            let m = where_mask(parents)?;
            let from_then = t[1].map(|tt| scale(tt, &m)).transpose()?;
            let from_else = t[2].map(|te| scale(te, &m.map(|m| 1.0 - m))).transpose()?;
            add_opt(
                from_then.map(|x| fit(x, b, &out_shape)).transpose()?,
                from_else.map(|x| fit(x, b, &out_shape)).transpose()?,
            )?
        }
        Op::Cpt { conditions, keys } => {
            // the selected outcome's tangent; zero (rather than none) if only unselected outcomes depend on wrt
            let selected = cpt_selected(*conditions, keys, parents)
                .map_err(|e| invalid_selection("cpt", e))?;
            match t[selected] {
                Some(ts) => Some(ts.clone()),
                None if t[*conditions..].iter().any(Option::is_some) => {
                    Some(DoubleTensor::zeros(&batched_shape(b, &out_shape)))
                }
                None => None,
            }
        }
        Op::Multiplexer => {
            // as Cpt: the selected option's tangent, zero if only unselected options depend on wrt
            let selected =
                multiplexer_selected(parents).map_err(|e| invalid_selection("multiplexer", e))?;
            match t[selected] {
                Some(ts) => Some(ts.clone()),
                None if t[1..].iter().any(Option::is_some) => {
                    Some(DoubleTensor::zeros(&batched_shape(b, &out_shape)))
                }
                None => None,
            }
        }
        Op::SetWithMask => {
            let set = set_mask(parents)?;
            let from_operand = t[0]
                .map(|t0| scale(t0, &set.map(|m| 1.0 - m)))
                .transpose()?;
            let from_value = t[2]
                .map(|tv| scale(&tv.reshape(&[b as isize])?, &set))
                .transpose()?;
            add_opt(from_operand, from_value)?
        }
        Op::ReplaceNan(_) => {
            let Some(t0) = t[0] else { return Ok(None) };
            Some(scale(t0, &not_nan_mask(parents)?)?)
        }
        Op::Lambda(lambda) => {
            let jacobian = lambda.jacobian.as_ref().ok_or_else(|| {
                GraphError::NotDifferentiable(format!("lambda {} has no jacobian", lambda.name))
            })?;
            let jacobians = jacobian(parents).map_err(GraphError::Lambda)?;
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
    graph: &Graph,
    id: NodeId,
    parents: &[&Value],
    out: &Value,
    cotangent: DoubleTensor,
    b: usize,
) -> Result<Vec<Option<DoubleTensor>>> {
    let node = &graph[id];
    let out_shape = node.shape.to_vec();
    let y = f64v(out)?;
    let op = &node.op;
    let none = || vec![None; parents.len()];
    let g = &cotangent;
    Ok(match op {
        Op::Proxy => vec![Some(cotangent)],
        Op::Unary(u) => {
            let x = f64v(parents[0])?;
            vec![Some(reduce_to(
                g.mul(&unary_derivative(*u, x, y)?)?,
                x.shape(),
            )?)]
        }
        Op::Binary(bop) => {
            let (l, r) = (f64v(parents[0])?, f64v(parents[1])?);
            let (dl, dr) = binary_derivatives(*bop, l, r, y)?;
            vec![
                Some(reduce_to(dl.reverse(g)?, l.shape())?),
                Some(reduce_to(dr.reverse(g)?, r.shape())?),
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
                    let expanded = g
                        .reshape(&batched(b, &kept))?
                        .broadcast_to(&batched_shape(b, x.shape()))?;
                    vec![Some(if *kind == ReduceOp::Mean {
                        expanded / count as f64
                    } else {
                        expanded
                    })]
                }
                ReduceOp::Product => {
                    // x̄_i = ȳ of i's group × (the product of the group's other elements)
                    let axes = product_axes(axes, x)?;
                    let kept: Vec<usize> = x
                        .shape()
                        .iter()
                        .enumerate()
                        .map(|(i, &d)| if axes.contains(&i) { 1 } else { d })
                        .collect();
                    let others = product_of_others(x, &axes)?;
                    vec![Some(g.reshape(&batched(b, &kept))?.mul(&others)?)]
                }
                ReduceOp::Max | ReduceOp::Min => {
                    let j = extremum_jacobian(x, *kind == ReduceOp::Max)?;
                    vec![Some(jacobian_reverse(g, &j, x.shape())?)]
                }
                ReduceOp::StdDev => {
                    vec![Some(jacobian_reverse(g, &std_dev_jacobian(x)?, x.shape())?)]
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
        Op::Scan { kind, axis } => {
            let x = f64v(parents[0])?;
            let axis = resolve_axis(*axis, x.rank())?;
            vec![Some(match kind {
                ScanOp::CumSum => reverse_cumsum(g, axis)?,
                ScanOp::CumProd => cumprod_reverse(g, x, y, axis)?,
            })]
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
        | Op::TrianglePart { .. } => gather_reverse(graph.index_map(id, parents)?, g, parents, b)?,
        Op::TensorMultiply {
            left_axes,
            right_axes,
        } => {
            // G [B, kept left..., kept right...]: L̄ contracts G's kept right axes with R's, R̄ its kept left
            // axes with L's; each is then permuted back to its operand's layout
            let (l, r) = (f64v(parents[0])?, f64v(parents[1])?);
            let c = Contraction::new(left_axes, right_axes, l.rank(), r.rank())?;
            let (kl, kr) = (c.kept_left.len(), c.kept_right.len());
            let g_kept_left: Vec<usize> = (1..1 + kl).collect();
            let g_kept_right: Vec<usize> = (1 + kl..1 + kl + kr).collect();
            let dl = g
                .tensor_multiply(r, &signed(&g_kept_right, 0), &signed(&c.kept_right, 0))?
                .permute(&Contraction::to_operand(&c.kept_left, &c.left, &c.right))?;
            let dr = g
                .tensor_multiply(l, &signed(&g_kept_left, 0), &signed(&c.kept_left, 0))?
                .permute(&Contraction::to_operand(&c.kept_right, &c.right, &c.left))?;
            vec![Some(dl), Some(dr)]
        }
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
                Some(reduce_to(da, a.shape())?),
                Some(reduce_to(db, bm.shape())?),
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
            let m = where_mask(parents)?;
            let then_shape = parents[1].shape();
            let else_shape = parents[2].shape();
            vec![
                None,
                Some(reduce_to(g.mul(&m)?, then_shape)?),
                Some(reduce_to(g.mul(&m.map(|m| 1.0 - m))?, else_shape)?),
            ]
        }
        Op::Cpt { conditions, keys } => {
            // the selected outcome gets the cotangent, the other outcomes zero (as forward mode)
            let selected = cpt_selected(*conditions, keys, parents)
                .map_err(|e| invalid_selection("cpt", e))?;
            route_to(parents.len(), *conditions, selected, cotangent)
        }
        Op::Multiplexer => {
            // as Cpt: the selected option gets the cotangent, the other options zero
            let selected =
                multiplexer_selected(parents).map_err(|e| invalid_selection("multiplexer", e))?;
            route_to(parents.len(), 1, selected, cotangent)
        }
        Op::SetWithMask => {
            let set = set_mask(parents)?;
            let to_operand = g.mul(&set.map(|m| 1.0 - m))?;
            // the value is a scalar (of any rank): sum over every non-batch axis
            let selected = g.mul(&set)?;
            let axes: Vec<isize> = (1..selected.rank() as isize).collect();
            let to_value = if axes.is_empty() {
                selected
            } else {
                selected.sum_axes(&axes)?
            }
            .reshape(&batched(b, parents[2].shape()))?;
            vec![Some(to_operand), None, Some(to_value)]
        }
        Op::ReplaceNan(_) => vec![Some(g.mul(&not_nan_mask(parents)?)?)],
        Op::Lambda(lambda) => {
            let jacobian = lambda.jacobian.as_ref().ok_or_else(|| {
                GraphError::NotDifferentiable(format!("lambda {} has no jacobian", lambda.name))
            })?;
            let jacobians = jacobian(parents).map_err(GraphError::Lambda)?;
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

/// Reverse mode of a selection (CPT, multiplexer): the cotangent to the selected parent, zero to the other
/// options (parents from `first`), and none to the selectors.
fn route_to(
    parents: usize,
    first: usize,
    selected: usize,
    cotangent: DoubleTensor,
) -> Vec<Option<DoubleTensor>> {
    let mut out = vec![None; parents];
    for (i, slot) in out.iter_mut().enumerate().skip(first) {
        if i != selected {
            *slot = Some(DoubleTensor::zeros(cotangent.shape()));
        }
    }
    out[selected] = Some(cotangent);
    out
}

/// The error for a CPT or multiplexer whose selection fails (which its evaluation would have reported first).
fn invalid_selection(op: &str, message: String) -> GraphError {
    GraphError::InvalidValue(format!("{op}: {message}"))
}

/// The where predicate as 1.0 / 0.0, broadcastable against the output.
fn where_mask(parents: &[&Value]) -> Result<DoubleTensor> {
    Ok(parents[0]
        .as_bool()
        .ok_or_else(|| GraphError::InvalidValue("where predicate must be bool".into()))?
        .to_double_mask())
}

/// 1.0 where a `set_with_mask` node takes the set value (mask == 1.0), else 0.0, shaped like the operand.
fn set_mask(parents: &[&Value]) -> Result<DoubleTensor> {
    Ok(f64v(parents[1])?.map(|m| if m == 1.0 { 1.0 } else { 0.0 }))
}

/// 1.0 where a `replace_nan` node's input isn't NaN (so passes through), else 0.0.
fn not_nan_mask(parents: &[&Value]) -> Result<DoubleTensor> {
    Ok(f64v(parents[0])?.map(|x| if x.is_nan() { 0.0 } else { 1.0 }))
}
