//! Loading a model saved in Keanu's schema: each saved node becomes the graph node with the same index.

use crate::Metadata;
use crate::codes::{decode, distribution_of_code};
use crate::error::{IoError, Result};
use crate::proto;
use crate::proto::node::Op as P;
use crate::save::FORMAT_VERSION;
use keanu_graph::op::Op;
use keanu_graph::{DType, Graph, GraphBuilder, NodeId, State, Value};
use keanu_tensor::{BoolTensor, DoubleTensor, IntTensor};

/// A loaded model: its graph, the saved values (observations are observed) and the saved metadata. Saved node `i`
/// is the graph's node `NodeId::from_index(i)`.
#[derive(Debug)]
#[non_exhaustive]
pub struct Loaded {
    /// The graph.
    pub graph: Graph,
    /// The saved values and observations.
    pub state: State,
    /// The saved metadata.
    pub metadata: Metadata,
}

/// A size or index from the file as a `usize`, or a decode error where it doesn't fit.
fn usize_of(v: u64) -> Result<usize> {
    usize::try_from(v).map_err(|_| IoError::Decode(format!("the integer {v} is too large")))
}

/// An integer from the file as an `isize` (an axis, offset or reshape dimension), or a decode error where it doesn't
/// fit (on 32-bit platforms).
fn isize_of(v: i64) -> Result<isize> {
    isize::try_from(v).map_err(|_| IoError::Decode(format!("the integer {v} is too large")))
}

fn usizes(values: &[u64]) -> Result<Vec<usize>> {
    values.iter().map(|&v| usize_of(v)).collect()
}

fn isizes(values: &[i64]) -> Result<Vec<isize>> {
    values.iter().map(|&v| isize_of(v)).collect()
}

/// A shape from the file: dimensions whose product (the element count) fits a `usize`.
fn shape_of(dims: &[u64]) -> Result<Vec<usize>> {
    let shape = usizes(dims)?;
    shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or_else(|| IoError::Decode(format!("the shape {shape:?} has too many elements")))?;
    Ok(shape)
}

fn value_of(tensor: &proto::Tensor) -> Result<Value> {
    use proto::tensor::Values;
    let shape = shape_of(&tensor.shape)?;
    let invalid = |e: keanu_tensor::TensorError| IoError::Decode(e.to_string());
    Ok(match &tensor.values {
        Some(Values::F64(v)) => {
            Value::F64(DoubleTensor::new(&shape, v.values.clone()).map_err(invalid)?)
        }
        Some(Values::I64(v)) => {
            Value::I64(IntTensor::new(&shape, v.values.clone()).map_err(invalid)?)
        }
        Some(Values::Bool(v)) => {
            Value::Bool(BoolTensor::new(&shape, v.values.clone()).map_err(invalid)?)
        }
        None => return Err(IoError::Decode("a tensor has no values".into())),
    })
}

/// Builds a graph from a decoded model.
pub(crate) fn load_model(model: &proto::Model) -> Result<Loaded> {
    match model.format_version {
        0 => return Err(IoError::Decode("the model has no format version".into())),
        found if found > FORMAT_VERSION => {
            return Err(IoError::UnsupportedVersion {
                found,
                supported: FORMAT_VERSION,
            });
        }
        _ => {}
    }
    let saved = model
        .graph
        .as_ref()
        .ok_or_else(|| IoError::Decode("the model has no graph".into()))?;
    let builder = GraphBuilder::new();
    for (index, node) in saved.nodes.iter().enumerate() {
        let invalid = |message: String| IoError::InvalidNode { index, message };
        let parents = node
            .parents
            .iter()
            .map(|&p| match usize::try_from(p) {
                Ok(p) if p < index => Ok(NodeId::from_index(p)),
                _ => Err(invalid(format!("parent {p} is not an earlier node"))),
            })
            .collect::<Result<Vec<_>>>()?;
        let op = node
            .op
            .as_ref()
            .ok_or_else(|| invalid("no operation (or one this version doesn't know)".into()))?;
        let id = add_node(&builder, op, &parents).map_err(|e| match e {
            IoError::Decode(message) => invalid(message),
            other => other,
        })?;
        debug_assert_eq!(id.index(), index);
        if !node.label.is_empty() {
            builder.set_label_of(id, &node.label)?;
        }
    }
    for stored in &saved.values {
        let node = usize::try_from(stored.node)
            .ok()
            .filter(|&n| n < saved.nodes.len())
            .map(NodeId::from_index)
            .ok_or_else(|| {
                IoError::Decode(format!(
                    "a value is stored for unknown node {}",
                    stored.node
                ))
            })?;
        let value = value_of(
            stored
                .value
                .as_ref()
                .ok_or_else(|| IoError::Decode("a stored value has no value".into()))?,
        )?;
        if stored.observed {
            builder.observe_of(node, value)?;
        } else {
            builder.set_value_of(node, value)?;
        }
    }
    let (graph, state) = builder.build()?;
    Ok(Loaded {
        graph,
        state,
        metadata: model.metadata.clone(),
    })
}

/// Adds the node of a saved operation.
fn add_node(builder: &GraphBuilder, op: &P, parents: &[NodeId]) -> Result<NodeId> {
    let op = match op {
        P::Constant(tensor) => Op::Constant(value_of(tensor)?),
        P::Unary(code) => Op::Unary(decode(*code)?),
        P::Binary(code) => Op::Binary(decode(*code)?),
        P::Compare(code) => Op::Compare(decode(*code)?),
        P::Mask(code) => Op::Mask(decode(*code)?),
        P::ApproxEq(p) => Op::ApproxEq(p.epsilon),
        P::Logic(code) => Op::Logic(decode(*code)?),
        P::Not(_) => Op::Not,
        P::All(_) => Op::All,
        P::Any(_) => Op::Any,
        P::Predicate(code) => Op::Predicate(decode(*code)?),
        P::Reduce(p) => Op::Reduce {
            kind: decode(p.kind)?,
            axes: p.axes.as_ref().map(|a| isizes(&a.values)).transpose()?,
        },
        P::Standardize(_) => Op::Standardize,
        P::Cumulative(p) => Op::Scan {
            kind: decode(p.kind)?,
            axis: isize_of(p.axis)?,
        },
        P::ArgReduce(p) => Op::ArgReduce {
            kind: decode(p.kind)?,
            axis: p.axis.map(isize_of).transpose()?,
        },
        P::Reshape(p) => Op::Reshape(isizes(&p.shape)?),
        P::Permute(p) => Op::Permute(usizes(&p.axes)?),
        P::Transpose(_) => Op::Transpose,
        P::Broadcast(p) => Op::Broadcast(shape_of(&p.shape)?),
        P::Slice(p) => Op::Slice(p.spec.clone()),
        P::Concat(p) => Op::Concat(isize_of(p.axis)?),
        P::Take(p) => Op::Take(usizes(&p.index)?),
        P::Gather(p) => Op::Gather {
            axis: isize_of(p.axis)?,
        },
        P::ScatterAdd(p) => Op::ScatterAdd {
            axis: isize_of(p.axis)?,
            size: usize_of(p.size)?,
        },
        P::Stack(p) => Op::Stack(isize_of(p.axis)?),
        P::Diag(_) => Op::Diag,
        P::DiagPart(_) => Op::DiagPart,
        P::Tril(p) => Op::Tril(isize_of(p.k)?),
        P::Triu(p) => Op::Triu(isize_of(p.k)?),
        P::FillTriangular(p) => Op::FillTriangular {
            upper: p.upper,
            lower: p.lower,
        },
        P::TrianglePart(p) => Op::TrianglePart { upper: p.upper },
        P::Matmul(p) => Op::MatMul {
            transpose_left: p.transpose_left,
            transpose_right: p.transpose_right,
        },
        P::TensorMultiply(p) => Op::TensorMultiply {
            left_axes: isizes(&p.left_axes)?,
            right_axes: isizes(&p.right_axes)?,
        },
        P::Inverse(_) => Op::Inverse,
        P::Determinant(_) => Op::Determinant,
        P::Cholesky(_) => Op::Cholesky,
        P::CholeskyInverse(_) => Op::CholeskyInverse,
        P::Cast(code) => Op::Cast(decode(*code)?),
        P::Where(_) => Op::Where,
        P::Multiplexer(_) => Op::Multiplexer,
        P::Cpt(p) => Op::Cpt {
            conditions: usize_of(p.conditions)?,
            keys: p.keys.iter().map(|k| k.values.clone()).collect(),
        },
        P::SetWithMask(_) => Op::SetWithMask,
        P::ReplaceNan(p) => Op::ReplaceNan(p.value),
        P::Assert(p) => Op::Assert(p.message.clone()),
        P::Proxy(p) => {
            let dtype: DType = decode(p.dtype)?;
            let proxy = builder.add_proxy(dtype, &shape_of(&p.shape)?);
            match parents {
                [] => {}
                [parent] => builder.connect_ids(proxy, *parent)?,
                _ => return Err(IoError::Decode("a proxy has more than one parent".into())),
            }
            return Ok(proxy);
        }
        P::Distribution(p) => {
            let kind = distribution_of_code(p.kind, p.kde.as_ref())?;
            return Ok(builder.add_distribution(kind, Some(&shape_of(&p.shape)?), parents)?);
        }
    };
    Ok(builder.add_node(op, parents)?)
}
