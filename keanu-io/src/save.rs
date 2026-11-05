//! Saving a graph in Keanu's schema: one saved node per graph node, in the same order, each with its typed
//! operation.

use crate::Metadata;
use crate::codes::{distribution_code, encode};
use crate::error::{IoError, Result};
use crate::proto;
use crate::proto::node::Op as P;
use keanu_graph::op::{DistributionKind, Op};
use keanu_graph::{Graph, NodeId, State, Value};

/// The schema version this crate writes, and the newest it reads.
pub(crate) const FORMAT_VERSION: u32 = 1;

fn dims(shape: &[usize]) -> Vec<u64> {
    shape.iter().map(|&d| d as u64).collect()
}

fn signed(values: &[isize]) -> Vec<i64> {
    values.iter().map(|&v| v as i64).collect()
}

pub(crate) fn tensor(value: &Value) -> Result<proto::Tensor> {
    use proto::tensor::Values;
    let (shape, values) = match value {
        Value::F64(t) => (
            t.shape(),
            Values::F64(proto::DoubleValues {
                values: t.as_slice().to_vec(),
            }),
        ),
        Value::I64(t) => (
            t.shape(),
            Values::I64(proto::IntValues {
                values: t.as_slice().to_vec(),
            }),
        ),
        Value::Bool(t) => (
            t.shape(),
            Values::Bool(proto::BoolValues {
                values: t.as_slice().to_vec(),
            }),
        ),
        _ => {
            return Err(IoError::NotSaveable(
                "a value of an unsupported type".into(),
            ));
        }
    };
    Ok(proto::Tensor {
        shape: dims(shape),
        values: Some(values),
    })
}

fn empty() -> proto::Empty {
    proto::Empty {}
}

/// The saved operation of a node.
fn save_op(graph: &Graph, id: NodeId) -> Result<P> {
    let node = &graph[id];
    let unsupported = |what: &str| {
        IoError::NotSaveable(format!("{what} (node {}) has no saved form", id.index()))
    };
    Ok(match node.op() {
        Op::Constant(value) => P::Constant(tensor(value)?),
        Op::Unary(op) => P::Unary(encode(op)?),
        Op::Binary(op) => P::Binary(encode(op)?),
        Op::Compare(op) => P::Compare(encode(op)?),
        Op::Mask(op) => P::Mask(encode(op)?),
        Op::ApproxEq(epsilon) => P::ApproxEq(proto::ApproxEq { epsilon: *epsilon }),
        Op::Logic(op) => P::Logic(encode(op)?),
        Op::Not => P::Not(empty()),
        Op::All => P::All(empty()),
        Op::Any => P::Any(empty()),
        Op::Predicate(p) => P::Predicate(encode(p)?),
        Op::Reduce { kind, axes } => P::Reduce(proto::Reduce {
            kind: encode(kind)?,
            axes: axes.as_ref().map(|axes| proto::Axes {
                values: signed(axes),
            }),
        }),
        Op::Standardize => P::Standardize(empty()),
        Op::Scan { kind, axis } => P::Cumulative(proto::Cumulative {
            kind: encode(kind)?,
            axis: *axis as i64,
        }),
        Op::ArgReduce { kind, axis } => P::ArgReduce(proto::ArgReduce {
            kind: encode(kind)?,
            axis: axis.map(|a| a as i64),
        }),
        Op::Reshape(shape) => P::Reshape(proto::Reshape {
            shape: signed(shape),
        }),
        Op::Permute(axes) => P::Permute(proto::Permute { axes: dims(axes) }),
        Op::Transpose => P::Transpose(empty()),
        Op::Broadcast(shape) => P::Broadcast(proto::Broadcast { shape: dims(shape) }),
        Op::Slice(spec) => P::Slice(proto::Slice { spec: spec.clone() }),
        Op::Concat(axis) => P::Concat(proto::Concat { axis: *axis as i64 }),
        Op::Take(index) => P::Take(proto::Take { index: dims(index) }),
        Op::Diag => P::Diag(empty()),
        Op::DiagPart => P::DiagPart(empty()),
        Op::Tril(k) => P::Tril(proto::Triangle { k: *k as i64 }),
        Op::Triu(k) => P::Triu(proto::Triangle { k: *k as i64 }),
        Op::FillTriangular { upper, lower } => P::FillTriangular(proto::FillTriangular {
            upper: *upper,
            lower: *lower,
        }),
        Op::TrianglePart { upper } => P::TrianglePart(proto::TrianglePart { upper: *upper }),
        Op::MatMul {
            transpose_left,
            transpose_right,
        } => P::Matmul(proto::MatMul {
            transpose_left: *transpose_left,
            transpose_right: *transpose_right,
        }),
        Op::TensorMultiply {
            left_axes,
            right_axes,
        } => P::TensorMultiply(proto::TensorMultiply {
            left_axes: signed(left_axes),
            right_axes: signed(right_axes),
        }),
        Op::Inverse => P::Inverse(empty()),
        Op::Determinant => P::Determinant(empty()),
        Op::Cholesky => P::Cholesky(empty()),
        Op::CholeskyInverse => P::CholeskyInverse(empty()),
        Op::Cast(dtype) => P::Cast(encode(dtype)?),
        Op::Where => P::Where(empty()),
        Op::Multiplexer => P::Multiplexer(empty()),
        Op::Cpt { conditions, keys } => P::Cpt(proto::Cpt {
            conditions: *conditions as u64,
            keys: keys
                .iter()
                .map(|key| proto::CptKey {
                    values: key.clone(),
                })
                .collect(),
        }),
        Op::SetWithMask => P::SetWithMask(empty()),
        Op::ReplaceNan(value) => P::ReplaceNan(proto::ReplaceNan { value: *value }),
        Op::Proxy => P::Proxy(proto::Proxy {
            dtype: encode(&node.dtype())?,
            shape: dims(node.shape()),
        }),
        Op::Assert(message) => P::Assert(proto::Assert {
            message: message.clone(),
        }),
        Op::Distribution(distribution) => {
            let kind = distribution_code(&distribution.kind)
                .ok_or_else(|| unsupported("a custom distribution"))?;
            let kde = match &distribution.kind {
                DistributionKind::Kde(kde) => Some(proto::Kde {
                    samples: kde.samples.clone(),
                    bandwidth: kde.bandwidth,
                }),
                _ => None,
            };
            P::Distribution(proto::Distribution {
                kind: kind as i32,
                shape: dims(&distribution.shape),
                kde,
            })
        }
        Op::Lambda(_) => return Err(unsupported("a lambda")),
        other => return Err(unsupported(other.name())),
    })
}

/// The saved form of a graph and the values in a state.
pub(crate) fn save_model(
    graph: &Graph,
    state: &State,
    metadata: &Metadata,
) -> Result<proto::Model> {
    if u32::try_from(graph.len()).is_err() {
        return Err(IoError::NotSaveable(format!(
            "the graph has {} nodes, more than the format allows (2³² - 1)",
            graph.len()
        )));
    }
    let index = |id: NodeId| id.index() as u32;
    let nodes = graph
        .ids()
        .map(|id| {
            let node = &graph[id];
            Ok(proto::Node {
                parents: node.parents().iter().map(|&p| index(p)).collect(),
                label: node.label().unwrap_or_default().to_owned(),
                op: Some(save_op(graph, id)?),
            })
        })
        .collect::<Result<_>>()?;
    let values = graph
        .ids()
        .filter_map(|id| {
            state.value(id).map(|value| {
                Ok(proto::StoredValue {
                    node: index(id),
                    value: Some(tensor(value)?),
                    observed: state.is_observed(id),
                })
            })
        })
        .collect::<Result<_>>()?;
    Ok(proto::Model {
        format_version: FORMAT_VERSION,
        graph: Some(proto::Graph { nodes, values }),
        metadata: metadata.clone(),
    })
}
