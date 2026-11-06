//! Loading a model saved in Keanu's schema: each saved node becomes the graph node with the same index.

use crate::Metadata;
use crate::codes::{decode, distribution_of_type};
use crate::error::{IoError, Result};
use crate::proto;
use crate::proto::node::Op as P;
use crate::save::FORMAT_VERSION;
use keanu_graph::ode::OdeOptions;
use keanu_graph::op::{OdeBody, Op, ScanBody};
use keanu_graph::{DType, Graph, GraphBuilder, NodeId, State, Value};
use keanu_tensor::{BoolTensor, DoubleTensor, IntTensor};
use std::sync::Arc;

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

/// Limits on what loading a file may create, protecting against untrusted files: a few bytes can describe nodes of
/// enormous shapes, or a scan of so many steps that evaluating it would practically never finish. Loading fails with
/// [`IoError::InvalidNode`] for a node beyond them. The defaults suit models of moderate size; raise them to load
/// larger trusted models.
///
/// Whatever the limits, no node may have more than `isize::MAX / 8` elements (the most one allocation can hold).
///
/// ```
/// let mut options = keanu_io::LoadOptions::default();
/// options.max_scan_steps = 1 << 30;
/// options.max_scan_work = 1 << 40;
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct LoadOptions {
    /// The most elements any node (including the nodes of scan bodies) may have. Default 2^31 (16 GiB of `f64`).
    pub max_elements: usize,
    /// The most steps any scan may have. Default 2^24.
    pub max_scan_steps: usize,
    /// The most work any scan may do, counted as its steps times the work of its body: the sum, over the body's
    /// nodes, of each node's element count (at least 1), with a nested scan counting its own work. Default 2^32.
    pub max_scan_work: u64,
    /// The largest step limit ([`OdeOptions::max_steps`]) any ODE may have. Default 10^7, ten times the default
    /// limit.
    pub max_ode_steps: usize,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            max_elements: 1 << 31,
            max_scan_steps: 1 << 24,
            max_scan_work: 1 << 32,
            max_ode_steps: 10_000_000,
        }
    }
}

/// The work of a scan, as [`LoadOptions::max_scan_work`] counts it (saturating).
fn scan_work(body: &ScanBody) -> u64 {
    let graph = body.graph();
    let per_step = graph
        .ids()
        .map(|id| match graph[id].op() {
            Op::Scan(inner) => scan_work(inner),
            _ => u64::try_from(graph[id].shape().iter().product::<usize>())
                .unwrap_or(u64::MAX)
                .max(1),
        })
        .fold(0u64, u64::saturating_add);
    u64::try_from(body.length())
        .unwrap_or(u64::MAX)
        .saturating_mul(per_step)
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

/// The most elements a shape from the file may have: at 8 bytes an element (an `f64` or `i64`), a value must fit in
/// `isize::MAX` bytes, the most one allocation can hold. The graph applies the same limit to every node it creates.
const MAX_ELEMENTS: usize = isize::MAX as usize / 8;

/// A shape from the file: dimensions whose product (the element count) is at most [`MAX_ELEMENTS`].
fn shape_of(dims: &[u64]) -> Result<Vec<usize>> {
    let shape = usizes(dims)?;
    shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .filter(|&n| n <= MAX_ELEMENTS)
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
pub(crate) fn load_model(model: &proto::Model, options: &LoadOptions) -> Result<Loaded> {
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
    let (graph, state) = load_graph(saved, options)?;
    Ok(Loaded {
        graph,
        state,
        metadata: model.metadata.clone(),
    })
}

/// Builds a graph and its state from a saved graph.
fn load_graph(saved: &proto::Graph, options: &LoadOptions) -> Result<(Graph, State)> {
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
        let id = add_node(&builder, op, &parents, options).map_err(|e| match e {
            IoError::Decode(message) => invalid(message),
            other => other,
        })?;
        debug_assert_eq!(id.index(), index);
        let shape = builder.shape(id).unwrap_or_default();
        let elements = shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d));
        if elements.is_none_or(|n| n > options.max_elements) {
            return Err(invalid(format!(
                "the shape {shape:?} has more elements than the limit of {}",
                options.max_elements
            )));
        }
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
    Ok(builder.build()?)
}

/// The body of a saved scan.
fn scan_body(p: &proto::Scan, options: &LoadOptions) -> Result<ScanBody> {
    let saved = p
        .body
        .as_ref()
        .ok_or_else(|| IoError::Decode("a scan has no body".into()))?;
    if !saved.values.is_empty() {
        return Err(IoError::Decode("a scan body has stored values".into()));
    }
    let id = |i: u32| -> Result<NodeId> { Ok(NodeId::from_index(usize_of(u64::from(i))?)) };
    let ids = |v: &[u32]| v.iter().map(|&i| id(i)).collect::<Result<Vec<_>>>();
    let length = usize_of(p.length)?;
    if length > options.max_scan_steps {
        return Err(IoError::Decode(format!(
            "{length} steps is more than the limit of {}",
            options.max_scan_steps
        )));
    }
    let (graph, _) = load_graph(saved, options)?;
    let body = ScanBody::new(
        graph,
        length,
        id(p.carry)?,
        ids(&p.xs)?,
        ids(&p.constants)?,
        id(p.output)?,
    )?;
    let work = scan_work(&body);
    if work > options.max_scan_work {
        return Err(IoError::Decode(format!(
            "the scan's work ({work}: steps times body elements) is more than the limit of {}",
            options.max_scan_work
        )));
    }
    Ok(body)
}

/// The body of a saved ODE.
fn ode_body(p: &proto::Ode, options: &LoadOptions) -> Result<OdeBody> {
    let saved = p
        .body
        .as_ref()
        .ok_or_else(|| IoError::Decode("an ODE has no body".into()))?;
    if !saved.values.is_empty() {
        return Err(IoError::Decode("an ODE body has stored values".into()));
    }
    let id = |i: u32| -> Result<NodeId> { Ok(NodeId::from_index(usize_of(u64::from(i))?)) };
    let parameters = p.parameters.iter().map(|&i| id(i)).collect::<Result<_>>()?;
    let max_steps = usize_of(p.max_steps)?;
    if max_steps > options.max_ode_steps {
        return Err(IoError::Decode(format!(
            "a step limit of {max_steps} is more than the limit of {}",
            options.max_ode_steps
        )));
    }
    let solver_options = OdeOptions::new(decode(p.solver)?)
        .with_tolerances(p.rtol, p.atol)
        .with_max_steps(max_steps);
    let (graph, _) = load_graph(saved, options)?;
    Ok(OdeBody::new(
        graph,
        id(p.time)?,
        id(p.state)?,
        parameters,
        id(p.output)?,
        solver_options,
    )?)
}

/// Adds the node of a saved operation.
fn add_node(
    builder: &GraphBuilder,
    op: &P,
    parents: &[NodeId],
    options: &LoadOptions,
) -> Result<NodeId> {
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
        P::Softmax(p) => Op::Softmax {
            axis: isize_of(p.axis)?,
        },
        P::LogSoftmax(p) => Op::LogSoftmax {
            axis: isize_of(p.axis)?,
        },
        P::Cumulative(p) => Op::Cumulative {
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
        P::SolveTriangular(p) => Op::SolveTriangular {
            lower: p.lower,
            transpose: p.transpose,
        },
        P::Solve(_) => Op::Solve,
        P::LogAbsDeterminant(_) => Op::LogAbsDeterminant,
        P::DeterminantSign(_) => Op::DeterminantSign,
        P::Eigvalsh(_) => Op::Eigvalsh,
        P::EighVectors(_) => Op::EighVectors,
        P::MatrixExp(_) => Op::MatrixExp,
        P::SparseMatmul(p) => Op::SparseMatMul(Arc::new(sparse_matrix(p)?)),
        P::Cast(code) => Op::Cast(decode(*code)?),
        P::Where(_) => Op::Where,
        P::Multiplexer(_) => Op::Multiplexer,
        P::Cpt(p) => {
            // checked here as well as by the graph, so a count from the file never reaches arity arithmetic
            let conditions = usize_of(p.conditions)?;
            if conditions >= parents.len() {
                return Err(IoError::Decode(format!(
                    "a cpt has {conditions} conditions but only {} parents",
                    parents.len()
                )));
            }
            if p.keys.iter().any(|k| k.values.len() != conditions) {
                return Err(IoError::Decode(
                    "every cpt key needs one value per condition".into(),
                ));
            }
            Op::Cpt {
                conditions,
                keys: p.keys.iter().map(|k| k.values.clone()).collect(),
            }
        }
        P::SetWithMask(_) => Op::SetWithMask,
        P::ReplaceNan(p) => Op::ReplaceNan(p.value),
        P::Bessel(p) => Op::Bessel {
            kind: decode(p.kind)?,
            order: p.order,
        },
        P::Assert(p) => Op::Assert(p.message.clone()),
        P::Factor(_) => Op::Factor,
        P::LogDensity(p) => {
            Op::LogDensity(distribution_of_type(p.kind.as_ref().ok_or_else(|| {
                IoError::Decode("a log density has no distribution".into())
            })?)?)
        }
        P::Proxy(p) => {
            let dtype: DType = decode(p.dtype)?;
            let proxy = builder.add_proxy(dtype, &shape_of(&p.shape)?)?;
            match parents {
                [] => {}
                [parent] => builder.connect_ids(proxy, *parent)?,
                _ => return Err(IoError::Decode("a proxy has more than one parent".into())),
            }
            return Ok(proxy);
        }
        P::Scan(p) => {
            Op::Scan(Arc::new(scan_body(p, options).map_err(|e| {
                IoError::Decode(format!("in the scan body: {e}"))
            })?))
        }
        P::Ode(p) => {
            Op::Ode(Arc::new(ode_body(p, options).map_err(|e| {
                IoError::Decode(format!("in the ODE body: {e}"))
            })?))
        }
        P::Distribution(p) => {
            let kind = distribution_of_type(&proto::DistributionType {
                kind: p.kind,
                kde: p.kde.clone(),
                base: p.base.clone().map(Box::new),
                adjacency: p.adjacency.clone(),
            })?;
            return Ok(builder.add_distribution(kind, Some(&shape_of(&p.shape)?), parents)?);
        }
    };
    Ok(builder.add_node(op, parents)?)
}

/// A sparse matrix from its saved triplets.
pub(crate) fn sparse_matrix(p: &proto::SparseMatrix) -> Result<keanu_tensor::SparseMatrix> {
    let (rows, cols) = (usize_of(p.rows)?, usize_of(p.cols)?);
    if p.row_indices.len() != p.values.len() || p.col_indices.len() != p.values.len() {
        return Err(IoError::Decode(format!(
            "a sparse matrix has {} row indices, {} column indices and {} values",
            p.row_indices.len(),
            p.col_indices.len(),
            p.values.len()
        )));
    }
    let triplets = p
        .row_indices
        .iter()
        .zip(&p.col_indices)
        .zip(&p.values)
        .map(|((&i, &j), &v)| Ok((usize_of(i)?, usize_of(j)?, v)))
        .collect::<Result<Vec<_>>>()?;
    keanu_tensor::SparseMatrix::from_triplets(rows, cols, &triplets)
        .map_err(|e| IoError::Decode(format!("invalid sparse matrix: {e}")))
}
