//! Missing observations: a graph rewrite that observes the known elements of an element-wise probabilistic node and
//! turns its missing elements into latents, which inference then imputes (as PyMC does for masked observations).
//!
//! For a node `y` of shape S with missing elements M and observed elements O (flat, row-major positions), the rewrite
//! [`observe_masked`] replaces `y` by three nodes:
//!
//! - `<label>_observed`, the same distribution with each parameter gathered at O, observed with the known values
//!   (shape [|O|]);
//! - `<label>_missing`, the same distribution with each parameter gathered at M, a latent (shape [|M|]);
//! - `<label>`, a deterministic node of shape S that puts each element back in its place, which every node that used
//!   `y` now reads.
//!
//! The joint density of the observed and missing parts is the density of `y`, so the model is unchanged; the
//! missing elements are sampled with everything else (NUTS for continuous ones, Gibbs steps for discrete ones), and
//! draws of `<label>` hold the observed values and the imputed ones in their original positions.

use crate::error::{GraphError, Result};
use crate::graph::{Graph, Node, NodeId, Parents};
use crate::op::log_density::has_elementwise_density;
use crate::op::{DistributionKind, DistributionOp, Op};
use crate::rewrite::{Writer, check_distinct, check_state, fill_values, mapped, stale};
use crate::state::State;
use crate::value::{DType, Value};
use keanu_distributions::KeanuRng;
use keanu_tensor::{BoolTensor, DoubleTensor, IntTensor, Shape};

/// An observation of a probabilistic node with some elements missing, for [`observe_masked`].
#[derive(Debug, Clone, PartialEq)]
pub struct MaskedObservation {
    node: NodeId,
    values: Value,
    missing: BoolTensor,
}

impl MaskedObservation {
    /// Observes `node` with `values`, except where `missing` is true: those elements are imputed. `values` and
    /// `missing` have the node's shape; the values at the missing elements are ignored.
    pub fn new(node: NodeId, values: impl Into<Value>, missing: BoolTensor) -> Self {
        Self {
            node,
            values: values.into(),
            missing,
        }
    }

    /// Observes a continuous `node` with `values`, whose NaN elements are missing.
    pub fn with_nan_missing(node: NodeId, values: DoubleTensor) -> Self {
        let missing = values.as_slice().iter().map(|v| v.is_nan()).collect();
        let missing =
            BoolTensor::new(values.shape(), missing).expect("the mask has the values' shape");
        Self::new(node, values, missing)
    }

    /// The node observed.
    pub fn node(&self) -> NodeId {
        self.node
    }
}

/// The record of an [`observe_masked`] rewrite: which node of the new graph each node of the original graph became,
/// and the parts of each masked node.
#[derive(Debug, Clone, PartialEq)]
pub struct Imputed {
    /// For each node of the original graph, its node in the new graph.
    mapping: Vec<Option<NodeId>>,
    nodes: Vec<ImputedNode>,
}

impl Imputed {
    /// The node of the new graph that a node of the original graph became: for a masked node with missing elements
    /// the deterministic node that recombines its parts (with its label and shape), otherwise the same operation on
    /// the corresponding parents, with the same label. `None` for ids that aren't nodes of the original graph.
    pub fn node(&self, original: NodeId) -> Option<NodeId> {
        self.mapping.get(original.index()).copied().flatten()
    }

    /// The masked nodes, in the order they were given.
    pub fn nodes(&self) -> &[ImputedNode] {
        &self.nodes
    }

    /// The record of a masked node, by its id in the original graph.
    pub fn get(&self, original: NodeId) -> Option<&ImputedNode> {
        self.nodes.iter().find(|n| n.original == original)
    }
}

/// One masked node: where its missing and observed elements are and which nodes of the new graph hold them.
#[derive(Debug, Clone, PartialEq)]
pub struct ImputedNode {
    original: NodeId,
    label: Option<String>,
    shape: Shape,
    node: NodeId,
    observed: Option<NodeId>,
    missing: Option<NodeId>,
    missing_indices: Vec<usize>,
    observed_indices: Vec<usize>,
}

impl ImputedNode {
    /// The node's id in the original graph.
    pub fn original(&self) -> NodeId {
        self.original
    }

    /// The node's label in the original graph, which the recombined node keeps.
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// The node's shape, S.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The node of the new graph holding the whole value, of shape S, with the observed values and the imputed ones
    /// in their original positions: record it while sampling to read the imputed draws in place. It is the
    /// recombining deterministic node when elements are missing, and the observed node itself when none are.
    pub fn node(&self) -> NodeId {
        self.node
    }

    /// The observed part (`<label>_observed`), shape [|O|], or `None` if every element is missing. When no element
    /// is missing it is the node itself, of shape S.
    pub fn observed(&self) -> Option<NodeId> {
        self.observed
    }

    /// The latent holding the missing elements (`<label>_missing`), shape [|M|], or `None` if no element is
    /// missing. Its element k is the missing element at flat position [`missing_indices`](Self::missing_indices)`[k]`.
    pub fn missing(&self) -> Option<NodeId> {
        self.missing
    }

    /// The flat (row-major) positions of the missing elements in the node's shape, in increasing order.
    pub fn missing_indices(&self) -> &[usize] {
        &self.missing_indices
    }

    /// The flat (row-major) positions of the observed elements, in increasing order: element j of the observed part
    /// is at position `observed_indices()[j]`.
    pub fn observed_indices(&self) -> &[usize] {
        &self.observed_indices
    }
}

/// Whether masked observations of `kind` can be split element by element: it has one value per element with an
/// element-wise density, and each parameter broadcasts to the value's shape element by element.
fn splits_elementwise(kind: &DistributionKind) -> bool {
    match kind {
        // the heterogeneous mixture's weights have a trailing component axis
        DistributionKind::MixtureOf(_) => false,
        DistributionKind::Categorical | DistributionKind::CategoricalLogit => true,
        k => has_elementwise_density(k),
    }
}

/// Observes nodes with some elements missing: returns a new graph and state in which, for each masked observation,
/// the known elements are observed and the missing ones are a latent to impute, and the [`Imputed`] record of how
/// the nodes correspond.
///
/// A masked node must be probabilistic with an element-wise distribution (the continuous and count distributions,
/// truncated, censored, zero-inflated, hurdle and transformed ones, the Bernoulli and the categorical, and their
/// logit forms), each of whose parameters broadcasts to the node's shape. It may be an outcome or a covariate given
/// a distribution: the nodes that used it read the recombined node, which has its label and shape. A node of shape S
/// with missing elements becomes three:
///
/// - `<label>_observed`, the same distribution with each parameter gathered at the observed elements, observed with
///   the known values (shape [|O|]);
/// - `<label>_missing`, the same distribution with each parameter gathered at the missing elements, a latent (shape
///   [|M|]);
/// - `<label>`, a deterministic node of shape S that puts each element back in its place.
///
/// The parts are unlabelled if the node is. A node with every element missing has no observed part; one with none missing is simply observed, and
/// keeps its node. Multivariate distributions (whose elements' densities aren't separate) can't be split, so a
/// missing component of a multivariate Gaussian, say, can't be imputed this way.
///
/// Every other node is copied with its label, value and observation; the values of deterministic nodes downstream
/// of a masked node are recomputed. Each missing part starts at a draw from its distribution given its parameters
/// (drawing any latent they depend on that has no value), so the new state is ready to sample.
///
/// The joint log probability is the original model's with the known elements observed, and the pointwise log
/// likelihood (keanu-algorithms' `PointwiseLogLik`) has a point for each observed element only.
///
/// ```
/// use keanu_graph::{GraphBuilder, MaskedObservation, Model, observe_masked};
/// use keanu_tensor::DoubleTensor;
///
/// let b = GraphBuilder::new();
/// let mu = b.gaussian(0.0, 10.0).label("mu");
/// let y = b.gaussian_with_shape(&[4], mu, 1.0).label("y");
/// let total = y.sum().label("total");
/// let (mu, y, total) = (mu.id(), y.id(), total.id());
/// let (graph, state) = b.build()?;
///
/// let values = DoubleTensor::vector(vec![1.0, f64::NAN, 3.0, f64::NAN]);
/// let (graph, state, imputed) =
///     observe_masked(&graph, &state, &[MaskedObservation::with_nan_missing(y, values)])?;
/// let y = imputed.get(y).unwrap();
/// assert_eq!(y.missing_indices(), &[1, 3]);
/// assert_eq!(graph.find("y_missing"), y.missing());
/// assert_eq!(graph[y.missing().unwrap()].shape(), &[2]);
/// // `y` is the recombined node, which `total` reads
/// assert_eq!(graph.find("y"), Some(y.node()));
/// assert_eq!(graph[imputed.node(total).unwrap()].parents(), &[y.node()]);
/// let model = Model::new(graph, &state)?;
/// assert_eq!(model.latents().len(), 2); // mu and y_missing
/// # Ok::<(), keanu_graph::GraphError>(())
/// ```
///
/// # Errors
///
/// [`GraphError::CannotImpute`] for a node that isn't probabilistic or whose distribution isn't element-wise (or has
/// a parameter that doesn't broadcast to the node's shape), [`GraphError::UnknownNode`] for an id that isn't a node of
/// the graph, [`GraphError::InvalidValue`] if `observations` is empty or repeats a node, values or a mask don't have
/// the node's type and shape, or the state isn't the graph's, [`GraphError::DuplicateLabel`] if a part's label is
/// already used, and any error drawing the missing parts' initial values.
pub fn observe_masked(
    graph: &Graph,
    state: &State,
    observations: &[MaskedObservation],
) -> Result<(Graph, State, Imputed)> {
    let ids: Vec<NodeId> = observations.iter().map(|o| o.node).collect();
    graph.check_ids(&ids)?;
    check_state(graph, state)?;
    if observations.is_empty() {
        return Err(GraphError::InvalidValue(
            "there are no masked observations".into(),
        ));
    }
    check_distinct(graph, &ids)?;
    let n = graph.len();
    let mut masked: Vec<Option<&MaskedObservation>> = vec![None; n];
    for observation in observations {
        check(graph, observation)?;
        masked[observation.node.0] = Some(observation);
    }
    let changed: Vec<bool> = masked.iter().map(Option::is_some).collect();
    let stale = stale(graph, state, &changed);

    let mut out = Writer::default();
    let mut mapping: Vec<Option<NodeId>> = vec![None; n];
    let mut records: Vec<ImputedNode> = Vec::with_capacity(observations.len());
    for id in graph.ids() {
        let node = &graph[id];
        let parents = node
            .parents
            .iter()
            .map(|&p| mapped(&mapping, p))
            .collect::<Result<Parents>>()?;
        let Some(observation) = masked[id.0] else {
            let value = if stale[id.0] {
                None
            } else {
                state.value(id).cloned()
            };
            mapping[id.0] = Some(out.copy(node, parents, value, state.is_observed(id)));
            continue;
        };
        let record = split(&mut out, node, id, &parents, observation)?;
        mapping[id.0] = Some(record.node);
        records.push(record);
    }
    let (graph, mut state) = out.finish()?;

    // start each missing part at a draw from its distribution, and compute what depends on it
    let missing: Vec<NodeId> = records.iter().filter_map(ImputedNode::missing).collect();
    if !missing.is_empty() {
        state.lazy_eval(&graph, &missing, &mut KeanuRng::seed_from_u64(0))?;
        fill_values(&graph.nodes, &mut state.values);
    }
    // the records by the order the observations were given
    records.sort_by_key(|r| ids.iter().position(|&id| id == r.original));
    Ok((
        graph,
        state,
        Imputed {
            mapping,
            nodes: records,
        },
    ))
}

/// An error unless the observation is of a probabilistic node with an element-wise distribution whose parameters
/// broadcast to its shape, with values and a mask of its type and shape.
fn check(graph: &Graph, observation: &MaskedObservation) -> Result<()> {
    let id = observation.node;
    let node = &graph[id];
    let reject = |reason: String| GraphError::CannotImpute {
        node: graph.describe(id),
        reason,
    };
    let Op::Distribution(distribution) = &node.op else {
        return Err(reject(format!(
            "is a deterministic node ({}); only probabilistic nodes can be observed with missing elements",
            node.op.name()
        )));
    };
    if !splits_elementwise(&distribution.kind) {
        return Err(reject(format!(
            "has a {} distribution, which doesn't have one value per element with an element-wise density, so its \
             missing elements can't be split from its observed ones",
            distribution.kind.name()
        )));
    }
    let shape = node.shape();
    for &p in &node.parents {
        let parameter = graph[p].shape();
        if keanu_tensor::broadcast_shape(parameter, shape)
            .ok()
            .as_deref()
            != Some(shape)
        {
            return Err(reject(format!(
                "has a parameter {} of shape {parameter:?}, which doesn't broadcast to its shape {shape:?}",
                graph.describe(p)
            )));
        }
    }
    let values = &observation.values;
    if values.dtype() != node.dtype() || values.shape() != shape {
        return Err(GraphError::InvalidValue(format!(
            "{} holds {:?} of shape {shape:?}, but the observed values are {:?} of shape {:?}",
            graph.describe(id),
            node.dtype(),
            values.dtype(),
            values.shape()
        )));
    }
    if observation.missing.shape() != shape {
        return Err(GraphError::InvalidValue(format!(
            "{} has shape {shape:?}, but the mask of missing elements has shape {:?}",
            graph.describe(id),
            observation.missing.shape()
        )));
    }
    Ok(())
}

/// Writes a masked node: observed when nothing is missing, else its observed part, missing part and recombination.
fn split(
    out: &mut Writer,
    node: &Node,
    original: NodeId,
    parents: &Parents,
    observation: &MaskedObservation,
) -> Result<ImputedNode> {
    let Op::Distribution(distribution) = &node.op else {
        unreachable!("checked to be probabilistic")
    };
    let kind = &distribution.kind;
    let shape: &[usize] = &node.shape;
    let (mut missing_indices, mut observed_indices) = (Vec::new(), Vec::new());
    for (i, &m) in observation.missing.as_slice().iter().enumerate() {
        if m {
            missing_indices.push(i);
        } else {
            observed_indices.push(i);
        }
    }
    let label = node.label.clone();
    let mut record = ImputedNode {
        original,
        label: label.clone(),
        shape: node.shape.clone(),
        node: NodeId(0),
        observed: None,
        missing: None,
        missing_indices,
        observed_indices,
    };
    if record.missing_indices.is_empty() {
        let id = out.copy(
            node,
            parents.clone(),
            Some(observation.values.clone()),
            true,
        );
        record.node = id;
        record.observed = Some(id);
        return Ok(record);
    }
    let suffixed = |suffix: &str| label.as_ref().map(|l| format!("{l}_{suffix}"));
    let n: usize = shape.iter().product();

    // each part: the distribution with every non-scalar parameter broadcast to S, flattened and gathered
    let part = |out: &mut Writer, indices: &[usize], label: Option<String>| -> Result<NodeId> {
        let index = out.constant(Value::I64(IntTensor::vector(as_i64(indices))), None);
        let mut gathered = Parents::new();
        for &p in parents {
            let mut q = p;
            if out.nodes[q.0].shape.is_empty() {
                gathered.push(q);
                continue;
            }
            if out.nodes[q.0].shape.as_slice() != shape {
                q = out.add(Op::Broadcast(shape.to_vec()), &[q])?;
            }
            if shape.len() != 1 {
                q = out.add(Op::Reshape(vec![n as isize]), &[q])?;
            }
            gathered.push(out.add(Op::Gather { axis: 0 }, &[q, index])?);
        }
        let types: Vec<(DType, &[usize])> = gathered
            .iter()
            .map(|q| (out.nodes[q.0].dtype, out.nodes[q.0].shape.as_slice()))
            .collect();
        let part_shape = DistributionOp::value_shape(kind, Some(&[indices.len()]), &types)?;
        Ok(out.push(
            Node {
                op: Op::Distribution(DistributionOp {
                    kind: kind.clone(),
                    shape: part_shape.clone(),
                }),
                parents: gathered,
                dtype: node.dtype,
                shape: part_shape,
                label,
            },
            None,
            false,
        ))
    };

    let missing = part(out, &record.missing_indices, suffixed("missing"))?;
    record.missing = Some(missing);
    let flat = if record.observed_indices.is_empty() {
        missing
    } else {
        let observed = part(out, &record.observed_indices, suffixed("observed"))?;
        out.values[observed.0] = Some(select(&observation.values, &record.observed_indices)?);
        out.observed[observed.0] = true;
        record.observed = Some(observed);
        // the missing elements then the observed ones, each picked out for its original position
        let joined = out.add(Op::Concat(0), &[missing, observed])?;
        let (mut m, mut o) = (0, record.missing_indices.len());
        let order: Vec<usize> = observation
            .missing
            .as_slice()
            .iter()
            .map(|&is_missing| {
                let slot = if is_missing { &mut m } else { &mut o };
                *slot += 1;
                *slot - 1
            })
            .collect();
        let order = out.constant(Value::I64(IntTensor::vector(as_i64(&order))), None);
        out.add(Op::Gather { axis: 0 }, &[joined, order])?
    };
    let whole = out.add(
        Op::Reshape(shape.iter().map(|&d| d as isize).collect()),
        &[flat],
    )?;
    out.nodes[whole.0].label = label;
    record.node = whole;
    Ok(record)
}

fn as_i64(indices: &[usize]) -> Vec<i64> {
    indices.iter().map(|&i| i as i64).collect()
}

/// The elements of `value` at flat positions `indices`, as a vector.
fn select(value: &Value, indices: &[usize]) -> Result<Value> {
    fn pick<T: Copy>(data: &[T], indices: &[usize]) -> Vec<T> {
        indices.iter().map(|&i| data[i]).collect()
    }
    let shape = [indices.len()];
    Ok(match value {
        Value::F64(t) => Value::F64(DoubleTensor::new(&shape, pick(t.as_slice(), indices))?),
        Value::I64(t) => Value::I64(IntTensor::new(&shape, pick(t.as_slice(), indices))?),
        Value::Bool(t) => Value::Bool(BoolTensor::new(&shape, pick(t.as_slice(), indices))?),
    })
}
