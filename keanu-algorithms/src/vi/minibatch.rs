//! Minibatches for stochastic variational inference: each iteration observes a random subset of a model's data
//! rows, with its log likelihood weighted up to the full data's.

use crate::error::{AlgorithmError, Result};
use keanu_distributions::KeanuRng;
use keanu_graph::op::Op;
use keanu_graph::{Model, NodeId, State, Value, downstream_lambda_section};
use keanu_tensor::IntTensor;

/// One group of data nodes that share their rows: observed nodes and covariates whose leading dimension (axis 0)
/// indexes the same N data rows, of which each iteration uses the same B.
///
/// In the graph every node of the group has leading dimension B (the batch size), and its full data has N rows
/// with the same trailing shape. Each batch sets every node to the same rows of its data, so row i of an
/// observation stays with row i of its covariates. The log probability terms of the group's probabilistic nodes
/// (its observed nodes, and covariates that are observed probabilistic nodes) and of its [factors](Self::factor)
/// are weighted by N / B.
///
/// Weighting is exact in expectation only when each of those terms is a sum of independent terms over the rows,
/// given the latents: a Gaussian, Poisson or Bernoulli observation of shape `[B]`, or a batch of B multivariate
/// events of shape `[B, k]`, but not one multivariate Gaussian (or Gaussian process) over the B rows.
#[derive(Debug, Clone, PartialEq)]
pub struct MinibatchGroup {
    batch_size: usize,
    observed: Vec<(NodeId, Value)>,
    covariates: Vec<(NodeId, Value)>,
    factors: Vec<NodeId>,
}

impl MinibatchGroup {
    /// An empty group whose batches hold `batch_size` rows.
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            observed: Vec::new(),
            covariates: Vec::new(),
            factors: Vec::new(),
        }
    }

    /// Adds an observed probabilistic node, with its full data (N rows along axis 0). Its log probability is
    /// weighted by N / B.
    #[must_use]
    pub fn observed(mut self, node: NodeId, data: impl Into<Value>) -> Self {
        self.observed.push((node, data.into()));
        self
    }

    /// Adds a covariate, with its full data (N rows along axis 0): a constant node, or an observed node. Its
    /// deterministic descendants are recomputed for each batch. An observed probabilistic covariate's log
    /// probability is weighted by N / B, as an observed node's.
    #[must_use]
    pub fn covariate(mut self, node: NodeId, data: impl Into<Value>) -> Self {
        self.covariates.push((node, data.into()));
        self
    }

    /// Adds a factor node whose term sums over the batch's rows (such as a custom log likelihood of the batched
    /// data), to be weighted by N / B. Factors are never weighted implicitly, even when they depend on the batched
    /// data; one that depends on it must be added here.
    #[must_use]
    pub fn factor(mut self, node: NodeId) -> Self {
        self.factors.push(node);
        self
    }

    /// The batch size B.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// The data nodes: observed nodes, then covariates.
    fn data(&self) -> impl Iterator<Item = &(NodeId, Value)> {
        self.observed.iter().chain(&self.covariates)
    }
}

/// The data a minibatch fit subsamples: one or more [`MinibatchGroup`]s, each with its own rows, batch size and
/// random order. Give it to [`Advi::with_minibatch`](super::Advi::with_minibatch); a single group converts into
/// one.
///
/// Within a group, batches are drawn without replacement: the rows are shuffled at the start of each epoch and
/// taken B at a time, and when fewer than B are left the epoch ends (they are skipped) and the rows are shuffled
/// again. Every batch is thus a uniformly random set of B distinct rows, so the weighted log likelihood is an
/// unbiased estimate of the full data's, and when B divides N the batches of an epoch cover every row once and their
/// weighted log likelihoods average to it exactly.
///
/// ```
/// use keanu_algorithms::vi::{Minibatch, MinibatchGroup};
/// use keanu_distributions::KeanuRng;
/// use keanu_graph::{GraphBuilder, Model};
/// use keanu_tensor::DoubleTensor;
///
/// // 1000 rows of (x, y), used 100 at a time
/// let xs = DoubleTensor::vector((0..1000).map(|i| i as f64 / 1000.0).collect());
/// let ys = &xs * 2.0;
/// let b = GraphBuilder::new();
/// let slope = b.gaussian(0.0, 10.0);
/// b.set_value(slope, DoubleTensor::scalar(1.0));
/// let x = b.constant(DoubleTensor::zeros(&[100]));
/// let y = b.gaussian(slope * x, 1.0);
/// b.observe(y, DoubleTensor::zeros(&[100]));
/// let (x, y) = (x.id(), y.id());
/// let (graph, mut state) = b.build()?;
/// let model = Model::new(graph, &state)?;
///
/// let minibatch = Minibatch::from(
///     MinibatchGroup::new(100).observed(y, ys).covariate(x, xs),
/// );
/// let mut batches = minibatch.batches(&model, &state)?;
/// assert_eq!(batches.weights(), &[(y, 10.0)]);
/// batches.next(&mut state, &mut KeanuRng::seed_from_u64(1))?;
/// // the batch's covariates and observations come from the same rows
/// let (x, y) = (state.f64(x).unwrap(), state.f64(y).unwrap());
/// assert!(x.as_slice().iter().zip(y.as_slice()).all(|(x, y)| *y == 2.0 * x));
/// # Ok::<(), keanu_algorithms::AlgorithmError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Minibatch {
    groups: Vec<MinibatchGroup>,
}

impl From<MinibatchGroup> for Minibatch {
    fn from(group: MinibatchGroup) -> Self {
        Self {
            groups: vec![group],
        }
    }
}

impl Minibatch {
    /// A minibatch of these groups.
    pub fn new(groups: impl IntoIterator<Item = MinibatchGroup>) -> Self {
        Self {
            groups: groups.into_iter().collect(),
        }
    }

    /// The groups.
    pub fn groups(&self) -> &[MinibatchGroup] {
        &self.groups
    }

    /// Checks the minibatch against `model` (in `state`) and returns the stream of its batches, which
    /// [`Batches::next`] observes one at a time. [`Advi`](super::Advi) does this itself; use it to drive another
    /// stochastic optimization, or to inspect the batches.
    ///
    /// # Errors
    ///
    /// [`AlgorithmError::InvalidArgument`] if there are no groups; a group's batch size is 0 or more than its N
    /// rows, it has no observed node or factor, or its data tensors disagree on N; a node's shape in the graph
    /// does not lead with B or differs from its data's trailing shape, or its data's type differs from the node's;
    /// a node appears twice; an observed node is not an observed probabilistic node of the model, a covariate is
    /// neither a constant nor observed, or a factor is not a factor of the model; or a latent, or an observed node or
    /// factor outside the group, depends on the group's data (its term would be a batch's but not weighted, or
    /// minibatching would need local latents, which it does not support).
    pub fn batches<'a>(&'a self, model: &'a Model, state: &State) -> Result<Batches<'a>> {
        let invalid = |m: String| Err(AlgorithmError::InvalidArgument(m));
        if self.groups.is_empty() {
            return invalid("a minibatch needs at least one group".into());
        }
        let graph = model.graph();
        let mut seen: Vec<NodeId> = Vec::new();
        let mut streams = Vec::with_capacity(self.groups.len());
        let mut weights = Vec::new();
        let mut original = Vec::new();
        for (g, group) in self.groups.iter().enumerate() {
            let b = group.batch_size;
            if b == 0 {
                return invalid(format!("group {g}: the batch size must be greater than 0"));
            }
            if group.observed.is_empty() && group.factors.is_empty() {
                return invalid(format!(
                    "group {g}: a group needs an observed node or a factor, whose log probability is weighted"
                ));
            }
            let Some((_, first)) = group.data().next() else {
                return invalid(format!(
                    "group {g}: a group needs at least one observed node or covariate with data"
                ));
            };
            let n = first.shape().first().copied().unwrap_or(0);
            for (node, data) in group.data() {
                let node = *node;
                let Some(meta) = graph.node(node) else {
                    return invalid(format!(
                        "group {g}: node #{} is not in the model's graph",
                        node.index()
                    ));
                };
                let name = graph.describe(node);
                let shape = meta.shape();
                if data.shape().first() != Some(&n) {
                    return invalid(format!(
                        "group {g}: the data of {name} has shape {:?}, but the group's data has {n} rows",
                        data.shape()
                    ));
                }
                if shape.first() != Some(&b) || shape[1..] != data.shape()[1..] {
                    return invalid(format!(
                        "group {g}: {name} has shape {shape:?} in the graph, but a batch of {b} rows of its data \
                         would have shape {:?}",
                        std::iter::once(b)
                            .chain(data.shape()[1..].iter().copied())
                            .collect::<Vec<_>>()
                    ));
                }
                if data.dtype() != meta.dtype() {
                    return invalid(format!(
                        "group {g}: {name} holds {:?} values, but its data holds {:?}",
                        meta.dtype(),
                        data.dtype()
                    ));
                }
            }
            if b > n {
                return invalid(format!(
                    "group {g}: the batch size {b} is more than the {n} data rows"
                ));
            }
            let weight = n as f64 / b as f64;
            for &(node, _) in &group.observed {
                if !(graph[node].is_probabilistic() && model.observed().contains(&node)) {
                    return invalid(format!(
                        "group {g}: {} is not an observed probabilistic node of the model; observe it (with any \
                         batch of its data) before building the model",
                        graph.describe(node)
                    ));
                }
                weights.push((node, weight));
            }
            for &(node, _) in &group.covariates {
                let constant = matches!(graph[node].op(), Op::Constant(_));
                if !(constant || model.observed().contains(&node)) {
                    return invalid(format!(
                        "group {g}: the covariate {} is neither a constant nor an observed node",
                        graph.describe(node)
                    ));
                }
                if graph[node].is_probabilistic() {
                    weights.push((node, weight));
                }
            }
            for &node in &group.factors {
                if !model.factors().contains(&node) {
                    return invalid(format!(
                        "group {g}: {} is not a factor of the model",
                        graph.describe(node)
                    ));
                }
                weights.push((node, weight));
            }
            let members: Vec<NodeId> = group
                .data()
                .map(|(node, _)| *node)
                .chain(group.factors.iter().copied())
                .collect();
            for &node in &members {
                if seen.contains(&node) {
                    return invalid(format!(
                        "{} appears more than once in the minibatch",
                        graph.describe(node)
                    ));
                }
                seen.push(node);
            }
            let sources: Vec<NodeId> = group.data().map(|(node, _)| *node).collect();
            let section = downstream_lambda_section(graph, state, &sources)?;
            for &node in section.latent_and_observed() {
                if members.contains(&node) {
                    continue;
                }
                let name = graph.describe(node);
                let problem = if model.latents().contains(&node) {
                    "it is a latent; minibatching supports only latents shared by every row (local latents \
                     would need a value for every row)"
                } else if graph[node].is_factor() {
                    "it is a factor outside the group, so its term would not be weighted; add it with \
                     MinibatchGroup::factor"
                } else {
                    "it is not in the group, so its term would not be weighted; add it as an observed node \
                     or covariate"
                };
                return invalid(format!(
                    "group {g}: {name} depends on the group's batched data, but {problem}"
                ));
            }
            for &node in &sources {
                let value = state
                    .value(node)
                    .cloned()
                    .ok_or(keanu_graph::GraphError::MissingValue(node))?;
                original.push((node, value));
            }
            streams.push(Stream {
                total: n,
                order: (0..n).collect(),
                // the first batch starts an epoch
                position: n,
                rows: Vec::new(),
            });
        }
        Ok(Batches {
            minibatch: self,
            model,
            streams,
            weights,
            original,
        })
    }
}

/// A group's place in its random order.
#[derive(Debug, Clone)]
struct Stream {
    total: usize,
    order: Vec<usize>,
    position: usize,
    rows: Vec<usize>,
}

/// The stream of a [`Minibatch`]'s batches, checked against a model: see [`Minibatch::batches`].
#[derive(Debug, Clone)]
pub struct Batches<'a> {
    minibatch: &'a Minibatch,
    model: &'a Model,
    streams: Vec<Stream>,
    weights: Vec<(NodeId, f64)>,
    /// The batched nodes' values before the first batch.
    original: Vec<(NodeId, Value)>,
}

impl Batches<'_> {
    /// The weight N / B of each weighted node (observed nodes, observed probabilistic covariates and factors), in
    /// group order: give them to [`Parameterisation::with_weights`](keanu_graph::Parameterisation::with_weights).
    pub fn weights(&self) -> &[(NodeId, f64)] {
        &self.weights
    }

    /// The number of batches in an epoch of group `group`: ⌊N / B⌋. Panics if there is no such group.
    pub fn batches_per_epoch(&self, group: usize) -> usize {
        self.streams[group].total / self.minibatch.groups[group].batch_size
    }

    /// The data rows of group `group`'s current batch, in the order they are laid out along the nodes' leading
    /// dimension (empty before the first batch). Panics if there is no such group.
    pub fn rows(&self, group: usize) -> &[usize] {
        &self.streams[group].rows
    }

    /// Observes the next batch of every group: sets each node of the group to the batch's rows of its data
    /// (observing observed nodes, setting constants) and recomputes their deterministic descendants. Shuffles a
    /// group's rows when it starts an epoch.
    ///
    /// # Errors
    ///
    /// Any error recomputing the descendants.
    pub fn next(&mut self, state: &mut State, rng: &mut KeanuRng) -> Result<()> {
        for (stream, group) in self.streams.iter_mut().zip(&self.minibatch.groups) {
            let b = group.batch_size;
            if stream.position + b > stream.total {
                // Fisher–Yates
                for i in (1..stream.total).rev() {
                    let j = rng.next_below(i as u64 + 1) as usize;
                    stream.order.swap(i, j);
                }
                stream.position = 0;
            }
            stream.rows = stream.order[stream.position..stream.position + b].to_vec();
            stream.position += b;
        }
        let rows: Vec<Vec<usize>> = self.streams.iter().map(|s| s.rows.clone()).collect();
        self.observe_rows(state, &rows)
    }

    /// Observes the given rows of each group (as [`next`](Self::next) observes a batch), without moving the
    /// streams.
    pub(crate) fn observe_rows(&self, state: &mut State, rows: &[Vec<usize>]) -> Result<()> {
        let mut changed = Vec::new();
        for (group, rows) in self.minibatch.groups.iter().zip(rows) {
            let indices = IntTensor::vector(rows.iter().map(|&r| r as i64).collect());
            for (node, data) in group.data() {
                let batch = gather_rows(data, &indices)?;
                set_data(self.model, state, *node, batch)?;
                changed.push(*node);
            }
        }
        state.cascade(self.model.graph(), &changed)?;
        Ok(())
    }

    /// The current batch's rows of every group.
    pub(crate) fn current_rows(&self) -> Vec<Vec<usize>> {
        self.streams.iter().map(|s| s.rows.clone()).collect()
    }

    /// Restores the batched nodes' values from before the first batch, and recomputes their deterministic
    /// descendants.
    ///
    /// # Errors
    ///
    /// Any error recomputing the descendants.
    pub fn restore(&self, state: &mut State) -> Result<()> {
        let mut changed = Vec::with_capacity(self.original.len());
        for (node, value) in &self.original {
            set_data(self.model, state, *node, value.clone())?;
            changed.push(*node);
        }
        state.cascade(self.model.graph(), &changed)?;
        Ok(())
    }
}

/// Observes an observed node's new data, or sets a constant's.
fn set_data(model: &Model, state: &mut State, node: NodeId, value: Value) -> Result<()> {
    let graph = model.graph();
    if state.is_observed(node) {
        state.observe(graph, node, value)?;
    } else {
        state.set(graph, node, value)?;
    }
    Ok(())
}

/// The rows `indices` of a tensor, along axis 0.
fn gather_rows(data: &Value, indices: &IntTensor) -> Result<Value> {
    Ok(match data {
        Value::F64(t) => Value::F64(t.gather(0, indices)?),
        Value::I64(t) => Value::I64(t.gather(0, indices)?),
        Value::Bool(t) => Value::Bool(t.gather(0, indices)?),
        other => {
            return Err(AlgorithmError::InvalidArgument(format!(
                "minibatch data of type {:?} is not supported",
                other.dtype()
            )));
        }
    })
}
