//! Samples of node values, as produced by samplers.

use crate::diagnostics::{self, ElementSummary};
use crate::error::{AlgorithmError, Result};
use crate::statistics;
use keanu_graph::{DType, Graph, GraphError, NodeId, State, Value};
use keanu_tensor::{BoolTensor, DoubleTensor, Element, IntTensor, Tensor};

/// The samples of one node.
#[derive(Debug, Clone, PartialEq)]
enum Column {
    F64(Vec<DoubleTensor>),
    I64(Vec<IntTensor>),
    Bool(Vec<BoolTensor>),
}

impl Column {
    fn new(dtype: DType) -> Self {
        match dtype {
            DType::F64 => Self::F64(Vec::new()),
            DType::I64 => Self::I64(Vec::new()),
            DType::Bool => Self::Bool(Vec::new()),
            // a dtype added later: an empty column that rejects every value, so `record` reports an error
            _ => Self::F64(Vec::new()),
        }
    }

    fn dtype(&self) -> DType {
        match self {
            Self::F64(_) => DType::F64,
            Self::I64(_) => DType::I64,
            Self::Bool(_) => DType::Bool,
        }
    }

    fn accepts(&self, value: &Value) -> bool {
        matches!(
            (self, value),
            (Self::F64(_), Value::F64(_))
                | (Self::I64(_), Value::I64(_))
                | (Self::Bool(_), Value::Bool(_))
        )
    }

    fn push(&mut self, value: &Value) -> bool {
        match (self, value) {
            (Self::F64(v), Value::F64(t)) => v.push(t.clone()),
            (Self::I64(v), Value::I64(t)) => v.push(t.clone()),
            (Self::Bool(v), Value::Bool(t)) => v.push(t.clone()),
            _ => return false,
        }
        true
    }

    fn get(&self, index: usize) -> Value {
        match self {
            Self::F64(v) => Value::F64(v[index].clone()),
            Self::I64(v) => Value::I64(v[index].clone()),
            Self::Bool(v) => Value::Bool(v[index].clone()),
        }
    }

    fn select(&self, indices: &[usize]) -> Self {
        fn pick<T: Clone>(v: &[T], indices: &[usize]) -> Vec<T> {
            indices.iter().map(|&i| v[i].clone()).collect()
        }
        match self {
            Self::F64(v) => Self::F64(pick(v, indices)),
            Self::I64(v) => Self::I64(pick(v, indices)),
            Self::Bool(v) => Self::Bool(pick(v, indices)),
        }
    }
}

/// Samples of a set of nodes, with the log probability recorded for each sample.
#[derive(Debug, Clone, PartialEq)]
pub struct NetworkSamples {
    nodes: Vec<NodeId>,
    columns: Vec<Column>,
    log_probs: Vec<f64>,
}

impl NetworkSamples {
    /// An empty set of samples of the given nodes. Fails with [`GraphError::UnknownNode`] if a node is not in the
    /// graph.
    pub fn new(graph: &Graph, nodes: &[NodeId]) -> Result<Self> {
        Ok(Self {
            nodes: nodes.to_vec(),
            columns: nodes
                .iter()
                .map(|&id| {
                    let node = graph.node(id).ok_or(GraphError::UnknownNode(id))?;
                    Ok(Column::new(node.dtype()))
                })
                .collect::<Result<_>>()?,
            log_probs: Vec::new(),
        })
    }

    /// Records the current values of the nodes as a new sample.
    pub fn record(&mut self, state: &State, log_prob: f64) -> Result<()> {
        let values: Vec<&Value> = self
            .nodes
            .iter()
            .map(|&id| {
                state
                    .value(id)
                    .ok_or(AlgorithmError::Graph(GraphError::MissingValue(id)))
            })
            .collect::<Result<_>>()?;
        // check every value before pushing any, so a mismatch leaves the samples unchanged
        if !self.columns.iter().zip(&values).all(|(c, v)| c.accepts(v)) {
            return Err(AlgorithmError::InvalidArgument(
                "a value's type doesn't match its node".into(),
            ));
        }
        for (column, value) in self.columns.iter_mut().zip(values) {
            let pushed = column.push(value);
            debug_assert!(pushed, "checked by accepts");
        }
        self.log_probs.push(log_prob);
        Ok(())
    }

    /// The number of samples.
    pub fn len(&self) -> usize {
        self.log_probs.len()
    }

    /// Whether there are no elements.
    pub fn is_empty(&self) -> bool {
        self.log_probs.is_empty()
    }

    /// The nodes that were sampled.
    pub fn nodes(&self) -> &[NodeId] {
        &self.nodes
    }

    /// The log probability recorded with each sample.
    pub fn log_probs(&self) -> &[f64] {
        &self.log_probs
    }

    fn column(&self, id: NodeId) -> Option<&Column> {
        self.nodes
            .iter()
            .position(|&n| n == id)
            .map(|i| &self.columns[i])
    }

    /// The samples of an f64 node.
    pub fn f64(&self, id: NodeId) -> Option<Samples<'_, f64>> {
        match self.column(id)? {
            Column::F64(v) => Some(Samples { values: v }),
            _ => None,
        }
    }

    /// The samples of an i64 node.
    pub fn i64(&self, id: NodeId) -> Option<Samples<'_, i64>> {
        match self.column(id)? {
            Column::I64(v) => Some(Samples { values: v }),
            _ => None,
        }
    }

    /// The samples of a bool node.
    pub fn bool(&self, id: NodeId) -> Option<Samples<'_, bool>> {
        match self.column(id)? {
            Column::Bool(v) => Some(Samples { values: v }),
            _ => None,
        }
    }

    /// The samples of the f64 node with this label.
    pub fn f64_by_label(&self, graph: &Graph, label: &str) -> Option<Samples<'_, f64>> {
        self.f64(graph.find(label)?)
    }

    fn select(&self, indices: &[usize]) -> Self {
        Self {
            nodes: self.nodes.clone(),
            columns: self.columns.iter().map(|c| c.select(indices)).collect(),
            log_probs: indices.iter().map(|&i| self.log_probs[i]).collect(),
        }
    }

    /// The samples without the first `count` (e.g. burn-in). Fails if there are fewer than `count` samples.
    pub fn drop(&self, count: usize) -> Result<Self> {
        if count > self.len() {
            return Err(AlgorithmError::InvalidArgument(format!(
                "cannot drop {count} of {} samples",
                self.len()
            )));
        }
        Ok(self.select(&(count..self.len()).collect::<Vec<_>>()))
    }

    /// Every `interval`-th sample, starting with the first, so ceil(len / interval) samples are kept.
    pub fn down_sample(&self, interval: usize) -> Result<Self> {
        if interval == 0 {
            return Err(AlgorithmError::InvalidArgument(
                "the down sample interval must be positive".into(),
            ));
        }
        Ok(self.select(&(0..self.len()).step_by(interval).collect::<Vec<_>>()))
    }

    /// The samples carried over to another graph, such as one a graph rewrite made from this one
    /// ([`intervene`](keanu_graph::intervene), [`observe_masked`](keanu_graph::observe_masked) or
    /// [`marginalize`](keanu_graph::marginalize)): each sampled node that `map` sends to a node of `graph` keeps its
    /// samples under the new id, and the others are left out. The log probabilities are kept as recorded (they are the
    /// original model's).
    ///
    /// # Errors
    ///
    /// [`GraphError::UnknownNode`] if `map` gives an id that isn't a node of `graph`, and
    /// [`AlgorithmError::InvalidArgument`] if it gives one whose type differs from the sampled node's or gives the same
    /// node for two sampled nodes.
    pub fn map_nodes(&self, graph: &Graph, map: impl Fn(NodeId) -> Option<NodeId>) -> Result<Self> {
        let mut nodes = Vec::new();
        let mut columns = Vec::new();
        for (&id, column) in self.nodes.iter().zip(&self.columns) {
            let Some(new) = map(id) else { continue };
            let node = graph.node(new).ok_or(GraphError::UnknownNode(new))?;
            if node.dtype() != column.dtype() {
                return Err(AlgorithmError::InvalidArgument(format!(
                    "node #{} is mapped to {}, which holds values of another type",
                    id.index(),
                    graph.describe(new)
                )));
            }
            if nodes.contains(&new) {
                return Err(AlgorithmError::InvalidArgument(format!(
                    "two sampled nodes are mapped to {}",
                    graph.describe(new)
                )));
            }
            nodes.push(new);
            columns.push(column.clone());
        }
        Ok(Self {
            nodes,
            columns,
            log_probs: self.log_probs.clone(),
        })
    }

    /// One sample, as a view of each node's value.
    pub fn sample(&self, index: usize) -> Option<NetworkSample<'_>> {
        (index < self.len()).then_some(NetworkSample {
            samples: self,
            index,
        })
    }

    /// Every sample in order.
    pub fn iter(&self) -> impl Iterator<Item = NetworkSample<'_>> + '_ {
        (0..self.len()).map(move |index| NetworkSample {
            samples: self,
            index,
        })
    }

    /// The fraction of samples for which the predicate holds.
    pub fn probability(&self, predicate: impl Fn(&NetworkSample<'_>) -> bool) -> f64 {
        self.iter().filter(|s| predicate(s)).count() as f64 / self.len() as f64
    }

    /// The sample with the highest log probability (the first, if there are ties).
    pub fn most_probable(&self) -> Option<NetworkSample<'_>> {
        let mut best: Option<usize> = None;
        for (i, &lp) in self.log_probs.iter().enumerate() {
            if best.is_none_or(|b| lp > self.log_probs[b]) {
                best = Some(i);
            }
        }
        best.map(|index| NetworkSample {
            samples: self,
            index,
        })
    }
}

/// One sample of every sampled node.
#[derive(Debug, Clone, Copy)]
pub struct NetworkSample<'a> {
    samples: &'a NetworkSamples,
    index: usize,
}

impl NetworkSample<'_> {
    /// The node's value in this sample.
    pub fn get(&self, id: NodeId) -> Option<Value> {
        self.samples.column(id).map(|c| c.get(self.index))
    }

    /// The f64 value, if there is one.
    pub fn f64(&self, id: NodeId) -> Option<&DoubleTensor> {
        match self.samples.column(id)? {
            Column::F64(v) => Some(&v[self.index]),
            _ => None,
        }
    }

    /// The i64 value, if there is one.
    pub fn i64(&self, id: NodeId) -> Option<&IntTensor> {
        match self.samples.column(id)? {
            Column::I64(v) => Some(&v[self.index]),
            _ => None,
        }
    }

    /// The bool value, if there is one.
    pub fn bool(&self, id: NodeId) -> Option<&BoolTensor> {
        match self.samples.column(id)? {
            Column::Bool(v) => Some(&v[self.index]),
            _ => None,
        }
    }

    /// The log probability recorded with this sample.
    pub fn log_prob(&self) -> f64 {
        self.samples.log_probs[self.index]
    }

    /// The index of this sample.
    pub fn index(&self) -> usize {
        self.index
    }
}

/// The samples of one node.
#[derive(Debug, Clone, Copy)]
pub struct Samples<'a, T: Element> {
    values: &'a [Tensor<T>],
}

impl<'a, T: Element> Samples<'a, T> {
    /// The number of elements.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether there are no elements.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The elements in row-major order.
    pub fn as_slice(&self) -> &'a [Tensor<T>] {
        self.values
    }

    /// The fraction of samples for which the predicate holds.
    pub fn probability(&self, predicate: impl Fn(&Tensor<T>) -> bool) -> f64 {
        self.values.iter().filter(|v| predicate(v)).count() as f64 / self.values.len() as f64
    }

    /// The most frequent value, or None if there are no samples. Of values with the same highest count, the one
    /// that appears first wins. Values are compared with `==`, so a tensor containing NaN is never equal to
    /// another (or itself) and counts once.
    pub fn mode(&self) -> Option<&'a Tensor<T>>
    where
        T: PartialOrd,
    {
        // sort sample indices so equal values are adjacent (the sort is stable, so each run of equal values
        // starts with its first appearance), then count the runs
        let mut order: Vec<usize> = (0..self.values.len()).collect();
        order.sort_by(|&a, &b| total_cmp(&self.values[a], &self.values[b]));
        // (count, first index) of the best run so far
        let mut best: Option<(usize, usize)> = None;
        let mut consider = |count: usize, first: usize| {
            if best.is_none_or(|(c, f)| count > c || (count == c && first < f)) {
                best = Some((count, first));
            }
        };
        let mut i = 0;
        while i < order.len() {
            let first = order[i];
            let value = &self.values[first];
            let mut j = i + 1;
            #[allow(clippy::eq_op)] // false for a tensor holding NaN
            let comparable = value == value;
            if comparable {
                while j < order.len() && self.values[order[j]] == *value {
                    j += 1;
                }
                consider(j - i, first);
            } else {
                // contains NaN: equal to nothing, so a run of its own
                consider(1, first);
            }
            i = j;
        }
        best.map(|(_, first)| &self.values[first])
    }
}

/// A total order on tensors consistent with `==` for values without NaN: by shape, then element by element, with
/// NaN after every number (and NaNs ordered equal, so a tensor holding NaN sorts next to similar ones without being
/// equal to them).
fn total_cmp<T: Element + PartialOrd>(a: &Tensor<T>, b: &Tensor<T>) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let element = |x: &T, y: &T| match x.partial_cmp(y) {
        Some(ordering) => ordering,
        #[allow(clippy::eq_op)]
        None => (x != x).cmp(&(y != y)),
    };
    a.shape().cmp(b.shape()).then_with(|| {
        a.as_slice()
            .iter()
            .zip(b.as_slice())
            .map(|(x, y)| element(x, y))
            .find(|&o| o != Ordering::Equal)
            .unwrap_or(Ordering::Equal)
    })
}

impl Samples<'_, f64> {
    fn shape(&self) -> &[usize] {
        self.values.first().map(|v| v.shape()).unwrap_or(&[])
    }

    /// The element-wise mean of the samples.
    pub fn mean(&self) -> Result<DoubleTensor> {
        let mut sum = DoubleTensor::zeros(self.shape());
        for v in self.values {
            sum = sum.add(v)?;
        }
        Ok(sum / self.values.len() as f64)
    }

    /// The element-wise sample variance (n - 1 denominator).
    pub fn variance(&self) -> Result<DoubleTensor> {
        let n = self.values.len() as f64;
        let mean = self.mean()?;
        // two passes: sum (x - mean)^2 / (n - 1). Java's E[x^2] - E[x]^2 loses all precision when the mean is
        // large relative to the spread (and can even go negative).
        let mut sum_sq = DoubleTensor::zeros(self.shape());
        for v in self.values {
            sum_sq = sum_sq.add(&v.sub(&mean)?.map(|x| x * x))?;
        }
        Ok(sum_sq / (n - 1.0))
    }

    /// The values of one element of the samples, in sample order. A scalar node's element is `&[]`.
    pub fn element(&self, index: &[usize]) -> Result<Vec<f64>> {
        let shape = self.shape();
        let index: &[usize] = if shape.is_empty() { &[] } else { index };
        self.values
            .iter()
            .map(|v| {
                v.get(index).ok_or_else(|| {
                    AlgorithmError::InvalidArgument(format!(
                        "index {index:?} is out of range for shape {shape:?}"
                    ))
                })
            })
            .collect()
    }

    /// The autocorrelation of one element of the samples at lags 0, 1, ..., len - 1.
    pub fn autocorrelation(&self, index: &[usize]) -> Result<Vec<f64>> {
        Ok(statistics::autocorrelation(&self.element(index)?))
    }

    /// The effective sample size of one element of the samples.
    pub fn effective_sample_size(&self, index: &[usize]) -> Result<f64> {
        Ok(statistics::effective_sample_size(&self.element(index)?))
    }

    /// The samples stacked into one tensor of shape [len, node shape...].
    pub fn to_tensor(&self) -> Result<DoubleTensor> {
        let mut shape = vec![self.values.len()];
        shape.extend_from_slice(self.shape());
        let data: Vec<f64> = self
            .values
            .iter()
            .flat_map(|v| v.as_slice().iter().copied())
            .collect();
        Ok(DoubleTensor::new(&shape, data)?)
    }
}

/// Samples from several independent chains over the same nodes, each with the same number of draws, for
/// convergence diagnostics across chains (see [`diagnostics`]).
///
/// ```
/// use keanu_algorithms::mcmc::chains::sample_chains;
/// use keanu_algorithms::mcmc::nuts::Nuts;
/// use keanu_graph::{GraphBuilder, Model};
/// use keanu_tensor::DoubleTensor;
///
/// let b = GraphBuilder::new();
/// let mu = b.gaussian(0.0, 1.0);
/// let x = b.gaussian(mu, 1.0);
/// b.observe(x, DoubleTensor::scalar(1.0));
/// let mu = mu.id();
/// let (graph, state) = b.build()?;
/// let model = Model::new(graph, &state)?;
/// let nuts = Nuts::default().with_adapt_count(100);
/// let chains = sample_chains(4, 7, &state, |state, rng| nuts.sample(&model, state, &[mu], 300, rng))?;
/// assert_eq!((chains.num_chains(), chains.draws()), (4, 300));
/// let r_hat = chains.rhat(mu, &[])?;
/// assert!(r_hat < 1.05);
/// println!("{}", chains.summary()?);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ChainSamples {
    chains: Vec<NetworkSamples>,
}

impl ChainSamples {
    /// Stacks the samples of each chain. Fails with [`AlgorithmError::InvalidArgument`] if there are no chains, or
    /// the chains did not sample the same nodes (in the same order, with the same types) or have different
    /// numbers of samples.
    pub fn new(chains: Vec<NetworkSamples>) -> Result<Self> {
        let Some(first) = chains.first() else {
            return Err(AlgorithmError::InvalidArgument(
                "there must be at least one chain".into(),
            ));
        };
        let same_columns = |c: &NetworkSamples| {
            c.nodes == first.nodes
                && c.columns
                    .iter()
                    .zip(&first.columns)
                    .all(|(a, b)| std::mem::discriminant(a) == std::mem::discriminant(b))
        };
        if let Some(i) = chains.iter().position(|c| !same_columns(c)) {
            return Err(AlgorithmError::InvalidArgument(format!(
                "chain {i} sampled different nodes from chain 0"
            )));
        }
        if let Some(i) = chains.iter().position(|c| c.len() != first.len()) {
            return Err(AlgorithmError::InvalidArgument(format!(
                "chain {i} has {} samples but chain 0 has {}",
                chains[i].len(),
                first.len()
            )));
        }
        Ok(Self { chains })
    }

    /// The samples of each chain.
    pub fn chains(&self) -> &[NetworkSamples] {
        &self.chains
    }

    /// The samples of each chain, by value.
    pub fn into_chains(self) -> Vec<NetworkSamples> {
        self.chains
    }

    /// The number of chains.
    pub fn num_chains(&self) -> usize {
        self.chains.len()
    }

    /// The number of samples in each chain.
    pub fn draws(&self) -> usize {
        self.chains[0].len()
    }

    /// The nodes that were sampled.
    pub fn nodes(&self) -> &[NodeId] {
        self.chains[0].nodes()
    }

    /// One element of a node in each chain: one series of draws per chain, as f64 (an integer node's values
    /// converted, and a boolean node's as 0 and 1, so that the diagnostics can be applied to discrete latents too).
    /// A scalar node's element is `&[]`. Fails if the node was not sampled or the index is out of range.
    pub fn element(&self, id: NodeId, index: &[usize]) -> Result<Vec<Vec<f64>>> {
        self.chains
            .iter()
            .map(|chain| {
                if let Some(samples) = chain.f64(id) {
                    samples.element(index)
                } else if let Some(samples) = chain.i64(id) {
                    discrete_element(samples.as_slice(), index, |x| x as f64)
                } else if let Some(samples) = chain.bool(id) {
                    discrete_element(samples.as_slice(), index, |x| f64::from(u8::from(x)))
                } else {
                    Err(AlgorithmError::InvalidArgument(format!(
                        "node {id:?} was not sampled"
                    )))
                }
            })
            .collect()
    }

    /// The rank-normalised split R-hat of one element ([`diagnostics::rhat`]).
    pub fn rhat(&self, id: NodeId, index: &[usize]) -> Result<f64> {
        diagnostics::rhat(&self.element(id, index)?)
    }

    /// The bulk effective sample size of one element ([`diagnostics::ess_bulk`]).
    pub fn ess_bulk(&self, id: NodeId, index: &[usize]) -> Result<f64> {
        diagnostics::ess_bulk(&self.element(id, index)?)
    }

    /// The tail effective sample size of one element ([`diagnostics::ess_tail`]).
    pub fn ess_tail(&self, id: NodeId, index: &[usize]) -> Result<f64> {
        diagnostics::ess_tail(&self.element(id, index)?)
    }

    /// The Monte Carlo standard error of the mean of one element ([`diagnostics::mcse_mean`]).
    pub fn mcse_mean(&self, id: NodeId, index: &[usize]) -> Result<f64> {
        diagnostics::mcse_mean(&self.element(id, index)?)
    }

    /// The summary statistics and diagnostics of one element.
    pub fn summarize(&self, id: NodeId, index: &[usize]) -> Result<ElementSummary> {
        diagnostics::summarize(&self.element(id, index)?)
    }

    /// The summary statistics and diagnostics of every element of every sampled f64 node, in node order and
    /// row-major element order. Int and bool nodes are skipped.
    pub fn summary(&self) -> Result<ChainSummary> {
        let first = &self.chains[0];
        let mut rows = Vec::new();
        for &node in first.nodes() {
            let Some(samples) = first.f64(node) else {
                continue;
            };
            for index in element_indices(samples.shape()) {
                let stats = self.summarize(node, &index)?;
                rows.push(SummaryRow { node, index, stats });
            }
        }
        Ok(ChainSummary { rows })
    }
}

/// One element of each of an integer or boolean node's draws, as f64.
fn discrete_element<T: Element>(
    values: &[Tensor<T>],
    index: &[usize],
    to_f64: impl Fn(T) -> f64,
) -> Result<Vec<f64>> {
    let shape = values.first().map(|v| v.shape()).unwrap_or(&[]);
    let index: &[usize] = if shape.is_empty() { &[] } else { index };
    values
        .iter()
        .map(|v| {
            v.get(index).map(&to_f64).ok_or_else(|| {
                AlgorithmError::InvalidArgument(format!(
                    "index {index:?} is out of range for shape {shape:?}"
                ))
            })
        })
        .collect()
}

/// Every index into a tensor of this shape, in row-major order (one empty index for a scalar).
fn element_indices(shape: &[usize]) -> Vec<Vec<usize>> {
    let mut indices = vec![Vec::new()];
    for &size in shape {
        indices = indices
            .into_iter()
            .flat_map(|prefix| {
                (0..size).map(move |i| {
                    let mut index = prefix.clone();
                    index.push(i);
                    index
                })
            })
            .collect();
    }
    indices
}

/// One row of a [`ChainSummary`]: the diagnostics of one element of one node.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SummaryRow {
    /// The node.
    pub node: NodeId,
    /// The element's index in the node's value (empty for a scalar node).
    pub index: Vec<usize>,
    /// Its summary statistics and diagnostics.
    pub stats: ElementSummary,
}

/// The summary statistics and diagnostics of every element of the f64 nodes of [`ChainSamples`]. Its `Display`
/// is a table with a row per element.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainSummary {
    rows: Vec<SummaryRow>,
}

impl ChainSummary {
    /// A row per element.
    pub fn rows(&self) -> &[SummaryRow] {
        &self.rows
    }

    /// The statistics of one element, if it is in the summary.
    pub fn get(&self, node: NodeId, index: &[usize]) -> Option<&ElementSummary> {
        self.rows
            .iter()
            .find(|r| r.node == node && r.index == index)
            .map(|r| &r.stats)
    }

    /// Whether every element is converged ([`ElementSummary::is_converged`]).
    pub fn is_converged(&self) -> bool {
        self.rows.iter().all(|r| r.stats.is_converged())
    }
}

impl std::fmt::Display for ChainSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:<12} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>9} {:>9} {:>7}",
            "node", "mean", "sd", "5%", "50%", "95%", "mcse_mean", "ess_bulk", "ess_tail", "r_hat"
        )?;
        for row in &self.rows {
            let name = if row.index.is_empty() {
                format!("{:?}", row.node)
            } else {
                format!("{:?}{:?}", row.node, row.index)
            };
            let s = &row.stats;
            write!(
                f,
                "\n{name:<12} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>9.0} {:>9.0} {:>7.3}",
                s.mean, s.sd, s.q5, s.q50, s.q95, s.mcse_mean, s.ess_bulk, s.ess_tail, s.r_hat
            )?;
        }
        Ok(())
    }
}
