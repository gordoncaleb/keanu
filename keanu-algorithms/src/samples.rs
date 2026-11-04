//! Samples of node values, as produced by samplers.

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

/// Samples of a set of nodes, with the log probability recorded for each sample (Java's `NetworkSamples`).
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

/// One sample of every sampled node (Java's `NetworkState`).
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
