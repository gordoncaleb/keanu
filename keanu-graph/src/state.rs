//! Node values and their propagation through a graph.

use crate::error::{GraphError, Result};
use crate::graph::{Graph, NodeId};
use crate::op::{Op, eval};
use crate::value::{DType, Value};
use keanu_distributions::KeanuRng;
use keanu_tensor::{BoolTensor, DoubleTensor, IntTensor};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// The values of a graph's nodes and which of them are observed.
///
/// A state is separate from its [`Graph`] so that several states (chains, particles, optimizer runs) can share one
/// graph; `Clone` it to get an independent copy.
#[derive(Debug, Clone, PartialEq)]
pub struct State {
    pub(crate) values: Vec<Option<Value>>,
    pub(crate) observed: Vec<bool>,
}

impl State {
    /// The value, or `None` if the node has no value (or is not a node of this state's graph).
    pub fn value(&self, id: NodeId) -> Option<&Value> {
        self.values.get(id.0)?.as_ref()
    }

    /// The f64 value, if there is one.
    pub fn f64(&self, id: NodeId) -> Option<&DoubleTensor> {
        self.value(id).and_then(Value::as_f64)
    }

    /// The i64 value, if there is one.
    pub fn i64(&self, id: NodeId) -> Option<&IntTensor> {
        self.value(id).and_then(Value::as_i64)
    }

    /// The bool value, if there is one.
    pub fn bool(&self, id: NodeId) -> Option<&BoolTensor> {
        self.value(id).and_then(Value::as_bool)
    }

    /// Whether the node has a value (false if it is not a node of this state's graph).
    pub fn has_value(&self, id: NodeId) -> bool {
        self.value(id).is_some()
    }

    /// Whether the node is observed (false if it is not a node of this state's graph).
    pub fn is_observed(&self, id: NodeId) -> bool {
        self.observed.get(id.0).copied().unwrap_or(false)
    }

    /// An error unless this state holds the values of `graph` and every id is a node of it, so that indexing by
    /// those ids (and their ancestors and descendants) can't go out of range.
    fn check_ids<'a>(
        &self,
        graph: &Graph,
        ids: impl IntoIterator<Item = &'a NodeId>,
    ) -> Result<()> {
        if self.values.len() != graph.len() {
            return Err(GraphError::InvalidValue(format!(
                "the state holds {} nodes but the graph has {}",
                self.values.len(),
                graph.len()
            )));
        }
        graph.check_ids(ids)
    }

    /// [`GraphError::UnknownNode`] unless the id is a node of this state.
    fn check_own(&self, id: NodeId) -> Result<()> {
        if id.0 < self.values.len() {
            Ok(())
        } else {
            Err(GraphError::UnknownNode(id))
        }
    }

    fn check(&self, graph: &Graph, id: NodeId, value: &Value) -> Result<()> {
        self.check_ids(graph, [&id])?;
        let node = &graph[id];
        if value.dtype() != node.dtype || value.shape() != node.shape() {
            return Err(GraphError::InvalidValue(format!(
                "node {id:?} holds {:?} of shape {:?}, got {:?} of shape {:?}",
                node.dtype,
                node.shape(),
                value.dtype(),
                value.shape()
            )));
        }
        Ok(())
    }

    /// Sets a node's value. Observed nodes keep their observed value (the call is ignored).
    pub fn set(&mut self, graph: &Graph, id: NodeId, value: impl Into<Value>) -> Result<()> {
        let value = value.into();
        self.check(graph, id, &value)?;
        if !self.observed[id.0] {
            self.values[id.0] = Some(value);
        }
        Ok(())
    }

    /// Sets a node's value and recomputes its deterministic descendants.
    pub fn set_and_cascade(
        &mut self,
        graph: &Graph,
        id: NodeId,
        value: impl Into<Value>,
    ) -> Result<()> {
        self.set(graph, id, value)?;
        self.cascade(graph, &[id])
    }

    /// Fixes a node's value. Probabilistic nodes and boolean deterministic nodes can be observed; numeric
    /// deterministic nodes cannot.
    pub fn observe(&mut self, graph: &Graph, id: NodeId, value: impl Into<Value>) -> Result<()> {
        self.check_ids(graph, [&id])?;
        let node = &graph[id];
        if !node.is_probabilistic() && node.dtype != DType::Bool {
            return Err(GraphError::NotObservable(format!(
                "node {id:?} ({}) is deterministic and numeric, so it cannot be observed",
                node.op.name()
            )));
        }
        let value = value.into();
        self.check(graph, id, &value)?;
        self.values[id.0] = Some(value);
        self.observed[id.0] = true;
        Ok(())
    }

    /// Stops observing a node, keeping its current value.
    pub fn unobserve(&mut self, id: NodeId) -> Result<()> {
        self.check_own(id)?;
        self.observed[id.0] = false;
        Ok(())
    }

    /// Removes a node's value (unless it is observed), so that the next evaluation recomputes or resamples it.
    pub fn clear(&mut self, id: NodeId) -> Result<()> {
        self.check_own(id)?;
        if !self.observed[id.0] {
            self.values[id.0] = None;
        }
        Ok(())
    }

    /// Computes the given nodes and their ancestors. Deterministic ancestors are recomputed, up to the nearest
    /// probabilistic ancestors; probabilistic nodes keep their values, or are sampled from their prior (given their
    /// lazily evaluated parents) if they have none.
    pub fn eval(&mut self, graph: &Graph, targets: &[NodeId], rng: &mut KeanuRng) -> Result<()> {
        self.check_ids(graph, targets)?;
        let plan = self.plan(graph, targets, false);
        self.execute(graph, &plan, rng)
    }

    /// Computes only the missing values the given nodes need, sampling probabilistic nodes without a value.
    pub fn lazy_eval(
        &mut self,
        graph: &Graph,
        targets: &[NodeId],
        rng: &mut KeanuRng,
    ) -> Result<()> {
        self.check_ids(graph, targets)?;
        let plan = self.plan(graph, targets, true);
        self.execute(graph, &plan, rng)
    }

    /// Recomputes the deterministic descendants of `from` (and `from` itself if deterministic), in topological
    /// order, stopping at probabilistic nodes.
    ///
    /// The cost is proportional to the descendants recomputed, not to the size of the graph: nodes are visited
    /// in increasing id order from a heap, and as a node's children have larger ids, every path to a node has been
    /// followed (and its duplicate heap entries are adjacent) by the time it is popped.
    pub fn cascade(&mut self, graph: &Graph, from: &[NodeId]) -> Result<()> {
        self.check_ids(graph, from)?;
        let mut pending: BinaryHeap<Reverse<NodeId>> = from.iter().map(|&id| Reverse(id)).collect();
        let mut last = None;
        while let Some(Reverse(id)) = pending.pop() {
            if last.replace(id) == Some(id) {
                continue;
            }
            if graph[id].is_probabilistic() {
                if !self.has_value(id) {
                    return Err(GraphError::InvalidValue(format!(
                        "probabilistic node {id:?} has no value to cascade from"
                    )));
                }
            } else {
                self.update(graph, id, None)?;
            }
            pending.extend(
                graph
                    .children_of(id)
                    .iter()
                    .filter(|&&child| !graph[child].is_probabilistic())
                    .map(|&child| Reverse(child)),
            );
        }
        Ok(())
    }

    /// The nodes an eval (or lazy eval) of the targets updates, in topological order: deterministic nodes to
    /// recompute and probabilistic nodes without a value to sample.
    ///
    /// Nodes are visited in decreasing id order from a heap, so (as parents have smaller ids) every request for a
    /// node has been made by the time it is popped, and the cost is proportional to the nodes visited rather than
    /// to the size of the graph. A node may be requested lazily and eagerly (which does more): eager requests sort
    /// first, and the node's other entries are skipped.
    fn plan(&self, graph: &Graph, targets: &[NodeId], lazy: bool) -> Vec<NodeId> {
        // (node, eager)
        let mut pending: BinaryHeap<(NodeId, bool)> = targets.iter().map(|&t| (t, !lazy)).collect();
        let mut plan = Vec::new();
        let mut last = None;
        while let Some((id, eager)) = pending.pop() {
            if last.replace(id) == Some(id) {
                continue;
            }
            let has_value = self.has_value(id);
            if !eager && has_value {
                continue;
            }
            let node = &graph[id];
            if node.is_probabilistic() {
                if !has_value {
                    plan.push(id);
                    pending.extend(node.parents.iter().map(|&p| (p, false)));
                }
            } else {
                plan.push(id);
                pending.extend(node.parents.iter().map(|&p| (p, eager)));
            }
        }
        plan.reverse();
        plan
    }

    fn execute(&mut self, graph: &Graph, plan: &[NodeId], rng: &mut KeanuRng) -> Result<()> {
        for &id in plan {
            self.update(graph, id, Some(rng))?;
        }
        Ok(())
    }

    /// Draws a value for a probabilistic node from its prior given its parents' current values, without changing
    /// the state. The parents must have values.
    pub fn sample_prior(&self, graph: &Graph, id: NodeId, rng: &mut KeanuRng) -> Result<Value> {
        self.check_ids(graph, [&id])?;
        let node = &graph[id];
        let Op::Distribution(distribution) = &node.op else {
            return Err(GraphError::InvalidValue(format!(
                "node {id:?} is not probabilistic, so it has no prior to sample from"
            )));
        };
        let parents: smallvec::SmallVec<[&Value; 4]> = node
            .parents
            .iter()
            .map(|&p| self.values[p.0].as_ref().ok_or(GraphError::MissingValue(p)))
            .collect::<Result<_>>()?;
        distribution.sample(&parents, rng)
    }

    /// Samples a probabilistic node with no value, or recomputes a deterministic one
    /// that isn't observed.
    fn update(&mut self, graph: &Graph, id: NodeId, rng: Option<&mut KeanuRng>) -> Result<()> {
        let node = &graph[id];
        if self.observed[id.0] {
            return Ok(());
        }
        let parents: smallvec::SmallVec<[&Value; 4]> = node
            .parents
            .iter()
            .map(|&p| {
                self.values[p.0].as_ref().ok_or_else(|| {
                    GraphError::InvalidValue(format!(
                        "node {p:?} (a parent of {id:?}) has no value"
                    ))
                })
            })
            .collect::<Result<_>>()?;
        let value = match &node.op {
            Op::Distribution(distribution) => {
                if self.values[id.0].is_some() {
                    return Ok(());
                }
                let rng = rng.ok_or_else(|| {
                    GraphError::InvalidValue(format!("probabilistic node {id:?} needs a value"))
                })?;
                distribution.sample(&parents, rng)?
            }
            // a constant's value is its state: setting it sticks, as in Java (calculate() returns getValue())
            Op::Constant(_) if self.values[id.0].is_some() => return Ok(()),
            Op::Proxy if parents.is_empty() => {
                return Err(GraphError::Proxy(format!(
                    "proxy node {id:?} was never connected"
                )));
            }
            op => eval(op, &parents)?,
        };
        drop(parents);
        self.values[id.0] = Some(value);
        Ok(())
    }
}
