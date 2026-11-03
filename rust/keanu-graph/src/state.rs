//! Node values and their propagation through a graph (Java's `VertexValuePropagation`).

use crate::error::{GraphError, Result};
use crate::graph::{Graph, NodeId};
use crate::op::{Op, eval};
use crate::value::{DType, Value};
use keanu_distributions::KeanuRng;
use keanu_tensor::{BoolTensor, DoubleTensor, IntTensor};

/// The values of a graph's nodes and which of them are observed.
///
/// A state is separate from its [`Graph`] so that several states (chains, particles, optimizer runs) can share one
/// graph; `Clone` it to get an independent copy.
#[derive(Debug, Clone, PartialEq)]
pub struct State {
    pub(crate) values: Vec<Option<Value>>,
    pub(crate) observed: Vec<bool>,
}

#[derive(Clone, Copy, PartialEq)]
enum Plan {
    None,
    /// recompute a deterministic node, or sample a probabilistic one that has no value
    Update,
}

impl State {
    pub fn value(&self, id: NodeId) -> Option<&Value> {
        self.values[id.0].as_ref()
    }

    pub fn f64(&self, id: NodeId) -> Option<&DoubleTensor> {
        self.value(id).and_then(Value::as_f64)
    }

    pub fn i64(&self, id: NodeId) -> Option<&IntTensor> {
        self.value(id).and_then(Value::as_i64)
    }

    pub fn bool(&self, id: NodeId) -> Option<&BoolTensor> {
        self.value(id).and_then(Value::as_bool)
    }

    pub fn has_value(&self, id: NodeId) -> bool {
        self.values[id.0].is_some()
    }

    pub fn is_observed(&self, id: NodeId) -> bool {
        self.observed[id.0]
    }

    fn check(graph: &Graph, id: NodeId, value: &Value) -> Result<()> {
        let node = graph.node(id);
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

    /// Sets a node's value. Observed nodes keep their observed value (the call is ignored), as in Java.
    pub fn set(&mut self, graph: &Graph, id: NodeId, value: impl Into<Value>) -> Result<()> {
        let value = value.into();
        Self::check(graph, id, &value)?;
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
        let node = graph.node(id);
        if !node.is_probabilistic() && node.dtype != DType::Bool {
            return Err(GraphError::NotObservable(format!(
                "node {id:?} ({}) is deterministic and numeric, so it cannot be observed",
                node.op.name()
            )));
        }
        let value = value.into();
        Self::check(graph, id, &value)?;
        self.values[id.0] = Some(value);
        self.observed[id.0] = true;
        Ok(())
    }

    /// Stops observing a node, keeping its current value.
    pub fn unobserve(&mut self, id: NodeId) {
        self.observed[id.0] = false;
    }

    /// Removes a node's value (unless it is observed), so that the next evaluation recomputes or resamples it.
    pub fn clear(&mut self, id: NodeId) {
        if !self.observed[id.0] {
            self.values[id.0] = None;
        }
    }

    /// Computes the given nodes and their ancestors. Deterministic ancestors are recomputed, up to the nearest
    /// probabilistic ancestors; probabilistic nodes keep their values, or are sampled from their prior (given their
    /// lazily evaluated parents) if they have none.
    pub fn eval(&mut self, graph: &Graph, targets: &[NodeId], rng: &mut KeanuRng) -> Result<()> {
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
        let plan = self.plan(graph, targets, true);
        self.execute(graph, &plan, rng)
    }

    /// Recomputes the deterministic descendants of `from` (and `from` itself if deterministic), in topological
    /// order, stopping at probabilistic nodes.
    pub fn cascade(&mut self, graph: &Graph, from: &[NodeId]) -> Result<()> {
        let mut plan = vec![Plan::None; graph.len()];
        let mut stack: Vec<NodeId> = from.to_vec();
        for &id in from {
            plan[id.0] = Plan::Update;
        }
        while let Some(id) = stack.pop() {
            for &child in graph.children(id) {
                if !graph.node(child).is_probabilistic() && plan[child.0] == Plan::None {
                    plan[child.0] = Plan::Update;
                    stack.push(child);
                }
            }
        }
        for id in graph.ids() {
            if plan[id.0] == Plan::Update {
                let node = graph.node(id);
                if node.is_probabilistic() {
                    if !self.has_value(id) {
                        return Err(GraphError::InvalidValue(format!(
                            "probabilistic node {id:?} has no value to cascade from"
                        )));
                    }
                } else {
                    self.update(graph, id, None)?;
                }
            }
        }
        Ok(())
    }

    /// Marks which nodes an eval or lazy eval of the targets updates.
    fn plan(&self, graph: &Graph, targets: &[NodeId], lazy: bool) -> Vec<Plan> {
        let mut plan = vec![Plan::None; graph.len()];
        // a node may be visited lazily and later eagerly (which does more), but not the other way round
        let mut visited_eagerly = vec![false; graph.len()];
        let mut visited_lazily = vec![false; graph.len()];
        let mut stack: Vec<(NodeId, bool)> = targets.iter().map(|&t| (t, lazy)).collect();
        while let Some((id, lazy)) = stack.pop() {
            if visited_eagerly[id.0] || (lazy && visited_lazily[id.0]) {
                continue;
            }
            if lazy {
                visited_lazily[id.0] = true;
            } else {
                visited_eagerly[id.0] = true;
            }
            let node = graph.node(id);
            let has_value = self.has_value(id);
            if lazy && has_value {
                continue;
            }
            if node.is_probabilistic() {
                if !has_value {
                    plan[id.0] = Plan::Update;
                    stack.extend(node.parents.iter().map(|&p| (p, true)));
                }
            } else {
                plan[id.0] = Plan::Update;
                stack.extend(node.parents.iter().map(|&p| (p, lazy)));
            }
        }
        plan
    }

    fn execute(&mut self, graph: &Graph, plan: &[Plan], rng: &mut KeanuRng) -> Result<()> {
        for id in graph.ids() {
            if plan[id.0] == Plan::Update {
                self.update(graph, id, Some(rng))?;
            }
        }
        Ok(())
    }

    /// Draws a value for a probabilistic node from its prior given its parents' current values, without changing
    /// the state. The parents must have values.
    pub fn sample_prior(&self, graph: &Graph, id: NodeId, rng: &mut KeanuRng) -> Result<Value> {
        let node = graph.node(id);
        let Op::Distribution(distribution) = &node.op else {
            return Err(GraphError::InvalidValue(format!(
                "node {id:?} is not probabilistic, so it has no prior to sample from"
            )));
        };
        let parents: Vec<&Value> = node
            .parents
            .iter()
            .map(|&p| {
                self.values[p.0]
                    .as_ref()
                    .ok_or(GraphError::MissingValue(p.0))
            })
            .collect::<Result<_>>()?;
        distribution.sample(&parents, rng)
    }

    /// Java's `updateVertexValue`: sample a probabilistic node with no value, or recompute a deterministic one
    /// that isn't observed.
    fn update(&mut self, graph: &Graph, id: NodeId, rng: Option<&mut KeanuRng>) -> Result<()> {
        let node = graph.node(id);
        if self.observed[id.0] {
            return Ok(());
        }
        let parents: Vec<&Value> = node
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
        self.values[id.0] = Some(value);
        Ok(())
    }
}
