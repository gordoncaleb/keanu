//! Machinery shared by the graph rewrites ([`marginalize`](crate::marginalize) and [`intervene`](crate::intervene)): a
//! writer that builds the new graph's nodes with their initial values and observations, and the old-to-new id mapping
//! helpers.

use crate::error::{GraphError, Result};
use crate::graph::{Graph, Node, NodeId, Parents};
use crate::op::{Op, eval, output_type};
use crate::state::State;
use crate::value::{DType, Value};
use keanu_tensor::Shape;
use std::collections::HashMap;

/// The nodes of a graph being written, with their initial values and observations.
#[derive(Default)]
pub(crate) struct Writer {
    pub(crate) nodes: Vec<Node>,
    pub(crate) values: Vec<Option<Value>>,
    pub(crate) observed: Vec<bool>,
}

impl Writer {
    pub(crate) fn push(&mut self, node: Node, value: Option<Value>, observed: bool) -> NodeId {
        self.nodes.push(node);
        self.values.push(value);
        self.observed.push(observed);
        NodeId(self.nodes.len() - 1)
    }

    /// A new deterministic node, its type and shape inferred.
    pub(crate) fn add(&mut self, op: Op, parents: &[NodeId]) -> Result<NodeId> {
        let (dtype, shape) = {
            let types: Vec<(DType, &[usize])> = parents
                .iter()
                .map(|p| (self.nodes[p.0].dtype, self.nodes[p.0].shape.as_slice()))
                .collect();
            output_type(&op, &types)?
        };
        Ok(self.push(
            Node {
                op,
                parents: Parents::from_slice(parents),
                dtype,
                shape,
                label: None,
            },
            None,
            false,
        ))
    }

    /// A new constant node holding `value`.
    pub(crate) fn constant(&mut self, value: Value, label: Option<String>) -> NodeId {
        let node = Node {
            dtype: value.dtype(),
            shape: Shape::from_slice(value.shape()),
            op: Op::Constant(value.clone()),
            parents: Parents::new(),
            label,
        };
        self.push(node, Some(value), false)
    }

    /// A copy of `node` (a node of another graph) with its parents replaced by `parents`, with `value` and
    /// `observed` as its initial value and observation.
    pub(crate) fn copy(
        &mut self,
        node: &Node,
        parents: Parents,
        value: Option<Value>,
        observed: bool,
    ) -> NodeId {
        self.push(
            Node {
                parents,
                ..node.clone()
            },
            value,
            observed,
        )
    }

    /// The finished graph and state. Each deterministic node without a value whose parents have values is computed,
    /// so that the new nodes are ready to use; a node that fails to evaluate is left without a value, for its next
    /// evaluation to report. Fails with [`GraphError::DuplicateLabel`] if two nodes have the same label.
    pub(crate) fn finish(mut self) -> Result<(Graph, State)> {
        fill_values(&self.nodes, &mut self.values);
        let mut labels = HashMap::new();
        for (i, node) in self.nodes.iter().enumerate() {
            if let Some(label) = &node.label {
                if labels.insert(label.clone(), NodeId(i)).is_some() {
                    return Err(GraphError::DuplicateLabel(label.clone()));
                }
            }
        }
        let graph = Graph::new(self.nodes, labels);
        Ok((
            graph,
            State {
                values: self.values,
                observed: self.observed,
            },
        ))
    }
}

/// Computes each deterministic node without a value whose parents have values, in id (topological) order; nodes
/// that fail to evaluate stay without one.
pub(crate) fn fill_values(nodes: &[Node], values: &mut [Option<Value>]) {
    for id in 0..nodes.len() {
        let node = &nodes[id];
        if values[id].is_some() || node.is_probabilistic() {
            continue;
        }
        let parents: Option<Vec<&Value>> =
            node.parents.iter().map(|p| values[p.0].as_ref()).collect();
        if let Some(Ok(value)) = parents.map(|parents| eval(&node.op, &parents)) {
            values[id] = Some(value);
        }
    }
}

/// The new node a parent was copied to, or an internal error if it wasn't copied before its child.
pub(crate) fn mapped(mapping: &[Option<NodeId>], p: NodeId) -> Result<NodeId> {
    mapping[p.0].ok_or_else(|| {
        GraphError::InvalidValue(format!(
            "internal error: node {p:?} was not copied before its child"
        ))
    })
}

/// An error unless the state holds the values of `graph`.
pub(crate) fn check_state(graph: &Graph, state: &State) -> Result<()> {
    if state.values.len() != graph.len() {
        return Err(GraphError::InvalidValue(format!(
            "the state holds {} nodes but the graph has {}",
            state.values.len(),
            graph.len()
        )));
    }
    Ok(())
}

/// For each node, whether its value may change because of the nodes marked in `changed`: those nodes and the
/// deterministic, unobserved nodes downstream of them (probabilistic and observed nodes keep their own values).
pub(crate) fn stale(graph: &Graph, state: &State, changed: &[bool]) -> Vec<bool> {
    let mut stale = changed.to_vec();
    for id in graph.ids() {
        let node = &graph[id];
        if stale[id.0] || node.is_probabilistic() || state.is_observed(id) {
            continue;
        }
        stale[id.0] = node.parents.iter().any(|p| stale[p.0]);
    }
    stale
}

/// An error if `ids` repeats a node.
pub(crate) fn check_distinct(graph: &Graph, ids: &[NodeId]) -> Result<()> {
    for (i, id) in ids.iter().enumerate() {
        if ids[..i].contains(id) {
            return Err(GraphError::InvalidValue(format!(
                "{} is listed more than once",
                graph.describe(*id)
            )));
        }
    }
    Ok(())
}
