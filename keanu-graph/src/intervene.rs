//! Interventions: the do-operator as a graph rewrite that sets nodes to fixed values, cutting them off from their
//! causes.
//!
//! Conditioning on X = x (observing it) asks what the other nodes are like among the cases where X happens to be x,
//! and so carries information back to X's causes. Intervening, do(X = x), asks what they would be if X were *set* to
//! x whatever its causes: X becomes a constant, its distribution (and its log probability term) is gone, and its
//! causes keep their own distributions. [`intervene`] rewrites a graph that way, and the posterior predictive of the
//! rewritten graph, given posterior draws of the original model mapped through the [`Intervened`] record (with the
//! latents downstream of the intervened nodes drawn again given their new value), answers interventional questions
//! (see `PredictiveSampler::interventional` in keanu-algorithms).

use crate::error::{GraphError, Result};
use crate::graph::{Graph, NodeId};
use crate::rewrite::{Writer, check_distinct, check_state, mapped, stale};
use crate::state::State;
use crate::value::Value;

/// The record of an [`intervene`] rewrite: which node of the new graph each node of the original graph became.
#[derive(Debug, Clone, PartialEq)]
pub struct Intervened {
    /// For each node of the original graph, its node in the new graph.
    mapping: Vec<Option<NodeId>>,
    /// The intervened nodes, by their ids in the original graph, in the order given.
    intervened: Vec<NodeId>,
    /// For each node of the original graph, whether it depends on an intervened node (and isn't one).
    downstream: Vec<bool>,
}

impl Intervened {
    /// The node of the new graph that a node of the original graph became: the same operation on the
    /// corresponding parents, with the same label, value and observation, or for an intervened node a constant
    /// holding its value, with its label. `None` for the nodes that were dropped because they only fed intervened
    /// nodes (see [`is_dropped`](Self::is_dropped)) and for ids that aren't nodes of the original graph.
    pub fn node(&self, original: NodeId) -> Option<NodeId> {
        self.mapping.get(original.index()).copied().flatten()
    }

    /// The intervened nodes, by their ids in the original graph, in the order they were given.
    pub fn intervened(&self) -> &[NodeId] {
        &self.intervened
    }

    /// Whether a node of the original graph was intervened on.
    pub fn is_intervened(&self, original: NodeId) -> bool {
        self.intervened.contains(&original)
    }

    /// Whether a node of the original graph was dropped: it influenced the rest of the model only through
    /// intervened nodes, so it has no part in the new graph.
    pub fn is_dropped(&self, original: NodeId) -> bool {
        original.index() < self.mapping.len() && self.mapping[original.index()].is_none()
    }

    /// The nodes of the original graph that were dropped, in id order.
    pub fn dropped(&self) -> Vec<NodeId> {
        self.mapping
            .iter()
            .enumerate()
            .filter(|(_, new)| new.is_none())
            .map(|(i, _)| NodeId(i))
            .collect()
    }

    /// Whether a node of the original graph depends, directly or through other nodes, on an intervened node (and
    /// isn't one itself). Its posterior draws were made given the intervened node's old value, so they don't carry
    /// over: a latent downstream must be drawn again from its distribution given its parents' new values.
    pub fn is_downstream(&self, original: NodeId) -> bool {
        self.downstream
            .get(original.index())
            .copied()
            .unwrap_or(false)
    }

    /// The nodes of the original graph downstream of an intervened node ([`is_downstream`](Self::is_downstream)),
    /// in id order.
    pub fn downstream(&self) -> Vec<NodeId> {
        self.downstream
            .iter()
            .enumerate()
            .filter(|(_, downstream)| **downstream)
            .map(|(i, _)| NodeId(i))
            .collect()
    }

    /// The node of the new graph whose values a draw of an original node carries over to: the node it became,
    /// unless it was intervened on (its value is now fixed), dropped, or is downstream of an intervened node (its
    /// draws depended on the old value; [`is_downstream`](Self::is_downstream)). Use it to map posterior draws of
    /// the original model onto the new graph.
    pub fn carried(&self, original: NodeId) -> Option<NodeId> {
        if self.is_intervened(original) || self.is_downstream(original) {
            None
        } else {
            self.node(original)
        }
    }
}

/// The do-operator: returns a new graph and state in which each node in `interventions` is a constant holding the
/// value given for it, and the [`Intervened`] record of how the nodes correspond.
///
/// - **Intervened nodes.** Each becomes a constant with the same label. A probabilistic node loses its distribution,
///   so its log probability term leaves the joint log probability; an observed node's observation is replaced by the
///   value; a deterministic node no longer computes from its parents. Its children read the constant.
/// - **Ancestors.** A node that influenced the rest of the model only through intervened nodes (directly, or
///   through other such nodes) is dropped: an unobserved node all of whose children are intervened or dropped.
///   Observed nodes and nodes without children are never dropped. [`Intervened::node`] gives `None` for the dropped
///   nodes.
/// - **Everything else** is copied with its label, value and observation, and keeps its dependence on its
///   parents, so latents downstream of an intervened node still depend on its new value. The values of
///   deterministic nodes downstream of an intervened node are recomputed.
///
/// The new graph's posterior predictive, given posterior draws of the original model mapped onto it with
/// [`Intervened::carried`] (which leaves out the nodes downstream of an intervened node), with the latents
/// downstream of the intervened nodes ([`Intervened::downstream`]) drawn again given their parents, is the
/// interventional distribution p(y | do(x)) averaged over the posterior.
///
/// ```
/// use keanu_graph::{GraphBuilder, Model, intervene};
/// use keanu_tensor::DoubleTensor;
///
/// // a confounder z drives both x and y; x also drives y
/// let b = GraphBuilder::new();
/// let z = b.gaussian(0.0, 1.0).label("z");
/// let x = b.gaussian(z, 1.0).label("x");
/// let y = b.gaussian(x + z, 1.0).label("y");
/// b.observe(y, DoubleTensor::scalar(2.0));
/// let x = x.id();
/// let (graph, state) = b.build()?;
///
/// let (graph, state, intervened) = intervene(&graph, &state, &[(x, DoubleTensor::scalar(1.0).into())])?;
/// let x = intervened.node(x).unwrap();
/// assert_eq!(graph.find("x"), Some(x));
/// assert!(!graph[x].is_probabilistic());
/// // z is now the only latent, and x's density is gone from the joint log probability
/// let model = Model::new(graph, &state)?;
/// assert_eq!(model.latents(), &[model.graph().find("z").unwrap()]);
/// # Ok::<(), keanu_graph::GraphError>(())
/// ```
///
/// # Errors
///
/// [`GraphError::UnknownNode`] for an id that isn't a node of the graph, [`GraphError::InvalidValue`] if
/// `interventions` is empty or repeats a node, a value doesn't have its node's type and shape, or the state isn't the
/// graph's.
pub fn intervene(
    graph: &Graph,
    state: &State,
    interventions: &[(NodeId, Value)],
) -> Result<(Graph, State, Intervened)> {
    let targets: Vec<NodeId> = interventions.iter().map(|(id, _)| *id).collect();
    graph.check_ids(&targets)?;
    check_state(graph, state)?;
    if interventions.is_empty() {
        return Err(GraphError::InvalidValue(
            "there are no interventions to make".into(),
        ));
    }
    check_distinct(graph, &targets)?;
    let n = graph.len();
    let mut value_of: Vec<Option<&Value>> = vec![None; n];
    for (id, value) in interventions {
        let node = &graph[*id];
        if value.dtype() != node.dtype() || value.shape() != node.shape() {
            return Err(GraphError::InvalidValue(format!(
                "{} holds {:?} of shape {:?}, but the intervention sets it to {:?} of shape {:?}",
                graph.describe(*id),
                node.dtype(),
                node.shape(),
                value.dtype(),
                value.shape()
            )));
        }
        value_of[id.0] = Some(value);
    }

    // the nodes whose influence on the model ran only through intervened nodes; children have larger ids
    let mut dropped = vec![false; n];
    for id in (0..n).rev() {
        let children = graph.children_of(NodeId(id));
        dropped[id] = value_of[id].is_none()
            && !state.is_observed(NodeId(id))
            && !children.is_empty()
            && children
                .iter()
                .all(|c| value_of[c.0].is_some() || dropped[c.0]);
    }
    let changed: Vec<bool> = value_of.iter().map(Option::is_some).collect();
    // the nodes that depend on an intervened node; parents have smaller ids
    let mut downstream = vec![false; n];
    for id in graph.ids() {
        downstream[id.0] = !changed[id.0]
            && graph[id]
                .parents
                .iter()
                .any(|p| changed[p.0] || downstream[p.0]);
    }
    let stale = stale(graph, state, &changed);

    let mut out = Writer::default();
    let mut mapping: Vec<Option<NodeId>> = vec![None; n];
    for id in graph.ids() {
        if dropped[id.0] {
            continue;
        }
        let node = &graph[id];
        mapping[id.0] = Some(match value_of[id.0] {
            Some(value) => out.constant(value.clone(), node.label.clone()),
            None => {
                let parents = node
                    .parents
                    .iter()
                    .map(|&p| mapped(&mapping, p))
                    .collect::<Result<_>>()?;
                let value = if stale[id.0] {
                    None
                } else {
                    state.value(id).cloned()
                };
                out.copy(node, parents, value, state.is_observed(id))
            }
        });
    }
    let (graph, state) = out.finish()?;
    Ok((
        graph,
        state,
        Intervened {
            mapping,
            intervened: targets,
            downstream,
        },
    ))
}
