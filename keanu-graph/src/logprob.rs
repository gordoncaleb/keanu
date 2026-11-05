//! Log probabilities of probabilistic nodes and their gradients with respect to latent nodes.

use crate::autodiff::{
    SecondOrderSeed, accumulate, forward_tangents, reverse_from_seeds, second_order_sweep,
};
use crate::error::{GraphError, Result};
use crate::graph::{Graph, NodeId};
use crate::op::{DistributionKind, LogProbGradientParts, Op, Support};
use crate::state::State;
use crate::value::{DType, Value};
use keanu_tensor::DoubleTensor;
use smallvec::SmallVec;
use std::collections::HashMap;

fn value(state: &State, id: NodeId) -> Result<&Value> {
    state.value(id).ok_or(GraphError::MissingValue(id))
}

fn kind(graph: &Graph, id: NodeId) -> Result<&DistributionKind> {
    match graph.node(id).ok_or(GraphError::UnknownNode(id))?.op() {
        Op::Distribution(op) => Ok(&op.kind),
        _ => Err(GraphError::InvalidValue(format!(
            "node #{} is not probabilistic, so it has no log probability",
            id.0
        ))),
    }
}

/// The log probability of a probabilistic node at its current value, given its parameters' current values
/// (summed over the elements of the node).
pub fn node_log_prob(graph: &Graph, state: &State, id: NodeId) -> Result<f64> {
    graph.check_ids([&id])?;
    node_log_prob_at(graph, state, id, value(state, id)?)
}

/// The log probability that a probabilistic node takes the value `v`, given its parameters' current values
/// (summed over the elements of the node). The node's own current value is ignored.
pub fn node_log_prob_at(graph: &Graph, state: &State, id: NodeId, v: &Value) -> Result<f64> {
    let kind = kind(graph, id)?;
    kind.log_prob(v, &parameters(graph, state, id)?)
}

/// The support of a probabilistic node's distribution given its parameters' current values (see [`Support`]).
/// An error if the node is not probabilistic or a parameter has no value.
pub fn node_support(graph: &Graph, state: &State, id: NodeId) -> Result<Support> {
    match graph.node(id).ok_or(GraphError::UnknownNode(id))?.op() {
        Op::Distribution(op) => op.kind.support(&parameters(graph, state, id)?),
        _ => Err(GraphError::InvalidValue(format!(
            "node #{} is not probabilistic, so it has no support",
            id.0
        ))),
    }
}

/// The values of a node's parameters (its parents), for a node of the graph.
fn parameters<'s>(graph: &Graph, state: &'s State, id: NodeId) -> Result<SmallVec<[&'s Value; 4]>> {
    graph[id]
        .parents()
        .iter()
        .map(|&p| value(state, p))
        .collect()
}

/// The total log probability of the given probabilistic nodes.
pub fn log_prob(graph: &Graph, state: &State, nodes: &[NodeId]) -> Result<f64> {
    nodes
        .iter()
        .map(|&id| node_log_prob(graph, state, id))
        .sum()
}

/// The joint log probability of every probabilistic node.
pub fn joint_log_prob(graph: &Graph, state: &State) -> Result<f64> {
    graph
        .probabilistic()
        .map(|id| node_log_prob(graph, state, id))
        .sum()
}

/// The log likelihood: the log probability of the observed probabilistic nodes.
pub fn log_likelihood(graph: &Graph, state: &State) -> Result<f64> {
    graph
        .probabilistic()
        .filter(|&id| state.is_observed(id))
        .map(|id| node_log_prob(graph, state, id))
        .sum()
}

/// The log prior: the log probability of the unobserved (latent) probabilistic nodes.
pub fn log_prior(graph: &Graph, state: &State) -> Result<f64> {
    graph
        .probabilistic()
        .filter(|&id| !state.is_observed(id))
        .map(|id| node_log_prob(graph, state, id))
        .sum()
}

/// The derivatives of one node's log probability with respect to its own value (`value`, None for non-f64
/// nodes and distributions without one) and each of its parents (`parameters`, None where there is no
/// derivative). Element-wise (shaped like the node's value) except where the distribution reduces itself
/// (multivariate Gaussian, Dirichlet).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct NodeLogProbGradient {
    /// With respect to `value`.
    pub value: Option<DoubleTensor>,
    /// With respect to `parameters`.
    pub parameters: Vec<Option<DoubleTensor>>,
}

/// ∂ log p(node) / ∂ value and ∂ log p(node) / ∂ parameter for each parameter.
pub fn node_log_prob_gradient(
    graph: &Graph,
    state: &State,
    id: NodeId,
) -> Result<NodeLogProbGradient> {
    let (value, parameters) = node_gradient_parts(graph, state, id)?;
    Ok(NodeLogProbGradient {
        value,
        parameters: parameters.into_vec(),
    })
}

fn node_gradient_parts(graph: &Graph, state: &State, id: NodeId) -> Result<LogProbGradientParts> {
    let kind = kind(graph, id)?;
    let v = value(state, id)?;
    kind.log_prob_gradient(v, &parameters(graph, state, id)?)
}

/// Sums a gradient over the dimensions that a tensor of `target` shape was broadcast along.
fn sum_to_shape(t: DoubleTensor, target: &[usize]) -> Result<DoubleTensor> {
    if t.shape() == target {
        return Ok(t);
    }
    if target.iter().all(|&d| d == 1) {
        // a scalar (or all-ones) parameter: the total
        return Ok(DoubleTensor::full(target, t.sum()));
    }
    let rank = t.rank();
    if target.len() > rank {
        return Ok(t.broadcast_to(target)?);
    }
    let offset = rank - target.len();
    let axes: Vec<isize> = (0..rank)
        .filter(|&i| i < offset || (target[i - offset] == 1 && t.shape()[i] != 1))
        .map(|i| i as isize)
        .collect();
    let summed = if axes.is_empty() {
        t
    } else {
        t.sum_axes(&axes)?
    };
    let shape: Vec<isize> = target.iter().map(|&d| d as isize).collect();
    Ok(summed.into_reshape(&shape)?)
}

/// The gradient of a joint log probability with respect to latent nodes, keyed by node.
#[derive(Debug, Clone, Default)]
pub struct LogProbGradient {
    /// Indexed by node id (a dense table: gradients are taken over every latent, so this beats hashing).
    gradients: Vec<Option<DoubleTensor>>,
    len: usize,
}

impl LogProbGradient {
    fn with_capacity(nodes: usize) -> Self {
        Self {
            gradients: vec![None; nodes],
            len: 0,
        }
    }

    /// The gradient with respect to a node, or None if the log probability does not depend on it.
    pub fn get(&self, node: NodeId) -> Option<&DoubleTensor> {
        self.gradients.get(node.0).and_then(Option::as_ref)
    }

    /// Removes and returns the gradient with respect to a node (None if there is none), so it can be used
    /// without copying.
    pub fn take(&mut self, node: NodeId) -> Option<DoubleTensor> {
        let gradient = self.gradients.get_mut(node.0).and_then(Option::take);
        if gradient.is_some() {
            self.len -= 1;
        }
        gradient
    }

    /// Iterates over the entries, in node order.
    pub fn iter(&self) -> impl Iterator<Item = (NodeId, &DoubleTensor)> + '_ {
        self.gradients
            .iter()
            .enumerate()
            .filter_map(|(i, t)| t.as_ref().map(|t| (NodeId(i), t)))
    }

    /// The number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether there are no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The entries as a map.
    pub fn into_map(self) -> HashMap<NodeId, DoubleTensor> {
        self.gradients
            .into_iter()
            .enumerate()
            .filter_map(|(i, t)| t.map(|t| (NodeId(i), t)))
            .collect()
    }

    fn add(&mut self, id: NodeId, gradient: DoubleTensor) -> Result<()> {
        if id.0 >= self.gradients.len() {
            self.gradients.resize(id.0 + 1, None);
        }
        let slot = &mut self.gradients[id.0];
        *slot = Some(match slot.take() {
            Some(existing) => existing.add(&gradient)?,
            None => {
                self.len += 1;
                gradient
            }
        });
        Ok(())
    }
}

/// True for nodes that log prob gradients can be taken with respect to: unobserved f64 probabilistic nodes.
fn is_latent_double(graph: &Graph, state: &State, id: NodeId) -> bool {
    let node = &graph[id];
    node.is_probabilistic() && node.dtype() == DType::F64 && !state.is_observed(id)
}

/// The gradient of the total log probability of the probabilistic nodes `of` with respect to the latents in
/// `wrt`.
///
/// For each node in `of`: the derivative with respect to its own value is included if it is an unobserved f64
/// node, and the derivatives with respect to its parameters are propagated in reverse mode through
/// deterministic parents to the f64 latents in `wrt` (stopping at probabilistic nodes). Integer and boolean
/// latents in `wrt` are skipped.
pub fn log_prob_gradient(
    graph: &Graph,
    state: &State,
    of: &[NodeId],
    wrt: &[NodeId],
) -> Result<LogProbGradient> {
    let gradient = gradient_from_parts(graph, state, of, wrt, |node| {
        node_gradient_parts(graph, state, node).map(Some)
    })?;
    Ok(gradient.expect("every node has gradient parts"))
}

/// The total log probability of the probabilistic nodes `of` and its gradient with respect to the latents in
/// `wrt`, exactly as [`log_prob`] (summed in the order of `of`) and [`log_prob_gradient`], but in one pass that
/// reads each node's value and parameters and builds its distribution once.
///
/// None if any node's log probability is impossible (-infinity or NaN); the remaining nodes are not evaluated.
pub(crate) fn log_prob_and_gradient(
    graph: &Graph,
    state: &State,
    of: &[NodeId],
    wrt: &[NodeId],
) -> Result<Option<(f64, LogProbGradient)>> {
    let mut log_prob = 0.0;
    let gradient = gradient_from_parts(graph, state, of, wrt, |node| {
        let (node_log_prob, parts) = kind(graph, node)?
            .log_prob_and_gradient(value(state, node)?, &parameters(graph, state, node)?)?;
        log_prob += node_log_prob;
        Ok(parts)
    })?;
    Ok(gradient.map(|gradient| (log_prob, gradient)))
}

/// The gradient of [`log_prob_gradient`], given each node's gradient parts by `parts_of`. None as soon as
/// `parts_of` returns None for a node.
fn gradient_from_parts(
    graph: &Graph,
    state: &State,
    of: &[NodeId],
    wrt: &[NodeId],
    mut parts_of: impl FnMut(NodeId) -> Result<Option<LogProbGradientParts>>,
) -> Result<Option<LogProbGradient>> {
    graph.check_ids(of.iter().chain(wrt))?;
    let mut is_wrt: smallvec::SmallVec<[bool; 32]> = smallvec::smallvec![false; graph.len()];
    let mut any_target = false;
    for &id in wrt {
        if is_latent_double(graph, state, id) {
            is_wrt[id.0] = true;
            any_target = true;
        }
    }
    let mut result = LogProbGradient::with_capacity(graph.len());
    // every parameter's upstream gradient is seeded into one reverse sweep (reverse mode is linear, so this is the
    // same sum as one sweep per parameter)
    let mut seeds: smallvec::SmallVec<[(NodeId, DoubleTensor); 8]> = smallvec::SmallVec::new();
    for &node in of {
        let Some((value_gradient, parameters)) = parts_of(node)? else {
            return Ok(None);
        };
        if let (true, Some(dx)) = (is_latent_double(graph, state, node), value_gradient) {
            result.add(node, dx)?;
        }
        if !any_target {
            continue;
        }
        for (&parent, dp) in graph[node].parents().iter().zip(parameters) {
            let Some(dp) = dp else { continue };
            let parent_node = &graph[parent];
            // a constant can't lead to a latent, so there's nothing to propagate
            if parent_node.dtype() != DType::F64 || matches!(parent_node.op(), Op::Constant(_)) {
                continue;
            }
            seeds.push((parent, sum_to_shape(dp, graph[parent].shape())?));
        }
    }
    reverse_from_seeds(graph, state, seeds, &is_wrt, |latent, partial| {
        result.add(latent, partial)
    })?;
    Ok(Some(result))
}

/// Hessian-vector products of a log probability: by latent id, the derivative of its gradient along each of `b`
/// directions, shaped `[b, latent shape...]` (None where zero).
pub(crate) type LogProbHvp = Vec<Option<DoubleTensor>>;

/// The gradient of the total log probability of the probabilistic nodes `of` with respect to the f64 latents in
/// `wrt` and its derivative along `b` directions (its Hessian-vector products), by forward-over-reverse
/// differentiation. The directions are given as each latent's batched tangent `[b, latent shape...]` in
/// `tangents`; a latent without one has a zero tangent. Unlike [`log_prob_gradient`], a node's derivative with
/// respect to its own value is included only for nodes in `wrt`.
///
/// # Errors
///
/// [`GraphError::NoSecondOrderRule`] naming the node if a distribution whose log density depends on the directions
/// (through its value or a parameter) or a deterministic node between such a parameter and a latent has no
/// second-order rule.
pub(crate) fn log_prob_hvp(
    graph: &Graph,
    state: &State,
    of: &[NodeId],
    wrt: &[NodeId],
    tangents: impl IntoIterator<Item = (NodeId, DoubleTensor)>,
    b: usize,
) -> Result<(LogProbGradient, LogProbHvp)> {
    graph.check_ids(of.iter().chain(wrt))?;
    let mut is_wrt = vec![false; graph.len()];
    for &id in wrt {
        if is_latent_double(graph, state, id) {
            is_wrt[id.0] = true;
        }
    }
    let seeds: Vec<(NodeId, DoubleTensor)> = tangents
        .into_iter()
        .filter(|(id, _)| is_wrt.get(id.0).copied().unwrap_or(false))
        .collect();
    let node_tangents = forward_tangents(graph, state, seeds, b)?;
    let mut gradient = LogProbGradient::with_capacity(graph.len());
    let mut hvp: LogProbHvp = vec![None; graph.len()];
    let mut seeds: Vec<SecondOrderSeed> = Vec::new();
    for &node in of {
        let (value_gradient, parameter_gradients) = node_gradient_parts(graph, state, node)?;
        let parents = graph[node].parents();
        let value_tangent = node_tangents[node.0].as_ref();
        let parent_tangents: SmallVec<[Option<&DoubleTensor>; 4]> = parents
            .iter()
            .map(|p| node_tangents[p.0].as_ref())
            .collect();
        // a node whose value and parameters don't move along the directions has no second-order terms
        let (value_hvp, parameter_hvps) =
            if value_tangent.is_none() && parent_tangents.iter().all(Option::is_none) {
                (None, smallvec::smallvec![None; parents.len()])
            } else {
                kind(graph, node)?
                    .log_prob_hvp(
                        value(state, node)?,
                        &parameters(graph, state, node)?,
                        value_tangent,
                        &parent_tangents,
                        b,
                    )
                    .map_err(|e| match e {
                        GraphError::NoSecondOrderRule(m) => GraphError::NoSecondOrderRule(format!(
                            "{m} of node {}",
                            graph.describe(node)
                        )),
                        other => other,
                    })?
            };
        if is_wrt[node.0] {
            if let Some(dx) = value_gradient {
                gradient.add(node, dx)?;
            }
            if let Some(h) = value_hvp {
                accumulate(&mut hvp[node.0], h)?;
            }
        }
        for ((&parent, dp), hp) in parents.iter().zip(parameter_gradients).zip(parameter_hvps) {
            let parent_node = &graph[parent];
            // a constant can't lead to a latent, so there's nothing to propagate
            if parent_node.dtype() != DType::F64 || matches!(parent_node.op(), Op::Constant(_)) {
                continue;
            }
            let dp = dp
                .map(|dp| sum_to_shape(dp, parent_node.shape()))
                .transpose()?;
            seeds.push((parent, dp, hp));
        }
    }
    second_order_sweep(
        graph,
        state,
        &node_tangents,
        seeds,
        b,
        &is_wrt,
        |latent, g, h| {
            if let Some(g) = g {
                let shape: Vec<isize> = graph[latent].shape().iter().map(|&d| d as isize).collect();
                gradient.add(latent, g.into_reshape(&shape)?)?;
            }
            if let Some(h) = h {
                accumulate(&mut hvp[latent.0], h)?;
            }
            Ok(())
        },
    )?;
    Ok((gradient, hvp))
}

/// The gradient of the joint log probability of every probabilistic node with respect to every latent f64 node.
pub fn joint_log_prob_gradient(graph: &Graph, state: &State) -> Result<LogProbGradient> {
    let of: Vec<NodeId> = graph.probabilistic().collect();
    let wrt: Vec<NodeId> = of
        .iter()
        .copied()
        .filter(|&id| is_latent_double(graph, state, id))
        .collect();
    log_prob_gradient(graph, state, &of, &wrt)
}

/// The gradient of the log likelihood (observed nodes only) with respect to every latent f64 node.
pub fn log_likelihood_gradient(graph: &Graph, state: &State) -> Result<LogProbGradient> {
    let probabilistic: Vec<NodeId> = graph.probabilistic().collect();
    let of: Vec<NodeId> = probabilistic
        .iter()
        .copied()
        .filter(|&id| state.is_observed(id))
        .collect();
    let wrt: Vec<NodeId> = probabilistic
        .into_iter()
        .filter(|&id| is_latent_double(graph, state, id))
        .collect();
    log_prob_gradient(graph, state, &of, &wrt)
}
