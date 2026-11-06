//! Small helpers shared by the algorithms.

use crate::error::{AlgorithmError, Result};
use crate::samples::NetworkSamples;
use keanu_distributions::KeanuRng;
use keanu_graph::{Graph, GraphError, Model, NodeId, Parameterisation, Space, State, Value};
use keanu_tensor::DoubleTensor;

/// The dot product of two vectors of the same length.
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The Euclidean norm of a vector.
pub(crate) fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// A node's name as an [`AlgorithmError`] carries it (in [`AlgorithmError::DiscreteLatents`] and
/// [`AlgorithmError::ModeOnBoundary`]): its label, unquoted, or `#` and its index. Messages name nodes with
/// [`Graph::describe`] instead, which quotes the label.
pub(crate) fn node_name(graph: &Graph, id: NodeId) -> String {
    match graph.node(id).and_then(|node| node.label()) {
        Some(label) => label.to_owned(),
        None => format!("#{}", id.index()),
    }
}

/// [`GraphError::NoLikelihood`] if the model has a simulator node: every algorithm but approximate Bayesian
/// computation needs the model's likelihood.
pub(crate) fn check_likelihood(model: &Model) -> Result<()> {
    Ok(model.check_likelihood()?)
}

/// Gradient based algorithms need a likelihood ([`check_likelihood`]), at least one latent, and only continuous
/// ones.
pub(crate) fn check_latents(model: &Model) -> Result<()> {
    check_likelihood(model)?;
    if model.latents().is_empty() {
        return Err(AlgorithmError::NoLatents);
    }
    let discrete = model.discrete_latents();
    if !discrete.is_empty() {
        let graph = model.graph();
        let names = discrete.iter().map(|&id| node_name(graph, id)).collect();
        return Err(AlgorithmError::DiscreteLatents(names));
    }
    Ok(())
}

/// What [`sample_approximation`] does with a draw where the model can't be evaluated (a failed ODE solve), whose
/// nodes have no values to record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Unevaluable {
    /// Draw again in its place, so that `count` samples are recorded: for approximations that can draw afresh
    /// (Laplace, ADVI). [`MAX_REDRAWS`] such draws in a row are the solver's error.
    Redraw,
    /// Leave it out, so fewer samples are recorded: for a fixed set of draws (Pathfinder's).
    Skip,
}

/// The most draws in a row that [`sample_approximation`] redraws because the model can't be evaluated there.
pub(crate) const MAX_REDRAWS: usize = 1000;

/// `count` draws from a fitted approximation (`draw` gives each in unconstrained space, `None` when there are no
/// more), recording the values of `nodes` at each with the model's log probability there: what
/// `LaplaceResult::sample`, `AdviResult::sample` and `PathfinderResult::sample` do. A draw where the model can't be
/// evaluated ([`GraphError::is_impossible_point`], a failed ODE solve) has no values to record, and is redrawn or
/// skipped as `unevaluable` says; draws that evaluate are recorded even where their log probability is −∞.
pub(crate) fn sample_approximation(
    param: &Parameterisation<'_>,
    mut draw: impl FnMut() -> Option<Vec<f64>>,
    state: &mut State,
    nodes: &[NodeId],
    count: usize,
    unevaluable: Unevaluable,
) -> Result<NetworkSamples> {
    if nodes.is_empty() {
        return Err(AlgorithmError::InvalidArgument("no nodes to sample".into()));
    }
    let model = param.model();
    check_nodes(model.graph(), nodes)?;
    let mut samples = NetworkSamples::new(model.graph(), nodes)?;
    let mut in_a_row = 0;
    while samples.len() < count {
        let Some(z) = draw() else { break };
        match param.to_state(&z, state) {
            Ok(()) => in_a_row = 0,
            Err(e) if e.is_impossible_point() => {
                in_a_row += 1;
                if unevaluable == Unevaluable::Redraw && in_a_row == MAX_REDRAWS {
                    return Err(e.into());
                }
                continue;
            }
            Err(e) => return Err(e.into()),
        }
        let log_prob = param.log_prob(&z, false, state)?;
        samples.record(state, log_prob)?;
    }
    Ok(samples)
}

/// [`GraphError::UnknownNode`] for the first of `nodes` that is not a node of the graph.
pub(crate) fn check_nodes(graph: &Graph, nodes: &[NodeId]) -> Result<()> {
    match nodes.iter().find(|&&id| !graph.contains(id)) {
        Some(&id) => Err(GraphError::UnknownNode(id).into()),
        None => Ok(()),
    }
}

/// Gives every node of the graph a value: latents without one are sampled from their prior and deterministic
/// nodes are computed. Samplers need this because a latent with a value can still have deterministic parents
/// without one.
pub(crate) fn evaluate_all(graph: &Graph, state: &mut State, rng: &mut KeanuRng) -> Result<()> {
    let all: Vec<NodeId> = graph.ids().collect();
    state.lazy_eval(graph, &all, rng)?;
    Ok(())
}

/// Computes every deterministic node that has no value but can be computed without sampling: one whose ancestors,
/// back to the nearest nodes with values, are all deterministic. Deterministic nodes of constants only (such as the
/// parts of a Gaussian process kernel that depend only on its inputs) have no value after `build`, and the
/// optimizers, which take no random number generator, need them as [`evaluate_all`] gives them to the samplers.
/// Latents without a value are left without one.
pub(crate) fn compute_deterministic(graph: &Graph, state: &mut State) -> Result<()> {
    // ids are in topological order, so each node's parents are decided before it
    let mut computable = vec![false; graph.ids().len()];
    let mut missing = Vec::new();
    for id in graph.ids() {
        let node = &graph[id];
        let has_value = state.has_value(id);
        computable[id.index()] = has_value
            || (!node.is_probabilistic() && node.parents().iter().all(|p| computable[p.index()]));
        if computable[id.index()] && !has_value {
            missing.push(id);
        }
    }
    if !missing.is_empty() {
        // every ancestor the lazy plan visits has a value or is deterministic, so nothing is sampled and the
        // generator is never used
        state.lazy_eval(graph, &missing, &mut KeanuRng::seed_from_u64(0))?;
    }
    Ok(())
}

/// The current values of `nodes`, paired with their ids.
pub(crate) fn current_values(state: &State, nodes: &[NodeId]) -> Result<Vec<(NodeId, Value)>> {
    nodes
        .iter()
        .map(|&id| {
            state
                .value(id)
                .cloned()
                .map(|v| (id, v))
                .ok_or(AlgorithmError::Graph(GraphError::MissingValue(id)))
        })
        .collect()
}

/// The shape of each latent's block of a parameterisation's flat vector, in layout order: the latent's own shape in
/// [`Space::Constrained`] (so algorithms that work on one tensor per latent see exactly the latents' values), and a
/// vector of its unconstrained coordinates in [`Space::Unconstrained`].
pub(crate) fn block_shapes(param: &Parameterisation<'_>) -> Vec<Vec<usize>> {
    let graph = param.model().graph();
    param
        .latents()
        .iter()
        .map(|&id| match param.space() {
            Space::Constrained => graph[id].shape().to_vec(),
            Space::Unconstrained => vec![param.range(id).map_or(0, |r| r.len())],
        })
        .collect()
}

/// The flat vector `z` as one tensor per block of `shapes`.
pub(crate) fn split(z: &[f64], shapes: &[Vec<usize>]) -> Vec<DoubleTensor> {
    let mut offset = 0;
    shapes
        .iter()
        .map(|shape| {
            let len: usize = shape.iter().product();
            let tensor = DoubleTensor::new(shape, z[offset..offset + len].to_vec())
                .expect("a block's shape matches its length");
            offset += len;
            tensor
        })
        .collect()
}

/// The blocks concatenated into one flat vector: the inverse of [`split`].
pub(crate) fn concat(blocks: &[DoubleTensor]) -> Vec<f64> {
    blocks
        .iter()
        .flat_map(|t| t.as_slice().iter().copied())
        .collect()
}

/// The point of the latents' current values. A value outside its latent's support (or on its boundary, in
/// unconstrained space) has no point, which is an impossible start.
pub(crate) fn start_point(param: &Parameterisation<'_>, state: &State) -> Result<Vec<f64>> {
    param.from_state(state).map_err(|e| match e {
        GraphError::Transform(_) => AlgorithmError::ImpossibleStart,
        other => other.into(),
    })
}

/// The latents (unobserved probabilistic nodes) among the ancestors of `targets`, up to the nearest probabilistic
/// ones, in id order.
pub(crate) fn latent_ancestors(graph: &Graph, state: &State, targets: &[NodeId]) -> Vec<NodeId> {
    let mut seen = vec![false; graph.len()];
    let mut stack = targets.to_vec();
    let mut latents = Vec::new();
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut seen[id.index()], true) {
            continue;
        }
        let node = &graph[id];
        if node.is_probabilistic() {
            if !state.is_observed(id) {
                latents.push(id);
            }
        } else {
            stack.extend(node.parents().iter().copied());
        }
    }
    latents.sort();
    latents
}
