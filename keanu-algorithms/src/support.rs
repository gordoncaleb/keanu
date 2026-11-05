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

/// Gradient based algorithms need at least one latent, and only continuous ones.
pub(crate) fn check_latents(model: &Model) -> Result<()> {
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

/// `count` independent draws from a fitted approximation (`draw` gives each in unconstrained space), recording
/// the values of `nodes` at each with the model's log probability there: what `LaplaceResult::sample` and
/// `AdviResult::sample` do.
pub(crate) fn sample_approximation(
    param: &Parameterisation<'_>,
    mut draw: impl FnMut() -> Vec<f64>,
    state: &mut State,
    nodes: &[NodeId],
    count: usize,
) -> Result<NetworkSamples> {
    if nodes.is_empty() {
        return Err(AlgorithmError::InvalidArgument("no nodes to sample".into()));
    }
    let model = param.model();
    check_nodes(model.graph(), nodes)?;
    let mut samples = NetworkSamples::new(model.graph(), nodes)?;
    for _ in 0..count {
        let z = draw();
        // a draw where the model can't be evaluated (a failed ODE solve) is recorded as impossible
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
