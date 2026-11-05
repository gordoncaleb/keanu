//! Small helpers shared by the algorithms.

use crate::error::{AlgorithmError, Result};
use keanu_distributions::KeanuRng;
use keanu_graph::{Graph, GraphError, NodeId, Parameterisation, Space, State, Value};
use keanu_tensor::DoubleTensor;

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
