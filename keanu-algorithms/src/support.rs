//! Small helpers shared by the algorithms.

use crate::error::{AlgorithmError, Result};
use keanu_distributions::KeanuRng;
use keanu_graph::{Graph, GraphError, NodeId, State, Value};
use keanu_tensor::DoubleTensor;

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

/// `values` (one f64 tensor per node) paired with the ids of `nodes`, ready to set in a state.
pub(crate) fn f64_values<'a>(
    nodes: &'a [NodeId],
    values: impl IntoIterator<Item = DoubleTensor> + 'a,
) -> impl Iterator<Item = (NodeId, Value)> + 'a {
    nodes
        .iter()
        .copied()
        .zip(values.into_iter().map(Value::F64))
}
