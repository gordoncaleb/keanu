//! Forward sampling: drawing samples from a model's prior.

use crate::error::{AlgorithmError, Result};
use crate::samples::NetworkSamples;
use crate::support::{MAX_REDRAWS, check_nodes};
use keanu_distributions::KeanuRng;
use keanu_graph::{
    Graph, GraphError, Model, NodeId, State, downstream_lambda_section, upstream_lambda_section,
};

/// Samples from the prior by sampling each latent in topological order and recomputing the deterministic nodes
/// in between.
///
/// Observations are kept as they are, so forward sampling only makes sense when no observation depends on a
/// latent; that case is an error.
///
/// A draw where the model can't be evaluated (a failed ODE solve, an impossible point: see
/// [`GraphError::is_impossible_point`]) is drawn again, as the samplers reject such points and the approximations
/// redraw them, so the samples are from the prior where the model can be evaluated. A thousand such draws in a row
/// are the solver's error.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ForwardSampler {
    /// Record the joint log probability of each sample (otherwise 0 is recorded).
    pub calculate_log_prob: bool,
}

/// Setters for each setting, for use as `ForwardSampler::default().with_...(...)`.
impl ForwardSampler {
    /// Sets [`calculate_log_prob`](Self::calculate_log_prob).
    #[must_use]
    pub fn with_calculate_log_prob(mut self, calculate_log_prob: bool) -> Self {
        self.calculate_log_prob = calculate_log_prob;
        self
    }
}

impl ForwardSampler {
    /// A forward sampler with the default settings, which record 0 rather than each sample's log probability
    /// (use [`with_calculate_log_prob`](Self::with_calculate_log_prob) to record it).
    pub fn new() -> Self {
        Self::default()
    }

    /// Draws `count` samples of `nodes`, leaving `state` at the last sample. Fails with
    /// [`GraphError::UnknownNode`] if a node is not in the model's graph.
    pub fn sample(
        &self,
        model: &Model,
        state: &mut State,
        nodes: &[NodeId],
        count: usize,
        rng: &mut KeanuRng,
    ) -> Result<NetworkSamples> {
        let graph = model.graph();
        check_nodes(graph, nodes)?;
        if model.latents().is_empty() {
            return Err(AlgorithmError::NoLatents);
        }

        // an observation with a random variable upstream of it can't be respected by sampling the prior
        let upstream_of_observed = upstream_lambda_section(graph, state, model.observed())?;
        if upstream_of_observed
            .latent_and_observed()
            .iter()
            .any(|&id| graph[id].is_probabilistic() && !state.is_observed(id))
        {
            return Err(AlgorithmError::InvalidArgument(
                "cannot forward sample when an observed node has a random variable upstream of it"
                    .into(),
            ));
        }

        let order = sampling_order(graph, state, model.latents(), nodes)?;

        // make sure every recorded node has a value, even those outside the sampling order (e.g. constants)
        state.lazy_eval(graph, nodes, rng)?;

        let mut samples = NetworkSamples::new(graph, nodes)?;
        let mut in_a_row = 0;
        while samples.len() < count {
            match self.draw(model, state, &order, rng) {
                Ok(log_prob) => {
                    in_a_row = 0;
                    samples.record(state, log_prob)?;
                }
                Err(e) if e.is_impossible_point() => {
                    in_a_row += 1;
                    if in_a_row == MAX_REDRAWS {
                        return Err(e.into());
                    }
                }
                Err(e) => return Err(e.into()),
            }
        }
        Ok(samples)
    }

    /// One draw: samples the nodes of `order` (in order) and gives the log probability to record.
    fn draw(
        &self,
        model: &Model,
        state: &mut State,
        order: &[NodeId],
        rng: &mut KeanuRng,
    ) -> std::result::Result<f64, GraphError> {
        let graph = model.graph();
        for &id in order {
            if state.is_observed(id) {
                continue;
            }
            // clearing and lazily evaluating samples a probabilistic node from its prior given its parents,
            // and recomputes a deterministic node from its parents
            state.clear(id)?;
            state.lazy_eval(graph, &[id], rng)?;
        }
        Ok(if self.calculate_log_prob {
            // only nodes the requested ones depend on were resampled, but the log probability covers the
            // whole model: recompute the deterministic nodes downstream of them so none is stale
            state.cascade(graph, order)?;
            model.log_prob(state)?
        } else {
            0.0
        })
    }
}

/// The nodes downstream of the latents that the requested nodes depend on, in topological (id) order.
fn sampling_order(
    graph: &Graph,
    state: &State,
    latents: &[NodeId],
    nodes: &[NodeId],
) -> Result<Vec<NodeId>> {
    let downstream = downstream_lambda_section(graph, state, latents)?;

    // everything the requested nodes depend on, including themselves
    let mut upstream = vec![false; graph.len()];
    let mut stack: Vec<NodeId> = nodes.to_vec();
    while let Some(id) = stack.pop() {
        if !std::mem::replace(&mut upstream[id.index()], true) {
            stack.extend_from_slice(graph[id].parents());
        }
    }

    Ok(downstream
        .nodes()
        .iter()
        .copied()
        .filter(|id| upstream[id.index()])
        .collect())
}
