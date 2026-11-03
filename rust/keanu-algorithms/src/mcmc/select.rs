//! Choosing a sampler for a model (Java's `Keanu.Sampling.MCMC.withDefaultConfigFor` and `DifferentiableChecker`).

use crate::error::Result;
use crate::mcmc::metropolis::MetropolisHastings;
use crate::mcmc::nuts::Nuts;
use crate::mcmc::{SamplingOptions, generate};
use crate::samples::NetworkSamples;
use keanu_distributions::KeanuRng;
use keanu_graph::op::Op;
use keanu_graph::{DType, Graph, Model, NodeId, State, is_differentiable};
use std::collections::HashSet;

/// True if the joint log probability can be differentiated with respect to every latent: every unobserved
/// probabilistic node is continuous, and between each probabilistic node and its probabilistic ancestors there are
/// only differentiable operations, or operations whose value can't change (they depend only on constants and
/// observations).
pub fn is_differentiable_wrt_latents(model: &Model, state: &State) -> bool {
    let graph = model.graph();
    let nodes: Vec<NodeId> = model
        .latents()
        .iter()
        .chain(model.observed())
        .copied()
        .collect();
    let all_continuous_or_observed = nodes
        .iter()
        .filter(|&&id| graph.node(id).is_probabilistic())
        .all(|&id| graph.node(id).dtype() == DType::F64 || state.is_observed(id));
    if !all_continuous_or_observed {
        return false;
    }

    let mut constant_cache: HashSet<NodeId> = HashSet::new();
    let mut visited: HashSet<NodeId> = HashSet::new();
    let mut queue: Vec<NodeId> = nodes
        .iter()
        .flat_map(|&id| graph.parents(id).iter().copied())
        .collect();
    while let Some(id) = queue.pop() {
        if !visited.insert(id) {
            continue;
        }
        if !is_differentiable(graph, id) && !is_constant(graph, state, id, &mut constant_cache) {
            return false;
        }
        if !graph.node(id).is_probabilistic() {
            queue.extend_from_slice(graph.parents(id));
        }
    }
    true
}

fn known_constant(graph: &Graph, state: &State, id: NodeId, cache: &HashSet<NodeId>) -> bool {
    matches!(graph.node(id).op(), Op::Constant(_)) || state.is_observed(id) || cache.contains(&id)
}

/// True if a node's value can't change: no unobserved probabilistic node is upstream of it.
fn is_constant(graph: &Graph, state: &State, id: NodeId, cache: &mut HashSet<NodeId>) -> bool {
    if known_constant(graph, state, id, cache) {
        return true;
    }
    let mut visited: Vec<NodeId> = Vec::new();
    let mut seen: HashSet<NodeId> = HashSet::new();
    let mut queue = vec![id];
    while let Some(v) = queue.pop() {
        if !seen.insert(v) {
            continue;
        }
        if graph.node(v).is_probabilistic() && !state.is_observed(v) {
            return false;
        }
        visited.push(v);
        if !known_constant(graph, state, v, cache) {
            queue.extend_from_slice(graph.parents(v));
        }
    }
    cache.extend(visited);
    true
}

/// A sampler chosen to suit a model: NUTS when the log probability is differentiable with respect to every latent,
/// otherwise Metropolis–Hastings, each with its default settings.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Sampler {
    /// The No-U-Turn sampler.
    Nuts(Nuts),
    /// Metropolis–Hastings.
    MetropolisHastings(MetropolisHastings),
}

impl Sampler {
    /// The default sampler for a model in this state.
    pub fn for_model(model: &Model, state: &State) -> Self {
        if is_differentiable_wrt_latents(model, state) {
            Self::Nuts(Nuts::default())
        } else {
            Self::MetropolisHastings(MetropolisHastings::default())
        }
    }

    /// Runs a chain of `count` steps, dropping and down-sampling as `options` says.
    pub fn sample(
        &self,
        model: &Model,
        state: &mut State,
        nodes: &[NodeId],
        count: usize,
        options: SamplingOptions,
        rng: &mut KeanuRng,
    ) -> Result<NetworkSamples> {
        match self {
            Self::Nuts(nuts) => {
                let mut sampler = nuts.sampler(model, state, nodes, rng)?;
                generate(&mut sampler, state, count, options, rng)
            }
            Self::MetropolisHastings(mh) => {
                mh.sample_with(model, state, nodes, count, options, rng)
            }
        }
    }
}
