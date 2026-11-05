//! Choosing a sampler for a model.

use crate::error::Result;
use crate::mcmc::SamplingOptions;
use crate::mcmc::compound::Compound;
use crate::mcmc::gibbs::Gibbs;
use crate::mcmc::metropolis::MetropolisHastings;
use crate::mcmc::nuts::Nuts;
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
        .filter(|&&id| graph[id].is_probabilistic())
        .all(|&id| graph[id].dtype() == DType::F64 || state.is_observed(id));
    if !all_continuous_or_observed {
        return false;
    }

    let mut constant_cache: HashSet<NodeId> = HashSet::new();
    let mut visited: HashSet<NodeId> = HashSet::new();
    let mut queue: Vec<NodeId> = nodes
        .iter()
        .flat_map(|&id| graph[id].parents().iter().copied())
        .collect();
    while let Some(id) = queue.pop() {
        if !visited.insert(id) {
            continue;
        }
        if !is_differentiable(graph, id) && !is_constant(graph, state, id, &mut constant_cache) {
            return false;
        }
        if !graph[id].is_probabilistic() {
            queue.extend_from_slice(graph[id].parents());
        }
    }
    true
}

/// True if the joint log probability can be differentiated with respect to the continuous latents `latents` with
/// every other latent held fixed: each of them is a continuous latent of the model, and every deterministic node
/// between them and a node of the log probability (a latent, an observed node or a factor) whose value depends on
/// them is differentiable. Operations that depend only on other latents (a comparison of a discrete change point
/// with the time, say) don't need to be. With every continuous latent of a model without discrete latents, this is
/// [`is_differentiable_wrt_latents`].
pub fn is_differentiable_wrt(model: &Model, state: &State, latents: &[NodeId]) -> bool {
    let graph = model.graph();
    let continuous = model.continuous_latents();
    if latents.is_empty() || latents.iter().any(|id| !continuous.contains(id)) {
        return false;
    }
    // the nodes whose value depends on the latents, through deterministic unobserved nodes (ids are topological)
    let mut depends = vec![false; graph.len()];
    for &id in latents {
        depends[id.index()] = true;
    }
    for id in graph.ids() {
        let node = &graph[id];
        if node.is_probabilistic() || state.is_observed(id) {
            continue;
        }
        depends[id.index()] = node.parents().iter().any(|p| depends[p.index()]);
    }
    // the deterministic nodes the log probability is computed from
    let mut visited: HashSet<NodeId> = HashSet::new();
    let mut queue: Vec<NodeId> = model
        .latents()
        .iter()
        .chain(model.observed())
        .flat_map(|&id| graph[id].parents().iter().copied())
        .chain(model.factors().iter().copied())
        .collect();
    while let Some(id) = queue.pop() {
        if !visited.insert(id) || graph[id].is_probabilistic() || state.is_observed(id) {
            continue;
        }
        if depends[id.index()] && !is_differentiable(graph, id) {
            return false;
        }
        queue.extend_from_slice(graph[id].parents());
    }
    true
}

fn known_constant(graph: &Graph, state: &State, id: NodeId, cache: &HashSet<NodeId>) -> bool {
    matches!(graph[id].op(), Op::Constant(_)) || state.is_observed(id) || cache.contains(&id)
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
        if graph[v].is_probabilistic() && !state.is_observed(v) {
            return false;
        }
        visited.push(v);
        if !known_constant(graph, state, v, cache) {
            queue.extend_from_slice(graph[v].parents());
        }
    }
    cache.extend(visited);
    true
}

/// A sampler chosen to suit a model ([`Sampler::for_model`]), or given by the caller.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Sampler {
    /// The No-U-Turn sampler.
    Nuts(Nuts),
    /// Metropolis–Hastings.
    MetropolisHastings(MetropolisHastings),
    /// Gibbs sampling of discrete latents.
    Gibbs(Gibbs),
    /// A compound sampler: a cycle of steps over blocks of the latents.
    Compound(Compound),
}

impl Sampler {
    /// The default sampler for a model in this state, each with its default settings:
    ///
    /// - every latent continuous: NUTS when the log probability is differentiable with respect to every latent
    ///   ([`is_differentiable_wrt_latents`]), otherwise Metropolis–Hastings;
    /// - every latent discrete, with finitely many values per element: [`Gibbs`] over all of them;
    /// - otherwise (both kinds, or discrete latents Gibbs can't enumerate): [`Compound::for_model`], typically
    ///   Gibbs or discrete random-walk steps for the discrete latents and NUTS for the continuous ones.
    pub fn for_model(model: &Model, state: &State) -> Self {
        let discrete = model.discrete_latents();
        if discrete.is_empty() {
            return if is_differentiable_wrt_latents(model, state) {
                Self::Nuts(Nuts::default())
            } else {
                Self::MetropolisHastings(MetropolisHastings::default())
            };
        }
        let compound = Compound::for_model(model, state);
        match compound.steps() {
            [crate::mcmc::compound::CompoundStep::Gibbs(gibbs)] => Self::Gibbs(gibbs.clone()),
            _ => Self::Compound(compound),
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
            Self::Nuts(nuts) => Ok(nuts
                .sample_with_statistics(model, state, nodes, count, options, rng)?
                .0),
            Self::MetropolisHastings(mh) => {
                mh.sample_with(model, state, nodes, count, options, rng)
            }
            Self::Gibbs(gibbs) => gibbs.sample_with(model, state, nodes, count, options, rng),
            Self::Compound(compound) => Ok(compound
                .sample_with_statistics(model, state, nodes, count, options, rng)?
                .0),
        }
    }
}
