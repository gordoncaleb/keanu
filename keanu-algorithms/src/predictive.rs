//! Predictive sampling: draws of observed nodes as the model would generate them, either from the prior or given
//! each draw of a posterior. Posterior draws can also be replayed on a separate prediction graph.

use crate::error::{AlgorithmError, Result};
use crate::samples::NetworkSamples;
use crate::support::{MAX_REDRAWS, check_nodes};
use keanu_distributions::KeanuRng;
use keanu_graph::{Graph, GraphError, Intervened, Model, NodeId, State};
use std::collections::{HashMap, HashSet};

/// Draws prior- and posterior-predictive samples of observed nodes (the "targets").
///
/// - [`posterior`](Self::posterior) takes posterior samples of the latents (from MCMC, say) and, for each, draws
///   the targets from the model given those latent values: the posterior predictive distribution, for checking
///   the model against the data it was fitted to.
/// - [`prior`](Self::prior) draws the latents from their priors and then the targets: the prior predictive
///   distribution, for checking that the priors generate plausible data.
///
/// Both work on a copy of the state, so the targets stay observed with their values. The draws are returned as
/// [`NetworkSamples`] of the targets, with a log probability of 0 recorded for each.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct PredictiveSampler {
    /// The number of posterior draws to use, evenly spaced through the posterior samples; `None` (the default)
    /// uses every sample. Used when predicting from posterior draws.
    pub draws: Option<usize>,
}

/// Setters for each setting, for use as `PredictiveSampler::default().with_...(...)`.
impl PredictiveSampler {
    /// Sets [`draws`](Self::draws) to `Some(draws)`.
    #[must_use]
    pub fn with_draws(mut self, draws: usize) -> Self {
        self.draws = Some(draws);
        self
    }
}

impl PredictiveSampler {
    /// A predictive sampler with the default settings, which use every posterior sample.
    pub fn new() -> Self {
        Self::default()
    }

    /// One draw of `targets` for each posterior sample (or for [`draws`](Self::draws) of them, evenly spaced).
    ///
    /// Each draw sets the sampled nodes' values in a copy of `state`, recomputes the deterministic nodes the
    /// targets depend on and samples the targets from their distributions given their parents. Observed nodes
    /// other than the targets keep their observed values; a target downstream of another target is drawn given
    /// that target's draw.
    ///
    /// Fails if `targets` is empty, if a target is not a node of the model's graph
    /// ([`GraphError::UnknownNode`]) or not an observed probabilistic node, if `samples` doesn't hold every latent
    /// the targets depend on, or if [`draws`](Self::draws) is zero or more than the number of samples. The latents
    /// of each draw are given, so a draw where the model can't be evaluated at them (a failed ODE solve, which a
    /// sampler never records) can't be drawn again, and is the solver's error
    /// ([`GraphError::SolverFailed`]).
    pub fn posterior(
        &self,
        model: &Model,
        state: &State,
        samples: &NetworkSamples,
        targets: &[NodeId],
        rng: &mut KeanuRng,
    ) -> Result<NetworkSamples> {
        let graph = model.graph();
        check_targets(graph, state, targets)?;
        let redrawn = vec![false; graph.len()];
        self.draw(graph, state, samples, targets, &redrawn, rng)
    }

    /// Replay posterior draws on a separate graph, for example one built on held-out data of a different shape.
    ///
    /// Each binding pairs a sampled source node with an unobserved probabilistic node of the prediction graph.
    /// Values must match the destination's type and shape. Targets are node IDs in the prediction graph.
    /// Deterministic targets are evaluated and probabilistic targets are drawn given their parents. A target
    /// cannot also be bound. Other observations keep their values. To draw a new latent upstream of a target,
    /// include it in the targets. Every other required latent must be bound.
    ///
    /// This uses a copy of `state`. Fails for an empty target list, an unknown target or destination, an absent
    /// source, repeated binding endpoints, a missing required binding, or an evaluation error.
    pub fn predict(
        &self,
        graph: &Graph,
        state: &State,
        samples: &NetworkSamples,
        bindings: &[(NodeId, NodeId)],
        targets: &[NodeId],
        rng: &mut KeanuRng,
    ) -> Result<NetworkSamples> {
        if targets.is_empty() {
            return Err(AlgorithmError::InvalidArgument(
                "no target nodes to draw".into(),
            ));
        }
        check_nodes(graph, targets)?;
        let sampled: HashSet<_> = samples.nodes().iter().copied().collect();
        let mut mapping = HashMap::with_capacity(bindings.len());
        for &(source, destination) in bindings {
            graph
                .node(destination)
                .ok_or(GraphError::UnknownNode(destination))?;
            if !sampled.contains(&source) || mapping.insert(source, destination).is_some() {
                return Err(AlgorithmError::InvalidArgument(format!(
                    "prediction source {source:?} must occur in the samples and be bound only once"
                )));
            }
            if !is_latent(graph, state, destination) || targets.contains(&destination) {
                return Err(AlgorithmError::InvalidArgument(format!(
                    "prediction destination {} must be an unobserved probabilistic node outside the targets",
                    graph.describe(destination)
                )));
            }
        }
        let mapped = samples.map_nodes(graph, |id| mapping.get(&id).copied())?;
        let redrawn = vec![false; graph.len()];
        self.draw(graph, state, &mapped, targets, &redrawn, rng)
    }

    /// [`posterior`](Self::posterior), with the latents marked in `redrawn` (by index) drawn again for each draw,
    /// from their distributions given their parents, rather than taken from `samples`.
    fn draw(
        &self,
        graph: &Graph,
        state: &State,
        samples: &NetworkSamples,
        targets: &[NodeId],
        redrawn: &[bool],
        rng: &mut KeanuRng,
    ) -> Result<NetworkSamples> {
        check_nodes(graph, samples.nodes())?;
        let indices = self.thinned(samples.len())?;

        let mut work = state.clone();
        for &id in targets {
            work.unobserve(id)?;
        }
        let sampled = membership(graph, samples.nodes());
        // Observations and carried latent draws are fixed, so their parents do not affect prediction.
        let ancestors = ancestors(graph, targets, |id| {
            work.is_observed(id)
                || (is_latent(graph, &work, id)
                    && sampled[id.index()]
                    && !redrawn[id.index()]
                    && !is_target(targets, id))
        });
        let missing: Vec<String> = ancestors
            .iter()
            .filter(|&&id| {
                is_latent(graph, &work, id)
                    && !is_target(targets, id)
                    && !sampled[id.index()]
                    && !redrawn[id.index()]
            })
            .map(|&id| graph.describe(id))
            .collect();
        if !missing.is_empty() {
            return Err(AlgorithmError::InvalidArgument(format!(
                "the targets depend on latents without posterior samples: {}",
                missing.join(", ")
            )));
        }
        // everything the targets depend on that isn't fixed by an observation or a posterior sample
        let to_redraw: Vec<NodeId> = ancestors
            .iter()
            .copied()
            .filter(|&id| {
                !work.is_observed(id)
                    && (!graph[id].is_probabilistic()
                        || is_target(targets, id)
                        || redrawn[id.index()])
            })
            .collect();

        let mut predictive = NetworkSamples::new(graph, targets)?;
        for index in indices {
            let sample = samples
                .sample(index)
                .ok_or_else(|| AlgorithmError::InvalidArgument("sample out of range".into()))?;
            for &id in samples.nodes() {
                if let Some(value) = sample.get(id) {
                    work.set(graph, id, value)?;
                }
            }
            redraw(graph, &mut work, &to_redraw, targets, rng)?;
            predictive.record(&work, 0.0)?;
        }
        Ok(predictive)
    }

    /// The posterior predictive under an intervention: one draw of `targets` for each posterior sample of the
    /// original model (or for [`draws`](Self::draws) of them), on the graph that
    /// [`intervene`](keanu_graph::intervene) made from it.
    ///
    /// `model` and `state` are the intervened graph's, and `intervened` the rewrite's record. `samples` are draws of
    /// the original model's latents (from NUTS on it, say), and `targets` are nodes of the original graph; both are
    /// carried over to the intervened graph ([`Intervened::carried`], [`NetworkSamples::map_nodes`]), and the draws
    /// are made as [`posterior`](Self::posterior) makes them, except that the latents downstream of an intervened
    /// node ([`Intervened::downstream`]), whose posterior draws were made given its old value, are drawn again in
    /// dependency order from their distributions given their parents. So each draw keeps the posterior's values of
    /// the parameters and the other latents, sets the intervened nodes to their values, draws the latents they
    /// affect (a mediator, say) given them, and then the targets: the interventional distribution
    /// p(targets | do(x), data). The returned samples are of the targets' nodes in the intervened graph
    /// (`intervened.node(target)`).
    ///
    /// Fails if a target was intervened on or dropped by the rewrite, and as [`posterior`](Self::posterior)
    /// otherwise.
    pub fn interventional(
        &self,
        model: &Model,
        state: &State,
        intervened: &Intervened,
        samples: &NetworkSamples,
        targets: &[NodeId],
        rng: &mut KeanuRng,
    ) -> Result<NetworkSamples> {
        let targets = targets
            .iter()
            .map(|&id| {
                let kept = intervened
                    .node(id)
                    .filter(|_| !intervened.is_intervened(id));
                kept.ok_or_else(|| {
                    AlgorithmError::InvalidArgument(format!(
                        "target #{} was {} by the intervention, so it has no draws",
                        id.index(),
                        if intervened.is_intervened(id) {
                            "set"
                        } else {
                            "dropped"
                        }
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let samples = samples.map_nodes(model.graph(), |id| intervened.carried(id))?;
        let mut redrawn = vec![false; model.graph().len()];
        for id in intervened.downstream() {
            if let Some(new) = intervened
                .node(id)
                .filter(|&new| new.index() < redrawn.len())
            {
                redrawn[new.index()] = true;
            }
        }
        check_targets(model.graph(), state, &targets)?;
        self.draw(model.graph(), state, &samples, &targets, &redrawn, rng)
    }

    /// `count` draws of `targets` from the prior: each draw samples the latents the targets depend on from their
    /// priors, recomputes the deterministic nodes in between and samples the targets.
    ///
    /// Observed nodes other than the targets keep their observed values, as in the
    /// [`ForwardSampler`](crate::ForwardSampler), so an observed node that a target depends on must not itself
    /// depend on a latent (make it a target too); observed nodes the targets don't depend on are ignored.
    ///
    /// A draw where the model can't be evaluated (a failed ODE solve) is drawn again, as the
    /// [`ForwardSampler`](crate::ForwardSampler) does, so the draws are from the prior where the model can be
    /// evaluated; a thousand such draws in a row are the solver's error.
    ///
    /// Fails if `targets` is empty, if a target is not a node of the model's graph
    /// ([`GraphError::UnknownNode`]) or not an observed probabilistic node, or if an observed node the targets
    /// depend on depends on a latent.
    pub fn prior(
        &self,
        model: &Model,
        state: &State,
        targets: &[NodeId],
        count: usize,
        rng: &mut KeanuRng,
    ) -> Result<NetworkSamples> {
        let graph = model.graph();
        check_targets(graph, state, targets)?;

        let mut work = state.clone();
        for &id in targets {
            work.unobserve(id)?;
        }
        let ancestors = ancestors(graph, targets, |_| false);
        // whether each ancestor depends on a latent; parents have smaller ids, so they are settled first
        let mut random = vec![false; graph.len()];
        for &id in &ancestors {
            random[id.index()] = graph[id]
                .parents()
                .iter()
                .any(|&p| random[p.index()] || is_latent(graph, &work, p));
        }
        let conflicting: Vec<String> = ancestors
            .iter()
            .filter(|&&id| work.is_observed(id) && random[id.index()])
            .map(|&id| graph.describe(id))
            .collect();
        if !conflicting.is_empty() {
            return Err(AlgorithmError::InvalidArgument(format!(
                "the targets depend on observed nodes that depend on a latent, so a prior draw can't keep their \
                 observed values; make them targets too: {}",
                conflicting.join(", ")
            )));
        }
        let to_redraw: Vec<NodeId> = ancestors
            .iter()
            .copied()
            .filter(|&id| !work.is_observed(id))
            .collect();

        let mut predictive = NetworkSamples::new(graph, targets)?;
        let mut in_a_row = 0;
        while predictive.len() < count {
            match redraw(graph, &mut work, &to_redraw, targets, rng) {
                Ok(()) => {
                    in_a_row = 0;
                    predictive.record(&work, 0.0)?;
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
        Ok(predictive)
    }

    /// The indices of the posterior samples to use.
    fn thinned(&self, len: usize) -> Result<Vec<usize>> {
        match self.draws {
            None => Ok((0..len).collect()),
            Some(draws) if draws == 0 || draws > len => {
                Err(AlgorithmError::InvalidSetting(format!(
                    "cannot use {draws} draws of {len} posterior samples; use between 1 and {len}"
                )))
            }
            // evenly spaced, starting with the first; no overflow as draws <= len
            Some(draws) => Ok((0..draws).map(|i| i * len / draws).collect()),
        }
    }
}

/// An error unless the targets are a non-empty list of observed probabilistic nodes of the graph.
fn check_targets(graph: &Graph, state: &State, targets: &[NodeId]) -> Result<()> {
    if targets.is_empty() {
        return Err(AlgorithmError::InvalidArgument(
            "no target nodes to draw".into(),
        ));
    }
    check_nodes(graph, targets)?;
    match targets
        .iter()
        .find(|&&id| !graph[id].is_probabilistic() || !state.is_observed(id))
    {
        Some(&id) => Err(AlgorithmError::InvalidArgument(format!(
            "predictive targets must be observed probabilistic nodes, but {} is not",
            graph.describe(id)
        ))),
        None => Ok(()),
    }
}

/// The targets and every node they depend on, in topological (id) order, not walking past nodes for which
/// `stop` holds (which are included).
fn ancestors(graph: &Graph, targets: &[NodeId], stop: impl Fn(NodeId) -> bool) -> Vec<NodeId> {
    let mut seen = vec![false; graph.len()];
    let mut stack = targets.to_vec();
    while let Some(id) = stack.pop() {
        if !std::mem::replace(&mut seen[id.index()], true) && !stop(id) {
            stack.extend_from_slice(graph[id].parents());
        }
    }
    graph.ids().filter(|id| seen[id.index()]).collect()
}

/// For each node of the graph, whether it is one of `nodes`.
fn membership(graph: &Graph, nodes: &[NodeId]) -> Vec<bool> {
    let mut member = vec![false; graph.len()];
    for id in nodes {
        member[id.index()] = true;
    }
    member
}

fn is_latent(graph: &Graph, state: &State, id: NodeId) -> bool {
    graph[id].is_probabilistic() && !state.is_observed(id)
}

fn is_target(targets: &[NodeId], id: NodeId) -> bool {
    targets.contains(&id)
}

/// Clears `nodes` (in topological order) and recomputes the targets, which samples each cleared probabilistic node
/// from its distribution given its parents and recomputes each cleared deterministic node.
fn redraw(
    graph: &Graph,
    state: &mut State,
    nodes: &[NodeId],
    targets: &[NodeId],
    rng: &mut KeanuRng,
) -> std::result::Result<(), GraphError> {
    for &id in nodes {
        state.clear(id)?;
    }
    state.lazy_eval(graph, targets, rng)?;
    Ok(())
}
