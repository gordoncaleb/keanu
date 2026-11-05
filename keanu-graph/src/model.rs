//! Probabilistic models over a graph: the latent and observed nodes, their joint log probability and its gradient,
//! and the lambda sections that let samplers update the log probability incrementally.

use crate::error::{GraphError, Result};
use crate::graph::{Graph, NodeId};
use crate::logprob::{LogProbGradient, log_prob_gradient, node_log_prob};
use crate::op::eval;
use crate::state::State;
use crate::value::{DType, Value};
use std::collections::HashSet;
use std::sync::Arc;

/// A set of nodes reached from some starting nodes by walking towards children (downstream) or parents
/// (upstream), stopping at (but including) probabilistic and observed nodes. The starting nodes are included.
/// Factor nodes are included where the walk reaches them, but don't stop it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LambdaSection {
    nodes: Vec<NodeId>,
    latent_and_observed: Vec<NodeId>,
}

impl LambdaSection {
    /// Every node in the section, in topological order.
    pub fn nodes(&self) -> &[NodeId] {
        &self.nodes
    }

    /// The probabilistic, observed and factor nodes in the section, in topological order.
    pub fn latent_and_observed(&self) -> &[NodeId] {
        &self.latent_and_observed
    }
}

fn is_stop(graph: &Graph, state: &State, id: NodeId) -> bool {
    graph[id].is_probabilistic() || state.is_observed(id)
}

fn lambda_section(
    graph: &Graph,
    state: &State,
    from: &[NodeId],
    downstream: bool,
) -> LambdaSection {
    let mut seen: HashSet<NodeId> = from.iter().copied().collect();
    let mut stack: Vec<NodeId> = from.to_vec();
    while let Some(id) = stack.pop() {
        let next = if downstream {
            graph.children_of(id)
        } else {
            graph[id].parents()
        };
        for &n in next {
            if seen.insert(n) && !is_stop(graph, state, n) {
                stack.push(n);
            }
        }
    }
    let mut nodes: Vec<NodeId> = seen.into_iter().collect();
    nodes.sort();
    let latent_and_observed = nodes
        .iter()
        .copied()
        .filter(|&id| is_stop(graph, state, id) || graph[id].is_factor())
        .collect();
    LambdaSection {
        nodes,
        latent_and_observed,
    }
}

/// The nodes whose log probability can change when the values of `from` change: walking towards children and
/// stopping at probabilistic and observed nodes (the factors reached are included). Fails if an id is not a node of the graph.
pub fn downstream_lambda_section(
    graph: &Graph,
    state: &State,
    from: &[NodeId],
) -> Result<LambdaSection> {
    graph.check_ids(from)?;
    Ok(lambda_section(graph, state, from, true))
}

/// The nodes `from` depends on, walking towards parents and stopping at probabilistic and observed nodes. Fails if
/// an id is not a node of the graph.
pub fn upstream_lambda_section(
    graph: &Graph,
    state: &State,
    from: &[NodeId],
) -> Result<LambdaSection> {
    graph.check_ids(from)?;
    Ok(lambda_section(graph, state, from, false))
}

/// True if a deterministic observed node's value differs from what its parents compute (a violated constraint).
fn contradicts_observation(graph: &Graph, state: &State, id: NodeId) -> Result<bool> {
    let node = &graph[id];
    let Some(observed) = state.value(id) else {
        return Ok(false);
    };
    let parents: Vec<&Value> = node
        .parents()
        .iter()
        .map(|&p| state.value(p).ok_or(GraphError::MissingValue(p)))
        .collect::<Result<_>>()?;
    let computed = eval(node.op(), &parents)?;
    Ok(&computed != observed)
}

/// A probabilistic model: a shared graph with its latent nodes (unobserved probabilistic nodes) and observed
/// nodes, as they were in the state the model was created from, and its factor nodes.
///
/// The joint log probability is the sum of the latent and observed nodes' log probabilities and every factor's
/// term (see [`Op::Factor`](crate::op::Op::Factor)); the log likelihood includes the factors too.
///
/// The model doesn't own values; every method takes the [`State`] to work on, so one model can serve many
/// chains.
#[derive(Debug, Clone)]
pub struct Model {
    graph: Arc<Graph>,
    latents: Vec<NodeId>,
    observed: Vec<NodeId>,
    /// The factor nodes.
    factors: Vec<NodeId>,
    /// For each node, the probabilistic, observed and factor nodes whose log probability depends on its value.
    sections: Vec<Option<Vec<NodeId>>>,
    /// Cached per model: latents, then observed nodes, then factors.
    joint: Vec<NodeId>,
    /// Observed nodes then factors.
    likelihood: Vec<NodeId>,
    /// The probabilistic nodes and factors of `joint`, and of `likelihood`.
    probabilistic_joint: Vec<NodeId>,
    probabilistic_observed: Vec<NodeId>,
    /// The latents holding f64 values.
    continuous: Vec<NodeId>,
}

impl Model {
    /// A model of every latent, observed and factor node of the graph, in the given state. Fails if there are no
    /// latent or observed nodes.
    pub fn new(graph: impl Into<Arc<Graph>>, state: &State) -> Result<Self> {
        let graph = graph.into();
        let latents: Vec<NodeId> = graph
            .probabilistic()
            .filter(|&id| !state.is_observed(id))
            .collect();
        let observed: Vec<NodeId> = graph.ids().filter(|&id| state.is_observed(id)).collect();
        if latents.is_empty() && observed.is_empty() {
            return Err(GraphError::InvalidValue(
                "a model needs at least one latent or observed node".into(),
            ));
        }
        let sections = graph
            .ids()
            .map(|id| {
                graph[id]
                    .is_probabilistic()
                    .then(|| lambda_section(&graph, state, &[id], true).latent_and_observed)
            })
            .collect();
        let factors: Vec<NodeId> = graph.factors().collect();
        let joint: Vec<NodeId> = latents
            .iter()
            .chain(&observed)
            .chain(&factors)
            .copied()
            .collect();
        let likelihood: Vec<NodeId> = observed.iter().chain(&factors).copied().collect();
        let contributes = |id: &NodeId| graph[*id].is_probabilistic() || graph[*id].is_factor();
        let probabilistic_joint = joint.iter().copied().filter(contributes).collect();
        let probabilistic_observed = likelihood.iter().copied().filter(contributes).collect();
        let continuous = latents
            .iter()
            .copied()
            .filter(|&id| graph[id].dtype() == DType::F64)
            .collect();
        Ok(Self {
            graph,
            latents,
            observed,
            factors,
            sections,
            joint,
            likelihood,
            probabilistic_joint,
            probabilistic_observed,
            continuous,
        })
    }

    /// The graph.
    pub fn graph(&self) -> &Arc<Graph> {
        &self.graph
    }

    /// The unobserved probabilistic nodes, in topological order.
    pub fn latents(&self) -> &[NodeId] {
        &self.latents
    }

    /// The latent nodes holding f64 values (those gradients can be taken with respect to).
    pub fn continuous_latents(&self) -> Vec<NodeId> {
        self.continuous.clone()
    }

    /// The latent nodes holding integer or boolean values.
    pub fn discrete_latents(&self) -> Vec<NodeId> {
        self.latents
            .iter()
            .copied()
            .filter(|&id| self.graph[id].dtype() != DType::F64)
            .collect()
    }

    /// The observed nodes (probabilistic, and observed boolean constraints), in topological order.
    pub fn observed(&self) -> &[NodeId] {
        &self.observed
    }

    /// The factor nodes (see [`Op::Factor`](crate::op::Op::Factor)), in topological order.
    pub fn factors(&self) -> &[NodeId] {
        &self.factors
    }

    /// The node with this label.
    pub fn find(&self, label: &str) -> Option<NodeId> {
        self.graph.find(label)
    }

    /// The log probability of the given latent, observed and factor nodes only: the sum of the probabilistic nodes'
    /// log probabilities and the factors' terms, or -∞ if any is impossible (-∞ or NaN, or a factor that is not
    /// finite) or any observed deterministic node among them contradicts its observation. Other nodes are ignored.
    ///
    /// With the nodes [`affected_by`](Self::affected_by) some latents, this is the part of the joint log probability
    /// that changes with them (their Markov blanket's terms), which is all a sampler updating those latents alone
    /// needs to compare their values.
    pub fn local_log_prob(&self, state: &State, nodes: &[NodeId]) -> Result<f64> {
        self.log_prob_of(state, nodes)
    }

    fn log_prob_of(&self, state: &State, nodes: &[NodeId]) -> Result<f64> {
        let mut sum = 0.0;
        for &id in nodes {
            let node = &self.graph[id];
            if node.is_probabilistic() || node.is_factor() {
                let lp = node_log_prob(&self.graph, state, id)?;
                if lp == f64::NEG_INFINITY || lp.is_nan() {
                    return Ok(f64::NEG_INFINITY);
                }
                sum += lp;
            } else if contradicts_observation(&self.graph, state, id)? {
                return Ok(f64::NEG_INFINITY);
            }
        }
        Ok(sum)
    }

    /// The joint log probability of the latent and observed nodes, with the factors' terms.
    pub fn log_prob(&self, state: &State) -> Result<f64> {
        self.log_prob_of(state, &self.joint)
    }

    /// The log likelihood: the log probability of the observed nodes, with the factors' terms.
    pub fn log_likelihood(&self, state: &State) -> Result<f64> {
        self.log_prob_of(state, &self.likelihood)
    }

    /// The gradient of the joint log probability with respect to each continuous latent.
    pub fn log_prob_gradient(&self, state: &State) -> Result<LogProbGradient> {
        log_prob_gradient(
            &self.graph,
            state,
            &self.probabilistic_joint,
            &self.continuous,
        )
    }

    /// The gradient of the log likelihood (with the factors' terms) with respect to each continuous latent.
    pub fn log_likelihood_gradient(&self, state: &State) -> Result<LogProbGradient> {
        log_prob_gradient(
            &self.graph,
            state,
            &self.probabilistic_observed,
            &self.continuous,
        )
    }

    /// Sets the values of some nodes, taking ownership of them, and recomputes their deterministic descendants.
    /// Pass `values.iter().cloned()` to keep borrowed values.
    pub fn set_and_cascade(
        &self,
        state: &mut State,
        values: impl IntoIterator<Item = (NodeId, Value)>,
    ) -> Result<()> {
        let mut from: smallvec::SmallVec<[NodeId; 8]> = smallvec::SmallVec::new();
        for (id, value) in values {
            state.set(&self.graph, id, value)?;
            from.push(id);
        }
        state.cascade(&self.graph, &from)
    }

    /// The probabilistic, observed and factor nodes whose log probability depends on the values of `nodes`. Fails if an
    /// id is not a node of the model's graph.
    pub fn affected_by(&self, nodes: &[NodeId]) -> Result<Vec<NodeId>> {
        let mut affected: Vec<NodeId> = Vec::new();
        for &id in nodes {
            match self.sections.get(id.0).ok_or(GraphError::UnknownNode(id))? {
                Some(section) => affected.extend_from_slice(section),
                None => affected.push(id),
            }
        }
        affected.sort();
        affected.dedup();
        Ok(affected)
    }

    /// If the current state is impossible (log prob of -infinity or NaN),
    /// repeatedly samples every latent from its prior, in dependency order, cascading each value, until the state is
    /// possible. Gives up with an error after `attempts` tries. Values missing from the state (e.g. latents that
    /// were never set) are sampled first.
    pub fn probe_for_non_zero_probability(
        &self,
        state: &mut State,
        attempts: usize,
        rng: &mut keanu_distributions::KeanuRng,
    ) -> Result<()> {
        let impossible = |state: &State| -> Result<bool> {
            let log_prob = self.log_prob(state)?;
            Ok(log_prob == f64::NEG_INFINITY || log_prob.is_nan())
        };
        // sample any latent without a value from its prior, and compute the deterministic nodes
        let all: Vec<NodeId> = (0..self.graph.len()).map(NodeId::from_index).collect();
        state.lazy_eval(&self.graph, &all, rng)?;
        let mut attempt = 0;
        while impossible(state)? {
            if attempt == attempts {
                return Err(GraphError::InvalidValue(format!(
                    "failed to find a state with non-zero probability in {attempts} attempts"
                )));
            }
            for &latent in &self.latents {
                let value = state.sample_prior(&self.graph, latent, rng)?;
                state.set_and_cascade(&self.graph, latent, value)?;
            }
            attempt += 1;
        }
        Ok(())
    }

    /// Applies a proposal (new values for some latents, cascading to their deterministic descendants) and returns
    /// the new joint log probability, computed incrementally from `log_prob_before` (the log probability before
    /// the proposal) by recomputing only the affected nodes.
    pub fn log_prob_after(
        &self,
        state: &mut State,
        proposal: &[(NodeId, Value)],
        log_prob_before: f64,
    ) -> Result<f64> {
        let ids: Vec<NodeId> = proposal.iter().map(|(id, _)| *id).collect();
        let affected = self.affected_by(&ids)?;
        let before = self.log_prob_of(state, &affected)?;
        self.set_and_cascade(state, proposal.iter().cloned())?;
        let after = self.log_prob_of(state, &affected)?;
        if !(before.is_finite() && after.is_finite() && log_prob_before.is_finite()) {
            // an impossible state on either side can't be updated incrementally
            return self.log_prob(state);
        }
        Ok(log_prob_before + (after - before))
    }

    /// The joint log probability and, unless it is impossible (-∞), its gradient with respect to each continuous
    /// latent: exactly [`Self::log_prob`] and [`Self::log_prob_gradient`], in one pass that evaluates each node's
    /// distribution once (for gradient based samplers, which need both at every step).
    pub fn log_prob_and_gradient(&self, state: &State) -> Result<(f64, Option<LogProbGradient>)> {
        self.log_prob_and_gradient_of(state, &self.continuous)
    }

    /// The joint log probability and, unless it is impossible (-∞), its gradient with respect to the continuous
    /// latents `wrt` only (a subset of [`Self::continuous_latents`]), the other latents held at their values in
    /// `state`: for samplers that update some latents by gradients and others by other steps. The gradient's entries
    /// for the latents in `wrt` are exact; any other entry is incomplete and should be ignored. With every
    /// continuous latent it is exactly [`Self::log_prob_and_gradient`].
    ///
    /// # Errors
    ///
    /// [`GraphError::InvalidValue`] if an id in `wrt` is not a continuous latent of the model, and any error
    /// evaluating the model.
    pub fn log_prob_and_gradient_wrt(
        &self,
        state: &State,
        wrt: &[NodeId],
    ) -> Result<(f64, Option<LogProbGradient>)> {
        if let Some(&id) = wrt.iter().find(|id| !self.continuous.contains(id)) {
            return Err(GraphError::InvalidValue(format!(
                "{} is not a continuous latent of the model",
                self.graph.describe(id)
            )));
        }
        self.log_prob_and_gradient_of(state, wrt)
    }

    /// [`Self::log_prob_and_gradient_wrt`] without checking `wrt`.
    pub(crate) fn log_prob_and_gradient_of(
        &self,
        state: &State,
        wrt: &[NodeId],
    ) -> Result<(f64, Option<LogProbGradient>)> {
        for &id in &self.observed {
            if !self.graph[id].is_probabilistic()
                && contradicts_observation(&self.graph, state, id)?
            {
                return Ok((f64::NEG_INFINITY, None));
            }
        }
        let joint = crate::logprob::log_prob_and_gradient(
            &self.graph,
            state,
            &self.probabilistic_joint,
            wrt,
        )?;
        Ok(match joint {
            Some((log_prob, gradient)) => (log_prob, Some(gradient)),
            None => (f64::NEG_INFINITY, None),
        })
    }

    /// The gradient of the joint log probability with respect to each continuous latent and its derivatives along
    /// `b` directions, given as each latent's batched tangent `[b, latent shape...]` (see
    /// [`log_prob_hvp`](crate::logprob::log_prob_hvp)). Observed deterministic nodes are not checked.
    pub(crate) fn log_prob_gradient_and_hvp(
        &self,
        state: &State,
        tangents: impl IntoIterator<Item = (NodeId, keanu_tensor::DoubleTensor)>,
        b: usize,
    ) -> Result<(LogProbGradient, crate::logprob::LogProbHvp)> {
        crate::logprob::log_prob_hvp(
            &self.graph,
            state,
            &self.probabilistic_joint,
            &self.continuous,
            tangents,
            b,
        )
    }
}
