//! Metropolis–Hastings sampling.

use crate::error::{AlgorithmError, Result};
use crate::mcmc::proposal::{PriorProposal, Proposal, ProposalDistribution, ProposalListener};
use crate::mcmc::{SamplingAlgorithm, SamplingOptions, generate, is_impossible};
use crate::samples::NetworkSamples;
use keanu_distributions::KeanuRng;
use keanu_graph::{Graph, Model, NodeId, State, Value, downstream_lambda_section};
use std::sync::Arc;

/// Which latent nodes each Metropolis–Hastings step proposes new values for (Java's `MHStepVariableSelector`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum VariableSelector {
    /// One latent per step, cycling through them in order.
    #[default]
    Single,
    /// Every latent at once.
    Full,
}

impl VariableSelector {
    /// The latents to propose at step `step`.
    pub fn select(self, latents: &[NodeId], step: usize) -> Vec<NodeId> {
        match self {
            Self::Single => vec![latents[step % latents.len()]],
            Self::Full => latents.to_vec(),
        }
    }
}

/// How a rejected proposal is undone (Java's `ProposalRejectionStrategy`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RejectionStrategy {
    /// Snapshot every value the proposal can change and restore them (Java's `RollBackToCachedValuesOnRejection`).
    #[default]
    RollbackToCachedValues,
    /// Restore the proposed nodes' old values and recompute their deterministic descendants (Java's
    /// `RollbackAndCascadeOnRejection`).
    RollbackAndCascade,
}

/// The result of one Metropolis–Hastings step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepResult {
    /// Whether the proposal was accepted.
    pub accepted: bool,
    /// The joint log probability after the step (the old one if the proposal was rejected).
    pub log_prob: f64,
}

/// One Metropolis–Hastings step (Java's `MetropolisHastingsStep`).
#[derive(Debug, Clone)]
pub struct MetropolisHastingsStep<'m> {
    model: &'m Model,
    proposal: Arc<dyn ProposalDistribution>,
    rejection: RejectionStrategy,
}

impl<'m> MetropolisHastingsStep<'m> {
    /// A step over `model` with the given proposal distribution and rejection strategy.
    pub fn new(
        model: &'m Model,
        proposal: Arc<dyn ProposalDistribution>,
        rejection: RejectionStrategy,
    ) -> Self {
        Self {
            model,
            proposal,
            rejection,
        }
    }

    /// Proposes new values for `nodes` and accepts them with the Metropolis–Hastings probability, with the change
    /// in log probability scaled by 1 / `temperature` (1 for ordinary sampling, as in simulated annealing).
    pub fn step(
        &self,
        state: &mut State,
        nodes: &[NodeId],
        log_prob_before: f64,
        temperature: f64,
        rng: &mut KeanuRng,
        mut listener: Option<&mut (dyn ProposalListener + '_)>,
    ) -> Result<StepResult> {
        let graph = self.model.graph();
        let proposal = self.proposal.propose(self.model, state, nodes, rng)?;
        if let Some(l) = listener.as_deref_mut() {
            l.on_proposal_created(&proposal);
        }
        let snapshot = match self.rejection {
            RejectionStrategy::RollbackToCachedValues => Some(snapshot(graph, state, &proposal)),
            RejectionStrategy::RollbackAndCascade => None,
        };

        let log_prob_after = self
            .model
            .log_prob_after(state, &proposal.to, log_prob_before)?;

        if !is_impossible(log_prob_after) {
            let delta = log_prob_after - log_prob_before;
            let q_old = self
                .proposal
                .log_prob_from_given_to(self.model, state, &proposal)?;
            let q_new = self
                .proposal
                .log_prob_to_given_from(self.model, state, &proposal)?;
            let log_r = delta / temperature + q_old - q_new;
            if log_r.exp() >= rng.next_f64() {
                return Ok(StepResult {
                    accepted: true,
                    log_prob: log_prob_after,
                });
            }
        }

        if let Some(l) = listener {
            l.on_proposal_rejected(&proposal);
        }
        match snapshot {
            Some(values) => {
                for (id, value) in values {
                    match value {
                        Some(v) => state.set(graph, id, v)?,
                        None => state.clear(id),
                    }
                }
            }
            None => self.model.set_and_cascade(state, &proposal.from)?,
        }
        Ok(StepResult {
            accepted: false,
            log_prob: log_prob_before,
        })
    }
}

/// The current values of every node a proposal can change: the proposed nodes and the deterministic nodes of
/// their downstream lambda sections.
fn snapshot(graph: &Graph, state: &State, proposal: &Proposal) -> Vec<(NodeId, Option<Value>)> {
    let nodes: Vec<NodeId> = proposal.nodes().collect();
    downstream_lambda_section(graph, state, &nodes)
        .nodes()
        .iter()
        .filter(|&&id| !state.is_observed(id))
        .map(|&id| (id, state.value(id).cloned()))
        .collect()
}

/// Metropolis–Hastings configuration (Java's `MetropolisHastings`). Defaults: prior proposals, one latent per
/// step and rollback to cached values. A configuration can be shared by many chains.
#[derive(Debug, Clone)]
pub struct MetropolisHastings {
    /// How new values are proposed.
    pub proposal: Arc<dyn ProposalDistribution>,
    /// Which latents each step proposes for.
    pub selector: VariableSelector,
    /// What happens when a proposal is rejected.
    pub rejection: RejectionStrategy,
}

impl Default for MetropolisHastings {
    fn default() -> Self {
        Self {
            proposal: Arc::new(PriorProposal),
            selector: VariableSelector::Single,
            rejection: RejectionStrategy::RollbackToCachedValues,
        }
    }
}

impl MetropolisHastings {
    /// The default settings: prior proposals, one latent per step, rollback to cached values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the proposal distribution.
    pub fn proposal(mut self, proposal: impl ProposalDistribution + 'static) -> Self {
        self.proposal = Arc::new(proposal);
        self
    }

    /// Sets which latents each step proposes for.
    pub fn selector(mut self, selector: VariableSelector) -> Self {
        self.selector = selector;
        self
    }

    /// Sets what happens when a proposal is rejected.
    pub fn rejection(mut self, rejection: RejectionStrategy) -> Self {
        self.rejection = rejection;
        self
    }

    /// A chain starting from the current state, recording the values of `nodes`.
    pub fn sampler<'m>(
        &self,
        model: &'m Model,
        state: &mut State,
        nodes: &[NodeId],
        rng: &mut KeanuRng,
    ) -> Result<MetropolisHastingsSampler<'m>> {
        if model.latents().is_empty() {
            return Err(AlgorithmError::NoLatents);
        }
        // every node needs a value: a latent with a value can still have deterministic parents without one
        let all: Vec<NodeId> = model.graph().ids().collect();
        state.lazy_eval(model.graph(), &all, rng)?;
        let log_prob = model.log_prob(state)?;
        Ok(MetropolisHastingsSampler {
            step: MetropolisHastingsStep::new(model, self.proposal.clone(), self.rejection),
            selector: self.selector,
            nodes: nodes.to_vec(),
            log_prob,
            step_count: 0,
            listener: None,
        })
    }

    /// Runs a chain of `count` steps from the current state, recording every step's values of `nodes`.
    pub fn sample(
        &self,
        model: &Model,
        state: &mut State,
        nodes: &[NodeId],
        count: usize,
        rng: &mut KeanuRng,
    ) -> Result<NetworkSamples> {
        self.sample_with(model, state, nodes, count, SamplingOptions::default(), rng)
    }

    /// Like [`sample`](Self::sample), dropping and down-sampling as `options` says.
    pub fn sample_with(
        &self,
        model: &Model,
        state: &mut State,
        nodes: &[NodeId],
        count: usize,
        options: SamplingOptions,
        rng: &mut KeanuRng,
    ) -> Result<NetworkSamples> {
        let mut sampler = self.sampler(model, state, nodes, rng)?;
        generate(&mut sampler, state, count, options, rng)
    }
}

/// A running Metropolis–Hastings chain (Java's `MetropolisHastingsSampler`).
pub struct MetropolisHastingsSampler<'m> {
    step: MetropolisHastingsStep<'m>,
    selector: VariableSelector,
    nodes: Vec<NodeId>,
    log_prob: f64,
    step_count: usize,
    listener: Option<&'m mut dyn ProposalListener>,
}

impl std::fmt::Debug for MetropolisHastingsSampler<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetropolisHastingsSampler")
            .field("selector", &self.selector)
            .field("nodes", &self.nodes)
            .field("log_prob", &self.log_prob)
            .field("step_count", &self.step_count)
            .finish_non_exhaustive()
    }
}

impl<'m> MetropolisHastingsSampler<'m> {
    /// Reports every proposal to `listener` (e.g. an [`AcceptanceRateTracker`](super::proposal::AcceptanceRateTracker)).
    pub fn with_listener(mut self, listener: &'m mut dyn ProposalListener) -> Self {
        self.listener = Some(listener);
        self
    }

    /// The joint log probability of the chain's current position.
    pub fn log_prob(&self) -> f64 {
        self.log_prob
    }
}

impl SamplingAlgorithm for MetropolisHastingsSampler<'_> {
    fn step(&mut self, state: &mut State, rng: &mut KeanuRng) -> Result<()> {
        let latents = self.step.model.latents();
        let chosen = self.selector.select(latents, self.step_count);
        let listener = self.listener.as_deref_mut();
        self.log_prob = self
            .step
            .step(state, &chosen, self.log_prob, 1.0, rng, listener)?
            .log_prob;
        self.step_count += 1;
        Ok(())
    }

    fn record(&self, state: &State, samples: &mut NetworkSamples) -> Result<()> {
        samples.record(state, self.log_prob)
    }

    fn graph(&self) -> &Graph {
        self.step.model.graph()
    }

    fn sampled_nodes(&self) -> &[NodeId] {
        &self.nodes
    }
}
