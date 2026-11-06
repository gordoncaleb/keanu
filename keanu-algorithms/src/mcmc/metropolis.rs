//! Metropolis–Hastings sampling.

use crate::error::{AlgorithmError, Result};
use crate::mcmc::proposal::{PriorProposal, Proposal, ProposalDistribution, ProposalListener};
use crate::mcmc::{SamplingAlgorithm, SamplingOptions, generate, is_impossible};
use crate::samples::NetworkSamples;
use crate::support::{check_nodes, evaluate_all};
use keanu_distributions::KeanuRng;
use keanu_graph::{Graph, Model, NodeId, State, Value, downstream_lambda_section};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

/// Which latent nodes each Metropolis–Hastings step proposes new values for.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
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

/// How a rejected proposal is undone.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum RejectionStrategy {
    /// Snapshot every value the proposal can change and restore them.
    #[default]
    RollbackToCachedValues,
    /// Restore the proposed nodes' old values and recompute their deterministic descendants.
    RollbackAndCascade,
}

/// The result of one Metropolis–Hastings step.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct StepResult {
    /// Whether the proposal was accepted.
    pub accepted: bool,
    /// The joint log probability after the step (the old one if the proposal was rejected).
    pub log_prob: f64,
}

/// One Metropolis–Hastings step.
#[derive(Debug)]
pub struct MetropolisHastingsStep<'m> {
    model: &'m Model,
    proposal: Arc<dyn ProposalDistribution>,
    rejection: RejectionStrategy,
    /// For [`RejectionStrategy::RollbackToCachedValues`]: for each list of proposed nodes seen so far, the nodes a
    /// proposal for them can change. A chain proposes for the same few lists over and over (one per latent, or all
    /// of them), so each is worked out once.
    changeable: Mutex<HashMap<Vec<NodeId>, Arc<[NodeId]>>>,
}

impl Clone for MetropolisHastingsStep<'_> {
    fn clone(&self) -> Self {
        Self::new(self.model, Arc::clone(&self.proposal), self.rejection)
    }
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
            changeable: Mutex::new(HashMap::new()),
        }
    }

    /// Proposes new values for `nodes` and accepts them with the Metropolis–Hastings probability, with the change
    /// in log probability scaled by 1 / `temperature` (1 for ordinary sampling, as in simulated annealing).
    ///
    /// If evaluating the proposal fails, the state is rolled back to its values before the proposal and the error
    /// is returned.
    pub fn step(
        &self,
        state: &mut State,
        nodes: &[NodeId],
        log_prob_before: f64,
        temperature: f64,
        rng: &mut KeanuRng,
        mut listener: Option<&mut (dyn ProposalListener + '_)>,
    ) -> Result<StepResult> {
        let proposal = self.proposal.propose(self.model, state, nodes, rng)?;
        if let Some(l) = listener.as_deref_mut() {
            l.on_proposal_created(&proposal);
        }
        let snapshot = match self.rejection {
            RejectionStrategy::RollbackToCachedValues => Some(self.snapshot(state, &proposal)?),
            RejectionStrategy::RollbackAndCascade => None,
        };

        let accepted = match self.accept(state, &proposal, log_prob_before, temperature, rng) {
            Ok(accepted) => accepted,
            Err(e) => {
                // don't leave the state part way through the proposal
                self.roll_back(state, snapshot, proposal)?;
                return Err(e);
            }
        };
        if let Some(log_prob) = accepted {
            return Ok(StepResult {
                accepted: true,
                log_prob,
            });
        }

        if let Some(l) = listener {
            l.on_proposal_rejected(&proposal);
        }
        self.roll_back(state, snapshot, proposal)?;
        Ok(StepResult {
            accepted: false,
            log_prob: log_prob_before,
        })
    }

    /// Applies the proposal and decides whether to accept it: the new log probability if accepted.
    fn accept(
        &self,
        state: &mut State,
        proposal: &Proposal,
        log_prob_before: f64,
        temperature: f64,
        rng: &mut KeanuRng,
    ) -> Result<Option<f64>> {
        let log_prob_after = self
            .model
            .log_prob_after(state, &proposal.to, log_prob_before)?;
        if is_impossible(log_prob_after) {
            return Ok(None);
        }
        let delta = log_prob_after - log_prob_before;
        let q_old = self
            .proposal
            .log_prob_from_given_to(self.model, state, proposal)?;
        let q_new = self
            .proposal
            .log_prob_to_given_from(self.model, state, proposal)?;
        let log_r = delta / temperature + q_old - q_new;
        Ok((log_r.exp() >= rng.next_f64()).then_some(log_prob_after))
    }

    /// Restores the values from before the proposal.
    fn roll_back(
        &self,
        state: &mut State,
        snapshot: Option<Vec<(NodeId, Option<Value>)>>,
        proposal: Proposal,
    ) -> Result<()> {
        match snapshot {
            Some(values) => {
                let graph = self.model.graph();
                for (id, value) in values {
                    match value {
                        Some(v) => state.set(graph, id, v)?,
                        None => state.clear(id)?,
                    }
                }
            }
            None => self.model.set_and_cascade(state, proposal.from)?,
        }
        Ok(())
    }

    /// The current values of every node a proposal can change: the proposed nodes and the unobserved
    /// deterministic nodes of their downstream lambda sections.
    fn snapshot(&self, state: &State, proposal: &Proposal) -> Result<Vec<(NodeId, Option<Value>)>> {
        let nodes: Vec<NodeId> = proposal.nodes().collect();
        let changeable = {
            let mut cache = self
                .changeable
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            match cache.get(&nodes) {
                Some(changeable) => Arc::clone(changeable),
                None => {
                    let changeable: Arc<[NodeId]> =
                        downstream_lambda_section(self.model.graph(), state, &nodes)?
                            .nodes()
                            .iter()
                            .copied()
                            .filter(|&id| !state.is_observed(id))
                            .collect();
                    cache.insert(nodes, Arc::clone(&changeable));
                    changeable
                }
            }
        };
        Ok(changeable
            .iter()
            .map(|&id| (id, state.value(id).cloned()))
            .collect())
    }
}

/// Metropolis–Hastings configuration. Defaults: prior proposals, one latent per
/// step and rollback to cached values. A configuration can be shared by many chains.
///
/// Set it up with `MetropolisHastings::new().proposal(...).selector(...).rejection(...)`.
#[derive(Debug, Clone)]
#[non_exhaustive]
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
    #[must_use]
    pub fn proposal(mut self, proposal: impl ProposalDistribution + 'static) -> Self {
        self.proposal = Arc::new(proposal);
        self
    }

    /// Sets the proposal distribution to one shared with the caller.
    #[must_use]
    pub fn shared_proposal(mut self, proposal: Arc<dyn ProposalDistribution>) -> Self {
        self.proposal = proposal;
        self
    }

    /// Sets which latents each step proposes for.
    #[must_use]
    pub fn selector(mut self, selector: VariableSelector) -> Self {
        self.selector = selector;
        self
    }

    /// Sets what happens when a proposal is rejected.
    #[must_use]
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
        check_nodes(model.graph(), nodes)?;
        crate::support::check_likelihood(model)?;
        if model.latents().is_empty() {
            return Err(AlgorithmError::NoLatents);
        }
        evaluate_all(model.graph(), state, rng)?;
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

/// A running Metropolis–Hastings chain.
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
