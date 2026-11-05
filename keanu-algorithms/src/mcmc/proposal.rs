//! Proposal distributions for Metropolis–Hastings, and listeners that observe proposals.

use crate::error::{AlgorithmError, Result};
use keanu_distributions::KeanuRng;
use keanu_graph::{DType, Model, NodeId, State, Value, node_log_prob_at};
use keanu_tensor::DoubleTensor;
use std::collections::HashMap;
use std::fmt::Debug;

/// New values for some latent nodes, together with their values before the proposal.
#[derive(Debug, Clone, PartialEq)]
pub struct Proposal {
    /// The proposed values, in the order the nodes were chosen.
    pub to: Vec<(NodeId, Value)>,
    /// The values before the proposal.
    pub from: Vec<(NodeId, Value)>,
    /// log q(to | from), when the proposal distribution has to evaluate it while the proposal is created.
    pub(crate) log_prob_to_given_from: Option<f64>,
}

impl Proposal {
    /// The nodes with a proposed value.
    pub fn nodes(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.to.iter().map(|(id, _)| *id)
    }
}

/// A distribution q(x' | x) of proposed values for the chosen latent nodes.
///
/// The Hastings correction is log q(x | x') - log q(x' | x): [`log_prob_from_given_to`] is called with the state
/// holding the proposed values, and [`log_prob_to_given_from`] may use a value recorded when the proposal was created.
///
/// [`log_prob_from_given_to`]: ProposalDistribution::log_prob_from_given_to
/// [`log_prob_to_given_from`]: ProposalDistribution::log_prob_to_given_from
pub trait ProposalDistribution: Debug + Send + Sync {
    /// Proposes new values for `nodes`, given the current state (which is not changed).
    fn propose(
        &self,
        model: &Model,
        state: &State,
        nodes: &[NodeId],
        rng: &mut KeanuRng,
    ) -> Result<Proposal>;

    /// log q(x | x'): the probability of proposing the old values from the new ones. Called with `state` holding
    /// the proposed values (and their deterministic consequences).
    fn log_prob_from_given_to(
        &self,
        model: &Model,
        state: &State,
        proposal: &Proposal,
    ) -> Result<f64>;

    /// log q(x' | x): the probability of the proposal itself.
    fn log_prob_to_given_from(
        &self,
        model: &Model,
        state: &State,
        proposal: &Proposal,
    ) -> Result<f64>;
}

fn as_f64(value: &Value) -> Result<&DoubleTensor> {
    value
        .as_f64()
        .ok_or_else(|| AlgorithmError::InvalidArgument("expected an f64 value".into()))
}

fn current(state: &State, id: NodeId) -> Result<Value> {
    state
        .value(id)
        .cloned()
        .ok_or_else(|| keanu_graph::GraphError::MissingValue(id).into())
}

/// Proposes each chosen node's new value from its prior given its parents' current values.
///
/// When several dependent nodes are proposed at once, each is drawn given the parents' values from *before* the
/// proposal, so q(x' | x) is evaluated, and recorded, while the proposal is created. The reverse move would draw
/// the old values given the proposed parents, which is how q(x | x') is evaluated.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PriorProposal;

impl ProposalDistribution for PriorProposal {
    fn propose(
        &self,
        model: &Model,
        state: &State,
        nodes: &[NodeId],
        rng: &mut KeanuRng,
    ) -> Result<Proposal> {
        let graph = model.graph();
        let mut to = Vec::with_capacity(nodes.len());
        let mut from = Vec::with_capacity(nodes.len());
        let mut log_prob_to_given_from = 0.0;
        for &id in nodes {
            let value = state.sample_prior(graph, id, rng)?;
            log_prob_to_given_from += node_log_prob_at(graph, state, id, &value)?;
            from.push((id, current(state, id)?));
            to.push((id, value));
        }
        Ok(Proposal {
            to,
            from,
            log_prob_to_given_from: Some(log_prob_to_given_from),
        })
    }

    fn log_prob_from_given_to(
        &self,
        model: &Model,
        state: &State,
        proposal: &Proposal,
    ) -> Result<f64> {
        let mut sum = 0.0;
        for (id, old) in &proposal.from {
            sum += node_log_prob_at(model.graph(), state, *id, old)?;
        }
        Ok(sum)
    }

    fn log_prob_to_given_from(
        &self,
        model: &Model,
        state: &State,
        proposal: &Proposal,
    ) -> Result<f64> {
        match proposal.log_prob_to_given_from {
            Some(recorded) => Ok(recorded),
            None => {
                let mut sum = 0.0;
                for (id, new) in &proposal.to {
                    sum += node_log_prob_at(model.graph(), state, *id, new)?;
                }
                Ok(sum)
            }
        }
    }
}

/// Proposes a random walk step: each chosen (f64) node's value plus Gaussian noise. The proposal is symmetric, so it adds no Hastings correction.
#[derive(Debug, Clone, PartialEq)]
pub struct GaussianProposal {
    default_sigma: Option<DoubleTensor>,
    sigmas: HashMap<NodeId, DoubleTensor>,
}

impl GaussianProposal {
    /// The same step size for every node.
    pub fn new(sigma: f64) -> Self {
        Self {
            default_sigma: Some(DoubleTensor::scalar(sigma)),
            sigmas: HashMap::new(),
        }
    }

    /// A step size per node; proposing any other node is an error.
    pub fn with_sigmas(sigmas: impl IntoIterator<Item = (NodeId, DoubleTensor)>) -> Self {
        Self {
            default_sigma: None,
            sigmas: sigmas.into_iter().collect(),
        }
    }

    /// Sets the step size of one node.
    pub fn sigma(mut self, node: NodeId, sigma: DoubleTensor) -> Self {
        self.sigmas.insert(node, sigma);
        self
    }

    fn sigma_for(&self, id: NodeId) -> Result<&DoubleTensor> {
        self.sigmas
            .get(&id)
            .or(self.default_sigma.as_ref())
            .ok_or_else(|| {
                AlgorithmError::InvalidArgument(format!(
                    "no Gaussian proposal sigma for node {id:?}"
                ))
            })
    }
}

impl ProposalDistribution for GaussianProposal {
    fn propose(
        &self,
        model: &Model,
        state: &State,
        nodes: &[NodeId],
        rng: &mut KeanuRng,
    ) -> Result<Proposal> {
        let mut to = Vec::with_capacity(nodes.len());
        let mut from = Vec::with_capacity(nodes.len());
        for &id in nodes {
            let node = model
                .graph()
                .node(id)
                .ok_or(keanu_graph::GraphError::UnknownNode(id))?;
            if node.dtype() != DType::F64 {
                return Err(AlgorithmError::InvalidArgument(format!(
                    "a Gaussian proposal can't be used for the discrete node {id:?}"
                )));
            }
            let value = current(state, id)?;
            let x = as_f64(&value)?;
            let step = rng.gaussian(x.shape()).mul(self.sigma_for(id)?)?;
            to.push((id, Value::F64(x.add(&step)?)));
            from.push((id, value));
        }
        Ok(Proposal {
            to,
            from,
            log_prob_to_given_from: None,
        })
    }

    // The proposal is symmetric, q(x | x') = q(x' | x), so the Hastings correction log q(x | x') - log q(x' | x) is
    // 0: both are given as 0 rather than evaluating two equal Gaussian densities (as Java does) to cancel them.
    fn log_prob_from_given_to(&self, _: &Model, _: &State, _: &Proposal) -> Result<f64> {
        Ok(0.0)
    }

    fn log_prob_to_given_from(&self, _: &Model, _: &State, _: &Proposal) -> Result<f64> {
        Ok(0.0)
    }
}

/// Observes proposals as a chain runs.
pub trait ProposalListener {
    /// Called when a proposal is created.
    fn on_proposal_created(&mut self, proposal: &Proposal);
    /// Called when a proposal is rejected.
    fn on_proposal_rejected(&mut self, proposal: &Proposal);
}

/// Counts proposals and rejections per node.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AcceptanceRateTracker {
    proposed: HashMap<NodeId, usize>,
    rejected: HashMap<NodeId, usize>,
}

impl AcceptanceRateTracker {
    /// A tracker with no proposals recorded.
    pub fn new() -> Self {
        Self::default()
    }

    /// The fraction of proposals for a node that were accepted, or None if none were made.
    pub fn acceptance_rate(&self, node: NodeId) -> Option<f64> {
        let proposed = *self.proposed.get(&node)?;
        let rejected = self.rejected.get(&node).copied().unwrap_or(0);
        Some(1.0 - rejected as f64 / proposed as f64)
    }
}

impl ProposalListener for AcceptanceRateTracker {
    fn on_proposal_created(&mut self, proposal: &Proposal) {
        for id in proposal.nodes() {
            *self.proposed.entry(id).or_default() += 1;
        }
    }

    fn on_proposal_rejected(&mut self, proposal: &Proposal) {
        for id in proposal.nodes() {
            *self.rejected.entry(id).or_default() += 1;
        }
    }
}
