//! Markov chain Monte Carlo: Metropolis–Hastings, differential evolution, slice sampling, static HMC, NUTS,
//! simulated annealing, and Gibbs and random-walk steps for discrete latents. Includes compound samplers and
//! shared machinery for turning a chain into [`NetworkSamples`].

pub mod annealing;
pub mod chains;
pub mod compound;
pub mod differential_evolution;
pub mod gibbs;
pub mod metropolis;
pub mod nuts;
pub mod proposal;
pub mod select;
pub mod slice;

pub use nuts::hmc;

use crate::error::{AlgorithmError, Result};
use crate::samples::NetworkSamples;
use keanu_distributions::KeanuRng;
use keanu_graph::{Graph, Model, NodeId, Parameterisation, Space, State};

/// A chain that can be advanced one step at a time.
pub trait SamplingAlgorithm {
    /// Advances the chain by one step, leaving `state` at the chain's current position.
    fn step(&mut self, state: &mut State, rng: &mut KeanuRng) -> Result<()>;

    /// Records the chain's current sample of the sampled nodes.
    fn record(&self, state: &State, samples: &mut NetworkSamples) -> Result<()>;

    /// The graph the chain samples.
    fn graph(&self) -> &Graph;

    /// The nodes whose values are recorded.
    fn sampled_nodes(&self) -> &[NodeId];
}

/// How many steps to throw away and how often to record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct SamplingOptions {
    /// Steps taken (and discarded) before the first recorded sample, e.g. for burn-in.
    pub drop: usize,
    /// Record every n-th step after the drop (1 records every step).
    pub down_sample_interval: usize,
}

/// Setters for each setting, for use as `SamplingOptions::default().with_...(...)`.
impl SamplingOptions {
    /// Sets [`drop`](Self::drop).
    #[must_use]
    pub fn with_drop(mut self, drop: usize) -> Self {
        self.drop = drop;
        self
    }

    /// Sets [`down_sample_interval`](Self::down_sample_interval).
    #[must_use]
    pub fn with_down_sample_interval(mut self, down_sample_interval: usize) -> Self {
        self.down_sample_interval = down_sample_interval;
        self
    }

    fn validate(self, total: usize) -> Result<()> {
        if self.down_sample_interval == 0 {
            return Err(AlgorithmError::InvalidSetting(
                "the down sample interval must be at least 1".into(),
            ));
        }
        if self.drop >= total {
            return Err(AlgorithmError::InvalidSetting(format!(
                "cannot drop {} of {total} samples",
                self.drop
            )));
        }
        Ok(())
    }
}

impl Default for SamplingOptions {
    fn default() -> Self {
        Self {
            drop: 0,
            down_sample_interval: 1,
        }
    }
}

/// Runs a chain for `total` steps: drops the first `options.drop`, then records every
/// `options.down_sample_interval`-th of the rest.
pub fn generate(
    algorithm: &mut impl SamplingAlgorithm,
    state: &mut State,
    total: usize,
    options: SamplingOptions,
    rng: &mut KeanuRng,
) -> Result<NetworkSamples> {
    options.validate(total)?;
    for _ in 0..options.drop {
        algorithm.step(state, rng)?;
    }
    let mut samples = NetworkSamples::new(algorithm.graph(), algorithm.sampled_nodes())?;
    for i in 0..total - options.drop {
        algorithm.step(state, rng)?;
        if i % options.down_sample_interval == 0 {
            algorithm.record(state, &mut samples)?;
        }
    }
    Ok(samples)
}

/// True for a log probability that means "impossible": not finite (-∞, NaN, or a density of +∞ at a boundary of
/// a support).
pub(crate) fn is_impossible(log_prob: f64) -> bool {
    !log_prob.is_finite()
}

/// Initialize a sampler that moves all continuous latents in the given coordinates.
fn continuous_target<'m>(
    model: &'m Model,
    state: &mut State,
    nodes: &[NodeId],
    space: Space,
    rng: &mut KeanuRng,
) -> Result<(Parameterisation<'m>, Vec<f64>)> {
    if nodes.is_empty() {
        return Err(AlgorithmError::InvalidArgument("no nodes to sample".into()));
    }
    crate::support::check_nodes(model.graph(), nodes)?;
    crate::support::check_latents(model)?;
    crate::support::evaluate_all(model.graph(), state, rng)?;
    let param = Parameterisation::with_space(model, state, space)?;
    let position = crate::support::start_point(&param, state)?;
    if position.is_empty() {
        return Err(AlgorithmError::NoLatents);
    }
    if position.iter().any(|x| !x.is_finite()) {
        return Err(AlgorithmError::ImpossibleStart);
    }
    Ok((param, position))
}
