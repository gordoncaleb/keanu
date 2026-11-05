//! Markov chain Monte Carlo: Metropolis–Hastings, simulated annealing, the No-U-Turn sampler, Gibbs and random-walk
//! steps for discrete latents and compound samplers made of them, plus the machinery they share for turning a chain
//! into [`NetworkSamples`].

pub mod annealing;
pub mod chains;
pub mod compound;
pub mod gibbs;
pub mod metropolis;
pub mod nuts;
pub mod proposal;
pub mod select;

use crate::error::{AlgorithmError, Result};
use crate::samples::NetworkSamples;
use keanu_distributions::KeanuRng;
use keanu_graph::{Graph, NodeId, State};

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
    if options.down_sample_interval == 0 {
        return Err(AlgorithmError::InvalidSetting(
            "the down sample interval must be at least 1".into(),
        ));
    }
    if options.drop >= total {
        return Err(AlgorithmError::InvalidSetting(format!(
            "cannot drop {} of {total} samples",
            options.drop
        )));
    }
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

/// True for a log probability that means "impossible": -∞ or NaN.
pub(crate) fn is_impossible(log_prob: f64) -> bool {
    log_prob == f64::NEG_INFINITY || log_prob.is_nan()
}
