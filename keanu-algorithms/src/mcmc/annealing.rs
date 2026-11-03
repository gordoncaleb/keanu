//! Simulated annealing: finding the maximum a posteriori state with Metropolis–Hastings at a falling temperature.

use crate::error::{AlgorithmError, Result};
use crate::mcmc::is_impossible;
use crate::mcmc::metropolis::{MetropolisHastings, MetropolisHastingsStep};
use keanu_distributions::KeanuRng;
use keanu_graph::{Model, NodeId, State, Value};

/// The temperature at each iteration.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum AnnealingSchedule {
    /// T(n) = start · (end / start)^(n / iterations), falling from `start` to `end` (Java's `exponentialSchedule`).
    Exponential {
        /// The number of iterations over which the temperature decays.
        iterations: usize,
        /// The starting temperature.
        start: f64,
        /// The final temperature.
        end: f64,
    },
}

impl AnnealingSchedule {
    /// Exponential decay from `start` to `end` over `iterations`.
    pub fn exponential(iterations: usize, start: f64, end: f64) -> Self {
        Self::Exponential {
            iterations,
            start,
            end,
        }
    }

    /// The temperature at the given iteration.
    pub fn temperature(&self, iteration: usize) -> f64 {
        match *self {
            Self::Exponential {
                iterations,
                start,
                end,
            } => {
                let minus_k = (end / start).ln() / iterations as f64;
                start * (minus_k * iteration as f64).exp()
            }
        }
    }
}

/// The most probable state found.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct MapEstimate {
    /// The latent values at the most probable state found.
    pub values: Vec<(NodeId, Value)>,
    /// Its joint log probability.
    pub log_prob: f64,
}

/// Simulated annealing (Java's `SimulatedAnnealing`), using a Metropolis–Hastings configuration for its proposals,
/// variable selection and rejection.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct SimulatedAnnealing {
    /// The Metropolis–Hastings settings each step uses.
    pub metropolis_hastings: MetropolisHastings,
}

/// Setters for each setting, for use as `SimulatedAnnealing::default().with_...(...)`.
impl SimulatedAnnealing {
    /// Sets [`metropolis_hastings`](Self::metropolis_hastings).
    #[must_use]
    pub fn with_metropolis_hastings(mut self, metropolis_hastings: MetropolisHastings) -> Self {
        self.metropolis_hastings = metropolis_hastings;
        self
    }
}

impl SimulatedAnnealing {
    /// Simulated annealing with the given Metropolis–Hastings settings.
    pub fn new(metropolis_hastings: MetropolisHastings) -> Self {
        Self {
            metropolis_hastings,
        }
    }

    /// Anneals for `iterations` steps with the default schedule, from temperature 2 down to 0.01.
    pub fn max_a_posteriori(
        &self,
        model: &Model,
        state: &mut State,
        iterations: usize,
        rng: &mut KeanuRng,
    ) -> Result<MapEstimate> {
        let schedule = AnnealingSchedule::exponential(iterations, 2.0, 0.01);
        self.max_a_posteriori_with(model, state, iterations, schedule, rng)
    }

    /// Anneals for `iterations` steps with the given schedule. `state` is left at the last step; the best state
    /// found is returned.
    pub fn max_a_posteriori_with(
        &self,
        model: &Model,
        state: &mut State,
        iterations: usize,
        schedule: AnnealingSchedule,
        rng: &mut KeanuRng,
    ) -> Result<MapEstimate> {
        let latents = model.latents();
        if latents.is_empty() {
            return Err(AlgorithmError::NoLatents);
        }
        // every node needs a value: a latent with a value can still have deterministic parents without one
        let all: Vec<NodeId> = model.graph().ids().collect();
        state.lazy_eval(model.graph(), &all, rng)?;

        let mut log_prob = model.log_prob(state)?;
        if is_impossible(log_prob) {
            return Err(AlgorithmError::ImpossibleStart);
        }
        let take = |state: &State| -> Result<Vec<(NodeId, Value)>> {
            latents
                .iter()
                .map(|&id| {
                    state
                        .value(id)
                        .cloned()
                        .map(|v| (id, v))
                        .ok_or(AlgorithmError::Graph(
                            keanu_graph::GraphError::MissingValue(id),
                        ))
                })
                .collect()
        };
        let mut best = MapEstimate {
            values: take(state)?,
            log_prob,
        };

        let mh = &self.metropolis_hastings;
        let step = MetropolisHastingsStep::new(model, mh.proposal.clone(), mh.rejection);
        for iteration in 0..iterations {
            let chosen = mh.selector.select(latents, iteration);
            let temperature = schedule.temperature(iteration);
            log_prob = step
                .step(state, &chosen, log_prob, temperature, rng, None)?
                .log_prob;
            if log_prob > best.log_prob {
                best = MapEstimate {
                    values: take(state)?,
                    log_prob,
                };
            }
        }
        Ok(best)
    }
}
