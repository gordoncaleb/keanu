//! Static Hamiltonian Monte Carlo with a fixed leapfrog count and optional step-size warmup.

use super::{
    AdaptiveStepSize, Integrator, LeapfrogState, PotentialState, QuadraticPotential,
    accept_with_log_probability, log_prob_and_gradient_at,
};
use crate::error::{AlgorithmError, Result};
use crate::mcmc::{SamplingAlgorithm, SamplingOptions, continuous_target, generate, latent_values};
use crate::samples::NetworkSamples;
use crate::support::block_shapes;
use keanu_distributions::KeanuRng;
use keanu_graph::{Graph, Model, NodeId, Parameterisation, Space, State};
use std::cell::RefCell;
use std::rc::Rc;

/// Static Hamiltonian Monte Carlo with an identity mass matrix. Every transition uses the same
/// number of leapfrog steps, with 10% step-size jitter to avoid repeated trajectory lengths.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct Hmc {
    /// Number of leapfrog steps per proposal.
    pub leapfrog_steps: usize,
    /// Nominal step size before jitter and warmup.
    pub step_size: f64,
    /// Initial transitions during which dual averaging targets 0.8 acceptance.
    pub adapt_count: usize,
    /// Coordinates in which to sample. Unconstrained by default.
    pub space: Space,
}

impl Default for Hmc {
    fn default() -> Self {
        Self {
            leapfrog_steps: 10,
            step_size: 0.25,
            adapt_count: 0,
            space: Space::Unconstrained,
        }
    }
}

impl Hmc {
    /// Sets [`leapfrog_steps`](Self::leapfrog_steps).
    #[must_use]
    pub fn with_leapfrog_steps(mut self, steps: usize) -> Self {
        self.leapfrog_steps = steps;
        self
    }

    /// Sets [`step_size`](Self::step_size).
    #[must_use]
    pub fn with_step_size(mut self, step_size: f64) -> Self {
        self.step_size = step_size;
        self
    }

    /// Sets [`adapt_count`](Self::adapt_count).
    #[must_use]
    pub fn with_adapt_count(mut self, adapt_count: usize) -> Self {
        self.adapt_count = adapt_count;
        self
    }

    /// Sets [`space`](Self::space).
    #[must_use]
    pub fn with_space(mut self, space: Space) -> Self {
        self.space = space;
        self
    }

    /// Starts a chain over all continuous latents, recording the given nodes.
    pub fn sampler<'m>(
        &self,
        model: &'m Model,
        state: &mut State,
        nodes: &[NodeId],
        rng: &mut KeanuRng,
    ) -> Result<HmcSampler<'m>> {
        if self.leapfrog_steps == 0 || !self.step_size.is_finite() || self.step_size <= 0.0 {
            return Err(AlgorithmError::InvalidSetting(
                "HMC needs a positive leapfrog count and a finite positive step size".into(),
            ));
        }
        let (param, z) = continuous_target(model, state, nodes, self.space, rng)?;
        let blocks: Vec<usize> = block_shapes(&param)
            .iter()
            .map(|shape| shape.iter().product())
            .collect();
        let (log_prob, gradient) = log_prob_and_gradient_at(&param, state, &z)?;
        if !log_prob.is_finite() {
            return Err(AlgorithmError::ImpossibleStart);
        }
        if gradient.iter().any(|x| !x.is_finite()) {
            return Err(AlgorithmError::NonFiniteGradient);
        }
        let potential = PotentialState::new(&QuadraticPotential::default(), 0, &blocks);
        let current = LeapfrogState::new(
            Rc::new(z.clone()),
            vec![0.0; z.len()],
            Rc::new(gradient),
            log_prob,
            &potential,
        )?;
        let sample_log_prob = log_prob - param.log_det_jacobian(&z)?;
        let left = latent_values(&param, state);
        Ok(HmcSampler {
            config: *self,
            left,
            param,
            potential,
            current,
            sample_log_prob,
            nodes: nodes.to_vec(),
            step_size: AdaptiveStepSize::new(self.step_size, 0.8),
            statistics: HmcStatistics {
                step_size: self.step_size,
                gradient_evaluations: 1,
                ..HmcStatistics::default()
            },
        })
    }

    /// Runs `count` transitions, discarding the first [`adapt_count`](Self::adapt_count) (warmup, while the step size
    /// still changes) and recording the rest.
    pub fn sample(
        &self,
        model: &Model,
        state: &mut State,
        nodes: &[NodeId],
        count: usize,
        rng: &mut KeanuRng,
    ) -> Result<NetworkSamples> {
        Ok(self
            .sample_with_statistics(
                model,
                state,
                nodes,
                count,
                SamplingOptions::default().with_drop(self.adapt_count),
                rng,
            )?
            .0)
    }

    /// Runs a chain and returns statistics for all transitions, including warmup. `options.drop` must cover the
    /// [`adapt_count`](Self::adapt_count) warmup transitions: draws made while the step size still changes do not come
    /// from a fixed transition kernel.
    pub fn sample_with_statistics(
        &self,
        model: &Model,
        state: &mut State,
        nodes: &[NodeId],
        count: usize,
        options: SamplingOptions,
        rng: &mut KeanuRng,
    ) -> Result<(NetworkSamples, HmcStatistics)> {
        if options.drop < self.adapt_count {
            return Err(AlgorithmError::InvalidSetting(format!(
                "HMC adapts its step size for {} transitions, so drop at least that many (drop is {})",
                self.adapt_count, options.drop
            )));
        }
        let mut sampler = self.sampler(model, state, nodes, rng)?;
        let samples = generate(&mut sampler, state, count, options, rng)?;
        Ok((samples, sampler.statistics))
    }
}

/// Summary of static HMC transitions, including warmup.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[non_exhaustive]
pub struct HmcStatistics {
    /// Number of attempted trajectories.
    pub transitions: usize,
    /// Number of accepted proposals.
    pub accepted: usize,
    /// Trajectories with an invalid point or an energy increase exceeding 1000.
    pub divergences: usize,
    /// Initial gradient plus one per attempted leapfrog step.
    pub gradient_evaluations: usize,
    /// The current nominal step size before jitter.
    pub step_size: f64,
}

/// An HMC chain. Keep the state at the position left by the previous step.
#[derive(Debug)]
pub struct HmcSampler<'m> {
    config: Hmc,
    /// The latent values this chain left in the state after its last transition.
    left: Vec<Option<keanu_tensor::DoubleTensor>>,
    param: Parameterisation<'m>,
    potential: PotentialState,
    current: LeapfrogState,
    sample_log_prob: f64,
    nodes: Vec<NodeId>,
    step_size: AdaptiveStepSize,
    statistics: HmcStatistics,
}

impl HmcSampler<'_> {
    /// Returns the statistics accumulated so far.
    pub fn statistics(&self) -> &HmcStatistics {
        &self.statistics
    }

    /// Adopts the state's position if something else moved the latents since the last transition.
    fn resync(&mut self, state: &mut State) -> Result<()> {
        if latent_values(&self.param, state) == self.left {
            return Ok(());
        }
        let z = self.param.from_state(state)?;
        self.statistics.gradient_evaluations += 1;
        let (log_prob, gradient) = log_prob_and_gradient_at(&self.param, state, &z)?;
        if !log_prob.is_finite() {
            return Err(AlgorithmError::ImpossibleStart);
        }
        let sample_log_prob = log_prob - self.param.log_det_jacobian(&z)?;
        self.current = LeapfrogState::new(
            Rc::new(z),
            self.current.momentum.clone(),
            Rc::new(gradient),
            log_prob,
            &self.potential,
        )?;
        self.sample_log_prob = sample_log_prob;
        Ok(())
    }
}

impl SamplingAlgorithm for HmcSampler<'_> {
    fn step(&mut self, state: &mut State, rng: &mut KeanuRng) -> Result<()> {
        self.resync(state)?;
        let result = self.transition(state, rng);
        self.left = latent_values(&self.param, state);
        result
    }

    fn record(&self, state: &State, samples: &mut NetworkSamples) -> Result<()> {
        samples.record(state, self.sample_log_prob)
    }

    fn graph(&self) -> &Graph {
        self.param.model().graph()
    }

    fn sampled_nodes(&self) -> &[NodeId] {
        &self.nodes
    }
}

impl HmcSampler<'_> {
    fn transition(&mut self, state: &mut State, rng: &mut KeanuRng) -> Result<()> {
        let mut point = LeapfrogState::new(
            Rc::clone(&self.current.position),
            self.potential.random_momentum(rng),
            Rc::clone(&self.current.gradient),
            self.current.log_prob,
            &self.potential,
        )?;
        let start_energy = point.energy;
        let integrator = Integrator {
            param: &self.param,
            potential: &self.potential,
            start_energy,
            max_energy_change: 1000.0,
            in_state: RefCell::new(None),
        };
        let dt = self.step_size.step_size * rng.next_f64_between(0.9, 1.1);
        let mut invalid = !start_energy.is_finite();
        for _ in 0..self.config.leapfrog_steps {
            if invalid {
                break;
            }
            self.statistics.gradient_evaluations += 1;
            point = match integrator.leapfrog(state, &point, dt) {
                Ok(point) => point,
                Err(error) => {
                    self.param.to_state(&self.current.position, state)?;
                    return Err(error);
                }
            };
            invalid = !point.energy.is_finite() || point.gradient.iter().any(|x| !x.is_finite());
        }
        let log_accept = if invalid {
            f64::NEG_INFINITY
        } else {
            (start_energy - point.energy).min(0.0)
        };
        if invalid || start_energy - point.energy < -integrator.max_energy_change {
            self.statistics.divergences += 1;
        }
        if accept_with_log_probability(log_accept, rng) {
            self.sample_log_prob = point.log_prob - self.param.log_det_jacobian(&point.position)?;
            self.current = point;
            self.statistics.accepted += 1;
        } else {
            self.param.to_state(&self.current.position, state)?;
        }
        self.statistics.transitions += 1;
        if self.statistics.transitions <= self.config.adapt_count {
            self.step_size.learn(log_accept.exp());
            if self.statistics.transitions == self.config.adapt_count {
                self.step_size.complete();
            }
        }
        self.statistics.step_size = self.step_size.step_size;
        Ok(())
    }
}
