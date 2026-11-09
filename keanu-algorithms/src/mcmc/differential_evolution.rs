//! Population differential-evolution Metropolis sampling for continuous latents.

use super::{SamplingOptions, continuous_target};
use crate::error::{AlgorithmError, Result};
use crate::samples::NetworkSamples;
use crate::support::{evaluate_all, start_point};
use keanu_distributions::KeanuRng;
use keanu_graph::{Model, NodeId, Space, State};

/// Proposes a scaled difference of two other walkers, plus independent Gaussian noise.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct DifferentialEvolutionMetropolis {
    /// Proposal scale. Defaults to 2.38 / √(2 × dimension).
    pub scale: Option<f64>,
    /// Standard deviation of the proposal's additive noise in each coordinate.
    pub noise: f64,
    /// Coordinates in which to sample. Unconstrained by default.
    pub space: Space,
}

impl Default for DifferentialEvolutionMetropolis {
    fn default() -> Self {
        Self {
            scale: None,
            noise: 1e-3,
            space: Space::Unconstrained,
        }
    }
}

impl DifferentialEvolutionMetropolis {
    /// Sets [`scale`](Self::scale).
    #[must_use]
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = Some(scale);
        self
    }

    /// Sets [`noise`](Self::noise).
    #[must_use]
    pub fn with_noise(mut self, noise: f64) -> Self {
        self.noise = noise;
        self
    }

    /// Sets [`space`](Self::space).
    #[must_use]
    pub fn with_space(mut self, space: Space) -> Self {
        self.space = space;
        self
    }

    /// Runs `count` sweeps and records one draw per walker per sweep.
    ///
    /// Supply at least three states for the same model and observations, with dispersed starting positions.
    /// Walkers form a single population, so use independently seeded populations for between-chain diagnostics.
    pub fn sample(
        &self,
        model: &Model,
        states: &mut [State],
        nodes: &[NodeId],
        count: usize,
        rng: &mut KeanuRng,
    ) -> Result<Vec<NetworkSamples>> {
        Ok(self
            .sample_with_statistics(model, states, nodes, count, SamplingOptions::default(), rng)?
            .0)
    }

    /// Runs the population with burn-in and thinning in units of sweeps. The statistics include all sweeps.
    pub fn sample_with_statistics(
        &self,
        model: &Model,
        states: &mut [State],
        nodes: &[NodeId],
        count: usize,
        options: SamplingOptions,
        rng: &mut KeanuRng,
    ) -> Result<(Vec<NetworkSamples>, DifferentialEvolutionStatistics)> {
        options.validate(count)?;
        let size = states.len();
        if size < 3 {
            return Err(AlgorithmError::InvalidArgument(
                "differential-evolution Metropolis needs at least three walkers".into(),
            ));
        }
        let positive = |x: f64| x.is_finite() && x > 0.0;
        if !positive(self.noise) || self.scale.is_some_and(|scale| !positive(scale)) {
            return Err(AlgorithmError::InvalidSetting(
                "differential-evolution Metropolis needs finite positive scale and noise".into(),
            ));
        }
        let graph = model.graph();
        for state in states.iter() {
            if graph.ids().any(|id| {
                state.is_observed(id) != states[0].is_observed(id)
                    || (state.is_observed(id) && state.value(id) != states[0].value(id))
            }) {
                return Err(AlgorithmError::InvalidArgument(
                    "differential-evolution walkers must have identical observations".into(),
                ));
            }
        }
        let (param, initial) = continuous_target(model, &mut states[0], nodes, self.space, rng)?;
        let scale = self
            .scale
            .unwrap_or_else(|| 2.38 / (2.0 * initial.len() as f64).sqrt());
        let mut positions = Vec::with_capacity(size);
        positions.push(initial);
        for state in &mut states[1..] {
            evaluate_all(graph, state, rng)?;
            let point = start_point(&param, state)?;
            if point.iter().any(|x| !x.is_finite()) {
                return Err(AlgorithmError::ImpossibleStart);
            }
            positions.push(point);
        }
        let mut log_probs = Vec::with_capacity(size);
        let mut sample_log_probs = Vec::with_capacity(size);
        for (position, state) in positions.iter().zip(states.iter_mut()) {
            let log_prob = param.log_prob(position, true, state)?;
            if !log_prob.is_finite() {
                return Err(AlgorithmError::ImpossibleStart);
            }
            log_probs.push(log_prob);
            sample_log_probs.push(log_prob - param.log_det_jacobian(position)?);
        }
        let mut samples = vec![NetworkSamples::new(graph, nodes)?; size];
        let mut statistics = DifferentialEvolutionStatistics {
            accepted: vec![0; size],
            evaluations: size,
            sweeps: count,
        };
        let mut trial = vec![0.0; param.dimension()];
        for sweep in 0..count {
            for (i, state) in states.iter_mut().enumerate() {
                // Select an ordered pair without replacement from the other walkers.
                let a = skip(rng.next_below((size - 1) as u64) as usize, i);
                let b = skip(rng.next_below((size - 2) as u64) as usize, i.min(a));
                let b = skip(b, i.max(a));
                for (j, x) in trial.iter_mut().enumerate() {
                    *x = positions[i][j]
                        + scale * (positions[a][j] - positions[b][j])
                        + self.noise * rng.next_gaussian();
                }
                if trial.iter().any(|x| !x.is_finite()) {
                    continue;
                }
                statistics.evaluations += 1;
                let log_prob = match param.log_prob(&trial, true, state) {
                    Ok(log_prob) => log_prob,
                    Err(error) => {
                        param.to_state(&positions[i], state)?;
                        return Err(error.into());
                    }
                };
                if log_prob.is_finite() && rng.next_f64_non_zero().ln() < log_prob - log_probs[i] {
                    sample_log_probs[i] = log_prob - param.log_det_jacobian(&trial)?;
                    std::mem::swap(&mut positions[i], &mut trial);
                    log_probs[i] = log_prob;
                    statistics.accepted[i] += 1;
                } else {
                    param.to_state(&positions[i], state)?;
                }
            }
            if sweep >= options.drop && (sweep - options.drop) % options.down_sample_interval == 0 {
                for (i, chain) in samples.iter_mut().enumerate() {
                    chain.record(&states[i], sample_log_probs[i])?;
                }
            }
        }
        Ok((samples, statistics))
    }
}

fn skip(index: usize, excluded: usize) -> usize {
    index + usize::from(index >= excluded)
}

/// Counts from a population run.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct DifferentialEvolutionStatistics {
    /// Number of accepted proposals per walker.
    pub accepted: Vec<usize>,
    /// Total joint density evaluations, including initial points.
    pub evaluations: usize,
    /// Number of attempted sweeps, with one proposal per walker in each sweep.
    pub sweeps: usize,
}
