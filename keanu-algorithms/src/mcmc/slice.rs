//! Coordinate-wise slice sampling with Neal's stepping-out and shrinkage procedure.

use super::{SamplingAlgorithm, SamplingOptions, continuous_target, generate, latent_values};
use crate::error::{AlgorithmError, Result};
use crate::samples::NetworkSamples;
use keanu_distributions::KeanuRng;
use keanu_graph::{Graph, Model, NodeId, Parameterisation, Space, State};
use keanu_tensor::DoubleTensor;

/// Gradient-free slice sampling. Each transition sweeps all continuous coordinates.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct Slice {
    /// Initial interval width in each sampling coordinate.
    pub width: f64,
    /// Maximum number of intervals in the stepping-out bracket, including the initial interval.
    pub max_intervals: usize,
    /// Coordinates in which to sample. Unconstrained by default.
    pub space: Space,
}

impl Default for Slice {
    fn default() -> Self {
        Self {
            width: 1.0,
            max_intervals: 100,
            space: Space::Unconstrained,
        }
    }
}

impl Slice {
    /// Sets [`width`](Self::width).
    #[must_use]
    pub fn with_width(mut self, width: f64) -> Self {
        self.width = width;
        self
    }

    /// Sets [`max_intervals`](Self::max_intervals).
    #[must_use]
    pub fn with_max_intervals(mut self, max_intervals: usize) -> Self {
        self.max_intervals = max_intervals;
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
    ) -> Result<SliceSampler<'m>> {
        if !self.width.is_finite() || self.width <= 0.0 || self.max_intervals == 0 {
            return Err(AlgorithmError::InvalidSetting(
                "slice sampling needs a finite positive width and at least one interval".into(),
            ));
        }
        let (param, position) = continuous_target(model, state, nodes, self.space, rng)?;
        let log_prob = param.log_prob(&position, true, state)?;
        if !log_prob.is_finite() {
            return Err(AlgorithmError::ImpossibleStart);
        }
        let sample_log_prob = log_prob - param.log_det_jacobian(&position)?;
        let left = latent_values(&param, state);
        Ok(SliceSampler {
            config: *self,
            left,
            param,
            position,
            log_prob,
            sample_log_prob,
            nodes: nodes.to_vec(),
            evaluations: 1,
        })
    }

    /// Records each sweep.
    pub fn sample(
        &self,
        model: &Model,
        state: &mut State,
        nodes: &[NodeId],
        count: usize,
        rng: &mut KeanuRng,
    ) -> Result<NetworkSamples> {
        generate(
            &mut self.sampler(model, state, nodes, rng)?,
            state,
            count,
            SamplingOptions::default(),
            rng,
        )
    }
}

/// A slice sampling chain. Keep the state at the position left by the previous sweep.
#[derive(Debug)]
pub struct SliceSampler<'m> {
    config: Slice,
    /// The latent values this chain left in the state after its last sweep.
    left: Vec<Option<DoubleTensor>>,
    param: Parameterisation<'m>,
    position: Vec<f64>,
    log_prob: f64,
    sample_log_prob: f64,
    nodes: Vec<NodeId>,
    evaluations: usize,
}

impl SliceSampler<'_> {
    /// The number of joint density evaluations, including the initial point.
    pub fn evaluations(&self) -> usize {
        self.evaluations
    }

    /// Adopts the state's position if something else moved the latents since the last sweep.
    fn resync(&mut self, state: &mut State) -> Result<()> {
        if latent_values(&self.param, state) == self.left {
            return Ok(());
        }
        let position = self.param.from_state(state)?;
        self.evaluations += 1;
        let log_prob = self.param.log_prob(&position, true, state)?;
        if !log_prob.is_finite() {
            return Err(AlgorithmError::ImpossibleStart);
        }
        self.sample_log_prob = log_prob - self.param.log_det_jacobian(&position)?;
        self.position = position;
        self.log_prob = log_prob;
        Ok(())
    }

    fn sweep(&mut self, state: &mut State, rng: &mut KeanuRng) -> Result<(Vec<f64>, f64, f64)> {
        let mut trial = self.position.clone();
        let mut log_prob = self.log_prob;
        for i in 0..trial.len() {
            let origin = trial[i];
            let height = rng.next_f64_non_zero().ln();
            let width = self.config.width;
            let mut left = origin - width * rng.next_f64();
            let mut right = left + width;
            // Randomize how the finite expansion budget is split to keep stepping out reversible.
            let left_budget = rng.next_below(self.config.max_intervals as u64) as usize;
            let right_budget = self.config.max_intervals - 1 - left_budget;
            let mut evaluate = |x| {
                trial[i] = x;
                self.evaluations += 1;
                self.param.log_prob(&trial, true, state)
            };
            let inside = |p: f64| p.is_finite() && p - log_prob >= height;
            for _ in 0..left_budget {
                if !left.is_finite() || !inside(evaluate(left)?) {
                    break;
                }
                left -= width;
            }
            for _ in 0..right_budget {
                if !right.is_finite() || !inside(evaluate(right)?) {
                    break;
                }
                right += width;
            }
            let mut accepted = None;
            for _ in 0..1000 {
                if !(right - left).is_finite() || left >= right || !(left..=right).contains(&origin)
                {
                    break;
                }
                let x = rng.next_f64_between(left, right);
                let p = evaluate(x)?;
                if inside(p) {
                    accepted = Some(p);
                    break;
                }
                if x < origin {
                    left = x;
                } else {
                    right = x;
                }
            }
            log_prob = accepted.ok_or_else(|| {
                AlgorithmError::SamplerStuck(format!(
                    "slice sampling could not shrink a finite bracket in coordinate {i}; check the interval width"
                ))
            })?;
        }
        let sample_log_prob = log_prob - self.param.log_det_jacobian(&trial)?;
        Ok((trial, log_prob, sample_log_prob))
    }
}

impl SamplingAlgorithm for SliceSampler<'_> {
    fn step(&mut self, state: &mut State, rng: &mut KeanuRng) -> Result<()> {
        self.resync(state)?;
        let result = match self.sweep(state, rng) {
            Ok((position, log_prob, sample_log_prob)) => {
                self.position = position;
                self.log_prob = log_prob;
                self.sample_log_prob = sample_log_prob;
                Ok(())
            }
            Err(error) => {
                self.param.to_state(&self.position, state)?;
                Err(error)
            }
        };
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
