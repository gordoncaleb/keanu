//! A particle filter for finding probable states of a model (Java's `ParticleFilter`).
//!
//! The model is considered one observation at a time, together with the latents that observation depends on
//! (ordered so that observations with the fewest new latents come first). For each increment every particle samples
//! the new latents from their prior, given the particle's existing latents, and is weighted by the log probability
//! of the part of the model considered so far. Low probability particles are culled and replaced by copies of
//! higher probability ones.

use crate::error::{AlgorithmError, Result};
use keanu_distributions::KeanuRng;
use keanu_graph::{Graph, Model, NodeId, State, Value, node_log_prob};
use std::collections::{BTreeMap, BTreeSet};

/// Particle filter settings. Defaults follow Java's `ParticleFilterBuilder`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ParticleFilter {
    /// The number of particles.
    pub particles: usize,
    /// Rounds of culling and resampling per increment.
    pub resampling_cycles: usize,
    /// The fraction of particles culled in each round.
    pub resampling_proportion: f64,
}

/// Setters for each setting, for use as `ParticleFilter::default().with_...(...)`.
impl ParticleFilter {
    /// Sets [`particles`](Self::particles).
    #[must_use]
    pub fn with_particles(mut self, particles: usize) -> Self {
        self.particles = particles;
        self
    }

    /// Sets [`resampling_cycles`](Self::resampling_cycles).
    #[must_use]
    pub fn with_resampling_cycles(mut self, resampling_cycles: usize) -> Self {
        self.resampling_cycles = resampling_cycles;
        self
    }

    /// Sets [`resampling_proportion`](Self::resampling_proportion).
    #[must_use]
    pub fn with_resampling_proportion(mut self, resampling_proportion: f64) -> Self {
        self.resampling_proportion = resampling_proportion;
        self
    }
}

impl Default for ParticleFilter {
    fn default() -> Self {
        Self {
            particles: 1000,
            resampling_cycles: 3,
            resampling_proportion: 0.5,
        }
    }
}

/// One candidate state: values for the latents considered so far, and its log probability.
#[derive(Debug, Clone, PartialEq)]
pub struct Particle {
    latents: BTreeMap<NodeId, Value>,
    observed: Vec<NodeId>,
    log_prob: f64,
}

impl Particle {
    fn empty() -> Self {
        Self {
            latents: BTreeMap::new(),
            observed: Vec::new(),
            log_prob: 0.0,
        }
    }

    /// The log probability of the latents and observations considered so far.
    pub fn log_prob(&self) -> f64 {
        self.log_prob
    }

    /// This particle's value of a latent.
    pub fn value(&self, id: NodeId) -> Option<&Value> {
        self.latents.get(&id)
    }

    /// The particle's latent values.
    pub fn latents(&self) -> impl Iterator<Item = (NodeId, &Value)> + '_ {
        self.latents.iter().map(|(&id, v)| (id, v))
    }

    /// Sets the particle's latent values in `state` and recomputes their deterministic descendants.
    pub fn apply(&self, model: &Model, state: &mut State) -> Result<()> {
        let values: Vec<(NodeId, Value)> = self
            .latents
            .iter()
            .map(|(&id, v)| (id, v.clone()))
            .collect();
        if !values.is_empty() {
            model.set_and_cascade(state, &values)?;
        }
        Ok(())
    }

    /// Applies the particle and recomputes its log probability.
    fn update_log_prob(&mut self, model: &Model, state: &mut State) -> Result<()> {
        self.apply(model, state)?;
        let graph = model.graph();
        let mut sum = 0.0;
        for &id in self.latents.keys().chain(&self.observed) {
            sum += node_log_prob(graph, state, id)?;
        }
        self.log_prob = sum;
        Ok(())
    }
}

/// For each observed node, the latents it depends on that no earlier observation depends on, in the order the
/// filter considers them: repeatedly the observation with the fewest remaining latent dependencies (ties broken
/// by node id). Java's `LatentIncrementSort`.
pub fn latent_increments(graph: &Graph, state: &State) -> Vec<(NodeId, Vec<NodeId>)> {
    // the latent ancestors of every node, in id (topological) order
    let mut latent_ancestors: Vec<BTreeSet<NodeId>> = vec![BTreeSet::new(); graph.len()];
    for id in graph.ids() {
        let mut ancestors = BTreeSet::new();
        for &parent in graph.parents(id) {
            ancestors.extend(latent_ancestors[parent.index()].iter().copied());
            if graph.node(parent).is_probabilistic() && !state.is_observed(parent) {
                ancestors.insert(parent);
            }
        }
        latent_ancestors[id.index()] = ancestors;
    }

    let mut remaining: BTreeMap<NodeId, BTreeSet<NodeId>> = graph
        .ids()
        .filter(|&id| state.is_observed(id) && graph.node(id).is_probabilistic())
        .map(|id| (id, latent_ancestors[id.index()].clone()))
        .collect();

    let mut order = Vec::new();
    while let Some((&next, _)) = remaining.iter().min_by_key(|(id, deps)| (deps.len(), **id)) {
        let deps = remaining.remove(&next).expect("present");
        for other in remaining.values_mut() {
            for d in &deps {
                other.remove(d);
            }
        }
        order.push((next, deps.into_iter().collect()));
    }
    order
}

impl ParticleFilter {
    fn validate(&self) -> Result<()> {
        if self.particles == 0 {
            return Err(AlgorithmError::InvalidSetting(
                "a particle filter needs at least one particle".into(),
            ));
        }
        if !(0.0..1.0).contains(&self.resampling_proportion) {
            return Err(AlgorithmError::InvalidSetting(
                "the resampling proportion must be in [0, 1)".into(),
            ));
        }
        Ok(())
    }

    /// Runs the filter and returns the particles, most probable first. `state` is used as scratch space; on return
    /// it holds the most probable particle's values.
    pub fn run(
        &self,
        model: &Model,
        state: &mut State,
        rng: &mut KeanuRng,
    ) -> Result<Vec<Particle>> {
        self.validate()?;
        let graph = model.graph();
        // every node needs a value so particles can sample given their parents
        let all: Vec<NodeId> = graph.ids().collect();
        state.lazy_eval(graph, &all, rng)?;

        let mut particles = vec![Particle::empty(); self.particles];
        for (observed, dependencies) in latent_increments(graph, state) {
            particles =
                self.update_particles(model, state, observed, &dependencies, &particles, rng)?;
        }
        particles.sort_by(|a, b| b.log_prob.total_cmp(&a.log_prob));
        if let Some(best) = particles.first() {
            best.apply(model, state)?;
        }
        Ok(particles)
    }

    fn update_particles(
        &self,
        model: &Model,
        state: &mut State,
        observed: NodeId,
        dependencies: &[NodeId],
        previous: &[Particle],
        rng: &mut KeanuRng,
    ) -> Result<Vec<Particle>> {
        let mut updated = sample_and_copy(previous, self.particles, rng);
        add_observation(model, state, &mut updated, observed, dependencies, rng)?;
        for _ in 0..self.resampling_cycles {
            updated.sort_by(|a, b| b.log_prob.total_cmp(&a.log_prob));
            let keep = (updated.len() as f64 * (1.0 - self.resampling_proportion)) as usize;
            updated.truncate(keep);
            let mut fresh = sample_and_copy(previous, self.particles - updated.len(), rng);
            add_observation(model, state, &mut fresh, observed, dependencies, rng)?;
            updated.extend(fresh);
        }
        Ok(updated)
    }
}

/// Extends each particle with an observation and samples the latents it newly depends on, parents first, given
/// the particle's own values for everything already sampled.
fn add_observation(
    model: &Model,
    state: &mut State,
    particles: &mut [Particle],
    observed: NodeId,
    dependencies: &[NodeId],
    rng: &mut KeanuRng,
) -> Result<()> {
    let mut ordered = dependencies.to_vec();
    ordered.sort();
    for particle in particles.iter_mut() {
        // the state holds whichever particle was evaluated last, so restore this one first
        particle.apply(model, state)?;
        particle.observed.push(observed);
        for &latent in &ordered {
            let value = state.sample_prior(model.graph(), latent, rng)?;
            model.set_and_cascade(state, &[(latent, value.clone())])?;
            particle.latents.insert(latent, value);
        }
        particle.update_log_prob(model, state)?;
    }
    Ok(())
}

/// exp(log prob - max log prob) for each particle, so that very negative log probabilities don't all underflow to
/// zero. If no particle is possible they are weighted equally.
pub fn normalised_weights(particles: &[Particle]) -> Vec<f64> {
    let max = particles
        .iter()
        .map(|p| p.log_prob)
        .fold(f64::NEG_INFINITY, f64::max);
    if max == f64::NEG_INFINITY || max.is_nan() {
        return vec![1.0; particles.len()];
    }
    particles.iter().map(|p| (p.log_prob - max).exp()).collect()
}

/// Draws `count` particles with replacement, in proportion to their weights.
fn sample_and_copy(particles: &[Particle], count: usize, rng: &mut KeanuRng) -> Vec<Particle> {
    let weights = normalised_weights(particles);
    let total: f64 = weights.iter().sum();
    (0..count)
        .map(|_| {
            let r = rng.next_f64() * total;
            let mut cumulative = 0.0;
            let mut chosen = particles.len() - 1;
            for (i, w) in weights.iter().enumerate() {
                cumulative += w;
                if cumulative > r {
                    chosen = i;
                    break;
                }
            }
            particles[chosen].clone()
        })
        .collect()
}
