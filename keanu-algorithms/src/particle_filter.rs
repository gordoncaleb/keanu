//! A particle filter for finding probable states of a model (Java's `ParticleFilter`).
//!
//! The model is considered one observation at a time, together with the latents that observation depends on
//! (ordered so that observations with the fewest new latents come first). For each increment every particle samples
//! the new latents from their prior, given the particle's existing latents, and is weighted by the log probability
//! of the part of the model considered so far. Low probability particles are culled and replaced by copies of
//! higher probability ones.

use crate::error::{AlgorithmError, Result};
use crate::support::evaluate_all;
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
    log_prob: f64,
}

impl Particle {
    fn empty() -> Self {
        Self {
            latents: BTreeMap::new(),
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
            model.set_and_cascade(state, values)?;
        }
        Ok(())
    }

    /// Adds the log probabilities of an increment's newly sampled latents and its observation, with the particle's
    /// values applied to `state`. The log probabilities of the latents and observations considered before don't
    /// change (they depend only on those latents' values and their ancestors', which are already sampled), so they
    /// aren't recomputed.
    fn add_log_prob(
        &mut self,
        graph: &Graph,
        state: &State,
        latents: &[NodeId],
        observed: NodeId,
    ) -> Result<()> {
        let mut increment = 0.0;
        for &id in latents.iter().chain([&observed]) {
            increment += node_log_prob(graph, state, id)?;
        }
        // a NaN log probability means the particle is unusable; treat it as impossible
        self.log_prob = finite_or_neg_inf(self.log_prob + increment);
        Ok(())
    }
}

/// For each observed node, the latents it depends on that no earlier observation depends on, in the order the
/// filter considers them: repeatedly the observation with the fewest remaining latent dependencies (ties broken
/// by node id). Java's `LatentIncrementSort`.
///
/// Only a count of remaining dependencies is kept per observation, so memory is linear in the size of the graph.
/// An observation's remaining latents are found when it is chosen, by walking up from it and stopping at latents
/// already taken (whose own latent ancestors were all taken with them); when a latent is taken, the observations
/// that depend on it are found by walking down from it.
pub fn latent_increments(graph: &Graph, state: &State) -> Vec<(NodeId, Vec<NodeId>)> {
    let is_latent = |id: NodeId| graph.node(id).is_probabilistic() && !state.is_observed(id);
    let mut observation_index: Vec<Option<usize>> = vec![None; graph.len()];
    let observations: Vec<NodeId> = graph
        .ids()
        .filter(|&id| state.is_observed(id) && graph.node(id).is_probabilistic())
        .collect();
    for (i, &id) in observations.iter().enumerate() {
        observation_index[id.index()] = Some(i);
    }

    let mut walker = Walker::new(graph.len());
    // the latent ancestors of an observation that aren't taken (an observation also depends on the latents its
    // observed ancestors depend on, so the walk goes through every kind of node)
    let untaken_latents = |walker: &mut Walker, from: NodeId, taken: &[bool]| {
        let mut found = Vec::new();
        walker.walk(
            from,
            |id| graph.parents(id),
            |id| {
                if !is_latent(id) {
                    return true;
                }
                if taken[id.index()] {
                    return false;
                }
                found.push(id);
                true
            },
        );
        found
    };

    let mut taken = vec![false; graph.len()];
    let mut remaining = vec![0usize; observations.len()];
    let mut queue: BTreeSet<(usize, NodeId)> = BTreeSet::new();
    for (i, &observation) in observations.iter().enumerate() {
        remaining[i] = untaken_latents(&mut walker, observation, &taken).len();
        queue.insert((remaining[i], observation));
    }

    let mut order = Vec::with_capacity(observations.len());
    while let Some((_, next)) = queue.pop_first() {
        let mut latents = untaken_latents(&mut walker, next, &taken);
        latents.sort();
        for &latent in &latents {
            taken[latent.index()] = true;
        }
        // each waiting observation downstream of a newly taken latent depends on one fewer
        for &latent in &latents {
            walker.walk(
                latent,
                |id| graph.children(id),
                |id| {
                    if let Some(i) = observation_index[id.index()]
                        && queue.remove(&(remaining[i], id))
                    {
                        remaining[i] -= 1;
                        queue.insert((remaining[i], id));
                    }
                    true
                },
            );
        }
        order.push((next, latents));
    }
    order
}

/// Depth-first walks over a graph that reuse their visited marks.
struct Walker {
    visited: Vec<u32>,
    walk: u32,
    stack: Vec<NodeId>,
}

impl Walker {
    fn new(nodes: usize) -> Self {
        Self {
            visited: vec![0; nodes],
            walk: 0,
            stack: Vec::new(),
        }
    }

    /// Visits each node reachable from `from` (excluding `from`) through `next` once; `visit` returns whether to
    /// carry on past the node.
    fn walk<'g>(
        &mut self,
        from: NodeId,
        next: impl Fn(NodeId) -> &'g [NodeId],
        mut visit: impl FnMut(NodeId) -> bool,
    ) {
        self.walk = self.walk.checked_add(1).unwrap_or_else(|| {
            self.visited.fill(0);
            1
        });
        self.stack.clear();
        self.stack.extend_from_slice(next(from));
        while let Some(id) = self.stack.pop() {
            if self.visited[id.index()] == self.walk {
                continue;
            }
            self.visited[id.index()] = self.walk;
            if visit(id) {
                self.stack.extend_from_slice(next(id));
            }
        }
    }
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
        evaluate_all(graph, state, rng)?;

        let mut particles = vec![Particle::empty(); self.particles];
        for (observed, dependencies) in latent_increments(graph, state) {
            particles =
                self.update_particles(model, state, observed, &dependencies, &particles, rng)?;
        }
        particles.sort_by(|a, b| rank(b).total_cmp(&rank(a)));
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
            updated.sort_by(|a, b| rank(b).total_cmp(&rank(a)));
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
        for &latent in &ordered {
            let value = state.sample_prior(model.graph(), latent, rng)?;
            model.set_and_cascade(state, [(latent, value.clone())])?;
            particle.latents.insert(latent, value);
        }
        particle.add_log_prob(model.graph(), state, &ordered, observed)?;
    }
    Ok(())
}

/// A log probability with NaN replaced by -infinity, so NaN particles are treated as impossible.
fn finite_or_neg_inf(log_prob: f64) -> f64 {
    if log_prob.is_nan() {
        f64::NEG_INFINITY
    } else {
        log_prob
    }
}

/// The key particles are ranked by: their log probability, with NaN as -infinity.
fn rank(particle: &Particle) -> f64 {
    finite_or_neg_inf(particle.log_prob)
}

/// exp(log prob - max log prob) for each particle, so that very negative log probabilities don't all underflow to
/// zero. A NaN log probability counts as -infinity (weight 0). If no particle is possible they are weighted
/// equally.
pub fn normalised_weights(particles: &[Particle]) -> Vec<f64> {
    let max = particles.iter().map(rank).fold(f64::NEG_INFINITY, f64::max);
    if max == f64::NEG_INFINITY {
        return vec![1.0; particles.len()];
    }
    particles.iter().map(|p| (rank(p) - max).exp()).collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn particle(log_prob: f64) -> Particle {
        Particle {
            log_prob,
            ..Particle::empty()
        }
    }

    #[test]
    fn nan_log_probs_get_zero_weight() {
        let particles = [particle(f64::NAN), particle(-1.0), particle(-2.0)];
        let weights = normalised_weights(&particles);
        assert_eq!(weights[0], 0.0);
        assert_eq!(weights[1], 1.0);
        assert!((weights[2] - (-1.0f64).exp()).abs() < 1e-15);
    }

    #[test]
    fn all_nan_or_impossible_log_probs_are_weighted_equally() {
        let particles = [particle(f64::NAN), particle(f64::NEG_INFINITY)];
        assert_eq!(normalised_weights(&particles), vec![1.0, 1.0]);
    }

    #[test]
    fn a_nan_particle_does_not_collapse_resampling() {
        // the possible particles must each be drawn; a NaN total would pick the last particle every time
        let particles = [particle(-1.0), particle(-1.0), particle(f64::NAN)];
        let mut rng = KeanuRng::seed_from_u64(1);
        let drawn = sample_and_copy(&particles, 100, &mut rng);
        assert!(drawn.iter().all(|p| !p.log_prob.is_nan()));
    }

    #[test]
    fn nan_particles_sort_last() {
        // total_cmp alone ranks NaN by its sign bit: +NaN above every number, -NaN below
        let mut particles = [
            particle(-f64::NAN),
            particle(-5.0),
            particle(f64::NAN),
            particle(-1.0),
        ];
        particles.sort_by(|a, b| rank(b).total_cmp(&rank(a)));
        assert_eq!(particles[0].log_prob, -1.0);
        assert_eq!(particles[1].log_prob, -5.0);
    }
}
