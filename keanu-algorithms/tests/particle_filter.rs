use keanu_algorithms::particle_filter::{ParticleFilter, latent_increments, normalised_weights};
use keanu_distributions::KeanuRng;
use keanu_graph::{Graph, GraphBuilder, Model, NodeId, State, node_log_prob};
use keanu_tensor::DoubleTensor;
use std::collections::{BTreeMap, BTreeSet};

fn scalar(v: &keanu_graph::Value) -> f64 {
    v.as_f64().unwrap().as_slice()[0]
}

#[test]
fn finds_the_most_probable_state() {
    // x ~ N(0, 1), y ~ N(x, 0.5) observed at 2: the posterior mode of x is 2 / (1 + 0.25) = 1.6
    let b = GraphBuilder::new();
    let x = b.gaussian(0.0, 1.0);
    let y = b.gaussian(x, 0.5);
    b.observe(y, DoubleTensor::scalar(2.0));
    let x = x.id();
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();

    let filter = ParticleFilter::default().with_particles(500);
    let particles = filter
        .run(&model, &mut state, &mut KeanuRng::seed_from_u64(1))
        .unwrap();
    assert_eq!(particles.len(), 500);
    assert!(
        particles
            .windows(2)
            .all(|w| w[0].log_prob() >= w[1].log_prob())
    );
    let best = scalar(particles[0].value(x).unwrap());
    assert!((best - 1.6).abs() < 0.15, "most probable x {best}");
    // the state is left at the most probable particle
    assert_eq!(state.f64(x).unwrap().as_slice()[0], best);
}

/// Regression: each particle must sample new latents given its own values for the latents it already has, not
/// the values of whichever particle was evaluated last.
#[test]
fn particles_sample_new_latents_given_their_own_parents() {
    let b = GraphBuilder::new();
    let a = b.gaussian(0.0, 10.0);
    let c = b.gaussian(a, 0.01);
    let y1 = b.gaussian(a, 5.0);
    let y2 = b.gaussian(c, 5.0);
    b.observe(y1, DoubleTensor::scalar(1.0));
    b.observe(y2, DoubleTensor::scalar(1.0));
    let (a, c) = (a.id(), c.id());
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();

    let increments = latent_increments(model.graph(), &state);
    assert_eq!(increments.len(), 2);
    assert_eq!(increments[0].1, vec![a]);
    assert_eq!(increments[1].1, vec![c]);

    let filter = ParticleFilter::default()
        .with_particles(200)
        .with_resampling_cycles(1);
    let particles = filter
        .run(&model, &mut state, &mut KeanuRng::seed_from_u64(2))
        .unwrap();
    for p in &particles {
        let (av, cv) = (scalar(p.value(a).unwrap()), scalar(p.value(c).unwrap()));
        assert!((av - cv).abs() < 0.1, "particle has a = {av} but c = {cv}");
    }
}

/// Regression: weights are normalised by the largest log probability, so very improbable particles don't all
/// underflow to zero weight.
#[test]
fn weights_do_not_underflow() {
    let b = GraphBuilder::new();
    let x = b.gaussian(0.0, 1.0);
    let y = b.gaussian_with_shape(&[400], x, 0.01);
    b.observe(y, DoubleTensor::full(&[400], 3.0));
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let particles = ParticleFilter::default()
        .with_particles(50)
        .run(&model, &mut state, &mut KeanuRng::seed_from_u64(3))
        .unwrap();
    // every log probability is hugely negative, yet the weights are well defined
    assert!(particles.iter().all(|p| p.log_prob() < -1000.0));
    let weights = normalised_weights(&particles);
    assert_eq!(weights.iter().copied().fold(0.0, f64::max), 1.0);
    assert!(weights.iter().all(|w| w.is_finite()));
}

#[test]
fn rejects_invalid_settings() {
    let b = GraphBuilder::new();
    let x = b.gaussian(0.0, 1.0);
    let y = b.gaussian(x, 1.0);
    b.observe(y, DoubleTensor::scalar(0.0));
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let filter = ParticleFilter::default().with_resampling_proportion(1.0);
    assert!(
        filter
            .run(&model, &mut state, &mut KeanuRng::seed_from_u64(1))
            .is_err()
    );
}

/// A random walk x_t ~ N(0.9 x_{t-1}, 1) observed through y_t ~ N(x_t, 0.5), with a deterministic node between
/// the steps, plus an observation z ~ N(y_last, 1) whose parent is itself an observation.
fn random_walk(steps: usize) -> (Model, State) {
    let b = GraphBuilder::new();
    let mut x = b.gaussian(0.0, 1.0);
    let mut y = b.gaussian(x, 0.5);
    b.observe(y, DoubleTensor::scalar(0.0));
    for t in 1..steps {
        x = b.gaussian(x * 0.9, 1.0);
        y = b.gaussian(x, 0.5);
        b.observe(y, DoubleTensor::scalar((t as f64 * 0.3).sin()));
    }
    let z = b.gaussian(y, 1.0);
    b.observe(z, DoubleTensor::scalar(0.5));
    let (graph, state) = b.build().unwrap();
    (Model::new(graph, &state).unwrap(), state)
}

/// Regression: each increment recomputed the log probability of everything considered so far (quadratic in the
/// length of a time series) after cascading the particle again. It now adds the new latents' and observation's log
/// probabilities; the total must still be the log probability of all of the particle's latents and observations.
#[test]
fn particle_log_probs_are_their_latents_and_observations_log_probs() {
    let (model, state) = random_walk(15);
    let mut scratch = state.clone();
    let particles = ParticleFilter::default()
        .with_particles(30)
        .run(&model, &mut scratch, &mut KeanuRng::seed_from_u64(4))
        .unwrap();
    let graph = model.graph();
    for particle in &particles {
        let mut state = scratch.clone();
        particle.apply(&model, &mut state).unwrap();
        let expected: f64 = particle
            .latents()
            .map(|(id, _)| id)
            .chain(model.observed().iter().copied())
            .map(|id| node_log_prob(graph, &state, id).unwrap())
            .sum();
        assert_eq!(particle.latents().count(), model.latents().len());
        assert!(
            (particle.log_prob() - expected).abs() <= 1e-12 * expected.abs(),
            "{} vs {expected}",
            particle.log_prob()
        );
    }
}

/// The original increment ordering, with every node's latent ancestors as a set.
fn reference_increments(graph: &Graph, state: &State) -> Vec<(NodeId, Vec<NodeId>)> {
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
        let deps = remaining.remove(&next).unwrap();
        for other in remaining.values_mut() {
            for d in &deps {
                other.remove(d);
            }
        }
        order.push((next, deps.into_iter().collect()));
    }
    order
}

/// Regression: `latent_increments` kept the set of latent ancestors of every node (quadratic memory on a chain).
/// It now keeps counts, and must order the increments exactly as before.
#[test]
fn latent_increments_match_the_set_based_ordering() {
    let (model, state) = random_walk(12);
    assert_eq!(
        latent_increments(model.graph(), &state),
        reference_increments(model.graph(), &state)
    );

    // a tree of shared latents with observations at several depths, ties and observed parents
    let b = GraphBuilder::new();
    let root = b.gaussian(0.0, 1.0);
    let a = b.gaussian(root, 1.0);
    let c = b.gaussian(root * 2.0, 1.0);
    let d = b.gaussian(a + c, 1.0);
    let e = b.gaussian(0.0, 1.0);
    let observations = [
        b.gaussian(d, 1.0),
        b.gaussian(a, 1.0),
        b.gaussian(e, 1.0),
        b.gaussian(c + e, 1.0),
        b.gaussian(root, 1.0),
    ];
    let downstream = b.gaussian(observations[0] + observations[2], 1.0);
    for o in observations.iter().chain([&downstream]) {
        b.observe(*o, DoubleTensor::scalar(1.0));
    }
    let (graph, state) = b.build().unwrap();
    let increments = latent_increments(&graph, &state);
    assert_eq!(increments, reference_increments(&graph, &state));
    assert_eq!(increments.len(), 6);
}
