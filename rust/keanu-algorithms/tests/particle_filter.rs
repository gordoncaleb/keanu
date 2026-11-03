use keanu_algorithms::particle_filter::{ParticleFilter, latent_increments, normalised_weights};
use keanu_distributions::KeanuRng;
use keanu_graph::{GraphBuilder, Model};
use keanu_tensor::DoubleTensor;

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

    let filter = ParticleFilter {
        particles: 500,
        ..ParticleFilter::default()
    };
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

    let filter = ParticleFilter {
        particles: 200,
        resampling_cycles: 1,
        ..ParticleFilter::default()
    };
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
    let particles = ParticleFilter {
        particles: 50,
        ..ParticleFilter::default()
    }
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
    let filter = ParticleFilter {
        resampling_proportion: 1.0,
        ..ParticleFilter::default()
    };
    assert!(
        filter
            .run(&model, &mut state, &mut KeanuRng::seed_from_u64(1))
            .is_err()
    );
}
