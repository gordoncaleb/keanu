use keanu_algorithms::{AlgorithmError, ForwardSampler};
use keanu_distributions::KeanuRng;
use keanu_graph::{GraphBuilder, GraphError, Model};
use keanu_tensor::DoubleTensor;

#[test]
fn samples_the_prior_and_deterministic_descendants() {
    let b = GraphBuilder::new();
    let a = b.gaussian(100.0, 1.0);
    let c = a + 5.0;
    let d = b.gaussian(c, 2.0);
    let (a, c, d) = (a.id(), c.id(), d.id());
    let (graph, mut state) = b.build().unwrap();
    let mut rng = KeanuRng::seed_from_u64(1);
    let all: Vec<_> = graph.ids().collect();
    state.lazy_eval(&graph, &all, &mut rng).unwrap();
    let model = Model::new(graph, &state).unwrap();

    let samples = ForwardSampler::new()
        .sample(&model, &mut state, &[a, c, d], 50_000, &mut rng)
        .unwrap();
    assert_eq!(samples.len(), 50_000);
    let mean = |id| {
        samples
            .f64(id)
            .unwrap()
            .mean()
            .unwrap()
            .to_scalar()
            .unwrap()
    };
    let var = |id| {
        samples
            .f64(id)
            .unwrap()
            .variance()
            .unwrap()
            .to_scalar()
            .unwrap()
    };
    assert!((mean(a) - 100.0).abs() < 0.03);
    assert!((mean(c) - 105.0).abs() < 0.03);
    assert!((mean(d) - 105.0).abs() < 0.06);
    // Var(d) = Var(a) + 2^2
    assert!((var(d) - 5.0).abs() < 0.15, "{}", var(d));
    // c is always a + 5
    for s in samples.iter() {
        assert_eq!(
            s.f64(c).unwrap().to_scalar().unwrap(),
            s.f64(a).unwrap().to_scalar().unwrap() + 5.0
        );
    }
    // log probs are only recorded on request
    assert!(samples.log_probs().iter().all(|&lp| lp == 0.0));
    let with_lp = ForwardSampler::default()
        .with_calculate_log_prob(true)
        .sample(&model, &mut state, &[a], 3, &mut rng)
        .unwrap();
    assert!(with_lp.log_probs().iter().all(|&lp| lp < 0.0));
}

#[test]
fn an_observation_downstream_of_a_latent_is_an_error() {
    let b = GraphBuilder::new();
    let a = b.gaussian(100.0, 1.0);
    let c = b.gaussian(a + 5.0, 1.0);
    b.observe(c, DoubleTensor::scalar(100.0));
    let a = a.id();
    let (graph, mut state) = b.build().unwrap();
    let mut rng = KeanuRng::seed_from_u64(1);
    let all: Vec<_> = graph.ids().collect();
    state.lazy_eval(&graph, &all, &mut rng).unwrap();
    let model = Model::new(graph, &state).unwrap();
    let error = ForwardSampler::new()
        .sample(&model, &mut state, &[a], 10, &mut rng)
        .unwrap_err();
    assert!(matches!(error, AlgorithmError::InvalidArgument(_)));
}

#[test]
fn several_observations_without_random_parents_are_fine() {
    // regression for Java's check, which rejected any model with two or more observations
    let b = GraphBuilder::new();
    let a = b.gaussian(100.0, 1.0);
    let one = b.gaussian(0.0, 1.0);
    let two = b.gaussian(one, 1.0);
    b.observe(one, DoubleTensor::scalar(1.0));
    b.observe(two, DoubleTensor::scalar(2.0));
    let joined = a + two;
    let joined = joined.id();
    let (graph, mut state) = b.build().unwrap();
    let mut rng = KeanuRng::seed_from_u64(2);
    let all: Vec<_> = graph.ids().collect();
    state.lazy_eval(&graph, &all, &mut rng).unwrap();
    let model = Model::new(graph, &state).unwrap();
    let samples = ForwardSampler::new()
        .sample(&model, &mut state, &[joined], 2_000, &mut rng)
        .unwrap();
    let mean = samples
        .f64(joined)
        .unwrap()
        .mean()
        .unwrap()
        .to_scalar()
        .unwrap();
    assert!((mean - 102.0).abs() < 0.1, "{mean}");
}

#[test]
fn hierarchical_prior() {
    // mu ~ N(0, 1), x ~ N(mu, 1): Var(x) = 2
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 1.0);
    let x = b.gaussian(mu, 1.0);
    let x = x.id();
    let (graph, mut state) = b.build().unwrap();
    let mut rng = KeanuRng::seed_from_u64(3);
    let all: Vec<_> = graph.ids().collect();
    state.lazy_eval(&graph, &all, &mut rng).unwrap();
    let model = Model::new(graph, &state).unwrap();
    let samples = ForwardSampler::new()
        .sample(&model, &mut state, &[x], 50_000, &mut rng)
        .unwrap();
    let variance = samples
        .f64(x)
        .unwrap()
        .variance()
        .unwrap()
        .to_scalar()
        .unwrap();
    assert!((variance - 2.0).abs() < 0.06, "{variance}");
}

#[test]
fn no_latents_is_an_error() {
    let b = GraphBuilder::new();
    let x = b.gaussian(0.0, 1.0);
    b.observe(x, DoubleTensor::scalar(0.5));
    let x = x.id();
    let (graph, mut state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    let error = ForwardSampler::new()
        .sample(&model, &mut state, &[x], 1, &mut KeanuRng::seed_from_u64(0))
        .unwrap_err();
    assert_eq!(error, AlgorithmError::NoLatents);
}

/// Regression: with log probabilities on, only the nodes the requested ones depend on were resampled, but the
/// log probability covers the whole model, so deterministic nodes downstream of a resampled latent were stale.
#[test]
fn log_probs_use_up_to_date_downstream_nodes() {
    let b = GraphBuilder::new();
    let a = b.gaussian(0.0, 1.0);
    let doubled = a * 2.0;
    let c = b.gaussian(doubled, 1.0);
    let (a, doubled, _c) = (a.id(), doubled.id(), c.id());
    let (graph, mut state) = b.build().unwrap();
    let mut rng = KeanuRng::seed_from_u64(3);
    let all: Vec<_> = graph.ids().collect();
    state.lazy_eval(&graph, &all, &mut rng).unwrap();
    let model = Model::new(graph, &state).unwrap();

    let samples = ForwardSampler::default()
        .with_calculate_log_prob(true)
        .sample(&model, &mut state, &[a], 20, &mut rng)
        .unwrap();
    for sample in samples.iter() {
        // the joint log probability with this value of a and everything downstream recomputed from it
        let mut expected = state.clone();
        model
            .set_and_cascade(&mut expected, [(a, sample.get(a).unwrap())])
            .unwrap();
        let expected = model.log_prob(&expected).unwrap();
        assert!(
            (sample.log_prob() - expected).abs() < 1e-12,
            "{} vs {expected}",
            sample.log_prob()
        );
    }
    let a_value = state.f64(a).unwrap().to_scalar().unwrap();
    assert_eq!(
        state.f64(doubled).unwrap().to_scalar().unwrap(),
        2.0 * a_value
    );
}

#[test]
fn an_unknown_node_is_an_error_not_a_panic() {
    // a node of a larger graph is not a node of this one
    let other = GraphBuilder::new();
    let _ = other.gaussian(0.0, 1.0);
    let foreign = other.gaussian(0.0, 1.0).id();

    let b = GraphBuilder::new();
    let a = b.gaussian(0.0, 1.0).id();
    let (graph, mut state) = b.build().unwrap();
    let mut rng = KeanuRng::seed_from_u64(1);
    let all: Vec<_> = graph.ids().collect();
    state.lazy_eval(&graph, &all, &mut rng).unwrap();
    let model = Model::new(graph, &state).unwrap();
    let error = ForwardSampler::new()
        .sample(&model, &mut state, &[a, foreign], 10, &mut rng)
        .unwrap_err();
    assert_eq!(
        error,
        AlgorithmError::Graph(GraphError::UnknownNode(foreign))
    );
}
