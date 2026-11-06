//! Masked observations under inference: the default sampler for the imputed latents, Gibbs on discrete missing
//! values against the exact posterior, and the pointwise log likelihood of the observed elements alone.

use keanu_algorithms::NetworkSamples;
use keanu_algorithms::loglik::PointwiseLogLik;
use keanu_algorithms::mcmc::SamplingOptions;
use keanu_algorithms::mcmc::select::Sampler;
use keanu_distributions::KeanuRng;
use keanu_graph::{GraphBuilder, MaskedObservation, Model, observe_masked};
use keanu_tensor::{BoolTensor, DoubleTensor};

#[test]
fn discrete_missing_values_are_sampled_by_gibbs() {
    // which of two coins? three tosses seen, two lost: each lost toss is heads with probability
    // P(biased | data) 0.9 + P(fair | data) 0.2
    let b = GraphBuilder::new();
    let biased = b.bernoulli(0.5).label("biased");
    let p = b.where_(biased, b.scalar(0.9), b.scalar(0.2));
    let tosses = b.bernoulli_with_shape(&[5], p).label("tosses");
    b.set_value(biased, BoolTensor::scalar(true));
    let (biased, tosses) = (biased.id(), tosses.id());
    let (graph, state) = b.build().unwrap();
    let seen = BoolTensor::vector(vec![true, true, true, false, false]);
    let lost = BoolTensor::vector(vec![false, false, false, true, true]);
    let (graph, mut state, imputed) = observe_masked(
        &graph,
        &state,
        &[MaskedObservation::new(tosses, seen, lost)],
    )
    .unwrap();
    let record = imputed.get(tosses).unwrap();
    let model = Model::new(graph, &state).unwrap();
    let sampler = Sampler::for_model(&model, &state);
    assert!(matches!(sampler, Sampler::Gibbs(_)), "{sampler:?}");

    let (biased, whole) = (imputed.node(biased).unwrap(), record.node());
    let samples = sampler
        .sample(
            &model,
            &mut state,
            &[biased, whole],
            20_000,
            SamplingOptions::default().with_drop(1000),
            &mut KeanuRng::seed_from_u64(3),
        )
        .unwrap();
    let p_biased = 0.9f64.powi(3) / (0.9f64.powi(3) + 0.2f64.powi(3));
    let p_heads = p_biased * 0.9 + (1.0 - p_biased) * 0.2;
    let draws = samples.bool(whole).unwrap();
    for i in 0..5 {
        let heads = draws.probability(|t| t.as_slice()[i]);
        let expected = if i < 3 { 1.0 } else { p_heads };
        assert!(
            (heads - expected).abs() < 0.02,
            "toss {i}: {heads} against {expected}"
        );
    }
    let biased = samples
        .bool(biased)
        .unwrap()
        .probability(|t| t.as_slice()[0]);
    assert!(
        (biased - p_biased).abs() < 0.02,
        "{biased} against {p_biased}"
    );
}

#[test]
fn continuous_missing_values_get_nuts_and_only_observed_elements_have_points() {
    let b = GraphBuilder::new();
    let mu = b.gaussian(0.0, 5.0).label("mu");
    let y = b.gaussian_with_shape(&[5], mu, 1.0).label("y");
    b.set_value(mu, DoubleTensor::scalar(0.0));
    let (mu, y) = (mu.id(), y.id());
    let (graph, state) = b.build().unwrap();
    let values = DoubleTensor::vector(vec![0.5, f64::NAN, 1.5, f64::NAN, -0.2]);
    let (graph, state, imputed) = observe_masked(
        &graph,
        &state,
        &[MaskedObservation::with_nan_missing(y, values)],
    )
    .unwrap();
    let model = Model::new(graph, &state).unwrap();
    assert!(matches!(
        Sampler::for_model(&model, &state),
        Sampler::Nuts(_)
    ));

    let mu = imputed.node(mu).unwrap();
    let mut samples = NetworkSamples::new(model.graph(), &[mu]).unwrap();
    let mut work = state.clone();
    for value in [0.3, 1.1] {
        work.set_and_cascade(model.graph(), mu, DoubleTensor::scalar(value))
            .unwrap();
        samples.record(&work, 0.0).unwrap();
    }
    let loglik = PointwiseLogLik::from_samples(&model, &state, &samples).unwrap();
    assert_eq!(
        loglik.names(),
        &["y_observed[0]", "y_observed[1]", "y_observed[2]"]
    );
    let ln_normal =
        |x: f64, m: f64| -0.5 * (x - m).powi(2) - 0.5 * (2.0 * std::f64::consts::PI).ln();
    for (draw, m) in [0.3, 1.1].into_iter().enumerate() {
        let expected: Vec<f64> = [0.5, 1.5, -0.2].iter().map(|&x| ln_normal(x, m)).collect();
        for (a, e) in loglik.chain(0).unwrap()[draw].iter().zip(&expected) {
            assert!((a - e).abs() < 1e-12);
        }
    }
}
