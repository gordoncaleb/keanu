//! The interventional predictive with a latent mediator: a latent downstream of the intervened node is drawn again
//! given its new value, not kept at its posterior draws (which were made given the old one).

use keanu_algorithms::{NetworkSamples, PredictiveSampler};
use keanu_distributions::KeanuRng;
use keanu_graph::{GraphBuilder, Model, intervene};
use keanu_tensor::DoubleTensor;

#[test]
fn the_interventional_predictive_redraws_a_latent_mediator_given_the_intervened_value() {
    // x ~ N(0, 1) observed 0, m ~ N(x, 0.1) latent, y ~ N(m, 0.1) observed 0: the posterior has m near 0, but under
    // do(x = 5) the mediator is near 5, and so is y
    let b = GraphBuilder::new();
    let x = b.gaussian(0.0, 1.0).label("x");
    let m = b.gaussian(x, 0.1).label("m");
    let y = b.gaussian(m, 0.1).label("y");
    b.observe(x, DoubleTensor::scalar(0.0));
    b.observe(y, DoubleTensor::scalar(0.0));
    b.set_value(m, DoubleTensor::scalar(0.0));
    let (x, m, y) = (x.id(), m.id(), y.id());
    let (graph, state) = b.build().unwrap();
    // posterior draws of m: N(0, 0.005)
    let mut samples = NetworkSamples::new(&graph, &[m]).unwrap();
    let mut work = state.clone();
    let mut rng = KeanuRng::seed_from_u64(3);
    for _ in 0..2000 {
        let value = 0.005f64.sqrt() * rng.next_gaussian();
        work.set(&graph, m, DoubleTensor::scalar(value)).unwrap();
        samples.record(&work, 0.0).unwrap();
    }

    let (graph_do, state_do, intervened) =
        intervene(&graph, &state, &[(x, DoubleTensor::scalar(5.0).into())]).unwrap();
    assert!(intervened.is_downstream(m) && intervened.is_downstream(y));
    assert!(!intervened.is_downstream(x));
    assert_eq!(intervened.downstream(), vec![m, y]);
    assert_eq!(intervened.carried(m), None);
    let model_do = Model::new(graph_do, &state_do).unwrap();
    let draws = PredictiveSampler::new()
        .interventional(&model_do, &state_do, &intervened, &samples, &[y], &mut rng)
        .unwrap();
    let ys: Vec<f64> = draws
        .f64(intervened.node(y).unwrap())
        .unwrap()
        .as_slice()
        .iter()
        .map(|t| t.to_scalar().unwrap())
        .collect();
    let n = ys.len() as f64;
    let mean = ys.iter().sum::<f64>() / n;
    let variance = ys.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    // y | do(x = 5) ~ N(5, 0.1² + 0.1²)
    assert!((mean - 5.0).abs() < 0.02, "mean {mean}");
    assert!((variance - 0.02).abs() < 0.003, "variance {variance}");
}
