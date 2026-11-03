use keanu_algorithms::{AlgorithmError, NetworkSamples};
use keanu_graph::{Graph, GraphBuilder, NodeId, State};
use keanu_tensor::{BoolTensor, DoubleTensor};

/// Samples of a vector node, an int node and a bool node, with values 0..count.
fn samples(count: usize) -> (Graph, NetworkSamples, NodeId, NodeId, NodeId) {
    let b = GraphBuilder::new();
    let x = b.gaussian_with_shape(&[2], 0.0, 1.0).label("x");
    let n = b.poisson(1.0);
    let flag = b.bernoulli(0.5);
    let (x, n, flag) = (x.id(), n.id(), flag.id());
    let (graph, mut state): (Graph, State) = b.build().unwrap();
    let mut samples = NetworkSamples::new(&graph, &[x, n, flag]);
    for i in 0..count {
        state
            .set(&graph, x, DoubleTensor::vector(vec![i as f64, -(i as f64)]))
            .unwrap();
        state
            .set(&graph, n, keanu_tensor::IntTensor::scalar((i % 3) as i64))
            .unwrap();
        state
            .set(&graph, flag, BoolTensor::scalar(i % 2 == 0))
            .unwrap();
        samples.record(&state, i as f64).unwrap();
    }
    (graph, samples, x, n, flag)
}

#[test]
fn down_sample_keeps_every_nth_starting_with_the_first() {
    let (_, s, x, _, _) = samples(5);
    let down = s.down_sample(2).unwrap();
    assert_eq!(down.len(), 3);
    let firsts: Vec<f64> = down
        .f64(x)
        .unwrap()
        .as_slice()
        .iter()
        .map(|t| t.as_slice()[0])
        .collect();
    assert_eq!(firsts, vec![0.0, 2.0, 4.0]);
    assert_eq!(down.log_probs(), &[0.0, 2.0, 4.0]);
    assert_eq!(s.down_sample(1).unwrap(), s);
    assert!(s.down_sample(0).is_err());
}

#[test]
fn drop_removes_the_first_samples() {
    let (_, s, x, _, _) = samples(5);
    let dropped = s.drop(2).unwrap();
    assert_eq!(dropped.len(), 3);
    assert_eq!(dropped.f64(x).unwrap().as_slice()[0].as_slice()[0], 2.0);
    assert_eq!(s.drop(5).unwrap().len(), 0);
    assert!(matches!(s.drop(6), Err(AlgorithmError::InvalidArgument(_))));
}

#[test]
fn means_variances_and_tensors() {
    let (graph, s, x, _, _) = samples(4);
    let xs = s.f64_by_label(&graph, "x").unwrap();
    assert_eq!(xs.mean().unwrap().as_slice(), &[1.5, -1.5]);
    let variance = xs.variance().unwrap();
    for v in variance.as_slice() {
        assert!((v - 5.0 / 3.0).abs() < 1e-12);
    }
    assert_eq!(xs.element(&[1]).unwrap(), vec![0.0, -1.0, -2.0, -3.0]);
    let stacked = s.f64(x).unwrap().to_tensor().unwrap();
    assert_eq!(stacked.shape(), &[4, 2]);
    assert!(xs.autocorrelation(&[0]).unwrap()[0] == 1.0);
    assert!(xs.element(&[5]).is_err());
}

#[test]
fn modes_probabilities_and_states() {
    let (_, s, x, n, flag) = samples(7);
    // n cycles 0, 1, 2, 0, 1, 2, 0
    assert_eq!(s.i64(n).unwrap().mode().unwrap().to_scalar(), Some(0));
    assert!(
        (s.bool(flag)
            .unwrap()
            .probability(|t| t.to_scalar() == Some(true))
            - 4.0 / 7.0)
            .abs()
            < 1e-15
    );
    assert!(
        (s.probability(|sample| sample.f64(x).unwrap().as_slice()[0] >= 5.0) - 2.0 / 7.0).abs()
            < 1e-15
    );
    let best = s.most_probable().unwrap();
    assert_eq!(best.index(), 6);
    assert_eq!(best.log_prob(), 6.0);
    assert_eq!(s.sample(3).unwrap().i64(n).unwrap().to_scalar(), Some(0));
    assert!(s.sample(7).is_none());
    assert_eq!(s.iter().count(), 7);
    // wrong type accessors
    assert!(s.f64(n).is_none());
    assert!(s.sample(0).unwrap().bool(x).is_none());
}
