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

#[test]
fn variance_is_accurate_for_a_large_mean() {
    let b = GraphBuilder::new();
    let x = b.gaussian(0.0, 1.0).id();
    let (graph, mut state): (Graph, State) = b.build().unwrap();
    let mut samples = NetworkSamples::new(&graph, &[x]);
    // spread 1 around 1e9: E[x^2] - E[x]^2 cancels catastrophically here
    for offset in [-1.0, 0.0, 1.0, 0.0] {
        state
            .set(&graph, x, DoubleTensor::scalar(1e9 + offset))
            .unwrap();
        samples.record(&state, 0.0).unwrap();
    }
    let variance = samples.f64(x).unwrap().variance().unwrap().as_slice()[0];
    assert!((variance - 2.0 / 3.0).abs() < 1e-9, "{variance}");
}

#[test]
fn a_type_mismatch_leaves_the_samples_unchanged() {
    let a = GraphBuilder::new();
    let (x, y) = (a.gaussian(0.0, 1.0).id(), a.gaussian(0.0, 1.0).id());
    let (graph_a, _): (Graph, State) = a.build().unwrap();
    // the same ids, but the second node is an int node
    let b = GraphBuilder::new();
    let (x_b, y_b) = (b.gaussian(0.0, 1.0).id(), b.uniform_int(0, 4).id());
    let (graph_b, mut state_b): (Graph, State) = b.build().unwrap();
    state_b
        .set(&graph_b, x_b, DoubleTensor::scalar(0.5))
        .unwrap();
    state_b
        .set(&graph_b, y_b, keanu_tensor::IntTensor::scalar(1))
        .unwrap();
    assert_eq!((x_b, y_b), (x, y));

    let mut samples = NetworkSamples::new(&graph_a, &[x, y]);
    assert!(matches!(
        samples.record(&state_b, 0.0),
        Err(AlgorithmError::InvalidArgument(_))
    ));
    assert_eq!(samples.len(), 0);
    assert_eq!(samples.f64(x).unwrap().len(), 0);
}

/// Samples of a scalar node with the given values.
fn scalar_samples(values: &[f64]) -> (NetworkSamples, NodeId) {
    let b = GraphBuilder::new();
    let x = b.gaussian(0.0, 1.0).id();
    let (graph, mut state): (Graph, State) = b.build().unwrap();
    let mut samples = NetworkSamples::new(&graph, &[x]);
    for &v in values {
        state.set(&graph, x, DoubleTensor::scalar(v)).unwrap();
        samples.record(&state, 0.0).unwrap();
    }
    (samples, x)
}

fn mode_of(values: &[f64]) -> Option<f64> {
    let (samples, x) = scalar_samples(values);
    samples
        .f64(x)
        .unwrap()
        .mode()
        .map(|t| t.to_scalar().unwrap())
}

/// The original quadratic mode: distinct values in order of first appearance, the first with the highest count.
fn reference_mode(values: &[f64]) -> Option<f64> {
    let mut distinct: Vec<(f64, usize)> = Vec::new();
    for &v in values {
        match distinct.iter_mut().find(|(d, _)| *d == v) {
            Some((_, count)) => *count += 1,
            None => distinct.push((v, 1)),
        }
    }
    let mut best: Option<(f64, usize)> = None;
    for (v, count) in distinct {
        if best.is_none_or(|(_, c)| count > c) {
            best = Some((v, count));
        }
    }
    best.map(|(v, _)| v)
}

/// Regression: `mode` compared every sample with every distinct value so far (quadratic). It now sorts, keeping
/// the same answer: of the values with the highest count, the one that appears first.
#[test]
fn mode_breaks_ties_by_first_appearance() {
    assert_eq!(mode_of(&[]), None);
    assert_eq!(mode_of(&[2.0, 1.0, 1.0, 2.0]), Some(2.0));
    assert_eq!(mode_of(&[3.0, 1.0, 2.0, 2.0, 1.0]), Some(1.0));
    assert_eq!(mode_of(&[3.0, 1.0, 2.0, 2.0, 1.0, 2.0]), Some(2.0));
    // -0 == 0, so they count together, as the first of them
    let zero = mode_of(&[5.0, -0.0, 0.0, 5.0, 0.0]).unwrap();
    assert!(zero == 0.0 && zero.is_sign_negative());
    // NaN is never equal to anything, so each counts once
    assert!(mode_of(&[f64::NAN, f64::NAN, 4.0]).unwrap().is_nan());
    assert_eq!(mode_of(&[f64::NAN, 3.0, f64::NAN, 3.0]), Some(3.0));

    // agrees with the original on many small sequences
    let mut seed = 12345u64;
    for _ in 0..500 {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let len = ((seed >> 33) % 12) as usize;
        let values: Vec<f64> = (0..len)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                match (seed >> 33) % 6 {
                    5 => f64::NAN,
                    k => k as f64 - 2.0,
                }
            })
            .collect();
        let (got, expected) = (mode_of(&values), reference_mode(&values));
        let same = match (got, expected) {
            (Some(g), Some(e)) => g.to_bits() == e.to_bits() || (g.is_nan() && e.is_nan()),
            (g, e) => g == e,
        };
        assert!(same, "{values:?}: {got:?} vs {expected:?}");
    }
}

#[test]
fn mode_of_vector_and_bool_samples() {
    let (_, s, x, _, flag) = samples(5);
    // every x is distinct, so the first wins
    assert_eq!(s.f64(x).unwrap().mode().unwrap().as_slice(), &[0.0, -0.0]);
    // flag is true, false, true, false, true
    assert_eq!(
        s.bool(flag).unwrap().mode().unwrap().to_scalar(),
        Some(true)
    );
}
