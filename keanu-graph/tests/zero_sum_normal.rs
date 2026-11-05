//! Zero-sum normal nodes: their samples sum to zero, their log probability is the distribution's, they map to
//! N − 1 unconstrained coordinates and back, and malformed shapes are build errors.

use keanu_distributions::KeanuRng;
use keanu_graph::{GraphBuilder, GraphError, Model, NodeId, Parameterisation, Space, Support};
use keanu_tensor::DoubleTensor;

mod common;
use common::t;

#[test]
fn samples_sum_to_zero_and_the_log_probability_is_the_distributions() {
    let b = GraphBuilder::new();
    let sigma = b.half_gaussian(1.0);
    b.set_value(sigma, DoubleTensor::scalar(0.7));
    let effects = b.zero_sum_normal_with_shape(&[3, 6], sigma);
    let (effects, sigma) = (effects.id(), sigma.id());
    let (graph, mut state) = b.build().unwrap();
    let all: Vec<NodeId> = graph.ids().collect();
    state
        .lazy_eval(&graph, &all, &mut KeanuRng::seed_from_u64(5))
        .unwrap();
    let value = state.f64(effects).unwrap().clone();
    assert_eq!(value.shape(), &[3, 6]);
    for row in value.as_slice().chunks_exact(6) {
        assert!(row.iter().sum::<f64>().abs() < 1e-12, "{row:?}");
    }
    let model = Model::new(graph, &state).unwrap();
    let expected = keanu_distributions::ZeroSumNormal::new(DoubleTensor::scalar(0.7))
        .log_prob(&value)
        .unwrap()
        + keanu_distributions::HalfGaussian::new(DoubleTensor::scalar(1.0))
            .log_prob(&DoubleTensor::scalar(0.7))
            .unwrap()
            .sum();
    let actual = model.log_prob(&state).unwrap();
    assert!((actual - expected).abs() < 1e-12, "{actual} vs {expected}");

    // in unconstrained space the effects have 3 × 5 coordinates, and the round trip keeps the value
    let param = Parameterisation::with_space(&model, &state, Space::Unconstrained).unwrap();
    assert_eq!(param.range(effects).unwrap().len(), 15);
    assert_eq!(param.range(sigma).unwrap().len(), 1);
    let z = param.from_state(&state).unwrap();
    let mut other = state.clone();
    param.to_state(&z, &mut other).unwrap();
    let back = other.f64(effects).unwrap();
    for (a, e) in back.as_slice().iter().zip(value.as_slice()) {
        assert!((a - e).abs() < 1e-12);
    }
    assert_eq!(
        keanu_graph::node_support(model.graph(), &state, effects).unwrap(),
        Support::SumToZero
    );
}

#[test]
fn a_value_that_does_not_sum_to_zero_is_impossible() {
    let b = GraphBuilder::new();
    let y = b.zero_sum_normal(3, 1.0);
    b.observe(y, t(&[3], &[1.0, 0.0, 0.0]));
    let (graph, state) = b.build().unwrap();
    let model = Model::new(graph, &state).unwrap();
    assert_eq!(model.log_prob(&state).unwrap(), f64::NEG_INFINITY);
}

#[test]
fn malformed_shapes_are_build_errors() {
    let message = |build: fn(&GraphBuilder)| {
        let b = GraphBuilder::new();
        build(&b);
        match b.build() {
            Err(GraphError::InvalidNode { op, message }) => {
                assert_eq!(op, "zero_sum_normal");
                message
            }
            Err(e) => panic!("expected an invalid node, got {e}"),
            Ok(_) => panic!("expected the build to fail"),
        }
    };
    assert!(
        message(|b| {
            b.zero_sum_normal_with_shape(&[], 1.0);
        })
        .contains("[..., N]")
    );
    assert!(
        message(|b| {
            b.zero_sum_normal(0, 1.0);
        })
        .contains("N > 0")
    );
    assert!(
        message(|b| {
            b.zero_sum_normal_with_shape(&[2, 4], DoubleTensor::zeros(&[2, 1]));
        })
        .contains("sigma")
    );
    // a sigma that doesn't broadcast at all
    let b = GraphBuilder::new();
    b.zero_sum_normal_with_shape(&[2, 4], DoubleTensor::vector(vec![1.0, 2.0, 3.0]));
    assert!(b.build().is_err());
}
