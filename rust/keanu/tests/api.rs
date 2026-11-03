//! API contract checks: thread safety and common traits of the public types, error source chaining, and the
//! setter style of the settings structs.

use keanu::prelude::*;
use std::error::Error as _;

fn send_sync<T: Send + Sync>() {}
fn debug_clone<T: std::fmt::Debug + Clone>() {}

#[test]
fn public_types_are_thread_safe() {
    send_sync::<DoubleTensor>();
    send_sync::<IntTensor>();
    send_sync::<BoolTensor>();
    send_sync::<Graph>();
    send_sync::<State>();
    send_sync::<Model>();
    send_sync::<Value>();
    send_sync::<KeanuRng>();
    send_sync::<NetworkSamples>();
    send_sync::<Nuts>();
    send_sync::<MetropolisHastings>();
    send_sync::<Adam>();
    send_sync::<ConjugateGradient>();
    send_sync::<Bobyqa>();
    send_sync::<OptimizedResult>();
    send_sync::<keanu::distributions::Gaussian>();
    send_sync::<keanu::Error>();
}

#[test]
fn settings_and_values_are_debug_and_clone() {
    debug_clone::<DoubleTensor>();
    debug_clone::<Value>();
    debug_clone::<State>();
    debug_clone::<KeanuRng>();
    debug_clone::<Nuts>();
    debug_clone::<MetropolisHastings>();
    debug_clone::<Adam>();
    debug_clone::<ConjugateGradient>();
    debug_clone::<Bobyqa>();
    debug_clone::<SamplingOptions>();
    debug_clone::<keanu::distributions::Gaussian>();
}

#[test]
fn settings_use_with_setters() {
    let nuts = Nuts::default()
        .with_adapt_count(10)
        .with_initial_step_size(0.5)
        .with_max_tree_height(3);
    assert_eq!(nuts.adapt_count, 10);
    assert_eq!(nuts.initial_step_size, Some(0.5));
    assert_eq!(nuts.max_tree_height, 3);
    let options = SamplingOptions::default()
        .with_drop(5)
        .with_down_sample_interval(2);
    assert_eq!((options.drop, options.down_sample_interval), (5, 2));
}

#[test]
fn wrapped_errors_are_transparent() {
    // a tensor error raised while building a graph surfaces through GraphError and keanu::Error
    let b = GraphBuilder::new();
    let a = b.constant(DoubleTensor::zeros(&[2, 3]));
    let c = b.constant(DoubleTensor::zeros(&[4]));
    let _ = a + c;
    let error: keanu::Error = b.build().unwrap_err().into();
    let message = error.to_string();
    assert!(
        message.contains("[2, 3]") && message.contains("[4]"),
        "{message}"
    );
    // transparent: the wrapper does not report the same error again as its source
    if let Some(source) = error.source() {
        assert_ne!(source.to_string(), message);
    }
}

#[test]
fn invalid_user_input_is_an_error_not_a_panic() {
    let t = DoubleTensor::new(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(DoubleTensor::new(&[3], vec![1.0]).is_err());
    assert!(t.reshape(&[3, -1]).is_err());
    assert!(t.permute(&[0, 0]).is_err());
    assert!(t.slice("0:1:0").is_err());
    assert!(t.slice("5").is_err());
    assert!(t.sum_axes(&[2]).is_err());
    assert!(t.matmul(&DoubleTensor::zeros(&[3, 1])).is_err());
    assert!(
        DoubleTensor::new(&[2, 2], vec![1.0, 2.0, 2.0, 4.0])
            .unwrap()
            .inverse()
            .is_err()
    );
    assert!(
        DoubleTensor::new(&[2, 2], vec![1.0, 2.0, 2.0, 1.0])
            .unwrap()
            .cholesky()
            .is_err()
    );
    assert!(
        IntTensor::vector(vec![1])
            .div(&IntTensor::vector(vec![0]))
            .is_err()
    );

    let b = GraphBuilder::new();
    let x = b.gaussian(0.0, 1.0).label("x");
    let _ = b.gaussian(0.0, 1.0).label("x");
    let _ = x;
    assert!(b.build().is_err(), "duplicate labels are rejected");

    // a model needs at least one latent or observed node (as Java's KeanuProbabilisticModel)
    let b = GraphBuilder::new();
    let _ = b.constant(DoubleTensor::scalar(1.0)) * 2.0;
    let (graph, state) = b.build().unwrap();
    assert!(Model::new(graph, &state).is_err());
}
