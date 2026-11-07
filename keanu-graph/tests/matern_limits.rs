//! The general Matérn kernel's limits: the covariance at zero distance is exactly σ² for any smoothness (also
//! below ½, where the correlation falls steeply away from zero distance), and the correlation never exceeds 1 (also
//! for very large smoothness).

use keanu_distributions::KeanuRng;
use keanu_graph::GraphBuilder;
use keanu_graph::gp::Kernel;
use keanu_tensor::DoubleTensor;

mod common;
use common::t;

#[test]
fn the_matern_covariance_at_zero_distance_is_the_squared_amplitude() {
    let x = t(&[3, 1], &[0.0, 0.0, 1.0]);
    let amplitude = 1.5;
    for nu in [0.01, 0.05, 0.2, 0.5, 0.7, 1.5, 3.0, 40.0] {
        let b = GraphBuilder::new();
        let k = Kernel::matern(&b, nu, amplitude, 0.8).covariance(x.clone(), x.clone());
        let k = k.id();
        let (graph, mut state) = b.build().unwrap();
        state
            .lazy_eval(&graph, &[k], &mut KeanuRng::seed_from_u64(1))
            .unwrap();
        let k = state.f64(k).unwrap();
        let k = k.as_slice();
        // [0, 0], [1, 1], [2, 2] and the repeated input [0, 1], [1, 0]
        for (i, j) in [(0, 0), (1, 1), (2, 2), (0, 1), (1, 0)] {
            assert_eq!(k[i * 3 + j], amplitude * amplitude, "ν = {nu}: [{i}, {j}]");
        }
        assert!(k[2] < amplitude * amplitude, "ν = {nu}");
    }
}

#[test]
fn the_matern_correlation_never_exceeds_one() {
    let b = GraphBuilder::new();
    let x = b.constant(DoubleTensor::vector(
        (0..200).map(|i| 1e-3 * f64::from(i)).collect(),
    ));
    let ids: Vec<_> = [1e3, 1e4, 3e4, 1e5]
        .iter()
        .map(|&nu| (nu, x.matern_correlation(nu).id()))
        .collect();
    let (graph, mut state) = b.build().unwrap();
    for (nu, id) in ids {
        state
            .lazy_eval(&graph, &[id], &mut KeanuRng::seed_from_u64(1))
            .unwrap();
        let values = state.f64(id).unwrap();
        assert!(
            values.as_slice().iter().all(|&f| f <= 1.0 && f > 0.0),
            "ν = {nu}: max {}",
            values.as_slice().iter().copied().fold(0.0, f64::max)
        );
    }
}
