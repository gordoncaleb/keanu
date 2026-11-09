use keanu_distributions::KeanuRng;
use keanu_graph::GraphBuilder;
use keanu_graph::gp::{InducingGp, Kernel, PeriodicHsgp};
use keanu_tensor::DoubleTensor;

fn matrix(shape: &[usize], values: &[f64]) -> DoubleTensor {
    DoubleTensor::new(shape, values.to_vec()).unwrap()
}

fn close(actual: &DoubleTensor, expected: &DoubleTensor, tolerance: f64) {
    assert_eq!(actual.shape(), expected.shape());
    for (&a, &b) in actual.as_slice().iter().zip(expected.as_slice()) {
        assert!((a - b).abs() < tolerance, "{a} vs {b}");
    }
}

#[test]
fn structured_gp_covariances_match_dense_kernel_calculations() {
    let b = GraphBuilder::new();
    let x = b.constant(DoubleTensor::vector(vec![-0.4, 0.1, 0.42, 1.3]));
    let z = DoubleTensor::vector(vec![0.0, 0.3, 0.6]);
    let kernel = Kernel::periodic(&b, 1.4, 0.8, 1.0);
    let dense = kernel.covariance(x, x);
    let fourier = PeriodicHsgp::new(&kernel, 12).unwrap().basis(x);
    let approximate = fourier.matmul_transposed(fourier, false, true);
    let inducing = InducingGp::new(&kernel, z.clone()).unwrap().basis(x);
    let reduced = inducing.matmul_transposed(inducing, false, true);
    let cross = kernel.covariance(x, z.clone());
    let gram = kernel.covariance(z.clone(), z)
        + b.constant(DoubleTensor::eye(3).unwrap() * keanu_graph::gp::DEFAULT_JITTER);
    let projection = cross.matmul(gram.solve(cross.transpose()));
    let ids = [dense.id(), approximate.id(), reduced.id(), projection.id()];
    let (graph, mut state) = b.build().unwrap();
    state
        .lazy_eval(&graph, &ids, &mut KeanuRng::seed_from_u64(2))
        .unwrap();
    let [dense, approximate, reduced, projection] = ids.map(|id| state.f64(id).unwrap());
    close(approximate, dense, 1e-10);
    close(reduced, projection, 1e-12);
}

#[test]
fn kronecker_solve_agrees_with_an_explicit_dense_system() {
    let a = matrix(&[2, 2], &[2.0, 0.4, -0.5, 1.3]);
    let b = matrix(&[3, 3], &[1.4, 0.3, 0.1, 0.7, 1.9, -0.6, 0.0, 0.2, 2.0]);
    let rhs = matrix(&[2, 3], &[1.0, -2.0, 3.0, -1.0, 1.5, 0.5]);
    let dense = DoubleTensor::new(
        &[6, 6],
        (0..36)
            .map(|p| {
                let (row, col) = (p / 6, p % 6);
                a.as_slice()[(row / 3) * 2 + col / 3] * b.as_slice()[(row % 3) * 3 + col % 3]
            })
            .collect(),
    )
    .unwrap();
    let expected = dense
        .solve(&rhs.reshape(&[6]).unwrap())
        .unwrap()
        .reshape(&[2, 3])
        .unwrap();
    let builder = GraphBuilder::new();
    let result = builder.constant(a).solve_kronecker(b, rhs).id();
    let (graph, mut state) = builder.build().unwrap();
    state
        .lazy_eval(&graph, &[result], &mut KeanuRng::seed_from_u64(3))
        .unwrap();
    close(state.f64(result).unwrap(), &expected, 1e-12);
}

#[test]
fn multivariate_local_level_matches_dense_gaussian_likelihood() {
    use keanu_graph::node_log_prob;

    let b = GraphBuilder::new();
    let mean = DoubleTensor::vector(vec![0.2, -0.4]);
    let initial = matrix(&[2, 2], &[0.7, 0.1, 0.1, 0.8]);
    let level = matrix(&[2, 2], &[0.3, -0.15, -0.15, 0.5]);
    let noise = matrix(&[2, 2], &[0.1, 0.02, 0.02, 0.2]);
    let data = matrix(&[3, 2], &[0.3, 0.1, 0.7, -0.2, 0.4, -0.8]);
    let series = b.multivariate_local_level(
        3,
        mean.clone(),
        initial.clone(),
        level.clone(),
        noise.clone(),
    );
    b.observe(series, data.clone());
    let dense = DoubleTensor::new(
        &[6, 6],
        (0..36)
            .map(|p| {
                let (row, col) = (p / 6, p % 6);
                let coordinate = 2 * (row % 2) + col % 2;
                initial.as_slice()[coordinate]
                    + (row / 2).min(col / 2) as f64 * level.as_slice()[coordinate]
                    + if row / 2 == col / 2 {
                        noise.as_slice()[coordinate]
                    } else {
                        0.0
                    }
            })
            .collect(),
    )
    .unwrap();
    let expected = keanu_distributions::MultivariateGaussian::new(
        mean.broadcast_to(&[3, 2]).unwrap().reshape(&[6]).unwrap(),
        &dense,
    )
    .log_prob(&data.reshape(&[6]).unwrap())
    .unwrap();
    let id = series.id();
    let (graph, mut state) = b.build().unwrap();
    state
        .lazy_eval(&graph, &[id], &mut KeanuRng::seed_from_u64(4))
        .unwrap();
    assert!((node_log_prob(&graph, &state, id).unwrap() - expected).abs() < 1e-12);
}

#[test]
fn periodic_basis_keeps_scaled_bessel_weights_for_short_length_scales() {
    let b = GraphBuilder::new();
    let length = 1e-12_f64;
    let kernel = Kernel::periodic(&b, 1.0, length, 1.0);
    let basis = PeriodicHsgp::new(&kernel, 5)
        .unwrap()
        .basis(DoubleTensor::vector(vec![0.0]));
    let variance = basis.matmul_transposed(basis, false, true).id();
    let (graph, mut state) = b.build().unwrap();
    state
        .lazy_eval(&graph, &[variance], &mut KeanuRng::seed_from_u64(5))
        .unwrap();
    let expected = (0..=5)
        .map(|j| {
            let multiplier = if j == 0 { 1.0 } else { 2.0 };
            multiplier * keanu_tensor::special::bessel_i_scaled(j as f64, 1.0 / (length * length))
        })
        .sum::<f64>();
    let actual = state.f64(variance).unwrap().as_slice()[0];
    assert!(
        (actual / expected - 1.0).abs() < 1e-12,
        "{actual} vs {expected}"
    );
}
