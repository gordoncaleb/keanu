//! The spatial distributions: ICAR, proper CAR and the BYM2 scaling factors, against dense computations.

use keanu_distributions::{
    Adjacency, Car, DistributionError, Icar, KeanuRng, MultivariateGaussian,
};
use keanu_tensor::DoubleTensor;

/// A deterministic vector with zero mean on each of the given groups.
fn centred(n: usize, groups: &[Vec<usize>], seed: u64) -> Vec<f64> {
    let mut rng = KeanuRng::seed_from_u64(seed);
    let mut x: Vec<f64> = (0..n).map(|_| rng.next_gaussian()).collect();
    for group in groups.iter().filter(|g| g.len() > 1) {
        let mean = group.iter().map(|&i| x[i]).sum::<f64>() / group.len() as f64;
        for &i in group {
            x[i] -= mean;
        }
    }
    x
}

/// xᵀ M x for a dense matrix M.
fn quadratic(m: &DoubleTensor, x: &[f64]) -> f64 {
    let n = x.len();
    let m = m.as_slice();
    (0..n)
        .map(|i| (0..n).map(|j| x[i] * m[i * n + j] * x[j]).sum::<f64>())
        .sum()
}

fn close(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() <= tolerance * b.abs().max(1.0)
}

/// The Moore–Penrose inverse of a connected graph's Laplacian, (Q + J/n)⁻¹ − J/n, with J the matrix of ones.
fn laplacian_pseudo_inverse(q: &DoubleTensor) -> DoubleTensor {
    let n = q.shape()[0];
    let j = DoubleTensor::full(&[n, n], 1.0 / n as f64);
    q.add(&j).unwrap().inverse().unwrap().sub(&j).unwrap()
}

#[test]
fn lattice_icar_density_is_minus_half_the_laplacian_quadratic_form_and_its_scaling_factor_matches()
{
    // AE1
    let a = Adjacency::lattice(10, 10).unwrap();
    assert_eq!(a.len(), 100);
    assert_eq!(a.edges().len(), 180);
    assert_eq!(a.components().len(), 1);
    assert_eq!(a.icar_rank(), 99);
    let q = a.laplacian().to_dense();
    let x = centred(100, a.components(), 1);
    let phi = DoubleTensor::vector(x.clone());
    let expected = -0.5 * quadratic(&q, &x);
    let log_p = Icar::new(&a, DoubleTensor::scalar(1.0))
        .log_prob(&phi)
        .unwrap();
    assert!(close(log_p, expected, 1e-12), "{log_p} != {expected}");
    // with a scale: the density of φ/σ and −(n − 1) ln σ
    let scaled = Icar::new(&a, DoubleTensor::scalar(2.0))
        .log_prob(&phi)
        .unwrap();
    assert!(close(scaled, expected / 4.0 - 99.0 * 2f64.ln(), 1e-12));

    // the BYM2 scaling factor: the geometric mean of the diagonal of Q's generalised inverse
    let inverse = laplacian_pseudo_inverse(&q);
    let diagonal: Vec<f64> = (0..100).map(|i| inverse.as_slice()[i * 100 + i]).collect();
    let geometric = (diagonal.iter().map(|v| v.ln()).sum::<f64>() / 100.0).exp();
    let factors = a.scaling_factors().unwrap();
    assert!(
        factors.iter().all(|&s| (s - geometric).abs() < 1e-10),
        "{} != {geometric}",
        factors[0]
    );
}

#[test]
fn several_components_and_islands() {
    // a triangle-free path 0 - 1 - 2, a pair 3 - 4 and an island 5
    let a = Adjacency::from_edges(6, &[(1, 0), (1, 2), (2, 1), (4, 3)]).unwrap();
    assert_eq!(a.edges(), &[(0, 1), (1, 2), (3, 4)]);
    assert_eq!(a.components(), &[vec![0, 1, 2], vec![3, 4], vec![5]]);
    assert_eq!(a.connected_groups().count(), 2);
    assert_eq!(a.icar_rank(), 4);
    let x = [0.5, -0.2, -0.3, 0.7, -0.7, 1.3];
    let sigma = 0.8;
    let icar = Icar::new(&a, DoubleTensor::scalar(sigma));
    let phi = DoubleTensor::vector(x.to_vec());
    let squares = 0.7f64.powi(2) + 0.1f64.powi(2) + 1.4f64.powi(2) + 1.3f64.powi(2);
    let expected = -0.5 * squares / (sigma * sigma) - 4.0 * sigma.ln();
    assert!(close(icar.log_prob(&phi).unwrap(), expected, 1e-12));
    // each component of two or more areas must sum to zero, but not the island
    let mut off = x;
    off[3] += 0.1;
    off[4] -= 0.05;
    assert_eq!(
        icar.log_prob(&DoubleTensor::vector(off.to_vec())).unwrap(),
        f64::NEG_INFINITY
    );
    let mut island = x;
    island[5] = -4.0;
    assert!(
        icar.log_prob(&DoubleTensor::vector(island.to_vec()))
            .unwrap()
            .is_finite()
    );

    // scaling factors: the pair's Laplacian [[1, -1], [-1, 1]] has generalised inverse Q/4, the island 1
    let factors = a.scaling_factors().unwrap();
    assert!((factors[3] - 0.25).abs() < 1e-12 && (factors[4] - 0.25).abs() < 1e-12);
    assert_eq!(factors[5], 1.0);
    let path = DoubleTensor::new(
        &[3, 3],
        vec![1.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 1.0],
    )
    .unwrap();
    let inverse = laplacian_pseudo_inverse(&path);
    let geometric = (0..3).map(|i| inverse.as_slice()[i * 4].ln()).sum::<f64>() / 3.0;
    assert!((factors[0] - geometric.exp()).abs() < 1e-12);
}

/// Central differences of a function of a vector.
fn numeric_gradient(f: impl Fn(&[f64]) -> f64, x: &[f64]) -> Vec<f64> {
    let h = 1e-6;
    (0..x.len())
        .map(|i| {
            let (mut up, mut down) = (x.to_vec(), x.to_vec());
            up[i] += h;
            down[i] -= h;
            (f(&up) - f(&down)) / (2.0 * h)
        })
        .collect()
}

#[test]
fn gradients_match_finite_differences() {
    let a = Adjacency::from_edges(6, &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5)]).unwrap();
    let x = centred(6, a.components(), 3);
    // the ICAR's density off its support is −∞, so its value gradient is checked against the quadratic form
    let icar_density = |x: &[f64], sigma: f64| {
        let mut s = 0.0;
        for &(i, j) in a.edges() {
            s += (x[i] - x[j]).powi(2);
        }
        -0.5 * s / (sigma * sigma) - a.icar_rank() as f64 * sigma.ln()
    };
    let sigma = 1.3;
    let g = Icar::new(&a, DoubleTensor::scalar(sigma))
        .d_log_prob(&DoubleTensor::vector(x.clone()))
        .unwrap();
    let numeric = numeric_gradient(|x| icar_density(x, sigma), &x);
    for (a, e) in g.x.as_slice().iter().zip(&numeric) {
        assert!(close(*a, *e, 1e-7), "{a} != {e}");
    }
    let d_sigma = numeric_gradient(|s| icar_density(&x, s[0]), &[sigma])[0];
    assert!(close(g.sigma.as_slice()[0], d_sigma, 1e-7));

    let car = |x: &[f64], tau: f64, alpha: f64| {
        Car::new(&a, DoubleTensor::scalar(tau), DoubleTensor::scalar(alpha))
            .log_prob(&DoubleTensor::vector(x.to_vec()))
            .unwrap()
    };
    let (tau, alpha) = (1.7, 0.6);
    let g = Car::new(&a, DoubleTensor::scalar(tau), DoubleTensor::scalar(alpha))
        .d_log_prob(&DoubleTensor::vector(x.clone()))
        .unwrap();
    for (a, e) in
        g.x.as_slice()
            .iter()
            .zip(numeric_gradient(|x| car(x, tau, alpha), &x))
    {
        assert!(close(*a, e, 1e-7), "{a} != {e}");
    }
    let d_tau = numeric_gradient(|t| car(&x, t[0], alpha), &[tau])[0];
    let d_alpha = numeric_gradient(|al| car(&x, tau, al[0]), &[alpha])[0];
    assert!(close(g.tau.as_slice()[0], d_tau, 1e-7));
    assert!(close(g.alpha.as_slice()[0], d_alpha, 1e-7));
}

/// A connected random graph of n nodes: a ring plus random chords.
fn random_graph(n: usize, chords: usize, seed: u64) -> Adjacency {
    let mut rng = KeanuRng::seed_from_u64(seed);
    let mut edges: Vec<(usize, usize)> = (0..n).map(|i| (i, (i + 1) % n)).collect();
    while edges.len() < n + chords {
        let i = rng.next_below(n as u64) as usize;
        let j = rng.next_below(n as u64) as usize;
        if i != j {
            edges.push((i, j));
        }
    }
    Adjacency::from_edges(n, &edges).unwrap()
}

#[test]
fn proper_car_density_is_the_dense_multivariate_normal_with_precision_tau_d_minus_alpha_w() {
    // AE2
    let a = random_graph(50, 40, 11);
    assert!(a.degrees().iter().all(|&d| d > 0.0));
    let w = a.neighbours().to_dense();
    let mut d = DoubleTensor::zeros(&[50, 50]).into_vec();
    for (i, &v) in a.degrees().iter().enumerate() {
        d[i * 51] = v;
    }
    let d = DoubleTensor::new(&[50, 50], d).unwrap();
    let mut rng = KeanuRng::seed_from_u64(2);
    for (tau, alpha) in [(1.0, 0.0), (2.5, 0.9), (0.4, -0.5), (3.0, 0.99)] {
        let precision = d.sub(&(w.clone() * alpha)).unwrap() * tau;
        let x: Vec<f64> = (0..50).map(|_| rng.next_gaussian()).collect();
        let (sign, log_det) = precision.slogdet().unwrap();
        assert_eq!(sign.as_slice(), &[1.0]);
        let expected = 0.5 * log_det.as_slice()[0]
            - 0.5 * quadratic(&precision, &x)
            - 25.0 * (2.0 * std::f64::consts::PI).ln();
        let car = Car::new(&a, DoubleTensor::scalar(tau), DoubleTensor::scalar(alpha));
        let log_p = car.log_prob(&DoubleTensor::vector(x.clone())).unwrap();
        assert!((log_p - expected).abs() < 1e-10, "{log_p} != {expected}");
        // and the library's dense multivariate normal with the inverse as its covariance
        let mvn =
            MultivariateGaussian::new(DoubleTensor::zeros(&[50]), precision.inverse().unwrap())
                .log_prob(&DoubleTensor::vector(x))
                .unwrap();
        assert!((log_p - mvn).abs() < 1e-8, "{log_p} != {mvn}");
    }
    // outside the range of a positive definite precision the density is NaN
    let lambda = a.car_eigenvalues().unwrap();
    assert!((lambda[lambda.len() - 1] - 1.0).abs() < 1e-12);
    for (tau, alpha) in [
        (1.0, 1.0),
        (1.0, 1.0 / lambda[0] - 0.01),
        (0.0, 0.5),
        (-1.0, 0.5),
    ] {
        let car = Car::new(&a, DoubleTensor::scalar(tau), DoubleTensor::scalar(alpha));
        assert!(
            car.log_prob(&DoubleTensor::zeros(&[50])).unwrap().is_nan(),
            "{tau} {alpha}"
        );
    }
}

/// The sample covariance of draws (rows), and the standard error of each entry for a Gaussian.
fn covariance(draws: &[Vec<f64>]) -> Vec<f64> {
    let (count, n) = (draws.len() as f64, draws[0].len());
    let mean: Vec<f64> = (0..n)
        .map(|i| draws.iter().map(|d| d[i]).sum::<f64>() / count)
        .collect();
    let mut c = vec![0.0; n * n];
    for d in draws {
        for i in 0..n {
            for j in 0..n {
                c[i * n + j] += (d[i] - mean[i]) * (d[j] - mean[j]) / (count - 1.0);
            }
        }
    }
    c
}

/// Checks a sample covariance against the truth within five standard errors, √((Σᵢᵢ Σⱼⱼ + Σᵢⱼ²) / N).
fn assert_covariance(what: &str, sample: &[f64], truth: &[f64], n: usize, count: usize) {
    for i in 0..n {
        for j in 0..n {
            let se = ((truth[i * n + i] * truth[j * n + j] + truth[i * n + j].powi(2))
                / count as f64)
                .sqrt();
            let (s, t) = (sample[i * n + j], truth[i * n + j]);
            assert!(
                (s - t).abs() < 5.0 * se,
                "{what} [{i}, {j}]: {s} vs {t} (se {se})"
            );
        }
    }
}

#[test]
fn icar_draws_sum_to_zero_per_component_with_covariance_sigma_squared_times_the_generalised_inverse()
 {
    let a = Adjacency::from_edges(7, &[(0, 1), (1, 2), (2, 3), (0, 2), (4, 5)]).unwrap();
    let sigma = 1.5;
    let icar = Icar::new(&a, DoubleTensor::scalar(sigma));
    let mut rng = KeanuRng::seed_from_u64(5);
    let count = 20_000;
    let draws: Vec<Vec<f64>> = (0..count)
        .map(|_| icar.sample(&[7], &mut rng).unwrap().into_vec())
        .collect();
    for d in &draws {
        assert!((d[0] + d[1] + d[2] + d[3]).abs() < 1e-12);
        assert!((d[4] + d[5]).abs() < 1e-12);
        assert!(
            icar.log_prob(&DoubleTensor::vector(d.clone()))
                .unwrap()
                .is_finite()
        );
    }
    // the truth: σ² Q⁺ on each component, σ² on the island, 0 between components
    let q = a.laplacian().to_dense();
    let mut truth = vec![0.0; 49];
    for group in a.components() {
        let m = group.len();
        let block: Vec<f64> = group
            .iter()
            .flat_map(|&i| group.iter().map(move |&j| (i, j)))
            .map(|(i, j)| q.as_slice()[i * 7 + j])
            .collect();
        let inverse = if m == 1 {
            DoubleTensor::new(&[1, 1], vec![1.0]).unwrap()
        } else {
            laplacian_pseudo_inverse(&DoubleTensor::new(&[m, m], block).unwrap())
        };
        for (r, &i) in group.iter().enumerate() {
            for (c, &j) in group.iter().enumerate() {
                truth[i * 7 + j] = sigma * sigma * inverse.as_slice()[r * m + c];
            }
        }
    }
    assert_covariance("ICAR", &covariance(&draws), &truth, 7, count);
}

#[test]
fn car_draws_have_the_inverse_precision_as_covariance() {
    let a = Adjacency::from_edges(5, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)]).unwrap();
    let (tau, alpha) = (2.0, 0.8);
    let car = Car::new(&a, DoubleTensor::scalar(tau), DoubleTensor::scalar(alpha));
    let mut rng = KeanuRng::seed_from_u64(6);
    let count = 20_000;
    let draws: Vec<Vec<f64>> = (0..count)
        .map(|_| car.sample(&[5], &mut rng).unwrap().into_vec())
        .collect();
    let w = a.neighbours().to_dense();
    let mut precision = (w * (-alpha * tau)).into_vec();
    for (i, &d) in a.degrees().iter().enumerate() {
        precision[i * 6] = tau * d;
    }
    let truth = DoubleTensor::new(&[5, 5], precision)
        .unwrap()
        .inverse()
        .unwrap();
    assert_covariance("CAR", &covariance(&draws), truth.as_slice(), 5, count);
}

#[test]
fn invalid_neighbourhoods_values_and_parameters_are_errors() {
    let invalid = |r: Result<Adjacency, DistributionError>| {
        matches!(r, Err(DistributionError::InvalidParameter(_)))
    };
    assert!(invalid(Adjacency::from_edges(3, &[(0, 3)])));
    assert!(invalid(Adjacency::from_edges(3, &[(1, 1)])));
    let matrix = |values: Vec<f64>| DoubleTensor::new(&[2, 2], values).unwrap();
    assert!(invalid(Adjacency::from_matrix(&matrix(vec![
        0.0, 1.0, 0.0, 0.0
    ]))));
    assert!(invalid(Adjacency::from_matrix(&matrix(vec![
        0.0, 0.5, 0.5, 0.0
    ]))));
    assert!(invalid(Adjacency::from_matrix(&matrix(vec![
        1.0, 0.0, 0.0, 0.0
    ]))));
    assert!(invalid(Adjacency::from_matrix(&DoubleTensor::zeros(&[
        2, 3
    ]))));
    assert!(invalid(Adjacency::from_matrix(&DoubleTensor::zeros(&[4]))));
    let from_matrix = Adjacency::from_matrix(&matrix(vec![0.0, 1.0, 1.0, 0.0])).unwrap();
    assert_eq!(from_matrix, Adjacency::from_edges(2, &[(0, 1)]).unwrap());
    assert!(Adjacency::from_edges(0, &[]).unwrap().is_empty());

    let a = Adjacency::from_edges(3, &[(0, 1)]).unwrap();
    // a proper CAR needs every area to have a neighbour
    assert!(a.car_eigenvalues().is_err());
    let car = Car::new(&a, DoubleTensor::scalar(1.0), DoubleTensor::scalar(0.5));
    assert!(car.log_prob(&DoubleTensor::zeros(&[3])).is_err());
    // shapes and scalars
    let icar = Icar::new(&a, DoubleTensor::scalar(1.0));
    assert!(icar.log_prob(&DoubleTensor::zeros(&[4])).is_err());
    assert!(
        icar.sample(&[2, 3], &mut KeanuRng::seed_from_u64(1))
            .is_err()
    );
    assert!(
        Icar::new(&a, DoubleTensor::vector(vec![1.0, 2.0]))
            .log_prob(&DoubleTensor::zeros(&[3]))
            .is_err()
    );
    // invalid scales: NaN densities, and errors when sampling
    for sigma in [0.0, -1.0, f64::NAN] {
        let icar = Icar::new(&a, DoubleTensor::scalar(sigma));
        assert!(icar.log_prob(&DoubleTensor::zeros(&[3])).unwrap().is_nan());
        assert!(icar.sample(&[3], &mut KeanuRng::seed_from_u64(1)).is_err());
    }
    let icar = Icar::new(&a, DoubleTensor::scalar(1.0));
    assert_eq!(
        icar.log_prob(&DoubleTensor::vector(vec![f64::INFINITY, 0.0, 0.0]))
            .unwrap(),
        f64::NEG_INFINITY
    );
    let ring = Adjacency::from_edges(3, &[(0, 1), (1, 2), (2, 0)]).unwrap();
    for (tau, alpha) in [(0.0, 0.5), (1.0, 1.0), (1.0, f64::NAN)] {
        let car = Car::new(
            &ring,
            DoubleTensor::scalar(tau),
            DoubleTensor::scalar(alpha),
        );
        assert!(car.sample(&[3], &mut KeanuRng::seed_from_u64(1)).is_err());
    }
}
