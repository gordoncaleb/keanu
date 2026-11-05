//! General linear solves, sign and log-determinants, and symmetric eigendecompositions.

use keanu_tensor::{DoubleTensor, TensorError};

/// Deterministic uniform values in [-1, 1).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    fn tensor(&mut self, shape: &[usize]) -> DoubleTensor {
        let len = shape.iter().product();
        DoubleTensor::new(shape, (0..len).map(|_| self.next()).collect()).unwrap()
    }
}

/// Well-conditioned (diagonally dominant) random matrices, not symmetric.
fn well_conditioned(rng: &mut Lcg, batch: &[usize], n: usize) -> DoubleTensor {
    let shape: Vec<usize> = batch.iter().copied().chain([n, n]).collect();
    rng.tensor(&shape)
        .add(&(DoubleTensor::eye(n).unwrap() * (n as f64 + 1.0)))
        .unwrap()
}

/// Random symmetric matrices.
fn symmetric(rng: &mut Lcg, batch: &[usize], n: usize) -> DoubleTensor {
    let shape: Vec<usize> = batch.iter().copied().chain([n, n]).collect();
    let a = rng.tensor(&shape);
    a.add(&a.transpose().unwrap()).unwrap()
}

fn assert_close(actual: &DoubleTensor, expected: &DoubleTensor, tolerance: f64) {
    assert_eq!(actual.shape(), expected.shape());
    for (i, (&a, &e)) in actual
        .as_slice()
        .iter()
        .zip(expected.as_slice())
        .enumerate()
    {
        assert!(
            (a - e).abs() <= tolerance * e.abs().max(1.0),
            "element {i}: {a} != {e}"
        );
    }
}

#[test]
fn solve_satisfies_the_system_batched() {
    let mut rng = Lcg(7);
    for n in [1, 2, 5, 12] {
        let a = well_conditioned(&mut rng, &[2, 3], n);
        let b = rng.tensor(&[2, 3, n, 4]);
        let x = a.solve(&b).unwrap();
        assert_eq!(x.shape(), &[2, 3, n, 4]);
        assert_close(&a.matmul(&x).unwrap(), &b, 1e-12);
    }
}

#[test]
fn solve_broadcasts_leading_dimensions_and_takes_vectors() {
    let mut rng = Lcg(11);
    let a = well_conditioned(&mut rng, &[3], 4);
    // one right-hand side matrix shared by every matrix in the batch
    let b = rng.tensor(&[4, 2]);
    let x = a.solve(&b).unwrap();
    assert_eq!(x.shape(), &[3, 4, 2]);
    assert_close(
        &a.matmul(&x).unwrap(),
        &b.broadcast_to(&[3, 4, 2]).unwrap(),
        1e-12,
    );
    // a batch of right-hand sides against one matrix
    let single = well_conditioned(&mut rng, &[], 4);
    let many = rng.tensor(&[5, 4, 1]);
    let x = single.solve(&many).unwrap();
    assert_eq!(x.shape(), &[5, 4, 1]);
    assert_close(&single.matmul(&x).unwrap(), &many, 1e-12);
    // a vector right-hand side
    let v = rng.tensor(&[4]);
    let x = a.solve(&v).unwrap();
    assert_eq!(x.shape(), &[3, 4]);
    let column = x.reshape(&[3, 4, 1]).unwrap();
    assert_close(
        &a.matmul(&column).unwrap().reshape(&[3, 4]).unwrap(),
        &v.broadcast_to(&[3, 4]).unwrap(),
        1e-12,
    );
}

#[test]
fn solve_needs_pivoting_and_rejects_singular_and_mismatched_systems() {
    // a zero leading element needs a row swap
    let a = DoubleTensor::new(&[2, 2], vec![0.0, 1.0, 2.0, 3.0]).unwrap();
    let b = DoubleTensor::new(&[2], vec![1.0, 8.0]).unwrap();
    assert_eq!(a.solve(&b).unwrap().as_slice(), &[2.5, 1.0]);

    let singular = DoubleTensor::new(&[2, 2], vec![1.0, 2.0, 2.0, 4.0]).unwrap();
    assert_eq!(singular.solve(&b), Err(TensorError::SingularMatrix));
    let zeros = DoubleTensor::zeros(&[3, 3]);
    assert_eq!(
        zeros.solve(&DoubleTensor::zeros(&[3, 1])),
        Err(TensorError::SingularMatrix)
    );
    assert!(matches!(
        a.solve(&DoubleTensor::zeros(&[3])),
        Err(TensorError::IncompatibleShapes { .. })
    ));
    assert!(matches!(
        DoubleTensor::zeros(&[2, 3]).solve(&b),
        Err(TensorError::InvalidArgument(_))
    ));
    // empty systems
    let empty = DoubleTensor::zeros(&[0, 0]);
    assert_eq!(
        empty.solve(&DoubleTensor::zeros(&[0, 2])).unwrap().shape(),
        &[0, 2]
    );
}

#[test]
fn slogdet_matches_the_determinant_for_both_signs() {
    let mut rng = Lcg(3);
    let a = well_conditioned(&mut rng, &[4], 5);
    // flipping one row of the first two matrices flips their determinants' signs
    let mut data = a.as_slice().to_vec();
    for m in 0..2 {
        for j in 0..5 {
            data[m * 25 + j] = -data[m * 25 + j];
        }
    }
    let a = DoubleTensor::new(&[4, 5, 5], data).unwrap();
    let det = a.determinant().unwrap();
    let (sign, log_abs) = a.slogdet().unwrap();
    assert_eq!(sign.shape(), &[4]);
    assert_eq!(log_abs.shape(), &[4]);
    for i in 0..4 {
        let d = det.as_slice()[i];
        assert_eq!(sign.as_slice()[i], d.signum());
        assert!((log_abs.as_slice()[i] - d.abs().ln()).abs() < 1e-12);
    }
    assert_eq!(&sign.as_slice()[..2], &[-1.0, -1.0]);

    // a row swap: det [[0, 1], [1, 0]] = -1
    let swap = DoubleTensor::new(&[2, 2], vec![0.0, 1.0, 1.0, 0.0]).unwrap();
    let (sign, log_abs) = swap.slogdet().unwrap();
    assert_eq!(
        (sign.as_slice(), log_abs.as_slice()),
        (&[-1.0][..], &[0.0][..])
    );
}

#[test]
fn slogdet_does_not_overflow_and_reports_singular_matrices() {
    // det = 1e400 overflows f64, its log does not
    let big = DoubleTensor::eye(4).unwrap() * 1e100;
    let (sign, log_abs) = big.slogdet().unwrap();
    assert_eq!(sign.as_slice(), &[1.0]);
    assert!((log_abs.as_slice()[0] - 400.0 * 10f64.ln()).abs() < 1e-9);
    // singular: sign 0, log|det| -inf, as numpy
    let singular = DoubleTensor::new(&[2, 2], vec![1.0, 2.0, 2.0, 4.0]).unwrap();
    let (sign, log_abs) = singular.slogdet().unwrap();
    assert_eq!(sign.as_slice(), &[0.0]);
    assert_eq!(log_abs.as_slice(), &[f64::NEG_INFINITY]);
    // 0 x 0: the empty product
    let (sign, log_abs) = DoubleTensor::zeros(&[3, 0, 0]).slogdet().unwrap();
    assert_eq!(sign.as_slice(), &[1.0; 3]);
    assert_eq!(log_abs.as_slice(), &[0.0; 3]);
    assert!(DoubleTensor::zeros(&[2, 3]).slogdet().is_err());
}

#[test]
fn eigh_reconstructs_the_matrix_with_ascending_eigenvalues() {
    let mut rng = Lcg(5);
    for n in [1, 2, 3, 6, 20] {
        let a = symmetric(&mut rng, &[2, 2], n);
        let (values, vectors) = a.eigh().unwrap();
        assert_eq!(values.shape(), &[2, 2, n]);
        assert_eq!(vectors.shape(), &[2, 2, n, n]);
        for lane in values.as_slice().chunks(n) {
            assert!(
                lane.windows(2).all(|w| w[0] <= w[1]),
                "{lane:?} not ascending"
            );
        }
        // A = Q Λ Qᵀ and Qᵀ Q = I
        let scaled = vectors
            .mul(&values.reshape(&[2, 2, 1, n as isize]).unwrap())
            .unwrap();
        let rebuilt = scaled.matmul_transposed(&vectors, false, true).unwrap();
        assert_close(&rebuilt, &a, 1e-10);
        let gram = vectors.matmul_transposed(&vectors, true, false).unwrap();
        assert_close(
            &gram,
            &DoubleTensor::eye(n)
                .unwrap()
                .broadcast_to(&[2, 2, n, n])
                .unwrap(),
            1e-10,
        );
        // each eigenvector's largest component is positive, so the signs are deterministic
        let q = vectors.as_slice();
        for m in 0..4 {
            for col in 0..n {
                let column: Vec<f64> = (0..n).map(|row| q[m * n * n + row * n + col]).collect();
                let largest = column.iter().copied().fold(0.0f64, |best, x| {
                    if x.abs() > best.abs() { x } else { best }
                });
                assert!(largest > 0.0);
            }
        }
    }
}

#[test]
fn eigh_reads_only_the_lower_triangle() {
    let a = DoubleTensor::new(&[2, 2], vec![2.0, 1.0, 1.0, 2.0]).unwrap();
    let garbage = DoubleTensor::new(&[2, 2], vec![2.0, 99.0, 1.0, 2.0]).unwrap();
    let (values, vectors) = garbage.eigh().unwrap();
    let (expected_values, expected_vectors) = a.eigh().unwrap();
    assert_close(&values, &expected_values, 1e-14);
    assert_close(&vectors, &expected_vectors, 1e-14);
    assert_close(
        &values,
        &DoubleTensor::new(&[2], vec![1.0, 3.0]).unwrap(),
        1e-14,
    );
}

#[test]
fn eigh_edge_cases() {
    // a diagonal matrix: sorted eigenvalues, permuted unit vectors
    let d = DoubleTensor::new(&[3, 3], vec![3.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 2.0]).unwrap();
    let (values, vectors) = d.eigh().unwrap();
    assert_close(
        &values,
        &DoubleTensor::new(&[3], vec![-1.0, 2.0, 3.0]).unwrap(),
        1e-14,
    );
    assert_close(
        &vectors,
        &DoubleTensor::new(&[3, 3], vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap(),
        1e-14,
    );
    // empty, non-finite and non-square inputs
    let (values, vectors) = DoubleTensor::zeros(&[2, 0, 0]).eigh().unwrap();
    assert_eq!(
        (values.shape(), vectors.shape()),
        (&[2, 0][..], &[2, 0, 0][..])
    );
    let nan = DoubleTensor::new(&[2, 2], vec![1.0, 0.0, f64::NAN, 1.0]).unwrap();
    let (values, vectors) = nan.eigh().unwrap();
    assert!(
        values
            .as_slice()
            .iter()
            .chain(vectors.as_slice())
            .all(|v| v.is_nan())
    );
    assert!(DoubleTensor::zeros(&[2, 3]).eigh().is_err());
    assert!(DoubleTensor::zeros(&[3]).eigh().is_err());
}
