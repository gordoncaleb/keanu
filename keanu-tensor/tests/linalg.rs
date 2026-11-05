//! General linear solves, sign and log-determinants, symmetric eigendecompositions and the matrix exponential.

use keanu_tensor::{DoubleTensor, TensorError};

mod common;
use common::assert_all_close_rel;

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

/// [`assert_all_close_rel`] on the elements, after checking the shapes agree.
fn assert_tensor_close_rel(actual: &DoubleTensor, expected: &DoubleTensor, tolerance: f64) {
    assert_eq!(actual.shape(), expected.shape());
    assert_all_close_rel(actual.as_slice(), expected.as_slice(), tolerance, "element");
}

#[test]
fn solve_satisfies_the_system_batched() {
    let mut rng = Lcg(7);
    for n in [1, 2, 5, 12] {
        let a = well_conditioned(&mut rng, &[2, 3], n);
        let b = rng.tensor(&[2, 3, n, 4]);
        let x = a.solve(&b).unwrap();
        assert_eq!(x.shape(), &[2, 3, n, 4]);
        assert_tensor_close_rel(&a.matmul(&x).unwrap(), &b, 1e-12);
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
    assert_tensor_close_rel(
        &a.matmul(&x).unwrap(),
        &b.broadcast_to(&[3, 4, 2]).unwrap(),
        1e-12,
    );
    // a batch of right-hand sides against one matrix
    let single = well_conditioned(&mut rng, &[], 4);
    let many = rng.tensor(&[5, 4, 1]);
    let x = single.solve(&many).unwrap();
    assert_eq!(x.shape(), &[5, 4, 1]);
    assert_tensor_close_rel(&single.matmul(&x).unwrap(), &many, 1e-12);
    // a vector right-hand side
    let v = rng.tensor(&[4]);
    let x = a.solve(&v).unwrap();
    assert_eq!(x.shape(), &[3, 4]);
    let column = x.reshape(&[3, 4, 1]).unwrap();
    assert_tensor_close_rel(
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
        assert_tensor_close_rel(&rebuilt, &a, 1e-10);
        let gram = vectors.matmul_transposed(&vectors, true, false).unwrap();
        assert_tensor_close_rel(
            &gram,
            &DoubleTensor::eye(n)
                .unwrap()
                .broadcast_to(&[2, 2, n, n])
                .unwrap(),
            1e-10,
        );
        // each eigenvector's first component within a relative 1e-10 of the largest magnitude is positive, so the
        // signs are deterministic
        let q = vectors.as_slice();
        for m in 0..4 {
            for col in 0..n {
                let column: Vec<f64> = (0..n).map(|row| q[m * n * n + row * n + col]).collect();
                let largest = column.iter().fold(0.0f64, |best, x| best.max(x.abs()));
                let leading = column
                    .iter()
                    .find(|x| x.abs() >= largest * (1.0 - 1e-10))
                    .unwrap();
                assert!(*leading > 0.0);
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
    assert_tensor_close_rel(&values, &expected_values, 1e-14);
    assert_tensor_close_rel(&vectors, &expected_vectors, 1e-14);
    assert_tensor_close_rel(
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
    assert_tensor_close_rel(
        &values,
        &DoubleTensor::new(&[3], vec![-1.0, 2.0, 3.0]).unwrap(),
        1e-14,
    );
    assert_tensor_close_rel(
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

#[test]
fn eigh_signs_are_stable_when_components_tie_in_magnitude() {
    // the eigenvectors of [[2, 1], [1, 2]] are (1, -1)/√2 and (1, 1)/√2: both components of each have equal
    // magnitude, so perturbing the matrix by rounding-sized amounts changes which one is (barely) largest. The first
    // component within a relative 1e-10 of the largest is made positive, so the signs don't depend on the rounding.
    let vectors = |e: f64| {
        DoubleTensor::new(&[2, 2], vec![2.0 + e, 1.0, 1.0, 2.0 - e])
            .unwrap()
            .eigh()
            .unwrap()
            .1
    };
    let exact = vectors(0.0);
    let s = std::f64::consts::FRAC_1_SQRT_2;
    // columns: (1, -1)/√2 for λ = 1 and (1, 1)/√2 for λ = 3, each with its first component positive
    assert_tensor_close_rel(
        &exact,
        &DoubleTensor::new(&[2, 2], vec![s, s, -s, s]).unwrap(),
        1e-12,
    );
    for e in [1e-14, -1e-14, 3e-15, -3e-15] {
        assert_tensor_close_rel(&vectors(e), &exact, 1e-10);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// the matrix exponential

fn matrix(n: usize, data: &[f64]) -> DoubleTensor {
    DoubleTensor::new(&[n, n], data.to_vec()).unwrap()
}

/// exp(A) by its Taylor series, summed until the terms vanish (accurate for the moderate norms used here).
fn taylor_exp(a: &DoubleTensor) -> DoubleTensor {
    let n = a.shape()[0];
    let mut sum = DoubleTensor::eye(n).unwrap();
    let mut term = DoubleTensor::eye(n).unwrap();
    for k in 1..200 {
        term = term.matmul(a).unwrap() * (1.0 / k as f64);
        sum = sum.add(&term).unwrap();
    }
    sum
}

#[test]
fn matrix_exp_of_diagonal_and_nilpotent_matrices_has_closed_forms() {
    let d = [1.0, -2.0, 0.5, 3.0];
    let diagonal = DoubleTensor::vector(d.to_vec()).diag().unwrap();
    let expected = DoubleTensor::vector(d.iter().map(|x| x.exp()).collect())
        .diag()
        .unwrap();
    assert_tensor_close_rel(&diagonal.matrix_exp().unwrap(), &expected, 1e-13);

    // N strictly upper triangular, N³ = 0: exp(N) = I + N + N² / 2
    let nilpotent = matrix(3, &[0.0, 2.0, -3.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0]);
    let expected = matrix(3, &[1.0, 2.0, 1.0, 0.0, 1.0, 4.0, 0.0, 0.0, 1.0]);
    assert_tensor_close_rel(&nilpotent.matrix_exp().unwrap(), &expected, 1e-13);
}

#[test]
fn matrix_exp_of_a_rotation_generator_is_a_rotation_at_every_degree_and_scaling() {
    // norms from below θ₃ to well above θ₁₃, so every Padé degree and the scaling path are used
    for angle in [0.01, 0.2, 0.9, 2.0, 5.0, 10.0, 100.0] {
        let generator = matrix(2, &[0.0, -angle, angle, 0.0]);
        let (c, s) = (f64::cos(angle), f64::sin(angle));
        let expected = matrix(2, &[c, -s, s, c]);
        let tolerance = if angle > 50.0 { 1e-11 } else { 1e-13 };
        assert_tensor_close_rel(&generator.matrix_exp().unwrap(), &expected, tolerance);
    }
}

#[test]
fn matrix_exp_of_a_two_state_generator_has_its_closed_form() {
    let (a, b) = (0.7, 0.3);
    for time in [0.01, 0.5, 2.0, 20.0] {
        let q = matrix(2, &[-a * time, a * time, b * time, -b * time]);
        let e = (-(a + b) * time).exp();
        let expected = matrix(
            2,
            &[b + a * e, a - a * e, b - b * e, a + b * e].map(|x| x / (a + b)),
        );
        assert_tensor_close_rel(&q.matrix_exp().unwrap(), &expected, 1e-13);
    }
}

#[test]
fn matrix_exp_matches_the_taylor_series_and_is_batched() {
    let mut rng = Lcg(11);
    // norms around each threshold
    let mut batch = Vec::new();
    for scale in [0.003, 0.06, 0.3, 0.6, 1.4, 3.0] {
        let a = rng.tensor(&[4, 4]) * scale;
        assert_tensor_close_rel(&a.matrix_exp().unwrap(), &taylor_exp(&a), 1e-13);
        batch.extend_from_slice(a.as_slice());
    }
    let batched = DoubleTensor::new(&[2, 3, 4, 4], batch).unwrap();
    let exp = batched.matrix_exp().unwrap();
    assert_eq!(exp.shape(), &[2, 3, 4, 4]);
    for (k, lane) in exp.as_slice().chunks(16).enumerate() {
        let single = DoubleTensor::new(&[4, 4], batched.as_slice()[k * 16..(k + 1) * 16].to_vec())
            .unwrap()
            .matrix_exp()
            .unwrap();
        assert_eq!(lane, single.as_slice());
    }
}

#[test]
fn matrix_exp_of_large_norms_uses_scaling_and_squaring() {
    let mut rng = Lcg(5);
    // symmetric: exp(S) = Q diag(exp λ) Qᵀ
    let s = symmetric(&mut rng, &[], 5) * 8.0;
    let (lambda, q) = s.eigh().unwrap();
    let expected = q
        .mul(&lambda.exp().reshape(&[1, 5]).unwrap())
        .unwrap()
        .matmul_transposed(&q, false, true)
        .unwrap();
    assert_tensor_close_rel(&s.matrix_exp().unwrap(), &expected, 1e-11);

    // a general matrix: exp(2A) = exp(A)², and exp(A) exp(-A) = I
    let a = rng.tensor(&[6, 6]) * 6.0;
    let exp = a.matrix_exp().unwrap();
    let squared = exp.matmul(&exp).unwrap();
    assert_tensor_close_rel(&(a.clone() * 2.0).matrix_exp().unwrap(), &squared, 1e-11);
    // to rounding relative to the sizes of the factors, as products of large matrices cancel
    let inverse = a.neg().matrix_exp().unwrap();
    let identity = exp.matmul(&inverse).unwrap();
    let size = exp.abs().max().unwrap() * inverse.abs().max().unwrap();
    let error = identity
        .sub(&DoubleTensor::eye(6).unwrap())
        .unwrap()
        .abs()
        .max()
        .unwrap();
    assert!(error < 1e-13 * size, "{error} vs {size}");
}

#[test]
fn rows_of_the_exponential_of_a_generator_sum_to_one() {
    // an illness-death style generator: rows sum to zero, off-diagonal rates non-negative
    let q = matrix(
        4,
        &[
            -0.9, 0.5, 0.3, 0.1, 0.2, -0.6, 0.3, 0.1, 0.0, 0.4, -0.5, 0.1, 0.0, 0.0, 0.0, 0.0,
        ],
    );
    for time in [0.1, 1.0, 10.0, 100.0] {
        let p = (q.clone() * time).matrix_exp().unwrap();
        for row in p.as_slice().chunks(4) {
            assert!(row.iter().all(|&x| x >= -1e-15), "{row:?}");
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-13, "row sum {sum} at t = {time}");
        }
    }
}

#[test]
fn matrix_exp_edge_cases_and_errors() {
    let empty = DoubleTensor::zeros(&[3, 0, 0]);
    assert_eq!(empty.matrix_exp().unwrap().shape(), &[3, 0, 0]);
    let zero = DoubleTensor::zeros(&[3, 3]);
    assert_eq!(
        zero.matrix_exp().unwrap().as_slice(),
        DoubleTensor::eye(3).unwrap().as_slice()
    );
    let nan = matrix(2, &[1.0, f64::NAN, 0.0, 1.0]).matrix_exp().unwrap();
    assert!(nan.as_slice().iter().all(|x| x.is_nan()));
    let infinite = matrix(2, &[1.0, f64::INFINITY, 0.0, 1.0])
        .matrix_exp()
        .unwrap();
    assert!(infinite.as_slice().iter().all(|x| x.is_nan()));
    assert!(matches!(
        DoubleTensor::zeros(&[2, 3]).matrix_exp(),
        Err(TensorError::InvalidArgument(_))
    ));
    assert!(DoubleTensor::zeros(&[3]).matrix_exp().is_err());
    assert!(matches!(
        DoubleTensor::zeros(&[2, 2]).matrix_exp_frechet(&DoubleTensor::zeros(&[3, 3])),
        Err(TensorError::IncompatibleShapes { .. })
    ));
    assert!(
        DoubleTensor::zeros(&[2, 2, 2])
            .matrix_exp_frechet(&DoubleTensor::zeros(&[3, 2, 2]))
            .is_err()
    );
}

/// (exp(A + hE) − exp(A − hE)) / 2h.
fn exp_difference(a: &DoubleTensor, e: &DoubleTensor, h: f64) -> DoubleTensor {
    let up = a.add(&(e.clone() * h)).unwrap().matrix_exp().unwrap();
    let down = a.sub(&(e.clone() * h)).unwrap().matrix_exp().unwrap();
    up.sub(&down).unwrap() * (0.5 / h)
}

#[test]
fn the_frechet_derivative_matches_finite_differences_and_closed_forms() {
    let mut rng = Lcg(3);
    for scale in [0.01, 0.5, 2.0, 12.0] {
        let a = rng.tensor(&[4, 4]) * scale;
        for e_scale in [1e-6, 1.0, 1e4] {
            let e = rng.tensor(&[4, 4]) * e_scale;
            let derivative = a.matrix_exp_frechet(&e).unwrap();
            let numeric = exp_difference(&a, &(e.clone() * (1.0 / e_scale)), 1e-5) * e_scale;
            let size = numeric.abs().max().unwrap_or(1.0).max(1e-300);
            for (x, y) in derivative.as_slice().iter().zip(numeric.as_slice()) {
                assert!(
                    (x - y).abs() <= 1e-7 * size,
                    "‖A‖ ~ {scale}, ‖E‖ ~ {e_scale}: {x} != {y}"
                );
            }
        }
        // A commutes with itself: L(A, A) = A exp(A)
        let expected = a.matmul(&a.matrix_exp().unwrap()).unwrap();
        assert_tensor_close_rel(&a.matrix_exp_frechet(&a).unwrap(), &expected, 1e-12);
    }
    // the adjoint: ⟨G, L(A, E)⟩ = ⟨L(Aᵀ, G), E⟩
    let (a, e, g) = (
        rng.tensor(&[3, 3]),
        rng.tensor(&[3, 3]),
        rng.tensor(&[3, 3]),
    );
    let inner = |x: &DoubleTensor, y: &DoubleTensor| x.mul(y).unwrap().sum();
    let left = inner(&g, &a.matrix_exp_frechet(&e).unwrap());
    let right = inner(&a.transpose().unwrap().matrix_exp_frechet(&g).unwrap(), &e);
    assert!((left - right).abs() < 1e-12 * left.abs().max(1.0));
    // zero directions
    assert_eq!(
        a.matrix_exp_frechet(&DoubleTensor::zeros(&[3, 3]))
            .unwrap()
            .as_slice(),
        &[0.0; 9]
    );
}

#[test]
fn the_frechet_derivative_broadcasts_batches() {
    let mut rng = Lcg(9);
    let a = rng.tensor(&[2, 3, 3]);
    let e = rng.tensor(&[4, 1, 3, 3]);
    let derivative = a.matrix_exp_frechet(&e).unwrap();
    assert_eq!(derivative.shape(), &[4, 2, 3, 3]);
    let lane = |t: &DoubleTensor, k: usize| {
        DoubleTensor::new(&[3, 3], t.as_slice()[k * 9..(k + 1) * 9].to_vec()).unwrap()
    };
    for i in 0..4 {
        for j in 0..2 {
            let single = lane(&a, j).matrix_exp_frechet(&lane(&e, i)).unwrap();
            assert_eq!(lane(&derivative, i * 2 + j).as_slice(), single.as_slice());
        }
    }
}

#[test]
fn the_second_frechet_derivative_matches_finite_differences_of_the_first() {
    let mut rng = Lcg(21);
    for scale in [0.05, 1.0, 8.0] {
        let a = rng.tensor(&[3, 3]) * scale;
        let (e1, e2) = (rng.tensor(&[3, 3]), rng.tensor(&[3, 3]) * 100.0);
        let second = a.matrix_exp_frechet_second(&e1, &e2).unwrap();
        let h = 1e-6;
        let up = a
            .add(&(e2.clone() * (h / 100.0)))
            .unwrap()
            .matrix_exp_frechet(&e1)
            .unwrap();
        let down = a
            .sub(&(e2.clone() * (h / 100.0)))
            .unwrap()
            .matrix_exp_frechet(&e1)
            .unwrap();
        let numeric = up.sub(&down).unwrap() * (50.0 / h);
        let size = numeric.abs().max().unwrap();
        for (x, y) in second.as_slice().iter().zip(numeric.as_slice()) {
            assert!((x - y).abs() <= 1e-6 * size, "‖A‖ ~ {scale}: {x} != {y}");
        }
        // symmetric in the directions
        let swapped = a.matrix_exp_frechet_second(&e2, &e1).unwrap();
        assert_tensor_close_rel(&swapped, &second, 1e-12);
    }
}
