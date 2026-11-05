//! `solve_triangular` against the explicit inverse, on small (direct substitution) and large (faer) sizes.

use keanu_tensor::{DoubleTensor, TensorError};

mod common;
use common::assert_all_close_1p;

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

    fn values(&mut self, len: usize) -> Vec<f64> {
        (0..len).map(|_| self.next()).collect()
    }
}

/// A batch of well-conditioned triangular matrices: diagonal in [1, 2) (random sign), off-diagonal small. The
/// other triangle holds garbage, which must not be read.
fn triangular(rng: &mut Lcg, batch: &[usize], n: usize, lower: bool) -> DoubleTensor {
    let count: usize = batch.iter().product();
    let mut data = Vec::with_capacity(count * n * n);
    for _ in 0..count {
        for i in 0..n {
            for j in 0..n {
                let in_triangle = if lower { j < i } else { j > i };
                data.push(if i == j {
                    let d = 1.5 + 0.5 * rng.next();
                    if rng.next() < 0.0 { -d } else { d }
                } else if in_triangle {
                    rng.next() / n as f64
                } else {
                    1e3 * rng.next()
                });
            }
        }
    }
    let shape: Vec<usize> = batch.iter().copied().chain([n, n]).collect();
    DoubleTensor::new(&shape, data).unwrap()
}

/// The same matrices with the unused triangle zeroed.
fn masked(t: &DoubleTensor, lower: bool) -> DoubleTensor {
    let n = t.shape()[t.rank() - 1];
    let data = t
        .as_slice()
        .iter()
        .enumerate()
        .map(|(index, &x)| {
            let (i, j) = ((index / n) % n, index % n);
            if (lower && j <= i) || (!lower && j >= i) {
                x
            } else {
                0.0
            }
        })
        .collect();
    DoubleTensor::new(t.shape(), data).unwrap()
}

fn reference(t: &DoubleTensor, b: &DoubleTensor, lower: bool, transpose: bool) -> DoubleTensor {
    let op = masked(t, lower);
    let op = if transpose {
        op.transpose().unwrap()
    } else {
        op
    };
    op.inverse().unwrap().matmul(b).unwrap()
}

/// [`assert_all_close_1p`] on the elements, after checking the shapes agree.
fn assert_tensor_close_1p(actual: &DoubleTensor, expected: &DoubleTensor, tolerance: f64) {
    assert_eq!(actual.shape(), expected.shape());
    assert_all_close_1p(actual.as_slice(), expected.as_slice(), tolerance, "element");
}

#[test]
fn matches_the_inverse_for_every_variant_and_size() {
    let mut rng = Lcg(42);
    for n in [1, 2, 5, 17, 64, 65, 130] {
        for lower in [true, false] {
            for transpose in [false, true] {
                let t = triangular(&mut rng, &[], n, lower);
                let b = DoubleTensor::new(&[n, 3], rng.values(n * 3)).unwrap();
                let x = t.solve_triangular(&b, lower, transpose).unwrap();
                assert_tensor_close_1p(&x, &reference(&t, &b, lower, transpose), 1e-10);
            }
        }
    }
}

#[test]
fn batch_dimensions_broadcast() {
    let mut rng = Lcg(7);
    for n in [4, 80] {
        let t = triangular(&mut rng, &[2, 1], n, true);
        let b = DoubleTensor::new(&[3, n, 2], rng.values(3 * n * 2)).unwrap();
        let x = t.solve_triangular(&b, true, true).unwrap();
        assert_eq!(x.shape(), &[2, 3, n, 2]);
        assert_tensor_close_1p(&x, &reference(&t, &b, true, true), 1e-10);

        // a single matrix against a batch of right-hand sides, and vice versa
        let single = triangular(&mut rng, &[], n, false);
        let x = single.solve_triangular(&b, false, false).unwrap();
        assert_tensor_close_1p(&x, &reference(&single, &b, false, false), 1e-10);
        let single_b = DoubleTensor::new(&[n, 1], rng.values(n)).unwrap();
        let x = t.solve_triangular(&single_b, true, false).unwrap();
        assert_eq!(x.shape(), &[2, 1, n, 1]);
        assert_tensor_close_1p(&x, &reference(&t, &single_b, true, false), 1e-10);
    }
}

#[test]
fn vector_right_hand_side() {
    let mut rng = Lcg(3);
    let n = 6;
    let t = triangular(&mut rng, &[2], n, true);
    let b = DoubleTensor::new(&[n], rng.values(n)).unwrap();
    let x = t.solve_triangular(&b, true, false).unwrap();
    assert_eq!(x.shape(), &[2, n]);
    let column = b.reshape(&[n as isize, 1]).unwrap();
    let expected = reference(&t, &column, true, false)
        .reshape(&[2, n as isize])
        .unwrap();
    assert_tensor_close_1p(&x, &expected, 1e-12);
}

#[test]
fn zero_diagonal_is_singular() {
    for n in [3, 70] {
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        data[(n - 1) * n + n - 1] = 0.0;
        let t = DoubleTensor::new(&[n, n], data).unwrap();
        let b = DoubleTensor::ones(&[n, 1]);
        for (lower, transpose) in [(true, false), (false, true)] {
            assert!(matches!(
                t.solve_triangular(&b, lower, transpose),
                Err(TensorError::SingularMatrix)
            ));
        }
        // even with no right-hand side columns to solve for
        let empty_b = DoubleTensor::new(&[n, 0], vec![]).unwrap();
        assert!(matches!(
            t.solve_triangular(&empty_b, true, false),
            Err(TensorError::SingularMatrix)
        ));
    }
}

#[test]
fn shape_errors() {
    let square = DoubleTensor::eye(3).unwrap();
    let not_square = DoubleTensor::ones(&[3, 2]);
    let b = DoubleTensor::ones(&[3, 2]);
    assert!(not_square.solve_triangular(&b, true, false).is_err());
    assert!(
        DoubleTensor::ones(&[3])
            .solve_triangular(&b, true, false)
            .is_err()
    );
    assert!(matches!(
        square.solve_triangular(&DoubleTensor::ones(&[2, 2]), true, false),
        Err(TensorError::IncompatibleShapes { .. })
    ));
    assert!(matches!(
        square.solve_triangular(&DoubleTensor::ones(&[4]), true, false),
        Err(TensorError::IncompatibleShapes { .. })
    ));
    assert!(
        square
            .solve_triangular(&DoubleTensor::scalar(1.0), true, false)
            .is_err()
    );
    // batch dimensions that do not broadcast
    let batched = DoubleTensor::ones(&[2, 3, 3]);
    assert!(
        batched
            .solve_triangular(&DoubleTensor::ones(&[4, 3, 1]), true, false)
            .is_err()
    );
}

#[test]
fn empty_matrices() {
    let t = DoubleTensor::new(&[0, 0], vec![]).unwrap();
    let x = t
        .solve_triangular(&DoubleTensor::new(&[0, 2], vec![]).unwrap(), true, false)
        .unwrap();
    assert_eq!(x.shape(), &[0, 2]);
    let x = t
        .solve_triangular(&DoubleTensor::new(&[0], vec![]).unwrap(), false, true)
        .unwrap();
    assert_eq!(x.shape(), &[0]);
    let batch = DoubleTensor::new(&[0, 3, 3], vec![]).unwrap();
    let x = batch
        .solve_triangular(&DoubleTensor::ones(&[3, 1]), true, false)
        .unwrap();
    assert_eq!(x.shape(), &[0, 3, 1]);
}

#[test]
fn cholesky_inverse_matches_the_inverse_above_the_faer_threshold() {
    // above 64 cholesky_inverse goes through faer's triangular solve and matmul
    let mut rng = Lcg(11);
    for n in [64, 65, 150] {
        let b = DoubleTensor::new(&[2, n, n], rng.values(2 * n * n)).unwrap();
        let eye = &DoubleTensor::eye(n).unwrap() * n as f64;
        let a = b
            .matmul_transposed(&b, false, true)
            .unwrap()
            .add(&eye)
            .unwrap();
        let inverse = a.cholesky().unwrap().cholesky_inverse().unwrap();
        assert_tensor_close_1p(&inverse, &a.inverse().unwrap(), 1e-12);
        let data = inverse.as_slice();
        for m in 0..2 {
            for i in 0..n {
                for j in 0..i {
                    assert_eq!(data[m * n * n + i * n + j], data[m * n * n + j * n + i]);
                }
            }
        }
    }
    let mut singular = vec![0.0; 70 * 70];
    singular[0] = 1.0;
    assert!(matches!(
        DoubleTensor::new(&[70, 70], singular)
            .unwrap()
            .cholesky_inverse(),
        Err(TensorError::SingularMatrix)
    ));
}
