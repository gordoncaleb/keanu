//! Sparse matrices: construction, validation and products against the dense ones.

use keanu_tensor::{DoubleTensor, SparseMatrix, TensorError};

mod common;
use common::assert_all_close_rel;

/// Deterministic uniform values in [0, 1).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A random sparse matrix with about a fraction `density` of its entries stored, as triplets in a scrambled order.
fn random(rng: &mut Lcg, rows: usize, cols: usize, density: f64) -> Vec<(usize, usize, f64)> {
    let mut triplets = Vec::new();
    for i in 0..rows {
        for j in 0..cols {
            if rng.next() < density {
                triplets.push((i, j, rng.next() * 4.0 - 2.0));
            }
        }
    }
    // reverse, so the triplets are not in row-major order
    triplets.reverse();
    triplets
}

fn dense_of(rows: usize, cols: usize, triplets: &[(usize, usize, f64)]) -> DoubleTensor {
    let mut data = vec![0.0; rows * cols];
    for &(i, j, v) in triplets {
        data[i * cols + j] = v;
    }
    DoubleTensor::new(&[rows, cols], data).unwrap()
}

fn uniform(rng: &mut Lcg, shape: &[usize]) -> DoubleTensor {
    let len = shape.iter().product();
    DoubleTensor::new(shape, (0..len).map(|_| rng.next() * 2.0 - 1.0).collect()).unwrap()
}

#[test]
fn products_match_dense_matrix_products() {
    let mut rng = Lcg(7);
    for (rows, cols, density) in [
        (5, 7, 0.3),
        (12, 4, 0.5),
        (1, 1, 1.0),
        (9, 9, 0.1),
        (6, 3, 0.0),
    ] {
        let triplets = random(&mut rng, rows, cols, density);
        let a = SparseMatrix::from_triplets(rows, cols, &triplets).unwrap();
        let dense = dense_of(rows, cols, &triplets);
        assert_eq!(a.to_dense(), dense);
        assert_eq!(a.nnz(), triplets.len());
        assert_eq!(SparseMatrix::from_dense(&dense).unwrap().to_dense(), dense);
        let at = dense.transpose().unwrap();
        assert_eq!(a.transpose().to_dense(), at);

        let x = uniform(&mut rng, &[cols]);
        let column = x.reshape(&[cols as isize, 1]).unwrap();
        assert_all_close_rel(
            a.matvec(&x).unwrap().as_slice(),
            dense.matmul(&column).unwrap().as_slice(),
            1e-14,
            "product",
        );
        let m = uniform(&mut rng, &[cols, 3]);
        let product = a.matmat(&m).unwrap();
        assert_eq!(product.shape(), &[rows, 3]);
        assert_all_close_rel(
            product.as_slice(),
            dense.matmul(&m).unwrap().as_slice(),
            1e-14,
            "product",
        );

        let y = uniform(&mut rng, &[rows]);
        let column = y.reshape(&[rows as isize, 1]).unwrap();
        assert_all_close_rel(
            a.transpose_matvec(&y).unwrap().as_slice(),
            at.matmul(&column).unwrap().as_slice(),
            1e-14,
            "product",
        );
        let m = uniform(&mut rng, &[rows, 2]);
        assert_all_close_rel(
            a.transpose_matmat(&m).unwrap().as_slice(),
            at.matmul(&m).unwrap().as_slice(),
            1e-14,
            "product",
        );

        // along a middle axis of a batch: each fibre is multiplied
        let batch = uniform(&mut rng, &[2, cols, 3]);
        let along = a.mul_along_axis(&batch, 1, false).unwrap();
        assert_eq!(along.shape(), &[2, rows, 3]);
        for k in 0..2 {
            let fibre = batch.slice(&format!("{k}")).unwrap();
            let expected = dense.matmul(&fibre).unwrap();
            let actual = along.slice(&format!("{k}")).unwrap();
            assert_all_close_rel(actual.as_slice(), expected.as_slice(), 1e-14, "product");
        }
    }
}

#[test]
fn empty_rows_and_all_zero_matrices_work() {
    // row 1 and the last row are empty
    let a = SparseMatrix::from_triplets(4, 3, &[(0, 1, 2.0), (2, 0, -1.0), (2, 2, 3.0)]).unwrap();
    assert_eq!(a.row_offsets(), &[0, 1, 1, 3, 3]);
    assert_eq!(a.col_indices(), &[1, 0, 2]);
    assert_eq!(a.values(), &[2.0, -1.0, 3.0]);
    let x = DoubleTensor::vector(vec![1.0, 2.0, 3.0]);
    assert_eq!(a.matvec(&x).unwrap().as_slice(), &[4.0, 0.0, 8.0, 0.0]);
    let zero = SparseMatrix::from_triplets(3, 2, &[]).unwrap();
    assert_eq!(zero.nnz(), 0);
    assert_eq!(
        zero.matvec(&DoubleTensor::vector(vec![1.0, 2.0]))
            .unwrap()
            .as_slice(),
        &[0.0; 3]
    );
    assert_eq!(zero.to_dense(), DoubleTensor::zeros(&[3, 2]));
    assert_eq!(
        SparseMatrix::from_dense(&DoubleTensor::zeros(&[3, 2])).unwrap(),
        zero
    );
    // degenerate shapes
    let none = SparseMatrix::from_triplets(0, 3, &[]).unwrap();
    assert_eq!(none.matvec(&x).unwrap().shape(), &[0]);
    let thin = SparseMatrix::from_triplets(2, 0, &[]).unwrap();
    assert_eq!(
        thin.matvec(&DoubleTensor::vector(vec![]))
            .unwrap()
            .as_slice(),
        &[0.0, 0.0]
    );
    assert_eq!(
        SparseMatrix::from_dense(&DoubleTensor::zeros(&[2, 0])).unwrap(),
        thin
    );
    // explicit zeros are kept
    assert_eq!(
        SparseMatrix::from_triplets(2, 2, &[(1, 1, 0.0)])
            .unwrap()
            .nnz(),
        1
    );
}

#[test]
fn malformed_input_is_rejected() {
    let invalid =
        |r: Result<SparseMatrix, TensorError>| matches!(r, Err(TensorError::InvalidArgument(_)));
    // out of range, repeated, non-finite
    assert!(invalid(SparseMatrix::from_triplets(2, 2, &[(2, 0, 1.0)])));
    assert!(invalid(SparseMatrix::from_triplets(2, 2, &[(0, 2, 1.0)])));
    assert!(invalid(SparseMatrix::from_triplets(
        2,
        2,
        &[(0, 1, 1.0), (0, 1, 2.0)]
    )));
    assert!(invalid(SparseMatrix::from_triplets(
        2,
        2,
        &[(0, 1, f64::NAN)]
    )));
    assert!(invalid(SparseMatrix::from_triplets(
        2,
        2,
        &[(0, 1, f64::INFINITY)]
    )));
    // CSR arrays: wrong offset count, offsets not from 0 to nnz, decreasing, unsorted or repeated columns, out of
    // range, mismatched lengths
    assert!(invalid(SparseMatrix::from_csr(
        2,
        2,
        vec![0, 1],
        vec![0],
        vec![1.0]
    )));
    assert!(invalid(SparseMatrix::from_csr(
        2,
        2,
        vec![1, 1, 1],
        vec![0],
        vec![1.0]
    )));
    assert!(invalid(SparseMatrix::from_csr(
        2,
        2,
        vec![0, 1, 2],
        vec![0],
        vec![1.0]
    )));
    assert!(invalid(SparseMatrix::from_csr(
        2,
        2,
        vec![0, 2, 1],
        vec![0, 1],
        vec![1.0, 2.0]
    )));
    assert!(invalid(SparseMatrix::from_csr(
        1,
        2,
        vec![0, 2],
        vec![1, 0],
        vec![1.0, 2.0]
    )));
    assert!(invalid(SparseMatrix::from_csr(
        1,
        2,
        vec![0, 2],
        vec![1, 1],
        vec![1.0, 2.0]
    )));
    assert!(invalid(SparseMatrix::from_csr(
        1,
        2,
        vec![0, 1],
        vec![2],
        vec![1.0]
    )));
    assert!(invalid(SparseMatrix::from_csr(
        1,
        2,
        vec![0, 1],
        vec![0, 1],
        vec![1.0]
    )));
    assert!(
        SparseMatrix::from_csr(2, 3, vec![0, 2, 3], vec![0, 2, 1], vec![1.0, 2.0, 3.0]).is_ok()
    );
    assert!(invalid(SparseMatrix::from_dense(&DoubleTensor::zeros(&[
        2
    ]))));
    assert!(invalid(SparseMatrix::from_dense(
        &DoubleTensor::new(&[1, 2], vec![1.0, f64::NAN]).unwrap()
    )));
    // products with mismatched shapes
    let a = SparseMatrix::from_triplets(2, 3, &[(0, 0, 1.0)]).unwrap();
    assert!(matches!(
        a.matvec(&DoubleTensor::zeros(&[2])),
        Err(TensorError::IncompatibleShapes { .. })
    ));
    assert!(a.matvec(&DoubleTensor::zeros(&[3, 1])).is_err());
    assert!(a.matmat(&DoubleTensor::zeros(&[3])).is_err());
    assert!(a.transpose_matvec(&DoubleTensor::zeros(&[3])).is_err());
    assert!(a.transpose_matmat(&DoubleTensor::zeros(&[2, 4])).is_ok());
    assert!(matches!(
        a.mul_along_axis(&DoubleTensor::zeros(&[3]), 1, false),
        Err(TensorError::InvalidAxis { .. })
    ));
}
