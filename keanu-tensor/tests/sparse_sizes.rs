//! Sparse matrices with sizes too large to represent or allocate are errors, not panics or aborts.

use keanu_tensor::SparseMatrix;

#[test]
fn a_sparse_matrix_too_large_to_build_is_an_error() {
    // rows + 1 overflows
    assert!(SparseMatrix::from_csr(usize::MAX, 1, vec![0], vec![], vec![]).is_err());
    assert!(SparseMatrix::from_triplets(usize::MAX, 1, &[]).is_err());
    // the row offsets can't be allocated
    assert!(SparseMatrix::from_triplets(1 << 62, 1, &[]).is_err());
    assert!(SparseMatrix::from_triplets(usize::MAX / 8, 1, &[(0, 0, 1.0)]).is_err());
    // huge columns need no allocation
    let wide = SparseMatrix::from_triplets(1, usize::MAX, &[(0, usize::MAX - 1, 1.0)]).unwrap();
    assert_eq!(wide.cols(), usize::MAX);
}
