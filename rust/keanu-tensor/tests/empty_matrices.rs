//! 0 × 0 matrices (and batches of them) are valid inputs: found by the no-panic property test in keanu-graph.

use keanu_tensor::DoubleTensor;

#[test]
fn linear_algebra_on_empty_matrices() {
    for shape in [vec![0, 0], vec![3, 0, 0]] {
        let empty = DoubleTensor::new(&shape, vec![]).unwrap();
        assert_eq!(empty.inverse().unwrap().shape(), shape.as_slice());
        assert_eq!(empty.cholesky().unwrap().shape(), shape.as_slice());
        assert_eq!(empty.cholesky_inverse().unwrap().shape(), shape.as_slice());
        // the determinant of a 0 × 0 matrix is the empty product, as in numpy
        let det = empty.determinant().unwrap();
        assert_eq!(det.shape(), &shape[..shape.len() - 2]);
        assert!(det.as_slice().iter().all(|&d| d == 1.0));
    }
}
