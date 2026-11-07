//! Sparse matrices in compressed sparse row (CSR) form.

use crate::error::{Result, TensorError};
use crate::tensor::DoubleTensor;

/// A sparse matrix of f64 values in compressed sparse row (CSR) form: for each row, the column indices of its
/// stored entries (strictly increasing) and their values.
///
/// It is a constant: a neighbourhood structure, a design matrix with mostly zeros, a graph Laplacian. Products with
/// dense tensors cost O(stored entries × columns of the right-hand side), not O(rows × columns).
///
/// ```
/// use keanu_tensor::{DoubleTensor, SparseMatrix};
///
/// // [[2, 0, 1],
/// //  [0, 0, 0],
/// //  [0, 3, 0]]
/// let a = SparseMatrix::from_triplets(3, 3, &[(0, 0, 2.0), (0, 2, 1.0), (2, 1, 3.0)])?;
/// let x = DoubleTensor::vector(vec![1.0, 2.0, 3.0]);
/// assert_eq!(a.matvec(&x)?.as_slice(), &[5.0, 0.0, 6.0]);
/// assert_eq!(a.transpose_matvec(&x)?.as_slice(), &[2.0, 9.0, 1.0]);
/// assert_eq!(a.to_dense().shape(), &[3, 3]);
/// # Ok::<(), keanu_tensor::TensorError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SparseMatrix {
    rows: usize,
    cols: usize,
    /// `row_offsets[i]..row_offsets[i + 1]` are row i's entries in `col_indices` and `values`.
    row_offsets: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<f64>,
}

fn invalid<T>(message: String) -> Result<T> {
    Err(TensorError::InvalidArgument(message))
}

/// `rows + 1`, the number of row offsets, or an error if it overflows.
fn offset_count(rows: usize) -> Result<usize> {
    rows.checked_add(1).map_or_else(
        || invalid(format!("a sparse matrix can't have {rows} rows")),
        Ok,
    )
}

/// A vector of `len` zeros, or an error (not an abort) if it can't be allocated.
fn zeros(len: usize) -> Result<Vec<usize>> {
    let mut v = Vec::new();
    if v.try_reserve_exact(len).is_err() {
        return invalid(format!(
            "a sparse matrix's {len} row offsets can't be allocated"
        ));
    }
    v.resize(len, 0);
    Ok(v)
}

impl SparseMatrix {
    /// A matrix from its CSR arrays: `row_offsets` has `rows + 1` non-decreasing entries starting at 0 and ending
    /// at the number of stored entries, and row i's entries are `col_indices[row_offsets[i]..row_offsets[i + 1]]`
    /// with the same range of `values`.
    ///
    /// # Errors
    ///
    /// [`TensorError::InvalidArgument`] if the offsets are malformed (including `rows + 1` overflowing), the arrays'
    /// lengths differ, a column index is out of range, a row's column indices are not strictly increasing (sorted
    /// and unique), or a value is not finite.
    pub fn from_csr(
        rows: usize,
        cols: usize,
        row_offsets: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<f64>,
    ) -> Result<Self> {
        let count = offset_count(rows)?;
        if row_offsets.len() != count {
            return invalid(format!(
                "a sparse matrix of {rows} rows needs {count} row offsets, not {}",
                row_offsets.len()
            ));
        }
        if col_indices.len() != values.len() {
            return invalid(format!(
                "a sparse matrix has {} column indices but {} values",
                col_indices.len(),
                values.len()
            ));
        }
        if row_offsets[0] != 0 || row_offsets[rows] != values.len() {
            return invalid(format!(
                "a sparse matrix's row offsets must run from 0 to the number of entries, {}, not {} to {}",
                values.len(),
                row_offsets[0],
                row_offsets[rows]
            ));
        }
        for (i, pair) in row_offsets.windows(2).enumerate() {
            if pair[0] > pair[1] {
                return invalid(format!("the row offsets decrease at row {i}"));
            }
            let row = &col_indices[pair[0]..pair[1]];
            if let Some(&j) = row.iter().find(|&&j| j >= cols) {
                return invalid(format!(
                    "column index {j} in row {i} is out of range for {cols} columns"
                ));
            }
            if row.windows(2).any(|w| w[0] >= w[1]) {
                return invalid(format!(
                    "the column indices of row {i} are not strictly increasing: {row:?}"
                ));
            }
        }
        if let Some(v) = values.iter().find(|v| !v.is_finite()) {
            return invalid(format!("a sparse matrix's values must be finite, not {v}"));
        }
        Ok(SparseMatrix {
            rows,
            cols,
            row_offsets,
            col_indices,
            values,
        })
    }

    /// A matrix from `(row, column, value)` triplets in any order. Explicit zeros are stored.
    ///
    /// # Errors
    ///
    /// [`TensorError::InvalidArgument`] if an index is out of range, a position is given twice, a value is not
    /// finite, or the row offsets (one per row, plus one) can't be allocated.
    pub fn from_triplets(
        rows: usize,
        cols: usize,
        triplets: &[(usize, usize, f64)],
    ) -> Result<Self> {
        if let Some(&(i, j, _)) = triplets.iter().find(|&&(i, j, _)| i >= rows || j >= cols) {
            return invalid(format!(
                "the entry ({i}, {j}) is out of range for a {rows} × {cols} matrix"
            ));
        }
        let mut sorted = triplets.to_vec();
        sorted.sort_by_key(|&(i, j, _)| (i, j));
        if let Some(w) = sorted
            .windows(2)
            .find(|w| (w[0].0, w[0].1) == (w[1].0, w[1].1))
        {
            return invalid(format!("the entry ({}, {}) is given twice", w[0].0, w[0].1));
        }
        let mut row_offsets = zeros(offset_count(rows)?)?;
        for &(i, _, _) in &sorted {
            row_offsets[i + 1] += 1;
        }
        for i in 0..rows {
            row_offsets[i + 1] += row_offsets[i];
        }
        let col_indices = sorted.iter().map(|&(_, j, _)| j).collect();
        let values = sorted.iter().map(|&(_, _, v)| v).collect();
        Self::from_csr(rows, cols, row_offsets, col_indices, values)
    }

    /// The nonzero entries of a dense matrix (shape `[rows, cols]`).
    ///
    /// # Errors
    ///
    /// [`TensorError::InvalidArgument`] if `dense` is not a matrix or has a non-finite element.
    pub fn from_dense(dense: &DoubleTensor) -> Result<Self> {
        let &[rows, cols] = dense.shape() else {
            return invalid(format!(
                "a sparse matrix is built from a matrix, not a tensor of shape {:?}",
                dense.shape()
            ));
        };
        let mut row_offsets = Vec::with_capacity(rows + 1);
        row_offsets.push(0);
        let (mut col_indices, mut values) = (Vec::new(), Vec::new());
        for row in dense.as_slice().chunks_exact(cols.max(1)).take(rows) {
            for (j, &v) in row.iter().enumerate() {
                if v != 0.0 {
                    col_indices.push(j);
                    values.push(v);
                }
            }
            row_offsets.push(values.len());
        }
        row_offsets.resize(rows + 1, values.len());
        Self::from_csr(rows, cols, row_offsets, col_indices, values)
    }

    /// The number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The number of stored entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// The row offsets: row i's entries are at `row_offsets()[i]..row_offsets()[i + 1]`.
    pub fn row_offsets(&self) -> &[usize] {
        &self.row_offsets
    }

    /// The column index of each stored entry, row by row.
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    /// The value of each stored entry, row by row.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// The stored entries as `(row, column, value)` triplets, row by row.
    pub fn triplets(&self) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
        (0..self.rows).flat_map(move |i| {
            let range = self.row_offsets[i]..self.row_offsets[i + 1];
            self.col_indices[range.clone()]
                .iter()
                .zip(&self.values[range])
                .map(move |(&j, &v)| (i, j, v))
        })
    }

    /// The matrix as a dense tensor of shape `[rows, cols]`.
    pub fn to_dense(&self) -> DoubleTensor {
        let mut data = vec![0.0; self.rows * self.cols];
        for (i, j, v) in self.triplets() {
            data[i * self.cols + j] = v;
        }
        DoubleTensor::new(&[self.rows, self.cols], data).expect("the shape matches the data")
    }

    /// The transposed matrix.
    pub fn transpose(&self) -> Self {
        let triplets: Vec<_> = self.triplets().map(|(i, j, v)| (j, i, v)).collect();
        Self::from_triplets(self.cols, self.rows, &triplets).expect("a transpose is valid")
    }

    /// A x for a vector x of `cols` elements.
    ///
    /// # Errors
    ///
    /// [`TensorError::IncompatibleShapes`] unless x has shape `[cols]`.
    pub fn matvec(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        self.require(x, 1)?;
        self.mul_along_axis(x, 0, false)
    }

    /// A X for a matrix X of `cols` rows.
    ///
    /// # Errors
    ///
    /// [`TensorError::IncompatibleShapes`] unless X has shape `[cols, k]`.
    pub fn matmat(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        self.require(x, 2)?;
        self.mul_along_axis(x, 0, false)
    }

    /// Aᵀ x for a vector x of `rows` elements.
    ///
    /// # Errors
    ///
    /// [`TensorError::IncompatibleShapes`] unless x has shape `[rows]`.
    pub fn transpose_matvec(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        self.require_transposed(x, 1)?;
        self.mul_along_axis(x, 0, true)
    }

    /// Aᵀ X for a matrix X of `rows` rows.
    ///
    /// # Errors
    ///
    /// [`TensorError::IncompatibleShapes`] unless X has shape `[rows, k]`.
    pub fn transpose_matmat(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        self.require_transposed(x, 2)?;
        self.mul_along_axis(x, 0, true)
    }

    /// The product of A (or Aᵀ if `transpose`) with `x` along one of its axes: the axis, of length `cols` (`rows`
    /// if transposed), is replaced by one of length `rows` (`cols`), every other axis kept, so that each fibre of x
    /// along the axis is multiplied by the matrix. With axis 0 this is the ordinary product A X.
    ///
    /// # Errors
    ///
    /// [`TensorError::InvalidAxis`] for an axis out of range and [`TensorError::IncompatibleShapes`] if its length
    /// doesn't match the matrix.
    pub fn mul_along_axis(
        &self,
        x: &DoubleTensor,
        axis: usize,
        transpose: bool,
    ) -> Result<DoubleTensor> {
        let shape = x.shape();
        if axis >= shape.len() {
            return Err(TensorError::InvalidAxis {
                axis: axis as isize,
                rank: shape.len(),
            });
        }
        let (input, output) = if transpose {
            (self.rows, self.cols)
        } else {
            (self.cols, self.rows)
        };
        if shape[axis] != input {
            return Err(self.incompatible(shape));
        }
        let outer: usize = shape[..axis].iter().product();
        let inner: usize = shape[axis + 1..].iter().product();
        let mut out_shape = shape.to_vec();
        out_shape[axis] = output;
        let xs = x.as_slice();
        let mut out = vec![0.0; outer * output * inner];
        for o in 0..outer {
            let x_block = &xs[o * input * inner..(o + 1) * input * inner];
            let out_block = &mut out[o * output * inner..(o + 1) * output * inner];
            for i in 0..self.rows {
                let range = self.row_offsets[i]..self.row_offsets[i + 1];
                for (&j, &v) in self.col_indices[range.clone()]
                    .iter()
                    .zip(&self.values[range])
                {
                    // A[i, j]: out[i] += v x[j], or transposed out[j] += v x[i]
                    let (to, from) = if transpose { (j, i) } else { (i, j) };
                    let source = &x_block[from * inner..(from + 1) * inner];
                    let target = &mut out_block[to * inner..(to + 1) * inner];
                    for (t, s) in target.iter_mut().zip(source) {
                        *t += v * s;
                    }
                }
            }
        }
        DoubleTensor::new(&out_shape, out)
    }

    fn incompatible(&self, shape: &[usize]) -> TensorError {
        TensorError::IncompatibleShapes {
            left: vec![self.rows, self.cols],
            right: shape.to_vec(),
        }
    }

    fn require(&self, x: &DoubleTensor, rank: usize) -> Result<()> {
        if x.rank() == rank && x.shape()[0] == self.cols {
            Ok(())
        } else {
            Err(self.incompatible(x.shape()))
        }
    }

    fn require_transposed(&self, x: &DoubleTensor, rank: usize) -> Result<()> {
        if x.rank() == rank && x.shape()[0] == self.rows {
            Ok(())
        } else {
            Err(self.incompatible(x.shape()))
        }
    }
}
