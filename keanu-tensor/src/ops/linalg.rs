//! Linear algebra on the last two dimensions, batched over (and broadcasting) any leading dimensions.

use crate::error::{Result, TensorError};
use crate::shape::{self, Shape};
use crate::tensor::{DoubleTensor, Storage, Tensor};

/// Scratch space for one matrix, on the stack up to 8 × 8.
type Scratch = smallvec::SmallVec<[f64; 64]>;
use faer::{MatRef, Side};

impl DoubleTensor {
    fn matrix_dims(&self, name: &str) -> Result<(usize, usize)> {
        if self.rank() < 2 {
            return Err(TensorError::InvalidArgument(format!(
                "{name} needs a tensor of rank >= 2, got shape {:?}",
                self.shape()
            )));
        }
        Ok((self.shape[self.rank() - 2], self.shape[self.rank() - 1]))
    }

    fn square_dims(&self, name: &str) -> Result<usize> {
        let (rows, cols) = self.matrix_dims(name)?;
        if rows != cols {
            return Err(TensorError::InvalidArgument(format!(
                "{name} needs square matrices, got shape {:?}",
                self.shape()
            )));
        }
        Ok(rows)
    }

    fn batch_shape(&self) -> &[usize] {
        &self.shape[..self.rank() - 2]
    }

    /// Matrix product of the last two dimensions, optionally transposing either operand. Leading (batch)
    /// dimensions broadcast. Both operands must have rank >= 2.
    pub fn matmul(&self, other: &Self) -> Result<Self> {
        self.matmul_transposed(other, false, false)
    }

    /// The matrix product, optionally transposing either operand.
    pub fn matmul_transposed(
        &self,
        other: &Self,
        transpose_left: bool,
        transpose_right: bool,
    ) -> Result<Self> {
        let (ar, ac) = self.matrix_dims("matmul")?;
        let (br, bc) = other.matrix_dims("matmul")?;
        let (m, k) = if transpose_left { (ac, ar) } else { (ar, ac) };
        let (k2, n) = if transpose_right { (bc, br) } else { (br, bc) };
        if k != k2 {
            return Err(TensorError::IncompatibleShapes {
                left: self.shape.to_vec(),
                right: other.shape.to_vec(),
            });
        }
        let batch = shape::broadcast_shape(self.batch_shape(), other.batch_shape())?;
        let batch_len = shape::length(&batch);
        let out_len = shape::checked_length(&[batch_len, m, n]).ok_or_else(|| {
            TensorError::IncompatibleShapes {
                left: self.shape.to_vec(),
                right: other.shape.to_vec(),
            }
        })?;
        let left_strides = shape::broadcast_strides(self.batch_shape(), &batch);
        let right_strides = shape::broadcast_strides(other.batch_shape(), &batch);
        let batch_strides = shape::row_major_strides(&batch);

        // zeroed once here: matmul_into accumulates small products into it
        let mut data: Storage<f64> = smallvec::smallvec![0.0; out_len];
        for (i, out) in data
            .chunks_mut(m.saturating_mul(n).max(1))
            .enumerate()
            .take(batch_len)
        {
            let (mut l, mut r) = (0, 0);
            for d in 0..batch.len() {
                let index = (i / batch_strides[d]) % batch[d];
                l += index * left_strides[d];
                r += index * right_strides[d];
            }
            matmul_into(
                &self.data[l * ar * ac..(l + 1) * ar * ac],
                (ar, ac, transpose_left),
                &other.data[r * br * bc..(r + 1) * br * bc],
                (br, bc, transpose_right),
                out,
                (m, k, n),
            );
        }
        let mut out_shape = batch;
        out_shape.extend([m, n]);
        Ok(Tensor::from_parts(out_shape, data))
    }

    /// Lower triangular Cholesky factor L (A = L Lᵀ) of each symmetric positive definite matrix. Only the lower
    /// triangle of the input is read.
    pub fn cholesky(&self) -> Result<Self> {
        let n = self.square_dims("cholesky")?;
        let mut data: Storage<f64> = smallvec::smallvec![0.0; self.len()];
        for (input, out) in self
            .data
            .chunks((n * n).max(1))
            .zip(data.chunks_mut((n * n).max(1)))
        {
            if n <= SMALL_CHOLESKY {
                cholesky_lower_into(input, n, out)?;
                continue;
            }
            let llt = MatRef::from_row_major_slice(input, n, n)
                .llt(Side::Lower)
                .map_err(|_| TensorError::NotPositiveDefinite)?;
            let l = llt.L();
            for row in 0..n {
                for col in 0..=row {
                    out[row * n + col] = l[(row, col)];
                }
            }
        }
        Ok(Tensor::from_parts(self.shape.clone(), data))
    }

    /// The inverse (L Lᵀ)⁻¹ of the matrices whose lower triangular Cholesky factors L are given. Only the lower
    /// triangle of the input is read. The result is the full symmetric inverse.
    pub fn cholesky_inverse(&self) -> Result<Self> {
        let n = self.square_dims("cholesky_inverse")?;
        let mut data: Storage<f64> = smallvec::smallvec![0.0; self.len()];
        for (l, out) in self
            .data
            .chunks((n * n).max(1))
            .zip(data.chunks_mut((n * n).max(1)))
        {
            if n > SMALL_TRIANGULAR_SOLVE {
                cholesky_inverse_large(l, n, out)?;
                continue;
            }
            // inverse of L by forward substitution
            let mut l_inv: Scratch = smallvec::smallvec![0.0; n * n];
            for j in 0..n {
                let diagonal = l[j * n + j];
                if diagonal == 0.0 {
                    return Err(TensorError::SingularMatrix);
                }
                l_inv[j * n + j] = 1.0 / diagonal;
                for i in j + 1..n {
                    let sum: f64 = (j..i).map(|k| l[i * n + k] * l_inv[k * n + j]).sum();
                    l_inv[i * n + j] = -sum / l[i * n + i];
                }
            }
            // A⁻¹ = L⁻ᵀ L⁻¹
            for i in 0..n {
                for j in 0..=i {
                    let value: f64 = (i..n).map(|k| l_inv[k * n + i] * l_inv[k * n + j]).sum();
                    out[i * n + j] = value;
                    out[j * n + i] = value;
                }
            }
        }
        Ok(Tensor::from_parts(self.shape.clone(), data))
    }

    /// Solves op(T) X = B for X, where T is the triangular matrix in the last two dimensions of self (`[..., n, n]`)
    /// and op(T) is T, or Tᵀ if `transpose`. Only the lower (if `lower`) or upper triangle of T, diagonal
    /// included, is read. B is `[..., n, k]`, or a single vector `[n]` (giving `[..., n]`). Leading (batch) dimensions
    /// broadcast, as in [`Self::matmul`]. A matrix with an exactly zero diagonal element is singular, which is
    /// an error.
    ///
    /// With L the [`Self::cholesky`] factor of A, `l.solve_triangular(b, true, false)` and then
    /// `.solve_triangular(.., true, true)` on the result solve A X = B without forming A⁻¹.
    pub fn solve_triangular(&self, b: &Self, lower: bool, transpose: bool) -> Result<Self> {
        let n = self.square_dims("solve_triangular")?;
        let incompatible = || TensorError::IncompatibleShapes {
            left: self.shape.to_vec(),
            right: b.shape.to_vec(),
        };
        let vector = b.rank() == 1;
        let (bn, k) = if vector {
            (b.shape[0], 1)
        } else {
            b.matrix_dims("solve_triangular")?
        };
        if bn != n {
            return Err(incompatible());
        }
        if self
            .data
            .chunks_exact((n * n).max(1))
            .any(|t| (0..n).any(|i| t[i * n + i] == 0.0))
        {
            return Err(TensorError::SingularMatrix);
        }
        let b_batch: &[usize] = if vector { &[] } else { b.batch_shape() };
        let batch = shape::broadcast_shape(self.batch_shape(), b_batch)?;
        let batch_len = shape::length(&batch);
        let out_len = shape::checked_length(&[batch_len, n, k]).ok_or_else(incompatible)?;
        let left_strides = shape::broadcast_strides(self.batch_shape(), &batch);
        let right_strides = shape::broadcast_strides(b_batch, &batch);
        let batch_strides = shape::row_major_strides(&batch);
        // forward substitution for a lower triangular op(T), back substitution for an upper one
        let forward = lower != transpose;

        let mut data: Storage<f64> = smallvec::smallvec![0.0; out_len];
        for (i, out) in data.chunks_mut((n * k).max(1)).enumerate().take(batch_len) {
            let (mut l, mut r) = (0, 0);
            for d in 0..batch.len() {
                let index = (i / batch_strides[d]) % batch[d];
                l += index * left_strides[d];
                r += index * right_strides[d];
            }
            out.copy_from_slice(&b.data[r * n * k..(r + 1) * n * k]);
            let t = &self.data[l * n * n..(l + 1) * n * n];
            if n <= SMALL_TRIANGULAR_SOLVE {
                substitute_in_place(t, n, transpose, forward, out, k);
                continue;
            }
            let t = MatRef::from_row_major_slice(t, n, n);
            let t = if transpose { t.transpose() } else { t };
            let rhs = faer::MatMut::from_row_major_slice_mut(out, n, k);
            let par = if n * n * k <= SEQUENTIAL_MATMUL {
                faer::Par::Seq
            } else {
                faer::get_global_parallelism()
            };
            if forward {
                faer::linalg::triangular_solve::solve_lower_triangular_in_place(t, rhs, par);
            } else {
                faer::linalg::triangular_solve::solve_upper_triangular_in_place(t, rhs, par);
            }
        }
        let mut out_shape = batch;
        out_shape.push(n);
        if !vector {
            out_shape.push(k);
        }
        Ok(Tensor::from_parts(out_shape, data))
    }

    /// Inverse of each square matrix. A matrix with an exactly zero pivot is singular, which is an error.
    pub fn inverse(&self) -> Result<Self> {
        let n = self.square_dims("inverse")?;
        let mut data: Storage<f64> = smallvec::smallvec![0.0; self.len()];
        for (input, out) in self
            .data
            .chunks((n * n).max(1))
            .zip(data.chunks_mut((n * n).max(1)))
        {
            let lu = Lu::factor(input, n);
            if lu.singular {
                return Err(TensorError::SingularMatrix);
            }
            lu.inverse_into(out);
        }
        Ok(Tensor::from_parts(self.shape.clone(), data))
    }

    /// Solves A X = B for X, where A is the square matrix in the last two dimensions of self (`[..., n, n]`), by LU
    /// factorisation with partial pivoting. B is `[..., n, k]`, or a single vector `[n]` (giving `[..., n]`). Leading
    /// (batch) dimensions broadcast, as in [`Self::matmul`]. A matrix with an exactly zero pivot is singular, which
    /// is an error ([`TensorError::SingularMatrix`], as in [`Self::inverse`]).
    pub fn solve(&self, b: &Self) -> Result<Self> {
        let n = self.square_dims("solve")?;
        let incompatible = || TensorError::IncompatibleShapes {
            left: self.shape.to_vec(),
            right: b.shape.to_vec(),
        };
        let vector = b.rank() == 1;
        let (bn, k) = if vector {
            (b.shape[0], 1)
        } else {
            b.matrix_dims("solve")?
        };
        if bn != n {
            return Err(incompatible());
        }
        let b_batch: &[usize] = if vector { &[] } else { b.batch_shape() };
        let batch = shape::broadcast_shape(self.batch_shape(), b_batch)?;
        let batch_len = shape::length(&batch);
        let out_len = shape::checked_length(&[batch_len, n, k]).ok_or_else(incompatible)?;
        // every matrix is factored once, however many right-hand sides use it
        let factors: Vec<Lu> = self
            .data
            .chunks_exact((n * n).max(1))
            .take(shape::length(self.batch_shape()))
            .map(|a| Lu::factor(a, n))
            .collect();
        if factors.iter().any(|lu| lu.singular) {
            return Err(TensorError::SingularMatrix);
        }
        let left_strides = shape::broadcast_strides(self.batch_shape(), &batch);
        let right_strides = shape::broadcast_strides(b_batch, &batch);
        let batch_strides = shape::row_major_strides(&batch);

        let mut data: Storage<f64> = smallvec::smallvec![0.0; out_len];
        if n > 0 && k > 0 {
            for (i, out) in data.chunks_mut(n * k).enumerate().take(batch_len) {
                let (mut l, mut r) = (0, 0);
                for d in 0..batch.len() {
                    let index = (i / batch_strides[d]) % batch[d];
                    l += index * left_strides[d];
                    r += index * right_strides[d];
                }
                out.copy_from_slice(&b.data[r * n * k..(r + 1) * n * k]);
                factors[l].solve_in_place(out, k);
            }
        }
        let mut out_shape = batch;
        out_shape.push(n);
        if !vector {
            out_shape.push(k);
        }
        Ok(Tensor::from_parts(out_shape, data))
    }

    /// The sign and natural log of the absolute value of the determinant of each square matrix, as numpy's
    /// `slogdet`: shape [..., n, n] gives two tensors of shape [...]. The log is summed from the LU factors, so it
    /// is finite where the determinant itself would overflow or underflow. A singular matrix (an exactly zero
    /// pivot, as in [`Self::inverse`]) has sign 0 and log|det| -∞; a 0 × 0 matrix has sign 1 and log|det| 0.
    pub fn slogdet(&self) -> Result<(Self, Self)> {
        let n = self.square_dims("slogdet")?;
        let batch: Shape = self.batch_shape().into();
        let count = shape::length(&batch);
        let mut signs: Storage<f64> = smallvec::smallvec![1.0; count];
        let mut logs: Storage<f64> = smallvec::smallvec![0.0; count];
        if n > 0 {
            for ((matrix, sign), log) in self
                .data
                .chunks_exact(n * n)
                .zip(signs.iter_mut())
                .zip(logs.iter_mut())
            {
                (*sign, *log) = Lu::factor(matrix, n).sign_and_log_abs_determinant();
            }
        }
        Ok((
            Tensor::from_parts(batch.clone(), signs),
            Tensor::from_parts(batch, logs),
        ))
    }

    /// The eigendecomposition A = Q diag(λ) Qᵀ of each symmetric matrix, by faer's self-adjoint eigensolver. Only
    /// the lower triangle of each matrix, diagonal included, is read. Returns the eigenvalues λ (`[..., n]`, in
    /// ascending order) and the orthonormal eigenvectors Q (`[..., n, n]`, one per column, in the same order). Each
    /// eigenvector's sign is chosen so that its first component (lowest row) whose magnitude is within a relative
    /// 1e-10 of the largest is positive, so the result is deterministic and components that tie in magnitude don't
    /// make the sign depend on rounding. Repeated eigenvalues have no unique eigenvectors: any orthonormal basis of
    /// their eigenspace may be returned. A matrix with a non-finite element in its lower triangle gives NaN
    /// eigenvalues and eigenvectors.
    ///
    /// The eigenvectors (and so their derivatives) are discontinuous where the sign convention switches (as the
    /// matrix varies, the component deciding the sign changes, flipping the eigenvector) and at repeated
    /// eigenvalues; elsewhere they are smooth.
    pub fn eigh(&self) -> Result<(Self, Self)> {
        let n = self.square_dims("eigh")?;
        let batch = self.batch_shape();
        let count = shape::length(batch);
        let mut values: Storage<f64> = smallvec::smallvec![0.0; count * n];
        let mut vectors: Storage<f64> = smallvec::smallvec![0.0; self.len()];
        if n > 0 {
            for ((a, lambda), q) in self
                .data
                .chunks_exact(n * n)
                .zip(values.chunks_exact_mut(n))
                .zip(vectors.chunks_exact_mut(n * n))
            {
                symmetric_eigen_into(a, n, lambda, q)?;
            }
        }
        let mut value_shape: Shape = batch.into();
        value_shape.push(n);
        Ok((
            Tensor::from_parts(value_shape, values),
            Tensor::from_parts(self.shape.clone(), vectors),
        ))
    }

    /// Determinant of each square matrix: shape [..., n, n] becomes [...]. Singular matrices give 0.
    pub fn determinant(&self) -> Result<Self> {
        let n = self.square_dims("determinant")?;
        let batch: Shape = self.batch_shape().into();
        let data: Storage<f64> = if n == 0 {
            // the determinant of a 0 × 0 matrix is 1 (the empty product), as in numpy
            smallvec::smallvec![1.0; shape::length(&batch)]
        } else {
            self.data
                .chunks(n * n)
                .take(shape::length(&batch))
                .map(|input| lu_determinant(input, n))
                .collect()
        };
        Ok(Tensor::from_parts(batch, data))
    }

    /// Generalised tensor contraction: sums the products over `left_axes` of self paired with `right_axes` of
    /// other. The result's dimensions are self's remaining dimensions followed by other's.
    pub fn tensor_multiply(
        &self,
        other: &Self,
        left_axes: &[isize],
        right_axes: &[isize],
    ) -> Result<Self> {
        if left_axes.len() != right_axes.len() {
            return Err(TensorError::InvalidArgument(
                "the same number of axes must be given for both tensors".into(),
            ));
        }
        let left: Vec<usize> = left_axes
            .iter()
            .map(|&a| shape::axis(a, self.rank()))
            .collect::<Result<_>>()?;
        let right: Vec<usize> = right_axes
            .iter()
            .map(|&a| shape::axis(a, other.rank()))
            .collect::<Result<_>>()?;
        for (&l, &r) in left.iter().zip(&right) {
            if self.shape[l] != other.shape[r] {
                return Err(TensorError::IncompatibleShapes {
                    left: self.shape.to_vec(),
                    right: other.shape.to_vec(),
                });
            }
        }
        let kept_left: Vec<usize> = (0..self.rank()).filter(|d| !left.contains(d)).collect();
        let kept_right: Vec<usize> = (0..other.rank()).filter(|d| !right.contains(d)).collect();
        let incompatible = || TensorError::IncompatibleShapes {
            left: self.shape.to_vec(),
            right: other.shape.to_vec(),
        };
        let out_dims: Shape = kept_left
            .iter()
            .map(|&d| self.shape[d])
            .chain(kept_right.iter().map(|&d| other.shape[d]))
            .collect();
        shape::checked_length(&out_dims).ok_or_else(incompatible)?;
        if self.is_empty() || other.is_empty() {
            // nothing to multiply (and the sizes below may not fit when a dimension is zero): every sum is empty
            return Ok(Tensor::zeros(&out_dims));
        }
        // explicit sizes rather than -1, which cannot be inferred when the contracted length is zero
        let size = |t: &Self, dims: &[usize]| {
            let dims: Shape = dims.iter().map(|&d| t.shape[d]).collect();
            shape::length(&dims) as isize
        };
        let contracted = size(self, &left);
        // the permuted copies are reshaped in place
        let lhs = self
            .permute(&[kept_left.as_slice(), &left].concat())?
            .into_reshape(&[size(self, &kept_left), contracted])?;
        let rhs = other
            .permute(&[right.as_slice(), &kept_right].concat())?
            .into_reshape(&[contracted, size(other, &kept_right)])?;
        let product = lhs.matmul(&rhs)?;
        Ok(Tensor::from_parts(out_dims, product.data))
    }
}

/// Factors `lu` (row-major n × n) in place into L and U with partial pivoting, P A = L U, recording the row
/// swapped with row k at step k in `pivots`. Returns whether a pivot was exactly zero (the matrix is singular).
fn lu_in_place(lu: &mut [f64], n: usize, pivots: &mut [usize]) -> bool {
    let mut singular = false;
    for k in 0..n {
        let mut pivot = k;
        let mut largest = lu[k * n + k].abs();
        for i in k + 1..n {
            let candidate = lu[i * n + k].abs();
            if candidate > largest {
                largest = candidate;
                pivot = i;
            }
        }
        pivots[k] = pivot;
        if pivot != k {
            for j in 0..n {
                lu.swap(k * n + j, pivot * n + j);
            }
        }
        let ukk = lu[k * n + k];
        if ukk == 0.0 {
            singular = true;
            continue;
        }
        // eliminate below the pivot, row by row (slices avoid bounds checks in the inner loop)
        let (above, below) = lu.split_at_mut((k + 1) * n);
        let pivot_row = &above[k * n + k + 1..(k + 1) * n];
        for row in below.chunks_exact_mut(n) {
            let factor = row[k] / ukk;
            row[k] = factor;
            if factor != 0.0 {
                for (a, &p) in row[k + 1..].iter_mut().zip(pivot_row) {
                    *a -= factor * p;
                }
            }
        }
    }
    singular
}

/// The determinant of one row-major n × n matrix by LU factorisation (0 if a pivot is exactly zero).
fn lu_determinant(matrix: &[f64], n: usize) -> f64 {
    let mut lu: Scratch = Scratch::from_slice(matrix);
    let mut pivots: smallvec::SmallVec<[usize; 8]> = smallvec::smallvec![0; n];
    if lu_in_place(&mut lu, n, &mut pivots) {
        return 0.0;
    }
    let swaps = pivots.iter().enumerate().filter(|&(i, &p)| i != p).count();
    let sign = if swaps % 2 == 0 { 1.0 } else { -1.0 };
    sign * (0..n).map(|i| lu[i * n + i]).product::<f64>()
}

/// LU factorisation with partial pivoting, P A = L U, on one row-major n × n matrix.
struct Lu {
    n: usize,
    lu: Scratch,
    /// row i was swapped with row pivots[i] at step i
    pivots: smallvec::SmallVec<[usize; 8]>,
    singular: bool,
}

impl Lu {
    fn factor(matrix: &[f64], n: usize) -> Self {
        let mut lu: Scratch = Scratch::from_slice(matrix);
        let mut pivots: smallvec::SmallVec<[usize; 8]> = smallvec::smallvec![0; n];
        let singular = lu_in_place(&mut lu, n, &mut pivots);
        Self {
            n,
            lu,
            pivots,
            singular,
        }
    }

    /// Solves A X = B in place in `x` (row-major n × k, holding B on entry). The factorisation must not be singular.
    fn solve_in_place(&self, x: &mut [f64], k: usize) {
        let n = self.n;
        let lu: &[f64] = &self.lu;
        for (row, &p) in self.pivots.iter().enumerate() {
            if p != row {
                for j in 0..k {
                    x.swap(row * k + j, p * k + j);
                }
            }
        }
        // L (unit diagonal) forward, then U backward, a row of right-hand sides at a time
        for i in 0..n {
            let (solved, rest) = x.split_at_mut(i * k);
            let row = &mut rest[..k];
            for (j, &l) in lu[i * n..i * n + i].iter().enumerate() {
                if l != 0.0 {
                    for (xi, &xj) in row.iter_mut().zip(&solved[j * k..(j + 1) * k]) {
                        *xi -= l * xj;
                    }
                }
            }
        }
        for i in (0..n).rev() {
            let (head, solved) = x.split_at_mut((i + 1) * k);
            let row = &mut head[i * k..];
            for (j, &u) in lu[i * n + i + 1..(i + 1) * n].iter().enumerate() {
                if u != 0.0 {
                    for (xi, &xj) in row.iter_mut().zip(&solved[j * k..(j + 1) * k]) {
                        *xi -= u * xj;
                    }
                }
            }
            let diagonal = lu[i * n + i];
            row.iter_mut().for_each(|xi| *xi /= diagonal);
        }
    }

    /// The sign (±1, or 0 if singular) and ln |det| (-∞ if singular) of the factored matrix.
    fn sign_and_log_abs_determinant(&self) -> (f64, f64) {
        if self.singular {
            return (0.0, f64::NEG_INFINITY);
        }
        let n = self.n;
        let swaps = self
            .pivots
            .iter()
            .enumerate()
            .filter(|&(i, &p)| i != p)
            .count();
        let mut sign = if swaps % 2 == 0 { 1.0 } else { -1.0 };
        let mut log = 0.0;
        for i in 0..n {
            let u = self.lu[i * n + i];
            sign *= u.signum();
            log += u.abs().ln();
        }
        (sign, log)
    }

    fn inverse_into(&self, out: &mut [f64]) {
        let n = self.n;
        let lu: &[f64] = &self.lu;
        let mut column_storage: smallvec::SmallVec<[f64; 8]> = smallvec::smallvec![0.0; n];
        let column: &mut [f64] = &mut column_storage;
        for c in 0..n {
            column.iter_mut().for_each(|x| *x = 0.0);
            column[c] = 1.0;
            for (k, &p) in self.pivots.iter().enumerate() {
                column.swap(k, p);
            }
            for i in 0..n {
                let row = &lu[i * n..i * n + i];
                let sum = row
                    .iter()
                    .zip(&column[..i])
                    .fold(column[i], |sum, (l, x)| sum - l * x);
                column[i] = sum;
            }
            for i in (0..n).rev() {
                let row = &lu[i * n + i + 1..(i + 1) * n];
                let sum = row
                    .iter()
                    .zip(&column[i + 1..])
                    .fold(column[i], |sum, (u, x)| sum - u * x);
                column[i] = sum / lu[i * n + i];
            }
            for r in 0..n {
                out[r * n + c] = column[r];
            }
        }
    }
}

/// Below this many multiply-adds a matrix product uses a plain loop: faer's dispatch (and its thread pool) costs
/// more than the arithmetic for the small matrices typical of models.
const SMALL_MATMUL: usize = 32 * 32 * 32;
/// Below this many multiply-adds faer runs sequentially; above it, on its global (rayon) parallelism.
const SEQUENTIAL_MATMUL: usize = 256 * 256 * 256;
/// Up to this size Cholesky uses a direct loop rather than faer.
const SMALL_CHOLESKY: usize = 64;

/// out[m × n] = op(A) op(B), row-major, where op optionally transposes. `a` is (rows, cols, transposed) of A as
/// stored, and similarly for `b`. `out` must be zeroed: small products accumulate into it.
fn matmul_into(
    a: &[f64],
    (ar, ac, ta): (usize, usize, bool),
    b: &[f64],
    (br, bc, tb): (usize, usize, bool),
    out: &mut [f64],
    (m, k, n): (usize, usize, usize),
) {
    let work = m * n * k;
    if work <= SMALL_MATMUL {
        debug_assert!(out.iter().all(|&x| x == 0.0));
        let a_at = |i: usize, p: usize| if ta { a[p * ac + i] } else { a[i * ac + p] };
        for i in 0..m {
            let row = &mut out[i * n..(i + 1) * n];
            for p in 0..k {
                let aip = a_at(i, p);
                if tb {
                    for (j, o) in row.iter_mut().enumerate() {
                        *o += aip * b[j * bc + p];
                    }
                } else {
                    let b_row = &b[p * bc..(p + 1) * bc];
                    for (o, &bpj) in row.iter_mut().zip(b_row) {
                        *o += aip * bpj;
                    }
                }
            }
        }
        return;
    }
    let lhs = MatRef::from_row_major_slice(a, ar, ac);
    let rhs = MatRef::from_row_major_slice(b, br, bc);
    let lhs = if ta { lhs.transpose() } else { lhs };
    let rhs = if tb { rhs.transpose() } else { rhs };
    let par = if work <= SEQUENTIAL_MATMUL {
        faer::Par::Seq
    } else {
        faer::get_global_parallelism()
    };
    faer::linalg::matmul::matmul(
        faer::MatMut::from_row_major_slice_mut(out, m, n),
        faer::Accum::Replace,
        lhs,
        rhs,
        1.0,
        par,
    );
}

/// Up to this size a triangular solve uses direct substitution rather than faer.
const SMALL_TRIANGULAR_SOLVE: usize = 64;

/// Solves op(T) X = B in place in `x` (row-major n × k, holding B on entry), where T is row-major n × n and op
/// transposes it if `transpose`. Substitutes forward (op(T) lower triangular) or backward, reading only that
/// triangle of op(T). The diagonal must be non-zero.
fn substitute_in_place(
    t: &[f64],
    n: usize,
    transpose: bool,
    forward: bool,
    x: &mut [f64],
    k: usize,
) {
    let at = |i: usize, j: usize| {
        if transpose {
            t[j * n + i]
        } else {
            t[i * n + j]
        }
    };
    let mut solve_row = |i: usize, solved: &mut dyn Iterator<Item = usize>| {
        for j in solved {
            let factor = at(i, j);
            if factor != 0.0 {
                let (row_i, row_j) = if j < i {
                    let (head, tail) = x.split_at_mut(i * k);
                    (&mut tail[..k], &head[j * k..(j + 1) * k])
                } else {
                    let (head, tail) = x.split_at_mut(j * k);
                    (&mut head[i * k..(i + 1) * k], &tail[..k])
                };
                for (xi, &xj) in row_i.iter_mut().zip(row_j) {
                    *xi -= factor * xj;
                }
            }
        }
        let diagonal = at(i, i);
        x[i * k..(i + 1) * k]
            .iter_mut()
            .for_each(|xi| *xi /= diagonal);
    };
    if forward {
        for i in 0..n {
            solve_row(i, &mut (0..i));
        }
    } else {
        for i in (0..n).rev() {
            solve_row(i, &mut (i + 1..n));
        }
    }
}

/// `cholesky_inverse` of one n × n factor with faer: L⁻¹ by a triangular solve against the identity, then
/// L⁻ᵀ L⁻¹, mirrored from its lower triangle so the result is exactly symmetric.
fn cholesky_inverse_large(l: &[f64], n: usize, out: &mut [f64]) -> Result<()> {
    if (0..n).any(|i| l[i * n + i] == 0.0) {
        return Err(TensorError::SingularMatrix);
    }
    let par = if n * n * n <= SEQUENTIAL_MATMUL {
        faer::Par::Seq
    } else {
        faer::get_global_parallelism()
    };
    let mut l_inv = faer::Mat::<f64>::identity(n, n);
    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
        MatRef::from_row_major_slice(l, n, n),
        l_inv.as_mut(),
        par,
    );
    faer::linalg::matmul::matmul(
        faer::MatMut::from_row_major_slice_mut(out, n, n),
        faer::Accum::Replace,
        l_inv.transpose(),
        l_inv.as_ref(),
        1.0,
        par,
    );
    for i in 0..n {
        for j in 0..i {
            out[j * n + i] = out[i * n + j];
        }
    }
    Ok(())
}

/// Lower Cholesky factor of one n × n matrix (reading only its lower triangle) into `out`, which must be zeroed.
fn cholesky_lower_into(a: &[f64], n: usize, out: &mut [f64]) -> Result<()> {
    for j in 0..n {
        let row_j = j * n;
        let diagonal = a[row_j + j] - out[row_j..row_j + j].iter().map(|l| l * l).sum::<f64>();
        if diagonal.is_nan() || diagonal <= 0.0 {
            return Err(TensorError::NotPositiveDefinite);
        }
        let ljj = diagonal.sqrt();
        out[row_j + j] = ljj;
        for i in j + 1..n {
            let row_i = i * n;
            let dot: f64 = out[row_i..row_i + j]
                .iter()
                .zip(&out[row_j..row_j + j])
                .map(|(x, y)| x * y)
                .sum();
            out[row_i + j] = (a[row_i + j] - dot) / ljj;
        }
    }
    Ok(())
}

/// The relative tolerance within which eigenvector components count as tied for the largest magnitude when
/// choosing the eigenvector's sign (see [`Tensor::eigh`]).
const SIGN_TOLERANCE: f64 = 1e-10;

/// The eigenvalues (ascending, into `values`) and eigenvectors (columns of row-major `vectors`) of one symmetric
/// n × n matrix, reading only its lower triangle. Each eigenvector is signed so that its first component within a
/// relative [`SIGN_TOLERANCE`] of the largest magnitude is positive.
fn symmetric_eigen_into(
    a: &[f64],
    n: usize,
    values: &mut [f64],
    vectors: &mut [f64],
) -> Result<()> {
    if (0..n).any(|i| a[i * n..=i * n + i].iter().any(|x| !x.is_finite())) {
        values.fill(f64::NAN);
        vectors.fill(f64::NAN);
        return Ok(());
    }
    let eigen = MatRef::from_row_major_slice(a, n, n)
        .self_adjoint_eigen(Side::Lower)
        .map_err(|_| {
            TensorError::InvalidArgument("the symmetric eigendecomposition did not converge".into())
        })?;
    let (s, u) = (eigen.S(), eigen.U());
    for col in 0..n {
        values[col] = s[col];
        let largest = (0..n).map(|row| u[(row, col)].abs()).fold(0.0, f64::max);
        // the first component within a relative SIGN_TOLERANCE of the largest decides the sign, so components
        // that tie in magnitude (up to rounding) don't make the sign depend on rounding
        let leading = (0..n)
            .map(|row| u[(row, col)])
            .find(|x| x.abs() >= largest * (1.0 - SIGN_TOLERANCE))
            .unwrap_or(0.0);
        let flip = if leading < 0.0 { -1.0 } else { 1.0 };
        for row in 0..n {
            vectors[row * n + col] = flip * u[(row, col)];
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------------
// the matrix exponential and its Fréchet derivatives

impl DoubleTensor {
    /// The matrix exponential exp(A) = Σₖ Aᵏ / k! of each square matrix, batched over leading dimensions, by
    /// scaling and squaring with Padé approximants (Higham, 2005, the algorithm of MATLAB's `expm`): the smallest
    /// degree m of 3, 5, 7, 9 whose threshold θₘ bounds ‖A‖₁ is used directly; otherwise A is scaled by 2⁻ˢ until
    /// ‖A / 2ˢ‖₁ ≤ θ₁₃, the degree 13 approximant is evaluated and squared s times. A matrix with a non-finite
    /// element gives NaN; a 0 × 0 matrix gives itself.
    pub fn matrix_exp(&self) -> Result<Self> {
        let n = self.square_dims("matrix_exp")?;
        let mut data: Storage<f64> = smallvec::smallvec![0.0; self.len()];
        if n > 0 {
            for (a, out) in self
                .data
                .chunks_exact(n * n)
                .zip(data.chunks_exact_mut(n * n))
            {
                out.copy_from_slice(&expm(a, n)?);
            }
        }
        Ok(Tensor::from_parts(self.shape.clone(), data))
    }

    /// The Fréchet derivative L(A, E) of the matrix exponential at each matrix A (self, `[..., n, n]`) in the
    /// direction E (`[..., n, n]`): the derivative of exp(A + εE) at ε = 0, which is the upper right block of
    /// exp([[A, E], [0, A]]) (Mathias, 1996). Leading (batch) dimensions broadcast. Since L(A, ·) is linear, E is
    /// scaled so that its norm doesn't exceed A's (or 1) before exponentiating, and the result scaled back. The
    /// adjoint of L(A, ·) in the trace inner product is L(Aᵀ, ·).
    pub fn matrix_exp_frechet(&self, direction: &Self) -> Result<Self> {
        broadcast_square("matrix_exp_frechet", &[self, direction], |m, n, out| {
            let (a, e) = (m[0], m[1]);
            let c = direction_scale(a, e, n);
            if c == 0.0 {
                out.fill(0.0);
                return Ok(());
            }
            let e: Scratch = e.iter().map(|x| x * c).collect();
            let block = block_upper_triangular(n, &[a, &e, a], 2);
            let exp = expm(&block, 2 * n)?;
            copy_block(&exp, 2 * n, n, 0, 1, out);
            out.iter_mut().for_each(|x| *x /= c);
            Ok(())
        })
    }

    /// The second Fréchet derivative L²(A; E₁, E₂) of the matrix exponential at each matrix A (self) in the
    /// directions E₁ and E₂: the mixed derivative of exp(A + αE₁ + βE₂) at α = β = 0, so the derivative of
    /// L(A, E₁) along E₂. It is the sum of the upper right blocks of exp([[A, E₁, 0], [0, A, E₂], [0, 0, A]]) and of
    /// the same with E₁ and E₂ exchanged. All three are `[..., n, n]`, with leading dimensions broadcasting; the
    /// directions are scaled as in [`Self::matrix_exp_frechet`].
    pub fn matrix_exp_frechet_second(&self, first: &Self, second: &Self) -> Result<Self> {
        broadcast_square(
            "matrix_exp_frechet_second",
            &[self, first, second],
            |m, n, out| {
                let a = m[0];
                let (c1, c2) = (direction_scale(a, m[1], n), direction_scale(a, m[2], n));
                if c1 == 0.0 || c2 == 0.0 {
                    out.fill(0.0);
                    return Ok(());
                }
                let e1: Scratch = m[1].iter().map(|x| x * c1).collect();
                let e2: Scratch = m[2].iter().map(|x| x * c2).collect();
                let mut ordered: Scratch = smallvec::smallvec![0.0; n * n];
                for (left, right) in [(&e1, &e2), (&e2, &e1)] {
                    let block = block_upper_triangular(n, &[a, left, &[], a, right, a], 3);
                    let exp = expm(&block, 3 * n)?;
                    copy_block(&exp, 3 * n, n, 0, 2, &mut ordered);
                    out.iter_mut().zip(&ordered).for_each(|(o, x)| *o += x);
                }
                let c = c1 * c2;
                out.iter_mut().for_each(|x| *x /= c);
                Ok(())
            },
        )
    }
}

/// Applies `f` to the matrices at each broadcast batch index of square tensors `[..., n, n]` of the same n,
/// giving one `[batch..., n, n]` tensor. `f` gets the matrices (row-major n × n) and a zeroed output matrix.
fn broadcast_square(
    name: &str,
    tensors: &[&DoubleTensor],
    mut f: impl FnMut(&[&[f64]], usize, &mut [f64]) -> Result<()>,
) -> Result<DoubleTensor> {
    let n = tensors[0].square_dims(name)?;
    let mut batch: Shape = tensors[0].batch_shape().into();
    for t in &tensors[1..] {
        if t.square_dims(name)? != n {
            return Err(TensorError::IncompatibleShapes {
                left: tensors[0].shape.to_vec(),
                right: t.shape.to_vec(),
            });
        }
        batch = shape::broadcast_shape(&batch, t.batch_shape())?;
    }
    let batch_len = shape::length(&batch);
    let strides: Vec<Shape> = tensors
        .iter()
        .map(|t| shape::broadcast_strides(t.batch_shape(), &batch))
        .collect();
    let batch_strides = shape::row_major_strides(&batch);
    let mut out_shape = batch.clone();
    out_shape.extend([n, n]);
    let mut data: Storage<f64> = smallvec::smallvec![0.0; batch_len * n * n];
    if n > 0 {
        let mut matrices: smallvec::SmallVec<[&[f64]; 3]> = smallvec::SmallVec::new();
        for (i, out) in data.chunks_exact_mut(n * n).enumerate() {
            matrices.clear();
            for (t, s) in tensors.iter().zip(&strides) {
                let offset: usize = (0..batch.len())
                    .map(|d| ((i / batch_strides[d]) % batch[d]) * s[d])
                    .sum();
                matrices.push(&t.data[offset * n * n..(offset + 1) * n * n]);
            }
            f(&matrices, n, out)?;
        }
    }
    Ok(Tensor::from_parts(out_shape, data))
}

/// The factor c scaling a direction E so that ‖cE‖₁ is at most max(‖A‖₁, 1): 0 for a zero direction, NaN for a
/// non-finite one.
fn direction_scale(a: &[f64], e: &[f64], n: usize) -> f64 {
    let ne = one_norm(e, n);
    if !ne.is_finite() {
        return f64::NAN;
    }
    let bound = one_norm(a, n).max(1.0);
    if ne == 0.0 {
        0.0
    } else if ne > bound {
        bound / ne
    } else {
        1.0
    }
}

/// The block upper triangular (k n) × (k n) matrix whose upper triangle of blocks, row by row, is `blocks` (an
/// empty slice is a zero block).
fn block_upper_triangular(n: usize, blocks: &[&[f64]], k: usize) -> Vec<f64> {
    let size = k * n;
    let mut out = vec![0.0; size * size];
    let mut next = blocks.iter();
    for block_row in 0..k {
        for block_col in block_row..k {
            let block = next.next().copied().unwrap_or(&[]);
            if block.is_empty() {
                continue;
            }
            for r in 0..n {
                let start = (block_row * n + r) * size + block_col * n;
                out[start..start + n].copy_from_slice(&block[r * n..(r + 1) * n]);
            }
        }
    }
    out
}

/// Copies block (`block_row`, `block_col`) of n × n blocks of a row-major `size` × `size` matrix into `out`.
fn copy_block(
    matrix: &[f64],
    size: usize,
    n: usize,
    block_row: usize,
    block_col: usize,
    out: &mut [f64],
) {
    for r in 0..n {
        let start = (block_row * n + r) * size + block_col * n;
        out[r * n..(r + 1) * n].copy_from_slice(&matrix[start..start + n]);
    }
}

/// The 1-norm (largest absolute column sum) of a row-major n × n matrix; NaN if an element is NaN.
fn one_norm(a: &[f64], n: usize) -> f64 {
    if a.iter().any(|x| x.is_nan()) {
        return f64::NAN;
    }
    (0..n)
        .map(|col| (0..n).map(|row| a[row * n + col].abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

/// θₘ for m = 3, 5, 7, 9 and 13: the largest ‖A‖₁ for which the degree m Padé approximant has backward error at
/// most the unit roundoff 2⁻⁵³ (Higham, 2005, Table 2.3).
const PADE_THETAS: [(usize, f64); 4] = [
    (3, 1.495585217958292e-2),
    (5, 2.53939833006323e-1),
    (7, 9.504178996162932e-1),
    (9, 2.097847961257068e0),
];
const THETA_13: f64 = 5.371920351148152e0;

/// The coefficients b₀, …, bₘ of the degree m diagonal Padé approximant to exp (Higham, 2005, Algorithm 2.3).
fn pade_coefficients(m: usize) -> &'static [f64] {
    match m {
        3 => &[120.0, 60.0, 12.0, 1.0],
        5 => &[30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0],
        7 => &[
            17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0, 1.0,
        ],
        9 => &[
            17643225600.0,
            8821612800.0,
            2075673600.0,
            302702400.0,
            30270240.0,
            2162160.0,
            110880.0,
            3960.0,
            90.0,
            1.0,
        ],
        _ => &[
            64764752532480000.0,
            32382376266240000.0,
            7771770303897600.0,
            1187353796428800.0,
            129060195264000.0,
            10559470521600.0,
            670442572800.0,
            33522128640.0,
            1323241920.0,
            40840800.0,
            960960.0,
            16380.0,
            182.0,
            1.0,
        ],
    }
}

/// The product of two row-major n × n matrices.
fn square_product(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    matmul_into(a, (n, n, false), b, (n, n, false), &mut out, (n, n, n));
    out
}

/// Σᵢ cᵢ Mᵢ (+ c I if `identity` is given) of row-major n × n matrices.
fn combination(terms: &[(f64, &[f64])], identity: f64, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    for &(c, m) in terms {
        out.iter_mut().zip(m).for_each(|(o, x)| *o += c * x);
    }
    for i in 0..n {
        out[i * n + i] += identity;
    }
    out
}

/// exp(A) of one row-major n × n matrix (n > 0) by scaling and squaring; NaN if an element is not finite.
fn expm(a: &[f64], n: usize) -> Result<Vec<f64>> {
    if a.iter().any(|x| !x.is_finite()) {
        return Ok(vec![f64::NAN; n * n]);
    }
    let norm = one_norm(a, n);
    let a2 = square_product(a, a, n);
    for &(m, theta) in &PADE_THETAS {
        if norm <= theta {
            return pade_low(a, &a2, n, m);
        }
    }
    let s = if norm > THETA_13 {
        (norm / THETA_13).log2().ceil().max(0.0) as i32
    } else {
        0
    };
    let mut x = if s == 0 {
        pade_13(a, &a2, n)?
    } else {
        let scale = 2f64.powi(-s);
        let scaled: Vec<f64> = a.iter().map(|x| x * scale).collect();
        let scaled2: Vec<f64> = a2.iter().map(|x| x * scale * scale).collect();
        pade_13(&scaled, &scaled2, n)?
    };
    for _ in 0..s {
        x = square_product(&x, &x, n);
    }
    Ok(x)
}

/// The degree m (3, 5, 7 or 9) Padé approximant r(A) = (V − U)⁻¹ (V + U), with U = A Σ b₂ⱼ₊₁ A²ʲ and
/// V = Σ b₂ⱼ A²ʲ. `a2` is A².
fn pade_low(a: &[f64], a2: &[f64], n: usize, m: usize) -> Result<Vec<f64>> {
    let b = pade_coefficients(m);
    // the even powers A², A⁴, ..., A^(m - 1)
    let mut powers: Vec<Vec<f64>> = vec![a2.to_vec()];
    while 2 * (powers.len() + 1) < m {
        let next = square_product(powers.last().expect("A² is first"), a2, n);
        powers.push(next);
    }
    let odd: Vec<(f64, &[f64])> = powers
        .iter()
        .enumerate()
        .map(|(j, p)| (b[2 * j + 3], p.as_slice()))
        .collect();
    let even: Vec<(f64, &[f64])> = powers
        .iter()
        .enumerate()
        .map(|(j, p)| (b[2 * j + 2], p.as_slice()))
        .collect();
    let u = square_product(a, &combination(&odd, b[1], n), n);
    let v = combination(&even, b[0], n);
    pade_quotient(&u, &v, n)
}

/// The degree 13 Padé approximant, evaluated with A², A⁴ and A⁶ only (Higham, 2005, equation 2.9). `a2` is A².
fn pade_13(a: &[f64], a2: &[f64], n: usize) -> Result<Vec<f64>> {
    let b = pade_coefficients(13);
    let a4 = square_product(a2, a2, n);
    let a6 = square_product(a4.as_slice(), a2, n);
    let inner_u = combination(&[(b[13], &a6), (b[11], &a4), (b[9], a2)], 0.0, n);
    let u = square_product(
        a,
        &combination(
            &[
                (1.0, &square_product(&a6, &inner_u, n)),
                (b[7], &a6),
                (b[5], &a4),
                (b[3], a2),
            ],
            b[1],
            n,
        ),
        n,
    );
    let inner_v = combination(&[(b[12], &a6), (b[10], &a4), (b[8], a2)], 0.0, n);
    let v = combination(
        &[
            (1.0, &square_product(&a6, &inner_v, n)),
            (b[6], &a6),
            (b[4], &a4),
            (b[2], a2),
        ],
        b[0],
        n,
    );
    pade_quotient(&u, &v, n)
}

/// (V − U)⁻¹ (V + U) by LU factorisation.
fn pade_quotient(u: &[f64], v: &[f64], n: usize) -> Result<Vec<f64>> {
    let denominator: Vec<f64> = v.iter().zip(u).map(|(v, u)| v - u).collect();
    let mut x: Vec<f64> = v.iter().zip(u).map(|(v, u)| v + u).collect();
    let lu = Lu::factor(&denominator, n);
    if lu.singular {
        return Err(TensorError::SingularMatrix);
    }
    lu.solve_in_place(&mut x, n);
    Ok(x)
}
