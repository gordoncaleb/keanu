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
