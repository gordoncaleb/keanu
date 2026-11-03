//! Linear algebra on the last two dimensions, batched over (and broadcasting) any leading dimensions.

use crate::error::{Result, TensorError};
use crate::shape::{self, Shape};
use crate::tensor::{DoubleTensor, Tensor};
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
        let left_strides = shape::broadcast_strides(self.batch_shape(), &batch);
        let right_strides = shape::broadcast_strides(other.batch_shape(), &batch);
        let batch_strides = shape::row_major_strides(&batch);

        let mut data = vec![0.0; batch_len * m * n];
        for (i, out) in data.chunks_mut((m * n).max(1)).enumerate().take(batch_len) {
            let (mut l, mut r) = (0, 0);
            for d in 0..batch.len() {
                let index = (i / batch_strides[d]) % batch[d];
                l += index * left_strides[d];
                r += index * right_strides[d];
            }
            let lhs =
                MatRef::from_row_major_slice(&self.data[l * ar * ac..(l + 1) * ar * ac], ar, ac);
            let rhs =
                MatRef::from_row_major_slice(&other.data[r * br * bc..(r + 1) * br * bc], br, bc);
            let lhs = if transpose_left { lhs.transpose() } else { lhs };
            let rhs = if transpose_right {
                rhs.transpose()
            } else {
                rhs
            };
            let product = lhs * rhs;
            for row in 0..m {
                for col in 0..n {
                    out[row * n + col] = product[(row, col)];
                }
            }
        }
        let mut out_shape = batch;
        out_shape.extend([m, n]);
        Ok(Tensor::from_parts(out_shape, data))
    }

    /// Lower triangular Cholesky factor L (A = L Lᵀ) of each symmetric positive definite matrix. Only the lower
    /// triangle of the input is read.
    pub fn cholesky(&self) -> Result<Self> {
        let n = self.square_dims("cholesky")?;
        let mut data = vec![0.0; self.len()];
        for (input, out) in self.data.chunks(n * n).zip(data.chunks_mut(n * n)) {
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
        let mut data = vec![0.0; self.len()];
        for (l, out) in self.data.chunks(n * n).zip(data.chunks_mut(n * n)) {
            // inverse of L by forward substitution
            let mut l_inv = vec![0.0; n * n];
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

    /// Inverse of each square matrix. A matrix with an exactly zero pivot is singular, which is an error.
    pub fn inverse(&self) -> Result<Self> {
        let n = self.square_dims("inverse")?;
        let mut data = vec![0.0; self.len()];
        for (input, out) in self.data.chunks(n * n).zip(data.chunks_mut(n * n)) {
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
        let data = self
            .data
            .chunks(n * n)
            .take(shape::length(&batch))
            .map(|input| {
                let lu = Lu::factor(input, n);
                if lu.singular { 0.0 } else { lu.determinant() }
            })
            .collect();
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
        let contracted: usize = left.iter().map(|&d| self.shape[d]).product();

        let lhs = self
            .permute(&[kept_left.clone(), left.clone()].concat())?
            .reshape(&[-1, contracted as isize])?;
        let rhs = other
            .permute(&[right.clone(), kept_right.clone()].concat())?
            .reshape(&[contracted as isize, -1])?;
        let product = lhs.matmul(&rhs)?;

        let out_shape: Vec<isize> = kept_left
            .iter()
            .map(|&d| self.shape[d] as isize)
            .chain(kept_right.iter().map(|&d| other.shape[d] as isize))
            .collect();
        product.reshape(&out_shape)
    }
}

/// LU factorisation with partial pivoting, P A = L U, on one row-major n × n matrix.
struct Lu {
    n: usize,
    lu: Vec<f64>,
    /// row i was swapped with row pivots[i] at step i
    pivots: Vec<usize>,
    singular: bool,
}

impl Lu {
    fn factor(matrix: &[f64], n: usize) -> Self {
        let mut lu = matrix.to_vec();
        let mut pivots = vec![0; n];
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
            for i in k + 1..n {
                let factor = lu[i * n + k] / ukk;
                lu[i * n + k] = factor;
                if factor != 0.0 {
                    for j in k + 1..n {
                        lu[i * n + j] -= factor * lu[k * n + j];
                    }
                }
            }
        }
        Self {
            n,
            lu,
            pivots,
            singular,
        }
    }

    fn determinant(&self) -> f64 {
        let n = self.n;
        let swaps = self
            .pivots
            .iter()
            .enumerate()
            .filter(|&(i, &p)| i != p)
            .count();
        let sign = if swaps % 2 == 0 { 1.0 } else { -1.0 };
        sign * (0..n).map(|i| self.lu[i * n + i]).product::<f64>()
    }

    fn inverse_into(&self, out: &mut [f64]) {
        let n = self.n;
        let mut column = vec![0.0; n];
        for c in 0..n {
            column.iter_mut().for_each(|x| *x = 0.0);
            column[c] = 1.0;
            for (k, &p) in self.pivots.iter().enumerate() {
                column.swap(k, p);
            }
            for i in 0..n {
                let row = &self.lu[i * n..i * n + i];
                let sum = row
                    .iter()
                    .zip(&column[..i])
                    .fold(column[i], |sum, (l, x)| sum - l * x);
                column[i] = sum;
            }
            for i in (0..n).rev() {
                let row = &self.lu[i * n + i + 1..(i + 1) * n];
                let sum = row
                    .iter()
                    .zip(&column[i + 1..])
                    .fold(column[i], |sum, (u, x)| sum - u * x);
                column[i] = sum / self.lu[i * n + i];
            }
            for r in 0..n {
                out[r * n + c] = column[r];
            }
        }
    }
}
