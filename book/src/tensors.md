# Tensors

Every node's value is a `Tensor`: a dense, row-major n-dimensional array of `f64` (`DoubleTensor`), `i64`
(`IntTensor`) or `bool` (`BoolTensor`). A scalar is a tensor of rank 0.

The semantics follow numpy:

- **Element-wise operations broadcast.** A `[2, 3]` tensor plus a `[3]` tensor adds the vector to each row.
- **Axes can be negative**, counting from the end.
- **Slicing uses numpy syntax**, including negative steps and `...`.
- **Linear algebra works on the last two dimensions** and broadcasts the others, so a `[10, 3, 3]` tensor is a
  batch of ten matrices.

```rust,ignore
{{#include ../../keanu/tests/book.rs:tensors}}
```

Operations that can fail (incompatible shapes, an invalid axis, a singular matrix, integer division by zero) return
`Result`; operations with a scalar, which can't fail, use the standard operators.

Some details worth knowing:

- `max`, `min` and their element-wise forms propagate NaN, and treat −0.0 as smaller than +0.0.
- `argmax` / `argmin` return the first NaN if there is one; `nan_argmax` / `nan_argmin` ignore NaN.
- Integer arithmetic wraps on overflow, and integer division truncates towards zero.
- `tril(k)` / `triu(k)` use numpy's convention for `k`.
- `solve(b)` solves A X = B by LU with partial pivoting, and `solve_triangular(b, lower, transpose)` solves with a
  triangular matrix (such as a Cholesky factor) reading only that triangle; both take a matrix `[..., n, k]` or a
  vector `[n]` right-hand side and broadcast leading dimensions. A singular matrix is a `SingularMatrix` error, as
  for `inverse`.
- `slogdet()` returns numpy's (sign, ln |det|), which stays finite where `determinant()` would overflow; a singular
  matrix has sign 0 and ln |det| −∞.
- `eigh()` returns the eigenvalues (ascending) and eigenvectors (columns, each signed so its first component within a
  relative 1e-10 of the largest magnitude is positive) of symmetric matrices, reading only the lower triangle. The
  eigenvectors are discontinuous where that sign convention switches and at repeated eigenvalues.
- `SparseMatrix` holds a constant matrix with mostly zero entries in compressed sparse row form, built from
  `(row, column, value)` triplets or a dense matrix. `matvec`, `matmat` and their transposed forms cost time
  proportional to the stored entries; in a model, `x.sparse_matmul(a)` multiplies a node by it, with exact first and
  second derivatives (see [Spatial models](spatial-models.md)).
