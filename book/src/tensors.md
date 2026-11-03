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
