//! Dense n-dimensional tensors for [Keanu](https://github.com/gordoncaleb/keanu).
//!
//! A [`Tensor<T>`] is a row-major array of `f64` ([`DoubleTensor`]), `i64` ([`IntTensor`]) or `bool`
//! ([`BoolTensor`]). Operations follow numpy conventions: element-wise operations broadcast, axes may be negative
//! (counting from the end), and slicing uses numpy syntax.
//!
//! Operations that can fail (incompatible shapes, invalid axes, singular matrices, ...) return
//! [`Result`]; operations between a tensor and a scalar use the standard operators.
//!
//! ```
//! use keanu_tensor::DoubleTensor;
//!
//! let a = DoubleTensor::new(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
//! let b = DoubleTensor::vector(vec![10.0, 20.0, 30.0]);
//! let c = a.add(&b)?;                // broadcasts b across the rows of a
//! assert_eq!(c.as_slice(), &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
//! assert_eq!(c.sum_axes(&[-1])?.as_slice(), &[66.0, 75.0]);
//! assert_eq!(a.slice("::-1, 0")?.as_slice(), &[4.0, 1.0]);
//! # Ok::<(), keanu_tensor::TensorError>(())
//! ```
//!
//! Semantics worth knowing:
//! - `max`/`min` (element-wise and reductions) propagate NaN, and treat -0.0 as smaller than +0.0.
//! - `argmax`/`argmin` return the first NaN if there is one; `nan_argmax`/`nan_argmin` ignore NaN.
//! - Integer arithmetic wraps on overflow; integer division truncates towards zero and division by zero is an error.
//! - Integer `pow` with a negative exponent truncates 1 / base^|exponent| towards zero.

#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

mod error;
/// The tensor operations, grouped by kind. They are all methods on [`Tensor`].
pub mod ops;
mod shape;
mod sparse;
pub mod special;
mod tensor;

pub use error::{Result, TensorError};
pub use ops::elementwise::{Numeric, int_pow};
pub use ops::linalg::LuFactorization;
pub use shape::{Shape, broadcast_shape};
pub use sparse::SparseMatrix;
pub use tensor::{BoolTensor, DoubleTensor, Element, IntTensor, Tensor};
