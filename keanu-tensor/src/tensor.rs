//! The tensor type and the generic kernels every operation is built from.

use crate::error::{Result, TensorError};
use crate::shape::{self, Shape};
use std::fmt;

/// Types a tensor can hold.
pub trait Element: Copy + PartialEq + fmt::Debug + Default + Send + Sync + 'static {}
impl Element for f64 {}
impl Element for i64 {}
impl Element for bool {}

/// A dense, row-major n-dimensional array.
///
/// Every operation returns a new tensor; tensors are never modified in place by operations, so a tensor can be
/// shared freely (it is `Send + Sync`).
#[derive(Clone, PartialEq)]
pub struct Tensor<T: Element> {
    pub(crate) shape: Shape,
    pub(crate) data: Storage<T>,
}

/// Element storage. Tensors with up to four elements (every scalar, and the small vectors typical of models) are
/// stored inline, so creating them doesn't allocate.
pub(crate) type Storage<T> = smallvec::SmallVec<[T; 4]>;

/// A tensor of `f64`.
pub type DoubleTensor = Tensor<f64>;
/// A tensor of `i64`.
pub type IntTensor = Tensor<i64>;
/// A tensor of `bool`.
pub type BoolTensor = Tensor<bool>;

impl<T: Element> fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape.as_slice())
            .field("data", &self.data)
            .finish()
    }
}

impl<T: Element> Tensor<T> {
    /// Creates a tensor from row-major data.
    pub fn new(shape: &[usize], data: Vec<T>) -> Result<Self> {
        if shape::checked_length(shape) != Some(data.len()) {
            return Err(TensorError::LengthMismatch {
                shape: shape.to_vec(),
                len: data.len(),
            });
        }
        Ok(Self {
            shape: Shape::from_slice(shape),
            data: Storage::from_vec(data),
        })
    }

    pub(crate) fn from_parts(shape: Shape, data: impl Into<Storage<T>>) -> Self {
        let data = data.into();
        debug_assert_eq!(shape::length(&shape), data.len());
        Self { shape, data }
    }

    /// A tensor whose element at flat (row-major) index `i` is `f(i)`. Small tensors don't allocate.
    ///
    /// # Panics
    /// If the number of elements in `shape` overflows `usize` (as `vec!` does for an impossible allocation).
    pub fn from_fn(shape: &[usize], f: impl FnMut(usize) -> T) -> Self {
        let n = shape::length(shape);
        Self {
            shape: Shape::from_slice(shape),
            data: (0..n).map(f).collect(),
        }
    }

    /// A rank 0 tensor holding one value.
    pub fn scalar(value: T) -> Self {
        Self {
            shape: Shape::new(),
            data: smallvec::smallvec![value],
        }
    }

    /// A rank 1 tensor.
    pub fn vector(data: Vec<T>) -> Self {
        Self {
            shape: Shape::from_slice(&[data.len()]),
            data: Storage::from_vec(data),
        }
    }

    /// A tensor with every element equal to `value`.
    ///
    /// # Panics
    /// If the number of elements in `shape` overflows `usize` (as `vec!` does for an impossible allocation).
    pub fn full(shape: &[usize], value: T) -> Self {
        Self {
            shape: Shape::from_slice(shape),
            data: smallvec::smallvec![value; shape::length(shape)],
        }
    }

    /// The shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The number of dimensions.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// The number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether there are no elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// True for a tensor with exactly one element, whatever its rank.
    pub fn is_length_one(&self) -> bool {
        self.data.len() == 1
    }

    /// The elements in row-major order.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// The elements in row-major order.
    pub fn into_vec(self) -> Vec<T> {
        self.data.into_vec()
    }

    /// The only element of a length one tensor.
    pub fn to_scalar(&self) -> Option<T> {
        if self.data.len() == 1 {
            Some(self.data[0])
        } else {
            None
        }
    }

    /// The element at a multi-dimensional index.
    pub fn get(&self, index: &[usize]) -> Option<T> {
        if index.len() != self.rank() || index.iter().zip(&self.shape).any(|(&i, &d)| i >= d) {
            return None;
        }
        let strides = shape::row_major_strides(&self.shape);
        Some(
            self.data[index
                .iter()
                .zip(&strides)
                .map(|(i, s)| i * s)
                .sum::<usize>()],
        )
    }

    /// Applies a function to every element.
    pub fn map<U: Element>(&self, f: impl Fn(T) -> U) -> Tensor<U> {
        Tensor {
            shape: self.shape.clone(),
            data: self.data.iter().map(|&x| f(x)).collect(),
        }
    }

    /// Like [`map`](Self::map) but the function can fail.
    pub fn try_map<U: Element>(&self, f: impl Fn(T) -> Result<U>) -> Result<Tensor<U>> {
        Ok(Tensor {
            shape: self.shape.clone(),
            data: self.data.iter().map(|&x| f(x)).collect::<Result<_>>()?,
        })
    }

    /// Combines two tensors element-wise with numpy broadcasting.
    pub fn zip_with<U: Element, V: Element>(
        &self,
        other: &Tensor<U>,
        f: impl Fn(T, U) -> V,
    ) -> Result<Tensor<V>> {
        if self.shape == other.shape {
            let data = self
                .data
                .iter()
                .zip(&other.data)
                .map(|(&a, &b)| f(a, b))
                .collect();
            return Ok(Tensor {
                shape: self.shape.clone(),
                data,
            });
        }
        let out = shape::broadcast_shape(&self.shape, &other.shape)?;
        let data = broadcast_zip(&self.data, &self.shape, &other.data, &other.shape, &out, f);
        Ok(Tensor { shape: out, data })
    }

    /// Like [`zip_with`](Self::zip_with) but the function can fail. The error is the first in row-major order of
    /// the result; once there is one, `f` isn't called again.
    pub fn try_zip_with<U: Element, V: Element>(
        &self,
        other: &Tensor<U>,
        f: impl Fn(T, U) -> Result<V>,
    ) -> Result<Tensor<V>> {
        let error = std::cell::OnceCell::new();
        let result = self.zip_with(other, |a, b| {
            if error.get().is_some() {
                return V::default();
            }
            f(a, b).unwrap_or_else(|e| {
                let _ = error.set(e);
                V::default()
            })
        })?;
        match error.into_inner() {
            Some(e) => Err(e),
            None => Ok(result),
        }
    }

    /// Applies a function to every element, reusing this tensor's buffer.
    pub(crate) fn map_in_place(mut self, f: impl Fn(T) -> T) -> Self {
        self.data.iter_mut().for_each(|x| *x = f(*x));
        self
    }

    /// Reads the tensor with arbitrary (possibly negative or zero) strides into a new contiguous tensor.
    pub(crate) fn strided_read(
        &self,
        out_shape: &[usize],
        read_strides: &[isize],
        offset: isize,
    ) -> Self {
        Tensor {
            shape: Shape::from_slice(out_shape),
            data: gather(&self.data, out_shape, read_strides, offset),
        }
    }
}

/// Element-wise combination over a broadcast shape. The last dimension runs as a tight inner loop in which each
/// operand is either contiguous or broadcast; the outer dimensions are walked with an odometer.
pub(crate) fn broadcast_zip<A: Copy, B: Copy, C>(
    a: &[A],
    a_shape: &[usize],
    b: &[B],
    b_shape: &[usize],
    out: &[usize],
    f: impl Fn(A, B) -> C,
) -> Storage<C> {
    let n = shape::length(out);
    if n <= INLINE {
        let mut data = Storage::new();
        broadcast_zip_into(a, a_shape, b, b_shape, out, f, &mut data);
        data
    } else {
        let mut data = Vec::with_capacity(n);
        broadcast_zip_into(a, a_shape, b, b_shape, out, f, &mut data);
        Storage::from_vec(data)
    }
}

/// Number of elements stored inline in a tensor.
const INLINE: usize = 4;

/// An output buffer for the kernels: a `Vec` for large results (extending it is faster) or inline `Storage`
/// for small ones (no allocation).
trait Sink<T>: Extend<T> {
    fn push_one(&mut self, value: T);
}

impl<T> Sink<T> for Vec<T> {
    fn push_one(&mut self, value: T) {
        self.push(value);
    }
}

impl<T> Sink<T> for Storage<T> {
    fn push_one(&mut self, value: T) {
        self.push(value);
    }
}

fn broadcast_zip_into<A: Copy, B: Copy, C>(
    a: &[A],
    a_shape: &[usize],
    b: &[B],
    b_shape: &[usize],
    out: &[usize],
    f: impl Fn(A, B) -> C,
    data: &mut impl Sink<C>,
) {
    let n = shape::length(out);
    if n == 0 {
        return;
    }
    if out.is_empty() {
        data.push_one(f(a[0], b[0]));
        return;
    }
    let sa = shape::broadcast_strides(a_shape, out);
    let sb = shape::broadcast_strides(b_shape, out);
    let rank = out.len();
    let inner = out[rank - 1];
    let (step_a, step_b) = (sa[rank - 1], sb[rank - 1]);
    let outer_shape = &out[..rank - 1];
    let mut index: Shape = smallvec::smallvec![0; rank - 1];
    let (mut ia, mut ib) = (0usize, 0usize);
    for _ in 0..shape::length(outer_shape) {
        match (step_a, step_b) {
            (1, 1) => data.extend(
                a[ia..ia + inner]
                    .iter()
                    .zip(&b[ib..ib + inner])
                    .map(|(&x, &y)| f(x, y)),
            ),
            (1, 0) => {
                let y = b[ib];
                data.extend(a[ia..ia + inner].iter().map(|&x| f(x, y)));
            }
            (0, 1) => {
                let x = a[ia];
                data.extend(b[ib..ib + inner].iter().map(|&y| f(x, y)));
            }
            _ => {
                let (x, y) = (a[ia], b[ib]);
                data.extend((0..inner).map(|_| f(x, y)));
            }
        }
        for d in (0..rank - 1).rev() {
            index[d] += 1;
            ia += sa[d];
            ib += sb[d];
            if index[d] < outer_shape[d] {
                break;
            }
            ia -= sa[d] * outer_shape[d];
            ib -= sb[d] * outer_shape[d];
            index[d] = 0;
        }
    }
}

/// Copies `src` read with arbitrary strides into a contiguous row-major buffer, with the last dimension as a
/// tight inner loop.
pub(crate) fn gather<T: Copy>(
    src: &[T],
    shape: &[usize],
    read: &[isize],
    offset: isize,
) -> Storage<T> {
    let n = shape::length(shape);
    if n <= INLINE {
        let mut out = Storage::new();
        gather_into(src, shape, read, offset, &mut out);
        out
    } else {
        let mut out = Vec::with_capacity(n);
        gather_into(src, shape, read, offset, &mut out);
        Storage::from_vec(out)
    }
}

fn gather_into<T: Copy>(
    src: &[T],
    shape: &[usize],
    read: &[isize],
    offset: isize,
    out: &mut impl Sink<T>,
) {
    let n = shape::length(shape);
    if n == 0 {
        return;
    }
    if shape.is_empty() {
        out.push_one(src[offset as usize]);
        return;
    }
    let rank = shape.len();
    let inner = shape[rank - 1];
    let step = read[rank - 1];
    let outer_shape = &shape[..rank - 1];
    let mut index: Shape = smallvec::smallvec![0; rank - 1];
    let mut position = offset;
    for _ in 0..shape::length(outer_shape) {
        if step == 1 {
            let start = position as usize;
            out.extend(src[start..start + inner].iter().copied());
        } else {
            let mut p = position;
            for _ in 0..inner {
                out.push_one(src[p as usize]);
                p += step;
            }
        }
        for d in (0..rank - 1).rev() {
            index[d] += 1;
            position += read[d];
            if index[d] < outer_shape[d] {
                break;
            }
            position -= read[d] * outer_shape[d] as isize;
            index[d] = 0;
        }
    }
}

/// Reduces over the dimensions marked in `reduce`, removing them. `init` starts each output element and `fold`
/// accumulates input elements into it.
pub(crate) fn reduce<T: Copy, A: Copy>(
    data: &[T],
    shape: &[usize],
    reduce: &[bool],
    init: A,
    fold: impl Fn(A, T) -> A,
) -> (Shape, Storage<A>) {
    let rank = shape.len();
    // the reduced dimensions kept with length 1, to walk the output alongside the input
    let kept_shape: Shape = (0..rank)
        .map(|i| if reduce[i] { 1 } else { shape[i] })
        .collect();
    let out_shape: Shape = (0..rank)
        .filter(|&i| !reduce[i])
        .map(|i| shape[i])
        .collect();
    let mut out = smallvec::smallvec![init; shape::length(&kept_shape)];
    if data.is_empty() {
        return (out_shape, out);
    }
    if rank == 0 {
        out[0] = fold(init, data[0]);
        return (out_shape, out);
    }
    let out_strides = shape::broadcast_strides(&kept_shape, shape);
    let inner = shape[rank - 1];
    let reduce_last = reduce[rank - 1];
    let outer_shape = &shape[..rank - 1];
    let mut index: Shape = smallvec::smallvec![0; rank - 1];
    let mut o = 0usize;
    for row in data.chunks_exact(inner.max(1)) {
        if inner == 0 {
            break;
        }
        if reduce_last {
            out[o] = row.iter().fold(out[o], |acc, &x| fold(acc, x));
        } else {
            for (acc, &x) in out[o..o + inner].iter_mut().zip(row) {
                *acc = fold(*acc, x);
            }
        }
        for d in (0..rank - 1).rev() {
            index[d] += 1;
            o += out_strides[d];
            if index[d] < outer_shape[d] {
                break;
            }
            o -= out_strides[d] * outer_shape[d];
            index[d] = 0;
        }
    }
    (out_shape, out)
}

/// Applies a running accumulation (e.g. cumulative sum) along one axis.
pub(crate) fn scan<T: Copy>(
    data: &[T],
    shape: &[usize],
    axis: usize,
    f: impl Fn(T, T) -> T,
) -> Storage<T> {
    let mut out = Storage::from_slice(data);
    let stride = shape::row_major_strides(shape)[axis];
    let n = shape[axis];
    let block = stride * n;
    if block == 0 {
        return out;
    }
    for start in (0..out.len()).step_by(block) {
        for k in 1..n {
            let (prev, cur) = (start + (k - 1) * stride, start + k * stride);
            for j in 0..stride {
                out[cur + j] = f(out[prev + j], out[cur + j]);
            }
        }
    }
    out
}

/// A scalar tensor holding the value.
impl<T: Element> From<T> for Tensor<T> {
    fn from(value: T) -> Self {
        Self::scalar(value)
    }
}
