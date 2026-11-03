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
    pub(crate) data: Vec<T>,
}

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
        if shape::length(shape) != data.len() {
            return Err(TensorError::LengthMismatch {
                shape: shape.to_vec(),
                len: data.len(),
            });
        }
        Ok(Self {
            shape: Shape::from_slice(shape),
            data,
        })
    }

    pub(crate) fn from_parts(shape: Shape, data: Vec<T>) -> Self {
        debug_assert_eq!(shape::length(&shape), data.len());
        Self { shape, data }
    }

    /// A rank 0 tensor holding one value.
    pub fn scalar(value: T) -> Self {
        Self {
            shape: Shape::new(),
            data: vec![value],
        }
    }

    /// A rank 1 tensor.
    pub fn vector(data: Vec<T>) -> Self {
        Self {
            shape: Shape::from_slice(&[data.len()]),
            data,
        }
    }

    /// A tensor with every element equal to `value`.
    pub fn full(shape: &[usize], value: T) -> Self {
        Self {
            shape: Shape::from_slice(shape),
            data: vec![value; shape::length(shape)],
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// The number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

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

    pub fn into_vec(self) -> Vec<T> {
        self.data
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

    /// Like [`zip_with`](Self::zip_with) but the function can fail.
    pub fn try_zip_with<U: Element, V: Element>(
        &self,
        other: &Tensor<U>,
        f: impl Fn(T, U) -> Result<V>,
    ) -> Result<Tensor<V>> {
        let out = shape::broadcast_shape(&self.shape, &other.shape)?;
        let results = broadcast_zip(&self.data, &self.shape, &other.data, &other.shape, &out, f);
        Ok(Tensor {
            shape: out,
            data: results.into_iter().collect::<Result<_>>()?,
        })
    }

    /// Reads the tensor with arbitrary (possibly negative or zero) strides into a new contiguous tensor.
    pub(crate) fn gather(
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
) -> Vec<C> {
    let n = shape::length(out);
    let mut data = Vec::with_capacity(n);
    if n == 0 {
        return data;
    }
    if out.is_empty() {
        data.push(f(a[0], b[0]));
        return data;
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
    data
}

/// Copies `src` read with arbitrary strides into a contiguous row-major buffer, with the last dimension as a
/// tight inner loop.
pub(crate) fn gather<T: Copy>(src: &[T], shape: &[usize], read: &[isize], offset: isize) -> Vec<T> {
    let n = shape::length(shape);
    let mut out = Vec::with_capacity(n);
    if n == 0 {
        return out;
    }
    if shape.is_empty() {
        out.push(src[offset as usize]);
        return out;
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
            out.extend_from_slice(&src[start..start + inner]);
        } else {
            let mut p = position;
            for _ in 0..inner {
                out.push(src[p as usize]);
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
    out
}

/// Reduces over the dimensions marked in `reduce`, removing them (or keeping them with length 1 if `keep_dims`).
/// `init` starts each output element and `fold` accumulates input elements into it.
pub(crate) fn reduce<T: Copy, A: Copy>(
    data: &[T],
    shape: &[usize],
    reduce: &[bool],
    keep_dims: bool,
    init: A,
    fold: impl Fn(A, T) -> A,
) -> (Shape, Vec<A>) {
    let rank = shape.len();
    let kept_shape: Shape = (0..rank)
        .map(|i| if reduce[i] { 1 } else { shape[i] })
        .collect();
    let out_shape: Shape = if keep_dims {
        kept_shape.clone()
    } else {
        (0..rank)
            .filter(|&i| !reduce[i])
            .map(|i| shape[i])
            .collect()
    };
    let mut out = vec![init; shape::length(&kept_shape)];
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
) -> Vec<T> {
    let mut out = data.to_vec();
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
