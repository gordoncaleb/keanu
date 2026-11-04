//! Reductions and scans.

use crate::error::{Result, TensorError};
use crate::ops::elementwise::Numeric;
use crate::shape;
use crate::tensor::{DoubleTensor, IntTensor, Tensor, reduce, scan};

impl<T: Numeric> Tensor<T> {
    /// Sum of every element.
    pub fn sum(&self) -> T {
        self.data.iter().fold(T::ZERO, |a, &b| a.add(b))
    }

    /// Sums over the given axes (negative counts from the end), removing them.
    pub fn sum_axes(&self, axes: &[isize]) -> Result<Self> {
        let mask = shape::axis_mask(axes, self.rank())?;
        let (s, data) = reduce(&self.data, &self.shape, &mask, false, T::ZERO, T::add);
        Ok(Tensor::from_parts(s, data))
    }

    /// Product of every element.
    pub fn product(&self) -> T {
        self.data.iter().fold(T::ONE, |a, &b| a.mul(b))
    }

    /// The product over the given axes, which are removed.
    pub fn product_axes(&self, axes: &[isize]) -> Result<Self> {
        let mask = shape::axis_mask(axes, self.rank())?;
        let (s, data) = reduce(&self.data, &self.shape, &mask, false, T::ONE, T::mul);
        Ok(Tensor::from_parts(s, data))
    }

    /// The largest element (NaN if any element is NaN), or None if empty.
    pub fn max(&self) -> Option<T> {
        self.data.iter().copied().reduce(T::max)
    }

    /// The smallest element (NaN if any element is NaN), or None if empty.
    pub fn min(&self) -> Option<T> {
        self.data.iter().copied().reduce(T::min)
    }

    /// Flat index of the largest element; the first NaN wins. None if empty.
    pub fn argmax(&self) -> Option<usize> {
        arg_best(&self.data, |v, best| v > best)
    }

    /// Flat index of the smallest element; the first NaN wins. None if empty.
    pub fn argmin(&self) -> Option<usize> {
        arg_best(&self.data, |v, best| v < best)
    }

    /// Index of the largest element along an axis; the first NaN wins. Fails if the axis is empty.
    pub fn argmax_axis(&self, axis: isize) -> Result<IntTensor> {
        self.arg_best_axis(axis, |v, best| v > best)
    }

    /// Index of the smallest element along an axis; the first NaN wins. Fails if the axis is empty.
    pub fn argmin_axis(&self, axis: isize) -> Result<IntTensor> {
        self.arg_best_axis(axis, |v, best| v < best)
    }

    fn arg_best_axis(&self, axis: isize, better: impl Fn(T, T) -> bool) -> Result<IntTensor> {
        let axis = shape::axis(axis, self.rank())?;
        let n = self.shape[axis];
        if n == 0 {
            return Err(TensorError::InvalidArgument(format!(
                "cannot take the argmax or argmin along empty axis {axis} of shape {:?}",
                self.shape()
            )));
        }
        let inner = shape::row_major_strides(&self.shape)[axis];
        let outer = shape::length(&self.shape[..axis]);
        let out_shape: shape::Shape = self
            .shape
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != axis)
            .map(|(_, &d)| d)
            .collect();
        let mut out = Vec::with_capacity(outer * inner);
        for o in 0..outer {
            for i in 0..inner {
                let base = o * n * inner + i;
                let lane = (0..n).map(|k| self.data[base + k * inner]);
                let best =
                    arg_best_iter(lane, &better).expect("a non-empty lane has a best element");
                out.push(best as i64);
            }
        }
        Ok(Tensor::from_parts(out_shape, out))
    }

    /// Cumulative sum along an axis.
    pub fn cumsum(&self, axis: isize) -> Result<Self> {
        let axis = shape::axis(axis, self.rank())?;
        Ok(Tensor::from_parts(
            self.shape.clone(),
            scan(&self.data, &self.shape, axis, T::add),
        ))
    }

    /// Cumulative product along an axis.
    pub fn cumprod(&self, axis: isize) -> Result<Self> {
        let axis = shape::axis(axis, self.rank())?;
        Ok(Tensor::from_parts(
            self.shape.clone(),
            scan(&self.data, &self.shape, axis, T::mul),
        ))
    }
}

fn arg_best<T: Numeric>(data: &[T], better: impl Fn(T, T) -> bool) -> Option<usize> {
    arg_best_iter(data.iter().copied(), &better)
}

/// Index of the best element: the first NaN if there is one, otherwise the first element that is better than
/// every element before it.
fn arg_best_iter<T: Numeric>(
    values: impl Iterator<Item = T>,
    better: &impl Fn(T, T) -> bool,
) -> Option<usize> {
    let mut best: Option<(usize, T)> = None;
    for (i, v) in values.enumerate() {
        match best {
            None => best = Some((i, v)),
            Some((_, b)) if !b.is_nan() && (v.is_nan() || better(v, b)) => best = Some((i, v)),
            _ => {}
        }
    }
    best.map(|(i, _)| i)
}

impl DoubleTensor {
    /// Mean of every element (NaN if empty).
    pub fn mean(&self) -> f64 {
        self.sum() / self.len() as f64
    }

    /// Mean over the given axes, removing them.
    pub fn mean_axes(&self, axes: &[isize]) -> Result<Self> {
        let mask = shape::axis_mask(axes, self.rank())?;
        let count: usize = self
            .shape
            .iter()
            .zip(&mask)
            .filter(|&(_, &m)| m)
            .map(|(&d, _)| d)
            .product();
        Ok(self.sum_axes(axes)?.map(|x| x / count as f64))
    }

    /// Sample standard deviation of every element (n - 1 denominator).
    pub fn std_dev(&self) -> f64 {
        let n = self.len() as f64;
        if self.len() < 2 {
            return if self.len() == 1 { 0.0 } else { f64::NAN };
        }
        let mean = self.mean();
        let sum_sq: f64 = self.data.iter().map(|&x| (x - mean) * (x - mean)).sum();
        (sum_sq / (n - 1.0)).sqrt()
    }

    /// (x - mean) / std_dev over every element.
    pub fn standardize(&self) -> Self {
        let (mean, sd) = (self.mean(), self.std_dev());
        self.map(|x| (x - mean) / sd)
    }

    /// Flat index of the largest element, ignoring NaN (0 if every element is NaN). None if empty.
    pub fn nan_argmax(&self) -> Option<usize> {
        nan_arg_best(&self.data, |v, b| v > b)
    }

    /// Flat index of the smallest element, ignoring NaN (0 if every element is NaN). None if empty.
    pub fn nan_argmin(&self) -> Option<usize> {
        nan_arg_best(&self.data, |v, b| v < b)
    }
}

fn nan_arg_best(data: &[f64], better: impl Fn(f64, f64) -> bool) -> Option<usize> {
    if data.is_empty() {
        return None;
    }
    let mut best: Option<(usize, f64)> = None;
    for (i, &v) in data.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        match best {
            Some((_, b)) if !better(v, b) => {}
            _ => best = Some((i, v)),
        }
    }
    Some(best.map(|(i, _)| i).unwrap_or(0))
}
