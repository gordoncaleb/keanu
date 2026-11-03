//! Spike A: a tensor built on `ndarray::ArrayD<f64>`, using ndarray's co-broadcasting arithmetic, axis
//! reductions, views and `dot`. Slicing reuses the numpy semantics from the in-house spike's `slicing` module
//! and translates them to ndarray slices, whose negative-step semantics differ from numpy's.

use ndarray::{ArrayD, Axis, IxDyn, Slice, SliceInfoElem};
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error(msg.into()))
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor(pub ArrayD<f64>);

fn normalise_axis(axis: isize, rank: usize) -> Result<usize> {
    let a = if axis < 0 { axis + rank as isize } else { axis };
    if a < 0 || a as usize >= rank {
        return err(format!("axis {axis} out of range for rank {rank}"));
    }
    Ok(a as usize)
}

impl Tensor {
    pub fn new(shape: &[usize], data: Vec<f64>) -> Result<Self> {
        ArrayD::from_shape_vec(IxDyn(shape), data).map(Tensor).map_err(|e| Error(e.to_string()))
    }

    pub fn scalar(value: f64) -> Self {
        Tensor(ArrayD::from_elem(IxDyn(&[]), value))
    }

    pub fn shape(&self) -> &[usize] {
        self.0.shape()
    }

    /// Row-major copy of the data (ndarray arrays may be views with arbitrary layout).
    pub fn data(&self) -> Vec<f64> {
        self.0.iter().copied().collect()
    }

    pub fn map(&self, f: impl Fn(f64) -> f64) -> Self {
        Tensor(self.0.mapv(f))
    }

    pub fn zip(&self, other: &Self, f: impl Fn(f64, f64) -> f64) -> Result<Self> {
        let shape = keanu_tensor_inhouse::broadcast_shape(self.shape(), other.shape()).map_err(|e| Error(e.0))?;
        let a = self.0.broadcast(IxDyn(&shape)).ok_or_else(|| Error("broadcast".into()))?;
        let b = other.0.broadcast(IxDyn(&shape)).ok_or_else(|| Error("broadcast".into()))?;
        Ok(Tensor(ndarray::Zip::from(&a).and(&b).map_collect(|&x, &y| f(x, y))))
    }

    pub fn plus(&self, o: &Self) -> Result<Self> {
        // ndarray's own co-broadcasting operator; errors (panics) on incompatible shapes, so check first
        keanu_tensor_inhouse::broadcast_shape(self.shape(), o.shape()).map_err(|e| Error(e.0))?;
        Ok(Tensor(&self.0 + &o.0))
    }
    pub fn times(&self, o: &Self) -> Result<Self> {
        keanu_tensor_inhouse::broadcast_shape(self.shape(), o.shape()).map_err(|e| Error(e.0))?;
        Ok(Tensor(&self.0 * &o.0))
    }
    pub fn div(&self, o: &Self) -> Result<Self> {
        keanu_tensor_inhouse::broadcast_shape(self.shape(), o.shape()).map_err(|e| Error(e.0))?;
        Ok(Tensor(&self.0 / &o.0))
    }
    pub fn pow(&self, o: &Self) -> Result<Self> {
        self.zip(o, f64::powf)
    }
    pub fn atan2(&self, y: &Self) -> Result<Self> {
        self.zip(y, |x, y| y.atan2(x))
    }

    pub fn sum(&self) -> Self {
        Self::scalar(self.0.sum())
    }

    pub fn sum_dims(&self, dims: &[isize]) -> Result<Self> {
        let rank = self.0.ndim();
        let mut axes = dims.iter().map(|&d| normalise_axis(d, rank)).collect::<Result<Vec<_>>>()?;
        axes.sort_unstable();
        axes.dedup();
        let mut out = self.0.clone();
        for &a in axes.iter().rev() {
            out = out.sum_axis(Axis(a));
        }
        Ok(Tensor(out))
    }

    pub fn cum_sum(&self, axis: isize) -> Result<Self> {
        let axis = normalise_axis(axis, self.0.ndim())?;
        let mut out = self.0.clone();
        out.accumulate_axis_inplace(Axis(axis), |&prev, curr| *curr += prev);
        Ok(Tensor(out))
    }

    pub fn arg_max(&self, axis: isize) -> Result<(Vec<usize>, Vec<i64>)> {
        let axis = normalise_axis(axis, self.0.ndim())?;
        let result = self.0.map_axis(Axis(axis), |lane| {
            let mut best = 0usize;
            let mut best_value = lane[0];
            for (k, &v) in lane.iter().enumerate().skip(1) {
                if !best_value.is_nan() && (v.is_nan() || v > best_value) {
                    best = k;
                    best_value = v;
                }
            }
            best as i64
        });
        Ok((result.shape().to_vec(), result.iter().copied().collect()))
    }

    pub fn reshape(&self, shape: &[i64]) -> Result<Self> {
        // ndarray has no -1 wildcard, so resolve it the same way as the in-house spike
        let resolved = keanu_tensor_inhouse::Tensor::new(&[self.0.len()], vec![0.0; self.0.len()])
            .and_then(|t| t.reshape(shape))
            .map_err(|e| Error(e.0))?;
        let data: Vec<f64> = self.0.iter().copied().collect();
        Self::new(resolved.shape(), data)
    }

    pub fn permute(&self, permutation: &[usize]) -> Result<Self> {
        let rank = self.0.ndim();
        let mut seen = vec![false; rank];
        if permutation.len() != rank || permutation.iter().any(|&p| p >= rank || std::mem::replace(&mut seen[p], true)) {
            return err(format!("invalid permutation {permutation:?} for rank {rank}"));
        }
        Ok(Tensor(self.0.view().permuted_axes(IxDyn(permutation)).as_standard_layout().into_owned()))
    }

    pub fn slice(&self, spec: &str) -> Result<Self> {
        let plan = keanu_tensor_inhouse::slicing::plan(spec, self.shape()).map_err(|e| Error(e.0))?;
        let elems: Vec<SliceInfoElem> = plan
            .iter()
            .map(|s| {
                if s.drop {
                    SliceInfoElem::Index(s.start)
                } else if s.len == 0 {
                    SliceInfoElem::from(Slice::new(0, Some(0), 1))
                } else if s.step > 0 {
                    let last = s.start + (s.len as isize - 1) * s.step;
                    SliceInfoElem::from(Slice::new(s.start, Some(last + 1), s.step))
                } else {
                    // ndarray: a negative step walks a forward range from its end, numpy walks from start
                    let lowest = s.start + (s.len as isize - 1) * s.step;
                    SliceInfoElem::from(Slice::new(lowest, Some(s.start + 1), s.step))
                }
            })
            .collect();
        let view = self.0.slice(elems.as_slice());
        Ok(Tensor(view.as_standard_layout().into_owned()))
    }

    pub fn matmul(&self, other: &Self, transpose_left: bool, transpose_right: bool) -> Result<Self> {
        let (a, b) = (self.shape(), other.shape());
        if a.len() < 2 || b.len() < 2 {
            return err("matrix multiply needs rank >= 2");
        }
        let batch_a = &a[..a.len() - 2];
        let batch_b = &b[..b.len() - 2];
        let batch = keanu_tensor_inhouse::broadcast_shape(batch_a, batch_b).map_err(|e| Error(e.0))?;
        let (ar, ac) = (a[a.len() - 2], a[a.len() - 1]);
        let (br, bc) = (b[b.len() - 2], b[b.len() - 1]);
        let (m, k) = if transpose_left { (ac, ar) } else { (ar, ac) };
        let (k2, n) = if transpose_right { (bc, br) } else { (br, bc) };
        if k != k2 {
            return err(format!("cannot matrix multiply {a:?} and {b:?}"));
        }
        let mut lhs_shape = batch.clone();
        lhs_shape.extend([ar, ac]);
        let mut rhs_shape = batch.clone();
        rhs_shape.extend([br, bc]);
        let lhs = self.0.broadcast(IxDyn(&lhs_shape)).ok_or_else(|| Error("broadcast".into()))?;
        let rhs = other.0.broadcast(IxDyn(&rhs_shape)).ok_or_else(|| Error("broadcast".into()))?;
        let batch_len: usize = batch.iter().product();
        let lhs = lhs.to_shape(IxDyn(&[batch_len, ar, ac])).map_err(|e| Error(e.to_string()))?;
        let rhs = rhs.to_shape(IxDyn(&[batch_len, br, bc])).map_err(|e| Error(e.to_string()))?;
        let mut out_shape = batch;
        out_shape.extend([m, n]);
        let mut out = ArrayD::<f64>::zeros(IxDyn(&[batch_len, m, n]));
        for i in 0..batch_len {
            let l = lhs.index_axis(Axis(0), i).into_dimensionality::<ndarray::Ix2>().unwrap();
            let r = rhs.index_axis(Axis(0), i).into_dimensionality::<ndarray::Ix2>().unwrap();
            let l = if transpose_left { l.reversed_axes() } else { l };
            let r = if transpose_right { r.reversed_axes() } else { r };
            out.index_axis_mut(Axis(0), i).assign(&l.dot(&r).into_dyn());
        }
        Ok(Tensor(out.into_shape_with_order(IxDyn(&out_shape)).map_err(|e| Error(e.to_string()))?))
    }

    pub fn cholesky(&self) -> Result<Self> {
        // ndarray has no decompositions without a LAPACK backend, so this goes through faer as in the in-house spike
        let s = self.shape().to_vec();
        if s.len() < 2 || s[s.len() - 1] != s[s.len() - 2] {
            return err("cholesky needs square matrices");
        }
        let data = self.data();
        keanu_tensor_inhouse::Tensor::new(&s, data)
            .and_then(|t| t.cholesky())
            .map_err(|e| Error(e.0))
            .and_then(|t| Self::new(&s, t.data().to_vec()))
    }
}
