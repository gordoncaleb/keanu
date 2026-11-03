//! Spike B: an in-house tensor. Contiguous row-major `Vec<f64>` plus a shape; every operation produces a new
//! contiguous tensor. Broadcasting walks the output with an odometer over per-operand strides (0 on broadcast
//! dimensions). Linear algebra goes through `faer` on row-major slices.

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
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f64>,
}

pub fn length(shape: &[usize]) -> usize {
    shape.iter().product()
}

pub fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    let mut acc = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc *= shape[i];
    }
    strides
}

pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut out = vec![0; rank];
    for i in 0..rank {
        let da = if i + a.len() >= rank { a[i + a.len() - rank] } else { 1 };
        let db = if i + b.len() >= rank { b[i + b.len() - rank] } else { 1 };
        out[i] = if da == db || db == 1 {
            da
        } else if da == 1 {
            db
        } else {
            return err(format!("cannot broadcast {a:?} with {b:?}"));
        };
    }
    Ok(out)
}

/// Strides for reading `shape` as if it had been broadcast to `out` (0 on broadcast dimensions).
fn broadcast_strides(shape: &[usize], out: &[usize]) -> Vec<usize> {
    let strides = row_major_strides(shape);
    let offset = out.len() - shape.len();
    (0..out.len())
        .map(|i| if i < offset || shape[i - offset] == 1 { 0 } else { strides[i - offset] })
        .collect()
}

fn normalise_axis(axis: isize, rank: usize) -> Result<usize> {
    let a = if axis < 0 { axis + rank as isize } else { axis };
    if a < 0 || a as usize >= rank {
        return err(format!("axis {axis} out of range for rank {rank}"));
    }
    Ok(a as usize)
}

impl Tensor {
    pub fn new(shape: &[usize], data: Vec<f64>) -> Result<Self> {
        if length(shape) != data.len() {
            return err(format!("shape {shape:?} needs {} values, got {}", length(shape), data.len()));
        }
        Ok(Self { shape: shape.to_vec(), data })
    }

    pub fn scalar(value: f64) -> Self {
        Self { shape: vec![], data: vec![value] }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[f64] {
        &self.data
    }

    pub fn map(&self, f: impl Fn(f64) -> f64) -> Self {
        Self { shape: self.shape.clone(), data: self.data.iter().map(|&x| f(x)).collect() }
    }

    /// Element-wise binary operation with numpy broadcasting.
    pub fn zip(&self, other: &Self, f: impl Fn(f64, f64) -> f64) -> Result<Self> {
        if self.shape == other.shape {
            let data = self.data.iter().zip(&other.data).map(|(&a, &b)| f(a, b)).collect();
            return Ok(Self { shape: self.shape.clone(), data });
        }
        let out = broadcast_shape(&self.shape, &other.shape)?;
        let sa = broadcast_strides(&self.shape, &out);
        let sb = broadcast_strides(&other.shape, &out);
        let n = length(&out);
        let mut data = Vec::with_capacity(n);
        if n == 0 {
            return Ok(Self { shape: out, data });
        }
        // Walk the outer dimensions with an odometer and run the last dimension as a tight inner loop:
        // each operand is either contiguous (stride 1) or broadcast (stride 0) along it.
        let rank = out.len();
        let inner = out[rank - 1];
        let (ia_step, ib_step) = (sa[rank - 1], sb[rank - 1]);
        let outer_shape = &out[..rank - 1];
        let outer = length(outer_shape);
        let mut index = vec![0usize; rank - 1];
        let (mut ia, mut ib) = (0usize, 0usize);
        for _ in 0..outer {
            match (ia_step, ib_step) {
                (1, 1) => data.extend(self.data[ia..ia + inner].iter().zip(&other.data[ib..ib + inner]).map(|(&a, &b)| f(a, b))),
                (1, 0) => {
                    let b = other.data[ib];
                    data.extend(self.data[ia..ia + inner].iter().map(|&a| f(a, b)));
                }
                (0, 1) => {
                    let a = self.data[ia];
                    data.extend(other.data[ib..ib + inner].iter().map(|&b| f(a, b)));
                }
                _ => {
                    let v = f(self.data[ia], other.data[ib]);
                    data.extend(std::iter::repeat_n(v, inner));
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
        Ok(Self { shape: out, data })
    }

    pub fn plus(&self, o: &Self) -> Result<Self> {
        self.zip(o, |a, b| a + b)
    }
    pub fn times(&self, o: &Self) -> Result<Self> {
        self.zip(o, |a, b| a * b)
    }
    pub fn div(&self, o: &Self) -> Result<Self> {
        self.zip(o, |a, b| a / b)
    }
    pub fn pow(&self, o: &Self) -> Result<Self> {
        self.zip(o, f64::powf)
    }
    /// Element-wise atan2(y, self), as in Keanu.
    pub fn atan2(&self, y: &Self) -> Result<Self> {
        self.zip(y, |x, y| y.atan2(x))
    }

    pub fn sum(&self) -> Self {
        Self::scalar(self.data.iter().sum())
    }

    /// Sums over the given dimensions (negative counts from the end), removing them.
    pub fn sum_dims(&self, dims: &[isize]) -> Result<Self> {
        let rank = self.shape.len();
        let mut reduce = vec![false; rank];
        for &d in dims {
            reduce[normalise_axis(d, rank)?] = true;
        }
        let out_shape: Vec<usize> = (0..rank).filter(|&i| !reduce[i]).map(|i| self.shape[i]).collect();
        let kept_shape: Vec<usize> = (0..rank).map(|i| if reduce[i] { 1 } else { self.shape[i] }).collect();
        let out_strides = broadcast_strides(&kept_shape, &self.shape);
        let mut data = vec![0.0; length(&out_shape)];
        if self.data.is_empty() || rank == 0 {
            if rank == 0 {
                data[0] = self.data[0];
            }
            return Ok(Self { shape: out_shape, data });
        }
        // the last dimension runs as an inner loop: either accumulated into one output (reduced) or added
        // element-wise into a contiguous output row (kept)
        let inner = self.shape[rank - 1];
        let reduce_last = reduce[rank - 1];
        let outer_shape = &self.shape[..rank - 1];
        let mut index = vec![0usize; rank - 1];
        let mut o = 0usize;
        for row in self.data.chunks_exact(inner) {
            if reduce_last {
                data[o] += row.iter().sum::<f64>();
            } else {
                for (acc, &x) in data[o..o + inner].iter_mut().zip(row) {
                    *acc += x;
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
        Ok(Self { shape: out_shape, data })
    }

    pub fn cum_sum(&self, axis: isize) -> Result<Self> {
        let axis = normalise_axis(axis, self.shape.len())?;
        let stride = row_major_strides(&self.shape)[axis];
        let n = self.shape[axis];
        let mut data = self.data.clone();
        for start in 0..data.len() {
            let position = (start / stride) % n;
            if position > 0 {
                data[start] += data[start - stride];
            }
        }
        Ok(Self { shape: self.shape.clone(), data })
    }

    /// Index of the maximum along an axis; the first NaN wins, as in numpy.
    pub fn arg_max(&self, axis: isize) -> Result<(Vec<usize>, Vec<i64>)> {
        let axis = normalise_axis(axis, self.shape.len())?;
        let strides = row_major_strides(&self.shape);
        let n = self.shape[axis];
        let out_shape: Vec<usize> = self.shape.iter().enumerate().filter(|&(i, _)| i != axis).map(|(_, &d)| d).collect();
        let outer = length(&self.shape[..axis]);
        let inner = strides[axis];
        let mut out = Vec::with_capacity(outer * inner);
        for o in 0..outer {
            for i in 0..inner {
                let base = o * n * inner + i;
                let mut best = 0usize;
                let mut best_value = self.data[base];
                for k in 1..n {
                    let v = self.data[base + k * inner];
                    if !best_value.is_nan() && (v.is_nan() || v > best_value) {
                        best = k;
                        best_value = v;
                    }
                }
                out.push(best as i64);
            }
        }
        Ok((out_shape, out))
    }

    /// Reshape with at most one -1 wildcard, which must divide evenly.
    pub fn reshape(&self, shape: &[i64]) -> Result<Self> {
        let wildcards = shape.iter().filter(|&&d| d == -1).count();
        if wildcards > 1 || shape.iter().any(|&d| d < -1) {
            return err(format!("invalid shape {shape:?}"));
        }
        let known: usize = shape.iter().filter(|&&d| d >= 0).map(|&d| d as usize).product();
        let total = self.data.len();
        let new_shape: Vec<usize> = if wildcards == 1 {
            if known == 0 || total % known != 0 {
                return err(format!("cannot reshape {:?} to {shape:?}", self.shape));
            }
            shape.iter().map(|&d| if d == -1 { total / known } else { d as usize }).collect()
        } else {
            shape.iter().map(|&d| d as usize).collect()
        };
        Self::new(&new_shape, self.data.clone())
    }

    pub fn permute(&self, permutation: &[usize]) -> Result<Self> {
        let rank = self.shape.len();
        let mut seen = vec![false; rank];
        if permutation.len() != rank || permutation.iter().any(|&p| p >= rank || std::mem::replace(&mut seen[p], true)) {
            return err(format!("invalid permutation {permutation:?} for rank {rank}"));
        }
        let strides = row_major_strides(&self.shape);
        let out_shape: Vec<usize> = permutation.iter().map(|&p| self.shape[p]).collect();
        let read: Vec<usize> = permutation.iter().map(|&p| strides[p]).collect();
        Ok(Self { shape: out_shape.clone(), data: gather(&self.data, &out_shape, &read, 0) })
    }

    /// numpy style slicing, e.g. `"1, ..., ::-1"`.
    pub fn slice(&self, spec: &str) -> Result<Self> {
        let plan = slicing::plan(spec, &self.shape)?;
        let strides = row_major_strides(&self.shape);
        let mut offset = 0isize;
        let mut out_shape = Vec::new();
        let mut read = Vec::new();
        for (d, s) in plan.iter().enumerate() {
            offset += s.start * strides[d] as isize;
            if !s.drop {
                out_shape.push(s.len);
                read.push(s.step * strides[d] as isize);
            }
        }
        Ok(Self { shape: out_shape.clone(), data: gather_signed(&self.data, &out_shape, &read, offset) })
    }

    /// Batched matrix multiply over the last two dimensions, broadcasting the batch dimensions.
    pub fn matmul(&self, other: &Self, transpose_left: bool, transpose_right: bool) -> Result<Self> {
        let (a, b) = (&self.shape, &other.shape);
        if a.len() < 2 || b.len() < 2 {
            return err("matrix multiply needs rank >= 2");
        }
        let (ar, ac) = (a[a.len() - 2], a[a.len() - 1]);
        let (br, bc) = (b[b.len() - 2], b[b.len() - 1]);
        let (m, k) = if transpose_left { (ac, ar) } else { (ar, ac) };
        let (k2, n) = if transpose_right { (bc, br) } else { (br, bc) };
        if k != k2 {
            return err(format!("cannot matrix multiply {a:?} and {b:?}"));
        }
        let batch = broadcast_shape(&a[..a.len() - 2], &b[..b.len() - 2])?;
        let batch_len = length(&batch);
        let sa = broadcast_strides(&a[..a.len() - 2], &batch);
        let sb = broadcast_strides(&b[..b.len() - 2], &batch);
        let batch_strides = row_major_strides(&batch);
        let mut data = vec![0.0; batch_len * m * n];
        for i in 0..batch_len {
            let (mut oa, mut ob) = (0, 0);
            for d in 0..batch.len() {
                let idx = (i / batch_strides[d]) % batch[d];
                oa += idx * sa[d];
                ob += idx * sb[d];
            }
            let lhs = faer::MatRef::from_row_major_slice(&self.data[oa * ar * ac..(oa + 1) * ar * ac], ar, ac);
            let rhs = faer::MatRef::from_row_major_slice(&other.data[ob * br * bc..(ob + 1) * br * bc], br, bc);
            let lhs = if transpose_left { lhs.transpose() } else { lhs };
            let rhs = if transpose_right { rhs.transpose() } else { rhs };
            let product = lhs * rhs;
            let block = &mut data[i * m * n..(i + 1) * m * n];
            for r in 0..m {
                for c in 0..n {
                    block[r * n + c] = product[(r, c)];
                }
            }
        }
        let mut shape = batch;
        shape.extend([m, n]);
        Ok(Self { shape, data })
    }

    /// Lower Cholesky factor of each matrix in a batch; only the lower triangle is read.
    pub fn cholesky(&self) -> Result<Self> {
        let s = &self.shape;
        if s.len() < 2 || s[s.len() - 1] != s[s.len() - 2] {
            return err("cholesky needs square matrices");
        }
        let n = s[s.len() - 1];
        let mut data = vec![0.0; self.data.len()];
        for (block, out) in self.data.chunks(n * n).zip(data.chunks_mut(n * n)) {
            let mat = faer::MatRef::from_row_major_slice(block, n, n);
            let llt = mat.llt(faer::Side::Lower).map_err(|e| Error(format!("not positive definite: {e:?}")))?;
            let l = llt.L();
            for r in 0..n {
                for c in 0..=r {
                    out[r * n + c] = l[(r, c)];
                }
            }
        }
        Ok(Self { shape: s.clone(), data })
    }
}

fn gather(src: &[f64], shape: &[usize], read: &[usize], offset: usize) -> Vec<f64> {
    let signed: Vec<isize> = read.iter().map(|&r| r as isize).collect();
    gather_signed(src, shape, &signed, offset as isize)
}

/// Copies `src` read with arbitrary (possibly negative) strides into a contiguous row-major buffer.
fn gather_signed(src: &[f64], shape: &[usize], read: &[isize], offset: isize) -> Vec<f64> {
    let n = length(shape);
    let mut out = Vec::with_capacity(n);
    if n == 0 {
        return out;
    }
    let mut index = vec![0usize; shape.len()];
    let mut position = offset;
    for _ in 0..n {
        out.push(src[position as usize]);
        for d in (0..shape.len()).rev() {
            index[d] += 1;
            position += read[d];
            if index[d] < shape[d] {
                break;
            }
            position -= read[d] * shape[d] as isize;
            index[d] = 0;
        }
    }
    out
}

/// numpy slice parsing and normalisation, shared by both spikes (they only differ in how they apply it).
pub mod slicing {
    use super::{Error, Result};

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct DimSlice {
        pub start: isize,
        pub step: isize,
        pub len: usize,
        pub drop: bool,
    }

    enum Item {
        Ellipsis,
        Index(isize),
        Range(Option<isize>, Option<isize>, Option<isize>),
    }

    fn parse_item(text: &str) -> Result<Item> {
        let parse = |s: &str| -> Result<Option<isize>> {
            if s.is_empty() { Ok(None) } else { s.parse().map(Some).map_err(|_| Error(format!("bad slice {s}"))) }
        };
        if text == "..." {
            return Ok(Item::Ellipsis);
        }
        let parts: Vec<&str> = text.split(':').collect();
        match parts.len() {
            1 => Ok(Item::Index(parse(parts[0])?.ok_or_else(|| Error("empty index".into()))?)),
            2 => Ok(Item::Range(parse(parts[0])?, parse(parts[1])?, None)),
            3 => Ok(Item::Range(parse(parts[0])?, parse(parts[1])?, parse(parts[2])?)),
            _ => Err(Error(format!("bad slice {text}"))),
        }
    }

    /// Resolves a slice string against a shape into one `DimSlice` per dimension.
    pub fn plan(spec: &str, shape: &[usize]) -> Result<Vec<DimSlice>> {
        let items = spec
            .split(',')
            .map(|s| s.trim())
            .map(parse_item)
            .collect::<Result<Vec<_>>>()?;
        let explicit = items.iter().filter(|i| !matches!(i, Item::Ellipsis)).count();
        if explicit > shape.len() {
            return Err(Error(format!("too many indices for shape {shape:?}")));
        }
        let mut plan = Vec::with_capacity(shape.len());
        let mut dim = 0;
        let all = |len: usize| DimSlice { start: 0, step: 1, len, drop: false };
        for item in &items {
            match *item {
                Item::Ellipsis => {
                    for _ in 0..shape.len() - explicit {
                        plan.push(all(shape[dim]));
                        dim += 1;
                    }
                }
                Item::Index(i) => {
                    let n = shape[dim] as isize;
                    let i = if i < 0 { i + n } else { i };
                    if i < 0 || i >= n {
                        return Err(Error(format!("index out of range for dimension of length {n}")));
                    }
                    plan.push(DimSlice { start: i, step: 1, len: 1, drop: true });
                    dim += 1;
                }
                Item::Range(start, stop, step) => {
                    let n = shape[dim] as isize;
                    let step = step.unwrap_or(1);
                    if step == 0 {
                        return Err(Error("slice step cannot be zero".into()));
                    }
                    let clamp = |v: isize| -> isize {
                        let v = if v < 0 { v + n } else { v };
                        if step > 0 { v.clamp(0, n) } else { v.clamp(-1, n - 1) }
                    };
                    let first = start.map(clamp).unwrap_or(if step > 0 { 0 } else { n - 1 });
                    let end = stop.map(clamp).unwrap_or(if step > 0 { n } else { -1 });
                    let len = if step > 0 {
                        if end > first { (end - first + step - 1) / step } else { 0 }
                    } else if first > end {
                        (first - end - step - 1) / -step
                    } else {
                        0
                    };
                    plan.push(DimSlice { start: first, step, len: len as usize, drop: false });
                    dim += 1;
                }
            }
        }
        while dim < shape.len() {
            plan.push(all(shape[dim]));
            dim += 1;
        }
        Ok(plan)
    }
}
