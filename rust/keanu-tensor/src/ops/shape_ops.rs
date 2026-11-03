//! Reshaping, permuting, broadcasting, triangles, diagonals, concatenation and numpy style slicing.

use crate::error::{Result, TensorError};
use crate::shape::{self, Shape};
use crate::tensor::{Element, Tensor};

impl<T: Element> Tensor<T> {
    /// Reshapes to a new shape with the same length. At most one dimension may be -1, which is inferred.
    pub fn reshape(&self, new_shape: &[isize]) -> Result<Self> {
        let invalid = || TensorError::InvalidReshape {
            from: self.shape.to_vec(),
            to: new_shape.to_vec(),
        };
        let wildcards = new_shape.iter().filter(|&&d| d == -1).count();
        if wildcards > 1 || new_shape.iter().any(|&d| d < -1) {
            return Err(invalid());
        }
        let known: usize = new_shape
            .iter()
            .filter(|&&d| d >= 0)
            .map(|&d| d as usize)
            .product();
        let resolved: Shape = if wildcards == 1 {
            if known == 0 || !self.len().is_multiple_of(known) {
                return Err(invalid());
            }
            new_shape
                .iter()
                .map(|&d| {
                    if d == -1 {
                        self.len() / known
                    } else {
                        d as usize
                    }
                })
                .collect()
        } else {
            new_shape.iter().map(|&d| d as usize).collect()
        };
        if shape::length(&resolved) != self.len() {
            return Err(invalid());
        }
        Ok(Tensor::from_parts(resolved, self.data.clone()))
    }

    /// Rearranges the dimensions: dimension `i` of the result is dimension `permutation[i]` of this tensor.
    pub fn permute(&self, permutation: &[usize]) -> Result<Self> {
        let rank = self.rank();
        let mut seen = vec![false; rank];
        let valid = permutation.len() == rank
            && permutation
                .iter()
                .all(|&p| p < rank && !std::mem::replace(&mut seen[p], true));
        if !valid {
            return Err(TensorError::InvalidPermutation {
                permutation: permutation.to_vec(),
                rank,
            });
        }
        let strides = shape::row_major_strides(&self.shape);
        let out_shape: Shape = permutation.iter().map(|&p| self.shape[p]).collect();
        let read: Vec<isize> = permutation.iter().map(|&p| strides[p] as isize).collect();
        Ok(self.gather(&out_shape, &read, 0))
    }

    /// Swaps the last two dimensions.
    pub fn transpose(&self) -> Result<Self> {
        let rank = self.rank();
        if rank < 2 {
            return Err(TensorError::InvalidArgument(
                "transpose needs a tensor of rank >= 2".into(),
            ));
        }
        let mut permutation: Vec<usize> = (0..rank).collect();
        permutation.swap(rank - 1, rank - 2);
        self.permute(&permutation)
    }

    /// Broadcasts to a shape, as numpy's `broadcast_to`.
    pub fn broadcast_to(&self, target: &[usize]) -> Result<Self> {
        let out = shape::broadcast_shape(&self.shape, target)?;
        if out.as_slice() != target {
            return Err(TensorError::IncompatibleShapes {
                left: self.shape.to_vec(),
                right: target.to_vec(),
            });
        }
        let strides = shape::broadcast_strides(&self.shape, target);
        let read: Vec<isize> = strides.iter().map(|&s| s as isize).collect();
        Ok(self.gather(target, &read, 0))
    }

    /// Joins tensors along an existing axis. All other dimensions must match.
    pub fn concat(axis: isize, tensors: &[&Self]) -> Result<Self> {
        let first = tensors
            .first()
            .ok_or_else(|| TensorError::InvalidArgument("nothing to concatenate".into()))?;
        let axis = shape::axis(axis, first.rank())?;
        for t in tensors {
            let compatible = t.rank() == first.rank()
                && (0..t.rank()).all(|d| d == axis || t.shape[d] == first.shape[d]);
            if !compatible {
                return Err(TensorError::IncompatibleShapes {
                    left: first.shape.to_vec(),
                    right: t.shape.to_vec(),
                });
            }
        }
        let outer = shape::length(&first.shape[..axis]);
        let mut out_shape = first.shape.clone();
        out_shape[axis] = tensors.iter().map(|t| t.shape[axis]).sum();
        let mut data = Vec::with_capacity(shape::length(&out_shape));
        for o in 0..outer {
            for t in tensors {
                let block = shape::length(&t.shape[axis..]);
                data.extend_from_slice(&t.data[o * block..(o + 1) * block]);
            }
        }
        Ok(Tensor::from_parts(out_shape, data))
    }

    fn check_matrix(&self, name: &str) -> Result<(usize, usize)> {
        if self.rank() < 2 {
            return Err(TensorError::InvalidArgument(format!(
                "{name} needs a tensor of rank >= 2"
            )));
        }
        Ok((self.shape[self.rank() - 2], self.shape[self.rank() - 1]))
    }

    /// Keeps the elements on and below the k-th diagonal of each matrix (j - i <= k), zeroing the rest,
    /// as numpy's `tril`. k > 0 includes diagonals above the main one.
    pub fn tril(&self, k: isize) -> Result<Self> {
        self.keep_where(|i, j| j as isize - i as isize <= k, "tril")
    }

    /// Keeps the elements on and above the k-th diagonal of each matrix (j - i >= k), zeroing the rest,
    /// as numpy's `triu`.
    pub fn triu(&self, k: isize) -> Result<Self> {
        self.keep_where(|i, j| j as isize - i as isize >= k, "triu")
    }

    fn keep_where(&self, keep: impl Fn(usize, usize) -> bool, name: &str) -> Result<Self> {
        let (rows, cols) = self.check_matrix(name)?;
        let mut data = self.data.clone();
        for block in data.chunks_mut((rows * cols).max(1)) {
            for i in 0..rows {
                for j in 0..cols {
                    if !keep(i, j) {
                        block[i * cols + j] = T::default();
                    }
                }
            }
        }
        Ok(Tensor::from_parts(self.shape.clone(), data))
    }

    /// The main diagonal of each matrix: shape [..., m, n] becomes [..., min(m, n)].
    pub fn diag_part(&self) -> Result<Self> {
        let (rows, cols) = self.check_matrix("diag_part")?;
        let k = rows.min(cols);
        let mut out_shape: Shape = self.shape[..self.rank() - 2].into();
        out_shape.push(k);
        let mut data = Vec::with_capacity(shape::length(&out_shape));
        for block in self.data.chunks((rows * cols).max(1)) {
            data.extend((0..k).map(|i| block[i * cols + i]));
        }
        Ok(Tensor::from_parts(out_shape, data))
    }

    /// A diagonal matrix from each vector: shape [..., n] becomes [..., n, n].
    pub fn diag(&self) -> Result<Self> {
        if self.rank() < 1 {
            return Err(TensorError::InvalidArgument(
                "diag needs a tensor of rank >= 1".into(),
            ));
        }
        let n = self.shape[self.rank() - 1];
        let mut out_shape = self.shape.clone();
        out_shape.push(n);
        let mut data = vec![T::default(); shape::length(&out_shape)];
        for (b, vector) in self.data.chunks(n.max(1)).enumerate() {
            for (i, &v) in vector.iter().enumerate() {
                data[b * n * n + i * n + i] = v;
            }
        }
        Ok(Tensor::from_parts(out_shape, data))
    }

    /// Fills the lower and/or upper triangle of an n × n matrix from vectors of length n(n + 1) / 2 in the last
    /// dimension. The lower triangle is filled column by column (equivalently the upper triangle row by row), so
    /// filling both gives a symmetric matrix.
    pub fn fill_triangular(&self, fill_upper: bool, fill_lower: bool) -> Result<Self> {
        if self.rank() < 1 {
            return Err(TensorError::InvalidArgument(
                "fill_triangular needs a tensor of rank >= 1".into(),
            ));
        }
        let m = self.shape[self.rank() - 1];
        let n = (((8 * m + 1) as f64).sqrt() as usize).saturating_sub(1) / 2;
        if n * (n + 1) / 2 != m {
            return Err(TensorError::InvalidArgument(format!(
                "{m} is not a triangular number"
            )));
        }
        let mut out_shape: Shape = self.shape[..self.rank() - 1].into();
        out_shape.extend([n, n]);
        let mut data = vec![T::default(); shape::length(&out_shape)];
        for (b, vector) in self.data.chunks(m.max(1)).enumerate() {
            let matrix = &mut data[b * n * n..(b + 1) * n * n];
            let mut k = 0;
            for row in 0..n {
                for col in row..n {
                    if fill_upper {
                        matrix[row * n + col] = vector[k];
                    }
                    if fill_lower {
                        matrix[col * n + row] = vector[k];
                    }
                    k += 1;
                }
            }
        }
        Ok(Tensor::from_parts(out_shape, data))
    }

    /// The upper or lower triangle (including the diagonal) of each square matrix as a vector of length
    /// n(n + 1) / 2: the upper triangle row by row, or the lower triangle column by column. This is the inverse of
    /// [`fill_triangular`](Self::fill_triangular).
    pub fn triangle_part(&self, upper: bool) -> Result<Self> {
        let (rows, cols) = self.check_matrix("triangle_part")?;
        if rows != cols {
            return Err(TensorError::InvalidArgument(
                "triangle_part needs square matrices".into(),
            ));
        }
        let n = rows;
        let mut out_shape: Shape = self.shape[..self.rank() - 2].into();
        out_shape.push(n * (n + 1) / 2);
        let mut data = Vec::with_capacity(shape::length(&out_shape));
        for matrix in self
            .data
            .chunks((n * n).max(1))
            .take(shape::length(&out_shape[..out_shape.len() - 1]))
        {
            for a in 0..n {
                for b in a..n {
                    // upper: row a, column b; lower: column a, row b
                    data.push(if upper {
                        matrix[a * n + b]
                    } else {
                        matrix[b * n + a]
                    });
                }
            }
        }
        Ok(Tensor::from_parts(out_shape, data))
    }

    /// numpy style slicing from a string, e.g. `"1, ..., ::-1"`. Integer indices remove their dimension.
    pub fn slice(&self, spec: &str) -> Result<Self> {
        let plan = slicing::plan(spec, &self.shape)?;
        let strides = shape::row_major_strides(&self.shape);
        let mut offset = 0isize;
        let mut out_shape = Shape::new();
        let mut read = Vec::new();
        for (d, s) in plan.iter().enumerate() {
            offset += s.start * strides[d] as isize;
            if !s.drop {
                out_shape.push(s.len);
                read.push(s.step * strides[d] as isize);
            }
        }
        Ok(self.gather(&out_shape, &read, offset))
    }
}

/// Parsing and resolution of numpy slice strings.
pub(crate) mod slicing {
    use crate::error::{Result, TensorError};

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub(crate) struct DimSlice {
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

    fn invalid(message: impl Into<String>) -> TensorError {
        TensorError::InvalidSlice(message.into())
    }

    fn parse_item(text: &str) -> Result<Item> {
        let parse = |s: &str| -> Result<Option<isize>> {
            if s.is_empty() {
                Ok(None)
            } else {
                s.parse()
                    .map(Some)
                    .map_err(|_| invalid(format!("cannot parse {s:?}")))
            }
        };
        if text == "..." {
            return Ok(Item::Ellipsis);
        }
        let parts: Vec<&str> = text.split(':').collect();
        match parts.as_slice() {
            [index] => Ok(Item::Index(
                parse(index)?.ok_or_else(|| invalid("empty index"))?,
            )),
            [start, stop] => Ok(Item::Range(parse(start)?, parse(stop)?, None)),
            [start, stop, step] => Ok(Item::Range(parse(start)?, parse(stop)?, parse(step)?)),
            _ => Err(invalid(format!("cannot parse {text:?}"))),
        }
    }

    /// Resolves a slice string against a shape into one `DimSlice` per dimension.
    pub(crate) fn plan(spec: &str, shape: &[usize]) -> Result<Vec<DimSlice>> {
        let items = spec
            .split(',')
            .map(str::trim)
            .map(parse_item)
            .collect::<Result<Vec<_>>>()?;
        if items.iter().filter(|i| matches!(i, Item::Ellipsis)).count() > 1 {
            return Err(invalid("more than one ellipsis"));
        }
        let explicit = items
            .iter()
            .filter(|i| !matches!(i, Item::Ellipsis))
            .count();
        if explicit > shape.len() {
            return Err(invalid(format!("too many indices for shape {shape:?}")));
        }
        let all = |len: usize| DimSlice {
            start: 0,
            step: 1,
            len,
            drop: false,
        };
        let mut plan = Vec::with_capacity(shape.len());
        let mut dim = 0;
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
                        return Err(invalid(format!(
                            "index out of range for a dimension of length {n}"
                        )));
                    }
                    plan.push(DimSlice {
                        start: i,
                        step: 1,
                        len: 1,
                        drop: true,
                    });
                    dim += 1;
                }
                Item::Range(start, stop, step) => {
                    let n = shape[dim] as isize;
                    let step = step.unwrap_or(1);
                    if step == 0 {
                        return Err(invalid("step cannot be zero"));
                    }
                    let clamp = |v: isize| {
                        let v = if v < 0 { v + n } else { v };
                        if step > 0 {
                            v.clamp(0, n)
                        } else {
                            v.clamp(-1, n - 1)
                        }
                    };
                    let first = start.map(clamp).unwrap_or(if step > 0 { 0 } else { n - 1 });
                    let end = stop.map(clamp).unwrap_or(if step > 0 { n } else { -1 });
                    let len = if step > 0 {
                        if end > first {
                            (end - first + step - 1) / step
                        } else {
                            0
                        }
                    } else if first > end {
                        (first - end - step - 1) / -step
                    } else {
                        0
                    };
                    plan.push(DimSlice {
                        start: first,
                        step,
                        len: len as usize,
                        drop: false,
                    });
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
