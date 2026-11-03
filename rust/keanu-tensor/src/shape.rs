//! Shape arithmetic: lengths, strides, broadcasting and axis normalisation.

use crate::error::{Result, TensorError};
use smallvec::SmallVec;

/// Dimensions of a tensor. Stored inline for rank <= 4 so small tensors don't allocate for their shape.
pub type Shape = SmallVec<[usize; 4]>;

pub(crate) fn length(shape: &[usize]) -> usize {
    shape.iter().product()
}

pub(crate) fn row_major_strides(shape: &[usize]) -> Shape {
    let mut strides: Shape = smallvec::smallvec![0; shape.len()];
    let mut acc = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc *= shape[i];
    }
    strides
}

/// The numpy broadcast of two shapes.
pub fn broadcast_shape(left: &[usize], right: &[usize]) -> Result<Shape> {
    let rank = left.len().max(right.len());
    let mut out: Shape = smallvec::smallvec![0; rank];
    for (i, dim) in out.iter_mut().enumerate() {
        let l = if i + left.len() >= rank {
            left[i + left.len() - rank]
        } else {
            1
        };
        let r = if i + right.len() >= rank {
            right[i + right.len() - rank]
        } else {
            1
        };
        *dim = if l == r || r == 1 {
            l
        } else if l == 1 {
            r
        } else {
            return Err(TensorError::IncompatibleShapes {
                left: left.to_vec(),
                right: right.to_vec(),
            });
        };
    }
    Ok(out)
}

/// Strides for reading `shape` as if broadcast to `out` (0 on broadcast dimensions).
pub(crate) fn broadcast_strides(shape: &[usize], out: &[usize]) -> Shape {
    let strides = row_major_strides(shape);
    let offset = out.len() - shape.len();
    (0..out.len())
        .map(|i| {
            if i < offset || shape[i - offset] == 1 {
                0
            } else {
                strides[i - offset]
            }
        })
        .collect()
}

/// Resolves a possibly negative axis (counting from the end) against a rank.
pub(crate) fn axis(axis: isize, rank: usize) -> Result<usize> {
    let a = if axis < 0 { axis + rank as isize } else { axis };
    if a < 0 || a as usize >= rank {
        return Err(TensorError::InvalidAxis { axis, rank });
    }
    Ok(a as usize)
}

/// Resolves a list of possibly negative axes into a per-dimension mask.
pub(crate) fn axis_mask(axes: &[isize], rank: usize) -> Result<SmallVec<[bool; 4]>> {
    let mut mask: SmallVec<[bool; 4]> = smallvec::smallvec![false; rank];
    for &a in axes {
        mask[axis(a, rank)?] = true;
    }
    Ok(mask)
}
