//! Shape arithmetic: lengths, strides, broadcasting and axis normalisation.

use crate::error::{Result, TensorError};
use smallvec::SmallVec;

/// Dimensions of a tensor. Stored inline for rank <= 4 so small tensors don't allocate for their shape.
pub type Shape = SmallVec<[usize; 4]>;

/// The number of elements in a shape, or None if it does not fit in a `usize`. A shape with a zero dimension
/// has no elements whatever its other dimensions.
pub(crate) fn checked_length(shape: &[usize]) -> Option<usize> {
    if shape.contains(&0) {
        return Some(0);
    }
    shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d))
}

/// The number of elements in the shape of an existing tensor, or one about to be allocated.
///
/// # Panics
/// If the length overflows `usize`, as `vec![x; n]` would for an impossible allocation.
pub(crate) fn length(shape: &[usize]) -> usize {
    checked_length(shape).expect("tensor length overflows usize")
}

pub(crate) fn row_major_strides(shape: &[usize]) -> Shape {
    let mut strides: Shape = smallvec::smallvec![0; shape.len()];
    let mut acc = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        // saturating: with a zero dimension the partial products can exceed usize, but then nothing is read
        acc = acc.saturating_mul(shape[i]);
    }
    strides
}

/// The numpy broadcast of two shapes. Fails if they are incompatible or the result's length overflows `usize`.
pub fn broadcast_shape(left: &[usize], right: &[usize]) -> Result<Shape> {
    let out = broadcast_dims(left, right)?;
    if checked_length(&out).is_none() {
        return Err(TensorError::IncompatibleShapes {
            left: left.to_vec(),
            right: right.to_vec(),
        });
    }
    Ok(out)
}

fn broadcast_dims(left: &[usize], right: &[usize]) -> Result<Shape> {
    // the common cases: equal shapes, or one side a scalar
    if left == right || right.is_empty() {
        return Ok(Shape::from_slice(left));
    }
    if left.is_empty() {
        return Ok(Shape::from_slice(right));
    }
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
