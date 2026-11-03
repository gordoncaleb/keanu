//! Element-wise helpers that broadcast several tensors together.

use crate::error::Result;
use keanu_tensor::{DoubleTensor, Element, Shape, Tensor, broadcast_shape};

pub(crate) fn broadcast_all(shapes: &[&[usize]]) -> Result<Shape> {
    // common case: every shape is the same as the largest one, or a scalar (rank 0)
    if let Some(largest) = shapes.iter().copied().max_by_key(|s| s.len()) {
        if shapes.iter().all(|s| s.is_empty() || *s == largest) {
            return Ok(Shape::from_slice(largest));
        }
    }
    let mut out = Shape::new();
    for s in shapes {
        out = broadcast_shape(&out, s)?;
    }
    Ok(out)
}

/// Read access to a tensor as if broadcast to a shape, without copying when it already has that shape or is a
/// single value.
enum Operand<'a, T> {
    Slice(&'a [T]),
    Scalar(T),
    Expanded(Vec<T>),
}

impl<T: Element> Operand<'_, T> {
    #[inline]
    fn at(&self, i: usize) -> T {
        match self {
            Operand::Slice(s) => s[i],
            Operand::Scalar(v) => *v,
            Operand::Expanded(v) => v[i],
        }
    }
}

fn operand<'a, T: Element>(t: &'a Tensor<T>, shape: &[usize]) -> Result<Operand<'a, T>> {
    Ok(if t.shape() == shape {
        Operand::Slice(t.as_slice())
    } else if t.len() == 1 {
        Operand::Scalar(t.as_slice()[0])
    } else {
        Operand::Expanded(t.broadcast_to(shape)?.into_vec())
    })
}

pub(crate) fn map2<A: Element, B: Element>(
    a: &Tensor<A>,
    b: &Tensor<B>,
    f: impl Fn(A, B) -> f64,
) -> Result<DoubleTensor> {
    Ok(a.zip_with(b, f)?)
}

pub(crate) fn map3<A: Element, B: Element, C: Element>(
    a: &Tensor<A>,
    b: &Tensor<B>,
    c: &Tensor<C>,
    f: impl Fn(A, B, C) -> f64,
) -> Result<DoubleTensor> {
    let shape = broadcast_all(&[a.shape(), b.shape(), c.shape()])?;
    let (a, b, c) = (
        operand(a, &shape)?,
        operand(b, &shape)?,
        operand(c, &shape)?,
    );
    Ok(DoubleTensor::from_fn(&shape, |i| {
        f(a.at(i), b.at(i), c.at(i))
    }))
}

/// Like [`map3`] but computes two outputs in one pass.
pub(crate) fn map3_pair<A: Element, B: Element, C: Element>(
    a: &Tensor<A>,
    b: &Tensor<B>,
    c: &Tensor<C>,
    f: impl Fn(A, B, C) -> (f64, f64),
) -> Result<(DoubleTensor, DoubleTensor)> {
    let shape = broadcast_all(&[a.shape(), b.shape(), c.shape()])?;
    let (a, b, c) = (
        operand(a, &shape)?,
        operand(b, &shape)?,
        operand(c, &shape)?,
    );
    let n: usize = shape.iter().product();
    let mut second = Vec::new();
    let first = if n <= 4 {
        // small: compute twice rather than allocate (f is cheap and pure)
        DoubleTensor::from_fn(&shape, |i| f(a.at(i), b.at(i), c.at(i)).0)
    } else {
        second.reserve(n);
        DoubleTensor::from_fn(&shape, |i| {
            let (x, y) = f(a.at(i), b.at(i), c.at(i));
            second.push(y);
            x
        })
    };
    let second = if n <= 4 {
        DoubleTensor::from_fn(&shape, |i| f(a.at(i), b.at(i), c.at(i)).1)
    } else {
        DoubleTensor::new(&shape, second)?
    };
    Ok((first, second))
}

pub(crate) fn map4<A: Element, B: Element, C: Element, D: Element>(
    a: &Tensor<A>,
    b: &Tensor<B>,
    c: &Tensor<C>,
    d: &Tensor<D>,
    f: impl Fn(A, B, C, D) -> f64,
) -> Result<DoubleTensor> {
    let shape = broadcast_all(&[a.shape(), b.shape(), c.shape(), d.shape()])?;
    let (a, b, c, d) = (
        operand(a, &shape)?,
        operand(b, &shape)?,
        operand(c, &shape)?,
        operand(d, &shape)?,
    );
    Ok(DoubleTensor::from_fn(&shape, |i| {
        f(a.at(i), b.at(i), c.at(i), d.at(i))
    }))
}

pub(crate) fn map5<A: Element, B: Element, C: Element, D: Element, E: Element>(
    a: &Tensor<A>,
    b: &Tensor<B>,
    c: &Tensor<C>,
    d: &Tensor<D>,
    e: &Tensor<E>,
    f: impl Fn(A, B, C, D, E) -> f64,
) -> Result<DoubleTensor> {
    let shape = broadcast_all(&[a.shape(), b.shape(), c.shape(), d.shape(), e.shape()])?;
    let (a, b, c, d, e) = (
        operand(a, &shape)?,
        operand(b, &shape)?,
        operand(c, &shape)?,
        operand(d, &shape)?,
        operand(e, &shape)?,
    );
    Ok(DoubleTensor::from_fn(&shape, |i| {
        f(a.at(i), b.at(i), c.at(i), d.at(i), e.at(i))
    }))
}

/// Draws one value per element of the broadcast of `shape` and the parameters' shapes.
pub(crate) fn sample_each<T: Element>(
    shape: &[usize],
    params: &[&DoubleTensor],
    mut draw: impl FnMut(&[f64]) -> Result<T>,
) -> Result<Tensor<T>> {
    let mut shapes: Vec<&[usize]> = vec![shape];
    shapes.extend(params.iter().map(|p| p.shape()));
    let out = broadcast_all(&shapes)?;
    let expanded = params
        .iter()
        .map(|p| operand(p, &out))
        .collect::<Result<Vec<_>>>()?;
    let len = out.iter().product();
    let mut values = Vec::with_capacity(params.len());
    let mut data = Vec::with_capacity(len);
    for i in 0..len {
        values.clear();
        values.extend(expanded.iter().map(|p| p.at(i)));
        data.push(draw(&values)?);
    }
    Ok(Tensor::new(&out, data)?)
}

/// `y * ln(x)`, taken to be 0 where `y` is 0 (even if `x` is 0).
pub(crate) fn safe_ln_times(x: f64, y: f64) -> f64 {
    if y == 0.0 { 0.0 } else { x.ln() * y }
}

pub(crate) fn ln_gamma(x: f64) -> f64 {
    keanu_tensor::special::ln_gamma(x)
}

pub(crate) fn digamma(x: f64) -> f64 {
    keanu_tensor::special::digamma(x)
}

pub(crate) fn scalar(v: f64) -> DoubleTensor {
    DoubleTensor::scalar(v)
}
