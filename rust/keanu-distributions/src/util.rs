//! Element-wise helpers that broadcast several tensors together.

use crate::error::Result;
use keanu_tensor::{DoubleTensor, Element, Shape, Tensor, broadcast_shape};

pub(crate) fn broadcast_all(shapes: &[&[usize]]) -> Result<Shape> {
    let mut out = Shape::new();
    for s in shapes {
        out = broadcast_shape(&out, s)?;
    }
    Ok(out)
}

/// Broadcasts tensors to a shape and returns their data in row-major order.
fn expand<T: Element>(t: &Tensor<T>, shape: &[usize]) -> Result<Vec<T>> {
    Ok(if t.shape() == shape {
        t.as_slice().to_vec()
    } else {
        t.broadcast_to(shape)?.into_vec()
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
    let (a, b, c) = (expand(a, &shape)?, expand(b, &shape)?, expand(c, &shape)?);
    let data = (0..a.len()).map(|i| f(a[i], b[i], c[i])).collect();
    Ok(DoubleTensor::new(&shape, data)?)
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
        expand(a, &shape)?,
        expand(b, &shape)?,
        expand(c, &shape)?,
        expand(d, &shape)?,
    );
    let data = (0..a.len()).map(|i| f(a[i], b[i], c[i], d[i])).collect();
    Ok(DoubleTensor::new(&shape, data)?)
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
        expand(a, &shape)?,
        expand(b, &shape)?,
        expand(c, &shape)?,
        expand(d, &shape)?,
        expand(e, &shape)?,
    );
    let data = (0..a.len())
        .map(|i| f(a[i], b[i], c[i], d[i], e[i]))
        .collect();
    Ok(DoubleTensor::new(&shape, data)?)
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
        .map(|p| expand(p, &out))
        .collect::<Result<Vec<_>>>()?;
    let len = out.iter().product();
    let mut values = Vec::with_capacity(params.len());
    let mut data = Vec::with_capacity(len);
    for i in 0..len {
        values.clear();
        values.extend(expanded.iter().map(|p| p[i]));
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
