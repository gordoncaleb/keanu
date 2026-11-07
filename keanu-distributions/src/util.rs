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
pub(crate) enum Operand<'a, T> {
    Slice(&'a [T]),
    Scalar(T),
    Expanded(Vec<T>),
}

impl<T: Element> Operand<'_, T> {
    #[inline]
    pub(crate) fn at(&self, i: usize) -> T {
        match self {
            Operand::Slice(s) => s[i],
            Operand::Scalar(v) => *v,
            Operand::Expanded(v) => v[i],
        }
    }
}

pub(crate) fn operand<'a, T: Element>(t: &'a Tensor<T>, shape: &[usize]) -> Result<Operand<'a, T>> {
    Ok(if t.shape() == shape {
        Operand::Slice(t.as_slice())
    } else if t.len() == 1 {
        Operand::Scalar(t.as_slice()[0])
    } else {
        Operand::Expanded(t.broadcast_to(shape)?.into_vec())
    })
}

/// A tuple of tensor references that broadcast together (see [`map`]).
pub(crate) trait Operands<'a> {
    /// A tuple of one element of each tensor.
    type Item;
    /// The tensors as broadcast.
    type Broadcast: Broadcast<Item = Self::Item>;
    /// The broadcast shape of the tensors, and the tensors broadcast to it.
    fn broadcast(self) -> Result<(Shape, Self::Broadcast)>;
}

/// Tensors broadcast to a common shape, read element by element.
pub(crate) trait Broadcast {
    /// A tuple of one element of each tensor.
    type Item;
    /// The elements at a flat index of the broadcast shape.
    fn at(&self, i: usize) -> Self::Item;
}

macro_rules! operands {
    ($($t:ident $v:ident),+) => {
        impl<'a, $($t: Element),+> Operands<'a> for ($(&'a Tensor<$t>,)+) {
            type Item = ($($t,)+);
            type Broadcast = ($(Operand<'a, $t>,)+);

            fn broadcast(self) -> Result<(Shape, Self::Broadcast)> {
                let ($($v,)+) = self;
                let shape = broadcast_all(&[$($v.shape()),+])?;
                let broadcast = ($(operand($v, &shape)?,)+);
                Ok((shape, broadcast))
            }
        }

        impl<'a, $($t: Element),+> Broadcast for ($(Operand<'a, $t>,)+) {
            type Item = ($($t,)+);

            #[inline]
            fn at(&self, i: usize) -> Self::Item {
                let ($($v,)+) = self;
                ($($v.at(i),)+)
            }
        }
    };
}

operands!(A a, B b);
operands!(A a, B b, C c);
operands!(A a, B b, C c, D d);
operands!(A a, B b, C c, D d, E e);
operands!(A a, B b, C c, D d, E e, F f);
operands!(A a, B b, C c, D d, E e, F f, G g);

/// `f` of each element of the broadcast of the `operands`, a tuple of tensor references.
pub(crate) fn map<'a, O: Operands<'a>>(
    operands: O,
    f: impl Fn(O::Item) -> f64,
) -> Result<DoubleTensor> {
    let (shape, broadcast) = operands.broadcast()?;
    Ok(DoubleTensor::from_fn(&shape, |i| f(broadcast.at(i))))
}

/// Like [`map`], but computes `N` outputs in one pass over the broadcast operands.
pub(crate) fn map_n<'a, O: Operands<'a>, const N: usize>(
    operands: O,
    f: impl Fn(O::Item) -> [f64; N],
) -> Result<[DoubleTensor; N]> {
    let (shape, broadcast) = operands.broadcast()?;
    let len: usize = shape.iter().product();
    let mut outputs: [Vec<f64>; N] = std::array::from_fn(|_| Vec::with_capacity(len));
    for i in 0..len {
        for (output, value) in outputs.iter_mut().zip(f(broadcast.at(i))) {
            output.push(value);
        }
    }
    let mut outputs = outputs.into_iter();
    Ok(std::array::from_fn(|_| {
        let data = outputs.next().expect("N outputs");
        DoubleTensor::new(&shape, data).expect("each output has one value per element of the shape")
    }))
}

/// The shape of a sample: the broadcast of the requested `shape` and the parameters' shapes. Samplers draw one
/// value per element of this shape so no draw is reused across broadcast elements.
pub(crate) fn sample_shape(shape: &[usize], params: &[&[usize]]) -> Result<Shape> {
    let mut shapes: Vec<&[usize]> = vec![shape];
    shapes.extend_from_slice(params);
    broadcast_all(&shapes)
}

/// `ln(1 + e^m)`, without overflow for large m or loss of precision for very negative m.
pub(crate) fn softplus(m: f64) -> f64 {
    if m > 0.0 {
        m + (-m).exp().ln_1p()
    } else {
        m.exp().ln_1p()
    }
}

/// `1 / (1 + e^-m)`, without overflow for any m.
pub(crate) fn sigmoid(m: f64) -> f64 {
    if m >= 0.0 {
        1.0 / (1.0 + (-m).exp())
    } else {
        let e = m.exp();
        e / (1.0 + e)
    }
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

/// An error unless `valid` holds for every element of `t`.
pub(crate) fn require<T: Element>(
    t: &Tensor<T>,
    valid: impl Fn(T) -> bool,
    message: impl FnOnce() -> String,
) -> Result<()> {
    if t.as_slice().iter().all(|&v| valid(v)) {
        Ok(())
    } else {
        crate::error::invalid_parameter(message())
    }
}

/// An error unless every element of `t` is positive (NaN is not).
pub(crate) fn require_positive(name: &str, t: &DoubleTensor) -> Result<()> {
    require(t, |v| v > 0.0, || format!("{name} must be positive"))
}

/// An error unless every element of `t` is a probability, in [0, 1] (NaN is not).
pub(crate) fn require_probability(name: &str, t: &DoubleTensor) -> Result<()> {
    require(
        t,
        |v| (0.0..=1.0).contains(&v),
        || format!("{name} must be in [0, 1]"),
    )
}

/// An error unless `max > min` element-wise, broadcasting (NaN bounds are not).
pub(crate) fn require_ordered<T: Element + PartialOrd>(
    min: &Tensor<T>,
    max: &Tensor<T>,
    message: &str,
) -> Result<()> {
    let shape = broadcast_all(&[min.shape(), max.shape()])?;
    let (lo, hi) = (operand(min, &shape)?, operand(max, &shape)?);
    let len: usize = shape.iter().product();
    if (0..len).all(|i| hi.at(i) > lo.at(i)) {
        Ok(())
    } else {
        crate::error::invalid_parameter(message)
    }
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

/// The least float greater than `x` (`f64::next_up`, which needs a newer Rust than the MSRV), for finite `x`.
pub(crate) fn next_up(x: f64) -> f64 {
    if x == 0.0 {
        f64::from_bits(1)
    } else if x > 0.0 {
        f64::from_bits(x.to_bits() + 1)
    } else {
        f64::from_bits(x.to_bits() - 1)
    }
}

/// The greatest float less than `x`, for finite `x`.
pub(crate) fn next_down(x: f64) -> f64 {
    -next_up(-x)
}

/// `x` moved strictly inside (lo, hi), for a draw that rounded onto a bound of an open support: the nearest float
/// inside. (Bounds that are adjacent floats leave nothing inside; `x` is then moved onto one of them.)
pub(crate) fn strictly_inside(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(next_up(lo)).min(next_down(hi))
}
