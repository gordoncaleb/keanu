//! Distribution parameters, owned or borrowed.

use keanu_tensor::{Element, Tensor};
use std::borrow::Cow;

/// A distribution parameter: a tensor the distribution either owns or borrows.
///
/// Distributions borrow their parameters when built from references, so evaluating a log probability from
/// tensors held elsewhere (for example a graph's node values) doesn't copy them.
pub type Param<'a, T = f64> = Cow<'a, Tensor<T>>;

/// An owned or borrowed tensor that converts into a [`Param`]. Every distribution constructor takes its
/// parameters as `impl IntoParam`, so both `Gaussian::new(mu, sigma)` and `Gaussian::new(&mu, &sigma)` work.
pub trait IntoParam<'a, T: Element> {
    /// The tensor as a parameter.
    fn into_param(self) -> Param<'a, T>;
}

impl<'a, T: Element> IntoParam<'a, T> for Tensor<T> {
    fn into_param(self) -> Param<'a, T> {
        Cow::Owned(self)
    }
}

impl<'a, T: Element> IntoParam<'a, T> for &'a Tensor<T> {
    fn into_param(self) -> Param<'a, T> {
        Cow::Borrowed(self)
    }
}

impl<'a, T: Element> IntoParam<'a, T> for Cow<'a, Tensor<T>> {
    fn into_param(self) -> Param<'a, T> {
        self
    }
}
