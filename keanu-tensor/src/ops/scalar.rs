//! Constructors and operators between a tensor and a scalar, which cannot fail.

use crate::ops::elementwise::Numeric;
use crate::tensor::{DoubleTensor, Tensor};
use std::ops::{Add, Div, Mul, Neg, Sub};

impl<T: Numeric> Tensor<T> {
    /// A tensor of zeros.
    pub fn zeros(shape: &[usize]) -> Self {
        Self::full(shape, T::ZERO)
    }
    /// A tensor of ones.
    pub fn ones(shape: &[usize]) -> Self {
        Self::full(shape, T::ONE)
    }
}

impl DoubleTensor {
    /// Values from `start` (inclusive) to `end` (exclusive) in steps of 1.
    pub fn arange(start: f64, end: f64) -> Self {
        let n = (end - start).ceil().max(0.0) as usize;
        Self::vector((0..n).map(|i| start + i as f64).collect())
    }

    /// The n × n identity matrix.
    pub fn eye(n: usize) -> Self {
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Tensor::from_parts([n, n].as_slice().into(), data)
    }
}

impl<T: Numeric> Neg for &Tensor<T> {
    type Output = Tensor<T>;
    fn neg(self) -> Tensor<T> {
        self.map(T::neg)
    }
}

impl<T: Numeric> Neg for Tensor<T> {
    type Output = Tensor<T>;
    fn neg(self) -> Tensor<T> {
        -&self
    }
}

macro_rules! scalar_ops {
    ($t:ty) => {
        impl Add<$t> for &Tensor<$t> {
            type Output = Tensor<$t>;
            fn add(self, rhs: $t) -> Tensor<$t> {
                self.map(|x| Numeric::add(x, rhs))
            }
        }
        impl Sub<$t> for &Tensor<$t> {
            type Output = Tensor<$t>;
            fn sub(self, rhs: $t) -> Tensor<$t> {
                self.map(|x| Numeric::sub(x, rhs))
            }
        }
        impl Mul<$t> for &Tensor<$t> {
            type Output = Tensor<$t>;
            fn mul(self, rhs: $t) -> Tensor<$t> {
                self.map(|x| Numeric::mul(x, rhs))
            }
        }
        impl Add<$t> for Tensor<$t> {
            type Output = Tensor<$t>;
            fn add(self, rhs: $t) -> Tensor<$t> {
                &self + rhs
            }
        }
        impl Sub<$t> for Tensor<$t> {
            type Output = Tensor<$t>;
            fn sub(self, rhs: $t) -> Tensor<$t> {
                &self - rhs
            }
        }
        impl Mul<$t> for Tensor<$t> {
            type Output = Tensor<$t>;
            fn mul(self, rhs: $t) -> Tensor<$t> {
                &self * rhs
            }
        }
    };
}

scalar_ops!(f64);
scalar_ops!(i64);

impl Div<f64> for &DoubleTensor {
    type Output = DoubleTensor;
    fn div(self, rhs: f64) -> DoubleTensor {
        self.map(|x| x / rhs)
    }
}

impl Div<f64> for DoubleTensor {
    type Output = DoubleTensor;
    fn div(self, rhs: f64) -> DoubleTensor {
        &self / rhs
    }
}
