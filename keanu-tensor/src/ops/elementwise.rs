//! Element-wise arithmetic, math functions, comparisons, logic and casts.

use crate::error::{Result, TensorError};
use crate::special;
use crate::tensor::{BoolTensor, DoubleTensor, Element, IntTensor, Tensor};

/// Element types with arithmetic: `f64` and `i64`.
///
/// Integer arithmetic wraps on overflow. `max`/`min` propagate NaN and order -0.0 below +0.0.
pub trait Numeric: Element + PartialOrd {
    /// The additive identity.
    const ZERO: Self;
    /// The multiplicative identity.
    const ONE: Self;
    /// self + other.
    fn add(self, other: Self) -> Self;
    /// self - other.
    fn sub(self, other: Self) -> Self;
    /// self × other.
    fn mul(self, other: Self) -> Self;
    /// The negation.
    fn neg(self) -> Self;
    /// The absolute value.
    fn abs(self) -> Self;
    /// The larger value (NaN if either is NaN; +0.0 is larger than -0.0).
    fn max(self, other: Self) -> Self;
    /// The smaller value (NaN if either is NaN; -0.0 is smaller than +0.0).
    fn min(self, other: Self) -> Self;
    /// True if this value should win an arg max / arg min (NaN wins, as in numpy).
    fn is_nan(self) -> bool;
    /// The value as f64.
    fn to_f64(self) -> f64;
}

impl Numeric for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    fn add(self, o: Self) -> Self {
        self + o
    }
    fn sub(self, o: Self) -> Self {
        self - o
    }
    fn mul(self, o: Self) -> Self {
        self * o
    }
    fn neg(self) -> Self {
        -self
    }
    fn abs(self) -> Self {
        f64::abs(self)
    }
    fn max(self, o: Self) -> Self {
        if self.is_nan() || o.is_nan() {
            f64::NAN
        } else if self == o {
            // -0.0 == 0.0: prefer +0.0
            if self.is_sign_negative() { o } else { self }
        } else if self > o {
            self
        } else {
            o
        }
    }
    fn min(self, o: Self) -> Self {
        if self.is_nan() || o.is_nan() {
            f64::NAN
        } else if self == o {
            if self.is_sign_negative() { self } else { o }
        } else if self < o {
            self
        } else {
            o
        }
    }
    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }
    fn to_f64(self) -> f64 {
        self
    }
}

impl Numeric for i64 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    fn add(self, o: Self) -> Self {
        self.wrapping_add(o)
    }
    fn sub(self, o: Self) -> Self {
        self.wrapping_sub(o)
    }
    fn mul(self, o: Self) -> Self {
        self.wrapping_mul(o)
    }
    fn neg(self) -> Self {
        self.wrapping_neg()
    }
    fn abs(self) -> Self {
        self.wrapping_abs()
    }
    fn max(self, o: Self) -> Self {
        Ord::max(self, o)
    }
    fn min(self, o: Self) -> Self {
        Ord::min(self, o)
    }
    fn is_nan(self) -> bool {
        false
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl<T: Numeric> Tensor<T> {
    /// Element-wise, broadcasting: self + other.
    pub fn add(&self, other: &Self) -> Result<Self> {
        self.zip_with(other, T::add)
    }
    /// Element-wise, broadcasting: self - other.
    pub fn sub(&self, other: &Self) -> Result<Self> {
        self.zip_with(other, T::sub)
    }
    /// Element-wise, broadcasting: self × other.
    pub fn mul(&self, other: &Self) -> Result<Self> {
        self.zip_with(other, T::mul)
    }
    /// Element-wise maximum (NaN if either is NaN).
    pub fn maximum(&self, other: &Self) -> Result<Self> {
        self.zip_with(other, T::max)
    }
    /// Element-wise minimum (NaN if either is NaN).
    pub fn minimum(&self, other: &Self) -> Result<Self> {
        self.zip_with(other, T::min)
    }
    /// Element-wise: the negation.
    pub fn neg(&self) -> Self {
        self.map(T::neg)
    }
    /// Element-wise: the absolute value.
    pub fn abs(&self) -> Self {
        self.map(T::abs)
    }

    /// Element-wise, broadcasting: whether self > other.
    pub fn gt(&self, other: &Self) -> Result<BoolTensor> {
        self.zip_with(other, |a, b| a > b)
    }
    /// Element-wise, broadcasting: whether self >= other.
    pub fn ge(&self, other: &Self) -> Result<BoolTensor> {
        self.zip_with(other, |a, b| a >= b)
    }
    /// Element-wise, broadcasting: whether self < other.
    pub fn lt(&self, other: &Self) -> Result<BoolTensor> {
        self.zip_with(other, |a, b| a < b)
    }
    /// Element-wise, broadcasting: whether self <= other.
    pub fn le(&self, other: &Self) -> Result<BoolTensor> {
        self.zip_with(other, |a, b| a <= b)
    }
    /// Element-wise equality (IEEE: NaN is never equal, -0.0 equals 0.0).
    pub fn elementwise_eq(&self, other: &Self) -> Result<BoolTensor> {
        self.zip_with(other, |a, b| a == b)
    }
    /// Element-wise, broadcasting: whether self != other.
    pub fn elementwise_ne(&self, other: &Self) -> Result<BoolTensor> {
        self.zip_with(other, |a, b| a != b)
    }
}

macro_rules! unary {
    ($($(#[$doc:meta])* $name:ident => $f:expr),* $(,)?) => {
        impl DoubleTensor {
            $($(#[$doc])* #[doc = concat!("Element-wise `", stringify!($name), "`.")] pub fn $name(&self) -> Self { self.map($f) })*
        }
    };
}

/// Sign of x: -1, 0 or 1, keeping the sign of zero and NaN.
fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        x
    }
}

unary! {
    reciprocal => |x| 1.0 / x,
    /// -1, 0 or 1 (keeping the sign of zero; NaN stays NaN).
    sign => sign,
    exp => f64::exp,
    exp2 => f64::exp2,
    exp_m1 => f64::exp_m1,
    ln => f64::ln,
    ln_1p => f64::ln_1p,
    log2 => f64::log2,
    log10 => f64::log10,
    sqrt => f64::sqrt,
    sin => f64::sin,
    cos => f64::cos,
    tan => f64::tan,
    asin => f64::asin,
    acos => f64::acos,
    atan => f64::atan,
    sinh => f64::sinh,
    cosh => f64::cosh,
    tanh => f64::tanh,
    asinh => f64::asinh,
    acosh => f64::acosh,
    atanh => f64::atanh,
    /// 1 / (1 + e^-x)
    sigmoid => |x| 1.0 / (1.0 + (-x).exp()),
    floor => f64::floor,
    ceil => f64::ceil,
    /// Rounds half away from zero.
    round => f64::round,
    /// ln Γ(x), NaN for x <= 0.
    ln_gamma => special::ln_gamma,
    digamma => special::digamma,
    trigamma => special::trigamma,
}

impl DoubleTensor {
    /// Element-wise, broadcasting: self / other.
    pub fn div(&self, other: &Self) -> Result<Self> {
        self.zip_with(other, |a, b| a / b)
    }
    /// Element-wise, broadcasting: self raised to the power other.
    pub fn pow(&self, exponent: &Self) -> Result<Self> {
        self.zip_with(exponent, f64::powf)
    }
    /// Element-wise atan2(y, self), the angle of the point (self, y).
    pub fn atan2(&self, y: &Self) -> Result<Self> {
        self.zip_with(y, |x, y| y.atan2(x))
    }
    /// ln(e^a + e^b) computed without overflow.
    pub fn log_add_exp(&self, other: &Self) -> Result<Self> {
        self.zip_with(other, |a, b| {
            let max = Numeric::max(a, b);
            if max.is_infinite() {
                max
            } else {
                max + ((a - max).exp() + (b - max).exp()).ln()
            }
        })
    }
    /// log2(2^a + 2^b) computed without overflow.
    pub fn log_add_exp2(&self, other: &Self) -> Result<Self> {
        self.zip_with(other, |a, b| {
            let max = Numeric::max(a, b);
            if max.is_infinite() {
                max
            } else {
                max + ((a - max).exp2() + (b - max).exp2()).ln() / std::f64::consts::LN_2
            }
        })
    }
    /// y * ln(self), taken to be 0 where y is 0 (even when self is 0).
    pub fn safe_ln_times(&self, y: &Self) -> Result<Self> {
        self.zip_with(y, |x, y| if y == 0.0 { 0.0 } else { x.ln() * y })
    }

    /// 1.0 where self > other, else 0.0.
    pub fn gt_mask(&self, other: &Self) -> Result<Self> {
        self.zip_with(other, |a, b| f64::from(u8::from(a > b)))
    }
    /// Element-wise, broadcasting: 1.0 where self >= other, else 0.0.
    pub fn ge_mask(&self, other: &Self) -> Result<Self> {
        self.zip_with(other, |a, b| f64::from(u8::from(a >= b)))
    }
    /// Element-wise, broadcasting: 1.0 where self < other, else 0.0.
    pub fn lt_mask(&self, other: &Self) -> Result<Self> {
        self.zip_with(other, |a, b| f64::from(u8::from(a < b)))
    }
    /// Element-wise, broadcasting: 1.0 where self <= other, else 0.0.
    pub fn le_mask(&self, other: &Self) -> Result<Self> {
        self.zip_with(other, |a, b| f64::from(u8::from(a <= b)))
    }

    /// Truncates towards zero; NaN becomes 0 and out of range values saturate.
    pub fn to_int(&self) -> IntTensor {
        self.map(|x| x as i64)
    }
}

/// Exact integer power. Negative exponents give 1 / base^|exponent| truncated towards zero (1 for base 1, ±1 for
/// base -1, 0 otherwise); zero to a negative power is an error. Overflow wraps.
pub fn int_pow(base: i64, exponent: i64) -> Result<i64> {
    if exponent < 0 {
        return match base {
            0 => Err(TensorError::DivisionByZero),
            1 => Ok(1),
            -1 => Ok(if exponent % 2 == 0 { 1 } else { -1 }),
            _ => Ok(0),
        };
    }
    let (mut result, mut b, mut e) = (1i64, base, exponent);
    while e > 0 {
        if e & 1 == 1 {
            result = result.wrapping_mul(b);
        }
        b = b.wrapping_mul(b);
        e >>= 1;
    }
    Ok(result)
}

impl IntTensor {
    /// Division truncating towards zero. Division by zero is an error.
    pub fn div(&self, other: &Self) -> Result<Self> {
        self.try_zip_with(other, |a, b| {
            if b == 0 {
                Err(TensorError::DivisionByZero)
            } else {
                Ok(a.wrapping_div(b))
            }
        })
    }
    /// Remainder with the sign of the dividend (as `%`). Division by zero is an error.
    pub fn rem(&self, other: &Self) -> Result<Self> {
        self.try_zip_with(other, |a, b| {
            if b == 0 {
                Err(TensorError::DivisionByZero)
            } else {
                Ok(a.wrapping_rem(b))
            }
        })
    }
    /// See [`int_pow`].
    pub fn pow(&self, exponent: &Self) -> Result<Self> {
        self.try_zip_with(exponent, int_pow)
    }
    /// Element-wise: the sign, -1, 0 or 1.
    pub fn sign(&self) -> Self {
        self.map(i64::signum)
    }
    /// Converts to f64.
    pub fn to_double(&self) -> DoubleTensor {
        self.map(|x| x as f64)
    }
}

impl BoolTensor {
    /// Element-wise, broadcasting: the logical and.
    pub fn and(&self, other: &Self) -> Result<Self> {
        self.zip_with(other, |a, b| a && b)
    }
    /// Element-wise, broadcasting: the logical or.
    pub fn or(&self, other: &Self) -> Result<Self> {
        self.zip_with(other, |a, b| a || b)
    }
    /// Element-wise, broadcasting: the logical exclusive or.
    pub fn xor(&self, other: &Self) -> Result<Self> {
        self.zip_with(other, |a, b| a != b)
    }
    /// Element-wise: the logical negation.
    pub fn not(&self) -> Self {
        self.map(|a| !a)
    }
    /// Whether every element is true.
    pub fn all(&self) -> bool {
        self.data.iter().all(|&a| a)
    }
    /// Whether any element is true.
    pub fn any(&self) -> bool {
        self.data.iter().any(|&a| a)
    }
    /// 1.0 where true, 0.0 where false.
    pub fn to_double_mask(&self) -> DoubleTensor {
        self.map(|a| f64::from(u8::from(a)))
    }
    /// 1 where true, 0 where false.
    pub fn to_int_mask(&self) -> IntTensor {
        self.map(i64::from)
    }
}
