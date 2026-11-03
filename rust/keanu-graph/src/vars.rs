//! Typed handles to nodes being built, with operators and methods that add new nodes.

use crate::builder::GraphBuilder;
use crate::graph::NodeId;
use crate::op::{ArgOp, BinaryOp, CompareOp, LogicOp, Op, Predicate, ReduceOp, ScanOp, UnaryOp};
use crate::value::DType;
use keanu_tensor::{BoolTensor, DoubleTensor, IntTensor};
use std::fmt;
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Rem, Sub};

/// A typed handle to a node in a [`GraphBuilder`].
pub trait Var<'b>: Copy {
    /// The type of the node's value.
    const DTYPE: DType;
    fn id(&self) -> NodeId;
    fn builder(&self) -> &'b GraphBuilder;
    #[doc(hidden)]
    fn wrap(builder: &'b GraphBuilder, id: NodeId) -> Self;
}

macro_rules! handle {
    ($(#[$doc:meta])* $name:ident, $dtype:expr) => {
        $(#[$doc])*
        #[derive(Clone, Copy)]
        pub struct $name<'b> {
            builder: &'b GraphBuilder,
            id: NodeId,
        }

        impl<'b> $name<'b> {
            pub(crate) fn new(builder: &'b GraphBuilder, id: NodeId) -> Self {
                Self { builder, id }
            }

            pub fn id(self) -> NodeId {
                self.id
            }

            /// The node's shape.
            pub fn shape(self) -> Vec<usize> {
                self.builder.shape(self.id)
            }

            /// Labels the node. Labels must be unique within a graph.
            pub fn label(self, label: &str) -> Self {
                self.builder.set_label(self.id, label);
                self
            }

            fn unary_node(self, op: Op, dtype: DType) -> NodeId {
                self.builder.add(op, &[self.id], dtype)
            }

            fn same(self, op: Op) -> Self {
                Self::new(self.builder, self.unary_node(op, $dtype))
            }

            pub fn reshape(self, shape: &[isize]) -> Self {
                self.same(Op::Reshape(shape.to_vec()))
            }
            pub fn permute(self, permutation: &[usize]) -> Self {
                self.same(Op::Permute(permutation.to_vec()))
            }
            /// Swaps the last two dimensions.
            pub fn transpose(self) -> Self {
                self.same(Op::Transpose)
            }
            pub fn broadcast_to(self, shape: &[usize]) -> Self {
                self.same(Op::Broadcast(shape.to_vec()))
            }
            /// numpy style slicing, e.g. `"1, ::-1, 1:3"`.
            pub fn slice(self, spec: &str) -> Self {
                self.same(Op::Slice(spec.to_owned()))
            }
            /// The element at an index, as a scalar.
            pub fn take(self, index: &[usize]) -> Self {
                self.same(Op::Take(index.to_vec()))
            }
            pub fn diag(self) -> Self {
                self.same(Op::Diag)
            }
            pub fn diag_part(self) -> Self {
                self.same(Op::DiagPart)
            }
            /// numpy's tril: keeps elements with column - row <= k.
            pub fn tril(self, k: isize) -> Self {
                self.same(Op::Tril(k))
            }
            /// numpy's triu: keeps elements with column - row >= k.
            pub fn triu(self, k: isize) -> Self {
                self.same(Op::Triu(k))
            }
            pub fn fill_triangular(self, upper: bool, lower: bool) -> Self {
                self.same(Op::FillTriangular { upper, lower })
            }
            pub fn triangle_part(self, upper: bool) -> Self {
                self.same(Op::TrianglePart { upper })
            }
        }

        impl<'b> Var<'b> for $name<'b> {
            const DTYPE: DType = $dtype;
            fn id(&self) -> NodeId {
                self.id
            }
            fn builder(&self) -> &'b GraphBuilder {
                self.builder
            }
            fn wrap(builder: &'b GraphBuilder, id: NodeId) -> Self {
                Self::new(builder, id)
            }
        }

        impl fmt::Debug for $name<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}({:?})", stringify!($name), self.id)
            }
        }
    };
}

handle!(
    /// A handle to an f64 node.
    DoubleVar,
    DType::F64
);
handle!(
    /// A handle to an i64 node.
    IntVar,
    DType::I64
);
handle!(
    /// A handle to a bool node.
    BoolVar,
    DType::Bool
);

/// Things that can be used where an f64 node is expected: handles, scalars and tensors (which become constants).
pub trait IntoDoubleVar<'b> {
    fn into_var(self, builder: &'b GraphBuilder) -> DoubleVar<'b>;
}

impl<'b> IntoDoubleVar<'b> for DoubleVar<'b> {
    fn into_var(self, _: &'b GraphBuilder) -> DoubleVar<'b> {
        self
    }
}

impl<'b> IntoDoubleVar<'b> for f64 {
    fn into_var(self, builder: &'b GraphBuilder) -> DoubleVar<'b> {
        builder.scalar(self)
    }
}

impl<'b> IntoDoubleVar<'b> for DoubleTensor {
    fn into_var(self, builder: &'b GraphBuilder) -> DoubleVar<'b> {
        builder.constant(self)
    }
}

/// Things that can be used where an i64 node is expected.
pub trait IntoIntVar<'b> {
    fn into_var(self, builder: &'b GraphBuilder) -> IntVar<'b>;
}

impl<'b> IntoIntVar<'b> for IntVar<'b> {
    fn into_var(self, _: &'b GraphBuilder) -> IntVar<'b> {
        self
    }
}

impl<'b> IntoIntVar<'b> for i64 {
    fn into_var(self, builder: &'b GraphBuilder) -> IntVar<'b> {
        builder.int_scalar(self)
    }
}

impl<'b> IntoIntVar<'b> for IntTensor {
    fn into_var(self, builder: &'b GraphBuilder) -> IntVar<'b> {
        builder.int_constant(self)
    }
}

macro_rules! numeric_methods {
    ($name:ident, $into:ident) => {
        impl<'b> $name<'b> {
            fn binary(self, op: BinaryOp, other: impl $into<'b>) -> Self {
                let other = other.into_var(self.builder);
                Self::new(
                    self.builder,
                    self.builder
                        .add(Op::Binary(op), &[self.id, other.id()], Self::DTYPE),
                )
            }
            fn compare(self, op: CompareOp, other: impl $into<'b>) -> BoolVar<'b> {
                let other = other.into_var(self.builder);
                BoolVar::new(
                    self.builder,
                    self.builder
                        .add(Op::Compare(op), &[self.id, other.id()], DType::Bool),
                )
            }
            fn reduce(self, kind: ReduceOp, axes: Option<&[isize]>) -> Self {
                self.same(Op::Reduce {
                    kind,
                    axes: axes.map(<[isize]>::to_vec),
                })
            }
            fn arg(self, kind: ArgOp, axis: Option<isize>) -> IntVar<'b> {
                IntVar::new(
                    self.builder,
                    self.unary_node(Op::ArgReduce { kind, axis }, DType::I64),
                )
            }

            pub fn abs(self) -> Self {
                self.same(Op::Unary(UnaryOp::Abs))
            }
            pub fn sign(self) -> Self {
                self.same(Op::Unary(UnaryOp::Sign))
            }
            pub fn pow(self, exponent: impl $into<'b>) -> Self {
                self.binary(BinaryOp::Pow, exponent)
            }
            /// Element-wise maximum (NaN if either is NaN), broadcasting.
            pub fn maximum(self, other: impl $into<'b>) -> Self {
                self.binary(BinaryOp::Max, other)
            }
            /// Element-wise minimum (NaN if either is NaN), broadcasting.
            pub fn minimum(self, other: impl $into<'b>) -> Self {
                self.binary(BinaryOp::Min, other)
            }
            pub fn gt(self, other: impl $into<'b>) -> BoolVar<'b> {
                self.compare(CompareOp::Gt, other)
            }
            pub fn ge(self, other: impl $into<'b>) -> BoolVar<'b> {
                self.compare(CompareOp::Ge, other)
            }
            pub fn lt(self, other: impl $into<'b>) -> BoolVar<'b> {
                self.compare(CompareOp::Lt, other)
            }
            pub fn le(self, other: impl $into<'b>) -> BoolVar<'b> {
                self.compare(CompareOp::Le, other)
            }
            pub fn elementwise_eq(self, other: impl $into<'b>) -> BoolVar<'b> {
                self.compare(CompareOp::Eq, other)
            }
            pub fn elementwise_ne(self, other: impl $into<'b>) -> BoolVar<'b> {
                self.compare(CompareOp::Ne, other)
            }
            /// |self - other| <= epsilon, element-wise.
            pub fn approx_eq(self, other: impl $into<'b>, epsilon: f64) -> BoolVar<'b> {
                let other = other.into_var(self.builder);
                BoolVar::new(
                    self.builder,
                    self.builder
                        .add(Op::ApproxEq(epsilon), &[self.id, other.id()], DType::Bool),
                )
            }
            /// Sum of every element.
            pub fn sum(self) -> Self {
                self.reduce(ReduceOp::Sum, None)
            }
            pub fn sum_axes(self, axes: &[isize]) -> Self {
                self.reduce(ReduceOp::Sum, Some(axes))
            }
            pub fn product(self) -> Self {
                self.reduce(ReduceOp::Product, None)
            }
            pub fn product_axes(self, axes: &[isize]) -> Self {
                self.reduce(ReduceOp::Product, Some(axes))
            }
            /// The largest element.
            pub fn max(self) -> Self {
                self.reduce(ReduceOp::Max, None)
            }
            /// The smallest element.
            pub fn min(self) -> Self {
                self.reduce(ReduceOp::Min, None)
            }
            pub fn cumsum(self, axis: isize) -> Self {
                self.same(Op::Scan {
                    kind: ScanOp::CumSum,
                    axis,
                })
            }
            pub fn cumprod(self, axis: isize) -> Self {
                self.same(Op::Scan {
                    kind: ScanOp::CumProd,
                    axis,
                })
            }
            /// Flat index of the largest element (the first NaN if any).
            pub fn argmax(self) -> IntVar<'b> {
                self.arg(ArgOp::ArgMax, None)
            }
            pub fn argmin(self) -> IntVar<'b> {
                self.arg(ArgOp::ArgMin, None)
            }
            pub fn argmax_axis(self, axis: isize) -> IntVar<'b> {
                self.arg(ArgOp::ArgMax, Some(axis))
            }
            pub fn argmin_axis(self, axis: isize) -> IntVar<'b> {
                self.arg(ArgOp::ArgMin, Some(axis))
            }
            /// 1 -> true, anything else -> false.
            pub fn to_bool(self) -> BoolVar<'b> {
                BoolVar::new(
                    self.builder,
                    self.unary_node(Op::Cast(DType::Bool), DType::Bool),
                )
            }
        }

        impl<'b, R: $into<'b>> Add<R> for $name<'b> {
            type Output = Self;
            fn add(self, rhs: R) -> Self {
                self.binary(BinaryOp::Add, rhs)
            }
        }
        impl<'b, R: $into<'b>> Sub<R> for $name<'b> {
            type Output = Self;
            fn sub(self, rhs: R) -> Self {
                self.binary(BinaryOp::Sub, rhs)
            }
        }
        impl<'b, R: $into<'b>> Mul<R> for $name<'b> {
            type Output = Self;
            fn mul(self, rhs: R) -> Self {
                self.binary(BinaryOp::Mul, rhs)
            }
        }
        impl<'b, R: $into<'b>> Div<R> for $name<'b> {
            type Output = Self;
            fn div(self, rhs: R) -> Self {
                self.binary(BinaryOp::Div, rhs)
            }
        }
        impl Neg for $name<'_> {
            type Output = Self;
            fn neg(self) -> Self {
                self.same(Op::Unary(UnaryOp::Neg))
            }
        }
    };
}

numeric_methods!(DoubleVar, IntoDoubleVar);
numeric_methods!(IntVar, IntoIntVar);

/// Scalar on the left: `2.0 * x`, `1.0 - x`, ...
macro_rules! scalar_left {
    ($scalar:ty, $var:ident) => {
        impl<'b> Add<$var<'b>> for $scalar {
            type Output = $var<'b>;
            fn add(self, rhs: $var<'b>) -> $var<'b> {
                rhs.builder.constant_like::<$var<'b>>(self) + rhs
            }
        }
        impl<'b> Sub<$var<'b>> for $scalar {
            type Output = $var<'b>;
            fn sub(self, rhs: $var<'b>) -> $var<'b> {
                rhs.builder.constant_like::<$var<'b>>(self) - rhs
            }
        }
        impl<'b> Mul<$var<'b>> for $scalar {
            type Output = $var<'b>;
            fn mul(self, rhs: $var<'b>) -> $var<'b> {
                rhs.builder.constant_like::<$var<'b>>(self) * rhs
            }
        }
        impl<'b> Div<$var<'b>> for $scalar {
            type Output = $var<'b>;
            fn div(self, rhs: $var<'b>) -> $var<'b> {
                rhs.builder.constant_like::<$var<'b>>(self) / rhs
            }
        }
    };
}

/// Scalars that can become constants of a handle's type.
#[doc(hidden)]
pub trait ScalarConstant<'b>: Sized {
    type Scalar;
    fn make(builder: &'b GraphBuilder, value: Self::Scalar) -> Self;
}

impl<'b> ScalarConstant<'b> for DoubleVar<'b> {
    type Scalar = f64;
    fn make(builder: &'b GraphBuilder, value: f64) -> Self {
        builder.scalar(value)
    }
}

impl<'b> ScalarConstant<'b> for IntVar<'b> {
    type Scalar = i64;
    fn make(builder: &'b GraphBuilder, value: i64) -> Self {
        builder.int_scalar(value)
    }
}

impl GraphBuilder {
    #[doc(hidden)]
    pub fn constant_like<'b, V: ScalarConstant<'b>>(&'b self, value: V::Scalar) -> V {
        V::make(self, value)
    }
}

scalar_left!(f64, DoubleVar);
scalar_left!(i64, IntVar);

impl<'b> DoubleVar<'b> {
    fn map(self, op: UnaryOp) -> Self {
        self.same(Op::Unary(op))
    }
    fn mask(self, op: CompareOp, other: impl IntoDoubleVar<'b>) -> Self {
        let other = other.into_var(self.builder);
        Self::new(
            self.builder,
            self.builder
                .add(Op::Mask(op), &[self.id, other.id()], DType::F64),
        )
    }
    fn predicate(self, p: Predicate) -> BoolVar<'b> {
        BoolVar::new(self.builder, self.unary_node(Op::Predicate(p), DType::Bool))
    }

    pub fn reciprocal(self) -> Self {
        self.map(UnaryOp::Reciprocal)
    }
    pub fn exp(self) -> Self {
        self.map(UnaryOp::Exp)
    }
    pub fn exp2(self) -> Self {
        self.map(UnaryOp::Exp2)
    }
    pub fn exp_m1(self) -> Self {
        self.map(UnaryOp::ExpM1)
    }
    pub fn ln(self) -> Self {
        self.map(UnaryOp::Ln)
    }
    pub fn ln_1p(self) -> Self {
        self.map(UnaryOp::Ln1p)
    }
    pub fn log2(self) -> Self {
        self.map(UnaryOp::Log2)
    }
    pub fn log10(self) -> Self {
        self.map(UnaryOp::Log10)
    }
    pub fn sqrt(self) -> Self {
        self.map(UnaryOp::Sqrt)
    }
    pub fn sin(self) -> Self {
        self.map(UnaryOp::Sin)
    }
    pub fn cos(self) -> Self {
        self.map(UnaryOp::Cos)
    }
    pub fn tan(self) -> Self {
        self.map(UnaryOp::Tan)
    }
    pub fn asin(self) -> Self {
        self.map(UnaryOp::Asin)
    }
    pub fn acos(self) -> Self {
        self.map(UnaryOp::Acos)
    }
    pub fn atan(self) -> Self {
        self.map(UnaryOp::Atan)
    }
    pub fn sinh(self) -> Self {
        self.map(UnaryOp::Sinh)
    }
    pub fn cosh(self) -> Self {
        self.map(UnaryOp::Cosh)
    }
    pub fn tanh(self) -> Self {
        self.map(UnaryOp::Tanh)
    }
    pub fn asinh(self) -> Self {
        self.map(UnaryOp::Asinh)
    }
    pub fn acosh(self) -> Self {
        self.map(UnaryOp::Acosh)
    }
    pub fn atanh(self) -> Self {
        self.map(UnaryOp::Atanh)
    }
    pub fn sigmoid(self) -> Self {
        self.map(UnaryOp::Sigmoid)
    }
    pub fn floor(self) -> Self {
        self.map(UnaryOp::Floor)
    }
    pub fn ceil(self) -> Self {
        self.map(UnaryOp::Ceil)
    }
    /// Rounds half away from zero.
    pub fn round(self) -> Self {
        self.map(UnaryOp::Round)
    }
    pub fn ln_gamma(self) -> Self {
        self.map(UnaryOp::LnGamma)
    }
    pub fn digamma(self) -> Self {
        self.map(UnaryOp::Digamma)
    }
    pub fn trigamma(self) -> Self {
        self.map(UnaryOp::Trigamma)
    }

    /// Element-wise atan2(y, self).
    pub fn atan2(self, y: impl IntoDoubleVar<'b>) -> Self {
        self.binary(BinaryOp::Atan2, y)
    }
    pub fn log_add_exp(self, other: impl IntoDoubleVar<'b>) -> Self {
        self.binary(BinaryOp::LogAddExp, other)
    }
    pub fn log_add_exp2(self, other: impl IntoDoubleVar<'b>) -> Self {
        self.binary(BinaryOp::LogAddExp2, other)
    }
    /// y * ln(self), taken to be 0 where y is 0.
    pub fn safe_ln_times(self, y: impl IntoDoubleVar<'b>) -> Self {
        self.binary(BinaryOp::SafeLnTimes, y)
    }

    /// 1.0 where self > other, else 0.0.
    pub fn gt_mask(self, other: impl IntoDoubleVar<'b>) -> Self {
        self.mask(CompareOp::Gt, other)
    }
    pub fn ge_mask(self, other: impl IntoDoubleVar<'b>) -> Self {
        self.mask(CompareOp::Ge, other)
    }
    pub fn lt_mask(self, other: impl IntoDoubleVar<'b>) -> Self {
        self.mask(CompareOp::Lt, other)
    }
    pub fn le_mask(self, other: impl IntoDoubleVar<'b>) -> Self {
        self.mask(CompareOp::Le, other)
    }

    pub fn is_nan(self) -> BoolVar<'b> {
        self.predicate(Predicate::IsNan)
    }
    pub fn not_nan(self) -> BoolVar<'b> {
        self.predicate(Predicate::NotNan)
    }
    pub fn is_finite(self) -> BoolVar<'b> {
        self.predicate(Predicate::IsFinite)
    }
    pub fn is_infinite(self) -> BoolVar<'b> {
        self.predicate(Predicate::IsInfinite)
    }
    pub fn is_positive_infinity(self) -> BoolVar<'b> {
        self.predicate(Predicate::IsPositiveInfinity)
    }
    pub fn is_negative_infinity(self) -> BoolVar<'b> {
        self.predicate(Predicate::IsNegativeInfinity)
    }

    pub fn mean(self) -> Self {
        self.reduce(ReduceOp::Mean, None)
    }
    pub fn mean_axes(self, axes: &[isize]) -> Self {
        self.reduce(ReduceOp::Mean, Some(axes))
    }
    /// Sample standard deviation of every element.
    pub fn std_dev(self) -> Self {
        self.reduce(ReduceOp::StdDev, None)
    }
    /// (x - mean) / std_dev over every element.
    pub fn standardize(self) -> Self {
        self.same(Op::Standardize)
    }
    /// Flat index of the largest element, ignoring NaN.
    pub fn nan_argmax(self) -> IntVar<'b> {
        self.arg(ArgOp::NanArgMax, None)
    }
    pub fn nan_argmin(self) -> IntVar<'b> {
        self.arg(ArgOp::NanArgMin, None)
    }

    pub fn matmul(self, other: impl IntoDoubleVar<'b>) -> Self {
        self.matmul_transposed(other, false, false)
    }
    pub fn matmul_transposed(
        self,
        other: impl IntoDoubleVar<'b>,
        transpose_left: bool,
        transpose_right: bool,
    ) -> Self {
        let other = other.into_var(self.builder);
        Self::new(
            self.builder,
            self.builder.add(
                Op::MatMul {
                    transpose_left,
                    transpose_right,
                },
                &[self.id, other.id()],
                DType::F64,
            ),
        )
    }
    pub fn tensor_multiply(
        self,
        other: impl IntoDoubleVar<'b>,
        left_axes: &[isize],
        right_axes: &[isize],
    ) -> Self {
        let other = other.into_var(self.builder);
        let op = Op::TensorMultiply {
            left_axes: left_axes.to_vec(),
            right_axes: right_axes.to_vec(),
        };
        Self::new(
            self.builder,
            self.builder.add(op, &[self.id, other.id()], DType::F64),
        )
    }
    pub fn inverse(self) -> Self {
        self.same(Op::Inverse)
    }
    pub fn determinant(self) -> Self {
        self.same(Op::Determinant)
    }
    pub fn cholesky(self) -> Self {
        self.same(Op::Cholesky)
    }
    /// The full symmetric inverse of the matrix whose lower Cholesky factor this is.
    pub fn cholesky_inverse(self) -> Self {
        self.same(Op::CholeskyInverse)
    }

    /// Truncates towards zero.
    pub fn to_int(self) -> IntVar<'b> {
        IntVar::new(
            self.builder,
            self.unary_node(Op::Cast(DType::I64), DType::I64),
        )
    }
    /// This value with `value` wherever `mask` is 1.0. The mask must have the same shape.
    pub fn set_with_mask(
        self,
        mask: impl IntoDoubleVar<'b>,
        value: impl IntoDoubleVar<'b>,
    ) -> Self {
        let (mask, value) = (mask.into_var(self.builder), value.into_var(self.builder));
        Self::new(
            self.builder,
            self.builder.add(
                Op::SetWithMask,
                &[self.id, mask.id(), value.id()],
                DType::F64,
            ),
        )
    }
    pub fn replace_nan(self, with: f64) -> Self {
        self.same(Op::ReplaceNan(with))
    }
}

impl<'b> IntVar<'b> {
    pub fn to_double(self) -> DoubleVar<'b> {
        DoubleVar::new(
            self.builder,
            self.unary_node(Op::Cast(DType::F64), DType::F64),
        )
    }
}

impl<'b, R: IntoIntVar<'b>> Rem<R> for IntVar<'b> {
    type Output = Self;
    fn rem(self, rhs: R) -> Self {
        self.binary(BinaryOp::Rem, rhs)
    }
}

/// Things that can be used where a bool node is expected.
pub trait IntoBoolVar<'b> {
    fn into_var(self, builder: &'b GraphBuilder) -> BoolVar<'b>;
}

impl<'b> IntoBoolVar<'b> for BoolVar<'b> {
    fn into_var(self, _: &'b GraphBuilder) -> BoolVar<'b> {
        self
    }
}

impl<'b> IntoBoolVar<'b> for bool {
    fn into_var(self, builder: &'b GraphBuilder) -> BoolVar<'b> {
        builder.bool_scalar(self)
    }
}

impl<'b> IntoBoolVar<'b> for BoolTensor {
    fn into_var(self, builder: &'b GraphBuilder) -> BoolVar<'b> {
        builder.bool_constant(self)
    }
}

impl<'b> BoolVar<'b> {
    fn logic(self, op: LogicOp, other: impl IntoBoolVar<'b>) -> Self {
        let other = other.into_var(self.builder);
        Self::new(
            self.builder,
            self.builder
                .add(Op::Logic(op), &[self.id, other.id()], DType::Bool),
        )
    }
    pub fn and(self, other: impl IntoBoolVar<'b>) -> Self {
        self.logic(LogicOp::And, other)
    }
    pub fn or(self, other: impl IntoBoolVar<'b>) -> Self {
        self.logic(LogicOp::Or, other)
    }
    pub fn xor(self, other: impl IntoBoolVar<'b>) -> Self {
        self.logic(LogicOp::Xor, other)
    }
    /// True if every element is true.
    pub fn all(self) -> Self {
        self.same(Op::All)
    }
    /// True if any element is true.
    pub fn any(self) -> Self {
        self.same(Op::Any)
    }
    pub fn elementwise_eq(self, other: impl IntoBoolVar<'b>) -> Self {
        let other = other.into_var(self.builder);
        Self::new(
            self.builder,
            self.builder.add(
                Op::Compare(CompareOp::Eq),
                &[self.id, other.id()],
                DType::Bool,
            ),
        )
    }
    /// 1.0 where true, 0.0 where false.
    pub fn to_double_mask(self) -> DoubleVar<'b> {
        DoubleVar::new(
            self.builder,
            self.unary_node(Op::Cast(DType::F64), DType::F64),
        )
    }
    /// 1 where true, 0 where false.
    pub fn to_int_mask(self) -> IntVar<'b> {
        IntVar::new(
            self.builder,
            self.unary_node(Op::Cast(DType::I64), DType::I64),
        )
    }
    /// The predicate itself, but evaluating it fails if any element is false.
    pub fn assert(self, message: Option<&str>) -> Self {
        self.same(Op::Assert(message.map(str::to_owned)))
    }
}

impl<'b, R: IntoBoolVar<'b>> BitAnd<R> for BoolVar<'b> {
    type Output = Self;
    fn bitand(self, rhs: R) -> Self {
        self.and(rhs)
    }
}

impl<'b, R: IntoBoolVar<'b>> BitOr<R> for BoolVar<'b> {
    type Output = Self;
    fn bitor(self, rhs: R) -> Self {
        self.or(rhs)
    }
}

impl<'b, R: IntoBoolVar<'b>> BitXor<R> for BoolVar<'b> {
    type Output = Self;
    fn bitxor(self, rhs: R) -> Self {
        self.xor(rhs)
    }
}

impl Not for BoolVar<'_> {
    type Output = Self;
    fn not(self) -> Self {
        self.same(Op::Not)
    }
}
