//! Typed handles to nodes being built, with operators and methods that add new nodes.

use crate::builder::GraphBuilder;
use crate::error::GraphError;
use crate::graph::NodeId;
use crate::op::{
    ArgOp, BesselOp, BinaryOp, CompareOp, CumulativeOp, LogicOp, Op, Predicate, ReduceOp, UnaryOp,
};
use crate::value::DType;
use keanu_tensor::{BoolTensor, DoubleTensor, IntTensor};
use std::fmt;
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Rem, Sub};

/// A typed handle to a node in a [`GraphBuilder`].
pub trait Var<'b>: Copy {
    /// The type of the node's value.
    const DTYPE: DType;
    /// The node this handle refers to.
    fn id(&self) -> NodeId;
    /// The builder the node belongs to.
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

            /// The node this handle refers to.
            pub fn id(self) -> NodeId {
                self.id
            }

            /// The node's shape.
            pub fn shape(self) -> Vec<usize> {
                self.builder.dtype_shape(self.id).1.to_vec()
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

            /// Reshapes to a shape with the same length (one dimension may be -1).
            pub fn reshape(self, shape: &[isize]) -> Self {
                self.same(Op::Reshape(shape.to_vec()))
            }
            /// Rearranges the dimensions.
            pub fn permute(self, permutation: &[usize]) -> Self {
                self.same(Op::Permute(permutation.to_vec()))
            }
            /// Swaps the last two dimensions.
            pub fn transpose(self) -> Self {
                self.same(Op::Transpose)
            }
            /// Broadcasts to the given shape.
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
            /// The elements selected along `axis` by `indices`, as numpy's `take(x, indices, axis)`: shape
            /// `shape[..axis] ++ indices.shape ++ shape[axis + 1..]`. `mu.gather(0, group)` is the hierarchical
            /// `mu[group]`. The indices may be a node whose value changes (e.g. a categorical latent); indices
            /// outside `0..shape[axis]` fail evaluation. Gradients flow to self, not to the indices.
            pub fn gather(self, axis: isize, indices: impl IntoIntVar<'b>) -> Self {
                let indices = indices.into_var(self.builder);
                Self::new(
                    self.builder,
                    self.builder
                        .add(Op::Gather { axis }, &[self.id, indices.id()], $dtype),
                )
            }
            /// A diagonal matrix from each vector in the last dimension.
            pub fn diag(self) -> Self {
                self.same(Op::Diag)
            }
            /// The main diagonal of each matrix.
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
            /// Fills the lower and/or upper triangle of a square matrix from vectors of length n(n + 1) / 2.
            pub fn fill_triangular(self, upper: bool, lower: bool) -> Self {
                self.same(Op::FillTriangular { upper, lower })
            }
            /// The elements of the upper or lower triangle of each matrix, as a vector.
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
    /// The handle, adding a constant node for plain values.
    fn into_var(self, builder: &'b GraphBuilder) -> DoubleVar<'b>;
}

impl<'b> IntoDoubleVar<'b> for DoubleVar<'b> {
    fn into_var(self, builder: &'b GraphBuilder) -> DoubleVar<'b> {
        builder.own(self)
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
    /// The handle, adding a constant node for plain values.
    fn into_var(self, builder: &'b GraphBuilder) -> IntVar<'b>;
}

impl<'b> IntoIntVar<'b> for IntVar<'b> {
    fn into_var(self, builder: &'b GraphBuilder) -> IntVar<'b> {
        builder.own(self)
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

            /// The adjoint of [`gather`](Self::gather): zeros of shape `shape[..axis] ++ [size] ++ shape[axis + k..]`
            /// (k the rank of `indices`) with each element added at its index along `axis`, so repeated indices
            /// accumulate (e.g. per-group totals). Self's dimensions `axis..axis + k` must equal the indices'
            /// shape; indices outside `0..size` fail evaluation.
            pub fn scatter_add(
                self,
                axis: isize,
                indices: impl IntoIntVar<'b>,
                size: usize,
            ) -> Self {
                let indices = indices.into_var(self.builder);
                Self::new(
                    self.builder,
                    self.builder.add(
                        Op::ScatterAdd { axis, size },
                        &[self.id, indices.id()],
                        Self::DTYPE,
                    ),
                )
            }

            /// A node computing the absolute value.
            pub fn abs(self) -> Self {
                self.same(Op::Unary(UnaryOp::Abs))
            }
            /// A node computing the sign (-1, 0 or 1; zero keeps its sign and NaN stays NaN).
            pub fn sign(self) -> Self {
                self.same(Op::Unary(UnaryOp::Sign))
            }
            /// A node computing self raised to the power other.
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
            /// A node computing whether self > other.
            pub fn gt(self, other: impl $into<'b>) -> BoolVar<'b> {
                self.compare(CompareOp::Gt, other)
            }
            /// A node computing whether self >= other.
            pub fn ge(self, other: impl $into<'b>) -> BoolVar<'b> {
                self.compare(CompareOp::Ge, other)
            }
            /// A node computing whether self < other.
            pub fn lt(self, other: impl $into<'b>) -> BoolVar<'b> {
                self.compare(CompareOp::Lt, other)
            }
            /// A node computing whether self <= other.
            pub fn le(self, other: impl $into<'b>) -> BoolVar<'b> {
                self.compare(CompareOp::Le, other)
            }
            /// A node computing whether self == other (IEEE: NaN is never equal).
            pub fn elementwise_eq(self, other: impl $into<'b>) -> BoolVar<'b> {
                self.compare(CompareOp::Eq, other)
            }
            /// A node computing whether self != other.
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
            /// The sum over the given axes, which are removed.
            pub fn sum_axes(self, axes: &[isize]) -> Self {
                self.reduce(ReduceOp::Sum, Some(axes))
            }
            /// The product of every element.
            pub fn product(self) -> Self {
                self.reduce(ReduceOp::Product, None)
            }
            /// The product over the given axes, which are removed.
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
            /// The cumulative sum along an axis.
            pub fn cumsum(self, axis: isize) -> Self {
                self.same(Op::Cumulative {
                    kind: CumulativeOp::CumSum,
                    axis,
                })
            }
            /// The cumulative product along an axis.
            pub fn cumprod(self, axis: isize) -> Self {
                self.same(Op::Cumulative {
                    kind: CumulativeOp::CumProd,
                    axis,
                })
            }
            /// Flat index of the largest element (the first NaN if any).
            pub fn argmax(self) -> IntVar<'b> {
                self.arg(ArgOp::ArgMax, None)
            }
            /// The flat index of the smallest element (the first NaN wins).
            pub fn argmin(self) -> IntVar<'b> {
                self.arg(ArgOp::ArgMin, None)
            }
            /// The index of the largest element along an axis (the first NaN wins).
            pub fn argmax_axis(self, axis: isize) -> IntVar<'b> {
                self.arg(ArgOp::ArgMax, Some(axis))
            }
            /// The index of the smallest element along an axis (the first NaN wins).
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

    /// A node computing 1 / x.
    pub fn reciprocal(self) -> Self {
        self.map(UnaryOp::Reciprocal)
    }
    /// A node computing eˣ.
    pub fn exp(self) -> Self {
        self.map(UnaryOp::Exp)
    }
    /// A node computing 2ˣ.
    pub fn exp2(self) -> Self {
        self.map(UnaryOp::Exp2)
    }
    /// A node computing eˣ - 1 (accurate near 0).
    pub fn exp_m1(self) -> Self {
        self.map(UnaryOp::ExpM1)
    }
    /// A node computing the natural logarithm.
    pub fn ln(self) -> Self {
        self.map(UnaryOp::Ln)
    }
    /// A node computing ln(1 + x) (accurate near 0).
    pub fn ln_1p(self) -> Self {
        self.map(UnaryOp::Ln1p)
    }
    /// A node computing the base 2 logarithm.
    pub fn log2(self) -> Self {
        self.map(UnaryOp::Log2)
    }
    /// A node computing the base 10 logarithm.
    pub fn log10(self) -> Self {
        self.map(UnaryOp::Log10)
    }
    /// A node computing the square root.
    pub fn sqrt(self) -> Self {
        self.map(UnaryOp::Sqrt)
    }
    /// A node computing the sine.
    pub fn sin(self) -> Self {
        self.map(UnaryOp::Sin)
    }
    /// A node computing the cosine.
    pub fn cos(self) -> Self {
        self.map(UnaryOp::Cos)
    }
    /// A node computing the tangent.
    pub fn tan(self) -> Self {
        self.map(UnaryOp::Tan)
    }
    /// A node computing the arc sine.
    pub fn asin(self) -> Self {
        self.map(UnaryOp::Asin)
    }
    /// A node computing the arc cosine.
    pub fn acos(self) -> Self {
        self.map(UnaryOp::Acos)
    }
    /// A node computing the arc tangent.
    pub fn atan(self) -> Self {
        self.map(UnaryOp::Atan)
    }
    /// A node computing the hyperbolic sine.
    pub fn sinh(self) -> Self {
        self.map(UnaryOp::Sinh)
    }
    /// A node computing the hyperbolic cosine.
    pub fn cosh(self) -> Self {
        self.map(UnaryOp::Cosh)
    }
    /// A node computing the hyperbolic tangent.
    pub fn tanh(self) -> Self {
        self.map(UnaryOp::Tanh)
    }
    /// A node computing the inverse hyperbolic sine.
    pub fn asinh(self) -> Self {
        self.map(UnaryOp::Asinh)
    }
    /// A node computing the inverse hyperbolic cosine.
    pub fn acosh(self) -> Self {
        self.map(UnaryOp::Acosh)
    }
    /// A node computing the inverse hyperbolic tangent.
    pub fn atanh(self) -> Self {
        self.map(UnaryOp::Atanh)
    }
    /// A node computing the logistic sigmoid 1 / (1 + e⁻ˣ).
    pub fn sigmoid(self) -> Self {
        self.map(UnaryOp::Sigmoid)
    }
    /// A node computing the largest integer not greater than each element.
    pub fn floor(self) -> Self {
        self.map(UnaryOp::Floor)
    }
    /// A node computing the smallest integer not less than each element.
    pub fn ceil(self) -> Self {
        self.map(UnaryOp::Ceil)
    }
    /// Rounds half away from zero.
    pub fn round(self) -> Self {
        self.map(UnaryOp::Round)
    }
    /// A node computing ln Γ(x).
    pub fn ln_gamma(self) -> Self {
        self.map(UnaryOp::LnGamma)
    }
    /// A node computing the digamma function ψ(x).
    pub fn digamma(self) -> Self {
        self.map(UnaryOp::Digamma)
    }
    /// A node computing the trigamma function ψ₁(x).
    pub fn trigamma(self) -> Self {
        self.map(UnaryOp::Trigamma)
    }
    /// A node computing the error function erf(x).
    pub fn erf(self) -> Self {
        self.map(UnaryOp::Erf)
    }
    /// A node computing the complementary error function erfc(x) = 1 - erf(x).
    pub fn erfc(self) -> Self {
        self.map(UnaryOp::Erfc)
    }
    /// A node computing the inverse error function erf⁻¹(x), for -1 <= x <= 1.
    pub fn erf_inv(self) -> Self {
        self.map(UnaryOp::ErfInv)
    }
    /// A node computing ln Φ(x), the log of the standard normal CDF, which stays finite and accurate deep in the
    /// lower tail where Φ underflows.
    pub fn log_ndtr(self) -> Self {
        self.map(UnaryOp::LogNdtr)
    }
    /// A node computing the standard normal quantile function Φ⁻¹(p), for 0 <= p <= 1.
    pub fn ndtri(self) -> Self {
        self.map(UnaryOp::Ndtri)
    }
    /// A node computing softplus(x) = ln(1 + eˣ).
    pub fn softplus(self) -> Self {
        self.map(UnaryOp::Softplus)
    }
    /// A node computing logit(p) = ln(p / (1 - p)), for 0 <= p <= 1.
    pub fn logit(self) -> Self {
        self.map(UnaryOp::Logit)
    }
    /// A node computing ln(1 - eˣ), for x <= 0.
    pub fn log1mexp(self) -> Self {
        self.map(UnaryOp::Log1mExp)
    }
    /// A node computing the principal branch W₀(x) of the Lambert W function, the solution w >= −1 of w eʷ = x,
    /// for x >= −1/e (NaN below), as [`special::lambert_w0`](keanu_tensor::special::lambert_w0).
    pub fn lambert_w0(self) -> Self {
        self.map(UnaryOp::LambertW0)
    }
    /// A node computing the lower branch W₋₁(x) of the Lambert W function, the solution w <= −1 of w eʷ = x, for
    /// −1/e <= x < 0 (NaN elsewhere), as [`special::lambert_wm1`](keanu_tensor::special::lambert_wm1).
    pub fn lambert_wm1(self) -> Self {
        self.map(UnaryOp::LambertWm1)
    }
    /// A node computing K_ν(x), the modified Bessel function of the second kind, of each element x >= 0 for a
    /// constant real order ν (see [`BesselOp::K`](crate::op::BesselOp::K)). Differentiable to second order in x
    /// (not in ν). A non-finite order is a build error.
    pub fn bessel_k(self, order: f64) -> Self {
        self.bessel(BesselOp::K, order)
    }
    /// A node computing ln K_ν(x) without overflow or underflow, for a constant real order ν (see
    /// [`BesselOp::LogK`](crate::op::BesselOp::LogK)).
    pub fn log_bessel_k(self, order: f64) -> Self {
        self.bessel(BesselOp::LogK, order)
    }
    /// A node computing ln I_ν(x), the log of the modified Bessel function of the first kind, of each element
    /// x >= 0 for a constant order ν >= 0 (see [`BesselOp::LogI`](crate::op::BesselOp::LogI)). Differentiable to
    /// second order in x (not in ν). A negative or non-finite order is a build error.
    pub fn log_bessel_i(self, order: f64) -> Self {
        self.bessel(BesselOp::LogI, order)
    }
    /// A node computing the Matérn correlation function 2^{1−ν}/Γ(ν) x^ν K_ν(x) of each element x >= 0 for a
    /// constant ν > 0 (1 at x = 0; see [`BesselOp::Matern`](crate::op::BesselOp::Matern)).
    pub fn matern_correlation(self, nu: f64) -> Self {
        self.bessel(BesselOp::Matern, nu)
    }
    fn bessel(self, kind: BesselOp, order: f64) -> Self {
        self.same(Op::Bessel { kind, order })
    }

    /// Element-wise atan2(y, self).
    pub fn atan2(self, y: impl IntoDoubleVar<'b>) -> Self {
        self.binary(BinaryOp::Atan2, y)
    }
    /// A node computing ln(eᵃ + eᵇ), without overflow.
    pub fn log_add_exp(self, other: impl IntoDoubleVar<'b>) -> Self {
        self.binary(BinaryOp::LogAddExp, other)
    }
    /// A node computing log₂(2ᵃ + 2ᵇ), without overflow.
    pub fn log_add_exp2(self, other: impl IntoDoubleVar<'b>) -> Self {
        self.binary(BinaryOp::LogAddExp2, other)
    }
    /// y * ln(self), taken to be 0 where y is 0.
    pub fn safe_ln_times(self, y: impl IntoDoubleVar<'b>) -> Self {
        self.binary(BinaryOp::SafeLnTimes, y)
    }
    /// A node computing ln B(self, b), the log of the beta function.
    pub fn log_beta(self, b: impl IntoDoubleVar<'b>) -> Self {
        self.binary(BinaryOp::LogBeta, b)
    }
    /// A node computing Owen's T function T(self, a) (h is self), as
    /// [`special::owens_t`](keanu_tensor::special::owens_t): differentiable to second order in both.
    pub fn owens_t(self, a: impl IntoDoubleVar<'b>) -> Self {
        self.binary(BinaryOp::OwensT, a)
    }

    /// 1.0 where self > other, else 0.0.
    pub fn gt_mask(self, other: impl IntoDoubleVar<'b>) -> Self {
        self.mask(CompareOp::Gt, other)
    }
    /// A node computing 1.0 where self >= other, else 0.0.
    pub fn ge_mask(self, other: impl IntoDoubleVar<'b>) -> Self {
        self.mask(CompareOp::Ge, other)
    }
    /// A node computing 1.0 where self < other, else 0.0.
    pub fn lt_mask(self, other: impl IntoDoubleVar<'b>) -> Self {
        self.mask(CompareOp::Lt, other)
    }
    /// A node computing 1.0 where self <= other, else 0.0.
    pub fn le_mask(self, other: impl IntoDoubleVar<'b>) -> Self {
        self.mask(CompareOp::Le, other)
    }

    /// A node computing whether each element is NaN.
    pub fn is_nan(self) -> BoolVar<'b> {
        self.predicate(Predicate::IsNan)
    }
    /// A node computing whether each element is not NaN.
    pub fn not_nan(self) -> BoolVar<'b> {
        self.predicate(Predicate::NotNan)
    }
    /// A node computing whether each element is finite.
    pub fn is_finite(self) -> BoolVar<'b> {
        self.predicate(Predicate::IsFinite)
    }
    /// A node computing whether each element is ±infinity.
    pub fn is_infinite(self) -> BoolVar<'b> {
        self.predicate(Predicate::IsInfinite)
    }
    /// A node computing whether each element is +infinity.
    pub fn is_positive_infinity(self) -> BoolVar<'b> {
        self.predicate(Predicate::IsPositiveInfinity)
    }
    /// A node computing whether each element is -infinity.
    pub fn is_negative_infinity(self) -> BoolVar<'b> {
        self.predicate(Predicate::IsNegativeInfinity)
    }

    /// The mean of every element.
    pub fn mean(self) -> Self {
        self.reduce(ReduceOp::Mean, None)
    }
    /// The mean over the given axes, which are removed.
    pub fn mean_axes(self, axes: &[isize]) -> Self {
        self.reduce(ReduceOp::Mean, Some(axes))
    }
    /// ln Σ eˣ over every element, computed stably (-∞ if every element is -∞). Its gradient is the softmax.
    pub fn log_sum_exp(self) -> Self {
        self.reduce(ReduceOp::LogSumExp, None)
    }
    /// ln Σ eˣ over the given axes, which are removed, computed stably.
    pub fn log_sum_exp_axes(self, axes: &[isize]) -> Self {
        self.reduce(ReduceOp::LogSumExp, Some(axes))
    }
    /// eˣ normalised to sum to 1 along an axis, computed stably (uniform where every element is -∞).
    pub fn softmax(self, axis: isize) -> Self {
        self.same(Op::Softmax { axis })
    }
    /// The log of the softmax along an axis, x - logsumexp(x), accurate for very negative values.
    pub fn log_softmax(self, axis: isize) -> Self {
        self.same(Op::LogSoftmax { axis })
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
    /// The flat index of the smallest element, ignoring NaN.
    pub fn nan_argmin(self) -> IntVar<'b> {
        self.arg(ArgOp::NanArgMin, None)
    }

    /// The matrix product over the last two dimensions; leading dimensions broadcast.
    pub fn matmul(self, other: impl IntoDoubleVar<'b>) -> Self {
        self.matmul_transposed(other, false, false)
    }
    /// The matrix product, optionally transposing either operand.
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
    /// The tensor contraction of self and other over the given pairs of axes.
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
    /// The inverse of each matrix (the last two dimensions).
    pub fn inverse(self) -> Self {
        self.same(Op::Inverse)
    }
    /// The determinant of each matrix (the last two dimensions).
    pub fn determinant(self) -> Self {
        self.same(Op::Determinant)
    }
    /// The lower triangular Cholesky factor of each symmetric positive definite matrix.
    pub fn cholesky(self) -> Self {
        self.same(Op::Cholesky)
    }
    /// Reduced QR with orthonormal Q and upper triangular R with a nonnegative diagonal. The input
    /// has shape `[..., m, n]` with `m >= n`. The factors have shapes `[..., m, n]` and `[..., n, n]`.
    /// Differentiation requires full column rank.
    pub fn thin_qr(self) -> (Self, Self) {
        let factors = self.same(Op::ThinQr);
        let [.., rows, _] = *self.shape() else {
            return (factors, factors); // adding the operation already recorded the shape error
        };
        (
            factors.slice(&format!("...,:{rows},:")),
            factors.slice(&format!("...,{rows}:,:")),
        )
    }
    /// The full symmetric inverse of the matrix whose lower Cholesky factor this is.
    pub fn cholesky_inverse(self) -> Self {
        self.same(Op::CholeskyInverse)
    }
    /// X solving op(T) X = rhs, where T is this triangular matrix (`[..., n, n]`) and op(T) is T, or Tᵀ if
    /// `transpose`. Only the lower (if `lower`) or upper triangle is read. `rhs` is `[..., n, k]` or a vector `[n]`;
    /// leading dimensions broadcast. With L = A.cholesky(), `l.solve_triangular(b, true, false)` and then
    /// `.solve_triangular(.., true, true)` on the result solve A X = B. See [`Op::SolveTriangular`].
    pub fn solve_triangular(
        self,
        rhs: impl IntoDoubleVar<'b>,
        lower: bool,
        transpose: bool,
    ) -> Self {
        self.with(Op::SolveTriangular { lower, transpose }, rhs)
    }
    /// X solving A X = rhs, where A is this square matrix (`[..., n, n]`), by LU factorisation. `rhs` is
    /// `[..., n, k]` or a vector `[n]`; leading dimensions broadcast. A singular matrix is an evaluation error.
    pub fn solve(self, rhs: impl IntoDoubleVar<'b>) -> Self {
        self.with(Op::Solve, rhs)
    }
    /// Solves a Kronecker system with square factors A (self) and B (right).
    /// Supply the row-major right-hand side as a matrix `[..., m, n]` and receive X of the same
    /// trailing shape, solving `A X Bᵀ = rhs`. The factors and rhs may have broadcast batch axes.
    /// This uses two ordinary solves without constructing the `(m n) × (m n)` Kronecker matrix.
    pub fn solve_kronecker(
        self,
        right: impl IntoDoubleVar<'b>,
        rhs: impl IntoDoubleVar<'b>,
    ) -> Self {
        let right = right.into_var(self.builder);
        let rhs = rhs.into_var(self.builder);
        if rhs.shape().len() < 2 {
            self.builder.record(GraphError::InvalidValue(
                "a Kronecker solve needs its right-hand side shaped as a matrix".into(),
            ));
            return rhs;
        }
        right.solve(self.solve(rhs).transpose()).transpose()
    }

    /// ln |det| of each matrix (the last two dimensions), finite where the determinant itself would overflow or
    /// underflow; -∞ for a singular matrix.
    pub fn log_abs_determinant(self) -> Self {
        self.same(Op::LogAbsDeterminant)
    }
    /// The sign of each matrix's determinant: 1, -1, or 0 for a singular matrix. Not differentiable.
    pub fn determinant_sign(self) -> Self {
        self.same(Op::DeterminantSign)
    }
    /// numpy's `slogdet`: (sign, ln |det|) of each matrix, as two nodes, [`Self::determinant_sign`] and
    /// [`Self::log_abs_determinant`]. The determinant is their product sign · exp(ln |det|).
    pub fn slogdet(self) -> (Self, Self) {
        (self.determinant_sign(), self.log_abs_determinant())
    }
    /// The eigenvalues (`[..., n]`, ascending) of each symmetric matrix, reading only its lower triangle.
    pub fn eigvalsh(self) -> Self {
        self.same(Op::Eigvalsh)
    }
    /// The orthonormal eigenvectors (`[..., n, n]`, one per column, in the order of [`Self::eigvalsh`]) of each
    /// symmetric matrix, reading only its lower triangle. Each is signed so its first component within a relative
    /// 1e-10 of the largest magnitude is positive. The eigenvectors and their gradients are discontinuous where that
    /// sign convention switches and at repeated eigenvalues, which give non-finite gradients.
    pub fn eigh_vectors(self) -> Self {
        self.same(Op::EighVectors)
    }
    /// The matrix exponential exp(A) = Σₖ Aᵏ / k! of each square matrix (the last two dimensions), by scaling and
    /// squaring with Padé approximants. With a transition-rate matrix Q of a continuous-time Markov chain,
    /// `(q * t).matrix_exp()` holds the transition probabilities over time t. See [`Op::MatrixExp`].
    pub fn matrix_exp(self) -> Self {
        self.same(Op::MatrixExp)
    }
    /// The product A·self with a constant sparse matrix A (m × n), for a vector self `[n]` (giving `[m]`) or a
    /// matrix `[n, k]` (giving `[m, k]`): the cost is proportional to A's stored entries, not to m × n. Use it for
    /// neighbourhood sums, sparse design matrices and graph Laplacians. Linear, so differentiable to any order.
    /// See [`Op::SparseMatMul`].
    pub fn sparse_matmul(self, a: impl Into<std::sync::Arc<keanu_tensor::SparseMatrix>>) -> Self {
        self.same(Op::SparseMatMul(a.into()))
    }
    /// The symmetric eigendecomposition A = Q diag(λ) Qᵀ as two nodes, (λ, Q): [`Self::eigvalsh`] and
    /// [`Self::eigh_vectors`].
    pub fn eigh(self) -> (Self, Self) {
        (self.eigvalsh(), self.eigh_vectors())
    }
    fn with(self, op: Op, other: impl IntoDoubleVar<'b>) -> Self {
        let other = other.into_var(self.builder);
        Self::new(
            self.builder,
            self.builder.add(op, &[self.id, other.id()], DType::F64),
        )
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
    /// Replaces NaN elements with a value.
    pub fn replace_nan(self, with: f64) -> Self {
        self.same(Op::ReplaceNan(with))
    }
}

impl<'b> IntVar<'b> {
    /// Converts to f64.
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
    /// The handle, adding a constant node for plain values.
    fn into_var(self, builder: &'b GraphBuilder) -> BoolVar<'b>;
}

impl<'b> IntoBoolVar<'b> for BoolVar<'b> {
    fn into_var(self, builder: &'b GraphBuilder) -> BoolVar<'b> {
        builder.own(self)
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
    /// A node computing the logical and.
    pub fn and(self, other: impl IntoBoolVar<'b>) -> Self {
        self.logic(LogicOp::And, other)
    }
    /// A node computing the logical or.
    pub fn or(self, other: impl IntoBoolVar<'b>) -> Self {
        self.logic(LogicOp::Or, other)
    }
    /// A node computing the logical exclusive or.
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
    /// A node computing whether self == other (IEEE: NaN is never equal).
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
