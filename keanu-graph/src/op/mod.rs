//! Node operations: what each node computes from its parents' values.

mod distribution;
pub(crate) mod eval;
mod shape;

pub(crate) use distribution::LogProbGradientParts;
pub use distribution::{CustomDistribution, CustomGradient, DistributionKind, DistributionOp};
pub(crate) use eval::eval;
pub(crate) use shape::output_type;

use crate::error::Result;
use crate::value::{DType, Value};
use std::fmt;
use std::sync::Arc;

/// Element-wise unary operations. All apply to f64; `Abs`, `Neg` and `Sign` also apply to i64.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum UnaryOp {
    /// |x|.
    Abs,
    /// -x.
    Neg,
    /// Sign (not differentiable).
    Sign,
    /// 1 / x.
    Reciprocal,
    /// Eˣ.
    Exp,
    /// 2ˣ.
    Exp2,
    /// Eˣ - 1.
    ExpM1,
    /// Natural logarithm.
    Ln,
    /// Ln(1 + x).
    Ln1p,
    /// Base 2 logarithm.
    Log2,
    /// Base 10 logarithm.
    Log10,
    /// Square root.
    Sqrt,
    /// Sine.
    Sin,
    /// Cosine.
    Cos,
    /// Tangent.
    Tan,
    /// Arc sine.
    Asin,
    /// Arc cosine.
    Acos,
    /// Arc tangent.
    Atan,
    /// Hyperbolic sine.
    Sinh,
    /// Hyperbolic cosine.
    Cosh,
    /// Hyperbolic tangent.
    Tanh,
    /// Inverse hyperbolic sine.
    Asinh,
    /// Inverse hyperbolic cosine.
    Acosh,
    /// Inverse hyperbolic tangent.
    Atanh,
    /// 1 / (1 + e⁻ˣ).
    Sigmoid,
    /// Floor (not differentiable).
    Floor,
    /// Ceiling (not differentiable).
    Ceil,
    /// Round half away from zero (not differentiable).
    Round,
    /// Ln Γ(x).
    LnGamma,
    /// Digamma ψ(x).
    Digamma,
    /// Trigamma ψ₁(x).
    Trigamma,
}

impl UnaryOp {
    pub(crate) fn applies_to_integers(self) -> bool {
        matches!(self, Self::Abs | Self::Neg | Self::Sign)
    }
}

/// Element-wise binary operations with broadcasting. `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Max` and `Min` apply to f64
/// and i64; `Rem` to i64 only; the rest to f64 only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BinaryOp {
    /// A + b.
    Add,
    /// A - b.
    Sub,
    /// A × b.
    Mul,
    /// A / b (integer division truncates; division by zero is an error).
    Div,
    /// Aᵇ.
    Pow,
    /// The larger of a and b (NaN if either is NaN).
    Max,
    /// The smaller of a and b (NaN if either is NaN).
    Min,
    /// Remainder with the sign of the dividend (integers).
    Rem,
    /// Atan2(b, a).
    Atan2,
    /// Ln(eᵃ + eᵇ).
    LogAddExp,
    /// Log₂(2ᵃ + 2ᵇ).
    LogAddExp2,
    /// B × ln(a), taken as 0 where b is 0.
    SafeLnTimes,
}

/// Element-wise comparisons with broadcasting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CompareOp {
    /// A > b.
    Gt,
    /// A >= b.
    Ge,
    /// A < b.
    Lt,
    /// A <= b.
    Le,
    /// A == b.
    Eq,
    /// A != b.
    Ne,
}

/// Element-wise boolean logic with broadcasting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LogicOp {
    /// Logical and.
    And,
    /// Logical or.
    Or,
    /// Logical exclusive or.
    Xor,
}

/// Element-wise tests of f64 values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Predicate {
    /// Whether NaN.
    IsNan,
    /// Whether not NaN.
    NotNan,
    /// Whether finite.
    IsFinite,
    /// Whether ±infinity.
    IsInfinite,
    /// Whether +infinity.
    IsPositiveInfinity,
    /// Whether -infinity.
    IsNegativeInfinity,
}

/// Reductions. `Mean` and `StdDev` apply to f64 only. `Max`, `Min` and `StdDev` reduce over every dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ReduceOp {
    /// Sum.
    Sum,
    /// Product.
    Product,
    /// Mean.
    Mean,
    /// The larger of a and b (NaN if either is NaN).
    Max,
    /// The smaller of a and b (NaN if either is NaN).
    Min,
    /// Sample standard deviation.
    StdDev,
}

/// Running accumulations along an axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ScanOp {
    /// Cumulative sum.
    CumSum,
    /// Cumulative product.
    CumProd,
}

/// Index of the largest or smallest element. The NaN-ignoring variants apply to f64 over every dimension only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ArgOp {
    /// Index of the largest element (the first NaN wins).
    ArgMax,
    /// Index of the smallest element (the first NaN wins).
    ArgMin,
    /// Index of the largest element, ignoring NaN.
    NanArgMax,
    /// Index of the smallest element, ignoring NaN.
    NanArgMin,
}

/// The function of a lambda node.
pub type LambdaFn = dyn Fn(&[Value]) -> std::result::Result<Value, String> + Send + Sync;

/// The Jacobian of a lambda: for each parent, ∂output/∂parent shaped `[output shape..., parent shape...]`
/// (`None` for parents the output doesn't depend on differentiably, e.g. non-f64 parents).
pub type LambdaJacobianFn = dyn Fn(&[Value]) -> std::result::Result<Vec<Option<keanu_tensor::DoubleTensor>>, String>
    + Send
    + Sync;

/// A node computed by a user supplied function of its parents' values.
#[derive(Clone)]
pub struct Lambda {
    pub(crate) name: String,
    pub(crate) function: Arc<LambdaFn>,
    pub(crate) jacobian: Option<Arc<LambdaJacobianFn>>,
    pub(crate) dtype: DType,
    pub(crate) shape: Vec<usize>,
}

impl fmt::Debug for Lambda {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Lambda")
            .field("name", &self.name)
            .field("differentiable", &self.jacobian.is_some())
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .finish()
    }
}

/// The operation a node performs. Parents are listed in the order each variant documents.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Op {
    /// No parents.
    Constant(Value),
    /// Parents: `input`
    Unary(UnaryOp),
    /// Parents: `left`, `right`
    Binary(BinaryOp),
    /// Parents: `left`, `right` -> bool
    Compare(CompareOp),
    /// Parents: `left`, `right` -> f64 (1.0 where the comparison holds), f64 only
    Mask(CompareOp),
    /// Parents: `left`, `right` -> bool, |left - right| <= epsilon
    ApproxEq(f64),
    /// Parents: `left`, `right`, bool
    Logic(LogicOp),
    /// Parents: `input`, bool
    Not,
    /// Parents: `input`, bool -> scalar bool
    All,
    /// Parents: `input`, bool -> scalar bool
    Any,
    /// Parents: `input`, f64 -> bool
    Predicate(Predicate),
    /// Parents: `input`. `axes: None` reduces every dimension.
    Reduce {
        /// The reduction.
        kind: ReduceOp,
        /// The axes to reduce over, or `None` for every axis.
        axes: Option<Vec<isize>>,
    },
    /// Parents: `input`, f64: (x - mean) / std_dev over every element
    Standardize,
    /// Parents: `input`
    /// Scan.
    Scan {
        /// The accumulation.
        kind: ScanOp,
        /// The axis to accumulate along.
        axis: isize,
    },
    /// Parents: `input` -> i64. `axis: None` gives the flat index.
    /// Arg reduce.
    ArgReduce {
        /// Which arg reduction.
        kind: ArgOp,
        /// The axis, or `None` for the flat index over every element.
        axis: Option<isize>,
    },
    /// Parents: `input`
    Reshape(Vec<isize>),
    /// Parents: `input`
    Permute(Vec<usize>),
    /// Parents: `input`
    Transpose,
    /// Parents: `input`
    Broadcast(Vec<usize>),
    /// Parents: `input`, numpy slice syntax
    Slice(String),
    /// Parents: `inputs...`
    Concat(isize),
    /// Parents: `input` -> the element at an index, as a scalar
    Take(Vec<usize>),
    /// Parents: `input`
    Diag,
    /// Parents: `input`
    DiagPart,
    /// Parents: `input`, numpy's tril
    Tril(isize),
    /// Parents: `input`, numpy's triu
    Triu(isize),
    /// Parents: `input`
    /// Fill triangular.
    FillTriangular {
        /// Whether to fill the upper triangle.
        upper: bool,
        /// Whether to fill the lower triangle.
        lower: bool,
    },
    /// Parents: `input`
    /// Triangle part.
    TrianglePart {
        /// Whether to take the upper (rather than lower) triangle.
        upper: bool,
    },
    /// Parents: `left`, `right`, f64
    MatMul {
        /// Whether to transpose the left operand.
        transpose_left: bool,
        /// Whether to transpose the right operand.
        transpose_right: bool,
    },
    /// Parents: `left`, `right`, f64
    TensorMultiply {
        /// The left operand's axes to contract.
        left_axes: Vec<isize>,
        /// The right operand's axes they pair with.
        right_axes: Vec<isize>,
    },
    /// Parents: `input`, f64
    Inverse,
    /// Parents: `input`, f64
    Determinant,
    /// Parents: `input`, f64
    Cholesky,
    /// Parents: `input`, f64 (a lower Cholesky factor) -> the full symmetric inverse
    CholeskyInverse,
    /// Parents: `input` -> the given type. Numbers to bool are true where equal to 1; bool to numbers is a 0/1 mask;
    /// f64 to i64 truncates.
    Cast(DType),
    /// [predicate (bool), then, else], broadcasting
    Where,
    /// [selector (scalar i64), options...] -> the selected option's value
    Multiplexer,
    /// Parents: `conditions...`, `outcomes...`, `default`: the outcome whose key matches the conditions' scalar values (booleans
    /// as 0/1), or the default.
    Cpt {
        /// The number of condition parents.
        conditions: usize,
        /// The key (condition values) of each outcome, in order.
        keys: Vec<Vec<i64>>,
    },
    /// [operand (f64), mask (f64), value (scalar f64)]: operand with value where mask == 1.0
    SetWithMask,
    /// Parents: `input`, f64
    ReplaceNan(f64),
    /// Parents: `parent`, connected after creation
    Proxy,
    /// Parents: `parents...`
    Lambda(Lambda),
    /// [predicate (bool)] -> the predicate, failing evaluation if any element is false
    Assert(Option<String>),
    /// Parents: `parameters...` probabilistic
    Distribution(DistributionOp),
}

impl Op {
    /// A short name for error messages.
    pub fn name(&self) -> &'static str {
        match self {
            Op::Constant(_) => "constant",
            Op::Unary(_) => "unary",
            Op::Binary(_) => "binary",
            Op::Compare(_) => "compare",
            Op::Mask(_) => "mask",
            Op::ApproxEq(_) => "approx_eq",
            Op::Logic(_) => "logic",
            Op::Not => "not",
            Op::All => "all",
            Op::Any => "any",
            Op::Predicate(_) => "predicate",
            Op::Reduce { .. } => "reduce",
            Op::Standardize => "standardize",
            Op::Scan { .. } => "scan",
            Op::ArgReduce { .. } => "arg_reduce",
            Op::Reshape(_) => "reshape",
            Op::Permute(_) => "permute",
            Op::Transpose => "transpose",
            Op::Broadcast(_) => "broadcast",
            Op::Slice(_) => "slice",
            Op::Concat(_) => "concat",
            Op::Take(_) => "take",
            Op::Diag => "diag",
            Op::DiagPart => "diag_part",
            Op::Tril(_) => "tril",
            Op::Triu(_) => "triu",
            Op::FillTriangular { .. } => "fill_triangular",
            Op::TrianglePart { .. } => "triangle_part",
            Op::MatMul { .. } => "matmul",
            Op::TensorMultiply { .. } => "tensor_multiply",
            Op::Inverse => "inverse",
            Op::Determinant => "determinant",
            Op::Cholesky => "cholesky",
            Op::CholeskyInverse => "cholesky_inverse",
            Op::Cast(_) => "cast",
            Op::Where => "where",
            Op::Multiplexer => "multiplexer",
            Op::Cpt { .. } => "cpt",
            Op::SetWithMask => "set_with_mask",
            Op::ReplaceNan(_) => "replace_nan",
            Op::Proxy => "proxy",
            Op::Lambda(_) => "lambda",
            Op::Assert(_) => "assert",
            Op::Distribution(_) => "distribution",
        }
    }

    /// True for nodes whose value is a random draw (sampled from their distribution, or observed).
    pub fn is_probabilistic(&self) -> bool {
        matches!(self, Op::Distribution(_))
    }
}

/// Internal: check a parent count when evaluating.
pub(crate) fn expect_parents(op: &Op, inputs: &[&Value], count: usize) -> Result<()> {
    if inputs.len() == count {
        Ok(())
    } else {
        Err(crate::error::GraphError::InvalidNode {
            op: op.name(),
            message: format!("expected {count} parents, got {}", inputs.len()),
        })
    }
}
