//! Node operations: what each node computes from its parents' values.

mod distribution;
mod eval;
mod shape;

pub use distribution::{DistributionKind, DistributionOp};
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
    Abs,
    Neg,
    Sign,
    Reciprocal,
    Exp,
    Exp2,
    ExpM1,
    Ln,
    Ln1p,
    Log2,
    Log10,
    Sqrt,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,
    Sigmoid,
    Floor,
    Ceil,
    Round,
    LnGamma,
    Digamma,
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
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Max,
    Min,
    Rem,
    Atan2,
    LogAddExp,
    LogAddExp2,
    SafeLnTimes,
}

/// Element-wise comparisons with broadcasting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompareOp {
    Gt,
    Ge,
    Lt,
    Le,
    Eq,
    Ne,
}

/// Element-wise boolean logic with broadcasting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LogicOp {
    And,
    Or,
    Xor,
}

/// Element-wise tests of f64 values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Predicate {
    IsNan,
    NotNan,
    IsFinite,
    IsInfinite,
    IsPositiveInfinity,
    IsNegativeInfinity,
}

/// Reductions. `Mean` and `StdDev` apply to f64 only. `Max`, `Min` and `StdDev` reduce over every dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Sum,
    Product,
    Mean,
    Max,
    Min,
    StdDev,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScanOp {
    CumSum,
    CumProd,
}

/// Index of the largest or smallest element. The NaN-ignoring variants apply to f64 over every dimension only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArgOp {
    ArgMax,
    ArgMin,
    NanArgMax,
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
    /// [input]
    Unary(UnaryOp),
    /// [left, right]
    Binary(BinaryOp),
    /// [left, right] -> bool
    Compare(CompareOp),
    /// [left, right] -> f64 (1.0 where the comparison holds), f64 only
    Mask(CompareOp),
    /// [left, right] -> bool, |left - right| <= epsilon
    ApproxEq(f64),
    /// [left, right], bool
    Logic(LogicOp),
    /// [input], bool
    Not,
    /// [input], bool -> scalar bool
    All,
    /// [input], bool -> scalar bool
    Any,
    /// [input], f64 -> bool
    Predicate(Predicate),
    /// [input]. `axes: None` reduces every dimension.
    Reduce {
        kind: ReduceOp,
        axes: Option<Vec<isize>>,
    },
    /// [input], f64: (x - mean) / std_dev over every element
    Standardize,
    /// [input]
    Scan { kind: ScanOp, axis: isize },
    /// [input] -> i64. `axis: None` gives the flat index.
    ArgReduce { kind: ArgOp, axis: Option<isize> },
    /// [input]
    Reshape(Vec<isize>),
    /// [input]
    Permute(Vec<usize>),
    /// [input]
    Transpose,
    /// [input]
    Broadcast(Vec<usize>),
    /// [input], numpy slice syntax
    Slice(String),
    /// [inputs...]
    Concat(isize),
    /// [input] -> the element at an index, as a scalar
    Take(Vec<usize>),
    /// [input]
    Diag,
    /// [input]
    DiagPart,
    /// [input], numpy's tril
    Tril(isize),
    /// [input], numpy's triu
    Triu(isize),
    /// [input]
    FillTriangular { upper: bool, lower: bool },
    /// [input]
    TrianglePart { upper: bool },
    /// [left, right], f64
    MatMul {
        transpose_left: bool,
        transpose_right: bool,
    },
    /// [left, right], f64
    TensorMultiply {
        left_axes: Vec<isize>,
        right_axes: Vec<isize>,
    },
    /// [input], f64
    Inverse,
    /// [input], f64
    Determinant,
    /// [input], f64
    Cholesky,
    /// [input], f64 (a lower Cholesky factor) -> the full symmetric inverse
    CholeskyInverse,
    /// [input] -> the given type. Numbers to bool are true where equal to 1; bool to numbers is a 0/1 mask;
    /// f64 to i64 truncates.
    Cast(DType),
    /// [predicate (bool), then, else], broadcasting
    Where,
    /// [selector (scalar i64), options...] -> the selected option's value
    Multiplexer,
    /// [conditions..., outcomes..., default]: the outcome whose key matches the conditions' scalar values (booleans
    /// as 0/1), or the default.
    Cpt {
        conditions: usize,
        keys: Vec<Vec<i64>>,
    },
    /// [operand (f64), mask (f64), value (scalar f64)]: operand with value where mask == 1.0
    SetWithMask,
    /// [input], f64
    ReplaceNan(f64),
    /// [parent], connected after creation
    Proxy,
    /// [parents...]
    Lambda(Lambda),
    /// [predicate (bool)] -> the predicate, failing evaluation if any element is false
    Assert(Option<String>),
    /// [parameters...] probabilistic
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
