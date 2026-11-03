use std::fmt;

/// Errors from tensor operations. Every fallible operation returns one of these rather than panicking.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TensorError {
    /// The number of values does not match the shape.
    LengthMismatch { shape: Vec<usize>, len: usize },
    /// Two shapes cannot be broadcast together, or a shape is otherwise incompatible with an operation.
    IncompatibleShapes { left: Vec<usize>, right: Vec<usize> },
    /// An axis or dimension is out of range for the tensor's rank.
    InvalidAxis { axis: isize, rank: usize },
    /// A reshape target is not valid for the tensor's length.
    InvalidReshape { from: Vec<usize>, to: Vec<isize> },
    /// A permutation is not a rearrangement of the tensor's dimensions.
    InvalidPermutation {
        permutation: Vec<usize>,
        rank: usize,
    },
    /// A slice specification could not be parsed or is out of range.
    InvalidSlice(String),
    /// The operation needs a tensor of a different rank or form (e.g. a square matrix).
    InvalidArgument(String),
    /// A matrix is singular.
    SingularMatrix,
    /// A matrix is not positive definite.
    NotPositiveDefinite,
    /// Integer division or remainder by zero, or zero raised to a negative power.
    DivisionByZero,
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { shape, len } => write!(
                f,
                "shape {shape:?} needs {} values but {len} were given",
                shape.iter().product::<usize>()
            ),
            Self::IncompatibleShapes { left, right } => {
                write!(f, "incompatible shapes {left:?} and {right:?}")
            }
            Self::InvalidAxis { axis, rank } => {
                write!(f, "axis {axis} is out of range for a rank {rank} tensor")
            }
            Self::InvalidReshape { from, to } => write!(f, "cannot reshape {from:?} to {to:?}"),
            Self::InvalidPermutation { permutation, rank } => write!(
                f,
                "{permutation:?} is not a permutation of {rank} dimensions"
            ),
            Self::InvalidSlice(message) => write!(f, "invalid slice: {message}"),
            Self::InvalidArgument(message) => f.write_str(message),
            Self::SingularMatrix => f.write_str("matrix is singular"),
            Self::NotPositiveDefinite => f.write_str("matrix is not positive definite"),
            Self::DivisionByZero => f.write_str("integer division by zero"),
        }
    }
}

impl std::error::Error for TensorError {}

pub type Result<T, E = TensorError> = std::result::Result<T, E>;
