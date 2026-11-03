use keanu_distributions::DistributionError;
use keanu_tensor::TensorError;
use std::fmt;

/// Errors from building or evaluating a graph.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum GraphError {
    /// A tensor operation failed (incompatible shapes, invalid axes, singular matrices, ...).
    Tensor(TensorError),
    /// A distribution could not be constructed or sampled.
    Distribution(DistributionError),
    /// A node was created with an invalid argument. `op` names the operation.
    InvalidNode { op: &'static str, message: String },
    /// A value of the wrong type or shape was given for a node.
    InvalidValue(String),
    /// Two nodes have the same label.
    DuplicateLabel(String),
    /// No node has this label.
    UnknownLabel(String),
    /// A node cannot be observed (only probabilistic and boolean nodes can).
    NotObservable(String),
    /// A proxy node was evaluated before being connected, or connected to an invalid parent.
    Proxy(String),
    /// A sequence (plate) could not be built, e.g. a proxy with no matching node in the previous item.
    Sequence(String),
    /// An assertion node's predicate was false.
    AssertionFailed(String),
    /// A lambda node's function failed.
    Lambda(String),
    /// There is no partial derivative: the "of" node does not depend on the "with respect to" node through
    /// differentiable nodes (e.g. it depends on it only via floor, a comparison or a probabilistic node).
    NoPartial { of: usize, wrt: usize },
    /// Auto diff needs a value that hasn't been computed; evaluate the graph first.
    MissingValue(usize),
    /// Auto diff was asked for on a node that isn't f64 or isn't differentiable.
    NotDifferentiable(String),
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tensor(e) => write!(f, "{e}"),
            Self::Distribution(e) => write!(f, "{e}"),
            Self::InvalidNode { op, message } => write!(f, "invalid {op} node: {message}"),
            Self::InvalidValue(m) => write!(f, "invalid value: {m}"),
            Self::DuplicateLabel(l) => write!(f, "more than one node is labelled {l:?}"),
            Self::UnknownLabel(l) => write!(f, "no node is labelled {l:?}"),
            Self::NotObservable(m) => write!(f, "{m}"),
            Self::Proxy(m) => write!(f, "{m}"),
            Self::Sequence(m) => write!(f, "sequence: {m}"),
            Self::AssertionFailed(m) => write!(f, "assertion failed: {m}"),
            Self::Lambda(m) => write!(f, "lambda failed: {m}"),
            Self::NoPartial { of, wrt } => write!(
                f,
                "no partial derivative of node #{of} with respect to node #{wrt}: it does not depend on it through differentiable nodes"
            ),
            Self::MissingValue(id) => write!(
                f,
                "node #{id} has no value; evaluate the graph before differentiating"
            ),
            Self::NotDifferentiable(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for GraphError {}

impl From<TensorError> for GraphError {
    fn from(e: TensorError) -> Self {
        Self::Tensor(e)
    }
}

impl From<DistributionError> for GraphError {
    fn from(e: DistributionError) -> Self {
        Self::Distribution(e)
    }
}

pub type Result<T, E = GraphError> = std::result::Result<T, E>;
