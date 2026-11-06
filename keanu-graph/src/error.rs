use crate::graph::NodeId;
use crate::ode::SolverFailed;
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
    InvalidNode {
        /// The operation.
        op: &'static str,
        /// What is wrong.
        message: String,
    },
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
    NoPartial {
        /// The index of the "of" node.
        of: NodeId,
        /// The index of the "with respect to" node.
        wrt: NodeId,
    },
    /// Auto diff needs a value that hasn't been computed; evaluate the graph first.
    MissingValue(NodeId),
    /// Auto diff was asked for on a node that isn't f64 or isn't differentiable.
    NotDifferentiable(String),
    /// A custom distribution failed.
    Custom(String),
    /// A handle from one [`GraphBuilder`](crate::GraphBuilder) was used with another. The node is the handle's.
    WrongBuilder(NodeId),
    /// A node id that isn't a node of the graph, builder or state it was used with (out of range, e.g. from another
    /// graph or made with [`NodeId::from_index`]).
    UnknownNode(NodeId),
    /// A constraint transform (see [`transform`](crate::transform)) could not be built for a support, or was given
    /// an input of the wrong size or a value outside its support.
    Transform(String),
    /// An exact second derivative (a Hessian-vector product) was needed through an op, distribution or bijector
    /// that has no second-order rule. The message names what lacks the rule and, for graph nodes, the node.
    NoSecondOrderRule(String),
    /// A discrete latent can't be marginalised (see [`marginalize`](crate::marginalize)): `node` blocks it (it may
    /// be the latent itself), for the `reason` given.
    CannotMarginalize {
        /// The latent, as [`Graph::describe`](crate::Graph::describe) names it.
        latent: String,
        /// The node that blocks it, named with its operation or distribution, such as `#12 (matmul)`.
        node: String,
        /// Why, completing a sentence whose subject is the node.
        reason: String,
    },
    /// An ODE solve (see [`Op::Ode`](crate::op::Op::Ode)) could not be completed at these values: the step limit
    /// was reached, the step size underflowed or the state became non-finite. The model has no usable value there,
    /// so models and inference algorithms treat it as an impossible point (see [`Self::is_impossible_point`]).
    SolverFailed(SolverFailed),
    /// A simulator node's function failed, or returned a value of the wrong type or shape (see
    /// [`Simulator`](crate::op::Simulator)).
    Simulator(String),
    /// A log probability was needed of a simulator node, which has none: the message names the node. Infer a
    /// simulator's parameters by approximate Bayesian computation instead (`keanu_algorithms::abc`).
    NoLikelihood(String),
}

impl GraphError {
    /// True for errors that mean a point has zero probability rather than that the model or its use is wrong:
    /// [`Self::SolverFailed`]. [`Model`](crate::Model)'s log probability methods,
    /// [`Parameterisation::log_prob_and_gradient`](crate::Parameterisation::log_prob_and_gradient) and the inference
    /// algorithms give −∞ for such points (a rejected proposal, a divergent NUTS trajectory, a rejected optimizer
    /// step) instead of failing.
    pub fn is_impossible_point(&self) -> bool {
        matches!(self, Self::SolverFailed(_))
    }
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
                "no partial derivative of node #{} with respect to node #{}: it does not depend on it through differentiable nodes",
                of.index(),
                wrt.index()
            ),
            Self::MissingValue(id) => write!(
                f,
                "node #{} has no value; evaluate the graph first",
                id.index()
            ),
            Self::NotDifferentiable(m) => write!(f, "{m}"),
            Self::Custom(m) => write!(f, "custom distribution: {m}"),
            Self::WrongBuilder(id) => write!(
                f,
                "node #{} belongs to a different GraphBuilder",
                id.index()
            ),
            Self::UnknownNode(id) => write!(f, "node #{} is not a node of this graph", id.index()),
            Self::Transform(m) => write!(f, "transform: {m}"),
            Self::NoSecondOrderRule(m) => write!(f, "no second-order rule for {m}"),
            Self::CannotMarginalize {
                latent,
                node,
                reason,
            } => write!(f, "cannot marginalise {latent}: {node} {reason}"),
            Self::SolverFailed(failed) => write!(f, "{failed}"),
            Self::Simulator(m) => write!(f, "simulator failed: {m}"),
            Self::NoLikelihood(node) => write!(
                f,
                "{node} is a simulator node, which has no likelihood; infer its parameters by approximate \
                 Bayesian computation (Smc::sample_abc with an Abc likelihood)"
            ),
        }
    }
}

/// Wrapped errors are transparent: `Display` shows the wrapped error's message and `source` is the wrapped
/// error's source, so error reports don't repeat a message.
impl std::error::Error for GraphError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Tensor(e) => e.source(),
            Self::Distribution(e) => e.source(),
            _ => None,
        }
    }
}

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

/// A `Result` whose error is [`GraphError`].
pub type Result<T, E = GraphError> = std::result::Result<T, E>;
