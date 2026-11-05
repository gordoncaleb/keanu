use keanu_graph::GraphError;
use keanu_tensor::TensorError;
use std::fmt;

/// Errors from inference algorithms.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum AlgorithmError {
    /// An error evaluating the graph.
    Graph(GraphError),
    /// The model has no latent nodes to optimize or sample.
    NoLatents,
    /// Gradient optimizers only work on continuous (f64) latents; these latents are discrete.
    DiscreteLatents(Vec<String>),
    /// The starting point has zero probability.
    ImpossibleStart,
    /// The gradient at the starting point is flat (its largest absolute element is given).
    FlatGradient {
        /// The largest absolute element of the gradient.
        largest: f64,
    },
    /// The gradient at the starting point has a NaN or infinite element.
    NonFiniteGradient,
    /// An algorithm setting is out of range.
    InvalidSetting(String),
    /// An argument is invalid for this operation.
    InvalidArgument(String),
    /// Variational inference could not evaluate its objective, the evidence lower bound (ELBO), or its gradient:
    /// the message says where.
    NonFiniteElbo(String),
    /// The Laplace approximation found its mode on the boundary of a latent's support (the latent is named), where
    /// a Gaussian approximation in unconstrained space is meaningless.
    ModeOnBoundary(String),
    /// The Hessian of the log density at the mode the Laplace approximation found is not negative definite (or not
    /// finite), so there is no Gaussian approximation there: the message names the latents along the offending
    /// direction and suggests remedies.
    IndefiniteHessian(String),
}

impl fmt::Display for AlgorithmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Graph(e) => write!(f, "{e}"),
            Self::NoLatents => f.write_str("the model has no latent nodes"),
            Self::DiscreteLatents(nodes) => {
                write!(
                    f,
                    "gradient optimizers need continuous latents, but these are discrete: {}",
                    nodes.join(", ")
                )
            }
            Self::ImpossibleStart => f.write_str("cannot start from a zero probability state"),
            Self::FlatGradient { largest } => {
                write!(
                    f,
                    "the initial gradient is very flat; its largest element is {largest}"
                )
            }
            Self::NonFiniteGradient => {
                f.write_str("the initial gradient has a NaN or infinite element")
            }
            Self::InvalidSetting(message) | Self::InvalidArgument(message) => f.write_str(message),
            Self::NonFiniteElbo(message) => {
                write!(f, "variational inference failed: {message}")
            }
            Self::ModeOnBoundary(node) => write!(
                f,
                "the mode of the Jacobian-adjusted density is on the boundary of the support of {node}; a Laplace \
                 approximation there is meaningless (sample the posterior instead, or give {node} a prior that keeps \
                 it away from the boundary)"
            ),
            Self::IndefiniteHessian(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for AlgorithmError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            // transparent: Display already shows the wrapped error's message
            Self::Graph(e) => e.source(),
            _ => None,
        }
    }
}

impl From<GraphError> for AlgorithmError {
    fn from(e: GraphError) -> Self {
        Self::Graph(e)
    }
}

impl From<keanu_distributions::DistributionError> for AlgorithmError {
    fn from(e: keanu_distributions::DistributionError) -> Self {
        Self::Graph(GraphError::Distribution(e))
    }
}

impl From<TensorError> for AlgorithmError {
    fn from(e: TensorError) -> Self {
        Self::Graph(GraphError::Tensor(e))
    }
}

/// A `Result` whose error is [`AlgorithmError`].
pub type Result<T, E = AlgorithmError> = std::result::Result<T, E>;
