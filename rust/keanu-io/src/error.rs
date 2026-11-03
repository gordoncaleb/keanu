use keanu_graph::GraphError;
use std::fmt;

/// Errors from loading or saving models.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum IoError {
    /// The bytes or text are not a valid saved model.
    Decode(String),
    /// A vertex type that can't be loaded.
    UnsupportedVertex(String),
    /// A vertex is missing a parameter, or has one of the wrong kind.
    InvalidParameter {
        vertex: String,
        parameter: String,
        message: String,
    },
    /// A parameter or value refers to a vertex that hasn't been defined (earlier in the file).
    UnknownVertex(String),
    /// A node that can't be saved (e.g. a lambda), or a value that doesn't fit the format.
    NotSaveable(String),
    /// The loaded graph is invalid.
    Graph(GraphError),
}

impl fmt::Display for IoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decode(m) => write!(f, "cannot decode the model: {m}"),
            Self::UnsupportedVertex(t) => write!(f, "vertex type {t} is not supported"),
            Self::InvalidParameter {
                vertex,
                parameter,
                message,
            } => write!(f, "{vertex}: parameter {parameter}: {message}"),
            Self::UnknownVertex(id) => write!(f, "unknown vertex {id}"),
            Self::NotSaveable(m) => write!(f, "cannot save: {m}"),
            Self::Graph(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for IoError {}

impl From<GraphError> for IoError {
    fn from(e: GraphError) -> Self {
        Self::Graph(e)
    }
}

pub type Result<T, E = IoError> = std::result::Result<T, E>;
