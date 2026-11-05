use keanu_graph::GraphError;
use std::fmt;

/// Errors from loading or saving models.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum IoError {
    /// The bytes or text are not a valid saved model.
    Decode(String),
    /// The model could not be written in the requested format.
    Encode(String),
    /// The file was written with a newer version of the format than this version of Keanu reads.
    UnsupportedVersion {
        /// The file's format version.
        found: u32,
        /// The newest format version this version of Keanu reads.
        supported: u32,
    },
    /// A saved node is invalid (e.g. it refers to a later node, or has an unknown operation).
    InvalidNode {
        /// The node's index in the file.
        index: usize,
        /// What is wrong.
        message: String,
    },
    /// A node that can't be saved (a lambda or a custom distribution), or a value that doesn't fit the format.
    NotSaveable(String),
    /// The loaded graph is invalid.
    Graph(GraphError),
}

impl fmt::Display for IoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decode(m) => write!(f, "cannot decode the model: {m}"),
            Self::Encode(m) => write!(f, "cannot encode the model: {m}"),
            Self::UnsupportedVersion { found, supported } => write!(
                f,
                "the model has format version {found}, but this version of Keanu reads versions up to {supported}"
            ),
            Self::InvalidNode { index, message } => write!(f, "node {index}: {message}"),
            Self::NotSaveable(m) => write!(f, "cannot save: {m}"),
            Self::Graph(e) => write!(f, "{e}"),
        }
    }
}

/// The wrapped graph error is transparent: `Display` shows its message and `source` is its source.
impl std::error::Error for IoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Graph(e) => e.source(),
            _ => None,
        }
    }
}

impl From<GraphError> for IoError {
    fn from(e: GraphError) -> Self {
        Self::Graph(e)
    }
}

/// A `Result` whose error is [`IoError`].
pub type Result<T, E = IoError> = std::result::Result<T, E>;
