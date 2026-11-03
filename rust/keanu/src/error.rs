use std::fmt;

/// Any error from Keanu, so that `?` works across the layers.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// A tensor operation failed.
    Tensor(crate::tensor::TensorError),
    /// A distribution could not be constructed or sampled.
    Distribution(crate::distributions::DistributionError),
    /// Building or evaluating a graph failed.
    Graph(crate::graph::GraphError),
    /// An optimizer or sampler failed.
    Algorithm(crate::algorithms::AlgorithmError),
    /// Saving or loading a model failed.
    Io(crate::io::IoError),
}

/// A `Result` whose error is [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tensor(e) => write!(f, "{e}"),
            Self::Distribution(e) => write!(f, "{e}"),
            Self::Graph(e) => write!(f, "{e}"),
            Self::Algorithm(e) => write!(f, "{e}"),
            Self::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        // transparent: Display already shows the wrapped error's message
        match self {
            Self::Tensor(e) => e.source(),
            Self::Distribution(e) => e.source(),
            Self::Graph(e) => e.source(),
            Self::Algorithm(e) => e.source(),
            Self::Io(e) => e.source(),
        }
    }
}

macro_rules! from {
    ($($variant:ident($t:ty)),*) => {
        $(impl From<$t> for Error {
            fn from(e: $t) -> Self {
                Self::$variant(e)
            }
        })*
    };
}

from!(
    Tensor(crate::tensor::TensorError),
    Distribution(crate::distributions::DistributionError),
    Graph(crate::graph::GraphError),
    Algorithm(crate::algorithms::AlgorithmError),
    Io(crate::io::IoError)
);
