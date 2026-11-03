use keanu_tensor::TensorError;
use std::fmt;

/// Errors from distributions.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum DistributionError {
    /// A tensor operation failed, e.g. because shapes are incompatible.
    Tensor(TensorError),
    /// A parameter is outside the range the distribution accepts (e.g. a non-positive scale when sampling).
    InvalidParameter(String),
    /// A value is not valid for the distribution (e.g. a Dirichlet value that does not sum to 1).
    InvalidValue(String),
}

impl fmt::Display for DistributionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tensor(e) => write!(f, "{e}"),
            Self::InvalidParameter(m) => write!(f, "invalid parameter: {m}"),
            Self::InvalidValue(m) => write!(f, "invalid value: {m}"),
        }
    }
}

impl std::error::Error for DistributionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Tensor(e) => Some(e),
            _ => None,
        }
    }
}

impl From<TensorError> for DistributionError {
    fn from(e: TensorError) -> Self {
        Self::Tensor(e)
    }
}

/// A `Result` whose error is [`DistributionError`].
pub type Result<T, E = DistributionError> = std::result::Result<T, E>;

pub(crate) fn invalid_parameter<T>(message: impl Into<String>) -> Result<T> {
    Err(DistributionError::InvalidParameter(message.into()))
}
