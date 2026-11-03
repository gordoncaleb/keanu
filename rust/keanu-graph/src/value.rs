use crate::error::{GraphError, Result};
use keanu_tensor::{BoolTensor, DoubleTensor, IntTensor};

/// The element type of a node's value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DType {
    /// `f64`
    F64,
    /// `i64`
    I64,
    /// `bool`
    Bool,
}

/// The value of a node: a tensor of one of the supported element types.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Value {
    /// An f64 tensor.
    F64(DoubleTensor),
    /// An i64 tensor.
    I64(IntTensor),
    /// A bool tensor.
    Bool(BoolTensor),
}

impl Value {
    /// The element type.
    pub fn dtype(&self) -> DType {
        match self {
            Self::F64(_) => DType::F64,
            Self::I64(_) => DType::I64,
            Self::Bool(_) => DType::Bool,
        }
    }

    /// The shape.
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::F64(t) => t.shape(),
            Self::I64(t) => t.shape(),
            Self::Bool(t) => t.shape(),
        }
    }

    /// The value as an f64 tensor, if it is one.
    pub fn as_f64(&self) -> Option<&DoubleTensor> {
        match self {
            Self::F64(t) => Some(t),
            _ => None,
        }
    }

    /// The value as an i64 tensor, if it is one.
    pub fn as_i64(&self) -> Option<&IntTensor> {
        match self {
            Self::I64(t) => Some(t),
            _ => None,
        }
    }

    /// The value as a bool tensor, if it is one.
    pub fn as_bool(&self) -> Option<&BoolTensor> {
        match self {
            Self::Bool(t) => Some(t),
            _ => None,
        }
    }

    /// The value converted to an f64 tensor of the same shape (integers as `f64`, booleans as 0.0 / 1.0).
    pub fn to_double_tensor(&self) -> DoubleTensor {
        match self {
            Self::F64(t) => t.clone(),
            Self::I64(t) => t.to_double(),
            Self::Bool(t) => t.to_double_mask(),
        }
    }

    /// A value of the given type and shape with every element equal to one (true for booleans).
    pub(crate) fn ones(dtype: DType, shape: &[usize]) -> Self {
        match dtype {
            DType::F64 => Self::F64(DoubleTensor::ones(shape)),
            DType::I64 => Self::I64(IntTensor::ones(shape)),
            DType::Bool => Self::Bool(BoolTensor::full(shape, true)),
        }
    }

    pub(crate) fn f64(&self) -> Result<&DoubleTensor> {
        self.as_f64().ok_or_else(|| {
            GraphError::InvalidValue(format!("expected f64, got {:?}", self.dtype()))
        })
    }

    pub(crate) fn i64(&self) -> Result<&IntTensor> {
        self.as_i64().ok_or_else(|| {
            GraphError::InvalidValue(format!("expected i64, got {:?}", self.dtype()))
        })
    }

    pub(crate) fn bool(&self) -> Result<&BoolTensor> {
        self.as_bool().ok_or_else(|| {
            GraphError::InvalidValue(format!("expected bool, got {:?}", self.dtype()))
        })
    }
}

impl From<DoubleTensor> for Value {
    fn from(t: DoubleTensor) -> Self {
        Self::F64(t)
    }
}

impl From<IntTensor> for Value {
    fn from(t: IntTensor) -> Self {
        Self::I64(t)
    }
}

impl From<BoolTensor> for Value {
    fn from(t: BoolTensor) -> Self {
        Self::Bool(t)
    }
}
