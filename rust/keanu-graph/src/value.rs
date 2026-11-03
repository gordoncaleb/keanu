use crate::error::{GraphError, Result};
use keanu_tensor::{BoolTensor, DoubleTensor, IntTensor};

/// The element type of a node's value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F64,
    I64,
    Bool,
}

/// The value of a node: a tensor of one of the supported element types.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    F64(DoubleTensor),
    I64(IntTensor),
    Bool(BoolTensor),
}

impl Value {
    pub fn dtype(&self) -> DType {
        match self {
            Self::F64(_) => DType::F64,
            Self::I64(_) => DType::I64,
            Self::Bool(_) => DType::Bool,
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Self::F64(t) => t.shape(),
            Self::I64(t) => t.shape(),
            Self::Bool(t) => t.shape(),
        }
    }

    pub fn as_f64(&self) -> Option<&DoubleTensor> {
        match self {
            Self::F64(t) => Some(t),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<&IntTensor> {
        match self {
            Self::I64(t) => Some(t),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<&BoolTensor> {
        match self {
            Self::Bool(t) => Some(t),
            _ => None,
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
