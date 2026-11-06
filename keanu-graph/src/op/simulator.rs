//! Simulator nodes: probabilistic nodes whose value is drawn by a user function of their parameters and a random
//! number generator, and which have no likelihood.

use crate::error::{GraphError, Result};
use crate::value::{DType, Value};
use keanu_distributions::KeanuRng;
use std::fmt;
use std::sync::Arc;

/// The function of a simulator node: draws a value from its parameters' values with the given generator.
///
/// It must return a value of the node's type and shape, and be thread safe: parallel chains call it at once.
pub type SimulatorFn =
    dyn Fn(&[&Value], &mut KeanuRng) -> std::result::Result<Value, String> + Send + Sync;

/// A simulator: the distribution of a probabilistic node that can be sampled (by a user function) but has no
/// likelihood, like PyMC's `pm.Simulator`.
///
/// A simulator node is made with [`GraphBuilder::simulator`](crate::GraphBuilder::simulator). Sampling it from its
/// prior (forward sampling, prior and posterior predictive draws) calls the function; its log probability is
/// [`GraphError::NoLikelihood`], so inference needs a likelihood-free method: approximate Bayesian computation,
/// which compares data simulated at each candidate parameter value with the observed data (`keanu_algorithms::abc`).
/// Simulator nodes can't be saved.
#[derive(Clone)]
pub struct Simulator {
    function: Arc<SimulatorFn>,
    dtype: DType,
}

impl Simulator {
    /// A simulator of values of type `dtype` drawn by `function`.
    pub fn new(
        dtype: DType,
        function: impl Fn(&[&Value], &mut KeanuRng) -> std::result::Result<Value, String>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self {
            function: Arc::new(function),
            dtype,
        }
    }

    /// The type of the values it simulates.
    pub fn value_type(&self) -> DType {
        self.dtype
    }

    /// Draws a value of the given shape from the parameters' values: [`GraphError::Simulator`] if the function
    /// fails or returns a value of another type or shape.
    pub fn simulate(
        &self,
        shape: &[usize],
        parameters: &[&Value],
        rng: &mut KeanuRng,
    ) -> Result<Value> {
        let value = (self.function)(parameters, rng).map_err(GraphError::Simulator)?;
        if value.dtype() != self.dtype || value.shape() != shape {
            return Err(GraphError::Simulator(format!(
                "the function returned a {:?} value of shape {:?}, expected {:?} of shape {shape:?}",
                value.dtype(),
                value.shape(),
                self.dtype
            )));
        }
        Ok(value)
    }
}

impl fmt::Debug for Simulator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Simulator")
            .field("dtype", &self.dtype)
            .finish_non_exhaustive()
    }
}
