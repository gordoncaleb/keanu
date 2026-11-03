//! Gradient based optimizers for finding the maximum a posteriori (MAP) or maximum likelihood (MLE) values of a
//! model's continuous latents.

mod adam;
mod conjugate_gradient;
mod convergence;

pub use adam::Adam;
pub use conjugate_gradient::{ConjugateGradient, UpdateFormula};
pub use convergence::{ConvergenceChecker, Norm};

use crate::error::{AlgorithmError, Result};
use keanu_graph::{Model, NodeId, State, Value};
use keanu_tensor::DoubleTensor;
use std::cell::Cell;

/// What to maximize.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fitness {
    /// The joint log probability (maximum a posteriori).
    LogProb,
    /// The log likelihood: the log probability of the observations (maximum likelihood).
    LogLikelihood,
}

/// The optimization algorithm.
#[derive(Debug, Clone, PartialEq)]
pub enum Algorithm {
    /// Adam.
    Adam(Adam),
    /// Nonlinear conjugate gradient.
    ConjugateGradient(ConjugateGradient),
}

impl From<Adam> for Algorithm {
    fn from(a: Adam) -> Self {
        Self::Adam(a)
    }
}

impl From<ConjugateGradient> for Algorithm {
    fn from(c: ConjugateGradient) -> Self {
        Self::ConjugateGradient(c)
    }
}

/// The result of an optimization. The state the optimizer was given is left at the optimum.
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizedResult {
    /// The optimized value of each continuous latent.
    pub values: Vec<(NodeId, DoubleTensor)>,
    /// The fitness (log probability or log likelihood) at the optimum.
    pub fitness: f64,
    /// Whether the algorithm's convergence criterion was met (rather than running out of evaluations).
    pub converged: bool,
    /// The number of fitness and gradient evaluations used.
    pub evaluations: usize,
}

impl OptimizedResult {
    /// The optimized value of a latent.
    pub fn get(&self, id: NodeId) -> Option<&DoubleTensor> {
        self.values.iter().find(|(n, _)| *n == id).map(|(_, v)| v)
    }
}

/// Finds the latent values that maximize the log probability or log likelihood of a model by gradient ascent
/// (Java's `GradientOptimizer`). The default algorithm is conjugate gradient.
#[derive(Debug, Clone)]
pub struct GradientOptimizer<'m> {
    model: &'m Model,
    algorithm: Algorithm,
    check_initial_conditions: bool,
}

/// Starting points whose largest absolute gradient element is at most this are considered flat.
const FLAT_GRADIENT: f64 = 1e-16;

impl<'m> GradientOptimizer<'m> {
    /// An optimizer for `model`, using conjugate gradient by default.
    pub fn new(model: &'m Model) -> Self {
        Self {
            model,
            algorithm: ConjugateGradient::default().into(),
            check_initial_conditions: true,
        }
    }

    /// Sets the optimization algorithm.
    pub fn algorithm(mut self, algorithm: impl Into<Algorithm>) -> Self {
        self.algorithm = algorithm.into();
        self
    }

    /// Whether to fail if the starting point is impossible or its gradient is flat (default true).
    pub fn check_initial_conditions(mut self, check: bool) -> Self {
        self.check_initial_conditions = check;
        self
    }

    /// Maximizes the joint log probability, starting from (and leaving `state` at) the latents' values.
    pub fn max_a_posteriori(&self, state: &mut State) -> Result<OptimizedResult> {
        self.optimize(state, Fitness::LogProb)
    }

    /// Maximizes the log likelihood, starting from (and leaving `state` at) the latents' values.
    pub fn max_likelihood(&self, state: &mut State) -> Result<OptimizedResult> {
        self.optimize(state, Fitness::LogLikelihood)
    }

    /// Optimizes the given fitness, leaving the state at the optimum.
    pub fn optimize(&self, state: &mut State, fitness: Fitness) -> Result<OptimizedResult> {
        let model = self.model;
        if model.latents().is_empty() {
            return Err(AlgorithmError::NoLatents);
        }
        let discrete = model.discrete_latents();
        if !discrete.is_empty() {
            let graph = model.graph();
            let names = discrete
                .iter()
                .map(|&id| {
                    graph
                        .node(id)
                        .label()
                        .map(str::to_owned)
                        .unwrap_or_else(|| format!("node {}", id.index()))
                })
                .collect();
            return Err(AlgorithmError::DiscreteLatents(names));
        }

        let objective = Objective::new(model, fitness);
        let start = objective.point(state)?;
        if self.check_initial_conditions {
            let initial = objective.value(state, &start)?;
            if initial == f64::NEG_INFINITY || initial.is_nan() {
                return Err(AlgorithmError::ImpossibleStart);
            }
            let gradient = objective.gradient(state, &start)?;
            let largest = gradient
                .iter()
                .flat_map(|g| g.as_slice().iter().map(|x| x.abs()))
                .fold(0.0, f64::max);
            if largest <= FLAT_GRADIENT {
                return Err(AlgorithmError::FlatGradient { largest });
            }
        }

        let (point, converged) = match &self.algorithm {
            Algorithm::Adam(adam) => adam.maximize(&objective, state, start)?,
            Algorithm::ConjugateGradient(cg) => cg.maximize(&objective, state, start)?,
        };
        let fitness = objective.value(state, &point)?;
        Ok(OptimizedResult {
            values: objective.latents.iter().copied().zip(point).collect(),
            fitness,
            converged,
            evaluations: objective.evaluations.get(),
        })
    }
}

/// The fitness of a model as a function of its continuous latents' values.
pub(crate) struct Objective<'m> {
    model: &'m Model,
    fitness: Fitness,
    latents: Vec<NodeId>,
    evaluations: Cell<usize>,
}

impl<'m> Objective<'m> {
    fn new(model: &'m Model, fitness: Fitness) -> Self {
        Self {
            model,
            fitness,
            latents: model.continuous_latents(),
            evaluations: Cell::new(0),
        }
    }

    pub(crate) fn evaluations(&self) -> usize {
        self.evaluations.get()
    }

    /// The latents' current values.
    fn point(&self, state: &State) -> Result<Vec<DoubleTensor>> {
        self.latents
            .iter()
            .map(|&id| {
                state.f64(id).cloned().ok_or(AlgorithmError::Graph(
                    keanu_graph::GraphError::MissingValue(id.index()),
                ))
            })
            .collect()
    }

    fn apply(&self, state: &mut State, point: &[DoubleTensor]) -> Result<()> {
        let values: Vec<(NodeId, Value)> = self
            .latents
            .iter()
            .copied()
            .zip(point.iter().cloned().map(Value::F64))
            .collect();
        self.model.set_and_cascade(state, &values)?;
        Ok(())
    }

    /// The fitness at a point (which is applied to the state).
    pub(crate) fn value(&self, state: &mut State, point: &[DoubleTensor]) -> Result<f64> {
        self.evaluations.set(self.evaluations.get() + 1);
        self.apply(state, point)?;
        Ok(match self.fitness {
            Fitness::LogProb => self.model.log_prob(state)?,
            Fitness::LogLikelihood => self.model.log_likelihood(state)?,
        })
    }

    /// The gradient of the fitness at a point (which is applied to the state), one tensor per latent. A latent the
    /// fitness doesn't depend on has a zero gradient.
    pub(crate) fn gradient(
        &self,
        state: &mut State,
        point: &[DoubleTensor],
    ) -> Result<Vec<DoubleTensor>> {
        self.evaluations.set(self.evaluations.get() + 1);
        self.apply(state, point)?;
        let gradient = match self.fitness {
            Fitness::LogProb => self.model.log_prob_gradient(state)?,
            Fitness::LogLikelihood => self.model.log_likelihood_gradient(state)?,
        };
        Ok(self
            .latents
            .iter()
            .zip(point)
            .map(|(&id, value)| {
                gradient
                    .get(id)
                    .cloned()
                    .unwrap_or_else(|| DoubleTensor::zeros(value.shape()))
            })
            .collect())
    }
}
