//! Gradient based optimizers for finding the maximum a posteriori (MAP) or maximum likelihood (MLE) values of a
//! model's continuous latents.

mod adam;
pub mod bobyqa;
mod conjugate_gradient;
mod convergence;

pub use adam::Adam;
pub use bobyqa::{Bobyqa, NonGradientOptimizer, OptimizerBounds};
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
#[non_exhaustive]
pub enum Algorithm {
    /// Adam.
    Adam(Adam),
    /// Nonlinear conjugate gradient.
    ConjugateGradient(ConjugateGradient),
}

/// Conjugate gradient, the default (as in Java).
impl Default for Algorithm {
    fn default() -> Self {
        Self::ConjugateGradient(ConjugateGradient::default())
    }
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
#[non_exhaustive]
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

    /// Whether to fail if the starting point is impossible or its gradient is flat or not finite (default true).
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
        check_latents(model)?;

        let objective = Objective::new(model, fitness);
        let start = objective.flat_point(state)?;
        // the initial-condition check's fitness and gradient are where the algorithm starts, so are passed on
        let known_start = if self.check_initial_conditions {
            let initial = start_value(&objective, state, &start)?;
            let gradient = objective.gradient_flat(state, &start)?;
            // checked separately: f64::max skips NaN, so a NaN gradient would pass the flatness check
            if !gradient.iter().all(|g| g.is_finite()) {
                return Err(AlgorithmError::NonFiniteGradient);
            }
            let largest = gradient.iter().copied().map(f64::abs).fold(0.0, f64::max);
            if largest <= FLAT_GRADIENT {
                return Err(AlgorithmError::FlatGradient { largest });
            }
            Some((initial, gradient))
        } else {
            None
        };

        let (point, fitness, converged) = match &self.algorithm {
            Algorithm::Adam(adam) => {
                let initial_gradient = known_start.map(|(_, g)| objective.unflatten(&g));
                let (point, converged) = adam.maximize(
                    &objective,
                    state,
                    objective.unflatten(&start),
                    initial_gradient,
                )?;
                // Adam's last gradient was taken before its last step, so the fitness at the end is unknown
                let fitness = objective.value(state, &point)?;
                (point, fitness, converged)
            }
            Algorithm::ConjugateGradient(cg) => {
                let (x, fitness, converged) = cg.maximize(&objective, state, start, known_start)?;
                objective.apply_flat(state, &x)?;
                (objective.unflatten(&x), fitness, converged)
            }
        };
        Ok(OptimizedResult {
            values: objective.latents.iter().copied().zip(point).collect(),
            fitness,
            converged,
            evaluations: objective.evaluations(),
        })
    }
}

/// The fitness at the starting point, failing if it is impossible (-infinity or NaN): the initial-condition check
/// shared by the optimizers.
pub(crate) fn start_value(
    objective: &Objective<'_>,
    state: &mut State,
    start: &[f64],
) -> Result<f64> {
    let initial = objective.value_flat(state, start)?;
    if initial == f64::NEG_INFINITY || initial.is_nan() {
        return Err(AlgorithmError::ImpossibleStart);
    }
    Ok(initial)
}

/// Optimizers need at least one latent, and only continuous ones.
pub(crate) fn check_latents(model: &Model) -> Result<()> {
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
    Ok(())
}

/// The fitness of a model as a function of its continuous latents' values, either as one tensor per latent or as
/// one flat vector of every latent's elements (in latent order, each row-major).
pub(crate) struct Objective<'m> {
    model: &'m Model,
    fitness: Fitness,
    latents: Vec<NodeId>,
    shapes: Vec<Vec<usize>>,
    evaluations: Cell<usize>,
}

impl<'m> Objective<'m> {
    pub(crate) fn new(model: &'m Model, fitness: Fitness) -> Self {
        let latents = model.continuous_latents();
        let shapes = latents
            .iter()
            .map(|&id| model.graph().node(id).shape().to_vec())
            .collect();
        Self {
            model,
            fitness,
            latents,
            shapes,
            evaluations: Cell::new(0),
        }
    }

    /// The number of fitness and gradient evaluations so far.
    pub(crate) fn evaluations(&self) -> usize {
        self.evaluations.get()
    }

    pub(crate) fn latents(&self) -> &[NodeId] {
        &self.latents
    }

    /// The latents' current values as one flat vector.
    pub(crate) fn flat_point(&self, state: &State) -> Result<Vec<f64>> {
        let len = self
            .shapes
            .iter()
            .map(|s| s.iter().product::<usize>())
            .sum();
        let mut x = Vec::with_capacity(len);
        for &id in &self.latents {
            let value = state.f64(id).ok_or(AlgorithmError::Graph(
                keanu_graph::GraphError::MissingValue(id),
            ))?;
            x.extend_from_slice(value.as_slice());
        }
        Ok(x)
    }

    /// The flat vector `x` as one tensor per latent.
    pub(crate) fn unflatten(&self, x: &[f64]) -> Vec<DoubleTensor> {
        let mut offset = 0;
        self.shapes
            .iter()
            .map(|shape| {
                let len: usize = shape.iter().product();
                let tensor = DoubleTensor::new(shape, x[offset..offset + len].to_vec())
                    .expect("a latent's shape matches its length");
                offset += len;
                tensor
            })
            .collect()
    }

    fn set(&self, state: &mut State, point: impl IntoIterator<Item = DoubleTensor>) -> Result<()> {
        let values: Vec<(NodeId, Value)> = self
            .latents
            .iter()
            .copied()
            .zip(point.into_iter().map(Value::F64))
            .collect();
        self.model.set_and_cascade(state, values)?;
        Ok(())
    }

    /// Sets the latents to the flat vector `x` and recomputes their deterministic descendants.
    pub(crate) fn apply_flat(&self, state: &mut State, x: &[f64]) -> Result<()> {
        self.set(state, self.unflatten(x))
    }

    fn fitness_of(&self, state: &State) -> Result<f64> {
        Ok(match self.fitness {
            Fitness::LogProb => self.model.log_prob(state)?,
            Fitness::LogLikelihood => self.model.log_likelihood(state)?,
        })
    }

    fn gradient_of(&self, state: &State) -> Result<keanu_graph::LogProbGradient> {
        Ok(match self.fitness {
            Fitness::LogProb => self.model.log_prob_gradient(state)?,
            Fitness::LogLikelihood => self.model.log_likelihood_gradient(state)?,
        })
    }

    fn count(&self) {
        self.evaluations.set(self.evaluations.get() + 1);
    }

    /// The fitness at a point, one tensor per latent (which is applied to the state).
    pub(crate) fn value(&self, state: &mut State, point: &[DoubleTensor]) -> Result<f64> {
        self.count();
        self.set(state, point.iter().cloned())?;
        self.fitness_of(state)
    }

    /// The fitness at the flat vector `x` (which is applied to the state).
    pub(crate) fn value_flat(&self, state: &mut State, x: &[f64]) -> Result<f64> {
        self.count();
        self.apply_flat(state, x)?;
        self.fitness_of(state)
    }

    /// The gradient of the fitness at a point (which is applied to the state), one tensor per latent. A latent the
    /// fitness doesn't depend on has a zero gradient.
    pub(crate) fn gradient(
        &self,
        state: &mut State,
        point: &[DoubleTensor],
    ) -> Result<Vec<DoubleTensor>> {
        self.count();
        self.set(state, point.iter().cloned())?;
        let gradient = self.gradient_of(state)?;
        Ok(self
            .latents
            .iter()
            .zip(&self.shapes)
            .map(|(&id, shape)| {
                gradient
                    .get(id)
                    .cloned()
                    .unwrap_or_else(|| DoubleTensor::zeros(shape))
            })
            .collect())
    }

    /// The gradient of the fitness at the flat vector `x` (which is applied to the state), flattened like `x`. A
    /// latent the fitness doesn't depend on has a zero gradient.
    pub(crate) fn gradient_flat(&self, state: &mut State, x: &[f64]) -> Result<Vec<f64>> {
        self.count();
        self.apply_flat(state, x)?;
        let gradient = self.gradient_of(state)?;
        let mut flat = Vec::with_capacity(x.len());
        for (&id, shape) in self.latents.iter().zip(&self.shapes) {
            match gradient.get(id) {
                Some(g) => flat.extend_from_slice(g.as_slice()),
                None => flat.extend(std::iter::repeat_n(0.0, shape.iter().product())),
            }
        }
        Ok(flat)
    }
}
