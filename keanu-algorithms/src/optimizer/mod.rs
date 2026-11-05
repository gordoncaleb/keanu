//! Gradient based optimizers for finding the maximum a posteriori (MAP) or maximum likelihood (MLE) values of a
//! model's continuous latents.
//!
//! # Unconstrained space
//!
//! By default the gradient optimizers move in unconstrained space ([`Space::Unconstrained`]): each latent is mapped
//! to ℝⁿ through the transform of its support (see [`Parameterisation`]), so every point they try is inside the
//! support. The fitness is the model's log probability (or log likelihood) at the constrained values, *without* the
//! log-Jacobian of the transforms, so an optimum inside the support is exactly the constrained optimum: the MAP of
//! `Gamma(theta = 0.5, k = 2)` is its mode, 0.5.
//!
//! A mode on the boundary of a support (a scale whose mode is 0) is at infinity in unconstrained space. Once a
//! coordinate of a bounded latent passes ±30 (where `exp` of it is below 10⁻¹³ or above 10¹³) the boundary is
//! checked for: if the gradient there points outwards and the fitness at twice that distance is no worse, the
//! optimizer stops and reports the latent in [`OptimizedResult::boundary`], with `converged` false and the values
//! left close to the boundary. Otherwise the optimum is in the interior, however far from 1 (a scale of 10¹⁵, say),
//! and the optimizer carries on. Optimizers can also stall in the flat tail on the way to a boundary, so when one
//! stops with a bounded coordinate beyond ±5 whose gradient still points outwards, that coordinate is tried further
//! out (at ±31, or twice its distance beyond ±30): if the fitness there is no worse, the result is the same boundary
//! result.
//!
//! [`GradientOptimizer::space`]`(Space::Constrained)` optimizes the latents' own values instead, as Keanu did before
//! transforms existed. [`NonGradientOptimizer`] always works on the constrained values, bounded by the supports.

mod adam;
pub mod bobyqa;
mod conjugate_gradient;
mod convergence;
mod lbfgs;
mod line_search;

pub use adam::Adam;
pub use bobyqa::{Bobyqa, NonGradientOptimizer, OptimizerBounds};
pub use conjugate_gradient::{ConjugateGradient, UpdateFormula};
pub use convergence::{ConvergenceChecker, Norm};
pub use lbfgs::Lbfgs;

use crate::error::{AlgorithmError, Result};
use crate::support::{
    block_shapes, check_latents, compute_deterministic, concat, split, start_point,
};
use keanu_graph::{Model, NodeId, Parameterisation, Space, State, Support, node_support};
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
    /// Limited-memory BFGS.
    Lbfgs(Lbfgs),
}

/// L-BFGS, the default.
impl Default for Algorithm {
    fn default() -> Self {
        Self::Lbfgs(Lbfgs::default())
    }
}

impl From<Lbfgs> for Algorithm {
    fn from(l: Lbfgs) -> Self {
        Self::Lbfgs(l)
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
    /// Whether the algorithm's convergence criterion was met (rather than running out of evaluations, or stopping
    /// at a [`boundary`](Self::boundary)).
    pub converged: bool,
    /// The number of fitness and gradient evaluations used.
    pub evaluations: usize,
    /// The latent whose optimum is on the boundary of its support, if the optimizer stopped because of one (see the
    /// [module documentation](self)). Its value is then close to the boundary rather than an interior optimum, and
    /// `converged` is false. Always `None` in [`Space::Constrained`] and for the [`NonGradientOptimizer`].
    pub boundary: Option<NodeId>,
}

impl OptimizedResult {
    /// The optimized value of a latent.
    pub fn get(&self, id: NodeId) -> Option<&DoubleTensor> {
        self.values.iter().find(|(n, _)| *n == id).map(|(_, v)| v)
    }
}

/// Finds the latent values that maximize the log probability or log likelihood of a model by gradient ascent
/// (Java's `GradientOptimizer`). The default algorithm is [`Lbfgs`], in unconstrained space (see the
/// [module documentation](self)).
#[derive(Debug, Clone)]
pub struct GradientOptimizer<'m> {
    model: &'m Model,
    algorithm: Algorithm,
    check_initial_conditions: bool,
    space: Space,
}

/// Starting points whose largest absolute gradient element is at most this are considered flat.
const FLAT_GRADIENT: f64 = 1e-16;

/// An unconstrained coordinate of a bounded latent beyond ± this is checked for heading to the boundary of its
/// support (see [`Objective::boundary`]).
const BOUNDARY: f64 = 30.0;

/// A bounded coordinate the optimizer stopped beyond ± this is checked for a boundary optimum (see
/// [`Objective::probe_boundary`]).
const PROBE: f64 = 5.0;

/// The most times the point a boundary probe tries is brought halfway back when its fitness is not finite.
const PROBE_HALVINGS: usize = 10;

impl<'m> GradientOptimizer<'m> {
    /// An optimizer for `model`, using L-BFGS in unconstrained space by default.
    pub fn new(model: &'m Model) -> Self {
        Self {
            model,
            algorithm: Algorithm::default(),
            check_initial_conditions: true,
            space: Space::Unconstrained,
        }
    }

    /// Sets the optimization algorithm.
    pub fn algorithm(mut self, algorithm: impl Into<Algorithm>) -> Self {
        self.algorithm = algorithm.into();
        self
    }

    /// Whether to fail if the starting point is impossible or its gradient is flat or not finite (default true).
    /// In unconstrained space a start outside a latent's support, or on its boundary, has no point to start from,
    /// so it is an [`AlgorithmError::ImpossibleStart`] either way.
    pub fn check_initial_conditions(mut self, check: bool) -> Self {
        self.check_initial_conditions = check;
        self
    }

    /// Sets the space to optimize in (default [`Space::Unconstrained`]; see the [module documentation](self)).
    pub fn space(mut self, space: Space) -> Self {
        self.space = space;
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
    ///
    /// # Errors
    ///
    /// [`AlgorithmError::ImpossibleStart`], [`AlgorithmError::FlatGradient`] and
    /// [`AlgorithmError::NonFiniteGradient`] for a bad starting point (see
    /// [`check_initial_conditions`](Self::check_initial_conditions)), errors for models without continuous latents
    /// or with discrete ones, and, in unconstrained space, an error from [`Parameterisation::new`] if a latent's
    /// support has no transform.
    pub fn optimize(&self, state: &mut State, fitness: Fitness) -> Result<OptimizedResult> {
        let model = self.model;
        check_latents(model)?;
        compute_deterministic(model.graph(), state)?;

        let param = Parameterisation::with_space(model, state, self.space)?;
        let objective = Objective::new(param, state, fitness, false)?;
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
                (concat(&point), fitness, converged)
            }
            Algorithm::ConjugateGradient(cg) => {
                let (x, fitness, converged) = cg.maximize(&objective, state, start, known_start)?;
                objective.apply_flat(state, &x)?;
                (x, fitness, converged)
            }
            Algorithm::Lbfgs(lbfgs) => {
                let (x, fitness, converged) =
                    lbfgs.maximize(&objective, state, start, known_start)?;
                objective.apply_flat(state, &x)?;
                (x, fitness, converged)
            }
        };
        let (point, fitness, boundary) = objective.probe_boundary(state, point, fitness)?;
        Ok(OptimizedResult {
            values: objective.constrained_values(&point)?,
            fitness,
            converged: converged && boundary.is_none(),
            evaluations: objective.evaluations(),
            boundary,
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

/// Where a bounded coordinate at `z` is tried to confirm it is heading for a boundary: ±([`BOUNDARY`] + 1) inside
/// the threshold, and twice as far out beyond it (so that an interior optimum just past the threshold, where the
/// fitness falls steeply further out, is not mistaken for one).
fn further_out(z: f64) -> f64 {
    if z.abs() <= BOUNDARY {
        (BOUNDARY + 1.0).copysign(z)
    } else {
        2.0 * z
    }
}

/// The coordinates of the flat vector that go to ±∞ as their latent approaches the boundary of its support: every
/// coordinate of a bounded latent except those of an identity (a real support, an angle, or the first element of an
/// ordered vector). None in constrained space.
fn boundary_coordinates(param: &Parameterisation<'_>, state: &State) -> Result<Vec<usize>> {
    if param.space() == Space::Constrained {
        return Ok(Vec::new());
    }
    let graph = param.model().graph();
    let mut coordinates = Vec::new();
    for &id in param.latents() {
        let range = param.range(id).unwrap_or_default();
        match node_support(graph, state, id)? {
            Support::Real | Support::Circular | Support::Discrete => {}
            Support::Ordered => {
                let row = graph[id].shape().last().copied().unwrap_or(1).max(1);
                coordinates.extend(range.clone().filter(|i| (i - range.start) % row != 0));
            }
            _ => coordinates.extend(range),
        }
    }
    Ok(coordinates)
}

/// The fitness of a model as a function of a point in a [`Parameterisation`]'s flat vector, either as that vector
/// or as one tensor per latent (see [`block_shapes`]).
///
/// With `jacobian` the log probability is that of the point, including the log-Jacobian of the transforms; MAP and
/// maximum likelihood use it without, so that interior optima are the constrained optima. The Jacobian applies to
/// the log probability only.
pub(crate) struct Objective<'m> {
    param: Parameterisation<'m>,
    fitness: Fitness,
    jacobian: bool,
    shapes: Vec<Vec<usize>>,
    boundary_coordinates: Vec<usize>,
    evaluations: Cell<usize>,
}

impl<'m> Objective<'m> {
    /// The objective for `fitness` over `param`, whose supports are read from `state`.
    pub(crate) fn new(
        param: Parameterisation<'m>,
        state: &State,
        fitness: Fitness,
        jacobian: bool,
    ) -> Result<Self> {
        if jacobian && fitness != Fitness::LogProb {
            return Err(AlgorithmError::InvalidArgument(
                "the Jacobian applies to the log probability only".into(),
            ));
        }
        Ok(Self {
            shapes: block_shapes(&param),
            boundary_coordinates: boundary_coordinates(&param, state)?,
            param,
            fitness,
            jacobian,
            evaluations: Cell::new(0),
        })
    }

    /// The number of fitness and gradient evaluations so far.
    pub(crate) fn evaluations(&self) -> usize {
        self.evaluations.get()
    }

    pub(crate) fn latents(&self) -> &[NodeId] {
        self.param.latents()
    }

    /// The parameterisation the objective is over.
    pub(crate) fn parameterisation(&self) -> &Parameterisation<'m> {
        &self.param
    }

    /// The parameterisation the objective is over, giving it up.
    pub(crate) fn into_parameterisation(self) -> Parameterisation<'m> {
        self.param
    }

    /// The point of the latents' current values: an impossible start if a value has none.
    pub(crate) fn flat_point(&self, state: &State) -> Result<Vec<f64>> {
        start_point(&self.param, state)
    }

    /// The flat vector `x` as one tensor per latent.
    pub(crate) fn unflatten(&self, x: &[f64]) -> Vec<DoubleTensor> {
        split(x, &self.shapes)
    }

    /// Each latent's constrained value at `x`.
    pub(crate) fn constrained_values(&self, x: &[f64]) -> Result<Vec<(NodeId, DoubleTensor)>> {
        Ok(self
            .latents()
            .iter()
            .copied()
            .zip(self.param.to_constrained(x)?)
            .collect())
    }

    /// The latent of the first coordinate of `x` that is past the boundary threshold ±[`BOUNDARY`] and heading
    /// for the boundary of its support, if any: its gradient points outwards and the fitness further out is no worse
    /// (see [`probe_boundary`](Self::probe_boundary)). A coordinate past the threshold whose optimum is in the
    /// interior (far from 1) is not a boundary. Leaves the state at `x`.
    pub(crate) fn boundary(&self, state: &mut State, x: &[f64]) -> Result<Option<NodeId>> {
        if !self
            .boundary_coordinates
            .iter()
            .any(|&i| x[i].abs() > BOUNDARY)
        {
            return Ok(None);
        }
        let f = self.value_flat(state, x)?;
        let (_, _, latent) = self.probe(state, x.to_vec(), f, BOUNDARY)?;
        self.apply_flat(state, x)?;
        Ok(latent)
    }

    /// Checks whether the point `x` (with fitness `f`) an algorithm stopped at is really heading for a boundary:
    /// optimizers can stall in the flat tail on the way, where steps shrink or the fitness stops changing, or stop
    /// at the threshold ([`boundary`](Self::boundary)). Each bounded coordinate already beyond ±[`PROBE`] whose
    /// gradient points further out is tried further out (see [`further_out`]); if the fitness there is no worse, the
    /// optimum is on the boundary and the optimizer moves there. Returns the final point, its fitness and the first
    /// latent found on a boundary, leaving the state at the point.
    fn probe_boundary(
        &self,
        state: &mut State,
        x: Vec<f64>,
        f: f64,
    ) -> Result<(Vec<f64>, f64, Option<NodeId>)> {
        self.probe(state, x, f, PROBE)
    }

    /// [`probe_boundary`](Self::probe_boundary) for the bounded coordinates beyond ±`threshold`.
    fn probe(
        &self,
        state: &mut State,
        mut x: Vec<f64>,
        mut f: f64,
        threshold: f64,
    ) -> Result<(Vec<f64>, f64, Option<NodeId>)> {
        let far: Vec<usize> = self
            .boundary_coordinates
            .iter()
            .copied()
            .filter(|&i| x[i].abs() > threshold)
            .collect();
        if far.is_empty() {
            return Ok((x, f, None));
        }
        let gradient = self.gradient_flat(state, &x)?;
        let mut boundary = None;
        for i in far {
            if x[i] * gradient[i] <= 0.0 {
                continue;
            }
            let mut y = x.clone();
            y[i] = further_out(x[i]);
            let mut fy = self.value_flat(state, &y)?;
            // far out the constrained value can underflow (or overflow) to an impossible point: come back towards
            // x until the fitness is finite
            for _ in 0..PROBE_HALVINGS {
                if fy.is_finite() {
                    break;
                }
                y[i] = 0.5 * (x[i] + y[i]);
                fy = self.value_flat(state, &y)?;
            }
            if fy >= f {
                (x, f) = (y, fy);
                boundary = boundary.or_else(|| self.param.coordinate_latent(i));
            }
        }
        self.apply_flat(state, &x)?;
        Ok((x, f, boundary))
    }

    /// As [`boundary`](Self::boundary), for a point given as one tensor per latent.
    pub(crate) fn boundary_of(
        &self,
        state: &mut State,
        point: &[DoubleTensor],
    ) -> Result<Option<NodeId>> {
        if self.boundary_coordinates.is_empty() {
            return Ok(None);
        }
        self.boundary(state, &concat(point))
    }

    /// Sets the latents to their values at the flat vector `x` and recomputes their deterministic descendants.
    pub(crate) fn apply_flat(&self, state: &mut State, x: &[f64]) -> Result<()> {
        Ok(self.param.to_state(x, state)?)
    }

    fn fitness_of(&self, state: &State, x: &[f64]) -> Result<f64> {
        Ok(match self.fitness {
            Fitness::LogProb if self.jacobian => {
                self.param.model().log_prob(state)? + self.param.log_det_jacobian(x)?
            }
            Fitness::LogProb => self.param.model().log_prob(state)?,
            Fitness::LogLikelihood => self.param.model().log_likelihood(state)?,
        })
    }

    fn count(&self) {
        self.evaluations.set(self.evaluations.get() + 1);
    }

    /// The fitness at a point, one tensor per latent (which is applied to the state).
    pub(crate) fn value(&self, state: &mut State, point: &[DoubleTensor]) -> Result<f64> {
        self.value_flat(state, &concat(point))
    }

    /// The fitness at the flat vector `x` (which is applied to the state).
    pub(crate) fn value_flat(&self, state: &mut State, x: &[f64]) -> Result<f64> {
        self.count();
        self.apply_flat(state, x)?;
        self.fitness_of(state, x)
    }

    /// The fitness and its gradient at the flat vector `x` (which is applied to the state), as
    /// [`value_flat`](Self::value_flat) and [`gradient_flat`](Self::gradient_flat) (and counted as both), in one
    /// pass for the log probability.
    pub(crate) fn value_and_gradient_flat(
        &self,
        state: &mut State,
        x: &[f64],
    ) -> Result<(f64, Vec<f64>)> {
        if self.fitness != Fitness::LogProb {
            let value = self.value_flat(state, x)?;
            return Ok((value, self.gradient_flat(state, x)?));
        }
        self.count();
        self.count();
        // where the log probability is impossible there is no gradient
        let (value, gradient) = self.param.log_prob_and_gradient(x, self.jacobian, state)?;
        Ok((value, gradient.unwrap_or_else(|| vec![f64::NAN; x.len()])))
    }

    /// The gradient of the fitness at a point (which is applied to the state), one tensor per latent. A latent the
    /// fitness doesn't depend on has a zero gradient.
    pub(crate) fn gradient(
        &self,
        state: &mut State,
        point: &[DoubleTensor],
    ) -> Result<Vec<DoubleTensor>> {
        Ok(self.unflatten(&self.gradient_flat(state, &concat(point))?))
    }

    /// The gradient of the fitness at the flat vector `x` (which is applied to the state), flattened like `x`. A
    /// latent the fitness doesn't depend on has a zero gradient.
    pub(crate) fn gradient_flat(&self, state: &mut State, x: &[f64]) -> Result<Vec<f64>> {
        self.count();
        if self.jacobian {
            // only the log probability takes the Jacobian; where it is impossible there is no gradient
            let (_, gradient) = self.param.log_prob_and_gradient(x, true, state)?;
            return Ok(gradient.unwrap_or_else(|| vec![f64::NAN; x.len()]));
        }
        self.apply_flat(state, x)?;
        let model = self.param.model();
        let gradient = match self.fitness {
            Fitness::LogProb => model.log_prob_gradient(state)?,
            Fitness::LogLikelihood => model.log_likelihood_gradient(state)?,
        };
        Ok(self.param.pull_back(x, &gradient)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use keanu_graph::GraphBuilder;

    /// With the Jacobian, the objective is the log density of the unconstrained point, and its gradient agrees with
    /// finite differences of it.
    #[test]
    fn the_jacobian_objective_is_the_density_of_the_point() {
        let b = GraphBuilder::new();
        let sigma = b.gamma(0.5, 2.0);
        b.set_value(sigma, DoubleTensor::scalar(2.0));
        let y = b.gaussian(0.0, sigma);
        b.observe(y, DoubleTensor::scalar(0.7));
        let sigma = sigma.id();
        let (graph, mut state) = b.build().unwrap();
        let model = Model::new(graph, &state).unwrap();
        let param = Parameterisation::new(&model, &state).unwrap();
        let objective = Objective::new(param, &state, Fitness::LogProb, true).unwrap();

        let z = [0.3];
        let sigma_value = 0.3f64.exp();
        let value = objective.value_flat(&mut state, &z).unwrap();
        let without = model.log_prob(&state).unwrap();
        assert!((value - (without + 0.3)).abs() < 1e-12, "log σ = z adds z");
        assert_eq!(state.f64(sigma).unwrap().to_scalar(), Some(sigma_value));

        let gradient = objective.gradient_flat(&mut state, &z).unwrap();
        let h = 1e-6;
        let numeric = (objective.value_flat(&mut state, &[z[0] + h]).unwrap()
            - objective.value_flat(&mut state, &[z[0] - h]).unwrap())
            / (2.0 * h);
        assert!(
            (gradient[0] - numeric).abs() < 1e-6,
            "{gradient:?} vs {numeric}"
        );

        let param = Parameterisation::new(&model, &state).unwrap();
        assert!(matches!(
            Objective::new(param, &state, Fitness::LogLikelihood, true),
            Err(AlgorithmError::InvalidArgument(_))
        ));
    }
}
