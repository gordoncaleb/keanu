//! Keanu's non-gradient optimizer: BOBYQA (bound optimisation by quadratic approximation).
//!
//! The solver in [`prima`] is a translation of PRIMA's BOBYQA and is distributed under PRIMA's BSD 3-Clause
//! licence (see `THIRD-PARTY-NOTICES.md` in this crate). This file is part of keanu-algorithms and MIT licensed.

pub mod prima;

use super::{Fitness, Objective, OptimizedResult, start_value};
use crate::error::{AlgorithmError, Result};
use keanu_graph::{Model, NodeId, Parameterisation, Space, State, Support, node_support};
use keanu_tensor::DoubleTensor;
use std::collections::HashMap;
use std::ops::ControlFlow;

/// BOBYQA settings (Java's `BOBYQA` builder).
///
/// As in PRIMA, some settings that can't be used as given are adjusted rather than rejected, without warning (for
/// n latent dimensions):
/// - `max_evaluations` below n + 3 is raised to n + 3, the fewest BOBYQA can work with;
/// - `interpolation_points` outside n + 2 to (n + 1)(n + 2) / 2, or not below the (adjusted) `max_evaluations`,
///   is replaced by the default, 2n + 1 (or `max_evaluations` - 1 if that is smaller);
/// - `initial_trust_region_radius` is reduced to fit the bounds (to a quarter of the smallest bound range if it is
///   more than half of it, and to the distance from the start to its nearest bound), and
///   `stopping_trust_region_radius` is reduced in proportion.
///
/// If the objective fails (returns an error) the optimization stops at once with that error.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Bobyqa {
    /// The maximum number of fitness evaluations (default: unlimited). At least n + 3 are used for n latent
    /// dimensions.
    pub max_evaluations: usize,
    /// Latents without explicit bounds are bounded to their starting value ± this (default: unbounded).
    pub bounds_range: f64,
    /// The initial trust region radius (default 10, as in Keanu; reduced to fit the bounds if needed).
    pub initial_trust_region_radius: f64,
    /// The final trust region radius: the accuracy required in the latents (default 1e-8).
    pub stopping_trust_region_radius: f64,
    /// The number of interpolation points, between n + 2 and (n + 1)(n + 2) / 2 for n latent dimensions
    /// (default 2n + 1). A number outside that range is replaced by the default.
    pub interpolation_points: Option<usize>,
}

/// Setters for each setting, for use as `Bobyqa::default().with_...(...)`.
impl Bobyqa {
    /// Sets [`max_evaluations`](Self::max_evaluations).
    #[must_use]
    pub fn with_max_evaluations(mut self, max_evaluations: usize) -> Self {
        self.max_evaluations = max_evaluations;
        self
    }

    /// Sets [`bounds_range`](Self::bounds_range).
    #[must_use]
    pub fn with_bounds_range(mut self, bounds_range: f64) -> Self {
        self.bounds_range = bounds_range;
        self
    }

    /// Sets [`initial_trust_region_radius`](Self::initial_trust_region_radius).
    #[must_use]
    pub fn with_initial_trust_region_radius(mut self, initial_trust_region_radius: f64) -> Self {
        self.initial_trust_region_radius = initial_trust_region_radius;
        self
    }

    /// Sets [`stopping_trust_region_radius`](Self::stopping_trust_region_radius).
    #[must_use]
    pub fn with_stopping_trust_region_radius(mut self, stopping_trust_region_radius: f64) -> Self {
        self.stopping_trust_region_radius = stopping_trust_region_radius;
        self
    }

    /// Sets [`interpolation_points`](Self::interpolation_points) to `Some(interpolation_points)`.
    #[must_use]
    pub fn with_interpolation_points(mut self, interpolation_points: usize) -> Self {
        self.interpolation_points = Some(interpolation_points);
        self
    }
}

impl Default for Bobyqa {
    fn default() -> Self {
        Self {
            max_evaluations: usize::MAX,
            bounds_range: f64::INFINITY,
            initial_trust_region_radius: 10.0,
            stopping_trust_region_radius: 1e-8,
            interpolation_points: None,
        }
    }
}

impl Bobyqa {
    fn validate(&self) -> Result<()> {
        let invalid = |m: &str| Err(AlgorithmError::InvalidSetting(m.to_owned()));
        if self.max_evaluations == 0 {
            return invalid("max_evaluations must be positive");
        }
        if self.bounds_range.is_nan() || self.bounds_range <= 0.0 {
            return invalid("bounds_range must be positive");
        }
        if !(self.initial_trust_region_radius > 0.0 && self.initial_trust_region_radius.is_finite())
        {
            return invalid("initial_trust_region_radius must be positive and finite");
        }
        if !(self.stopping_trust_region_radius > 0.0
            && self.stopping_trust_region_radius <= self.initial_trust_region_radius)
        {
            return invalid(
                "stopping_trust_region_radius must be positive and at most initial_trust_region_radius",
            );
        }
        Ok(())
    }
}

/// Bounds on individual latents for the non-gradient optimizer (Java's `OptimizerBounds`). A bound is a scalar
/// (applied to every element) or a tensor shaped like the latent.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OptimizerBounds {
    bounds: HashMap<NodeId, (DoubleTensor, DoubleTensor)>,
}

impl OptimizerBounds {
    /// No bounds.
    pub fn new() -> Self {
        Self::default()
    }

    /// Bounds a latent between `min` and `max`. A lower bound of -infinity or an upper bound of +infinity means
    /// no bound on that side; a NaN bound, a lower bound that is not below its upper bound, or bounds that don't
    /// overlap the latent's support make the optimization fail.
    pub fn add(mut self, latent: NodeId, min: DoubleTensor, max: DoubleTensor) -> Self {
        self.bounds.insert(latent, (min, max));
        self
    }

    /// Bounds every element of a latent between two scalars.
    pub fn add_scalar(self, latent: NodeId, min: f64, max: f64) -> Self {
        self.add(latent, DoubleTensor::scalar(min), DoubleTensor::scalar(max))
    }

    /// The bounds of a latent, if any.
    pub fn get(&self, latent: NodeId) -> Option<&(DoubleTensor, DoubleTensor)> {
        self.bounds.get(&latent)
    }
}

/// Finds the latent values that maximize the log probability or log likelihood of a model without using
/// gradients (Java's `NonGradientOptimizer`), with BOBYQA.
///
/// BOBYQA works on the latents' constrained values ([`Space::Constrained`]): it is a bounded optimizer, so instead
/// of transforming the latents it bounds each element of a latent with a lower or interval support (a scale, a
/// probability) to that support, intersected with any explicit [`OptimizerBounds`] and with the
/// [`bounds_range`](Bobyqa::bounds_range) around the start. It can still evaluate a support's bound itself, where
/// the log probability may be impossible. If any latent's support moves with another latent (`x ~ Uniform(0, hi)`
/// with a latent `hi`) no box can follow it, so no latent is bounded by its support.
#[derive(Debug, Clone)]
pub struct NonGradientOptimizer<'m> {
    model: &'m Model,
    algorithm: Bobyqa,
    bounds: OptimizerBounds,
    check_initial_conditions: bool,
}

impl<'m> NonGradientOptimizer<'m> {
    /// An optimizer for `model` with the default BOBYQA settings and no bounds.
    pub fn new(model: &'m Model) -> Self {
        Self {
            model,
            algorithm: Bobyqa::default(),
            bounds: OptimizerBounds::new(),
            check_initial_conditions: true,
        }
    }

    /// Sets the BOBYQA settings.
    pub fn algorithm(mut self, algorithm: Bobyqa) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Sets bounds on individual latents.
    pub fn bounds(mut self, bounds: OptimizerBounds) -> Self {
        self.bounds = bounds;
        self
    }

    /// Whether to fail if the starting point is impossible (default true).
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
        self.algorithm.validate()?;
        crate::support::check_latents(self.model)?;
        crate::support::compute_deterministic(self.model.graph(), state)?;
        let param = Parameterisation::with_space(self.model, state, Space::Constrained)?;
        let objective = Objective::new(param, state, fitness, false)?;
        let x0 = objective.flat_point(state)?;
        // the initial-condition check's value is reused for the solver's first evaluation, which is at x0
        let mut known_start = if self.check_initial_conditions {
            Some(start_value(&objective, state, &x0)?)
        } else {
            None
        };
        let (xl, xu) = self.flat_bounds(state, objective.latents(), &objective.unflatten(&x0))?;

        let mut failure: Option<AlgorithmError> = None;
        // the last and the best points evaluated, with their fitness, so the fitness of the point the solver
        // returns is known without evaluating it again
        let mut last: Option<(Vec<f64>, f64)> = None;
        let mut best: Option<(Vec<f64>, f64)> = None;
        let settings = prima::Settings {
            rhobeg: Some(self.algorithm.initial_trust_region_radius),
            rhoend: Some(self.algorithm.stopping_trust_region_radius),
            maxfun: Some(self.algorithm.max_evaluations),
            npt: self.algorithm.interpolation_points,
            ftarget: None,
            honour_x0: Some(true),
        };
        let outcome = prima::minimize_with_termination(
            |x| {
                let value = match known_start.take() {
                    Some(value) if x == x0.as_slice() => value,
                    _ => match objective.value_flat(state, x) {
                        Ok(value) => value,
                        Err(e) => {
                            // stop at once: there is no point continuing once the objective fails
                            failure = Some(e);
                            return ControlFlow::Break(f64::NAN);
                        }
                    },
                };
                if best.as_ref().is_none_or(|(_, b)| value > *b || b.is_nan()) {
                    best = Some((x.to_vec(), value));
                }
                last = Some((x.to_vec(), value));
                ControlFlow::Continue(-value)
            },
            &x0,
            &xl,
            &xu,
            &settings,
        );
        if let Some(e) = failure {
            return Err(e);
        }
        if outcome.status == prima::ExitStatus::NoSpaceBetweenBounds {
            return Err(AlgorithmError::InvalidArgument(
                "every lower bound must be below its upper bound".into(),
            ));
        }
        let known = [last, best]
            .into_iter()
            .flatten()
            .find(|(x, _)| *x == outcome.x)
            .map(|(_, value)| value);
        let fitness = match known {
            Some(value) => {
                objective.apply_flat(state, &outcome.x)?;
                value
            }
            // the solver's point is (in rare cases) recomputed from a shifted base, so may differ in its last bits
            // from any point it evaluated
            None => objective.value_flat(state, &outcome.x)?,
        };
        Ok(OptimizedResult {
            values: objective
                .latents()
                .iter()
                .copied()
                .zip(objective.unflatten(&outcome.x))
                .collect(),
            fitness,
            // an impossible optimum (only possible with the initial-condition check off) never counts as converged
            converged: fitness.is_finite()
                && matches!(
                    outcome.status,
                    prima::ExitStatus::SmallTrustRegionRadius | prima::ExitStatus::TargetReached
                ),
            evaluations: objective.evaluations(),
            boundary: None,
        })
    }

    /// Flattened lower and upper bounds: explicit bounds where given, otherwise start ± bounds_range, each
    /// intersected with the latent's support. Supports are only used if none of them moves with another latent
    /// (exactly when an unconstrained [`Parameterisation`] exists): a moving bound can't be a fixed box.
    fn flat_bounds(
        &self,
        state: &State,
        latents: &[NodeId],
        start: &[DoubleTensor],
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let fixed_supports = Parameterisation::new(self.model, state).is_ok();
        let mut xl = Vec::new();
        let mut xu = Vec::new();
        for (&id, value) in latents.iter().zip(start) {
            let offset = xl.len();
            match self.bounds.get(id) {
                Some((min, max)) => {
                    for (bound, out, name) in [(min, &mut xl, "lower"), (max, &mut xu, "upper")] {
                        if bound.as_slice().iter().any(|b| b.is_nan()) {
                            return Err(AlgorithmError::InvalidArgument(format!(
                                "the {name} bound of a latent is NaN"
                            )));
                        }
                        if bound.shape() == value.shape() {
                            out.extend_from_slice(bound.as_slice());
                        } else if let Some(v) = bound.to_scalar() {
                            out.extend(std::iter::repeat_n(v, value.len()));
                        } else {
                            return Err(AlgorithmError::InvalidArgument(format!(
                                "{name} bound shape {:?} does not match latent shape {:?}",
                                bound.shape(),
                                value.shape()
                            )));
                        }
                    }
                }
                None => {
                    xl.extend(
                        value
                            .as_slice()
                            .iter()
                            .map(|v| v - self.algorithm.bounds_range),
                    );
                    xu.extend(
                        value
                            .as_slice()
                            .iter()
                            .map(|v| v + self.algorithm.bounds_range),
                    );
                }
            }
            if !fixed_supports {
                continue;
            }
            let (lower, upper) = support_bounds(self.model, state, id, value)?;
            let explicit = self.bounds.get(id).is_some();
            for (i, (lo, hi)) in lower.into_iter().zip(upper).enumerate() {
                let (l, u) = (&mut xl[offset + i], &mut xu[offset + i]);
                // explicit bounds the wrong way round are reported as such, below
                if explicit && *l < *u && (*l).max(lo) >= (*u).min(hi) {
                    return Err(AlgorithmError::InvalidArgument(format!(
                        "the bounds [{l}, {u}] of latent {} do not overlap its support [{lo}, {hi}]",
                        self.model.graph().describe(id)
                    )));
                }
                *l = l.max(lo);
                *u = u.min(hi);
            }
        }
        Ok((xl, xu))
    }
}

/// The lower and upper bound of each element of a latent (shaped like `value`) from its support: finite for a lower
/// or interval support, infinite otherwise.
fn support_bounds(
    model: &Model,
    state: &State,
    id: NodeId,
    value: &DoubleTensor,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let n = value.len();
    let unbounded = || vec![f64::INFINITY; n];
    let elements = |bound: &DoubleTensor| -> Vec<f64> {
        match bound.to_scalar() {
            Some(b) => vec![b; n],
            None => match bound.broadcast_to(value.shape()) {
                Ok(b) => b.as_slice().to_vec(),
                // a support whose bounds don't fit the value can't bound it
                Err(_) => vec![f64::NAN; n],
            },
        }
    };
    let (lower, upper) = match node_support(model.graph(), state, id)? {
        Support::GreaterThan(lo) => (elements(&lo), unbounded()),
        Support::Interval(lo, hi) => (elements(&lo), elements(&hi)),
        _ => (vec![f64::NEG_INFINITY; n], unbounded()),
    };
    // NaN bounds (from a support that doesn't fit) impose nothing
    let lower = lower
        .into_iter()
        .map(|b| if b.is_nan() { f64::NEG_INFINITY } else { b })
        .collect();
    let upper = upper
        .into_iter()
        .map(|b| if b.is_nan() { f64::INFINITY } else { b })
        .collect();
    Ok((lower, upper))
}
