//! Keanu's non-gradient optimizer: BOBYQA (bound optimisation by quadratic approximation).
//!
//! The solver in [`prima`] is a translation of PRIMA's BOBYQA and is distributed under PRIMA's BSD 3-Clause
//! licence (see `THIRD-PARTY-NOTICES.md` in this crate). This file is part of keanu-algorithms and MIT licensed.

pub mod prima;

use super::{Fitness, Objective, OptimizedResult};
use crate::error::{AlgorithmError, Result};
use keanu_graph::{Model, NodeId, State};
use keanu_tensor::DoubleTensor;
use std::collections::HashMap;

/// BOBYQA settings (Java's `BOBYQA` builder).
#[derive(Debug, Clone, PartialEq)]
pub struct Bobyqa {
    /// The maximum number of fitness evaluations (default: unlimited).
    pub max_evaluations: usize,
    /// Latents without explicit bounds are bounded to their starting value ± this (default: unbounded).
    pub bounds_range: f64,
    /// The initial trust region radius (default 10, as in Keanu; reduced to fit the bounds if needed).
    pub initial_trust_region_radius: f64,
    /// The final trust region radius: the accuracy required in the latents (default 1e-8).
    pub stopping_trust_region_radius: f64,
    /// The number of interpolation points, between n + 2 and (n + 1)(n + 2) / 2 for n latent dimensions
    /// (default 2n + 1).
    pub interpolation_points: Option<usize>,
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

    /// Bounds a latent between `min` and `max`.
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
        super::check_latents(self.model)?;
        let objective = Objective::new(self.model, fitness);
        let start = objective.point(state)?;
        if self.check_initial_conditions {
            let initial = objective.value(state, &start)?;
            if initial == f64::NEG_INFINITY || initial.is_nan() {
                return Err(AlgorithmError::ImpossibleStart);
            }
        }

        let shapes: Vec<Vec<usize>> = start.iter().map(|t| t.shape().to_vec()).collect();
        let x0: Vec<f64> = start
            .iter()
            .flat_map(|t| t.as_slice().iter().copied())
            .collect();
        let (xl, xu) = self.flat_bounds(objective.latents(), &start)?;

        let unflatten = |x: &[f64]| -> Vec<DoubleTensor> {
            let mut offset = 0;
            shapes
                .iter()
                .map(|shape| {
                    let len: usize = shape.iter().product();
                    let t = DoubleTensor::new(shape, x[offset..offset + len].to_vec())
                        .expect("shape and length agree");
                    offset += len;
                    t
                })
                .collect()
        };

        let mut failure: Option<AlgorithmError> = None;
        let settings = prima::Settings {
            rhobeg: Some(self.algorithm.initial_trust_region_radius),
            rhoend: Some(self.algorithm.stopping_trust_region_radius),
            maxfun: Some(self.algorithm.max_evaluations),
            npt: self.algorithm.interpolation_points,
            ftarget: None,
            honour_x0: Some(true),
        };
        let outcome = prima::minimize(
            |x| {
                if failure.is_some() {
                    return f64::NAN;
                }
                match objective.value(state, &unflatten(x)) {
                    Ok(v) => -v,
                    Err(e) => {
                        failure = Some(e);
                        f64::NAN
                    }
                }
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
        let point = unflatten(&outcome.x);
        let fitness = objective.value(state, &point)?;
        Ok(OptimizedResult {
            values: objective.latents().iter().copied().zip(point).collect(),
            fitness,
            converged: matches!(
                outcome.status,
                prima::ExitStatus::SmallTrustRegionRadius | prima::ExitStatus::TargetReached
            ),
            evaluations: objective.evaluations(),
        })
    }

    /// Flattened lower and upper bounds: explicit bounds where given, otherwise start ± bounds_range.
    fn flat_bounds(
        &self,
        latents: &[NodeId],
        start: &[DoubleTensor],
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let mut xl = Vec::new();
        let mut xu = Vec::new();
        for (&id, value) in latents.iter().zip(start) {
            match self.bounds.get(id) {
                Some((min, max)) => {
                    for (bound, out, name) in [(min, &mut xl, "lower"), (max, &mut xu, "upper")] {
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
        }
        Ok((xl, xu))
    }
}
