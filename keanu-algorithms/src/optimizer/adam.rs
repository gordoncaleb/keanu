use super::{ConvergenceChecker, Objective};
use crate::error::{AlgorithmError, Result};
use keanu_graph::State;
use keanu_tensor::DoubleTensor;

/// The Adam optimizer (Kingma & Ba, 2014), with Keanu's defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Adam {
    /// Step size (default 0.001).
    pub alpha: f64,
    /// Decay of the first moment estimate (default 0.9).
    pub beta1: f64,
    /// Decay of the second moment estimate (default 0.999).
    pub beta2: f64,
    /// Guards the division by the second moment (default 1e-8).
    pub epsilon: f64,
    /// The maximum number of iterations (default unlimited).
    pub max_evaluations: usize,
    /// When to stop (default: relative change, L2 norm, below 1e-6).
    pub convergence: ConvergenceChecker,
}

/// Setters for each setting, for use as `Adam::default().with_...(...)`.
impl Adam {
    /// Sets [`alpha`](Self::alpha).
    #[must_use]
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Sets [`beta1`](Self::beta1).
    #[must_use]
    pub fn with_beta1(mut self, beta1: f64) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Sets [`beta2`](Self::beta2).
    #[must_use]
    pub fn with_beta2(mut self, beta2: f64) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Sets [`epsilon`](Self::epsilon).
    #[must_use]
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Sets [`max_evaluations`](Self::max_evaluations).
    #[must_use]
    pub fn with_max_evaluations(mut self, max_evaluations: usize) -> Self {
        self.max_evaluations = max_evaluations;
        self
    }

    /// Sets [`convergence`](Self::convergence).
    #[must_use]
    pub fn with_convergence(mut self, convergence: ConvergenceChecker) -> Self {
        self.convergence = convergence;
        self
    }
}

impl Default for Adam {
    fn default() -> Self {
        Self {
            alpha: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            max_evaluations: usize::MAX,
            convergence: ConvergenceChecker::default(),
        }
    }
}

impl Adam {
    fn validate(&self) -> Result<()> {
        let invalid = |m: &str| Err(AlgorithmError::InvalidSetting(m.into()));
        if self.alpha.is_nan() || self.alpha <= 0.0 {
            return invalid("alpha must be positive");
        }
        if !(0.0..1.0).contains(&self.beta1) || !(0.0..1.0).contains(&self.beta2) {
            return invalid("beta1 and beta2 must be in [0, 1)");
        }
        if self.epsilon.is_nan() || self.epsilon <= 0.0 {
            return invalid("epsilon must be positive");
        }
        if self.max_evaluations == 0 {
            return invalid("max_evaluations must be positive");
        }
        Ok(())
    }

    /// Returns the final position and whether it converged. `initial_gradient`, if known, is the gradient at
    /// `start`, so the first step needn't evaluate it. Stops, without converging, once a step passes the
    /// objective's boundary threshold.
    pub(crate) fn maximize(
        &self,
        objective: &Objective<'_>,
        state: &mut State,
        start: Vec<DoubleTensor>,
        mut initial_gradient: Option<Vec<DoubleTensor>>,
    ) -> Result<(Vec<DoubleTensor>, bool)> {
        self.validate()?;
        let mut theta = start;
        let mut m: Vec<DoubleTensor> = theta
            .iter()
            .map(|t| DoubleTensor::zeros(t.shape()))
            .collect();
        let mut v = m.clone();
        let (mut beta1_t, mut beta2_t) = (1.0, 1.0);
        let mut converged = false;
        let mut t = 1usize;
        while !converged && t <= self.max_evaluations {
            let gradients = match initial_gradient.take() {
                Some(gradient) => gradient,
                None => objective.gradient(state, &theta)?,
            };
            beta1_t *= self.beta1;
            beta2_t *= self.beta2;
            let b = (1.0 - beta1_t) / (1.0 - beta2_t).sqrt();
            let mut next = Vec::with_capacity(theta.len());
            for i in 0..theta.len() {
                m[i] = (&m[i] * self.beta1).add(&(&gradients[i] * (1.0 - self.beta1)))?;
                v[i] = (&v[i] * self.beta2)
                    .add(&(gradients[i].map(|g| g * g) * (1.0 - self.beta2)))?;
                let step = (&m[i] * self.alpha).div(&(v[i].sqrt() * b + self.epsilon))?;
                next.push(theta[i].add(&step)?);
            }
            converged = self.convergence.has_converged(&theta, &next);
            theta = next;
            t += 1;
            if objective.boundary_of(&theta).is_some() {
                // heading for the boundary of a support, where the optimum is at infinity: the caller reports it
                return Ok((theta, false));
            }
        }
        Ok((theta, converged))
    }
}
