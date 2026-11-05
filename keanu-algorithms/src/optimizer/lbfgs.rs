use super::Objective;
use super::line_search::{LineSearch, Outcome};
use crate::error::{AlgorithmError, Result};
use keanu_graph::State;
use std::collections::VecDeque;

/// Limited-memory BFGS (Nocedal & Wright, *Numerical Optimization*, Algorithm 7.4), with Stan's settings: the
/// default gradient algorithm.
///
/// Each iteration takes a quasi-Newton direction from the last `history_size` steps and gradient changes (the
/// two-loop recursion, with the initial inverse Hessian scaled by sᵀy / yᵀy), and a step along it satisfying the
/// strong Wolfe conditions (c1 = 1e-4, c2 = 0.9; bracketing and zoom with cubic interpolation). A pair whose
/// curvature yᵀs is not positive is not kept. If no step along the quasi-Newton direction decreases the objective
/// sufficiently, the history is cleared and the search restarts along the gradient; if that fails too, the
/// optimizer stops without converging. Trial points where the fitness or its gradient is not finite shorten the
/// step.
///
/// As in Stan, it converges when, between iterations, the fitness changes by less than `tol_obj`, or by less than
/// `tol_rel_obj` × ε relative to its magnitude (ε being machine epsilon, about 2.2e-16); or the gradient's norm
/// is below `tol_grad`, or gᵀ H g (with H the approximate inverse Hessian) relative to the fitness's magnitude is
/// below `tol_rel_grad` × ε; or the point moves less than `tol_param`. It stops without converging after
/// `max_iterations` iterations, and at the first point past the boundary threshold of a bounded latent in
/// unconstrained space that is heading for the boundary (see the [module documentation](super)).
///
/// A starting point whose fitness is not finite is an [`AlgorithmError::ImpossibleStart`], even without the
/// initial-condition check. Each trial point costs one fitness and one gradient evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Lbfgs {
    /// The number of past steps the inverse Hessian approximation is built from (default 5).
    pub history_size: usize,
    /// The maximum number of iterations (default 2000).
    pub max_iterations: usize,
    /// Absolute change in the fitness (default 1e-12).
    pub tol_obj: f64,
    /// Relative change in the fitness, in units of machine epsilon (default 1e4).
    pub tol_rel_obj: f64,
    /// Norm of the gradient (default 1e-8).
    pub tol_grad: f64,
    /// Relative gradient magnitude, in units of machine epsilon (default 1e7).
    pub tol_rel_grad: f64,
    /// Norm of the change in the point (default 1e-8).
    pub tol_param: f64,
}

impl Default for Lbfgs {
    fn default() -> Self {
        Self {
            history_size: 5,
            max_iterations: 2000,
            tol_obj: 1e-12,
            tol_rel_obj: 1e4,
            tol_grad: 1e-8,
            tol_rel_grad: 1e7,
            tol_param: 1e-8,
        }
    }
}

/// Setters for each setting, for use as `Lbfgs::default().with_...(...)`.
impl Lbfgs {
    /// Sets [`history_size`](Self::history_size).
    #[must_use]
    pub fn with_history_size(mut self, history_size: usize) -> Self {
        self.history_size = history_size;
        self
    }

    /// Sets [`max_iterations`](Self::max_iterations).
    #[must_use]
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Sets [`tol_obj`](Self::tol_obj).
    #[must_use]
    pub fn with_tol_obj(mut self, tol_obj: f64) -> Self {
        self.tol_obj = tol_obj;
        self
    }

    /// Sets [`tol_rel_obj`](Self::tol_rel_obj).
    #[must_use]
    pub fn with_tol_rel_obj(mut self, tol_rel_obj: f64) -> Self {
        self.tol_rel_obj = tol_rel_obj;
        self
    }

    /// Sets [`tol_grad`](Self::tol_grad).
    #[must_use]
    pub fn with_tol_grad(mut self, tol_grad: f64) -> Self {
        self.tol_grad = tol_grad;
        self
    }

    /// Sets [`tol_rel_grad`](Self::tol_rel_grad).
    #[must_use]
    pub fn with_tol_rel_grad(mut self, tol_rel_grad: f64) -> Self {
        self.tol_rel_grad = tol_rel_grad;
        self
    }

    /// Sets [`tol_param`](Self::tol_param).
    #[must_use]
    pub fn with_tol_param(mut self, tol_param: f64) -> Self {
        self.tol_param = tol_param;
        self
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn negated(a: &[f64]) -> Vec<f64> {
    a.iter().map(|x| -x).collect()
}

/// The last few steps s = x₊ − x and gradient changes y = g₊ − g, with ρ = 1 / yᵀs.
#[derive(Debug)]
struct History {
    pairs: VecDeque<(Vec<f64>, Vec<f64>, f64)>,
    capacity: usize,
}

impl History {
    fn new(capacity: usize) -> Self {
        Self {
            pairs: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    fn clear(&mut self) {
        self.pairs.clear();
    }

    /// Keeps the pair if its curvature yᵀs is positive (so the approximation stays positive definite), dropping
    /// the oldest pair when full. Returns whether it was kept.
    fn push(&mut self, s: Vec<f64>, y: Vec<f64>) -> bool {
        let ys = dot(&y, &s);
        if !(ys > 0.0 && ys.is_finite()) {
            return false;
        }
        if self.pairs.len() == self.capacity {
            self.pairs.pop_front();
        }
        self.pairs.push_back((s, y, 1.0 / ys));
        true
    }

    /// −H g by the two-loop recursion, with H₀ = γ I and γ = sᵀy / yᵀy of the newest pair (I when empty).
    fn direction(&self, g: &[f64]) -> Vec<f64> {
        let mut q = g.to_vec();
        let mut alphas = Vec::with_capacity(self.pairs.len());
        for (s, y, rho) in self.pairs.iter().rev() {
            let a = rho * dot(s, &q);
            q.iter_mut().zip(y).for_each(|(qi, yi)| *qi -= a * yi);
            alphas.push(a);
        }
        let gamma = self
            .pairs
            .back()
            .map_or(1.0, |(s, y, _)| dot(s, y) / dot(y, y));
        q.iter_mut().for_each(|qi| *qi *= gamma);
        for ((s, y, rho), a) in self.pairs.iter().zip(alphas.iter().rev()) {
            let b = rho * dot(y, &q);
            q.iter_mut().zip(s).for_each(|(qi, si)| *qi += (a - b) * si);
        }
        negated(&q)
    }
}

impl Lbfgs {
    fn validate(&self) -> Result<()> {
        let tolerances = [
            self.tol_obj,
            self.tol_rel_obj,
            self.tol_grad,
            self.tol_rel_grad,
            self.tol_param,
        ];
        if self.history_size == 0 || self.max_iterations == 0 {
            return Err(AlgorithmError::InvalidSetting(
                "history_size and max_iterations must be positive".into(),
            ));
        }
        if tolerances.iter().any(|t| t.is_nan() || *t < 0.0) {
            return Err(AlgorithmError::InvalidSetting(
                "the L-BFGS tolerances must be zero or positive".into(),
            ));
        }
        Ok(())
    }

    /// Maximizes the objective from `start`. Returns the final position, the fitness there and whether it
    /// converged. `known_start`, if given, is the fitness and gradient at `start` (from the initial-condition
    /// check), so they aren't evaluated again.
    pub(crate) fn maximize(
        &self,
        objective: &Objective<'_>,
        state: &mut State,
        start: Vec<f64>,
        known_start: Option<(f64, Vec<f64>)>,
    ) -> Result<(Vec<f64>, f64, bool)> {
        self.validate()?;
        let (fitness, gradient) = match known_start {
            Some(known) => known,
            None => objective.value_and_gradient_flat(state, &start)?,
        };
        if !fitness.is_finite() {
            return Err(AlgorithmError::ImpossibleStart);
        }
        // Keanu maximizes the fitness: minimize its negation
        let (x, f, converged) = self.minimize(
            state,
            start,
            -fitness,
            negated(&gradient),
            |state, x| {
                let (value, gradient) = objective.value_and_gradient_flat(state, x)?;
                Ok((-value, negated(&gradient)))
            },
            |state, x| Ok(objective.boundary(state, x)?.is_some()),
        )?;
        Ok((x, -f, converged))
    }

    /// Minimizes f from `x` (where f is `f` and its gradient `g`, both finite in f's case). `evaluate` returns f
    /// and its gradient at a point; `stop` says whether to stop (without converging) at a point the line search
    /// accepted. Both are given `context` (such as the state they evaluate in). Returns the final point, f there,
    /// and whether it converged.
    fn minimize<C>(
        &self,
        context: &mut C,
        mut x: Vec<f64>,
        mut f: f64,
        mut g: Vec<f64>,
        mut evaluate: impl FnMut(&mut C, &[f64]) -> Result<(f64, Vec<f64>)>,
        mut stop: impl FnMut(&mut C, &[f64]) -> Result<bool>,
    ) -> Result<(Vec<f64>, f64, bool)> {
        // there is nowhere to go along a NaN or infinite gradient
        if !g.iter().all(|gi| gi.is_finite()) {
            return Ok((x, f, false));
        }
        if norm(&g) <= self.tol_grad {
            return Ok((x, f, true));
        }
        let line_search = LineSearch::default();
        let mut history = History::new(self.history_size);
        let mut direction = negated(&g);
        for _ in 0..self.max_iterations {
            let mut slope = dot(&g, &direction);
            if slope.is_nan() || slope >= 0.0 {
                // not a descent direction (rounding in the recursion): restart along the gradient
                history.clear();
                direction = negated(&g);
                slope = -dot(&g, &g);
            }
            let trial = loop {
                // a quasi-Newton step has a natural length of 1; a gradient step starts no longer than 1
                let initial = if history.is_empty() {
                    (1.0 / norm(&g)).min(1.0)
                } else {
                    1.0
                };
                let outcome = line_search.search(f, slope, initial, |alpha| {
                    let point: Vec<f64> = x
                        .iter()
                        .zip(&direction)
                        .map(|(xi, di)| xi + alpha * di)
                        .collect();
                    let (value, gradient) = evaluate(context, &point)?;
                    Ok((value, dot(&gradient, &direction), (point, gradient)))
                })?;
                match outcome {
                    Outcome::Wolfe(trial) | Outcome::Decrease(trial) => break trial,
                    Outcome::Failed if !history.is_empty() => {
                        history.clear();
                        direction = negated(&g);
                        slope = -dot(&g, &g);
                    }
                    Outcome::Failed => return Ok((x, f, false)),
                }
            };
            let (next, g_next) = trial.data;
            let f_next = trial.value;
            if stop(context, &next)? {
                return Ok((next, f_next, false));
            }
            let s: Vec<f64> = next.iter().zip(&x).map(|(a, b)| a - b).collect();
            let y: Vec<f64> = g_next.iter().zip(&g).map(|(a, b)| a - b).collect();
            let step = norm(&s);
            history.push(s, y);
            direction = history.direction(&g_next);

            let change = (f - f_next).abs();
            let scale = f.abs().max(f_next.abs()).max(1.0);
            let relative_gradient = -dot(&g_next, &direction) / f_next.abs().max(1.0);
            let converged = change < self.tol_obj
                || change / scale < self.tol_rel_obj * f64::EPSILON
                || norm(&g_next) < self.tol_grad
                || relative_gradient < self.tol_rel_grad * f64::EPSILON
                || step < self.tol_param;
            x = next;
            f = f_next;
            g = g_next;
            if converged {
                return Ok((x, f, true));
            }
        }
        Ok((x, f, false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::Fitness;
    use keanu_graph::{GraphBuilder, Model, Parameterisation};
    use keanu_tensor::DoubleTensor;

    /// The generalised Rosenbrock function and its gradient.
    fn rosenbrock(x: &[f64]) -> (f64, Vec<f64>) {
        let n = x.len();
        let mut f = 0.0;
        let mut g = vec![0.0; n];
        for i in 0..n - 1 {
            let a = x[i + 1] - x[i] * x[i];
            let b = 1.0 - x[i];
            f += 100.0 * a * a + b * b;
            g[i] += -400.0 * a * x[i] - 2.0 * b;
            g[i + 1] += 200.0 * a;
        }
        (f, g)
    }

    fn minimize_rosenbrock(start: Vec<f64>) -> (Vec<f64>, bool, usize) {
        let (f, g) = rosenbrock(&start);
        let mut evaluations = 0;
        let (x, _, converged) = Lbfgs::default()
            .minimize(
                &mut (),
                start,
                f,
                g,
                |_, x| {
                    evaluations += 1;
                    Ok(rosenbrock(x))
                },
                |_, _| Ok(false),
            )
            .unwrap();
        (x, converged, evaluations)
    }

    #[test]
    fn rosenbrock_converges_to_all_ones() {
        let mut ten = [-1.2, 1.0].repeat(5);
        ten[9] = 1.0;
        for start in [vec![-1.2, 1.0], ten, vec![0.0; 10]] {
            let n = start.len();
            let (x, converged, evaluations) = minimize_rosenbrock(start);
            assert!(converged, "{n} dimensions");
            // the fitness converges (to a relative change of 1e4 ε) where Rosenbrock's narrow valley still leaves
            // the point ~1e-5 from the optimum
            assert!(
                x.iter().all(|xi| (xi - 1.0).abs() < 1e-4),
                "{n} dimensions: {x:?}"
            );
            assert!(evaluations < 500, "{n} dimensions: {evaluations}");
        }
    }

    #[test]
    fn pairs_without_positive_curvature_are_skipped() {
        let mut history = History::new(2);
        assert!(!history.push(vec![1.0, 0.0], vec![-1.0, 0.0]));
        assert!(!history.push(vec![1.0, 0.0], vec![0.0, 1.0]));
        assert!(!history.push(vec![1.0, 0.0], vec![f64::NAN, 0.0]));
        assert!(history.is_empty());
        // with no history the direction is steepest descent
        assert_eq!(history.direction(&[2.0, -1.0]), vec![-2.0, 1.0]);

        // a pair from f = x² + 2y² (Hessian diag(2, 4)) is kept; a pair with yᵀs ≤ 0 later leaves it in place
        assert!(history.push(vec![1.0, 0.0], vec![2.0, 0.0]));
        let before = history.direction(&[2.0, 4.0]);
        assert!(!history.push(vec![0.0, 1.0], vec![0.0, -4.0]));
        assert_eq!(history.direction(&[2.0, 4.0]), before);
        // the second pair completes the curvature: the direction is the exact Newton step to the minimum
        assert!(history.push(vec![0.0, 1.0], vec![0.0, 4.0]));
        let d = history.direction(&[2.0, 4.0]);
        assert!(
            (d[0] + 1.0).abs() < 1e-12 && (d[1] + 1.0).abs() < 1e-12,
            "{d:?}"
        );
        // the oldest pair is dropped when full
        assert!(history.push(vec![1.0, 1.0], vec![2.0, 4.0]));
        assert_eq!(history.pairs.len(), 2);
    }

    /// The objective with the Jacobian (as the Laplace approximation uses) optimizes the density of the
    /// unconstrained point: for Gamma(θ = 0.5, k = 2) that is the mode of the density of log σ, kθ = 1.
    #[test]
    fn the_jacobian_objective_is_maximized() {
        let b = GraphBuilder::new();
        let sigma = b.gamma(0.5, 2.0);
        b.set_value(sigma, DoubleTensor::scalar(2.0));
        let (graph, mut state) = b.build().unwrap();
        let model = Model::new(graph, &state).unwrap();
        let param = Parameterisation::new(&model, &state).unwrap();
        let objective = Objective::new(param, &state, Fitness::LogProb, true).unwrap();
        let start = objective.flat_point(&state).unwrap();
        let (x, fitness, converged) = Lbfgs::default()
            .maximize(&objective, &mut state, start, None)
            .unwrap();
        assert!(converged);
        assert!(x[0].abs() < 1e-4, "log σ = {}", x[0]);
        assert_eq!(objective.value_flat(&mut state, &x).unwrap(), fitness);
    }
}
